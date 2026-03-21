# =============================================================================
# benchmark_secnn.py  —  Downstream beam prediction evaluation
#                        using a custom SECNN model
#
# This script evaluates the CA-pretrained LWM v1.1 backbone on the
# Beam Prediction downstream task using a Squeeze-and-Excitation CNN (SECNN)
# as the downstream model head.
#
# Pipeline
# --------
#   1. Load DeepMIMO data for the selected scenarios.
#   2. Generate beam prediction labels.
#   3. Tokenize channels via lwm1_1.input_preprocess.tokenizer (no masking).
#   4. Run LWM inference to get embeddings (cls_emb or channel_emb or raw).
#   5. Split into train/val/test at several split ratios.
#   6. Train SECNN head at each split ratio from scratch (reproducible seed).
#   7. Report test Accuracy, Top-3 Accuracy, and Weighted F1-Score.
#   8. Save results to CSV and plots to PNG.
#
# Compared models
# ---------------
#   • base_lwm1_1 : original LWM v1.1 pretrained backbone (model.pth)
#   • ca_lwm1_1   : LWM v1.1 backbone pretrained with CoordAtt
#                   (model_weights_ca_e2e.pth)
#   • raw         : no backbone — raw channel representation baseline
#
# Usage (from project root):
#   python -m lwm1_1_ca.benchmark_secnn [options]
#   python -m lwm1_1_ca.benchmark_secnn --help
# =============================================================================

import argparse
import csv
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# LWM 1.1 modules
from lwm1_1.input_preprocess import tokenizer
from lwm1_1.lwm_model import lwm as LWM11
from lwm1_1.inference import lwm_inference

# LWM 1.1 CA model (for loading CA-pretrained weights)
from lwm1_1_ca.torch_pipeline_v11 import LWM11WithPrepatchCA


# =============================================================================
# SECNN: Squeeze-and-Excitation 1D Convolutional Neural Network
# =============================================================================

class SEBlock1D(nn.Module):
    """
    1D Squeeze-and-Excitation block.

    Given a feature map of shape (B, C, L), this block:
      1. Global average-pools to (B, C, 1) — 'squeeze'
      2. Passes through two FC layers with ReLU and Sigmoid — 'excitation'
      3. Multiplies the original feature map by the channel weights — 'recalibration'

    This is the 1D analogue of SENet (Hu et al., CVPR 2018), adapted for
    sequence / spectral features rather than image feature maps.

    Parameters
    ----------
    channels  : Number of input and output channels.
    reduction : Bottleneck reduction ratio (default 8).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)             # (B, C, L) -> (B, C, 1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, L)
        Returns:
            x with channel-wise attention applied, same shape.
        """
        scale = self.squeeze(x).squeeze(-1)         # (B, C)
        scale = self.excitation(scale).unsqueeze(-1) # (B, C, 1)
        return x * scale                             # (B, C, L) — broadcast


class SEResBlock1D(nn.Module):
    """
    1D Residual block with a Squeeze-Excitation attention mechanism.

    Structure:
        Conv1d(in, out, k=3) → BN → ReLU → Conv1d(out, out, k=3) → BN → SE → + shortcut → ReLU

    When in_channels != out_channels, the shortcut uses a 1×1 conv + BN.
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels,  out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.se    = SEBlock1D(out_channels, reduction=reduction)

        # Shortcut projection when channel sizes differ
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)                   # channel attention
        out = out + self.shortcut(identity)  # residual connection
        return F.relu(out)


class SECNN(nn.Module):
    """
    Squeeze-and-Excitation 1D CNN for wireless channel feature classification.

    Architecture:
      Stem conv + MaxPool
      → 3 SE-Residual stages (32 → 64 → 128 channels)
      → Global Average Pool
      → FC(128) + BN + ReLU + Dropout
      → FC(num_classes)

    Input shape convention: (B, sequence_length, input_channels)
    The first layer transposes to (B, input_channels, sequence_length) for Conv1d.

    This architecture is analogous to the res1dcnn in lwm1_1/main.py but with
    Squeeze-Excitation blocks replacing plain residual blocks, giving improved
    channel-wise feature recalibration.

    Parameters
    ----------
    input_channels  : Number of feature dimensions per time step.
    sequence_length : Number of time steps (patches or embedding dims).
    num_classes     : Number of output classes (beam indices).
    se_reduction    : SE bottleneck reduction ratio (default 8).
    dropout         : Dropout rate before the final classifier (default 0.5).
    """

    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        num_classes: int,
        se_reduction: int = 8,
        dropout: float = 0.5,
    ):
        super().__init__()

        # --- Stem ---
        self.stem_conv = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_bn   = nn.BatchNorm1d(32)
        self.stem_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # --- SE-Residual stages ---
        self.stage1 = self._make_stage(32,  32,  n_blocks=2, reduction=se_reduction)
        self.stage2 = self._make_stage(32,  64,  n_blocks=3, reduction=se_reduction)
        self.stage3 = self._make_stage(64,  128, n_blocks=4, reduction=se_reduction)

        # Compute the flattened feature size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, sequence_length)
            feat = self._conv_forward(dummy)
            flat_size = feat.numel()

        # --- Classifier head ---
        self.fc1     = nn.Linear(flat_size, 128)
        self.bn_fc   = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(128, num_classes)

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int, reduction: int):
        blocks = [SEResBlock1D(in_ch, out_ch, reduction=reduction)]
        for _ in range(1, n_blocks):
            blocks.append(SEResBlock1D(out_ch, out_ch, reduction=reduction))
        return nn.Sequential(*blocks)

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through all conv layers and return flattened features."""
        x = self.stem_pool(F.relu(self.stem_bn(self.stem_conv(x))))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = F.adaptive_avg_pool1d(x, 1)  # (B, 128, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, sequence_length, input_channels) — note the shape convention.
        Returns
            logits : (B, num_classes)
        """
        x = x.transpose(1, 2)       # (B, input_channels, sequence_length)
        x = self._conv_forward(x)   # (B, 128, 1)
        x = x.view(x.size(0), -1)   # (B, flat_size)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)


# =============================================================================
# Data utilities
# =============================================================================

def get_data_loaders(
    data: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    train_ratio: float,
    seed: int = 42,
) -> tuple:
    """Split (data, labels) into train / val / test DataLoaders."""
    dataset = TensorDataset(data, labels)
    n = len(dataset)
    n_train = max(1, int(train_ratio * n))
    n_remain = n - n_train
    n_val = n_remain // 2
    n_test = n_remain - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  generator=generator),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
    )


# =============================================================================
# Training & evaluation
# =============================================================================

def train_and_eval_secnn(
    data: torch.Tensor,
    labels: torch.Tensor,
    input_type: str,
    num_classes: int,
    split_ratio: float,
    num_epochs: int,
    batch_size: int,
    device: str,
    seed: int = 42,
    initial_lr: float = 1e-3,
) -> dict:
    """
    Train SECNN from scratch at a given split ratio and evaluate on held-out test set.

    Returns a dict with keys:
        train_losses, val_losses, val_f1s, test_acc, test_top3_acc, test_f1
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Determine SECNN input shape from the data tensor ----
    # data can be:
    #   cls_emb     → (N, 128)       → treat as (N, 1, 128) for Conv1d
    #   channel_emb → (N, 512, 128)  → sequence_length=512, input_channels=128
    #   raw         → (N, 512, 32)   → sequence_length=512, input_channels=32
    if data.dim() == 2:
        # cls_emb: add a dummy sequence dim
        data = data.unsqueeze(1)          # (N, 1, 128)

    seq_len = data.shape[1]
    input_ch = data.shape[2]

    train_loader, val_loader, test_loader = get_data_loaders(
        data, labels, batch_size, split_ratio, seed=seed
    )

    model = SECNN(
        input_channels=input_ch,
        sequence_length=seq_len,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_f1s = [], [], []

    for epoch in range(1, num_epochs + 1):
        # -- Train --
        model.train()
        ep_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * x.size(0)
        train_losses.append(ep_loss / len(train_loader.dataset))
        scheduler.step()

        # -- Validate --
        model.eval()
        v_loss, v_preds, v_tgts = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += criterion(out, y).item() * x.size(0)
                v_preds.extend(out.argmax(1).cpu().tolist())
                v_tgts.extend(y.cpu().tolist())
        val_losses.append(v_loss / len(val_loader.dataset))
        val_f1s.append(f1_score(v_tgts, v_preds, average="weighted", zero_division=0))

    # -- Test --
    model.eval()
    correct1 = correct3 = total = 0
    t_preds, t_tgts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            # Top-1 accuracy
            top1 = out.argmax(1)
            correct1 += (top1 == y).sum().item()
            # Top-3 accuracy
            top3 = out.topk(min(3, out.size(1)), dim=1).indices
            correct3 += sum(y[i] in top3[i] for i in range(len(y)))
            total += y.size(0)
            t_preds.extend(top1.cpu().tolist())
            t_tgts.extend(y.cpu().tolist())

    return {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "val_f1s":      val_f1s,
        "test_acc":     100.0 * correct1 / total,
        "test_top3_acc":100.0 * correct3 / total,
        "test_f1":      f1_score(t_tgts, t_preds, average="weighted", zero_division=0),
    }


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LWM v1.1 base vs CA-pretrained LWM v1.1 on Beam Prediction "
            "using a custom SECNN downstream model."
        )
    )

    parser.add_argument("--scenario",      type=str,   default="city_0_newyork",
                        help="Single DeepMIMO scenario for downstream evaluation.")
    parser.add_argument("--bs-idx",        type=int,   default=3,
                        help="Base-station index to use from the scenario.")
    parser.add_argument("--n-beams",       type=int,   default=64)
    parser.add_argument(
        "--input-types",
        nargs="+",
        default=["cls_emb", "channel_emb", "raw"],
        choices=["cls_emb", "channel_emb", "raw"],
    )
    parser.add_argument(
        "--split-ratios",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.05, 0.1, 0.2, 0.4],
    )
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     type=str,   default=None)

    # ---- Checkpoint paths ----
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default="lwm1_1/model.pth",
        help="Path to original LWM v1.1 pretrained weights (.pth).",
    )
    parser.add_argument(
        "--ca-ckpt",
        type=str,
        default="lwm1_1_ca/model_weights_ca_e2e.pth",
        help="Path to LWM v1.1 CA pretrained weights (.pth).",
    )

    # ---- Output ----
    parser.add_argument("--save-csv",  type=str, default="lwm1_1_ca/results/benchmark_secnn.csv")
    parser.add_argument("--save-plot", type=str, default="lwm1_1_ca/results/benchmark_secnn.png")

    return parser.parse_args()


# =============================================================================
# Embedding extraction helpers
# =============================================================================

def _load_lwm11(ckpt_path: str, device: str) -> nn.Module:
    """Load base LWM v1.1 weights, stripping DataParallel prefix if present."""
    model = LWM11().to(device)
    state = torch.load(ckpt_path, map_location=device)
    # Strip 'module.' prefix added by DataParallel
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def _load_lwm11_ca(ckpt_path: str, device: str) -> nn.Module:
    """Load CA-pretrained LWM v1.1 weights into LWM11WithPrepatchCA."""
    model = LWM11WithPrepatchCA(gen_raw=True).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # Strip torch.compile's _orig_mod prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def get_embeddings_base(
    model: nn.Module,
    preprocessed_data: torch.Tensor,
    input_type: str,
    device: str,
) -> torch.Tensor:
    """
    Extract embeddings from the base LWM v1.1 model using lwm_inference.

    Returns (N, ...) float tensor. Shape depends on input_type:
        cls_emb     → (N, 128)
        channel_emb → (N, 512, 128)
        raw         → (N, 512, 32) — returns raw tokenised channels
    """
    if input_type == "raw":
        return preprocessed_data[:, 1:, :].float()   # strip CLS, keep patches

    wrapped = nn.DataParallel(model)
    emb = lwm_inference(wrapped, preprocessed_data, input_type=input_type, device=device)
    return emb.float()


def get_embeddings_ca(
    ca_model: LWM11WithPrepatchCA,
    channels_ri: torch.Tensor,
    input_type: str,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Extract embeddings from the CA-pretrained model.

    The CA model operates on (B, 2, H, W) raw channel tensors.
    Output shapes match base model for fair comparison:
        cls_emb     → (N, 128)
        channel_emb → (N, seq_len-1, 128)
        raw         → (N, n_patches, patch_size)
    """
    from lwm1_1_ca.torch_pipeline_v11 import (
        ensure_ri_channels, add_complex_noise_ri, channels_to_patches
    )

    loader = DataLoader(TensorDataset(channels_ri), batch_size=batch_size, shuffle=False)
    all_embs = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            if input_type == "raw":
                # Return CA-processed patches without LWM encoding
                ri = ensure_ri_channels(batch)
                ca_out = ca_model.coordatt(ri)
                patches = channels_to_patches(ca_out, patch_size=ca_model.patch_size)
                all_embs.append(patches.cpu())
            else:
                # Full forward: CA → patches → masking (gen_raw=True, no masking) → LWM
                _, _, output = ca_model(batch)  # output: (B, seq_len, 128)
                if input_type == "cls_emb":
                    all_embs.append(output[:, 0, :].cpu())
                elif input_type == "channel_emb":
                    all_embs.append(output[:, 1:, :].cpu())

    return torch.cat(all_embs, dim=0).float()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] Device: {device}")
    print(f"[Benchmark] Scenario: {args.scenario}  |  n_beams: {args.n_beams}")

    # ---- 1. Tokenize downstream data (using lwm1_1's tokenizer) ----
    print("\n[Data] Tokenizing downstream data …")
    preprocessed_data, labels, raw_chs = tokenizer(
        selected_scenario_names=[args.scenario],
        bs_idxs=[args.bs_idx],
        load_data=False,
        task="Beam Prediction",
        n_beams=args.n_beams,
        mask=False,   # gen_raw=True equivalent — no masking for inference
    )
    # preprocessed_data: (N, seq_len, patch_size) tensor
    # labels: (N,) long tensor
    print(f"[Data] Samples: {len(preprocessed_data)} | Labels: {labels.shape}")

    labels = labels.long()
    n_classes = args.n_beams + 1  # +1 to match lwm1_1/main.py convention

    # ---- Build (N, 2, H, W) channel tensor for CA model ----
    # raw_chs is (N, 2*H*W) flat — reshape to (N, 2, 32, 32) for the CA model
    # We infer H=W=32 from standard config (n_ant=32, n_sub=32)
    N = raw_chs.shape[0]
    HW = raw_chs.shape[1] // 2        # half = H*W
    H = W = int(HW ** 0.5)
    channels_ri = torch.zeros(N, 2, H, W, dtype=torch.float32)
    channels_ri[:, 0] = raw_chs[:, :HW].view(N, H, W)   # real
    channels_ri[:, 1] = raw_chs[:, HW:].view(N, H, W)   # imag

    # ---- 2. Load models ----
    print("\n[Model] Loading base LWM v1.1 …")
    base_model = _load_lwm11(args.base_ckpt, device)

    print(f"[Model] Loading CA-pretrained LWM v1.1 from {args.ca_ckpt} …")
    ca_model = _load_lwm11_ca(args.ca_ckpt, device)

    # ---- 3. Results container ----
    os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
    csv_rows = []
    plot_data = {}   # {model_name: {split_ratio: test_f1}}

    for model_name, lwm_model, is_ca in [
        ("base_lwm1_1", base_model, False),
        ("ca_lwm1_1",   ca_model,   True),
    ]:
        plot_data[model_name] = {}
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        for input_type in args.input_types:
            print(f"\n  Input type: {input_type}")

            # ---- Extract embeddings ----
            if is_ca:
                emb = get_embeddings_ca(lwm_model, channels_ri, input_type, device)
            else:
                emb = get_embeddings_base(lwm_model, preprocessed_data, input_type, device)

            # ---- Train SECNN at each split ratio ----
            for sr in args.split_ratios:
                print(f"    Split ratio: {sr:.3f} … ", end="", flush=True)
                t0 = time.time()
                results = train_and_eval_secnn(
                    data=emb,
                    labels=labels,
                    input_type=input_type,
                    num_classes=n_classes,
                    split_ratio=sr,
                    num_epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=device,
                    seed=args.seed,
                )
                elapsed = time.time() - t0
                print(
                    f"Acc={results['test_acc']:.2f}%  "
                    f"Top3={results['test_top3_acc']:.2f}%  "
                    f"F1={results['test_f1']:.4f}  "
                    f"({elapsed:.1f}s)"
                )

                csv_rows.append({
                    "model":       model_name,
                    "input_type":  input_type,
                    "split_ratio": sr,
                    "test_acc":    results["test_acc"],
                    "test_top3":   results["test_top3_acc"],
                    "test_f1":     results["test_f1"],
                })

                if input_type == "cls_emb":
                    plot_data[model_name][sr] = results["test_f1"]

    # ---- 4. Save CSV ----
    with open(args.save_csv, "w", newline="") as f:
        fieldnames = ["model", "input_type", "split_ratio", "test_acc", "test_top3", "test_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[Results] CSV saved to: {args.save_csv}")

    # ---- 5. Plot F1 vs split ratio (cls_emb only) ----
    if args.save_plot:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        styles = {
            "base_lwm1_1": {"color": "steelblue",  "marker": "o", "linestyle": "-"},
            "ca_lwm1_1":   {"color": "darkorange", "marker": "s", "linestyle": "--"},
        }
        for model_name, ratio_f1 in plot_data.items():
            ratios = sorted(ratio_f1.keys())
            f1s    = [ratio_f1[r] for r in ratios]
            ax.plot(ratios, f1s, label=model_name, linewidth=2, markersize=7,
                    **styles.get(model_name, {}))

        ax.set_xscale("log")
        ax.set_xlabel("Train Split Ratio", fontsize=13)
        ax.set_ylabel("Weighted F1-Score", fontsize=13)
        ax.set_title(
            f"Beam Prediction — LWM 1.1 vs CA-LWM 1.1\n"
            f"Scenario: {args.scenario}  |  SECNN (cls_emb)  |  n_beams={args.n_beams}",
            fontsize=12,
        )
        ax.legend(fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(labelsize=11)
        plt.tight_layout()
        plt.savefig(args.save_plot)
        print(f"[Results] Plot saved to: {args.save_plot}")
        plt.close(fig)

    print("\nBenchmark complete.")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    main()
