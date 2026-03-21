# =============================================================================
# pretraining.py  —  End-to-end pretraining of LWM 1.1 + Coordinate Attention
#
# This script pretrains LWM v1.1 with a Coordinate Attention (CA) front-end
# using the Masked Channel Modelling (MCM) self-supervised objective.
#
# It is a direct port of lwm_ca/pretraining_e2e.py adapted for LWM v1.1:
#   • Uses the SAME 16 DeepMIMO scenarios as lwm_ca
#   • Uses the SAME hyperparameters (lr, batch_size, epochs, scheduler…)
#   • Uses lwm1_1.lwm_model.lwm as the backbone (instead of lwm.lwm_model.lwm)
#   • Wraps the backbone with CoordAtt via LWM11WithPrepatchCA
#   • Produces the same output artefacts: .pth weights + .log training log
#
# Usage (from project root):
#   python -m lwm1_1_ca.pretraining [options]
#
# See --help for all CLI options. Quick start:
#   python -m lwm1_1_ca.pretraining \
#       --epochs 100 \
#       --batch-size 64 \
#       --save-path lwm1_1_ca/model1_1_weights_ca_e2e.pth
# =============================================================================

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Optional: TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None  # TensorBoard not installed — silently disabled

# Optional: AMP (mixed precision)
try:
    from torch import amp as torch_amp
except Exception:
    torch_amp = None  # Older PyTorch — fallback to cuda.amp below

# ---------------------------------------------------------------------------
# LWM 1.1 CA model + original LWM data preprocessing
# ---------------------------------------------------------------------------
# We reuse lwm.input_preprocess for data loading (same module lwm_ca uses).
# This guarantees IDENTICAL data — same scenarios, same cleaning, same
# real/imag layout — so the CA pretraining is perfectly comparable.
from lwm.input_preprocess import DeepMIMO_data_gen, deepmimo_data_cleaning
from lwm1_1_ca.torch_pipeline_v11 import LWM11WithPrepatchCA


# =============================================================================
# Utilities
# =============================================================================

def default_num_workers() -> int:
    """Return a sensible default for DataLoader num_workers."""
    cpu_count = os.cpu_count() or 2
    return max(1, min(8, cpu_count // 2))


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.

    All defaults match those used in lwm_ca/pretraining_e2e.py so that
    running either script with default args produces a fair comparison.
    """
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pretraining for Coordinate Attention + LWM v1.1.\n"
            "Uses the same scenarios, hyperparameters, and loss normalisation "
            "as lwm_ca/pretraining_e2e.py."
        )
    )

    # ---- Dataset ----
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            # Exact same 16 scenarios used in lwm_ca/pretraining_e2e.py
            "O1_3p5_v1",
            "O1_3p5_v2",
            "Boston5G_3p5",
            "asu_campus1",
            "city_0_newyork",
            "city_1_losangeles",
            "city_2_chicago",
            "city_3_houston",
            "city_4_phoenix",
            "city_5_philadelphia",
            "city_6_miami",
            "city_8_dallas",
            "city_9_sanfrancisco",
            "city_10_austin",
            "city_13_columbus",
            "city_17_seattle",
        ],
        help="DeepMIMO scenario names to load for pretraining.",
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        default=None,
        help=(
            "Path to the DeepMIMO scenarios root folder. "
            "If None (default), the lwm.input_preprocess module will resolve "
            "it automatically from ../scenarios relative to the lwm package."
        ),
    )
    parser.add_argument(
        "--channels-cache",
        type=str,
        default=None,
        help=(
            "Optional .npy path for caching the stacked (N,2,H,W) channel array. "
            "If the file already exists it is loaded; otherwise it is generated "
            "and saved. Greatly speeds up repeated pretraining runs."
        ),
    )

    # ---- Training hyperparameters (identical to lwm_ca defaults) ----
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--train-ratio",  type=float, default=0.8)
    parser.add_argument("--val-ratio",    type=float, default=0.2)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--step-size",    type=int,   default=10,   help="StepLR step size.")
    parser.add_argument("--gamma",        type=float, default=0.9,  help="StepLR decay factor.")
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--snr-db",       type=float, default=None, help="AWGN noise level (dB). None = no noise.")

    # ---- Hardware ----
    parser.add_argument("--device",              type=str, default=None,
                        help="'cuda', 'cpu', etc. None = auto-detect.")
    parser.add_argument("--num-workers",         type=int, default=default_num_workers())
    parser.add_argument("--prefetch-factor",     type=int, default=2)
    parser.add_argument("--persistent-workers",  action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory",          action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp",                 action=argparse.BooleanOptionalAction, default=False,
                        help="Automatic Mixed Precision (requires CUDA).")
    parser.add_argument("--tf32",                action=argparse.BooleanOptionalAction, default=True,
                        help="Allow TF32 matmul on Ampere+ GPUs.")
    parser.add_argument("--cudnn-benchmark",     action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-compile",       action=argparse.BooleanOptionalAction, default=False,
                        help="Enable torch.compile (requires PyTorch 2.x).")
    parser.add_argument("--compile-mode",        type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune"])

    # ---- Logging ----
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Print step-level stats every N batches (0 = off).")
    parser.add_argument("--log-file",     type=str, default=None,
                        help="Path for the training log file. Defaults to <save-path>.log.")
    parser.add_argument("--tensorboard",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tb-logdir",    type=str, default="runs/lwm1_1_ca_pretraining")

    # ---- Scheduler mode ----
    parser.add_argument(
        "--scheduler-step-per-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Step the LR scheduler after every batch (True) or after every epoch (False).",
    )

    # ---- Checkpointing ----
    parser.add_argument("--save-path",  type=str, default="lwm1_1_ca/model1_1_weights_ca_e2e.pth",
                        help="Where to save final model weights.")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save checkpoint every N epochs (0 = only at end).")

    return parser.parse_args()


# =============================================================================
# Data Loading
# =============================================================================

def load_channels_ri(
    scenarios: list,
    dataset_folder: str | None = None,
    cache_path: str | None = None,
) -> np.ndarray:
    """
    Load, clean, and stack DeepMIMO channel data across multiple scenarios.

    Returns a (N, 2, H, W) float32 ndarray where:
        N = total users (all scenarios combined)
        channel 0 = real part
        channel 1 = imaginary part
        H = n_antennas (32)
        W = n_subcarriers (32)

    Caching
    -------
    If cache_path is provided:
    - On first call: generates data, saves to cache_path.npy.
    - On subsequent calls: loads from cache and skips DeepMIMO generation.
    This is highly recommended for large multi-scenario training runs.

    Args:
        scenarios     : List of DeepMIMO scenario names.
        dataset_folder: Optional path to DeepMIMO scenarios root folder.
        cache_path    : Optional .npy file path for caching.

    Returns:
        channels_ri : ndarray of shape (N, 2, H, W), dtype float32.
    """
    # --- Try to load from cache first ---
    if cache_path:
        expanded = os.path.expanduser(cache_path)
        candidates = [expanded, expanded + ".npy"]
        for path in candidates:
            if os.path.exists(path):
                print(f"[Data] Loading cached channels from: {path}")
                return np.load(path)

    # --- Generate from DeepMIMO ---
    print(f"[Data] Generating channels for {len(scenarios)} scenarios …")
    data_parts = []
    for name in scenarios:
        print(f"  → {name}")
        # DeepMIMO_data_gen from lwm.input_preprocess uses the same
        # scenarios/ folder and parameters as the original lwm_ca runs.
        kwargs = {}
        if dataset_folder is not None:
            kwargs["dataset_folder"] = dataset_folder
        deepmimo_data = DeepMIMO_data_gen(name, **kwargs)
        cleaned = deepmimo_data_cleaning(deepmimo_data)  # removes inactive users, scales by 1e6
        data_parts.append(cleaned)

    channels = np.vstack(data_parts)                         # (N, 1, H, W) complex
    real = channels.real.astype(np.float32)
    imag = channels.imag.astype(np.float32)
    channels_ri = np.stack([real, imag], axis=1)             # (N, 2, H, W)

    # --- Save to cache ---
    if cache_path:
        save_path = os.path.expanduser(cache_path)
        cache_dir = os.path.dirname(save_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.save(save_path, channels_ri)
        print(f"[Data] Cached channels saved to: {save_path}.npy")

    return channels_ri


def split_data(
    dataset: torch.utils.data.Dataset,
    train_ratio: float,
    val_ratio: float,
    seed: int = 0,
) -> tuple:
    """
    Split a TensorDataset into train / val / test subsets.

    Args:
        dataset    : Full dataset.
        train_ratio: Fraction for training.
        val_ratio  : Fraction for validation.
        seed       : Random seed for reproducibility.

    Returns:
        (train_data, val_data, test_data) — same types as input dataset.
    """
    n = len(dataset)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    device: str = "cuda",
    amp: bool = False,
    scaler=None,
    log_interval: int = 0,
    log_fn=print,
    writer=None,
    epoch_idx: int = 0,
    non_blocking: bool = False,
    scheduler_step_per_batch: bool = True,
) -> tuple:
    """
    Run one training epoch.

    Loss: MSE(logits_lm, masked_tokens) / Var(masked_tokens)
    This is the exact normalisation used in lwm_ca/pretraining_e2e.py.

    Returns:
        avg_loss : float — mean loss over all batches.
        metrics  : dict  — timing and throughput stats.
    """
    model.train()
    criterion = nn.MSELoss()

    running_loss = 0.0
    data_time_sum = step_time_sum = samples_sum = 0.0
    log_loss = log_data_time = log_step_time = log_samples = 0.0
    end = time.perf_counter()

    for step, (channels,) in enumerate(dataloader):
        # ---- data loading time ----
        data_t = time.perf_counter() - end
        data_time_sum += data_t
        log_data_time += data_t

        channels = channels.to(device, non_blocking=non_blocking)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()

        # ---- forward + loss ----
        optimizer.zero_grad(set_to_none=True)

        # Mixed-precision context manager (gracefully handles old PyTorch)
        if torch_amp is not None:
            autocast_ctx = torch_amp.autocast(device_type="cuda", enabled=amp)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=amp)

        with autocast_ctx:
            # LWM11WithPrepatchCA returns (logits_lm, masked_tokens, output)
            logits_lm, masked_tokens, _ = model(channels)
            # Normalised MSE — variance normalisation makes the loss scale-invariant
            loss = criterion(logits_lm, masked_tokens) / torch.var(masked_tokens)

        # ---- backward + optimiser step ----
        if amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # ---- LR scheduler (per-batch mode) ----
        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        step_t = time.perf_counter() - start
        step_time_sum += step_t
        log_step_time += step_t

        batch_sz = channels.size(0)
        samples_sum += batch_sz
        log_samples += batch_sz
        running_loss += loss.item()
        log_loss += loss.item()

        # ---- Step-level logging ----
        if log_interval and (step + 1) % log_interval == 0:
            avg_l = log_loss / log_interval
            avg_d = log_data_time / log_interval
            avg_s = log_step_time / log_interval
            throughput = log_samples / log_step_time if log_step_time > 0 else 0.0

            if writer is not None:
                g = epoch_idx * len(dataloader) + step + 1
                writer.add_scalar("train/loss_step",     avg_l,               g)
                writer.add_scalar("train/data_time_ms",  avg_d * 1000,        g)
                writer.add_scalar("train/step_time_ms",  avg_s * 1000,        g)
                writer.add_scalar("train/throughput",    throughput,          g)
                writer.add_scalar("train/lr",            optimizer.param_groups[0]["lr"], g)

            log_fn(
                f"  step {step + 1:>5}/{len(dataloader)} | "
                f"loss {avg_l:.4f} | "
                f"data {avg_d * 1000:.1f} ms | "
                f"step {avg_s * 1000:.1f} ms | "
                f"{throughput:.1f} samples/s"
            )
            log_loss = log_data_time = log_step_time = log_samples = 0.0

        end = time.perf_counter()

    avg_loss = running_loss / len(dataloader)
    metrics = {
        "data_time":  data_time_sum / len(dataloader),
        "step_time":  step_time_sum / len(dataloader),
        "throughput": samples_sum / step_time_sum if step_time_sum > 0 else 0.0,
    }
    return avg_loss, metrics


# =============================================================================
# Validation Loop
# =============================================================================

def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader | None,
    device: str = "cuda",
    amp: bool = False,
    non_blocking: bool = False,
) -> tuple:
    """
    Evaluate the model on validation (or test) data without gradient updates.

    Returns:
        avg_loss : float — mean loss.
        metrics  : dict  — timing stats.
    """
    if dataloader is None or len(dataloader) == 0:
        return 0.0, {"data_time": 0.0, "step_time": 0.0, "throughput": 0.0}

    model.eval()
    criterion = nn.MSELoss()

    running_loss = 0.0
    data_time_sum = step_time_sum = samples_sum = 0.0
    end = time.perf_counter()

    with torch.inference_mode():
        for (channels,) in dataloader:
            data_t = time.perf_counter() - end
            data_time_sum += data_t

            channels = channels.to(device, non_blocking=non_blocking)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()

            if torch_amp is not None:
                autocast_ctx = torch_amp.autocast(device_type="cuda", enabled=amp)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=amp)

            with autocast_ctx:
                logits_lm, masked_tokens, _ = model(channels)
                loss = criterion(logits_lm, masked_tokens) / torch.var(masked_tokens)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            step_t = time.perf_counter() - start
            step_time_sum += step_t
            samples_sum += channels.size(0)
            running_loss += loss.item()
            end = time.perf_counter()

    avg_loss = running_loss / len(dataloader)
    metrics = {
        "data_time":  data_time_sum / len(dataloader),
        "step_time":  step_time_sum / len(dataloader),
        "throughput": samples_sum / step_time_sum if step_time_sum > 0 else 0.0,
    }
    return avg_loss, metrics


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # ---- Device ----
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    # ---- Load data ----
    channels_ri = load_channels_ri(
        scenarios=args.scenarios,
        dataset_folder=args.dataset_folder,
        cache_path=args.channels_cache,
    )
    print(f"[Data] Channels shape: {channels_ri.shape} (N, 2, H, W)")

    tensor = torch.from_numpy(channels_ri)                           # (N, 2, H, W) float32
    dataset = torch.utils.data.TensorDataset(tensor)

    train_data, val_data, test_data = split_data(
        dataset, args.train_ratio, args.val_ratio, seed=args.seed
    )

    # ---- DataLoaders ----
    num_workers = max(0, int(args.num_workers))
    pin_memory = bool(args.pin_memory and device.startswith("cuda"))
    non_blocking = pin_memory and device.startswith("cuda")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"]    = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True,  **loader_kwargs)
    val_loader   = torch.utils.data.DataLoader(val_data,   shuffle=False, **loader_kwargs) if len(val_data)  > 0 else None
    test_loader  = torch.utils.data.DataLoader(test_data,  shuffle=False, **loader_kwargs) if len(test_data) > 0 else None

    # ---- Model ----
    model = LWM11WithPrepatchCA(gen_raw=False, snr_db=args.snr_db).to(device)
    print(
        f"[Model] LWM11WithPrepatchCA created. "
        f"Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Optional: torch.compile for PyTorch 2.x
    if args.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f"[Model] torch.compile enabled (mode={args.compile_mode})")
        except Exception as exc:
            print(f"[Warning] torch.compile failed, continuing uncompiled: {exc}")

    # ---- Optimiser & Scheduler ----
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # ---- AMP Scaler ----
    amp_enabled = bool(args.amp and device.startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # ---- Logging setup ----
    log_file = args.log_file
    if log_file is None:
        base = args.save_path or "lwm1_1_ca/pretraining"
        log_file = os.path.splitext(base)[0] + ".log"
    log_file = os.path.expanduser(log_file)
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_fp = open(log_file, "a", encoding="utf-8")

    # ---- TensorBoard (optional) ----
    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            print("[Warning] TensorBoard requested but 'tensorboard' package not installed.")
        else:
            tb_dir = os.path.expanduser(args.tb_logdir)
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)

    def log(*msg):
        """Write a message to stdout AND the log file simultaneously."""
        print(*msg)
        log_fp.write(" ".join(str(m) for m in msg) + "\n")
        log_fp.flush()

    # ---- Print run configuration ----
    log("=" * 60)
    log("LWM v1.1 + Coordinate Attention — Pretraining")
    log("=" * 60)
    log(f"Log file    : {log_file}")
    log(f"Device      : {device}")
    log(f"Scenarios   : {args.scenarios}")
    log(f"Num workers : {num_workers} | Pin memory: {pin_memory}")
    log(f"AMP         : {'enabled' if amp_enabled else 'disabled'}")
    log(f"Epochs      : {args.epochs}")
    log(f"Batch size  : {args.batch_size}")
    log(f"LR          : {args.lr} | Weight decay: {args.weight_decay}")
    log(f"Scheduler   : StepLR(step={args.step_size}, gamma={args.gamma})")
    log(f"Scheduler per batch: {args.scheduler_step_per_batch}")
    log(f"Seed        : {args.seed}")
    log(f"SNR (dB)    : {args.snr_db}")
    log(f"Save path   : {args.save_path}")
    log("-" * 60)
    log(
        f"Dataset sizes — train: {len(train_data)}, "
        f"val: {len(val_data)}, test: {len(test_data)}"
    )
    log(f"Batches per epoch — train: {len(train_loader)}, "
        f"val: {len(val_loader) if val_loader else 0}, "
        f"test: {len(test_loader) if test_loader else 0}")
    log("=" * 60)

    # ====================================================================
    # Main training loop
    # ====================================================================
    for epoch in range(args.epochs):
        log(f"\nEpoch {epoch + 1}/{args.epochs}")
        log("-" * 40)
        log(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        log(f"Batch Size   : {args.batch_size}")

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        # ---- Train ----
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device=device,
            amp=amp_enabled,
            scaler=scaler,
            log_interval=args.log_interval,
            log_fn=log,
            writer=writer,
            epoch_idx=epoch,
            non_blocking=non_blocking,
            scheduler_step_per_batch=args.scheduler_step_per_batch,
        )

        # Epoch-level LR scheduler step (if not per-batch)
        if scheduler is not None and not args.scheduler_step_per_batch:
            scheduler.step()

        log(
            f"Training Loss: {train_loss:.6f} | "
            f"data {train_metrics['data_time'] * 1000:.1f} ms | "
            f"step {train_metrics['step_time'] * 1000:.1f} ms | "
            f"{train_metrics['throughput']:.1f} samples/s"
        )

        if writer is not None:
            writer.add_scalar("train/loss_epoch",          train_loss,                         epoch + 1)
            writer.add_scalar("train/throughput_epoch",    train_metrics["throughput"],         epoch + 1)

        if device.startswith("cuda"):
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            log(f"Peak GPU memory: {peak_mem:.1f} MB")
            if writer is not None:
                writer.add_scalar("train/peak_gpu_mem_mb", peak_mem, epoch + 1)

        # ---- Validate ----
        if val_loader is not None:
            val_loss, val_metrics = validate_epoch(
                model, val_loader, device=device, amp=amp_enabled, non_blocking=non_blocking
            )
            log(
                f"Validation Loss: {val_loss:.6f} | "
                f"data {val_metrics['data_time'] * 1000:.1f} ms | "
                f"step {val_metrics['step_time'] * 1000:.1f} ms | "
                f"{val_metrics['throughput']:.1f} samples/s"
            )
            if writer is not None:
                writer.add_scalar("val/loss_epoch", val_loss, epoch + 1)

        # ---- Intermediate checkpoint ----
        if args.save_every and (epoch + 1) % args.save_every == 0 and args.save_path:
            ckpt_dir = os.path.dirname(args.save_path)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            # Unwrap torch.compile wrapper if present
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt_path = os.path.splitext(args.save_path)[0] + f"_epoch{epoch + 1}.pth"
            torch.save(save_model.state_dict(), ckpt_path)
            log(f"Checkpoint saved: {ckpt_path}")

    # ====================================================================
    # Test evaluation (on held-out set after all epochs)
    # ====================================================================
    if test_loader is not None:
        test_loss, test_metrics = validate_epoch(
            model, test_loader, device=device, amp=amp_enabled, non_blocking=non_blocking
        )
        log(
            f"\nTest Loss: {test_loss:.6f} | "
            f"data {test_metrics['data_time'] * 1000:.1f} ms | "
            f"step {test_metrics['step_time'] * 1000:.1f} ms | "
            f"{test_metrics['throughput']:.1f} samples/s"
        )
        if writer is not None:
            writer.add_scalar("test/loss",       test_loss,                      0)
            writer.add_scalar("test/throughput", test_metrics["throughput"],     0)

    # ====================================================================
    # Save final model weights
    # ====================================================================
    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(save_model.state_dict(), args.save_path)
        log(f"\nFinal model saved to: {args.save_path}")

    if writer is not None:
        writer.close()
    log_fp.close()
    print("Pretraining complete.")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    main()
