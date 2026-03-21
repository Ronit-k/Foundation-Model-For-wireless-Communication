# =============================================================================
# torch_pipeline_v11.py  —  End-to-end CA + LWM 1.1 pretraining pipeline
#
# This module mirrors lwm_ca/torch_pipeline.py but is adapted for LWM v1.1:
#
#   Key differences vs. lwm_ca/torch_pipeline.py
#   ------------------------------------------------
#   1. Uses `lwm1_1.lwm_model.lwm` whose forward() returns a 3-TUPLE:
#         (logits_lm, output_tensor, attention_maps_list)
#      The original lwm returns a 2-tuple.
#   2. The patch format is IDENTICAL to the original lwm tokenizer:
#         real and imag parts are concatenated side-by-side (flat)
#         then divided into fixed 16-element patches.
#      This ensures the pretrained model can directly feed into downstream
#      tasks that use the same patch representation.
#   3. Masking strategy: 15% MCM with symmetric real/imag masking (same as
#      lwm_ca). lwm1_1's own tokenizer uses 40%, but for pretraining we keep
#      the exact same regime as lwm_ca so results are comparable.
#
# =============================================================================

import os
import sys

# ---------------------------------------------------------------------------
# Make the project root importable when running this file directly.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

from lwm1_1_ca.coordatt import CoordAtt
from lwm1_1.lwm_model import lwm as LWM11  # LWM v1.1 – returns 3-tuple in forward


# ---------------------------------------------------------------------------
# Helper: normalise an arbitrary channel representation to (B, 2, H, W)
# ---------------------------------------------------------------------------

def ensure_ri_channels(channels: torch.Tensor) -> torch.Tensor:
    """
    Convert any supported input format to a float32 (B, 2, H, W) tensor
    where channel 0 = real part, channel 1 = imaginary part.

    Supported input shapes
    ----------------------
    • (B, H, W)          – real-valued, imag assumed zero
    • (B, 1, H, W)       – same, singleton channel dim squeezed
    • (B, 2, H, W)       – already in the desired format
    • (B, H, W) complex  – split into real/imag
    • (B, 1, H, W) complex – same after squeeze
    """
    # Drop a trailing singleton spatial dim left over by some loaders
    if channels.dim() == 5 and channels.size(2) == 1:
        channels = channels.squeeze(2)

    if channels.is_complex():
        return torch.stack([channels.real, channels.imag], dim=1).float()

    if channels.dim() == 4 and channels.size(1) == 2:
        return channels.float()  # already (B, 2, H, W)

    if channels.dim() == 4 and channels.size(1) == 1:
        # Single real channel — pad imaginary with zeros
        return torch.cat([channels, torch.zeros_like(channels)], dim=1).float()

    if channels.dim() == 3:
        # (B, H, W) real — pad imaginary with zeros
        return torch.stack([channels, torch.zeros_like(channels)], dim=1).float()

    raise ValueError(
        f"Cannot convert tensor of shape {tuple(channels.shape)} to (B,2,H,W). "
        "Expected (B,H,W), (B,1,H,W), (B,2,H,W), or their complex equivalents."
    )


# ---------------------------------------------------------------------------
# Helper: AWGN noise injection in real/imag domain
# ---------------------------------------------------------------------------

def add_complex_noise_ri(channels_ri: torch.Tensor, snr_db: float | None) -> torch.Tensor:
    """
    Add complex AWGN to a (B, 2, H, W) real/imag channel tensor.

    The noise power is computed per-sample so every sample has the exact
    requested SNR regardless of its absolute power.

    Args:
        channels_ri : (B, 2, H, W) float tensor – ch[:, 0] = real, ch[:, 1] = imag.
        snr_db      : Signal-to-Noise Ratio in decibels. If None, returns channels_ri unchanged.

    Returns:
        Noisy version of channels_ri with the same shape.
    """
    if snr_db is None:
        return channels_ri

    real = channels_ri[:, 0]
    imag = channels_ri[:, 1]

    # Per-sample signal power across both polarisations
    power = (real ** 2 + imag ** 2).mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = power / snr_linear
    noise_std = torch.sqrt(noise_power / 2.0)  # split equally between I and Q

    out = channels_ri.clone()
    out[:, 0] = real + torch.randn_like(real) * noise_std
    out[:, 1] = imag + torch.randn_like(imag) * noise_std
    return out


# ---------------------------------------------------------------------------
# Patching: (B, 2, H, W) -> (B, n_patches, patch_rows * patch_cols * 2)
# ---------------------------------------------------------------------------

def channels_to_patches_v11(
    channels_ri: torch.Tensor,
    patch_rows: int = 4,
    patch_cols: int = 4,
) -> torch.Tensor:
    """
    Divide the (B, 2, H, W) tensor into spatial patches, matching
    lwm1_1/input_preprocess.py's `patch_maker` exact format.

    Format:
        For each (patch_rows, patch_cols) block in the HxW grid,
        flatten the real and imaginary parts and interleave them.
        Resulting patch size = patch_rows * patch_cols * 2.

    For H=32, W=32, patch_rows=4, patch_cols=4:
        n_patches = (32/4) * (32/4) = 64
        patch_size = 4 * 4 * 2 = 32 elements per patch.

    Args:
        channels_ri : (B, 2, H, W) tensor.
        patch_rows  : Spatial rows per patch.
        patch_cols  : Spatial cols per patch.

    Returns:
        patches : (B, n_patches, patch_size) tensor.
    """
    B, C, H, W = channels_ri.shape
    assert C == 2, "channels_ri must have 2 channels (real, imag)"

    # Pad if not perfectly divisible
    pad_h = (patch_rows - (H % patch_rows)) % patch_rows
    pad_w = (patch_cols - (W % patch_cols)) % patch_cols

    if pad_h > 0 or pad_w > 0:
        channels_ri = torch.nn.functional.pad(channels_ri, (0, pad_w, 0, pad_h))
        H, W = channels_ri.shape[2:]

    # Separate real and imag
    real = channels_ri[:, 0]  # (B, H, W)
    imag = channels_ri[:, 1]  # (B, H, W)

    # Interleave real and imaginary parts along the column dimension
    # Create (B, H, W*2)
    interleaved = torch.empty(B, H, W * 2, dtype=channels_ri.dtype, device=channels_ri.device)
    interleaved[:, :, 0::2] = real
    interleaved[:, :, 1::2] = imag

    # Extract patches block by block
    n_patches_h = H // patch_rows
    n_patches_w = W // patch_cols

    patches = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            # Extract the block
            r_start = i * patch_rows
            r_end = r_start + patch_rows
            # W axis was doubled due to interleaving
            c_start = j * patch_cols * 2
            c_end = c_start + (patch_cols * 2)

            block = interleaved[:, r_start:r_end, c_start:c_end]  # (B, patch_rows, patch_cols * 2)
            patches.append(block.reshape(B, -1))                  # (B, patch_size)

    # Stack along sequence dimension
    # Shape: (B, n_patches, patch_size)
    return torch.stack(patches, dim=1)


# ---------------------------------------------------------------------------
# Masking: LWM 1.1 Masking Strategy
# ---------------------------------------------------------------------------

def mask_patches_v11(
    patches: torch.Tensor,
    mask_ratio: float = 0.40,
    gen_raw: bool = False,
) -> tuple:
    """
    Apply masking matching lwm1_1/input_preprocess.py's `make_sample` format.

    Strategy
    --------------------------------
    • Prepend a [CLS] token (constant 0.2 vector) → seq_len = n_patches + 1.
    • Select `mask_ratio` fraction of patches to mask.
    • Positional targets are selected randomly from 1 to n_patches (0 is CLS).
    • Each masked position follows BERT-style noise:
        - 80% replaced with [MASK] token (constant 0.1 vector)
        - 10% replaced with a random uniform vector
        - 10% left unchanged
    • If gen_raw=True, no masking replacement is applied.

    Args:
        patches    : (B, n_patches, patch_size) tensor.
        mask_ratio : Fraction of patches to mask (default 0.40).
        gen_raw    : If True, skip masking replacement (tokens are unchanged).

    Returns:
        input_ids    : (B, n_patches+1, patch_size)
        masked_tokens: (B, n_masks, patch_size)
        masked_pos   : (B, n_masks) – indices (1-indexed).
    """
    B, n_patches, patch_size = patches.shape

    n_masks = int(mask_ratio * n_patches)
    if n_masks < 1:
        raise ValueError(f"mask_ratio={mask_ratio} yields 0 masked tokens.")

    # --- Prepend [CLS] token (value = 0.2) ---
    cls_token = torch.full(
        (B, 1, patch_size), 0.2, device=patches.device, dtype=patches.dtype
    )
    input_ids = torch.cat([cls_token, patches], dim=1)  # (B, n_patches+1, patch_size)

    mask_vec = torch.full((patch_size,), 0.1, device=patches.device, dtype=patches.dtype)

    # --- Sample masked positions (1 to n_patches) ---
    # rand tensor shapes: (B, n_patches)
    rand = torch.rand(B, n_patches, device=patches.device)
    # Get top k indices
    # Shift by 1 because idx 0 is [CLS]
    masked_pos = rand.topk(n_masks, dim=1).indices + 1  # (B, n_masks)

    # --- Collect original values at masked positions (detached, for loss) ---
    masked_tokens = torch.gather(
        input_ids, 1,
        masked_pos.unsqueeze(-1).expand(-1, -1, patch_size)
    ).detach()  # (B, n_masks, patch_size)

    # --- Apply BERT-style masking ---
    if not gen_raw:
        rand_mask = torch.rand(B, n_masks, device=patches.device)
        # 80% [MASK], 10% random, 10% untouched
        mask_mask   = (rand_mask >= 0.1) & (rand_mask < 0.9)
        random_mask = rand_mask < 0.1

        batch_idx = (
            torch.arange(B, device=patches.device)[:, None]
            .expand_as(masked_pos)
        )

        if mask_mask.any():
            input_ids[batch_idx[mask_mask], masked_pos[mask_mask]] = mask_vec

        if random_mask.any():
            random_vals = torch.rand(
                B, n_masks, patch_size,
                device=patches.device, dtype=patches.dtype
            )
            input_ids[batch_idx[random_mask], masked_pos[random_mask]] = \
                random_vals[random_mask]

    return input_ids, masked_tokens, masked_pos


# ---------------------------------------------------------------------------
# Main model: CA + LWM 1.1 end-to-end
# ---------------------------------------------------------------------------

class LWM11WithPrepatchCA(nn.Module):
    """
    End-to-end Coordinate Attention + LWM v1.1 pretraining model.

    Architecture
    ------------
                        ┌──────────────┐
    (B,2,H,W) ──────▶  │  CoordAtt    │  (B,2,H,W)
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  Patching +  │  (B, n_patches+1, patch_size)
                        │  MCM Masking │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  LWM v1.1   │
                        │  Transformer │
                        └──────┬───────┘
                               │
                        (logits_lm, encoder_output, attn_maps)

    The CA module is trained jointly with the Transformer via the MCM loss,
    so attention weights are learned to emphasise channel features that are
    most informative for masked-token reconstruction.

    Parameters
    ----------
    patch_size         : Elements per patch (default 16, matching lwm_ca).
    mask_ratio         : Fraction of real patches to mask (default 0.15).
    gen_raw            : If True, disable masking (for inference/embedding).
    snr_db             : If not None, adds AWGN at this SNR before CA.
    coordatt_reduction : Bottleneck reduction ratio for CoordAtt (default 32).
    """

    def __init__(
        self,
        patch_rows: int = 4,
        patch_cols: int = 4,
        mask_ratio: float = 0.40,
        gen_raw: bool = False,
        snr_db: float | None = None,
        coordatt_reduction: int = 32,
    ):
        super().__init__()
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.mask_ratio = mask_ratio
        self.gen_raw = gen_raw
        self.snr_db = snr_db

        # -- CoordAtt: 2-channel in, 2-channel out (real + imag)
        self.coordatt = CoordAtt(2, 2, reduction=coordatt_reduction)

        # -- LWM v1.1 Transformer backbone
        self.lwm = LWM11()

    def forward(self, channels: torch.Tensor) -> tuple:
        """
        Args:
            channels : Raw channel tensor. Any format accepted by ensure_ri_channels().
                       Typically (B, 2, H, W) float32 from the DataLoader.

        Returns:
            logits_lm    : (B, n_masks, patch_size) – predicted patch values at
                           masked positions (used for MCM loss).
            masked_tokens: (B, n_masks, patch_size) – ground-truth patch values
                           at masked positions (target for MCM loss).
            output       : (B, seq_len, d_model) – full encoder output from LWM
                           (useful for feature extraction in downstream tasks).
        """
        # 1️⃣ Normalise to (B, 2, H, W)
        channels_ri = ensure_ri_channels(channels)

        # 2️⃣ Optional AWGN noise injection
        if self.snr_db is not None:
            channels_ri = add_complex_noise_ri(channels_ri, self.snr_db)

        # 3️⃣ Coordinate Attention — recalibrate spatial channel features
        ca_out = self.coordatt(channels_ri)          # (B, 2, H, W)

        # 4️⃣ Flatten into patches
        patches = channels_to_patches_v11(
            ca_out, patch_rows=self.patch_rows, patch_cols=self.patch_cols
        )

        # 5️⃣ MCM masking
        input_ids, masked_tokens, masked_pos = mask_patches_v11(
            patches, mask_ratio=self.mask_ratio, gen_raw=self.gen_raw
        )

        # 6️⃣ Sequence-length safety check (LWM has a max positional embedding)
        max_len = self.lwm.embedding.pos_embed.num_embeddings
        if input_ids.size(1) > max_len:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds LWM1.1 max_len={max_len}. "
                "Reduce H, W or patch size."
            )

        # 7️⃣ LWM v1.1 forward — returns (logits_lm, encoder_output, attn_maps)
        logits_lm, output, _attn_maps = self.lwm(input_ids, masked_pos)

        return logits_lm, masked_tokens, output
