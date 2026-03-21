import torch
import torch.nn as nn

# Import Coordinate Attention from the original folder (unchanged)
try:
    from lwm_ca.coordatt import CoordAtt
except ImportError:
    # Fallback if running from a different root
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lwm_ca.coordatt import CoordAtt

# Import the NEW Axial LWM model
from .lwm_axial_model import lwm


def ensure_ri_channels(channels):
    """
    Normalize channels to (B, 2, H, W) float tensor (real, imag).
    Accepts complex (B, H, W) or real (B, 2, H, W).
    """
    if channels.dim() == 5 and channels.size(2) == 1:
        channels = channels.squeeze(2)
    if channels.is_complex():
        real = channels.real
        imag = channels.imag
        return torch.stack([real, imag], dim=1).float()

    if channels.dim() == 4 and channels.size(1) == 2:
        return channels.float()

    if channels.dim() == 4 and channels.size(1) == 1:
        zeros = torch.zeros_like(channels)
        return torch.cat([channels, zeros], dim=1).float()

    if channels.dim() == 3:
        zeros = torch.zeros_like(channels)
        return torch.stack([channels, zeros], dim=1).float()

    raise ValueError(f"Unsupported channel shape: {tuple(channels.shape)}")


def add_complex_noise_ri(channels_ri, snr_db):
    """Add complex Gaussian noise to (B, 2, H, W) real/imag channels."""
    real = channels_ri[:, 0]
    imag = channels_ri[:, 1]
    power = (real**2 + imag**2).mean(dim=(1, 2), keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = power / snr_linear
    noise_std = torch.sqrt(noise_power / 2)

    noise_real = torch.randn_like(real) * noise_std
    noise_imag = torch.randn_like(imag) * noise_std

    out = channels_ri.clone()
    out[:, 0] = real + noise_real
    out[:, 1] = imag + noise_imag
    return out


def channels_to_patches(channels_ri, patch_size=16):
    """
    Convert (B, 2, Ant, Sub) into patches ordered by [Antenna, Component, Freq].
    Input: (B, 2, 32, 32) real/imag.
    Output: (B, 128, 16) flattened patches.
    sequence: Ant0_Real_P0, Ant0_Real_P1, Ant0_Imag_P0, Ant0_Imag_P1, Ant1...
    """
    batch_size, channels, n_ant, n_sub = channels_ri.shape
    
    # 1. Permute to [B, Ant, 2, Sub]
    # We want Antenna as the outer dimension.
    x = channels_ri.permute(0, 2, 1, 3) # [B, 32, 2, 32]
    
    # 2. Unfold subcarriers into patches
    # Sub (32) -> 2 patches of size 16
    # [B, 32, 2, 2, 16]
    n_patches_per_sub = n_sub // patch_size
    x = x.reshape(batch_size, n_ant, 2, n_patches_per_sub, patch_size)
    
    # 3. Flatten dimensions 2 and 3 (Component and FreqPatch)
    # [B, 32, 4, 16]
    # The inner order is Real_P0, Real_P1, Imag_P0, Imag_P1
    x = x.reshape(batch_size, n_ant, 2 * n_patches_per_sub, patch_size)
    
    # 4. Flatten all patches
    # [B, 128, 16]
    return x.reshape(batch_size, -1, patch_size)


def mask_patches(patches, mask_ratio=0.15, gen_raw=False):
    """
    Apply MCM-style masking to patches.

    Returns input_ids, masked_tokens, masked_pos.
    """
    batch_size, n_patches, patch_size = patches.shape
    real_tokens = n_patches // 2
    n_masks_half = int(mask_ratio * real_tokens)
    if n_masks_half < 1:
        raise ValueError("Mask ratio yields zero masked tokens.")

    # Use learned CLS/Mask embeddings passed from model, or default fixed (legacy)
    # Ideally, masking happens AFTER embedding in the model using learnable tokens.
    # But for compatibility with this pipeline, we use fixed placeholders here
    # and let the model potentially replace them or learn from them.
    # For now, we stick to the fixed values to minimize regression, 
    # BUT we will rely on the model to use learnable embeddings if implemented there.
    # The User requested "CLS = 0.2 vector" fix. 
    # We will just mark positions here and let the model handle the embedding replacement?
    # No, this function returns input_ids (values). 
    # We will continue returning values, but the Model should likely override the Embedding for these special positions.
    
    cls_token_val = 0.2
    mask_token_val = 0.1
    
    cls_token = torch.full(
        (batch_size, 1, patch_size), cls_token_val, device=patches.device, dtype=patches.dtype
    )
    input_ids = torch.cat([cls_token, patches], dim=1)

    n_masks = n_masks_half * 2
    mask_vec = torch.full((patch_size,), mask_token_val, device=patches.device, dtype=patches.dtype)

    # Real tokens are at indices [0, 1] of every 4-block.
    # Total patches = Ant * 4.
    # We want to sample N distinct (Ant, SubFreq) pairs.
    # Ant in range [0, 32), SubFreq in range [0, 2).
    # Then pos_real = Ant * 4 + SubFreq.
    # Then pos_imag = pos_real + 2.
    
    rand = torch.rand(batch_size, real_tokens, device=patches.device) # [B, 64]
    
    # We need to map [0..63] to the valid Real indices.
    # valid_real_indices = [0, 1, 4, 5, 8, 9, ...]
    # Formula: idx -> (idx // 2) * 4 + (idx % 2)
    
    # Select top-k indices from the range [0..63]
    selected_indices = rand.topk(n_masks_half, dim=1).indices # [B, N_Masks_Half] (values 0..63)
    
    # Map to physical indices
    pos_real = (selected_indices // 2) * 4 + (selected_indices % 2)
    pos_imag = pos_real + 2
    
    masked_pos = torch.cat([pos_real, pos_imag], dim=1) + 1  # shift for CLS
    
    masked_tokens = torch.gather(
        input_ids, 1, masked_pos.unsqueeze(-1).expand(-1, -1, patch_size)
    ).detach()

    if not gen_raw:
        rand_mask = torch.rand(batch_size, n_masks, device=patches.device)
        random_mask = rand_mask < 0.1
        mask_mask = (rand_mask >= 0.1) & (rand_mask < 0.9)

        batch_idx = torch.arange(batch_size, device=patches.device)[:, None].expand_as(
            masked_pos
        )
        if mask_mask.any():
            input_ids[batch_idx[mask_mask], masked_pos[mask_mask]] = mask_vec
        if random_mask.any():
            random_vals = torch.rand(
                batch_size,
                n_masks,
                patch_size,
                device=patches.device,
                dtype=patches.dtype,
            )
            input_ids[batch_idx[random_mask], masked_pos[random_mask]] = random_vals[
                random_mask
            ]

    return input_ids, masked_tokens, masked_pos


class LWMWithPrepatchAxial(nn.Module):
    """End-to-end CA + Axial LWM model with torch-native patching and masking."""

    def __init__(
        self,
        patch_size=16,
        mask_ratio=0.15,
        gen_raw=False,
        snr_db=None,
        coordatt_reduction=32,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.gen_raw = gen_raw
        self.snr_db = snr_db
        self.coordatt = CoordAtt(2, 2, reduction=coordatt_reduction)
        self.lwm = lwm()

    def forward(self, channels):
        channels_ri = ensure_ri_channels(channels)
        if self.snr_db is not None:
            channels_ri = add_complex_noise_ri(channels_ri, self.snr_db)

        ca_out = self.coordatt(channels_ri)
        patches = channels_to_patches(ca_out, patch_size=self.patch_size)
        input_ids, masked_tokens, masked_pos = mask_patches(
            patches, mask_ratio=self.mask_ratio, gen_raw=self.gen_raw
        )

        max_len = (self.lwm.embedding.antennas * self.lwm.embedding.freq_groups) + 1
        if input_ids.size(1) > max_len:
            raise ValueError(f"Sequence length {input_ids.size(1)} exceeds max_len {max_len}.")

        logits_lm, output = self.lwm(input_ids, masked_pos)
        return logits_lm, masked_tokens, output
