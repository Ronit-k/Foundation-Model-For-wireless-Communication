import torch
import torch.nn as nn

from .coordatt import CoordAtt
from lwm.lwm_model import lwm


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
    """Convert (B, 2, H, W) into patches (B, n_patches, patch_size)."""
    batch_size, _, height, width = channels_ri.shape
    flat = channels_ri.view(batch_size, 2, height * width)
    flat = torch.cat([flat[:, 0], flat[:, 1]], dim=1)
    if flat.size(1) % patch_size != 0:
        raise ValueError("Flattened channel length is not divisible by patch size.")
    return flat.view(batch_size, -1, patch_size)


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

    cls_token = torch.full(
        (batch_size, 1, patch_size), 0.2, device=patches.device, dtype=patches.dtype
    )
    input_ids = torch.cat([cls_token, patches], dim=1)

    n_masks = n_masks_half * 2
    mask_vec = torch.full((patch_size,), 0.1, device=patches.device, dtype=patches.dtype)

    # Sample masked positions without replacement in the real half.
    rand = torch.rand(batch_size, real_tokens, device=patches.device)
    pos_real = rand.topk(n_masks_half, dim=1).indices
    pos_imag = pos_real + real_tokens
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


class LWMWithPrepatchCA(nn.Module):
    """End-to-end CA + LWM model with torch-native patching and masking."""

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

        max_len = self.lwm.embedding.pos_embed.num_embeddings
        if input_ids.size(1) > max_len:
            raise ValueError(f"Sequence length {input_ids.size(1)} exceeds max_len {max_len}.")

        logits_lm, output = self.lwm(input_ids, masked_pos)
        return logits_lm, masked_tokens, output
