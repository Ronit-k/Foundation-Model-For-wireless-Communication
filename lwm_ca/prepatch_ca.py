import numpy as np
import torch

from .coordatt import CoordAtt


def build_coordatt(in_channels=2, out_channels=2, reduction=32, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return CoordAtt(in_channels, out_channels, reduction=reduction).to(device)


def _normalize_channel_shape(channels):
    data = np.asarray(channels)
    if data.ndim == 4 and data.shape[1] == 1:
        return data[:, 0]
    if data.ndim != 3:
        raise ValueError(f"Expected shape (N,H,W) or (N,1,H,W). Got {data.shape}.")
    return data


def apply_coordatt_prepatch(channels, coordatt=None, device=None):
    """
    Apply coordinate attention to complex channel maps before patching.

    Args:
        channels (np.ndarray or torch.Tensor): Complex or real array with shape (N,H,W).
        coordatt (CoordAtt, optional): Prebuilt CoordAtt module.
        device (str, optional): Torch device for CoordAtt.

    Returns:
        np.ndarray: Complex array with shape (N,H,W) after CA.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if coordatt is None:
        coordatt = build_coordatt(device=device)

    if isinstance(channels, torch.Tensor):
        data = channels.detach().cpu().numpy()
    else:
        data = np.asarray(channels)

    data = _normalize_channel_shape(data)

    if np.iscomplexobj(data):
        real = data.real
        imag = data.imag
    else:
        real = data
        imag = np.zeros_like(data)

    stacked = np.stack([real, imag], axis=1)  # (N, 2, H, W)
    tensor = torch.from_numpy(stacked).float().to(device)

    coordatt = coordatt.to(device)
    coordatt.eval()
    with torch.no_grad():
        out = coordatt(tensor)

    out_np = out.cpu().numpy()
    out_real = out_np[:, 0]
    out_imag = out_np[:, 1]
    return (out_real + 1j * out_imag).astype(np.csingle)
