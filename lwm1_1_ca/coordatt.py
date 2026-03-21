# =============================================================================
# coordatt.py  —  Coordinate Attention module for LWM 1.1 + CA pretraining
#
# Coordinate Attention (CA) learns to encode spatial location information into
# channel attention maps, separately along the Height (H) and Width (W) axes.
# Unlike SENet (global pooling), CA captures directional & positional context,
# making it well-suited for channel matrices shaped (N, 2, H, W) where H = n_ant
# and W = n_subcarriers.
#
# Reference: Hou et al., "Coordinate Attention for Efficient Mobile Network
#             Design", CVPR 2021.
#
# This file is identical to lwm_ca/coordatt.py — it is kept as a self-contained
# module inside lwm1_1_ca so no cross-package dependency on lwm_ca is needed.
# =============================================================================

import torch
import torch.nn as nn


class HSigmoid(nn.Module):
    """
    Hard-Sigmoid approximation: relu6(x+3)/6.
    Faster than the standard sigmoid, avoids exp() computation,
    and retains a piecewise-linear shape that is easier to quantise.
    """

    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3) / 6


class HSwish(nn.Module):
    """
    Hard-Swish: x * HSigmoid(x).
    Used as the internal activation inside CoordAtt.
    """

    def __init__(self, inplace: bool = True):
        super().__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """
    Coordinate Attention block operating on (N, C, H, W) tensors.

    For LWM 1.1 pretraining the input is the real/imaginary channel tensor
    of shape (N, 2, 32, 32) — 2 feature channels, 32 antennas × 32 subcarriers.

    Mechanism
    ---------
    1. Pool H-direction: AdaptiveAvgPool2d((None, 1))  -> shape (N, C, H, 1)
    2. Pool W-direction: AdaptiveAvgPool2d((1, None))  -> shape (N, C, 1, W)
       Then permute to (N, C, W, 1) and concatenate with H-pool along H.
    3. Shared 1×1 conv (with BN + HSwish) reduces channels to mid_channels.
    4. Split back into [h, w] halves; apply separate 1×1 convs → sigmoid → scales.
    5. Output = identity * a_h * a_w  (spatial + channel recalibration).

    Parameters
    ----------
    in_channels  : Number of input feature channels (C_in).
    out_channels : Number of output feature channels (C_out, usually = C_in).
    reduction    : Reduction ratio for the bottleneck (default 32).
                   mid_channels = max(8, in_channels // reduction).
    """

    def __init__(self, in_channels: int, out_channels: int, reduction: int = 32):
        super().__init__()

        # Directional average-pooling
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # collapses W -> (N, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # collapses H -> (N, C, 1, W)

        # Bottleneck: shared 1×1 conv that processes the concatenated H+W context
        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = HSwish()

        # Separate 1×1 convs to produce directional attention maps
        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor of same shape (N, C, H, W), recalibrated by CA.
        """
        identity = x  # save for the element-wise multiply at the end
        _, _, h, w = x.shape

        # --- Step 1: pool along each spatial axis separately ---
        x_h = self.pool_h(x)                        # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)    # (N, C, W, 1)  <- transposed

        # --- Step 2: concatenate height and width contexts along 'height' dim ---
        y = torch.cat([x_h, x_w], dim=2)            # (N, C, H+W, 1)

        # --- Step 3: shared transformation ---
        y = self.conv1(y)   # (N, mid, H+W, 1)
        y = self.bn1(y)
        y = self.act(y)

        # --- Step 4: split back and compute attention weights ---
        x_h, x_w = torch.split(y, [h, w], dim=2)   # (N,mid,H,1), (N,mid,W,1)
        x_w = x_w.permute(0, 1, 3, 2)               # (N, mid, 1, W)

        a_h = torch.sigmoid(self.conv_h(x_h))       # (N, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(x_w))       # (N, C, 1, W)

        # --- Step 5: recalibrate ---
        # Broadcasting: a_h * a_w produces (N, C, H, W) attention map
        return identity * a_h * a_w
