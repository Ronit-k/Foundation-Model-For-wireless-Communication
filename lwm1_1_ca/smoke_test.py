"""
smoke_test.py — Quick verification that lwm1_1_ca imports and runs correctly.

Run from the project root with:
    conda activate lwm_cuda
    python lwm1_1_ca/smoke_test.py
"""
import sys
import os

# Make sure project root is on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

print("=" * 55)
print("  lwm1_1_ca Smoke Test")
print("=" * 55)

# ── Test 1: CoordAtt ─────────────────────────────────────
print("\n[1] CoordAtt ...")
from lwm1_1_ca.coordatt import CoordAtt, HSwish, HSigmoid

ca = CoordAtt(2, 2, reduction=32)
x = torch.randn(4, 2, 32, 32)
y = ca(x)
assert y.shape == x.shape, f"shape mismatch: {y.shape}"
print(f"  ✓ CoordAtt(4,2,32,32) → {tuple(y.shape)}")

# ── Test 2: Pipeline helpers ──────────────────────────────
print("\n[2] Pipeline helpers ...")
from lwm1_1_ca.torch_pipeline_v11 import (
    ensure_ri_channels,
    add_complex_noise_ri,
    channels_to_patches_v11,
    mask_patches_v11,
)

ri = torch.randn(4, 2, 32, 32)

noisy = add_complex_noise_ri(ri, snr_db=20.0)
assert noisy.shape == ri.shape
print(f"  ✓ add_complex_noise_ri → {tuple(noisy.shape)}")

patches = channels_to_patches_v11(ri, patch_rows=4, patch_cols=4)
assert patches.shape == (4, 64, 32), f"Got {patches.shape}"
print(f"  ✓ channels_to_patches  → {tuple(patches.shape)}")

input_ids, masked_tokens, masked_pos = mask_patches_v11(patches, mask_ratio=0.40)
print(f"  ✓ mask_patches         → input_ids={tuple(input_ids.shape)}, masked={tuple(masked_tokens.shape)}")

# Verify ensure_ri_channels handles several input types
for shape, desc in [
    ((3, 2, 32, 32), "(B,2,H,W)"),
    ((3, 32, 32),    "(B,H,W)"),
]:
    t = torch.randn(*shape)
    out = ensure_ri_channels(t)
    assert out.shape[1] == 2, f"Expected 2 channels, got {out.shape}"
    print(f"  ✓ ensure_ri_channels({desc}) → {tuple(out.shape)}")

# ── Test 3: Full model forward + backward ─────────────────
print("\n[3] LWM11WithPrepatchCA full forward + backward ...")
from lwm1_1_ca.torch_pipeline_v11 import LWM11WithPrepatchCA

model = LWM11WithPrepatchCA(gen_raw=False, snr_db=None)
n_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ Model built — {n_params:,} parameters")

x = torch.randn(2, 2, 32, 32)
logits, mt, output = model(x)
print(f"  ✓ logits        : {tuple(logits.shape)}")
print(f"  ✓ masked_tokens : {tuple(mt.shape)}")
print(f"  ✓ encoder output: {tuple(output.shape)}")

loss = nn.MSELoss()(logits, mt) / torch.var(mt)
loss.backward()
print(f"  ✓ MCM loss = {loss.item():.5f}  |  backward() OK")

# ── Test 4: gen_raw (no masking) ─────────────────────────
print("\n[4] gen_raw=True (inference mode, no masking) ...")
model_raw = LWM11WithPrepatchCA(gen_raw=True)
with torch.no_grad():
    logits_r, mt_r, out_r = model_raw(x)
print(f"  ✓ logits={tuple(logits_r.shape)}, output={tuple(out_r.shape)}")

# ── Test 5: Package __init__ exports ─────────────────────
print("\n[5] Package __init__ exports ...")
from lwm1_1_ca import CoordAtt, LWM11WithPrepatchCA  # noqa
print("  ✓ CoordAtt, LWM11WithPrepatchCA importable from lwm1_1_ca")

print()
print("=" * 55)
print("  All tests PASSED ✓")
print("=" * 55)
