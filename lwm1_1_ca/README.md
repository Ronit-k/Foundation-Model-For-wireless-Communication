# LWM v1.1 + Coordinate Attention (CA) — Pretraining Package

> **`lwm1_1_ca/`** — Self-supervised pretraining of Large Wireless Model v1.1 with Coordinate Attention, adapted from `lwm_ca/` for the newer LWM 1.1 architecture.

---

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Datasets](#datasets)
4. [Project Structure](#project-structure)
5. [Requirements](#requirements)
6. [Pretraining](#pretraining)
7. [Downstream Evaluation (SECNN)](#downstream-evaluation-secnn)
8. [Hyperparameters](#hyperparameters)
9. [Output Files](#output-files)
10. [How It Compares to lwm_ca](#how-it-compares-to-lwm_ca)

---

## Overview

This package inserts a **Coordinate Attention (CA)** module between the raw wireless channel and the patching step of LWM v1.1, then pre-trains the joint model using **Masked Channel Modelling (MCM)** — the same self-supervised task as the original LWM.

The CA module learns to selectively recalibrate the antenna × subcarrier channel matrix along both spatial axes before it is tokenised and fed into the LWM Transformer. This allows the backbone to focus on the most informative spatial features in the channel, improving downstream task performance.

The downstream evaluation uses a custom **SECNN** (Squeeze-Excitation CNN) head for the beam prediction task, comparing:
- **base_lwm1_1**: original pretrained LWM v1.1 + SECNN head
- **ca_lwm1_1**: CA-pretrained LWM v1.1 + SECNN head

---

## Architecture

```
Raw Channel (N, 2, H, W)
         │
         ▼
  ┌─────────────┐
  │  CoordAtt   │  ← Coordinate Attention (H=32 antennas, W=32 subcarriers)
  │  (2→2 ch)   │    Learns H-direction and W-direction attention separately
  └──────┬──────┘
         │ (N, 2, H, W) — spatially recalibrated
         ▼
  ┌─────────────┐
  │  Patching   │  4x4 spatial blocks, interleaved real/imag
  │  + CLS      │  patch_size = 32, n_patches = 64
  └──────┬──────┘
         │
         ▼
  ┌──────────────────┐
  │  MCM Masking     │  40% random uniform masking
  │  (BERT-style)    │  80% → [MASK], 10% → random, 10% → unchanged
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │  LWM v1.1        │  12-layer Transformer Encoder
  │  Transformer     │  d_model=128, 8 heads, max_len=513
  └──────┬───────────┘
         │
         ▼
  (logits_lm, masked_tokens, output)
         │
  Loss: MSE(logits_lm, masked_tokens) / Var(masked_tokens)
```

### Coordinate Attention Detail

CoordAtt (Hou et al., CVPR 2021) encodes location information into channel attention by:
1. Pooling along H → captures row (antenna) context
2. Pooling along W → captures column (subcarrier) context
3. Concatenating and transforming jointly via a shared 1×1 conv + BN + HSwish
4. Producing separate H-direction and W-direction attention maps via sigmoid
5. Multiplying the input by both maps: `out = x * a_h * a_w`

Unlike SENet (global pooling only), CoordAtt preserves spatial structure, making it well-suited to the antenna × subcarrier layout of OFDM channel matrices.

---

## Datasets

### Pretraining Scenarios

The pretraining leverages a superset of 25 DeepMIMO scenarios from both `lwm` and `lwm1_1`, ensuring the most comprehensive pretraining possible:

| Scenario Base | Variants Included | Antennas | Subcarriers |
|---|---|---|---|
| `O1_3p5` | Base, B, v1, v2 | 32 | 32 |
| `Boston5G_3p5` | Base | 32 | 32 |
| `asu_campus1` | Base | 32 | 32 |
| `city_X` | Cities 0-15, 17-19 | 32 | 32 |

All data files are in `../scenarios/` relative to the project root.

**Channel dimensions**: Each scenario produces channels of shape (N_users, 1, 32, 32) — 32 BS antennas × 32 OFDM subcarriers. After cleaning and scaling by 1e6, the real and imaginary parts are stacked to form (N, 2, 32, 32).

### Downstream Evaluation Scenario

Default: `city_0_newyork`, BS index 3, 64 beams. Configurable via CLI.

---

## Project Structure

```
lwm1_1_ca/
├── __init__.py               # Package exports
├── coordatt.py               # Coordinate Attention (CoordAtt, HSwish, HSigmoid)
├── torch_pipeline_v11.py     # End-to-end pipeline for LWM 1.1 (channel→CA→patch→mask→LWM)
├── pretraining.py            # Self-supervised pretraining script (CLI)
├── benchmark_secnn.py        # Downstream beam prediction benchmark using SECNN
├── model_weights_ca_e2e.pth  # Trained weights (generated after pretraining)
├── model_weights_ca_e2e.log  # Training log (generated after pretraining)
└── results/
    ├── benchmark_secnn.csv   # Benchmark results table
    └── benchmark_secnn.png   # F1 vs split ratio plot
```

---

## Requirements

```
python >= 3.9
torch >= 1.12
numpy
tqdm
matplotlib
scikit-learn
DeepMIMOv3
```

Install: `pip install torch numpy tqdm matplotlib scikit-learn`

DeepMIMOv3 must be installed separately. Scenario data must be present in `../scenarios/`.

---

## Pretraining

Run from the **project root** (`Foundation-Model-For-Wireless-Communication/`):

```bash
# Default: 100 epochs, batch=64, all 16 scenarios, CUDA auto-detected
python -m lwm1_1_ca.pretraining

# Full explicit command (same defaults):
python -m lwm1_1_ca.pretraining \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --step-size 10 \
    --gamma 0.9 \
    --seed 0 \
    --train-ratio 0.8 \
    --val-ratio 0.2 \
    --save-path lwm1_1_ca/model_weights_ca_e2e.pth

# With channel caching (recommended for repeated runs):
python -m lwm1_1_ca.pretraining \
    --channels-cache /tmp/lwm1_1_channels.npy

# With TensorBoard:
python -m lwm1_1_ca.pretraining \
    --tensorboard \
    --tb-logdir runs/lwm1_1_ca_pretraining

# CPU smoke test (single epoch, small batch):
python -m lwm1_1_ca.pretraining \
    --scenarios city_0_newyork \
    --epochs 1 \
    --batch-size 8 \
    --device cpu \
    --save-path /tmp/test_weights.pth
```

### Key CLI Options

| Option | Default | Description |
|---|---|---|
| `--scenarios` | all 16 | Space-separated scenario names |
| `--epochs` | 100 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-4 | Adam learning rate |
| `--weight-decay` | 1e-5 | Adam weight decay |
| `--step-size` | 10 | StepLR step size |
| `--gamma` | 0.9 | StepLR decay factor |
| `--snr-db` | None | AWGN noise level (None = no noise) |
| `--channels-cache` | None | Cache path to avoid re-generating data |
| `--amp` | False | Enable Automatic Mixed Precision |
| `--save-path` | `lwm1_1_ca/model_weights_ca_e2e.pth` | Model checkpoint path |
| `--save-every` | 0 | Epoch interval for intermediate checkpoints |
| `--log-interval` | 50 | Steps between step-level log prints |

---

## Downstream Evaluation (SECNN)

```bash
# Compare base LWM 1.1 vs CA-pretrained LWM 1.1 on beam prediction
python -m lwm1_1_ca.benchmark_secnn \
    --scenario city_0_newyork \
    --bs-idx 3 \
    --n-beams 64 \
    --input-types cls_emb channel_emb raw \
    --split-ratios 0.005 0.01 0.05 0.1 0.2 0.4 \
    --epochs 30 \
    --batch-size 32 \
    --base-ckpt lwm1_1/model.pth \
    --ca-ckpt lwm1_1_ca/model_weights_ca_e2e.pth \
    --save-csv lwm1_1_ca/results/benchmark_secnn.csv \
    --save-plot lwm1_1_ca/results/benchmark_secnn.png
```

### SECNN Architecture

The Squeeze-Excitation CNN (SECNN) downstream head uses SENet-style channel attention inside each residual block:

```
Input → Conv1d(stem) → MaxPool
      → SEResBlock Stage1 (32 ch)
      → SEResBlock Stage2 (64 ch)
      → SEResBlock Stage3 (128 ch)
      → GlobalAvgPool → FC(128) → BN → ReLU → Dropout
      → FC(num_classes)
```

Each `SEResBlock` adds a Squeeze-Excitation layer after the second conv:
1. **Squeeze**: global average pool → (B, C, 1)
2. **Excitation**: FC(C→C/8) → ReLU → FC(C/8→C) → Sigmoid
3. **Recalibrate**: multiply feature map by excitation weights

This is distinct from the plain `res1dcnn` in `lwm1_1/main.py` — the SE blocks provide additional feature recalibration that complements the spatial attention introduced by CoordAtt in the backbone.

---

## Hyperparameters

All pretraining hyperparameters are identical to `lwm_ca/pretraining_e2e.py` for a fair comparison:

| Parameter | Value |
|---|---|
| Epochs | 100 |
| Batch size | 64 |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| LR scheduler | StepLR |
| Step size | 10 epochs |
| Gamma | 0.9 |
| Train/Val split | 80% / 20% |
| Patch grid / size | 4x4 spatial / 32 elements |
| MCM mask ratio | 40% uniform random |
| CLS token value | 0.2 |
| MASK token value | 0.1 |
| Loss | MSE / Var (normalised) |
| Seed | 0 |

---

## Output Files

After pretraining:

| File | Description |
|---|---|
| `model_weights_ca_e2e.pth` | Full model state dict (CA + LWM 1.1) |
| `model_weights_ca_e2e.log` | Training log with loss, LR, GPU memory per epoch |

After benchmark:

| File | Description |
|---|---|
| `results/benchmark_secnn.csv` | Test Acc, Top-3 Acc, F1 for all models/ratios |
| `results/benchmark_secnn.png` | F1 vs split ratio comparison plot |

---

## How It Compares to `lwm_ca`

| Aspect | `lwm_ca` | `lwm1_1_ca` |
|---|---|---|
| LWM backbone | LWM 1.0 (`lwm.lwm_model.lwm`) | LWM 1.1 (`lwm1_1.lwm_model.lwm`) |
| `forward()` return | `(logits, output)` 2-tuple | `(logits, output, attn_maps)` 3-tuple |
| Data API | `lwm.input_preprocess.DeepMIMO_data_gen(name)` | same (shared code) |
| Patch format | flat real\|imag, patch_size=16 | 4x4 interleaved, patch_size=32 |
| MCM masking | 15% symmetric | 40% uniform random |
| Loss | MSE / Var | identical |
| Downstream model | FCN (from lwm.utils) | SECNN (custom SE-Residual CNN) |
| Scenarios | 16 scenarios | 25 scenarios (superset) |
| Hyperparameters | all defaults | all matched |

The only intentional differences are:
1. The LWM backbone version (1.0 → 1.1).
2. The downstream evaluation model (FCN → SECNN).

This makes the performance comparison between `lwm_ca` and `lwm1_1_ca` directly meaningful.
