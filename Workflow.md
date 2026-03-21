# CA-LWM Training & Benchmarking

This folder contains training, inference utilities, and benchmarking scripts for a **Coordinate-Attention (CA) augmented Large Wireless Model (LWM)**. It includes:
- End-to-end CA + LWM pretraining using a torch-native patching pipeline.
- Benchmarking against a base LWM on downstream tasks (LoS/NLoS, Beam Prediction).
- Plotting helpers for benchmark results.

> Run scripts from this directory (`CA_LWM_Traing`) so the default relative paths to checkpoints and outputs resolve correctly.

---

## Repository Layout

```
CA_LWM_Traing/
  lwm/                 # Base LWM model + preprocessing utilities
  lwm_ca/              # CA modules + e2e pretraining + benchmark scripts
  results/             # Benchmark CSVs and plots (outputs)
  runs/                # TensorBoard logs (optional)
```

Key files:
- `lwm_ca/pretraining_e2e.py` — end-to-end CA + LWM pretraining.
- `lwm_ca/benchmark.py` — compare base LWM vs CA-LWM on downstream tasks.
- `lwm_ca/plot_benchmark_results.py` — plot benchmark CSVs.

---

## Requirements

Recommended environment:
- Python 3.9+
- PyTorch (CUDA optional but recommended)
- NumPy, tqdm, matplotlib
- pandas (used by some utilities)
- DeepMIMOv3 (for scenario generation)

Example (adjust as needed):
```
pip install torch numpy tqdm matplotlib pandas
```
DeepMIMOv3 is required to generate scenarios. Ensure it is installed and the dataset scenarios are available on disk.

---

## Dataset Setup

The scripts rely on DeepMIMO scenarios on disk.

Default dataset location used by scripts:
```
/home/audbhav22/foundation_model/Dataset
```

You can change the dataset location in two ways:
1) Pass `--dataset-folder` to scripts.
2) Set `LWM_SCENARIOS_DIR` (used by some utilities in `lwm/input_preprocess.py`).

Example:
```
export LWM_SCENARIOS_DIR=/path/to/DeepMIMO
```

---

## End-to-End CA + LWM Pretraining

Script: `lwm_ca/pretraining_e2e.py`

This performs:
- Loading DeepMIMO channels
- Pre-patch coordinate attention (CA)
- Torch-native patching + masking
- LWM training on the masked channel modeling task

Example:
```
python lwm_ca/pretraining_e2e.py \
  --scenarios O1_3p5_v1 O1_3p5_v2 Boston5G_3p5 \
  --epochs 100 \
  --batch-size 64 \
  --device cuda \
  --snr-db 20 \
  --save-path lwm_ca/model_weights_ca_e2e.pth
```

Useful options:
- `--channels-cache PATH.npy` caches the stacked channel tensor to avoid repeated DeepMIMO generation.
- `--tensorboard --tb-logdir runs/lwm_ca_pretraining` for logging.
- `--torch-compile` to enable `torch.compile` (PyTorch 2.x).

Output:
- Checkpoint at `--save-path` (default: `lwm_ca/model_weights_ca_e2e.pth`).

---

## Benchmarking (Base LWM vs CA-LWM)

Script: `lwm_ca/benchmark.py`

This compares downstream performance using:
- **Task**: `los` (LoS/NLoS Classification) or `beam` (Beam Prediction)
- **Input types**: `raw`, `cls_emb`, `channel_emb`
- **Splits**: configurable train/test ratios

Example:
```
python lwm_ca/benchmark.py \
  --scenarios O1_3p5B \
  --task beam \
  --n-beams 16 \
  --input-types cls_emb channel_emb \
  --split-ratios 0.05 0.1 0.2 \
  --epochs 100 \
  --batch-size 512 \
  --compare base ca \
  --ca-mode e2e \
  --base-ckpt ./lwm/model_weights.pth \
  --ca-ckpt ./lwm_ca/model_weights_ca_e2e.pth \
  --save-csv results/benchmark_results.csv
```

Notes:
- The benchmark now loads DeepMIMO data **once per run** and reuses it for labels and tokenization.
- `--ca-mode prepatch` uses pre-patch CA (`tokenizer_ca`).
- `--ca-mode e2e` uses the end-to-end CA model (`LWMWithPrepatchCA`).

Output:
- CSV at `--save-csv` (default: `results/benchmark_results.csv`).

---

## Plotting Benchmark Results

Script: `lwm_ca/plot_benchmark_results.py`

Example:
```
python lwm_ca/plot_benchmark_results.py \
  --csv results/benchmark_results.csv \
  --out results/benchmark_plot.png \
  --title "Benchmark F1 vs Split Ratio"
```

If `--out` is omitted, a plot window is shown instead.

---

## Checkpoints & Outputs

Default checkpoints (relative to this folder):
- Base LWM: `./lwm/model_weights.pth`
- CA-LWM (e2e): `./lwm_ca/model_weights_ca_e2e.pth`

Outputs:
- Benchmark CSVs: `results/benchmark_results.csv`
- Plots: `results/*.png` (if you save plots)

---

## Troubleshooting

- **DeepMIMOv3 not found**: ensure it is installed in your Python environment.
- **Dataset not found**: confirm `--dataset-folder` points to the DeepMIMO scenario root.
- **CUDA out of memory**: lower `--batch-size` or use `--device cpu`.
- **Slow generation**: use `--channels-cache` in pretraining to reuse stacked channels.

---

## Typical Workflow

1) Pretrain CA-LWM:
```
python lwm_ca/pretraining_e2e.py --save-path lwm_ca/model_weights_ca_e2e.pth
```

2) Benchmark against base LWM:
```
python lwm_ca/benchmark.py --compare base ca --save-csv results/benchmark_results.csv
```

3) Plot results:
```
python lwm_ca/plot_benchmark_results.py --csv results/benchmark_results.csv --out results/benchmark_plot.png
```




```bash
python3 -m lwm_axial.pretraining_axial     --save-path lwm_axial/model_weights_axial.pth     --tb-logdir runs/lwm_axial_pretraining     --batch-size 512     --channels-cache /tmp/channels_ri.npy --save-every 5 --no-scheduler-step-per-batch --tensorboard  --lr 1e-4
```

---

## Dataset Details: O1_3p5B

The `O1_3p5B` scenario is a key dataset for benchmarking `LWM`.

- **Scenario Type**: Outdoor 3.5GHz
- **Dimensions**:
  - **Rows**: 2751
  - **Users per Row**: 181
  - **Total Users**: ~498,000
- **Antenna Configuration**:
  - **Base Station**: 32 Antennas (Vertical array)
  - **User Equipment**: 1 Antenna
- **OFDM**:
  - **Subcarriers**: 32
  - **Bandwidth**: Scaled based on SCS (30kHz)
- **Path Features**:
  - **Grid**: Random uniform sampling capable
  - **Multipath**: Top 20 paths considered