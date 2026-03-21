import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Setup path to import from root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None

try:
    from torch import amp as torch_amp
except Exception:  # pragma: no cover - fallback for older torch
    torch_amp = None

# Imports
from lwm.input_preprocess import DeepMIMO_data_gen, deepmimo_data_cleaning
from lwm_axial.torch_pipeline_axial import LWMWithPrepatchAxial


def default_num_workers():
    # 24 cores available. Start with 8 workers to keep GPU fed without oversubscription.
    return 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end pretraining for Axial LWM with Coordinate Attention."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
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
        help="DeepMIMO scenarios to use for pretraining.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to write training logs (default: save_path + .log).",
    )
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--tb-logdir",
        type=str,
        default="runs/lwm_axial_pretraining",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--scheduler-step-per-batch", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        default="/home/audbhav22/foundation_model/Dataset",
        help="Path to DeepMIMO scenarios root (contains scenario folders).",
    )
    parser.add_argument(
        "--channels-cache",
        type=str,
        default=None,
        help="Optional path to cache the stacked channels (np.save/.npy).",
    )
    parser.add_argument("--save-path", type=str, default="lwm_axial/model_weights_axial.pth")
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args()


def load_channels_ri(scenarios, dataset_folder=None, cache_path=None):
    if cache_path:
        cache_path = os.path.expanduser(cache_path)
        load_path = cache_path
        if not os.path.exists(load_path):
            if not load_path.endswith(".npy") and os.path.exists(load_path + ".npy"):
                load_path = load_path + ".npy"
            else:
                load_path = None
        if load_path:
            print(f"Loading cached channels from {load_path}")
            return np.load(load_path)

    print("Loading data from scenarios...")
    data = []
    for name in scenarios:
        deepmimo_data = DeepMIMO_data_gen(name, dataset_folder=dataset_folder)
        cleaned = deepmimo_data_cleaning(deepmimo_data)
        data.append(cleaned)
    channels = np.vstack(data)
    real = channels.real.astype(np.float32)
    imag = channels.imag.astype(np.float32)
    channels_ri = np.stack([real, imag], axis=1)
    if cache_path:
        cache_path = os.path.expanduser(cache_path)
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, channels_ri)
        print(f"Cached channels to {cache_path}")
    return channels_ri


def split_data(dataset, train_ratio, val_ratio, seed=0):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - val_size - train_size
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler=None,
    device="cuda",
    amp=False,
    scaler=None,
    log_interval=0,
    log_fn=print,
    writer=None,
    epoch_idx=0,
    non_blocking=False,
    scheduler_step_per_batch=True,
    grad_clip=0.0,
):
    model.train()
    running_loss = 0.0
    criterion = nn.MSELoss()
    data_time_sum = 0.0
    step_time_sum = 0.0
    samples_sum = 0
    log_loss = 0.0
    log_data_time = 0.0
    log_step_time = 0.0
    log_samples = 0

    end = time.perf_counter()

    for step, (channels,) in enumerate(dataloader):
        data_time = time.perf_counter() - end
        data_time_sum += data_time
        log_data_time += data_time

        channels = channels.to(device, non_blocking=non_blocking)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        if torch_amp is not None:
            autocast_ctx = torch_amp.autocast(device_type="cuda", enabled=amp)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=amp)
        with autocast_ctx:
            logits_lm, masked_tokens, _ = model(channels)
            loss = criterion(logits_lm, masked_tokens) / torch.var(masked_tokens)

        if amp and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None and scheduler_step_per_batch:
            scheduler.step()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        step_time = time.perf_counter() - start
        step_time_sum += step_time
        log_step_time += step_time

        batch_size = channels.size(0)
        samples_sum += batch_size
        log_samples += batch_size

        running_loss += loss.item()
        log_loss += loss.item()

        if log_interval and (step + 1) % log_interval == 0:
            avg_loss = log_loss / log_interval
            avg_data = log_data_time / log_interval
            avg_step = log_step_time / log_interval
            throughput = log_samples / log_step_time if log_step_time > 0 else 0.0
            if writer is not None:
                global_step = epoch_idx * len(dataloader) + step + 1
                writer.add_scalar("train/loss_step", avg_loss, global_step)
                writer.add_scalar("train/data_time_ms", avg_data * 1000, global_step)
                writer.add_scalar("train/step_time_ms", avg_step * 1000, global_step)
                writer.add_scalar("train/throughput", throughput, global_step)
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], global_step
                )
            log_fn(
                f"  step {step + 1:>5}/{len(dataloader)} | "
                f"loss {avg_loss:.4f} | "
                f"data {avg_data * 1000:.1f} ms | "
                f"step {avg_step * 1000:.1f} ms | "
                f"{throughput:.1f} samples/s"
            )
            log_loss = 0.0
            log_data_time = 0.0
            log_step_time = 0.0
            log_samples = 0

        end = time.perf_counter()

    avg_loss = running_loss / len(dataloader)
    metrics = {
        "data_time": data_time_sum / len(dataloader),
        "step_time": step_time_sum / len(dataloader),
        "throughput": samples_sum / step_time_sum if step_time_sum > 0 else 0.0,
    }
    return avg_loss, metrics


def validate_epoch(model, dataloader, device="cuda", amp=False, non_blocking=False):
    model.eval()
    running_loss = 0.0
    criterion = nn.MSELoss()
    data_time_sum = 0.0
    step_time_sum = 0.0
    samples_sum = 0
    if dataloader is None or len(dataloader) == 0:
        return 0.0, {"data_time": 0.0, "step_time": 0.0, "throughput": 0.0}

    end = time.perf_counter()

    with torch.inference_mode():
        for (channels,) in dataloader:
            data_time = time.perf_counter() - end
            data_time_sum += data_time

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
            step_time = time.perf_counter() - start
            step_time_sum += step_time

            samples_sum += channels.size(0)
            running_loss += loss.item()

            end = time.perf_counter()

    avg_loss = running_loss / len(dataloader)
    metrics = {
        "data_time": data_time_sum / len(dataloader),
        "step_time": step_time_sum / len(dataloader),
        "throughput": samples_sum / step_time_sum if step_time_sum > 0 else 0.0,
    }
    return avg_loss, metrics


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if args.tf32 else "highest")

    channels_ri = load_channels_ri(
        args.scenarios,
        dataset_folder=args.dataset_folder,
        cache_path=args.channels_cache,
    )
    if len(channels_ri) == 0:
        print("Error: No data loaded.")
        return

    tensor = torch.from_numpy(channels_ri)
    dataset = torch.utils.data.TensorDataset(tensor)

    train_data, val_data, test_data = split_data(
        dataset, args.train_ratio, args.val_ratio, seed=args.seed
    )

    num_workers = max(0, int(args.num_workers))
    pin_memory = bool(args.pin_memory and device.startswith("cuda"))
    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers

    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, **loader_kwargs
    )
    val_loader = None
    if len(val_data) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_data, shuffle=False, **loader_kwargs
        )
    test_loader = None
    if len(test_data) > 0:
        test_loader = torch.utils.data.DataLoader(
            test_data, shuffle=False, **loader_kwargs
        )

    print(f"Initializing Axial LWM model on {device}...")
    model = LWMWithPrepatchAxial(gen_raw=False, snr_db=args.snr_db).to(device)
    
    if args.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except Exception as exc:
            # torch.compile may fail if Triton can't find a compiler.
            print(f"Warning: torch.compile failed, continuing uncompiled: {exc}")
            
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    amp_enabled = bool(args.amp and device.startswith("cuda"))
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
    non_blocking = pin_memory and device.startswith("cuda")

    log_file = args.log_file
    if log_file is None:
        base_path = args.save_path or "pretraining_axial"
        log_file = os.path.splitext(base_path)[0] + ".log"
    log_file = os.path.expanduser(log_file)
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_fp = open(log_file, "a", encoding="utf-8")
    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            print("Warning: TensorBoard requested but tensorboard is not installed.")
        else:
            tb_logdir = os.path.expanduser(args.tb_logdir)
            os.makedirs(tb_logdir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_logdir)

    def log(*message):
        print(*message)
        log_fp.write(" ".join(str(part) for part in message) + "\n")
        log_fp.flush()

    log(f"Logging to: {log_file}")
    if writer is not None:
        log(f"TensorBoard: enabled ({args.tb_logdir})")
    elif args.tensorboard:
        log("TensorBoard: requested but unavailable (install tensorboard)")
    log(f"Device: {device}")
    log(f"Num workers: {num_workers} | Pin memory: {pin_memory}")
    log(
        "Dataset sizes:",
        f"train={len(train_data)}, val={len(val_data)}, test={len(test_data)}",
    )
    log(f"Train batches per epoch: {len(train_loader)}")
    log(
        "Val/Test batches per epoch:",
        f"val={len(val_loader) if val_loader is not None else 0}, "
        f"test={len(test_loader) if test_loader is not None else 0}",
    )
    if amp_enabled:
        log("AMP: enabled")
    if args.torch_compile:
        log(f"Torch compile: enabled ({args.compile_mode})")

    for epoch in range(args.epochs):
        log(f"Epoch {epoch + 1}/{args.epochs}")
        log("-" * 20)
        log(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        log(f"Batch Size: {args.batch_size}")
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device=device,
            amp=amp_enabled,
            scaler=scaler,
            log_interval=args.log_interval,
            log_fn=log,
            writer=writer,
            epoch_idx=epoch,
            non_blocking=non_blocking,
            scheduler_step_per_batch=args.scheduler_step_per_batch,
            grad_clip=args.grad_clip,
        )
        if scheduler is not None and not args.scheduler_step_per_batch:
            scheduler.step()
        log(
            f"Training Loss: {train_loss:.4f} | "
            f"data {train_metrics['data_time'] * 1000:.1f} ms | "
            f"step {train_metrics['step_time'] * 1000:.1f} ms | "
            f"{train_metrics['throughput']:.1f} samples/s"
        )
        if writer is not None:
            writer.add_scalar("train/loss_epoch", train_loss, epoch + 1)
            writer.add_scalar(
                "train/data_time_ms_epoch", train_metrics["data_time"] * 1000, epoch + 1
            )
            writer.add_scalar(
                "train/step_time_ms_epoch", train_metrics["step_time"] * 1000, epoch + 1
            )
            writer.add_scalar(
                "train/throughput_epoch", train_metrics["throughput"], epoch + 1
            )
        if device.startswith("cuda"):
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            log(f"Peak GPU memory: {peak_mem:.1f} MB")
            if writer is not None:
                writer.add_scalar("train/peak_gpu_mem_mb", peak_mem, epoch + 1)

        if val_loader is not None:
            val_loss, val_metrics = validate_epoch(
                model,
                val_loader,
                device=device,
                amp=amp_enabled,
                non_blocking=non_blocking,
            )
            log(
                f"Validation Loss: {val_loss:.4f} | "
                f"data {val_metrics['data_time'] * 1000:.1f} ms | "
                f"step {val_metrics['step_time'] * 1000:.1f} ms | "
                f"{val_metrics['throughput']:.1f} samples/s"
            )
            if writer is not None:
                writer.add_scalar("val/loss_epoch", val_loss, epoch + 1)
                writer.add_scalar(
                    "val/data_time_ms_epoch", val_metrics["data_time"] * 1000, epoch + 1
                )
                writer.add_scalar(
                    "val/step_time_ms_epoch", val_metrics["step_time"] * 1000, epoch + 1
                )
                writer.add_scalar(
                    "val/throughput_epoch", val_metrics["throughput"], epoch + 1
                )

        if args.save_every and (epoch + 1) % args.save_every == 0:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(save_model.state_dict(), args.save_path)

    if test_loader is not None:
        test_loss, test_metrics = validate_epoch(
            model,
            test_loader,
            device=device,
            amp=amp_enabled,
            non_blocking=non_blocking,
        )
        log(
            f"Test Loss: {test_loss:.4f} | "
            f"data {test_metrics['data_time'] * 1000:.1f} ms | "
            f"step {test_metrics['step_time'] * 1000:.1f} ms | "
            f"{test_metrics['throughput']:.1f} samples/s"
        )
        if writer is not None:
            writer.add_scalar("test/loss", test_loss, 0)
            writer.add_scalar("test/data_time_ms", test_metrics["data_time"] * 1000, 0)
            writer.add_scalar("test/step_time_ms", test_metrics["step_time"] * 1000, 0)
            writer.add_scalar("test/throughput", test_metrics["throughput"], 0)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(save_model.state_dict(), args.save_path)
        log(f"Saved model to {args.save_path}")

    if writer is not None:
        writer.close()
    log_fp.close()


if __name__ == "__main__":
    main()
