import argparse
import csv
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from lwm.input_preprocess import (
    DeepMIMO_data_gen,
    deepmimo_data_cleaning,
    label_gen,
    tokenizer,
)
from lwm.inference import create_raw_dataset
from lwm.lwm_model import lwm
from lwm.utils import FCN, get_data_loaders, train_model
from lwm_ca.tokenizer_ca import tokenizer_ca
from lwm_ca.torch_pipeline import (
    LWMWithPrepatchCA,
    add_complex_noise_ri,
    channels_to_patches,
    ensure_ri_channels,
)


def load_state_dict_flexible(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark base LWM vs CA-augmented LWM on downstream tasks."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "O1_3p5B",
        ],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="beam",
        help="los or beam (or full task name).",
    )
    parser.add_argument("--n-beams", type=int, default=16)
    parser.add_argument(
        "--input-types",
        nargs="+",
        default=["cls_emb", "channel_emb"],
        choices=["raw", "cls_emb", "channel_emb"],
    )
    parser.add_argument(
        "--split-ratios",
        nargs="+",
        type=float,
        default=[0.05,],
    )
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dataset-folder",
        type=str,
        default="/home/audbhav22/foundation_model/Dataset",
        help="Path to DeepMIMO scenarios root (contains scenario folders).",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        default=["base", "ca"],
        choices=["base", "ca"],
    )
    parser.add_argument(
        "--ca-mode",
        type=str,
        default="e2e",
        choices=["prepatch", "e2e"],
    )
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default="./lwm/model_weights.pth",
    )
    parser.add_argument(
        "--ca-ckpt",
        type=str,
        default="./lwm_ca/model_weights_ca_e2e.pth",
    )
    parser.add_argument("--save-csv", type=str, default="results/benchmark_results.csv")
    return parser.parse_args()


def normalize_task(task):
    task_lower = task.strip().lower()
    if task_lower in {"los", "los/nlos", "los_nlos"}:
        return "LoS/NLoS Classification"
    if task_lower in {"beam", "beam_prediction"}:
        return "Beam Prediction"
    return task


def load_deepmimo_data(scenarios, dataset_folder=None):
    if isinstance(scenarios, str):
        scenarios = [scenarios]
    return [DeepMIMO_data_gen(name, dataset_folder=dataset_folder) for name in scenarios]


def stack_cleaned_channels(deepmimo_data):
    cleaned = [deepmimo_data_cleaning(dm) for dm in deepmimo_data]
    return np.vstack(cleaned)


def channels_to_ri(channels):
    real = channels.real.astype(np.float32)
    imag = channels.imag.astype(np.float32)
    return np.stack([real, imag], axis=1)


def load_channels_ri(scenarios, dataset_folder=None, deepmimo_data=None):
    if deepmimo_data is None:
        data = []
        for name in scenarios:
            deepmimo_data_item = DeepMIMO_data_gen(name, dataset_folder=dataset_folder)
            cleaned = deepmimo_data_cleaning(deepmimo_data_item)
            data.append(cleaned)
        channels = np.vstack(data)
    else:
        channels = stack_cleaned_channels(deepmimo_data)
    return channels_to_ri(channels)


def create_labels_from_data(task, scenarios, deepmimo_data, n_beams):
    if isinstance(scenarios, str):
        scenarios = [scenarios]
    if len(scenarios) != len(deepmimo_data):
        raise ValueError("Scenario list length does not match DeepMIMO data list.")

    labels = []
    for scenario_name, data in zip(scenarios, deepmimo_data):
        labels.extend(label_gen(task, data, scenario_name, n_beams=n_beams, visualize=False))
    return torch.tensor(labels).long()


def get_lwm_embeddings(preprocessed_chs, model, device, batch_size):
    input_ids, _, masked_pos = zip(*preprocessed_chs)
    dataset = TensorDataset(
        torch.tensor(input_ids).float(),
        torch.tensor(masked_pos).long(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    outputs = []
    model.eval()
    with torch.no_grad():
        for input_ids_batch, masked_pos_batch in loader:
            input_ids_batch = input_ids_batch.to(device)
            masked_pos_batch = masked_pos_batch.to(device)
            _, output = model(input_ids_batch, masked_pos_batch)
            outputs.append(output.cpu())

    return torch.cat(outputs, dim=0).float()


def build_datasets_from_preprocessed(
    preprocessed_chs,
    input_types,
    model_ckpt,
    device,
    batch_size,
):
    datasets = {}
    need_embeddings = any(t in {"cls_emb", "channel_emb"} for t in input_types)

    embeddings = None
    if need_embeddings:
        model = lwm.from_pretrained(ckpt_name=model_ckpt, device=device)
        embeddings = get_lwm_embeddings(
            preprocessed_chs, model, device, batch_size=batch_size
        ).cpu()

    if "raw" in input_types:
        datasets["raw"] = create_raw_dataset(preprocessed_chs, device).cpu()
    if embeddings is not None:
        if "cls_emb" in input_types:
            datasets["cls_emb"] = embeddings[:, 0]
        if "channel_emb" in input_types:
            datasets["channel_emb"] = embeddings[:, 1:]
    return datasets


def build_datasets_ca_e2e(
    channels_ri,
    input_types,
    model_ckpt,
    snr_db,
    device,
    batch_size=64,
):
    datasets = {}
    need_raw = "raw" in input_types
    need_embeddings = any(t in {"cls_emb", "channel_emb"} for t in input_types)

    tensor = torch.from_numpy(channels_ri)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensor), batch_size=batch_size, shuffle=False
    )

    model = LWMWithPrepatchCA(gen_raw=True, snr_db=snr_db).to(device)
    model = load_state_dict_flexible(model, model_ckpt, device)
    model.eval()

    raw_batches = []
    embedding_batches = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            if need_raw:
                channels = ensure_ri_channels(batch)
                if snr_db is not None:
                    channels = add_complex_noise_ri(channels, snr_db)
                ca_out = model.coordatt(channels)
                patch_batch = channels_to_patches(ca_out, patch_size=model.patch_size)
                raw_batches.append(patch_batch.cpu())
            if need_embeddings:
                _, _, output = model(batch)
                embedding_batches.append(output.cpu())

    if need_raw:
        datasets["raw"] = torch.cat(raw_batches, dim=0)
    if need_embeddings:
        embeddings = torch.cat(embedding_batches, dim=0)
        if "cls_emb" in input_types:
            datasets["cls_emb"] = embeddings[:, 0]
        if "channel_emb" in input_types:
            datasets["channel_emb"] = embeddings[:, 1:]
    return datasets


def train_fcn_for_splits(
    dataset,
    labels,
    input_type,
    split_ratios,
    n_trials,
    num_classes,
    epochs,
    batch_size,
    device,
):
    dataset = dataset.view(dataset.size(0), -1)
    input_dim = dataset.shape[-1]
    f1_scores = np.zeros((n_trials, len(split_ratios)))
    lr = 0.0001 if input_type == "raw" else 0.001

    for split_idx, split_ratio in enumerate(split_ratios):
        for trial in range(n_trials):
            torch.manual_seed(trial)
            train_loader, test_loader = get_data_loaders(
                dataset, labels, batch_size=batch_size, split_ratio=split_ratio
            )
            model = FCN(input_dim=input_dim, num_classes=num_classes)
            _, test_f1_scores = train_model(
                model,
                train_loader,
                test_loader,
                epochs=epochs,
                lr=lr,
                device=device,
                decay_step=30,
                decay_rate=0.5,
            )
            f1_scores[trial, split_idx] = test_f1_scores[0, -1]

    return f1_scores


def write_csv(path, results, split_ratios):
    rows = []
    for model_name, input_map in results.items():
        for input_type, f1_scores in input_map.items():
            mean = f1_scores.mean(axis=0)
            std = f1_scores.std(axis=0)
            for ratio, mean_val, std_val in zip(split_ratios, mean, std):
                rows.append(
                    {
                        "model": model_name,
                        "input_type": input_type,
                        "split_ratio": ratio,
                        "f1_mean": float(mean_val),
                        "f1_std": float(std_val),
                    }
                )

    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "input_type", "split_ratio", "f1_mean", "f1_std"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    task = normalize_task(args.task)
    num_classes = 2 if task == "LoS/NLoS Classification" else args.n_beams
    deepmimo_data = load_deepmimo_data(args.scenarios, dataset_folder=args.dataset_folder)
    cleaned_channels = stack_cleaned_channels(deepmimo_data)
    labels = create_labels_from_data(
        task,
        args.scenarios,
        deepmimo_data,
        n_beams=args.n_beams,
    ).cpu()

    results = {}

    if "base" in args.compare:
        results["base"] = {}
        preprocessed_base = tokenizer(
            selected_scenario_names=None,
            manual_data=cleaned_channels,
            gen_raw=True,
            snr_db=args.snr_db,
            dataset_folder=args.dataset_folder,
        )
        datasets_base = build_datasets_from_preprocessed(
            preprocessed_base,
            args.input_types,
            args.base_ckpt,
            device,
            args.batch_size,
        )
        for input_type in args.input_types:
            dataset = datasets_base[input_type]
            results["base"][input_type] = train_fcn_for_splits(
                dataset,
                labels,
                input_type,
                args.split_ratios,
                args.n_trials,
                num_classes,
                args.epochs,
                args.batch_size,
                device,
            )

    if "ca" in args.compare:
        results["ca"] = {}
        if args.ca_mode == "prepatch":
            preprocessed_ca = tokenizer_ca(
                selected_scenario_names=None,
                manual_data=cleaned_channels,
                gen_raw=True,
                snr_db=args.snr_db,
                device=device,
                dataset_folder=args.dataset_folder,
            )
            datasets_ca = build_datasets_from_preprocessed(
                preprocessed_ca,
                args.input_types,
                args.ca_ckpt,
                device,
                args.batch_size,
            )
        else:
            channels_ri = channels_to_ri(cleaned_channels)
            datasets_ca = build_datasets_ca_e2e(
                channels_ri,
                args.input_types,
                args.ca_ckpt,
                args.snr_db,
                device,
                batch_size=args.batch_size,
            )

        for input_type in args.input_types:
            dataset = datasets_ca[input_type]
            results["ca"][input_type] = train_fcn_for_splits(
                dataset,
                labels,
                input_type,
                args.split_ratios,
                args.n_trials,
                num_classes,
                args.epochs,
                args.batch_size,
                device,
            )

    for model_name, input_map in results.items():
        print(f"\n{model_name} results:")
        for input_type, f1_scores in input_map.items():
            mean = f1_scores.mean(axis=0)
            std = f1_scores.std(axis=0)
            print(f"  {input_type}:")
            for ratio, mean_val, std_val in zip(args.split_ratios, mean, std):
                print(f"    split={ratio:.4f} f1={mean_val:.4f} +/- {std_val:.4f}")

    if args.save_csv:
        write_csv(args.save_csv, results, args.split_ratios)


if __name__ == "__main__":
    main()
