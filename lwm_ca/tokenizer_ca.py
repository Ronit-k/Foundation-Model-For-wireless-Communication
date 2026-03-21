import numpy as np
import torch
from tqdm import tqdm

from lwm.input_preprocess import DeepMIMO_data_gen
from lwm.input_preprocess import deepmimo_data_cleaning
from lwm.input_preprocess import make_sample
from lwm.input_preprocess import patch_maker as base_patch_maker

from .prepatch_ca import apply_coordatt_prepatch, build_coordatt


def tokenizer_ca(
    selected_scenario_names=None,
    manual_data=None,
    gen_raw=True,
    snr_db=None,
    device=None,
    coordatt=None,
    dataset_folder=None,
):
    """
    Tokenizer with pre-patch coordinate attention.

    Args:
        selected_scenario_names (str or list): DeepMIMO scenarios to use.
        manual_data (array-like): Optional custom data of shape (N,H,W).
        gen_raw (bool): If False, enables masking for MCM.
        snr_db (float, optional): Noise level in dB.
        device (str, optional): Torch device for CA.
        coordatt (CoordAtt, optional): Prebuilt CoordAtt module.
        dataset_folder (str, optional): Path to DeepMIMO scenarios root.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if coordatt is None:
        coordatt = build_coordatt(device=device)

    if manual_data is not None:
        channels = np.array(manual_data)
        channels = apply_coordatt_prepatch(channels, coordatt=coordatt, device=device)
        patches = base_patch_maker(channels, snr_db=snr_db)
    else:
        if isinstance(selected_scenario_names, str):
            selected_scenario_names = [selected_scenario_names]
        deepmimo_data = [
            DeepMIMO_data_gen(name, dataset_folder=dataset_folder)
            for name in selected_scenario_names
        ]
        cleaned = [deepmimo_data_cleaning(dm) for dm in deepmimo_data]

        patch_list = []
        for ch in cleaned:
            ca_ch = apply_coordatt_prepatch(ch, coordatt=coordatt, device=device)
            patch_list.append(base_patch_maker(ca_ch, snr_db=snr_db))
        patches = np.vstack(patch_list)

    patch_size = patches.shape[2]
    n_patches = patches.shape[1]
    n_masks_half = int(0.15 * n_patches / 2)

    word2id = {"[CLS]": 0.2 * np.ones((patch_size)), "[MASK]": 0.1 * np.ones((patch_size))}

    preprocessed_data = []
    for user_idx in tqdm(range(len(patches)), desc="Processing items"):
        sample = make_sample(
            user_idx,
            patches,
            word2id,
            n_patches,
            n_masks_half,
            patch_size,
            gen_raw=gen_raw,
        )
        preprocessed_data.append(sample)

    return preprocessed_data
