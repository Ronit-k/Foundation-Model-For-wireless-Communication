# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:13:29 2024

This script generates preprocessed data from wireless communication scenarios, 
including channel generation, patch generation, masking, and preparing raw
channels for the Transformer-based LWM model. 

@author: Sadjad Alikhani
"""
from matplotlib import patches
import numpy as np
import os
from tqdm import tqdm
import time
import pickle
import DeepMIMOv3
import torch
from collections import defaultdict
from utils import generate_gaussian_noise, plot_coverage
#%% Scenarios List
def scenarios_list():
    """Returns an array of available scenarios."""
    return np.array([
        "city_0_newyork", "city_1_losangeles", "city_2_chicago", "city_3_houston", 
        "city_4_phoenix", "city_5_philadelphia", "city_6_miami", "city_7_sandiego", 
        "city_8_dallas", "city_9_sanfrancisco", "city_10_austin", "city_11_santaclara", 
        "city_12_fortworth", "city_13_columbus", "city_14_charlotte_3p5", "city_15_indianapolis", 
        "city_17_seattle", "city_18_denver", "city_19_oklahoma", "O1_3p5B", "O1_3p5"
    ])
    # scen_list = np.array([
    #     'city_0_newyork', 
    #     'city_1_losangeles', 
    #     'city_2_chicago', 
    #     'city_3_houston', 
    #     'city_4_phoenix', 
    #     'city_5_philadelphia', 
    #     'city_6_miami', 
    #     'city_7_sandiego',
    #     'city_8_dallas', 
    #     'city_9_sanfrancisco', 
    #     'city_10_austin', 
    #     'city_11_santaclara', 
    #     'city_12_fortworth', 
    #     'city_13_columbus', 
    #     'city_14_charlotte',
    #     'city_15_indianapolis',
    #     'city_16_sanfrancisco',  
    #     'city_17_seattle', 
    #     'city_18_denver', 
    #     'city_19_oklahoma', 
    #     'asu_campus1_v1',
    #     'asu_campus1_v2',
    #     'asu_campus1_v3',
    #     'asu_campus1_v4',
    #     'asu_campus1_v5',
    #     'asu_campus1_v6',
    #     'asu_campus1_v7',
    #     'asu_campus1_v8',
    #     'asu_campus1_v9',
    #     'asu_campus1_v10',
    #     'asu_campus1_v11',
    #     'asu_campus1_v12',
    #     'asu_campus1_v13',
    #     'asu_campus1_v14',
    #     'asu_campus1_v15',
    #     'asu_campus1_v16',
    #     'asu_campus1_v17',
    #     'asu_campus1_v18',
    #     'asu_campus1_v19',
    #     'asu_campus1_v20',
    #     'Boston5G_3p5_v1',
    #     'Boston5G_3p5_v2',
    #     'Boston5G_3p5_v3',
    #     'Boston5G_3p5_v4',
    #     'Boston5G_3p5_v5',
    #     'Boston5G_3p5_v6',
    #     'Boston5G_3p5_v7',
    #     'Boston5G_3p5_v8',
    #     'Boston5G_3p5_v9',
    #     'Boston5G_3p5_v10',
    #     'Boston5G_3p5_v11',
    #     'Boston5G_3p5_v12',
    #     'Boston5G_3p5_v13',
    #     'Boston5G_3p5_v14',
    #     'Boston5G_3p5_v15',
    #     'Boston5G_3p5_v16',
    #     'Boston5G_3p5_v17',
    #     'Boston5G_3p5_v18',
    #     'Boston5G_3p5_v19',
    #     'Boston5G_3p5_v20',
    #     'O1_3p5_v1',
    #     'O1_3p5_v2',
    #     'O1_3p5_v3',
    #     'O1_3p5_v4',
    #     'O1_3p5_v5',
    #     'O1_3p5_v6',
    #     'O1_3p5_v7',
    #     'O1_3p5_v8',
    #     'O1_3p5_v9',
    #     'O1_3p5_v10',
    #     'O1_3p5_v11',
    #     'O1_3p5_v12',
    #     'O1_3p5_v13',
    #     'O1_3p5_v14',
    #     'O1_3p5_v15',
    #     'O1_3p5_v16',
    #     'O1_3p5_v17',
    #     'O1_3p5_v18',
    #     'O1_3p5_v19',
    #     'O1_3p5_v20',
    #     'asu_campus1',
    #     'O1_3p5',
    #     'Boston5G_3p5',
    #     'city_0_newyork_v16x64', 
    #     'city_1_losangeles_v16x64', 
    #     'city_2_chicago_v16x64', 
    #     'city_3_houston_v16x64', 
    #     'city_4_phoenix_v16x64', 
    #     'city_5_philadelphia_v16x64', 
    #     'city_6_miami_v16x64', 
    #     'city_7_sandiego_v16x64',
    #     'city_8_dallas_v16x64', 
    #     'city_9_sanfrancisco_v16x64'
    #     ])
    # return scen_list
#%% Token Generation
def patch_gen(N_ROWS=4, N_COLUMNS=4, selected_scenario_names=None, 
              manual_data=None, bs_idxs=[1,2,3], load_data=False, 
              save_dir="data", task="LoS/NLoS Classification",
              n_beams=64, o1_bs_idx=[4]):
    
    os.makedirs(save_dir, exist_ok=True)
    
    if manual_data is not None:
        patches = patch_maker(np.expand_dims(np.array(manual_data), axis=1))
    else:
        deepmimo_data = []
        for scenario_name in selected_scenario_names:
            if "O1" in scenario_name: # make an exception for bs idxs of the o1 scenario
                if o1_bs_idx is None:
                    bs_idxs = [4, 15]
                else:
                    bs_idxs = o1_bs_idx
            for bs_idx in bs_idxs:
                if has_version_suffix(scenario_name) and bs_idx in [2,3]:
                    continue
                if not load_data:
                    print(f"\nGenerating data for scenario: {scenario_name}, BS #{bs_idx}")
                    data, n_ant_bs, n_subcarriers = DeepMIMO_data_gen(scenario_name, bs_idx)
                    file_name = f"{save_dir}/{scenario_name}_ant{n_ant_bs}_sub{n_subcarriers}_bs{bs_idx}.npy"
                    np.save(file_name, data)
                    print(f"Data saved to {file_name}")
                    deepmimo_data.append(data) 
                else:
                    n_ant_bs, n_subcarriers = parametersv2(scenario_name, bs_idx)
                    print(f"\nLoading data for scenario: {scenario_name}, BS #{bs_idx}")
                    file_name = f"{save_dir}/{scenario_name}_ant{n_ant_bs}_sub{n_subcarriers}_bs{bs_idx}.npy"
                    data = np.load(file_name, allow_pickle=True).item()
                    print(f"Data loaded from {file_name}")
                    deepmimo_data.append(data)
            
        cleaned_deepmimo_data = [deepmimo_data_cleaning(deepmimo_data[scenario_idx]) for scenario_idx in range(len(deepmimo_data))] #n_scenarios*n_bs_idxs
        patches = [patch_maker(cleaned_deepmimo_data[scenario_idx], N_ROWS, N_COLUMNS) for scenario_idx in range(len(deepmimo_data))]
        raw_chs = torch.tensor(cleaned_deepmimo_data[0]).squeeze(1)
        raw_chs = raw_chs.view(raw_chs.size(0), -1)
        raw_chs = torch.hstack((raw_chs.real, raw_chs.imag))
        
        if task:
            labels = [label_gen(task, deepmimo_data[scenario_idx], selected_scenario_names[scenario_idx], n_beams=n_beams) for scenario_idx in range(len(deepmimo_data))]
            #return patches, torch.tensor(labels[0]), raw_chs.view(raw_chs.size(0), -1)
            all_labels = np.concatenate(labels, axis=0)
            return patches, torch.tensor(all_labels), raw_chs.view(raw_chs.size(0), -1)
        else:
            return patches, raw_chs.view(raw_chs.size(0), -1)
#%%
def tokenizer(selected_scenario_names, 
              bs_idxs=[1,2,3], 
              load_data=False, 
              task="LoS/NLoS Classification", 
              n_beams=64,
              MAX_LEN=513, 
              masking_percent=.40, 
              mask=False, 
              manual_data=None,
              seed=42,
              snr=None):

    patches, labels, raw_chs = patch_gen(
        selected_scenario_names=selected_scenario_names,
        manual_data=manual_data,
        bs_idxs=bs_idxs,
        load_data=load_data,
        task=task,
        n_beams=n_beams
    ) 

    patches = [patch for patch_list in patches for patch in patch_list]
    print("Total number of samples:", len(patches))

    grouped_data = defaultdict(list)  # Group samples by sequence length
    grouped_data_2 = []
    
    for user_idx in tqdm(range(len(patches)), desc="Processing items"):
        patch_size = patches[user_idx].shape[1]
        n_patches = patches[user_idx].shape[0]
        n_masks_half = int(masking_percent * n_patches)

        word2id = {
            '[CLS]': 0.2 * np.ones((patch_size)),
            '[MASK]': 0.1 * np.ones((patch_size))
        }

        sample = make_sample(
            user_idx, patches, word2id, n_patches, n_masks_half, patch_size, MAX_LEN, mask=mask, seed=seed
        )
        
        if mask:
            seq_length = len(sample[0]) 
            grouped_data[seq_length].append(sample)
        else:
            grouped_data_2.append(sample)
    
    if mask:
        # Normalize keys to 0, 1, 2, ...
        normalized_grouped_data = {i: grouped_data[key] for i, key in enumerate(sorted(grouped_data.keys()))}
    else: 
        normalized_grouped_data = torch.stack(grouped_data_2, dim=0)
        # normalized_grouped_data = grouped_data_2
        if snr is not None:
            normalized_grouped_data += generate_gaussian_noise(normalized_grouped_data, snr)
    # normalized_grouped_data = {i: grouped_data[key] for i, key in enumerate(sorted(grouped_data.keys()))}
    
    return normalized_grouped_data, labels, raw_chs
#%% REMOVE ZERO CHANNELS AND SCALE
def deepmimo_data_cleaning(deepmimo_data):
    idxs = np.where(deepmimo_data['user']['LoS'] != -1)[0]
    cleaned_deepmimo_data = deepmimo_data['user']['channel'][idxs]
    return np.array(cleaned_deepmimo_data) * 1e6
#%%
def make_sample(user_idx, patch, word2id, n_patches, n_masks, patch_size, MAX_LEN, mask=True, seed=None):

    if seed is not None:
        np.random.seed(seed)  

    # Step 1: Retrieve tokens and prepend [CLS]
    tokens = patch[user_idx]
    input_ids = np.vstack((word2id['[CLS]'], tokens))

    # Step 2: Mask real and imaginary patches
    tokens_size = int(n_patches)  # int(n_patches / 2)
    masked_pos = np.random.choice(range(1, tokens_size), size=n_masks, replace=False)

    masked_tokens = []
    for pos in masked_pos:
        original_masked_tokens = input_ids[pos].copy()
        masked_tokens.append(original_masked_tokens)
        if mask:
            rnd_num = np.random.rand()
            if rnd_num < 0.1:
                input_ids[pos] = np.random.rand(patch_size)  # Replace with random values
            elif rnd_num < 0.9:
                input_ids[pos] = word2id['[MASK]']  # Replace with [MASK]
    
    if not mask:
        return torch.tensor(input_ids)
    else:
        return [input_ids, masked_tokens, masked_pos]
#%% Patch GENERATION
def patch_maker(original_ch, patch_rows, patch_cols):
    # Step 1: Remove the singleton channel dimension
    n_samples, _, n_rows, n_cols = original_ch.shape  # Unpack shape
    original_ch = original_ch[:, 0]  # Remove the singleton dimension

    # Step 2: Split into real and imaginary parts and interleave them
    flat_real = original_ch.real
    flat_imag = original_ch.imag

    # Interleave real and imaginary parts along the last axis
    interleaved = np.empty((n_samples, n_rows, n_cols * 2), dtype=np.float32)
    interleaved[:, :, 0::2] = flat_real
    interleaved[:, :, 1::2] = flat_imag

    # Step 3: Compute the number of patches along rows and columns
    n_patches_rows = int(np.ceil(n_rows / patch_rows))
    n_patches_cols = int(np.ceil(n_cols / patch_cols))

    # Step 4: Pad the matrix if necessary to make it divisible by patch size
    padded_rows = n_patches_rows * patch_rows - n_rows
    padded_cols = n_patches_cols * patch_cols - n_cols
    if padded_rows > 0 or padded_cols > 0:
        interleaved = np.pad(
            interleaved,
            ((0, 0), (0, padded_rows), (0, padded_cols * 2)),  # Double padding for interleaved axis
            mode='constant',
            constant_values=0,
        )

    # Step 5: Create patches by dividing into blocks
    n_samples, padded_rows, padded_cols = interleaved.shape
    padded_cols //= 2  # Adjust for interleaving (real and imaginary parts count as one)
    patches = []

    for i in range(0, padded_rows, patch_rows):
        for j in range(0, padded_cols, patch_cols):
            patch = interleaved[:, i:i + patch_rows, j * 2:(j + patch_cols) * 2]
            patches.append(patch.reshape(n_samples, -1))  # Flatten each patch

    # Step 6: Stack patches to form the final array
    patches = np.stack(patches, axis=1)  # Shape: (num_samples, n_patches, patch_rows * patch_cols * 2)

    return patches
#%% Data Generation for Scenario Areas
def DeepMIMO_data_gen(scenario, bs_idx):
    import DeepMIMOv3
    parameters, row_column_users = get_parameters(scenario, bs_idx)
    deepMIMO_dataset = DeepMIMOv3.generate_data(parameters)
    
    if "O1" in scenario:
        hops = [2, 2]
    else:
        hops = [1, 1]
        
    uniform_idxs = uniform_sampling(deepMIMO_dataset, hops, len(parameters['user_rows']), 
                                    users_per_row=row_column_users[scenario]['n_per_row'])
    data = select_by_idx(deepMIMO_dataset, uniform_idxs)[0]  
    
    n_ant_bs = parameters['bs_antenna']['shape'][0]
    n_subcarriers = parameters['OFDM']['subcarriers']
    
    return data, n_ant_bs, n_subcarriers
#%%
def parametersv2(scenario, bs_idx):
    parameters, _ = get_parameters(scenario, bs_idx)
    n_ant_bs = parameters['bs_antenna']['shape'][0]
    n_subcarriers = parameters['OFDM']['subcarriers']
    return n_ant_bs, n_subcarriers
#%%%
def get_parameters(scenario, bs_idx=1):
    
    n_ant_ue = 1
    scs = 30e3
        
    row_column_users = scenario_prop()
    
    parameters = DeepMIMOv3.default_params()

    # parameters['dataset_folder'] = './scenarios'
    # Get directory of this file (lwm/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one level up (to LWM/)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    # Point to scenarios folder
    parameters['dataset_folder'] = os.path.join(project_root, "scenarios")

    parameters['scenario'] = scenario.split("_v")[0]
    
    n_ant_bs = row_column_users[scenario]['n_ant_bs']
    n_subcarriers = row_column_users[scenario]['n_subcarriers']
    parameters['active_BS'] = np.array([bs_idx])
    
    if isinstance(row_column_users[scenario]['n_rows'], int):
        parameters['user_rows'] = np.arange(row_column_users[scenario]['n_rows'])
    else:
        parameters['user_rows'] = np.arange(row_column_users[scenario]['n_rows'][0],
                                            row_column_users[scenario]['n_rows'][1])

    parameters['bs_antenna']['shape'] = np.array([n_ant_bs, 1]) # Horizontal, Vertical 
    parameters['bs_antenna']['rotation'] = np.array([0,0,-135]) # (x,y,z)
    parameters['ue_antenna']['shape'] = np.array([n_ant_ue, 1])
    parameters['enable_BS2BS'] = False
    parameters['OFDM']['subcarriers'] = n_subcarriers
    parameters['OFDM']['selected_subcarriers'] = np.arange(n_subcarriers)
    
    parameters['OFDM']['bandwidth'] = scs * n_subcarriers / 1e9
    parameters['num_paths'] = 20
    
    return parameters, row_column_users
#%% Sampling and Data Selection
def uniform_sampling(dataset, sampling_div, n_rows, users_per_row):
    cols = np.arange(users_per_row, step=sampling_div[0])
    rows = np.arange(n_rows, step=sampling_div[1])
    uniform_idxs = np.array([j + i * users_per_row for i in rows for j in cols])
    return uniform_idxs

def select_by_idx(dataset, idxs):
    dataset_t = []  # Trimmed dataset
    for bs_idx in range(len(dataset)):
        dataset_t.append({})
        for key in dataset[bs_idx].keys():
            dataset_t[bs_idx]['location'] = dataset[bs_idx]['location']
            dataset_t[bs_idx]['user'] = {k: dataset[bs_idx]['user'][k][idxs] for k in dataset[bs_idx]['user']}
    return dataset_t
#%%
def inverse_patch_maker(patches, original_shape, patch_rows, patch_cols):
    """
    Reconstructs the original channel matrix from patches.

    Args:
        patches (numpy array): Patches of shape (num_samples, n_patches, patch_rows * patch_cols * 2).
        original_shape (tuple): Original shape of the channel matrix (num_samples, 1, n_rows, n_cols).
        patch_rows (int): Number of rows in each patch.
        patch_cols (int): Number of columns in each patch.

    Returns:
        numpy array: Reconstructed complex-valued channel matrix of shape (num_samples, 1, n_rows, n_cols).
    """
    n_samples, n_patches, patch_size = patches.shape
    _, _, n_rows, n_cols = original_shape

    # Ensure patch dimensions match
    assert patch_rows * patch_cols * 2 == patch_size, "Patch size mismatch with provided dimensions."

    # Compute the number of patches along rows and columns
    n_patches_rows = int(np.ceil(n_rows / patch_rows))
    n_patches_cols = int(np.ceil(n_cols / patch_cols))

    # Reassemble interleaved array from patches
    interleaved = np.zeros((n_samples, n_patches_rows * patch_rows, n_patches_cols * patch_cols * 2), dtype=np.float32)
    patch_idx = 0

    for i in range(n_patches_rows):
        for j in range(n_patches_cols):
            patch = patches[:, patch_idx, :].reshape(n_samples, patch_rows, patch_cols * 2)
            interleaved[:, i * patch_rows:(i + 1) * patch_rows, j * patch_cols * 2:(j + 1) * patch_cols * 2] = patch
            patch_idx += 1

    # Remove padding if necessary
    interleaved = interleaved[:, :n_rows, :n_cols * 2]

    # Separate real and imaginary parts
    flat_real = interleaved[:, :, 0::2]
    flat_imag = interleaved[:, :, 1::2]

    # Reconstruct the complex-valued original channel
    reconstructed = flat_real + 1j * flat_imag

    # Add the singleton channel dimension back
    reconstructed = reconstructed[:, np.newaxis, :, :]  # Shape: (num_samples, 1, n_rows, n_cols)

    return reconstructed
#%%
def label_gen(task, data, scenario, n_beams=64):
    
    idxs = np.where(data['user']['LoS'] != -1)[0]
            
    if task == 'LoS/NLoS Classification':
        label = data['user']['LoS'][idxs]
        
        losChs = np.where(data['user']['LoS'] == -1, np.nan, data['user']['LoS'])
        plot_coverage(data['user']['location'], losChs, cbar_title='LoS status')
        
    elif task == 'Beam Prediction':
        parameters, row_column_users = get_parameters(scenario, bs_idx=1)
        n_users = len(data['user']['channel'])
        n_subbands = 1
        fov = 180

        # Setup Beamformers
        beam_angles = np.around(np.arange(-fov/2, fov/2+.1, fov/(n_beams-1)), 2)

        F1 = np.array([steering_vec(parameters['bs_antenna']['shape'], 
                                    phi=azi*np.pi/180, 
                                    kd=2*np.pi*parameters['bs_antenna']['spacing']).squeeze()
                       for azi in beam_angles])

        full_dbm = np.zeros((n_beams, n_subbands, n_users), dtype=float)
        for ue_idx in tqdm(range(n_users), desc='Computing the channel for each user'):
            if data['user']['LoS'][ue_idx] == -1:
                full_dbm[:,:,ue_idx] = np.nan
            else:
                chs = F1 @ data['user']['channel'][ue_idx]
                full_linear = np.abs(np.mean(chs.squeeze().reshape((n_beams, n_subbands, -1)), axis=-1))
                full_dbm[:,:,ue_idx] = np.around(20*np.log10(full_linear) + 30, 1)

        best_beams = np.argmax(np.mean(full_dbm,axis=1), axis=0)
        best_beams = best_beams.astype(float)
        best_beams[np.isnan(full_dbm[0,0,:])] = np.nan
        # max_bf_pwr = np.max(np.mean(full_dbm,axis=1), axis=0) 
        
        plot_coverage(data['user']['location'], best_beams, tx_pos=data['location'], 
                      tx_ori=parameters['bs_antenna']['rotation']*np.pi/180, 
                      cbar_title='Best beam index')
        
        label = best_beams[idxs]
        
    return label.astype(int)
#%%
def steering_vec(array, phi=0, theta=0, kd=np.pi):
    idxs = DeepMIMOv3.ant_indices(array)
    resp = DeepMIMOv3.array_response(idxs, phi, theta+np.pi/2, kd)
    return resp / np.linalg.norm(resp)
#%%
import re
def has_version_suffix(s):
    pattern = r"_v([1-9]|1[0-9]|20)$"
    return bool(re.search(pattern, s))
#%%
def scenario_prop():
    row_column_users = {
    'city_0_newyork': {
        'n_rows': 44,
        'n_per_row': 117,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_1_losangeles': {
        'n_rows': 57,
        'n_per_row': 81,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_2_chicago': {
        'n_rows': 56,
        'n_per_row': 80,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_3_houston': {
        'n_rows': 62,
        'n_per_row': 81,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_4_phoenix': {
        'n_rows': 79,
        'n_per_row': 86,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_5_philadelphia': {
        'n_rows': 96,
        'n_per_row': 66,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_6_miami': {
        'n_rows': 80,
        'n_per_row': 87,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_7_sandiego': {
        'n_rows': 71,
        'n_per_row': 83,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_8_dallas': {
        'n_rows': 83,
        'n_per_row': 76,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_9_sanfrancisco': {
        'n_rows': 79,
        'n_per_row': 83,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_10_austin': {
        'n_rows': 102,
        'n_per_row': 55,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_11_santaclara': {
        'n_rows': 47,
        'n_per_row': 114,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_12_fortworth': {
        'n_rows': 86,
        'n_per_row': 72,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_13_columbus': {
        'n_rows': 71,
        'n_per_row': 96,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_15_indianapolis': {
        'n_rows': 80,
        'n_per_row': 79,
        'n_ant_bs': 32,
        'n_subcarriers': 32 
    },
    'city_17_seattle': {
        'n_rows': 74,
        'n_per_row': 82,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_18_denver': {
        'n_rows': 85,
        'n_per_row': 82,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'city_19_oklahoma': {
        'n_rows': 82,
        'n_per_row': 75,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'O1_3p5B': {
        'n_rows': 2751,
        'n_per_row': 181,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    },
    'O1_3p5': {
        'n_rows': 2751,
        'n_per_row': 181,
        'n_ant_bs': 32,
        'n_subcarriers': 32
    }}
    # row_column_users = {
    # 'city_0_newyork': {
    #     'n_rows': 109,
    #     'n_per_row': 291,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 32
    # },
    # 'city_1_losangeles': {
    #     'n_rows': 142,
    #     'n_per_row': 201,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 64
    # },
    # 'city_2_chicago': {
    #     'n_rows': 139,
    #     'n_per_row': 200,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 128
    # },
    # 'city_3_houston': {
    #     'n_rows': 154,
    #     'n_per_row': 202,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 256
    # },
    # 'city_4_phoenix': {
    #     'n_rows': 198,
    #     'n_per_row': 214,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 512
    # },
    # 'city_5_philadelphia': {
    #     'n_rows': 239,
    #     'n_per_row': 164,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 1024
    # },
    # 'city_6_miami': {
    #     'n_rows': 199,
    #     'n_per_row': 216 ,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 32
    # },
    # 'city_7_sandiego': {
    #     'n_rows': 207,
    #     'n_per_row': 176,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 64
    # },
    # 'city_8_dallas': {
    #     'n_rows': 207,
    #     'n_per_row': 190,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 128
    # },
    # 'city_9_sanfrancisco': {
    #     'n_rows': 196,
    #     'n_per_row': 206,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 256
    # },
    # 'city_10_austin': {
    #     'n_rows': 255,
    #     'n_per_row': 137,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 512
    # },
    # 'city_11_santaclara': {
    #     'n_rows': 117,
    #     'n_per_row': 285,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 32
    # },
    # 'city_12_fortworth': {
    #     'n_rows': 214,
    #     'n_per_row': 179,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 64
    # },
    # 'city_13_columbus': {
    #     'n_rows': 178,
    #     'n_per_row': 240,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 128
    # },
    # 'city_14_charlotte': {
    #     'n_rows': 216,
    #     'n_per_row': 177,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 256
    # },
    # 'city_15_indianapolis': {
    #     'n_rows': 200,
    #     'n_per_row': 196,
    #     'n_ant_bs': 64,
    #     'n_subcarriers': 32
    # },
    # 'city_16_sanfrancisco': {
    #     'n_rows': 201,
    #     'n_per_row': 208,
    #     'n_ant_bs': 64,
    #     'n_subcarriers': 64
    # },
    # 'city_17_seattle': {
    #     'n_rows': 185,
    #     'n_per_row': 205,
    #     'n_ant_bs': 64,
    #     'n_subcarriers': 128
    # },
    # 'city_18_denver': {
    #     'n_rows': 212,
    #     'n_per_row': 204,
    #     'n_ant_bs': 128,
    #     'n_subcarriers': 32
    # },
    # 'city_19_oklahoma': {
    #     'n_rows': 204,
    #     'n_per_row': 188,
    #     'n_ant_bs': 128,
    #     'n_subcarriers': 64
    # },
    # 'asu_campus1_v1': {
    #     'n_rows': [0, 1*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 32
    # },
    # 'asu_campus1_v2': {
    #     'n_rows': [1*int(321/20), 2*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 64
    # },
    # 'asu_campus1_v3': {
    #     'n_rows': [2*int(321/20), 3*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 128
    # },
    # 'asu_campus1_v4': {
    #     'n_rows': [3*int(321/20), 4*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 256
    # },
    # 'asu_campus1_v5': {
    #     'n_rows': [4*int(321/20), 5*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 512
    # },
    # 'asu_campus1_v6': {
    #     'n_rows': [5*int(321/20), 6*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 1024
    # },
    # 'asu_campus1_v7': {
    #     'n_rows': [6*int(321/20), 7*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 32
    # },
    # 'asu_campus1_v8': {
    #     'n_rows': [7*int(321/20), 8*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs':16,
    #     'n_subcarriers': 64
    # },
    # 'asu_campus1_v9': {
    #     'n_rows': [8*int(321/20), 9*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 128
    # },
    # 'asu_campus1_v10': {
    #     'n_rows': [9*int(321/20), 10*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 256
    # },
    # 'asu_campus1_v11': {
    #     'n_rows': [10*int(321/20), 11*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 512
    # },
    # 'asu_campus1_v12': {
    #     'n_rows': [11*int(321/20), 12*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 32
    # },
    # 'asu_campus1_v13': {
    #     'n_rows': [12*int(321/20), 13*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 64
    # },
    # 'asu_campus1_v14': {
    #     'n_rows': [13*int(321/20), 14*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 128
    # },
    # 'asu_campus1_v15': {
    #     'n_rows': [14*int(321/20), 15*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 256
    # },
    # 'asu_campus1_v16': {
    #     'n_rows': [15*int(321/20), 16*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 64,
    #     'n_subcarriers': 32
    # },
    # 'asu_campus1_v17': {
    #     'n_rows': [16*int(321/20), 17*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 64,
    #     'n_subcarriers': 64 
    # },
    # 'asu_campus1_v18': {
    #     'n_rows': [17*int(321/20), 18*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 64,
    #     'n_subcarriers': 128
    # },
    # 'asu_campus1_v19': {
    #     'n_rows': [18*int(321/20), 19*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 128,
    #     'n_subcarriers': 32
    # },
    # 'asu_campus1_v20': {
    #     'n_rows': [19*int(321/20), 20*int(321/20)],
    #     'n_per_row': 411,
    #     'n_ant_bs': 128,
    #     'n_subcarriers': 64
    # },
    # 'Boston5G_3p5_v1': {
    #     'n_rows': [812, 812 + 1*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 32
    # },
    # 'Boston5G_3p5_v2': {
    #     'n_rows': [812 + 1*int((1622-812)/20), 812 + 2*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 64
    # },
    # 'Boston5G_3p5_v3': {
    #     'n_rows': [812 + 2*int((1622-812)/20), 812 + 3*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 128
    # },
    # 'Boston5G_3p5_v4': {
    #     'n_rows': [812 + 3*int((1622-812)/20), 812 + 4*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 256
    # },
    # 'Boston5G_3p5_v5': {
    #     'n_rows': [812 + 4*int((1622-812)/20), 812 + 5*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 512
    # },
    # 'Boston5G_3p5_v6': {
    #     'n_rows': [812 + 5*int((1622-812)/20), 812 + 6*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 1024
    # },
    # 'Boston5G_3p5_v7': {
    #     'n_rows': [812 + 6*int((1622-812)/20), 812 + 7*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 32
    # },
    # 'Boston5G_3p5_v8': {
    #     'n_rows': [812 + 7*int((1622-812)/20), 812 + 8*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs':16,
    #     'n_subcarriers': 64
    # },
    # 'Boston5G_3p5_v9': {
    #     'n_rows': [812 + 8*int((1622-812)/20), 812 + 9*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 128
    # },
    # 'Boston5G_3p5_v10': {
    #     'n_rows': [812 + 9*int((1622-812)/20), 812 + 10*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 256
    # },
    # 'Boston5G_3p5_v11': {
    #     'n_rows': [812 + 10*int((1622-812)/20), 812 + 11*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 512
    # },
    # 'Boston5G_3p5_v12': {
    #     'n_rows': [812 + 11*int((1622-812)/20), 812 + 12*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 32
    # },
    # 'Boston5G_3p5_v13': {
    #     'n_rows': [812 + 12*int((1622-812)/20), 812 + 13*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 64
    # },
    # 'Boston5G_3p5_v14': {
    #     'n_rows': [812 + 13*int((1622-812)/20), 812 + 14*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 128
    # },
    # 'Boston5G_3p5_v15': {
    #     'n_rows': [812 + 14*int((1622-812)/20), 812 + 15*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 256
    # },
    # 'Boston5G_3p5_v16': {
    #     'n_rows': [812 + 15*int((1622-812)/20), 812 + 16*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 64,
    #     'n_subcarriers': 32
    # },
    # 'Boston5G_3p5_v17': {
    #     'n_rows': [812 + 16*int((1622-812)/20), 812 + 17*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 64,
    #     'n_subcarriers': 64 
    # },
    # 'Boston5G_3p5_v18': {
    #     'n_rows': [812 + 17*int((1622-812)/20), 812 + 18*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 64,
    #     'n_subcarriers': 128
    # },
    # 'Boston5G_3p5_v19': {
    #     'n_rows': [812 + 18*int((1622-812)/20), 812 + 19*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 128,
    #     'n_subcarriers': 32
    # },
    # 'Boston5G_3p5_v20': {
    #     'n_rows': [812 + 19*int((1622-812)/20), 812 + 20*int((1622-812)/20)],
    #     'n_per_row': 595,
    #     'n_ant_bs': 128,
    #     'n_subcarriers': 64
    # },
    # 'O1_3p5_v1': {
    #     'n_rows': [0*int(3852/12), 1*int(3852/12)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 32
    # },
    # 'O1_3p5_v2': {
    #     'n_rows': [1*int(3852/12), 2*int(3852/12)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 64
    # },
    # 'O1_3p5_v3': {
    #     'n_rows': [2*int(3852/12), 3*int(3852/12)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 128
    # },
    # 'O1_3p5_v4': {
    #     'n_rows': [3*int(3852/12), 4*int(3852/12)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 256
    # },
    # 'O1_3p5_v5': {
    #     'n_rows': [4*int(3852/12), 5*int(3852/12)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 512
    # },
    # 'O1_3p5_v6': {
    #     'n_rows': [5*int(3852/12), 6*int(3852/12)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 8,
    #     'n_subcarriers': 1024
    # },
    # 'O1_3p5_v7': {
    #     'n_rows': [6*int(3852/12), 7*int(3852/12)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 32
    # },
    # 'O1_3p5_v8': {
    #     'n_rows': [7*int(3852/12), 8*int(3852/12)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 64
    # },
    # 'O1_3p5_v9': {
    #     'n_rows': [8*int(3852/12), 9*int(3852/12)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 128
    # },
    # 'O1_3p5_v10': {
    #     'n_rows': [9*int(3852/12), 10*int(3852/12)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 256
    # },
    # 'O1_3p5_v11': {
    #     'n_rows': [10*int(3852/12), 11*int(3852/12)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 512
    # },
    # 'O1_3p5_v12': {
    #     'n_rows': [11*int(3852/12), 12*int(3852/12)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 32
    # },
    # 'O1_3p5_v13': {
    #     'n_rows': [12*int(3852/12)+0*int(1351/10), 12*int(3852/12)+1*int(1351/10)],
    #     'n_per_row': 361,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 64
    # },
    # 'O1_3p5_v14': {
    #     'n_rows': [12*int(3852/12)+1*int(1351/10), 12*int(3852/12)+2*int(1351/10)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 128
    # },
    # 'O1_3p5_v15': {
    #     'n_rows': [12*int(3852/12)+2*int(1351/10), 12*int(3852/12)+3*int(1351/10)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 32,
    #     'n_subcarriers': 256
    # },
    # 'O1_3p5_v16': {
    #     'n_rows': [12*int(3852/12)+3*int(1351/10), 12*int(3852/12)+4*int(1351/10)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 64,
    #     'n_subcarriers': 32
    # },
    # 'O1_3p5_v17': {
    #     'n_rows': [12*int(3852/12)+4*int(1351/10), 12*int(3852/12)+5*int(1351/10)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 64,
    #     'n_subcarriers': 64
    # },
    # 'O1_3p5_v18': {
    #     'n_rows': [12*int(3852/12)+5*int(1351/10), 12*int(3852/12)+6*int(1351/10)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 64,
    #     'n_subcarriers': 128
    # },
    # 'O1_3p5_v19': {
    #     'n_rows': [12*int(3852/12)+6*int(1351/10), 12*int(3852/12)+7*int(1351/10)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 128,
    #     'n_subcarriers': 32
    # },
    # 'O1_3p5_v20': {
    #     'n_rows': [12*int(3852/12)+7*int(1351/10), 12*int(3852/12)+8*int(1351/10)],
    #     'n_per_row': 181,
    #     'n_ant_bs': 128,
    #     'n_subcarriers': 64
    # },
    # 'city_0_newyork_v16x64': {
    #     'n_rows': 109,
    #     'n_per_row': 291,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 64
    # },
    # 'city_1_losangeles_v16x64': {
    #     'n_rows': 142,
    #     'n_per_row': 201,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 64
    # },
    # 'city_2_chicago_v16x64': {
    #     'n_rows': 139,
    #     'n_per_row': 200,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 64
    # },
    # 'city_3_houston_v16x64': {
    #     'n_rows': 154,
    #     'n_per_row': 202,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 64
    # },
    # 'city_4_phoenix_v16x64': {
    #     'n_rows': 198,
    #     'n_per_row': 214,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 64
    # },
    # 'city_5_philadelphia_v16x64': {
    #     'n_rows': 239,
    #     'n_per_row': 164,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 64
    # },
    # 'city_6_miami_v16x64': {
    #     'n_rows': 199,
    #     'n_per_row': 216,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 64
    # },
    # 'city_7_sandiego_v16x64': {
    #     'n_rows': 207,
    #     'n_per_row': 176,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 64
    # },
    # 'city_8_dallas_v16x64': {
    #     'n_rows': 207,
    #     'n_per_row': 190,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 64
    # },
    # 'city_9_sanfrancisco_v16x64': {
    #     'n_rows': 196,
    #     'n_per_row': 206,
    #     'n_ant_bs': 16,
    #     'n_subcarriers': 64
    # }}
    return row_column_users