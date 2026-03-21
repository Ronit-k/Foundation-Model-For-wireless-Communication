
import numpy as np
import sys
import os
import torch

# Add root to path
sys.path.append("/home/audbhav22/foundation_model/CA_LWM_Traing")

from lwm.input_preprocess import DeepMIMO_data_gen, deepmimo_data_cleaning

def check_shapes():
    print("Generating one scenario to check shapes...")
    # Generate for one small scenario
    scenario = "city_18_denver" 
    
    # We might need the dataset folder
    dataset_folder = "/home/audbhav22/foundation_model/Dataset"
    
    try:
        deepmimo_data = DeepMIMO_data_gen(scenario, dataset_folder=dataset_folder)
        cleaned = deepmimo_data_cleaning(deepmimo_data)
        
        print(f"\n--- Shape Analysis ---")
        print(f"Cleaned Data Shape: {cleaned.shape}")
        # Expected: [N_Users, ...]
        
        # Check first element structure
        user0 = cleaned[0]
        print(f"User 0 Shape: {user0.shape}")
        
        # Determine flattening order
        flat = user0.reshape(-1)
        print(f"User 0 Flattened Shape: {flat.shape}")
        
        # Check Real/Imag stacking
        # patch_maker does:
        # flat_channels = original_ch.reshape((original_ch.shape[0], -1))
        # np.hstack((flat_channels.real, flat_channels.imag))
        
        print("\n--- Flattening Simulation ---")
        reshaped = cleaned.reshape((cleaned.shape[0], -1))
        print(f"Reshaped (Batch, -1) Shape: {reshaped.shape}")
        
        stacked = np.hstack((reshaped.real, reshaped.imag))
        print(f"Stacked Real/Imag Shape: {stacked.shape}")
        
        n_patches = stacked.shape[1] // 16
        print(f"Calculated Patches: {n_patches}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_shapes()
