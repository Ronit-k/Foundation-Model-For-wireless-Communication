import torch
import torch.nn as nn
import numpy as np

def compute_physics_bias(
    n_antennas=32, 
    n_freq_groups=4, 
    sigma_spatial=2.0, 
    tau_c=4.0, 
    device='cpu'
):
    """
    Computes a Physics Prior Bias Matrix (Phi) for attention scores.
    Shape: [1, 1, SeqLen, SeqLen] where SeqLen = (Ant * Freq) + 1.
    
    Args:
        n_antennas (int): Number of antenna elements.
        n_freq_groups (int): Number of frequency chunks/patches.
        sigma_spatial (float): Decay factor for spatial distance.
        tau_c (float): Decay factor for frequency coherence.
        device: Torch device.
        
    Returns:
        torch.Tensor: Bias matrix of shape [1, 1, SeqLen, SeqLen].
    """
    seq_len = (n_antennas * n_freq_groups) + 1
    
    # 0. Initialize Bias Matrix
    # We use a large negative number for masking, but here we want a bias.
    # Phi should be in range [0, 1] (before scaling by lambda).
    # But wait, we want to ENCOURAGE near attention, so values should be HIGH for near, LOW for far.
    # Exponentials exp(-d) are exactly that.
    phi = torch.zeros((seq_len, seq_len), device=device)
    
    # Grid indices (excluding CLS at 0)
    # ant_idx[i] = antenna index of token i
    # freq_idx[i] = frequency index of token i
    
    # Layout: [CLS, (Ant0, Freq0), (Ant0, Freq1)... (Ant1, Freq0)...]
    # token_idx k > 0 maps to: 
    #   ant = (k-1) // n_freq_groups
    #   freq = (k-1) % n_freq_groups
    
    idxs = torch.arange(n_antennas * n_freq_groups, device=device)
    ant_indices = idxs // n_freq_groups
    freq_indices = idxs % n_freq_groups
    
    # 1. Spatial Correlation (Antenna Distance)
    # Distance matrix: |ant_i - ant_j|
    dist_mat = torch.abs(ant_indices.unsqueeze(0) - ant_indices.unsqueeze(1)).float()
    spatial_bias = torch.exp(- (dist_mat**2) / (2 * sigma_spatial**2))
    
    # 2. Frequency Selectivity (Coherence)
    # Delay/Freq Distance: |freq_i - freq_j|
    freq_diff_mat = torch.abs(freq_indices.unsqueeze(0) - freq_indices.unsqueeze(1)).float()
    freq_bias = torch.exp(- freq_diff_mat / tau_c)
    
    # 3. Combine Priors (Hadamard Product? or Sum?)
    # Physics: Correlation is usually strictly multiplicative (separable) or coupled.
    # Independent fading -> Multiplicative.
    # Combined Bias = Spatial_Decay * Freq_Decay
    grid_bias = spatial_bias * freq_bias
    
    # 4. Fill Matrix (Skip CLS)
    phi[1:, 1:] = grid_bias
    
    # 5. Handle CLS Token
    # CLS should attend to everyone (Global Context).
    # Let's give it a neutral bias of 1.0 (no decay).
    phi[0, :] = 1.0
    phi[:, 0] = 1.0
    
    # Expand to [1, 1, S, S] for broadcasting over Batch and Heads
    return phi.unsqueeze(0).unsqueeze(0)

if __name__ == "__main__":
    # Sanity check
    bias = compute_physics_bias()
    print(f"Bias Shape: {bias.shape}")
    print(f"Max: {bias.max()}, Min: {bias.min()}")
