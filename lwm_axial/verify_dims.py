
import numpy as np
import sys
import os

# Mock DeepMIMOv3 behavior based on parameters
def generate_mock_data():
    n_ant = 32
    n_sub = 32
    
    # Create distinguishable data
    # Channel shape is usually [n_users, n_ant, n_sub] or similar.
    # We will simulate one user.
    # Let's verify what patch_maker expects.
    # It expects `original_ch`.
    
    # Let's CREATE a dummy original_ch with verifiable patterns.
    # Assume distinct values for each ant/sub pair.
    
    # CASE 1: [Ant, Sub]
    ch_1 = np.zeros((1, n_ant, n_sub), dtype=complex)
    for a in range(n_ant):
        for s in range(n_sub):
            ch_1[0, a, s] = complex(a, s)
            
    # CASE 2: [Sub, Ant]
    ch_2 = np.zeros((1, n_sub, n_ant), dtype=complex)
    for s in range(n_sub):
        for a in range(n_ant):
            ch_2[0, s, a] = complex(s, a) # Swapped
            
    return ch_1, ch_2

def test_flattening(ch, name):
    print(f"\n--- Testing {name} with shape {ch.shape} ---")
    flat = ch.reshape((ch.shape[0], -1))
    # First few elements
    print(f"First 5 elements of flatten: {flat[0, :5]}")
    
    # Check what 'patch_maker' does
    flat_complex = np.hstack((flat.real, flat.imag))
    # flat_complex is [N, 2048]
    
    patch_size = 16
    n_patches = flat_complex.shape[1] // patch_size
    print(f"Num patches: {n_patches}")
    
    # Let's see patch 0
    p0 = flat_complex[:, 0:patch_size]
    print(f"Patch 0 first 5: {p0[0, :5]}")

    # My Axial Attention reshapes [128, D] -> [4, 32, D] (Freq=4, Ant=32)
    # This implies the Outer Dim is Freq (4), Inner Dim is Ant (32).
    # Meaning the data is ordered: Freq1_Ant1, Freq1_Ant2 ... Freq1_Ant32, Freq2_Ant1...
    
    # If flatten is Row-Major (default), and shape is [Ant, Sub]:
    # It iterates Sub (inner) then Ant (outer).
    # Sequence: Ant0_Sub0, Ant0_Sub1, Ant0_Sub2 ... 
    # This keeps Ant constant and varies Sub.
    
    # If shape is [Sub, Ant]:
    # Sequence: Sub0_Ant0, Sub0_Ant1, Sub0_Ant2 ... 
    # This keeps Sub constant and varies Ant.
    
if __name__ == "__main__":
    c1, c2 = generate_mock_data()
    test_flattening(c1, "Assuming [1, Ant, Sub]")
    test_flattening(c2, "Assuming [1, Sub, Ant]")
