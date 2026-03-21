import torch
import sys
import os

# Ensure we can import modules from current directory
sys.path.append(os.getcwd())

from lwm_axial.lwm_axial_model import lwm

def test_model():
    print("Initializing LWM with Axial Linear Attention...")
    model = lwm()
    print("Model initialized.")

    batch_size = 2
    seq_len = 129
    element_length = 16
    
    # Input tensor: (Batch, Seq_Len, Element_Length)
    # LWM takes raw patch features, not token IDs (based on Embedding.proj usage)
    dummy_input = torch.randn(batch_size, seq_len, element_length)
    
    # Masked positions: dummy indices to gather
    # Gathers 5 tokens per sample
    dummy_masked_pos = torch.randint(0, seq_len, (batch_size, 5)).float() 
    # Note: Model expects float? Code says `masked_pos.long()` inside forward, implying it might accept float input but casts it.
    
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        logits, output = model(dummy_input, dummy_masked_pos)
        print("Forward pass successful!")
        print(f"Logits shape: {logits.shape}")
        print(f"Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Forward pass FAILED.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
