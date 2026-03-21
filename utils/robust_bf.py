#%% PACKAGES & MODEULS
import numpy as np
import torch
from input_preprocess import DeepMIMO_data_gen, deepmimo_data_cleaning, tokenizer
from inference import lwm_inference, create_raw_dataset
from lwm_model import lwm

#%% DEEPMIMO DATA GENERATION
scenario_names = np.array([
    "city_18_denver", "city_15_indianapolis", "city_19_oklahoma", 
    "city_12_fortworth", "city_11_santaclara", "city_7_sandiego"
])

bf_scenario_idx = 3
scenario_idxs = np.array([bf_scenario_idx])  
selected_scenario_names = scenario_names[scenario_idxs]

deepmimo_data = [DeepMIMO_data_gen(scenario_name) for scenario_name in selected_scenario_names]
cleaned_deepmimo_data = [deepmimo_data_cleaning(deepmimo_data[scenario_idx]) for scenario_idx in range(len(deepmimo_data))]

#%% FUNCTION FOR MRT BEAMFORMING
def compute_mrt_beamforming(channel_data, snr_db=None):

    channel_data = torch.tensor(channel_data[0])
    mrt_vectors = []
    snr_linear = 10 ** (snr_db / 10) if snr_db is not None else None

    for idx in range(channel_data.shape[0]):
        channel = channel_data[idx, 0, :, :]  # Shape: (32, 32)

        if snr_db is not None:
            # Add complex Gaussian noise to the channel
            noise_power = torch.mean(torch.abs(channel) ** 2) / snr_linear
            noise = torch.sqrt(noise_power / 2) * (
                torch.randn_like(channel) + 1j * torch.randn_like(channel)
            )
            channel = channel + noise

        # Compute MRT beamforming vector for each user
        h_avg = torch.mean(channel, dim=1, keepdim=True)  # Shape: (32, 1)
        h_conj = torch.conj(h_avg)  # Conjugate of averaged channel vector
        mrt_vector = h_conj / torch.norm(h_conj, dim=0, keepdim=True)  # Normalize

        mrt_vectors.append(mrt_vector)

    return torch.stack(mrt_vectors, dim=0)  # Shape: (N, 32, 1)

#%% GENERATE BEAMFORMING VECTORS
beamforming_vectors = compute_mrt_beamforming(cleaned_deepmimo_data)

#%% GENERATE LWM EMBEDDINGS FROM MASKED INPUT CHANNELS
preprocessed_chs = tokenizer(
    selected_scenario_names=selected_scenario_names, 
    manual_data=None, 
    gen_raw=False) # gen_raw=False masks 15% of the input patches, and LWM will act as a denoiser

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading the LWM model on {device} ...")
model = lwm.from_pretrained(device=device)

input_types = ['cls_emb', 'channel_emb', 'raw']
selected_input_type = input_types[1] 

if selected_input_type in ['cls_emb', 'channel_emb']:
    dataset = lwm_inference(preprocessed_chs, selected_input_type, model, device)
else:
    dataset = create_raw_dataset(preprocessed_chs, device)
