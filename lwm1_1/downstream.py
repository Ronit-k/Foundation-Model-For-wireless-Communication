# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:11:58 2025

This script evaluates downstream task performance by comparing models trained 
on raw channel representations versus those trained on LWM embeddings.

@author: Sadjad Alikhani
"""
#%% IMPORT PACKAGES & MODULES
from input_preprocess import tokenizer, scenarios_list
from inference import lwm_inference
from utils import prepare_loaders
from train import finetune
import lwm_model
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#%% DOWNSTERAM DATA GENERATION
n_beams = 16
task = ['Beam Prediction', 'LoS/NLoS Classification'][1]
task_type = ["classification", "regression"][0]
visualization_method = ["pca", "umap", "tsne"][2]
input_types = ["cls_emb", "channel_emb", "raw"]
train_ratios = [.001, .01, .05, .1, .25, .5, .8]
fine_tuning_status = [None, ["layers.8", "layers.9", "layers.10", "layers.11"], "full"]
selected_scenario_names = [scenarios_list()[6]] 
preprocessed_data, labels, raw_chs = tokenizer(
    selected_scenario_names, 
    bs_idxs=[3], 
    load_data=False, 
    task=task, 
    n_beams=n_beams,
    manual_data=None)
#%% LOAD THE MODEL
gpu_ids = [0]
device = torch.device("cuda:0")
model = lwm_model.lwm().to(device)

model_name = "model.pth"
state_dict = torch.load(f"models/{model_name}", map_location=device)
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

model = nn.DataParallel(model, gpu_ids)
print(f"Model loaded successfully on GPU {device.index}")
#%% 2D EMBEDDING SPACE VISUALIZATIONN BEFORE FINE-TUNING
chs = lwm_inference(
    model, 
    preprocessed_data, 
    input_type="cls_emb", 
    device=device, 
    batch_size=64, 
    visualization=False, 
    labels=labels, 
    visualization_method=visualization_method)
#%% FINE-TUNE
results = np.zeros((len(fine_tuning_status), len(input_types), len(train_ratios)))
for fine_tuning_stat_idx, fine_tuning_stat in enumerate(fine_tuning_status):
    for input_type_idx, input_type in enumerate(input_types):
        
        if input_type == "raw" and fine_tuning_stat is not None:
            continue
        
        selected_patches_idxs = None
        for train_ratio_idx, train_ratio in enumerate(train_ratios):
            
            print(f"\nfine-tuning status: {fine_tuning_stat}")
            print(f"input type: {input_type}")
            print(f"train ratio: {train_ratio}\n")
            
            # PREPARE LOADERS
            train_loader, val_loader, samples, target = prepare_loaders(
                preprocessed_data=preprocessed_data,
                labels=labels,
                selected_patches_idxs=selected_patches_idxs,
                input_type=input_type,
                task_type=task_type,
                train_ratio=train_ratio,
                batch_size=128,
                seed=42
            )
            
            # FINE-TUNE LWM 
            fine_tuned_model, best_model_path, train_losses, val_losses, f1_scores, attn_maps_ft = finetune(
                base_model=model, 
                train_loader=train_loader,
                val_loader=val_loader,
                task_type=task_type,
                input_type=input_type,
                num_classes=n_beams if task=='Beam Prediction' else 2 if task=='LoS/NLoS Classification' else None,
                output_dim=target.shape[-1] if task_type =='regression' else None,
                use_custom_head=True,
                fine_tune_layers=fine_tuning_stat,
                optimizer_config={"lr": 1e-3},
                epochs=15,
                device=device,
                task=task
            )
            
            results[fine_tuning_stat_idx][input_type_idx][train_ratio_idx] = f1_scores[-1]

markers = ['o', 's', 'D'] 
labels = ['CLS Emb', 'CHS Emb', 'Raw']
fine_tuning_status_labels = ['No FT', 'Partial FT', 'Full FT']
line_styles = ['-', '--', ':'] 
colors = plt.cm.viridis(np.linspace(0, 0.8, len(labels)))
plt.figure(figsize=(12, 8), dpi=500)
for ft_idx, (ft_status_label, line_style) in enumerate(zip(fine_tuning_status_labels, line_styles)):
    for idx, (marker, label, color) in enumerate(zip(markers, labels, colors)):
        # For "Raw Channels," only plot "No Fine-Tuning" case
        if label == "Raw" and ft_status_label != "No FT":
            continue
        # Simplify label for "Raw Channels" without fine-tuning
        plot_label = label if label != "Raw Channels" or ft_status_label != "No Fine-Tuning" else "Raw Channels"
        plt.plot(
            train_ratios, 
            results[ft_idx, idx], 
            marker=marker, 
            linestyle=line_style, 
            label=f"{plot_label} ({ft_status_label})" if label != "Raw Channels" else plot_label, 
            color=color, 
            linewidth=3, 
            markersize=9
        )
plt.xscale('log')
plt.xlabel("Train Ratio", fontsize=20)
plt.ylabel("F1-Score", fontsize=20)
plt.legend(fontsize=17, loc="best")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.tight_layout()
plt.show()
#%% 2D EMBEDDING SPACE VISUALIZATIONN AFTER FINE-TUNING
chs = lwm_inference(
    fine_tuned_model.model, 
    preprocessed_data, 
    input_type="cls_emb", 
    device=device, 
    batch_size=64, 
    visualization=False, 
    labels=labels, 
    visualization_method=visualization_method)