# -*- coding: utf-8 -*-
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import subprocess
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split, TensorDataset
import csv, json, time
from sklearn.metrics import f1_score
from tqdm import tqdm  # Progress bar
import torch.optim as optim
import math
plt.show = lambda: None


from lwm.input_preprocess import tokenizer, create_labels, DeepMIMO_data_gen
from lwm_ca.torch_pipeline import LWMWithPrepatchCA
from benchmark import  stack_cleaned_channels, channels_to_ri, build_datasets_ca_e2e  # or copy this function

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using",device)
#######SELECT INPUT##############################################
# choose one: 'cls_emb', 'channel_emb', or 'raw'
input_types = ['cls_emb', 'channel_emb', 'raw']
selected_input_type = input_types[0] 
################Select Tasks#####################################
tasks = ['LoS/NLoS Classification', 'Beam Prediction']
task = tasks[1] # Choose 0 for LoS/NLoS labels or 1 for beam prediction labels.
num_epochs = 30
batch_size = 32  # Set a value (adjust as needed)
print(
    "---------------------------- training Details ----------------------------\n"
    f"epochs: {num_epochs}, "
    f"batch size: {batch_size}, "
    f"input type: {selected_input_type}\n"
    f"task: {task}"
)

# Define scenario names and select one (or more).
scenario_names = np.array([
    "city_0_newyork", "city_1_losangeles", "city_2_chicago", "city_3_houston",
    "city_4_phoenix", "city_5_philadelphia", "city_6_miami", "city_7_sandiego",
    "city_8_dallas", "city_9_sanfrancisco", "city_10_austin", "city_11_santaclara",
    "city_12_fortworth", "city_13_columbus", "city_15_indianapolis", "city_17_seattle",
    "city_18_denver", "city_19_oklahoma", "O1_3p5B", "O1_3p5"])
#################################################### Select the first scenario (index 0) – adjust as needed##################################################
scenario_idxs = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])[0:19]
selected_scenario_names = scenario_names[scenario_idxs]
print("selected scenarios: ")
for i in selected_scenario_names: print(i, end=", ")

deepmimo_data = [DeepMIMO_data_gen(scenario_name) for scenario_name in selected_scenario_names]
cleaned_channels = stack_cleaned_channels(deepmimo_data)
print(cleaned_channels.shape)
print("\nusing",selected_input_type,"as input")

#generate targets/lables
labels = create_labels(task, selected_scenario_names, n_beams=64)
print("using",selected_input_type,"for",task,"task")

channels_ri = channels_to_ri(cleaned_channels)
ca_ckpt = "./model_weights_ca_e2e.pth"
snr_db = None
datasets_ca = build_datasets_ca_e2e(
                channels_ri,
                input_types,                            #cls_emb / channel_emb
                ca_ckpt, #path to weights
                snr_db,                            #snr_db
                device,
                batch_size,
            )
# print(datasets_ca)
dataset = datasets_ca[selected_input_type]

# Initial log (Header)
message = (
    "---------------------------- training Details ----------------------------\n"
    f"Dataset Size: {len(dataset)}, shape: {dataset.shape}\n"
    f"epochs: {num_epochs}, "
    f"batch size: {batch_size}, "
    f"input type: {selected_input_type}\n"
    f"task: {task}"
)

# Write header to file
with open("results.txt", "a") as f:
    f.write("\n" + message)
print("\n\ninitiated results.txt with\n", message, '\n'*3)

#function to combine data and labels and split in the given train ratio ratio
def get_data_loaders(data_tensor, labels_tensor, batch_size, train_ratio):
    dataset = TensorDataset(data_tensor, labels_tensor)
    N = len(dataset)

    train_size = int(train_ratio * N)
    remaining = N - train_size
    val_size = remaining // 2
    test_size = remaining - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset,[train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Mapping for beam prediction input types.
mapping = {
    'cls_emb': {'input_channels': 1, 'sequence_length': 64},
    'channel_emb': {'input_channels': 64, 'sequence_length': 128},
    'raw': {'input_channels': 16, 'sequence_length': 128}
}
input_type = selected_input_type  # use the same type as for data generation
params = mapping.get(input_type, mapping[selected_input_type]) #change if chosen anything else
n_beams = 64  # adjust as needed
initial_lr = 0.001
num_classes = n_beams + 1  # as defined in your code
print(selected_input_type)

# ----------------------------------
# 1. SE LAYER
# ----------------------------------
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


# ----------------------------------
# 2. RESIDUAL BLOCK (WITH OPTIONAL SE)
# ----------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, use_se=False):
        super().__init__()

        self.conv1 = nn.Conv1d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)

        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)

        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(out_c) if use_se else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm1d(out_c)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply SE attention (if enabled)
        out = self.se(out)

        out += identity
        out = self.relu(out)

        return out


# ----------------------------------
# 3. MAIN MODEL
# ----------------------------------
class SEResNet1D(nn.Module):
    def __init__(self, input_channels, sequence_length, num_classes):
        super().__init__()

        # No early downsampling
        self.conv1 = nn.Conv1d(
            input_channels, 64,
            kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (No SE)
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

        # Stage 2 (SE starts)
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, use_se=True),
            ResidualBlock(128, 128, use_se=True)
        )

        # Stage 3 (SE)
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, use_se=True),
            ResidualBlock(256, 256, use_se=True)
        )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.1)

        # Infer FC size automatically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, sequence_length)
            dummy = self._forward_conv(dummy)
            flatten_size = dummy.view(1, -1).size(1)

        self.fc = nn.Linear(flatten_size, num_classes)

    def _forward_conv(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return x

    def forward(self, x):
        # Input shape: [B, L, C]
        x = x.transpose(1, 2)  # -> [B, C, L]

        x = self._forward_conv(x)
        x = x.flatten(1)
        x = self.dropout(x)

        return self.fc(x)


print("Final SE-ResNet1D Model Defined.")


# ----------------------------------
# 4. LABEL SMOOTHING LOSS
# ----------------------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# Function to plot training metrics.
def plot_training_metrics(epochs, train_losses, val_losses, val_f1_scores, save_path=None):
    plt.figure(figsize=(12, 5))
    # Loss plot.
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    # F1 score plot.
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_f1_scores, label='Validation Weighted F1', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

matplotlib.use('Agg')

split_ratios = [0.005, 0.01, 0.05, 0.1, 0.2, 0.4]
for split_ratio in split_ratios:

    # Instantiate the beam prediction model.)
    beam_model = SEResNet1D(params['input_channels'], params['sequence_length'], num_classes).to(device)
    optimizer = Adam(beam_model.parameters(), lr=initial_lr, weight_decay=1e-4)
    # scheduler = MultiStepLR(optimizer, milestones=[15, 35], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
        T_0=10,      # Restart every 10 epochs
        T_mult=2,    # Double the restart interval (10, 20, 40...)
        eta_min=1e-6 # Minimum LR
    )
    print("Advanced Optimizer and Scheduler Initialized.")

    # Get DataLoaders for the current split_ratio
    train_loader, val_loader, test_loader = get_data_loaders(dataset, labels, batch_size ,train_ratio=split_ratio)
    criterion = LabelSmoothingLoss(classes=num_classes, smoothing=0.1)
    train_losses = []
    val_losses = []
    val_f1_scores = []
    epochs_list = []
    print("train - test - val",len(train_loader),len(test_loader),len(val_loader))

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(1, num_epochs + 1):
        beam_model.train()
        running_loss = 0.0
        # Training with tqdm progress bar.
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False):
            data, target = data.to(device), target.to(device)
            # Adjust input shape based on type.
            if input_type == 'raw':
                data = data.view(data.size(0), params['sequence_length'], params['input_channels'])
            elif input_type == 'cls_emb':
                data = data.unsqueeze(2)
            optimizer.zero_grad()
            outputs = beam_model(data)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(beam_model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # Validation loop with tqdm.
        beam_model.eval()
        val_running_loss = 0.0
        all_preds = []
        all_targets = []
        for data, target in tqdm(val_loader, desc=f"Epoch {epoch} Validation", leave=False):
            data, target = data.to(device), target.to(device)
            if input_type == 'raw':
                data = data.view(data.size(0), params['sequence_length'], params['input_channels'])
            elif input_type == 'cls_emb':
                data = data.unsqueeze(2)
            outputs = beam_model(data)
            loss = criterion(outputs, target)
            val_running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
        val_loss = val_running_loss / len(val_loader.dataset)
        val_f1 = f1_score(all_targets, all_preds, average='weighted')

        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1_scores.append(val_f1)

        print(f"Epoch {epoch}/{num_epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Weighted F1: {val_f1:.4f}")

    # -----------------------------
    # Test Loop (After Training)
    # -----------------------------
    all_preds = []
    all_targets = []
    correct = 0
    total = 0

    beam_model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if input_type == 'raw':
                data = data.view(data.size(0),
                                 params['sequence_length'],
                                 params['input_channels'])
            elif input_type == 'cls_emb':
                data = data.unsqueeze(2)

            outputs = beam_model(data)
            _, predicted = torch.max(outputs, 1)

            # Accuracy calculation logic
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Compute final metrics
    accuracy = 100 * correct / total
    test_f1 = f1_score(all_targets, all_preds, average='weighted')

    # -----------------------------
    # Save results to file
    # -----------------------------
    with open("results.txt", "a") as f:
        f.write(
            f"\nSplit Ratio: {split_ratio} | "
            f"Test Accuracy: {accuracy:.2f}% | "
            f"Test F1: {test_f1:.4f}\n"
        )

    print("Results saved to results.txt")

    # -----------------------------
    # Save plot instead of showing
    # -----------------------------
    # Create a new figure explicitly
    fig = plt.figure()

    # Pass the figure or just run the plot command.
    # ENSURE plot_training_metrics DOES NOT CALL plt.show() internally.
    plot_training_metrics(epochs_list, train_losses, val_losses, val_f1_scores)

    # Save and close
    plt.savefig(f"{selected_input_type}_{split_ratio}.png", bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved as {selected_input_type}_{split_ratio}.png")
