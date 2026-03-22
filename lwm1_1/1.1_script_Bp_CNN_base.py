# %%
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
matplotlib.use('Agg')


from lwm1_1.input_preprocess import tokenizer
from lwm1_1.inference import lwm_inference
from lwm1_1.utils import prepare_loaders
# from lwm1_1.train import finetune
from lwm1_1.lwm_model import lwm


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

# %%
# Define scenario names and select one (or more).
scenario_names = np.array([
    "city_0_newyork", "city_1_losangeles", "city_2_chicago", "city_3_houston",
    "city_4_phoenix", "city_5_philadelphia", "city_6_miami", "city_7_sandiego",
    "city_8_dallas", "city_9_sanfrancisco", "city_10_austin", "city_11_santaclara",
    "city_12_fortworth", "city_13_columbus", "city_15_indianapolis", "city_17_seattle",
    "city_18_denver", "city_19_oklahoma", "O1_3p5B"])
#################################################### Select the first scenario (index 0) – adjust as needed##################################################
scenario_idxs = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])[18:19]
selected_scenario_names = scenario_names[scenario_idxs]
print("selected scenarios: ")
for i in selected_scenario_names: print(i, end=", ")

# %%
n_beams = 64  # Set the number of beams for beam prediction task (adjust as needed)
task_type = ["classification", "regression"][0]
preprocessed_data, labels, raw_chs = tokenizer(
    selected_scenario_names,
    bs_idxs=[1],
    load_data=False,
    task=task,
    n_beams=n_beams,
    manual_data=None)
print(preprocessed_data.shape)
print(labels.shape)
# print(preprocessed_data.shape[1])

# %%
#%% LOAD THE MODEL
gpu_ids = [0]
device = torch.device("cuda:0")
model = lwm().to(device)

model_name = "model.pth"
state_dict = torch.load(f"{model_name}", map_location=device) #
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()} # Remove 'module.' prefix if it exists
model.load_state_dict(new_state_dict) # Load the model weights

# lwm_model = nn.DataParallel(model, gpu_ids) # Wrap the model with DataParallel for multi-GPU support
lwm_model = model  # remove DataParallel to save memory
print(f"Model loaded successfully on GPU {device.index}") 

# %%
with torch.no_grad():
    dataset = lwm_inference(lwm_model, preprocessed_data, selected_input_type, device)

torch.cuda.empty_cache()
# At this point, `dataset` should be a torch Dataset yielding (data, target) pairs.

# %%
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
print(selected_input_type, "dataset shape:", dataset.shape)

# %%
unique_labels, counts = torch.unique(labels, return_counts=True)
x_values = unique_labels.cpu().numpy()
y_values = counts.cpu().numpy()

# 2. Increase figure width (e.g., to 16 or 18) to create more horizontal space
plt.figure(figsize=(18, 7)) 

# 3. Create bars
bars = plt.bar(x_values, y_values, color='skyblue', edgecolor='black', alpha=0.8)

# Adding some padding with 'labelpad' if needed
plt.xticks(x_values, rotation=0, fontsize=9) 

# 5. Annotate each bar with its frequency count
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), 
             va='bottom', ha='center', fontsize=7, rotation=0)

# 6. Add labels and title
plt.xlabel('Beam Index (Labels)', labelpad=15)
plt.ylabel('Frequency')
plt.title(f'Frequency Distribution of Beam Labels ($N = {len(labels)}$)')

# 7. Use tight_layout to ensure no labels are cut off at the edges
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

# %%
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


# %%
###############CHANGE MAPPING ACCORDINGLY#######################333

# Mapping for beam prediction input types.
mapping = {
    'cls_emb': {'input_channels': 1, 'sequence_length': 128},   # CLS embedding dim = 128
    'channel_emb': {'input_channels': 128, 'sequence_length': 64},  
    'raw': {'input_channels': 32, 'sequence_length': 512}  # depends on antenna config
}
input_type = selected_input_type  # use the same type as for data generation
params = mapping.get(input_type, mapping[selected_input_type]) #change if chosen anything else
initial_lr = 0.001
num_classes = n_beams + 1  # as defined in your code
print(selected_input_type)

# %%
# Define Residual Block and the 1D CNN model.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        x = F.relu(x)
        return x

class res1dcnn(nn.Module):
    def __init__(self, input_channels, sequence_length, num_classes):
        super(res1dcnn, self).__init__()
        # Initial convolution and pooling layers.
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # Residual layers.
        self.layer1 = self._make_layer(32, 32, 2)
        self.layer2 = self._make_layer(32, 64, 3)
        self.layer3 = self._make_layer(64, 128, 4)
        # Compute flattened feature size.
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, sequence_length)
            dummy_output = self.compute_conv_output(dummy_input)
            self.flatten_size = dummy_output.numel()
        # Fully connected layers.
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = [ResidualBlock(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def compute_conv_output(self, x):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool1d(x, 1)
        return x
    
    def forward(self, x):
        # Expect x shape: [batch, sequence_length, input_channels]
        x = x.transpose(1, 2)  # -> [batch, input_channels, sequence_length]
        x = self.compute_conv_output(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# %%
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

# %%
# Define the split ratios to iterate over
split_ratios = [0.005, 0.01, 0.05, 0.1, 0.2, 0.4]

for split_ratio in split_ratios:
    print(f"\n--- Starting training for split ratio: {split_ratio} ---")

    # Instantiate the model FRESH for every split ratio (train from scratch)
    beam_model = res1dcnn(params['input_channels'], params['sequence_length'], num_classes).to(device)
    optimizer = Adam(beam_model.parameters(), lr=initial_lr)
    scheduler = MultiStepLR(optimizer, milestones=[15, 35], gamma=0.1)
    
    # Get DataLoaders for the current split_ratio
    train_loader, val_loader, test_loader = get_data_loaders(dataset, labels, batch_size=batch_size, train_ratio=split_ratio)
    
    print(f"train: {len(train_loader)} | validate: {len(val_loader)} | test: {len(test_loader)}")
    
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    val_losses = []
    val_f1_scores = []
    epochs_list = []

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
    beam_model.eval()
    test_running_loss = 0.0
    all_preds_test = []
    all_targets_test = []
    correct = 0
    total = 0
    
    for data, target in tqdm(test_loader, desc="Testing"):
        data, target = data.to(device), target.to(device)
        if input_type == 'raw':
            data = data.view(data.size(0), params['sequence_length'], params['input_channels'])
        elif input_type == 'cls_emb':
            data = data.unsqueeze(2)
        outputs = beam_model(data)
        loss = criterion(outputs, target)
        test_running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(outputs, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        all_preds_test.extend(predicted.cpu().numpy())
        all_targets_test.extend(target.cpu().numpy())
        
    test_loss = test_running_loss / len(test_loader.dataset)
    accuracy = 100 * correct / total
    test_f1 = f1_score(all_targets_test, all_preds_test, average='weighted')
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Test F1: {test_f1:.4f}")

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
    # Save plot
    # -----------------------------
    fig = plt.figure()
    plot_training_metrics(epochs_list, train_losses, val_losses, val_f1_scores)
    plt.savefig(f"{selected_input_type}_{split_ratio}.png", bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved as {selected_input_type}_{split_ratio}.png")


