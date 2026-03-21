import subprocess
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import csv, json, time
from sklearn.metrics import f1_score
from tqdm import tqdm  # Progress bar

# ---------------------------------------------------------
# DATA GENERATION PART: Clone repositories and prepare data
# ---------------------------------------------------------

def clone_dataset_scenario(repo_url, model_repo_dir="./LWM", scenarios_dir="scenarios"):
    """
    Clones all scenarios from a repository, ensuring all files (small and large) are downloaded.
    """
    current_dir = os.path.basename(os.getcwd())
    if current_dir == "LWM":
        model_repo_dir = "."
    scenarios_path = os.path.join(model_repo_dir, scenarios_dir)
    os.makedirs(scenarios_path, exist_ok=True)
    original_dir = os.getcwd()
    try:
        if os.path.exists(scenarios_path):
            shutil.rmtree(scenarios_path)
        print("Cloning entire repository into temporary directory ...")
        subprocess.run(["git", "clone", repo_url, scenarios_path], check=True)
        os.chdir(scenarios_path)
        print("Pulling all files using Git LFS ...")
        subprocess.run(["git", "lfs", "install"], check=True)
        subprocess.run(["git", "lfs", "pull"], check=True)
        print(f"Successfully cloned all scenarios into {scenarios_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning scenarios: {str(e)}")
    finally:
        if os.path.exists(scenarios_path):
            shutil.rmtree(scenarios_path)
        os.chdir(original_dir)

# Clone model repository if not already present.
model_repo_url = "https://huggingface.co/wi-lab/lwm"
model_repo_dir = "./LWM"
if not os.path.exists(model_repo_dir):
    print(f"Cloning model repository from {model_repo_url}...")
    subprocess.run(["git", "clone", model_repo_url, model_repo_dir], check=True)

# Clone dataset repository (scenarios)
dataset_repo_url = "https://huggingface.co/datasets/wi-lab/lwm"
clone_dataset_scenario(dataset_repo_url, model_repo_dir)

# Change working directory to model repository.
if os.path.exists(model_repo_dir):
    os.chdir(model_repo_dir)
    print(f"Changed working directory to {os.getcwd()}")
else:
    print(f"Directory {model_repo_dir} does not exist. Please check if the repository is cloned properly.")

# Import tokenizer and LWM model from the repository.
from input_preprocess import tokenizer
from lwm_model import lwm

# Define scenario names and select one (or more).
scenario_names = np.array([
    "city_18_denver", "city_15_indianapolis", "city_19_oklahoma", 
    "city_12_fortworth", "city_11_santaclara", "city_7_sandiego"
])
# Select the first scenario (index 0) – adjust as needed.
scenario_idxs = np.array([0, 1, 2, 3, 4, 5])
selected_scenario_names = scenario_names[scenario_idxs]

snr_db = None
preprocessed_chs, deepmimo_data = tokenizer(
    selected_scenario_names=selected_scenario_names,
    manual_data=None,
    gen_raw=True,
    snr_db=snr_db
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading the LWM model on {device} ...")
lwm_model = lwm.from_pretrained(device=device)

# Import inference functions and generate the dataset.
from inference import lwm_inference, create_raw_dataset
input_types = ['cls_emb', 'channel_emb', 'raw']
selected_input_type = input_types[2]  # choose one: 'cls_emb', 'channel_emb', or 'raw'
if selected_input_type in ['cls_emb', 'channel_emb']:
    dataset = lwm_inference(preprocessed_chs, selected_input_type, lwm_model, device)
else:
    dataset = create_raw_dataset(preprocessed_chs, device)
# At this point, `dataset` should be a torch Dataset yielding (data, target) pairs.

# ---------------------------------------------------------
# TRAINING PART: Beam Prediction Model (1D CNN) with Weighted F1 and Visualizations
# ---------------------------------------------------------

# Mapping for beam prediction input types.
mapping = {
    'cls_emb': {'input_channels': 1, 'sequence_length': 64},
    'channel_emb': {'input_channels': 64, 'sequence_length': 128},
    'raw': {'input_channels': 16, 'sequence_length': 128}
}
input_type = selected_input_type  # use the same type as for data generation
params = mapping.get(input_type, mapping['raw'])
n_beams = 16  # adjust as needed
initial_lr = 0.001
num_classes = n_beams + 1  # as defined in your code

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

# Instantiate the beam prediction model.
beam_model = res1dcnn(params['input_channels'], params['sequence_length'], num_classes).to(device)
optimizer = Adam(beam_model.parameters(), lr=initial_lr)
scheduler = MultiStepLR(optimizer, milestones=[15, 35], gamma=0.1)
num_epochs = 50

# Create DataLoaders (assuming `dataset` is a torch Dataset with (data, target) pairs).
batch_size = 32  # adjust as needed
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset, batch_size=batch_size)
test_loader = DataLoader(dataset, batch_size=batch_size)

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

# Training loop with weighted F1 computation.
criterion = nn.CrossEntropyLoss()
train_losses = []
val_losses = []
val_f1_scores = []
epochs_list = []

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

# Plot training metrics.
plot_training_metrics(epochs_list, train_losses, val_losses, val_f1_scores, save_path="training_metrics.png")

# Test evaluation with tqdm.
beam_model.eval()
test_running_loss = 0.0
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
test_loss = test_running_loss / len(test_loader.dataset)
accuracy = 100 * correct / total
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")