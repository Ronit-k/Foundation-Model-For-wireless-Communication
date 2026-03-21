import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
#%%
def plot_dimensionality_reduction(tensor, method='all', labels=None, input_type='Unknown', task='Unknown'):
    """
    Plots 2D projections of high-dimensional data using PCA, t-SNE, or UMAP.
    
    Parameters:
    tensor (torch.Tensor): Input data of shape (n_samples, n_features).
    method (str or list): One of ['pca', 'tsne', 'umap'] or 'all' for all three.
    labels (array-like): Optional labels for coloring the scatter plot.
    input_type (str): Type of input data for title.
    task (str): Task description for title.
    """
    tensor = tensor.view(tensor.size(0), -1)
    # Convert to numpy if it's a PyTorch tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    
    methods = []
    if method == 'all':
        methods = ['pca', 'tsne', 'umap']
    elif isinstance(method, str):
        methods = [method]
    elif isinstance(method, list):
        methods = method
    
    plt.figure(figsize=(6 * len(methods), 5))
    plt.suptitle(f"Input: {input_type}, Task: {task}", fontsize=16)
    
    for i, m in enumerate(methods):
        if m == 'pca':
            reducer = PCA(n_components=2)
            title = 'PCA'
        elif m == 'tsne':
            reducer = TSNE(n_components=2, perplexity=2, random_state=42)
            title = 't-SNE'
        elif m == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            title = 'UMAP'
        else:
            raise ValueError(f"Unknown method: {m}")
        
        reduced_data = reducer.fit_transform(tensor)
        
        plt.subplot(1, len(methods), i + 1)
        
        if labels is not None:
            unique_labels = np.unique(labels)
            cmap = plt.get_cmap('Spectral', len(unique_labels)) 
            
            scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap=cmap, alpha=0.75)
            
            cbar = plt.colorbar(scatter, ticks=unique_labels)
            cbar.set_ticklabels(unique_labels)
        else:
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.75)
        
        plt.title(title, fontsize=14)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
#%%
def plot_coverage(receivers, coverage_map, dpi=200, figsize=(6, 4), cbar_title=None, title=None,
                  scatter_size=12, transmitter_position=None, transmitter_orientation=None, 
                  legend=False, limits=None, proj_3d=False, equal_aspect=False, tight_layout=True, 
                  colormap='tab20'):
    # Set up plot parameters
    plot_params = {'cmap': colormap}
    if limits:
        plot_params['vmin'], plot_params['vmax'] = limits[0], limits[1]

    # Extract coordinates
    x, y = receivers[:, 0], receivers[:, 1]

    # Create figure and axis
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize,
                           subplot_kw={})
    
    # Plot the coverage map
    ax.scatter(x, y, c=coverage_map, s=scatter_size, marker='s', edgecolors='black', linewidth=.15, **plot_params)
    
    # Set axis labels
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # Add legend if requested
    if legend:
        ax.legend(loc='upper center', ncols=10, framealpha=0.5)

    # Adjust plot limits
    if tight_layout:
        padding = 1
        mins = np.min(receivers, axis=0) - padding
        maxs = np.max(receivers, axis=0) + padding

        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[1], maxs[1]])

    # Set equal aspect ratio for 2D plots
    if equal_aspect:
        ax.set_aspect('equal')

    # Show plot
    plt.show()
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Data Preparation
def get_data_loaders(data_tensor, labels_tensor, batch_size=32, split_ratio=0.8):
    dataset = TensorDataset(data_tensor, labels_tensor)
    
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class FCN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)


# Training Function
def train_model(model, train_loader, test_loader, epochs=20, lr=0.001, device="cpu", decay_step=10, decay_rate=0.5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)
    criterion = nn.CrossEntropyLoss()
    
    train_losses, test_f1_scores = [], []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        scheduler.step()
        
        # Evaluate on test set
        f1 = evaluate_model(model, test_loader, device)
        test_f1_scores.append(f1)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]:.4f}, F1-score: {f1:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return train_losses, np.array([test_f1_scores])

# Model Evaluation
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(batch_y.cpu().numpy())

    return f1_score(all_labels, all_preds, average='weighted')

# Visualization
import matplotlib.cm as cm

def plot_metrics(test_f1_scores, input_types, n_train=None, flag=0):
    """
    Plots the F1-score over epochs or number of training samples.

    Parameters:
    test_f1_scores (list): List of F1-score values per epoch or training samples.
    input_types (list): List of input type names.
    n_train (list, optional): Number of training samples (used when flag=1).
    flag (int): 0 for plotting F1-score over epochs, 1 for F1-score over training samples.
    """
    plt.figure(figsize=(7, 5), dpi=200)
    colors = cm.get_cmap('Spectral', test_f1_scores.shape[0])  # Using Spectral colormap
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h']  # Different markers for curves
    
    for r in range(test_f1_scores.shape[0]):
        color = colors(0.5 if test_f1_scores.shape[0] == 1 else r / (test_f1_scores.shape[0] - 1))  # Normalize color index
        marker = markers[r % len(markers)]  # Cycle through markers
        if flag == 0:
            if test_f1_scores.shape[0] == 1:
                plt.plot(test_f1_scores[r], linewidth=2, color=color, label=f"{input_types[r]}")
            else:
                plt.plot(test_f1_scores[r], linewidth=2, marker=marker, markersize=5, markeredgewidth=1.5, 
                         markeredgecolor=color, color=color, label=f"{input_types[r]}")
        else:
            plt.plot(n_train, test_f1_scores[r], linewidth=2, marker=marker, markersize=6, markeredgewidth=1.5, 
                     markeredgecolor=color, markerfacecolor='none', color=color, label=f"{input_types[r]}")
            plt.xscale('log') 
    
    x_label = "Epochs" if flag == 0 else "Number of training samples"
    plt.xlabel(f"{x_label}", fontsize=12)
    plt.ylabel("F1-score", fontsize=12)
    
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

#%%
def classify_by_euclidean_distance(train_loader, test_loader, device="cpu"):
    """
    Classifies test samples based on the Euclidean distance to the mean of training samples from each class.
    Computes the F1-score for evaluation.

    Parameters:
    - train_loader (DataLoader): DataLoader for training data.
    - test_loader (DataLoader): DataLoader for test data.
    - device (str): Device to run computations on ("cpu" or "cuda").

    Returns:
    - predictions (torch.Tensor): Predicted class for each test sample.
    - f1 (float): Weighted F1-score.
    """

    # Store all training data and labels
    train_data_list, train_labels_list = [], []
    for batch_x, batch_y in train_loader:
        train_data_list.append(batch_x.to(device))
        train_labels_list.append(batch_y.to(device))
    
    train_data = torch.cat(train_data_list)
    train_labels = torch.cat(train_labels_list)

    unique_classes = torch.unique(train_labels)
    class_means = {}

    # Compute mean feature vector for each class
    for cls in unique_classes:
        class_means[cls.item()] = train_data[train_labels == cls].mean(dim=0)

    # Convert class means to tensor for vectorized computation
    class_means_tensor = torch.stack([class_means[cls.item()] for cls in unique_classes])

    # Store all test data and labels
    test_data_list, test_labels_list = [], []
    for batch_x, batch_y in test_loader:
        test_data_list.append(batch_x.to(device))
        test_labels_list.append(batch_y.to(device))
    
    test_data = torch.cat(test_data_list)
    test_labels = torch.cat(test_labels_list)

    # Compute Euclidean distance between each test sample and all class means
    dists = torch.cdist(test_data, class_means_tensor)  # Shape (n_test, n_classes)

    # Assign the class with the minimum distance
    predictions = unique_classes[torch.argmin(dists, dim=1)]

    # Compute F1-score
    f1 = f1_score(test_labels.cpu().numpy(), predictions.cpu().numpy(), average='weighted')

    return f1
#%%
def generate_gaussian_noise(data, snr_db):
    """
    Generate complex-valued Gaussian noise given an SNR and apply it to the data.

    Args:
        data (np.ndarray): Input data array of shape (n_samples, n_features), assumed to be complex-valued.
        snr_db (float): Signal-to-Noise Ratio in decibels (dB).

    Returns:
        np.ndarray: Complex-valued Gaussian noise of the same shape as data.
    """
    # Compute signal power
    signal_power = np.mean(np.abs(data) ** 2, axis=1, keepdims=True)  # Shape: (n_samples, 1)

    # Compute noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate complex Gaussian noise (real + imaginary parts)
    noise_real = np.random.randn(*data.shape) * np.sqrt(noise_power / 2)
    noise_imag = np.random.randn(*data.shape) * np.sqrt(noise_power / 2)

    # Combine real and imaginary parts to form complex noise
    noise = noise_real + 1j * noise_imag

    return noise
