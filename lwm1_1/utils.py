from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch
import numpy as np
#%%
def create_dataloader(grouped_data, batch_size, shuffle):

    dataloaders = {}

    for seq_length, group in grouped_data.items():
        
        print(f"dataloader in progress ...\nkey: {seq_length}")
        
        ## Uncomment the following line if you run out of memory during pre-training
        # batch_size = batch_size // 8 if seq_length >= 5 else batch_size
        
        # Unpack samples for the current group
        input_ids, masked_tokens, masked_pos = zip(*group)

        # Convert to tensors
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.float32)
        masked_tokens_tensor = torch.tensor(masked_tokens, dtype=torch.float32)
        masked_pos_tensor = torch.tensor(masked_pos, dtype=torch.long)

        # Create TensorDataset and DataLoader
        dataset = TensorDataset(input_ids_tensor, masked_tokens_tensor, masked_pos_tensor)
        dataloaders[seq_length] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    return dataloaders
#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#%%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def visualize_embeddings(embeddings, labels, method="pca", label=None):
    """
    Visualize embeddings using PCA, UMAP, or t-SNE with color-coded labels.

    Args:
        embeddings (torch.Tensor or np.ndarray): Embeddings to visualize, shape (n_samples, n_features).
        labels (torch.Tensor or np.ndarray): Class labels corresponding to embeddings, shape (n_samples,).
        method (str): Dimensionality reduction method ('pca', 'umap', or 'tsne').
        title (str): Title of the plot.
    """
    # Convert to numpy if input is a torch.Tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Apply the selected dimensionality reduction method
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
    elif method.lower() == "umap":
        reducer = umap.UMAP(n_components=2, n_neighbors=16, random_state=42)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, init="random")
    else:
        raise ValueError("Invalid method. Choose from 'pca', 'umap', or 'tsne'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    # Create a scatter plot with color-coding based on labels
    plt.figure(figsize=(10, 8))
    num_classes = len(np.unique(labels))
    colors = plt.cm.get_cmap("tab10", num_classes)

    for class_idx in range(num_classes):
        class_points = reduced_embeddings[labels == class_idx]
        plt.scatter(
            class_points[:, 0], class_points[:, 1],
            label=f"Class {class_idx}",
            alpha=0.6
        )

    # Customize the plot
    plt.title(f"{label} ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.show()
#%%
def generate_gaussian_noise(data, snr_db):
    """
    Generate Gaussian noise given an SNR and apply it to the data.

    Args:
        data (torch.Tensor): Input data tensor of shape (n_samples, seq_len, feature_dim).
        snr_db (float): Signal-to-Noise Ratio in decibels (dB).

    Returns:
        torch.Tensor: Data with Gaussian noise applied.
    """
    # Separate the input data to exclude the first channel
    a = data[:, 1:, :]  # Shape: (n_samples, seq_len-1, feature_dim)
    flat_data = a.view(a.size(0), -1)  # Flatten data to calculate power
    signal_power = torch.mean(flat_data**2, dim=1, keepdim=True)  # Shape: (n_samples, 1)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(flat_data) * torch.sqrt(noise_power)
    noise = noise.view_as(a)
    noise = torch.cat((torch.zeros_like(data[:, :1, :]), noise), dim=1)  # Add zero noise for the first channel

    return noise
#%%
def plot_coverage(rxs, cov_map, dpi=200, figsize=(6,4), cbar_title=None, title=False,
                  scat_sz=.5, tx_pos=None, tx_ori=None, legend=False, lims=None,
                  proj_3D=False, equal_aspect=False, tight=True, cmap='tab20'):
    
    plt_params = {'cmap': cmap}
    if lims:
        plt_params['vmin'], plt_params['vmax'] = lims[0], lims[1]
    
    n = 3 if proj_3D else 2 # n coordinates to consider 2 = xy | 3 = xyz
    
    xyz = {'x': rxs[:,0], 'y': rxs[:,1]}
    if proj_3D:
        xyz['zs'] = rxs[:,2]
        
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize,
                           subplot_kw={'projection': '3d'} if proj_3D else {})
    
    im = plt.scatter(**xyz, c=cov_map, s=scat_sz, marker='s', **plt_params)

    cbar = plt.colorbar(im, label='' if not cbar_title else cbar_title)
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    # TX position
    if tx_pos is not None:
        ax.scatter(*tx_pos[:n], marker='P', c='r', label='TX')
    
    # TX orientation
    if tx_ori is not None and tx_pos is not None: # ori = [azi, el]
        # positive azimuths point left (like positive angles in a unit circle)    
        # positive elevations point up
        r = 30 # ref size of pointing direction
        tx_lookat = np.copy(tx_pos)
        tx_lookat[:2] += r * np.array([np.cos(tx_ori[2]), np.sin(tx_ori[2])]) # azimuth
        tx_lookat[2] += r * np.sin(tx_ori[1]) # elevation
        
        line_components = [[tx_pos[i], tx_lookat[i]] for i in range(n)]
        line = {key:val for key,val in zip(['xs', 'ys', 'zs'], line_components)}
        if n == 2:
            ax.plot(line_components[0], line_components[1], c='k', alpha=.5, zorder=3)
        else:
            ax.plot(**line, c='k', alpha=.5, zorder=3)
        
    if title:
        ax.set_title(title)
    
    if legend:
        plt.legend(loc='upper center', ncols=10, framealpha=.5)
    
    if tight:
        s = 1
        mins, maxs = np.min(rxs, axis=0)-s, np.max(rxs, axis=0)+s
        if not proj_3D:
            plt.xlim([mins[0], maxs[0]])
            plt.ylim([mins[1], maxs[1]])
        else:
            ax.axes.set_xlim3d([mins[0], maxs[0]])
            ax.axes.set_ylim3d([mins[1], maxs[1]])
            if tx_pos is None:
                ax.axes.set_zlim3d([mins[2], maxs[2]])
            else:
                ax.axes.set_zlim3d([np.min([mins[2], tx_pos[2]]),
                                    np.max([mins[2], tx_pos[2]])])
    
    if equal_aspect and not proj_3D: # disrups the plot
        plt.axis('scaled')
    
    return fig, ax, cbar
#%%
def prepare_loaders(
    preprocessed_data, 
    labels=None, 
    selected_patches_idxs=None, 
    input_type="raw", 
    task_type="classification", 
    feature_selection=False, 
    train_ratio=0.8, 
    batch_size=64, 
    seed=42  # Default seed for reproducibility
):
    """
    Prepares datasets and data loaders for training and validation.

    Args:
        preprocessed_data (torch.Tensor): The input data, either raw or preprocessed.
        labels (torch.Tensor, optional): The labels for classification tasks.
        selected_patches_idxs (torch.Tensor, optional): Indices of selected patches for feature selection.
        input_type (str): "raw" or "processed" to specify input data type.
        task_type (str): "classification" or "regression".
        feature_selection (bool): Whether to perform feature selection based on selected_patches_idxs.
        train_ratio (float): Proportion of data to use for training (remaining for validation).
        batch_size (int): Batch size for data loaders.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Prepare samples
    if input_type == "raw":
        if feature_selection and selected_patches_idxs is not None:
            batch_indices = torch.arange(preprocessed_data.size(0)).unsqueeze(1)  # Shape: [batch_size, 1]
            samples = torch.tensor(preprocessed_data[batch_indices, selected_patches_idxs], dtype=torch.float32)
        else:
            samples = torch.tensor(preprocessed_data[:, 1:], dtype=torch.float32)  # raw_chs
    else:
        samples = torch.tensor(preprocessed_data, dtype=torch.float32)

    # Prepare dataset
    if task_type == "classification":
        if labels is None:
            raise ValueError("Labels are required for classification tasks.")
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(samples, labels)
        target = 0  # REVISE if needed
    elif task_type == "regression":
        target = samples[:, 1:, :].view(samples.size(0), -1)  # Reshape for regression targets
        dataset = TensorDataset(samples, target)
    else:
        raise ValueError("Invalid task_type. Choose 'classification' or 'regression'.")

    # Set random seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Split dataset into training and validation
    n_samples = len(dataset)
    train_size = int(train_ratio * n_samples)
    val_size = n_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    return train_loader, val_loader, samples, target