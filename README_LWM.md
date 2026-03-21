---
tags:
- transformers
- wireless-communication
- few-shot-learning
- limited-data
- feature-extraction
- pytorch
#license: mit
datasets:
- DeepMIMO
---

# 📡 **LWM: Large Wireless Model**

**[🚀 Click here to try the Interactive Demo!](https://huggingface.co/spaces/wi-lab/lwm-interactive-demo)**

**[🚀 Click here to try the Colab Notebook!](https://colab.research.google.com/drive/1a_eNi-HG79CY-iwnnlyR41uL8PrG7EIj?usp=sharing)**

LWM is a powerful **pre-trained** model developed as a **universal feature extractor** for wireless channels. As the world's first foundation model crafted for this domain, LWM leverages transformer architectures to extract refined representations from simulated datasets, such as DeepMIMO and Sionna, and real-world wireless data.

<!--
### LWM Tutorial Series

Explore LWM concepts and applications in this compact video series:

<table>
  <tr>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=3sxJR86EFOo" target="_blank">
        <img src="https://img.youtube.com/vi/3sxJR86EFOo/0.jpg" width="180"/>
        <div style="margin-top:4px;padding:4px 12px;background:#f97316;color:white;border-radius:6px;font-weight:600;">▶ Watch</div>
      </a>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=Coqcya9NzFs" target="_blank">
        <img src="https://img.youtube.com/vi/Coqcya9NzFs/0.jpg" width="180"/>
        <div style="margin-top:4px;padding:4px 12px;background:#f97316;color:white;border-radius:6px;font-weight:600;">▶ Watch</div>
      </a>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=e9KvAXMUuQg" target="_blank">
        <img src="https://img.youtube.com/vi/e9KvAXMUuQg/0.jpg" width="180"/>
        <div style="margin-top:4px;padding:4px 12px;background:#f97316;color:white;border-radius:6px;font-weight:600;">▶ Watch</div>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=ZB5WVvo6q6U" target="_blank">
        <img src="https://img.youtube.com/vi/ZB5WVvo6q6U/0.jpg" width="180"/>
        <div style="margin-top:4px;padding:4px 12px;background:#f97316;color:white;border-radius:6px;font-weight:600;">▶ Watch</div>
      </a>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=5oNnJjos0mo" target="_blank">
        <img src="https://img.youtube.com/vi/5oNnJjos0mo/0.jpg" width="180"/>
        <div style="margin-top:4px;padding:4px 12px;background:#f97316;color:white;border-radius:6px;font-weight:600;">▶ Watch</div>
      </a>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=_RObWck3MMw" target="_blank">
        <img src="https://img.youtube.com/vi/_RObWck3MMw/0.jpg" width="180"/>
        <div style="margin-top:4px;padding:4px 12px;background:#f97316;color:white;border-radius:6px;font-weight:600;">▶ Watch</div>
      </a>
    </td>
  </tr>
</table>
-->

### How is LWM built?

The LWM model’s structure is based on transformers, allowing it to capture both **fine-grained and global dependencies** within channel data. Unlike traditional models that are limited to specific tasks, LWM employs a **self-supervised** approach through our proposed technique, Masked Channel Modeling (MCM). This method trains the model on unlabeled data by predicting masked channel segments, enabling it to learn intricate relationships between antennas and subcarriers. Utilizing **bidirectional attention**, LWM interprets the full context by attending to both preceding and succeeding channel segments, resulting in embeddings that encode comprehensive spatial information, making them applicable to a variety of scenarios.

### What does LWM offer?

LWM provides a universal feature extraction framework that can be applied across diverse **wireless communication and sensing** tasks. It is built to handle complex wireless environments, capturing channel characteristics in a way that facilitates robust performance across different scenarios and conditions.

Trained on hundreds of thousands of wireless channel samples, LWM has been designed to generalize across varied environments—from dense urban areas to synthetic setups, ensuring its adaptability and consistency across a broad spectrum of wireless tasks.

### How is LWM used?

LWM is designed to be easily integrated into downstream applications as a source of high-quality **embeddings** that encapsulate complex channel features. By feeding raw wireless channel data into the pre-trained model, users obtain embeddings that capture essential spatial relationships and interactions within the channel environment.

These embeddings provide a versatile and contextualized representation of wireless data, which can be leveraged across different applications. By utilizing the pre-trained model in this way, users can **reduce the need for extensive labeled data** while benefiting from embeddings that retain the critical properties of the original channel.

### Advantages of Using LWM

- **Various Tasks**: Self-supervised and pre-trained without labels, LWM excels in a wide range of wireless tasks, offering flexibility and performance
- **Limited Data**: With LWM embeddings, downstream tasks achieve high accuracy with less data, cutting reliance on large datasets
- **Various Environments**: Pre-trained on diverse data, LWM excels in various environments from urban to rural areas, ensuring reliable performance

Join the growing community of researchers using LWM for their wireless communication and sensing research, and unlock a new level of performance and insight in your models!
---

Please cite the following paper if you use the LWM model or any modified parts:
```
@misc{alikhani2024largewirelessmodellwm,
      title={Large Wireless Model (LWM): A Foundation Model for Wireless Channels}, 
      author={Sadjad Alikhani and Gouranga Charan and Ahmed Alkhateeb},
      year={2024},
      eprint={2411.08872},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2411.08872}, 
}
```

## 🛠 **How to Use**

### 1. **Install Conda**

First, ensure that you have a package manager like **Conda** installed to manage your Python environments and packages. You can install **Conda** via **Anaconda** or **Miniconda**.

- **Anaconda** includes a comprehensive scientific package suite. Download it [here](https://www.anaconda.com/products/distribution).
- **Miniconda** is a lightweight version that includes only Conda and Python. Download it [here](https://docs.conda.io/en/latest/miniconda.html).

Once installed, you can use Conda to manage environments.

---

### 2. **Create a New Environment**

After installing Conda, follow these steps to create a new environment and install the required packages.

#### **Step 1: Create a new environment**

To begin, open the **Anaconda PowerShell Prompt** and create a new Conda environment named `lwm_env`:

```bash
conda create -n lwm_env
```

#### **Step 2: Activate the environment**

Activate the environment:

```bash
conda activate lwm_env
```

---

### 3. **Install Required Packages**

Once the environment is activated, install the necessary packages.

#### **Install CUDA-enabled PyTorch**

Although inference can run efficiently on a CPU, you may need a GPU for training more resource-intensive downstream tasks. Visit [this page](https://pytorch.org/get-started/locally/) and select the appropriate options based on your system's specifications. The website will generate a tailored installation command.

For instance, on an NVIDIA system, you can use a command like the following with the appropriate CUDA version for your system:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

This command installs PyTorch with CUDA support for GPU-accelerated training. Ensure that the specified CUDA version is compatible with your system, adjusting it if necessary.

> **Note:** If you encounter issues installing CUDA-enabled PyTorch, verify your CUDA version compatibility. It might also be due to conflicting installation attempts—try a fresh environment.

#### **Install Other Required Packages via Conda Forge**

```bash
conda install python numpy pandas matplotlib tqdm -c conda-forge
```

#### **Install DeepMIMOv3 with pip**

```bash
pip install DeepMIMOv3
```

---

### 4. **Clone the Dataset Scenarios**

The following functions will help you clone specific dataset scenarios from a repository:

```python
import subprocess
import os
import shutil

def clone_dataset_scenario(repo_url, model_repo_dir="./LWM", scenarios_dir="scenarios"):
    """
    Clones all scenarios from a repository, ensuring all files (small and large) are downloaded.

    Args:
        repo_url (str): URL of the Git repository
        model_repo_dir (str): Path to the model repository
        scenarios_dir (str): Directory name for storing scenarios
    """
    # Ensure we're in the correct directory structure
    current_dir = os.path.basename(os.getcwd())
    if current_dir == "LWM":
        model_repo_dir = "."

    # Create the scenarios directory if it doesn't exist
    scenarios_path = os.path.join(model_repo_dir, scenarios_dir)
    os.makedirs(scenarios_path, exist_ok=True)

    # Store the original working directory
    original_dir = os.getcwd()

    try:
        # Clean up any existing temp directory
        if os.path.exists(scenarios_path):
            shutil.rmtree(scenarios_path)

        # Clone the entire repository (including all files)
        print(f"Cloning entire repository into temporary directory...")
        subprocess.run([
            "git", "clone",
            repo_url,
            scenarios_path
        ], check=True)

        # Navigate to the temporary clone directory
        os.chdir(scenarios_path)

        # Pull all files using Git LFS
        print(f"Pulling all files using Git LFS...")
        subprocess.run(["git", "lfs", "install"], check=True)  # Ensure LFS is installed
        subprocess.run(["git", "lfs", "pull"], check=True)  # Pull all LFS files

        print(f"Successfully cloned all scenarios into {scenarios_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error cloning scenarios: {str(e)}")
    finally:
        # Clean up temporary directory
        if os.path.exists(scenarios_path):
            shutil.rmtree(scenarios_path)
        # Return to original directory
        os.chdir(original_dir)
```

---

### 5. **Clone the Model Repository**

Now, clone the **LWM** model repository to your local system.

```bash
# Step 1: Clone the model repository (if not already cloned)
model_repo_url = "https://huggingface.co/wi-lab/lwm"
model_repo_dir = "./LWM"

if not os.path.exists(model_repo_dir):
    print(f"Cloning model repository from {model_repo_url}...")
    subprocess.run(["git", "clone", model_repo_url, model_repo_dir], check=True)
```

---

### 6. **Clone the Desired Dataset Scenarios**

You can now clone specific scenarios from the DeepMIMO dataset, as detailed in the table below:

📊 **Dataset Overview**

| 📊 **Dataset** | 🏙️ **City**         | 👥 **Number of Users** | 🔗 **DeepMIMO Page**                                                                                       |
|----------------|----------------------|------------------------|------------------------------------------------------------------------------------------------------------|
| Dataset 0      | 🌆 Denver             | 1354                   | [DeepMIMO City Scenario 18](https://www.deepmimo.net/scenarios/deepmimo-city-scenario18/)                   |
| Dataset 1      | 🏙️ Indianapolis       | 3248                   | [DeepMIMO City Scenario 15](https://www.deepmimo.net/scenarios/deepmimo-city-scenario15/)                   |
| Dataset 2      | 🌇 Oklahoma           | 3455                   | [DeepMIMO City Scenario 19](https://www.deepmimo.net/scenarios/deepmimo-city-scenario19/)                   |
| Dataset 3      | 🌆 Fort Worth         | 1902                   | [DeepMIMO City Scenario 12](https://www.deepmimo.net/scenarios/deepmimo-city-scenario12/)                   |
| Dataset 4      | 🌉 Santa Clara        | 2689                   | [DeepMIMO City Scenario 11](https://www.deepmimo.net/scenarios/deepmimo-city-scenario11/)                   |
| Dataset 5      | 🌅 San Diego          | 2192                   | [DeepMIMO City Scenario 7](https://www.deepmimo.net/scenarios/deepmimo-city-scenario7/)                     |

It is important to note that these six datasets were **not** used during the pre-training of the LWM model, and the high-quality embeddings produced are a testament to LWM’s robust generalization capabilities rather than overfitting.

The operational settings below were used in generating the datasets for both the pre-training of LWM and the downstream tasks. If you intend to use custom datasets, please ensure they adhere to these configurations:

#### **Operational Settings**:
- *Antennas at BS*: 32
- *Antennas at UEs*: 1
- *Subcarriers*: 32
- *Paths*: 20
- *Frequency*: 3.5GHz (By the way, our results are consistent across different frequency ranges.)
  
#### **Clone the Scenarios:**
```python
import numpy as np
dataset_repo_url = "https://huggingface.co/datasets/wi-lab/lwm"  # Base URL for dataset repo

# Clone the requested scenarios
clone_dataset_scenario(dataset_repo_url, model_repo_dir)
```

---

### 7. **Change the Working Directory to LWM**

```bash
if os.path.exists(model_repo_dir):
    os.chdir(model_repo_dir)
    print(f"Changed working directory to {os.getcwd()}")
else:
    print(f"Directory {model_repo_dir} does not exist. Please check if the repository is cloned properly.")
```

---

### 8. **Tokenize and Load the Model**

Before we dive into tokenizing the dataset and loading the model, let's understand how the tokenization process is adapted to the wireless communication context. In this case, **tokenization** refers to segmenting each wireless channel into patches, similar to how Vision Transformers (ViTs) work with images. Each wireless channel is structured as a 32x32 matrix, where rows represent antennas and columns represent subcarriers.

The tokenization process involves **dividing the channel matrix into patches**, with each patch containing information from 16 consecutive subcarriers. These patches are then **embedded** into a 64-dimensional space, providing the Transformer with a richer context for each patch. In this process, **positional encodings** are added to preserve the structural relationships within the channel, ensuring the Transformer captures both spatial and frequency dependencies.

If you choose to apply **Masked Channel Modeling (MCM)** during inference (by setting `gen_raw=False`), LWM will mask certain patches, as it did during pre-training. However, for standard inference, masking isn't necessary unless you want to test LWM's robustness to noisy inputs! The printed LWM loss after inference could show you how well it has predicted the masked patches.

Now, let's move on to tokenize the dataset and load the pre-trained LWM model.

```python
from input_preprocess import tokenizer
from lwm_model import lwm
import torch

scenario_names = np.array([
    "city_18_denver", "city_15_indianapolis", "city_19_oklahoma", 
    "city_12_fortworth", "city_11_santaclara", "city_7_sandiego"
])
scenario_idxs = np.array([0, 1, 2, 3, 4, 5])  # Select the scenario indexes
selected_scenario_names = scenario_names[scenario_idxs]

preprocessed_chs = tokenizer(
    selected_scenario_names=selected_scenario_names,  # Selects predefined DeepMIMOv3 scenarios. Set to None to load your own dataset.
    manual_data=None,  # If using a custom dataset, ensure it is a wireless channel dataset of size (N,32,32) based on the settings provided above.
    gen_raw=True  # Set gen_raw=False to apply masked channel modeling (MCM), as used in LWM pre-training. For inference, masking is unnecessary unless you want to evaluate LWM's ability to handle noisy inputs.
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading the LWM model on {device}...")
model = lwm.from_pretrained(device=device)
```

With this setup, you're ready to pass your tokenized wireless channels through the pre-trained model, extracting rich, context-aware embeddings that are ready for use in downstream tasks.

---

### 9. **Perform Inference**

Before running the inference, it's important to understand the benefits of the different embedding types. The **CLS embeddings (cls_emb)** provide a highly compressed, holistic view of the entire wireless channel, making them ideal for tasks requiring a general understanding, such as classification or high-level decision-making. On the other hand, **channel embeddings (channel_emb)** capture detailed spatial and frequency information from the wireless channel, making them more suitable for complex tasks like beamforming or channel prediction.

You can now perform inference on the preprocessed data using the LWM model.

```python
from inference import lwm_inference, create_raw_dataset
input_types = ['cls_emb', 'channel_emb', 'raw']
selected_input_type = input_types[1]  # Change the index to select LWM CLS embeddings, LWM channel embeddings, or the original input channels.

if selected_input_type in ['cls_emb', 'channel_emb']:
    dataset = lwm_inference(preprocessed_chs, selected_input_type, model, device)
else:
    dataset = create_raw_dataset(preprocessed_chs, device)
```

By selecting either `cls_emb` or `channel_emb`, you leverage the pre-trained model's rich feature extraction capabilities to transform raw channels into highly informative embeddings. If you prefer to work with the original raw data, you can choose the `raw` input type.

---

### 10. **Generate Labels if Necessary**
If your dataset requires labels, you can easily generate them using DeepMIMO data. Here's an example to create labels for either LoS/NLoS classification or beam prediction, depending on the scenario selected:
```python
from input_preprocess import create_labels
tasks = ['LoS/NLoS Classification', 'Beam Prediction']
task = tasks[1] # Choose 0 for LoS/NLoS labels or 1 for beam prediction labels.
labels = create_labels(task, selected_scenario_names, n_beams=64) # For beam prediction, n_beams specifies the number of beams in the codebook. If you're generating labels for LoS/NLoS classification, you can leave this value unchanged as it doesn't impact the label generation.
```

---

### 11. **Leverage the Dataset for Downstream Tasks**

LWM, pre-trained on a vast and diverse dataset using self-supervised learning, does not rely on labeled data. During inference, it transforms raw channels into rich embeddings in real time, capturing both general and intricate patterns within the wireless channels. These embeddings can be directly applied to various downstream tasks, offering a more powerful alternative to using the original channel data.

---

### 12. **Explore the Interactive Demo**

To experience **LWM** interactively, visit our demo hosted on Hugging Face Spaces:

[**Try the Interactive Demo!**](https://huggingface.co/spaces/wi-lab/lwm-interactive-demo)

---

You are now ready to explore the power of **LWM** in wireless communications! Start processing datasets and generate high-quality embeddings to advance your research or applications.

If you have questions or need assistance, feel free to:
- Visit the [Hugging Face Discussions](https://huggingface.co/wi-lab/lwm/discussions) for community support.
- Check out the [LWM website FAQ](https://lwm-wireless.net/community).
- Contact us directly via email at [lwmwireless@gmail.com](mailto:lwmwireless@gmail.com).
