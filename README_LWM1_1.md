---
tags:
- transformers
- wireless-communication
- few-shot-learning
- limited-data
- feature-extraction
- pytorch
datasets:
- DeepMIMO
base_model:
- wi-lab/lwm
---

# **LWM 1.1**

**[🚀 Click here to try the Interactive Demo Based on LWM 1.0!](https://huggingface.co/spaces/wi-lab/lwm-interactive-demo)**

**[🚀 Click here to try the Colab Notebook!](https://colab.research.google.com/drive/1uA4ua8xqdc5XUZjzqIK8fRp8FhYtTxKB?authuser=1#scrollTo=4xPULSHkyWv1)**

LWM 1.1 is an **updated pre-trained model** designed for **feature extraction** in wireless channels. Extending LWM 1.0, this version introduces key modifications to improve **scalability**, **generalization**, and **efficiency** across diverse channel configurations. The model is pre-trained on an expanded dataset covering multiple **(N, SC) pairs**, ensuring robustness to varying antenna and subcarrier configurations. LWM 1.1 retains its transformer-based architecture and **Masked Channel Modeling (MCM)** pretraining approach, enabling it to learn structured representations from both **simulated (e.g., DeepMIMO) and real-world** wireless channels. The model supports variable-length inputs, incorporates **bucket-based batching** for memory efficiency, and enables fine-tuning for task-specific adaptation.

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

### 🎥 LWM Tutorial Series

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

### **How is LWM 1.1 built?**  

LWM 1.1 is a **transformer-based architecture** designed to model **spatial and frequency dependencies** in wireless channel data. It utilizes an enhanced **Masked Channel Modeling (MCM)** pretraining approach, with an increased masking ratio to improve feature learning and generalization. The introduction of **2D patch segmentation** allows the model to jointly process spatial (antenna) and frequency (subcarrier) relationships, providing a more structured representation of the channel. Additionally, **bucket-based batching** is employed to efficiently handle variable-sized inputs without excessive padding, ensuring memory-efficient training and inference. These modifications enable LWM 1.1 to extract meaningful embeddings from a wide range of wireless scenarios, improving its applicability across different system configurations.

### **What does LWM 1.1 offer?**  

LWM 1.1 serves as a **general-purpose feature extractor** for wireless communication and sensing tasks. Pretrained on an expanded and more diverse dataset, it effectively captures channel characteristics across various environments, including **dense urban areas, simulated settings, and real-world deployments**. The model's increased capacity and optimized pretraining strategy improve the quality of extracted representations, enhancing its applicability for downstream tasks.  

### **How is LWM 1.1 used?**  

LWM 1.1 is designed for seamless integration into **wireless communication pipelines** as a pre-trained **embedding extractor**. By processing raw channel data, the model generates structured representations that encode **spatial, frequency, and propagation characteristics**. These embeddings can be directly used for downstream tasks, reducing the need for extensive labeled data while improving model efficiency and generalization across different system configurations.

### **Advantages of Using LWM 1.1**

- **Enhanced Flexibility**: Handles diverse channel configurations with no size limitations.
- **Refined Embeddings**: Improved feature extraction through advanced pretraining and increased model capacity.
- **Efficient Processing**: Memory-optimized with bucket-based batching for variable-sized inputs.
- **Broad Generalization**: Trained on a larger, more diverse dataset for reliable performance across environments.
- **Task Adaptability**: Fine-tuning options enable seamless integration into a wide range of applications.

For example, the following figure demonstrates the advantages of using **LWM-based highly compact CLS embeddings** and **high-dimensional channel embeddings** over raw channels for the LoS/NLoS classification task. The raw dataset is derived from channels of size (32, 32) between BS 3 and 8,299 users in the densified Denver scenario of the DeepMIMO dataset.

<p align="center">
  <img src="https://huggingface.co/wi-lab/lwm-v1.1/resolve/main/images/los_perf.png" alt="LoS/NLoS Classification Performance" width="600"/>
</p>

<p align="center">
  <strong>Figure:</strong> This figure shows the F1-score comparison of models trained with wireless channels and their LWM embeddings for LoS/NLoS classification.
</p>

---

# **Key Improvements in LWM-v1.1**  

### **1️⃣ Expanded Input Flexibility**  
- **Removed Fixed Channel Size Constraints**: Supports multiple **(N, SC)** configurations instead of being restricted to (32, 32).  
- **Increased Sequence Length**: Extended from **128 to 512**, allowing the model to process larger input dimensions efficiently.  

### **2️⃣ Enhanced Dataset and Pretraining**  
- **Broader Dataset Coverage**: Increased the number of training scenarios from **15 to 140**, improving generalization across environments.  
- **Higher Masking Ratio in MCM**: Increased from **15% to 40%**, making the **Masked Channel Modeling (MCM)** task more challenging and effective for feature extraction.  
- **Larger Pretraining Dataset**: Expanded from **820K to 1.05M** samples for more robust representation learning.  

### **3️⃣ Improved Model Architecture**  
- **Increased Model Capacity**: Parameter count expanded from **600K to 2.5M**, enhancing representational power.  
- **2D Patch Segmentation**: Instead of segmenting channels along a single dimension (antennas or subcarriers), patches now span **both antennas and subcarriers**, improving spatial-frequency feature learning.  

### **4️⃣ Optimized Training and Efficiency**  
- **Adaptive Learning Rate Schedule**: Implemented **AdamW with Cosine Decay**, improving convergence stability.  
- **Computational Efficiency**: Reduced the number of attention heads per layer from **12 to 8**, balancing computational cost with feature extraction capability.  

---

### **Comparison of LWM Versions**  

| Feature                     | LWM 1.0                | **LWM 1.1**          |  
|-----------------------------|-------------------------|-----------------------|  
| Channel Size Limitation     | Fixed at (32, 32)       | **Supports multiple (N, SC) pairs**  |  
| Sequence Length Support     | 128 (16-dimensional)    | **512 (32-dimensional)**               |
| Pre-training Samples        | 820K                    | **1.05M**             |  
| Pre-training Scenarios      | 15                      | **140**               |  
| Masking Ratio               | 15%                     | **40%**               |  
| Embedding size              | 64                      | **128**               | 
| Number of Parameters        | 600K                    | **2.5M**              | 
| Segmentation                | 1D                      | **2D**                |   

---

# **Detailed Changes in LWM 1.1**

### **No Channel Size Limitation**  
In **LWM 1.0**, the model was pre-trained on a single (N, SC) = (32, 32) pair, which limited its generalization to other channel configurations. Wireless communication systems in the real world exhibit vast variability in the number of antennas (N) at base stations and subcarriers (SC). To address this limitation, **LWM 1.1** was pre-trained on **20 distinct (N, SC) pairs**, ranging from smaller setups like (8, 32) to more complex setups like (128, 64). This variety enables the model to effectively handle diverse channel configurations and ensures robust generalization without overfitting to specific configurations.

To handle variable-sized inputs efficiently, we implemented **bucket-based batching**, where inputs of similar sizes are grouped together. For example, channels with sizes (32, 64) and (16, 128) are placed in the same bucket, avoiding the excessive padding common in traditional batching approaches. This not only saves memory but also ensures computational efficiency during training. Furthermore, validation samples were drawn as **20% of each bucket**, maintaining a balanced evaluation process across all input sizes.

This approach eliminates the rigidity of fixed channel sizes and positions LWM 1.1 as a versatile model capable of adapting to real-world wireless systems with varying configurations.

### **Larger and More Diverse Pretraining Dataset**  
Generalization is a critical aspect of any foundation model. In **LWM 1.1**, we significantly expanded the training dataset to cover more diverse scenarios and environments. We added **seven new city scenarios**—Charlotte, Denver, Oklahoma, Indianapolis, Fort Worth, Santa Clara, and San Diego—to enrich the model’s exposure to a variety of urban layouts. To enhance the spatial resolution of the training data, we reduced the grid spacing between user locations in the DeepMIMO city scenarios from **2.5m to 1m**, resulting in a higher density of user positions. This adjustment required re-performing ray tracing for all scenarios to generate high-resolution wireless channel data.

Additionally, we introduced **channels from multiple base stations** in each scenario, with distinct (N, SC) pairs to ensure the model encounters a broad range of channel characteristics. This expansion resulted in a total of **1.3 million pre-training samples**, with 20% allocated for validation. This diversity mirrors the variability found in real-world deployments, such as urban, suburban, and rural environments. By exposing LWM 1.1 to this diversity, the model gains the ability to generalize across environments with distinct propagation characteristics, making it more reliable and versatile.

For the full list of pretraining scenarios and specifications, visit:  
[**LWM 1.1 Training Scenarios**](https://lwm-wireless.net/models/LWM1.0/small/model-training)

### **Fine-Tuning for Task-Specific Embedding Generation**  
While pretraining provides a robust feature extractor, downstream tasks often require tailored embeddings. In **LWM 1.1**, we introduced **fine-tuning options** that give users the flexibility to customize the model for specific tasks. Users can now **freeze specific layers** of the model, allowing the remaining layers to adapt to task-specific requirements. This feature is particularly valuable for tasks prone to overfitting, such as **LoS/NLoS classification**, where excessive training on all layers can lead to suboptimal generalization.

To further streamline task-specific adaptation, we provided **default classification and regression heads** for downstream tasks. Users can also define their own custom heads to suit unique requirements, ensuring maximum flexibility and adaptability.

### **Increased Model Capacity**  
LWM 1.1 significantly enhances the model's ability to extract complex features by increasing the **embedding size from 64 to 128**. This increase more than quadruples the model's parameter count, raising it from **600K to 2.5M**. The larger embedding size allows the model to represent more intricate relationships within channel data, improving its performance on challenging tasks such as **beam prediction** and **channel estimation**.

This change directly impacts the quality of the embeddings, making them more expressive and robust across a variety of downstream tasks, even in scenarios with limited labeled data.

### **Challenging MCM Task with Higher Masking Ratio**  
The **Masked Channel Modeling (MCM)** task lies at the core of LWM’s pretraining methodology. In **LWM 1.1**, we made the task more challenging by increasing the **masking ratio from 15% to 40%**. This means that a larger portion of the channel data is masked during training, requiring the model to infer the missing information from contextual dependencies.

This enhancement forces the model to rely on deeper spatial relationships between antennas and subcarriers, rather than learning superficial patterns. As a result, LWM 1.1 produces embeddings that are more robust and better equipped to handle real-world scenarios with incomplete or noisy data.

### **Support for Larger Input Sizes**  
Wireless communication systems are increasingly handling larger channels with higher dimensions. To accommodate these demands, we increased the **maximum sequence length** from **128 to 512** in **LWM 1.1**. This change enables the model to process larger and more detailed channel data without modification, broadening its applicability to high-dimensional wireless tasks. This ensures that LWM-v1.1 remains relevant as the scale and complexity of wireless systems continue to grow.

### **2D Patch Segmentation for Realistic Learning**  
In **LWM 1.0**, patches were segmented based on a single dimension, typically grouping elements from different subcarriers within the same antenna. In **LWM 1.1**, we introduced **2D patch segmentation**, where patches now combine elements from both antennas and subcarriers. This reflects real-world wireless channel dependencies more accurately, as the relationship between antennas and subcarriers is critical in practical deployments.

This multidimensional segmentation increases the complexity of the MCM task, requiring the model to learn deeper and more meaningful dependencies within the data. By better aligning the training methodology with real-world conditions, LWM 1.1 further enhances its ability to generalize and perform in practical scenarios.

### **Optimized Training Strategy**  
Training large models requires carefully designed optimization techniques to ensure smooth convergence and generalization. In **LWM 1.1**, we adopted the **AdamW optimizer**, which improves weight regularization and prevents overfitting compared to traditional Adam. The learning rate schedule was also refined, incorporating an **5-step warmup phase** followed by **cosine decay**. This strategy ensures that the model transitions smoothly from the initial training phase to convergence, maintaining stability and improving overall performance.

### **Improved Computational Efficiency**  
To balance computational efficiency with performance, we reduced the number of **attention heads per layer from 12 to 8** in **LWM 1.1**. This reduction decreases the computational load during both training and inference, making the model more efficient without significantly affecting its ability to extract meaningful features. The streamlined architecture ensures that LWM 1.1 is not only powerful but also practical for deployment in resource-constrained environments.

### **Why These Changes Were Necessary**  
The updates in LWM 1.1 were driven by real-world demands for greater flexibility, scalability, and performance in wireless communication tasks. Removing channel size limitations and diversifying the dataset address the variability inherent in wireless environments. Increasing model capacity and enhancing the MCM task improve the quality of embeddings, while optimized training strategies and computational efficiency make the model practical for a wide range of applications. These changes make LWM 1.1 a significant step forward, ensuring its relevance and impact in advancing wireless communication research.

## **Conclusion**  
**LWM 1.1** represents a major leap forward in wireless communication modeling, offering robust scalability, increased generalization, and adaptability to a wide variety of tasks. From enriched training datasets and challenging pretraining objectives to enhanced model capacity and efficient input handling, LWM 1.1 provides a powerful foundation for wireless communication research and applications.  

### **Try It Now!**  
Explore **LWM 1.1** on Hugging Face with preloaded datasets, fine-tuning options, and pretrained models to kickstart your projects.  
[👉 Access the model here!](https://huggingface.co/wi-lab/lwm-v1.1/tree/main)

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

---

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

# Function to clone a specific dataset scenario folder
def clone_dataset_scenario(scenario_name, repo_url, model_repo_dir="./LWM", scenarios_dir="scenarios"):
    current_dir = os.path.basename(os.getcwd())
    if current_dir == "LWM":
        model_repo_dir = "."

    # Create the scenarios directory if it doesn't exist
    scenarios_path = os.path.join(model_repo_dir, scenarios_dir)
    if not os.path.exists(scenarios_path):
        os.makedirs(scenarios_path)

    scenario_path = os.path.join(scenarios_path, scenario_name)

    # Initialize sparse checkout for the dataset repository
    if not os.path.exists(os.path.join(scenarios_path, ".git")):
        print(f"Initializing sparse checkout in {scenarios_path}...")
        subprocess.run(["git", "clone", "--sparse", repo_url, "."], cwd=scenarios_path, check=True)
        subprocess.run(["git", "sparse-checkout", "init", "--cone"], cwd=scenarios_path, check=True)
        subprocess.run(["git", "lfs", "install"], cwd=scenarios_path, check=True)  # Install Git LFS if needed

    # Add the requested scenario folder to sparse checkout
    print(f"Adding {scenario_name} to sparse checkout...")
    subprocess.run(["git", "sparse-checkout", "add", scenario_name], cwd=scenarios_path, check=True)
    
    # Pull large files if needed (using Git LFS)
    subprocess.run(["git", "lfs", "pull"], cwd=scenarios_path, check=True)

    print(f"Successfully cloned {scenario_name} into {scenarios_path}.")

def clone_dataset_scenarios(selected_scenario_names, dataset_repo_url, model_repo_dir):
    for scenario_name in selected_scenario_names:
        clone_dataset_scenario(scenario_name, dataset_repo_url, model_repo_dir)
```

---

### 5. **Clone the Model Repository**

Now, clone the **LWM-v1.1** model repository to your local system.

```bash
# Step 1: Clone the model repository (if not already cloned)
model_repo_url = "https://huggingface.co/wi-lab/lwm-v1.1"
model_repo_dir = "./LWM-v1.1"

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

If you plan to use custom datasets, please ensure that your complex channel contains at most **8196 elements** (N * SC <= 8196). In **LWM-v1.0**, the input was restricted to complex channels of size (N, SC) = (32, 32). However, with **LWM-v1.1**, you can now feed complex channels of arbitrary sizes, providing greater flexibility for your specific use case! 😊
  
#### **Clone the Scenarios:**
```python
import numpy as np
dataset_repo_url = "https://huggingface.co/datasets/wi-lab/lwm"  # Base URL for dataset repo
scenario_names = np.array(["city_6_miami"])

scenario_idxs = np.array([0])  # Select the scenario index
selected_scenario_names = scenario_names[scenario_idxs]

# Clone the requested scenarios
clone_dataset_scenarios(selected_scenario_names, dataset_repo_url, model_repo_dir)
```

---

## **7. Change the Working Directory to LWM**

Before proceeding, ensure you are in the correct working directory for the **LWM** repository:

```python
import os

if os.path.exists(model_repo_dir):
    os.chdir(model_repo_dir)
    print(f"Changed working directory to {os.getcwd()}")
else:
    print(f"Directory {model_repo_dir} does not exist. Please check if the repository is cloned properly.")
```

This ensures that all paths and dependencies align with the repository structure.

---

Next, we proceed in two distinct directions, each focusing on a critical aspect of **LWM-v1.1**:

1. **INFERENCE AND DOWNSTREAM TASKS**: Utilize the pre-trained LWM-v1.1 model to perform inference and adapt it for specific tasks such as classification or regression.  
2. **PRE-TRAINING LWM-v1.1**: Explore the process of pre-training the model from scratch, including the techniques and datasets used to develop its foundational capabilities.  

The corresponding scripts for these processes can be found in the **`downstream.py`** and **`main.py`** files available at [**Hugging Face Repository**](https://huggingface.co/wi-lab/lwm-v1.1/tree/main). The following sections provide complementary explanations to support their use.

---

# **1. INFERENCE & DOWNSTREAM TASKS**

### **Loading Required Packages and Modules**

To set up your environment for downstream tasks, import the necessary modules and suppress unnecessary warnings:

```python
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
```

### **Setting Parameters for Downstream Tasks**

Define the parameters for your downstream task. This includes selecting the desired task, visualization method, and data input types. Additionally, you can either use default tasks or manually define labels for custom tasks. If your primary goal is to extract **LWM embeddings**, you can skip task definitions and labels.

```python
n_beams = 16
task = ['Beam Prediction', 'LoS/NLoS Classification'][1]  # Default: LoS/NLoS Classification
task_type = ["classification", "regression"][0]  # Default: Classification
visualization_method = ["pca", "umap", "tsne"][2]  # Default: TSNE
input_types = ["cls_emb", "channel_emb", "raw"]  # Supported input types
train_ratios = [.001, .01, .05, .1, .25, .5, .8]  # Fraction of data for training
fine_tuning_status = [None, ["layers.8", "layers.9", "layers.10", "layers.11"], "full"]  # Fine-tuning configurations
selected_scenario_names = [scenarios_list()[6]]  # Choose a specific scenario
```

#### **Parameters**

1. **`n_beams`**:  
   - Specifies the number of beams in the codebook for the **Beam Prediction** task.
   - For example, `16` beams indicate 16 possible output classes when predicting the optimal beam index.

2. **`task`**:  
   - Defines the downstream task to perform:
     - `'Beam Prediction'`: Predicts the optimal beam index from sub-6GHz channels for mmWave communications.
     - `'LoS/NLoS Classification'`: Classifies channels into **Line-of-Sight (LoS)** or **Non-Line-of-Sight (NLoS)**.  
   - Here, **LoS/NLoS Classification** is selected (`[1]`).

3. **`task_type`**:  
   - Specifies whether the task involves **classification** (discrete outputs) or **regression** (continuous outputs).  
   - In this case, the task is a **classification problem** (`[0]`).

4. **`visualization_method`**:  
   - Determines how the channel embeddings will be visualized during evaluation:
     - `"pca"`: Principal Component Analysis for linear dimensionality reduction.
     - `"umap"`: Uniform Manifold Approximation and Projection for capturing non-linear structures.
     - `"tsne"`: t-distributed Stochastic Neighbor Embedding, ideal for clustering visualization.  
   - Here, **t-SNE** is used (`[2]`).

5. **`input_types`**:  
   - Lists the types of inputs supported by the model:
     - `"cls_emb"`: CLS token embeddings of size (n_samples, 128) representing holistic channel features.
     - `"channel_emb"`: Lower-level embeddings of szie (n_samples, n_patches, 128) derived from channel patches.
     - `"raw"`: Raw wireless channel data without preprocessing.  
   - These input types enable flexibility in evaluating and fine-tuning the model.

6. **`train_ratios`**:  
   - Specifies the fraction of the dataset used for training:
     - Values like `0.001` (0.1%) simulate data-limited scenarios, while `0.8` (80%) allows training with most of the dataset.  
   - This parameter is particularly useful for analyzing model performance under varying levels of labeled data availability. The LWM model is proven to perform most effectively compared to raw channel representations in data-limited scenarios.

7. **`fine_tuning_status`**:  
   - Determines how the pretrained **LWM-v1.1** model will be fine-tuned:
     - `None`: Uses the pretrained model as-is, without fine-tuning.
     - `["layers.8", "layers.9", "layers.10", "layers.11"]`: Fine-tunes only the last four encoder layers, suitable for task-specific adaptation. The set of desired layers can be selected ("layers.0" to "layers.11)".
     - `"full"`: Fine-tunes the entire model, ideal for significant task adaptation.  
   - These configurations help balance performance improvements with computational efficiency.

8. **`selected_scenario_names`**:  
   - Specifies the scenario(s) from the dataset to use for training and evaluation.  
   - **`scenarios_list()`**: A utility function that provides all available scenarios in the dataset.  
   - `[6]`: Selects the 6th scenario, representing a specific wireless environment and base station configuration. In this case, Scenario 6 corresponds to channels of size (16, 32) between BS 3 and users in the densified **Miami** scenario. The dataset is available at [**Hugging Face Datasets**](https://huggingface.co/datasets/wi-lab/lwm/tree/main).

---

#### **Preprocessing**

The `tokenizer` function processes the raw wireless channel data based on the selected parameters:

```python
preprocessed_data, labels, raw_chs = tokenizer(
    selected_scenario_names, 
    bs_idxs=[3], 
    load_data=False, 
    task=task, 
    n_beams=n_beams
)
```

1. **`selected_scenario_names`**: Defines the scenario(s) to tokenize.
2. **`bs_idxs`**: Specifies the base station(s) to include in the scenario.  
   - `[3]`: Includes only the 3rd base station.
3. **`load_data`**:  
   - `False`: Specifies that the function should generate the densified DeepMIMO scenario and save it. If the scenario has already been pre-saved, set this parameter to `True`.
4. **`task`**: Sets the downstream task (e.g., Beam Prediction or LoS/NLoS Classification).
5. **`n_beams`**: Specifies the number of beams for **Beam Prediction** tasks.

**Outputs**:
- **`preprocessed_data`**: Tokenized wireless channel data, formatted for the model.
- **`labels`**: Labels corresponding to the task (e.g., beam indexes or LoS/NLoS categories).
- **`raw_chs`**: Original raw wireless channel data for comparison or visualization.

---

### **Loading the Pretrained LWM-v1.1 Model**

Load the **LWM-v1.1** pretrained model and prepare it for downstream tasks. The model is initialized on the specified GPU(s) or CPU if no GPU is available.

```python
from lwm_model import lwm  # Adjust the import path as needed

gpu_ids = [0]
device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = lwm().to(device)

# Load the pretrained model state
model_name = "model.pth"
state_dict_path = f"models/{model_name}"
state_dict = torch.load(state_dict_path, map_location=device)

# Clean state dictionary for DataParallel compatibility
clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict)

# Use multiple GPUs if specified
if len(gpu_ids) > 1:
    model = nn.DataParallel(model, device_ids=gpu_ids)

print(f"Model loaded successfully on device: {device}")
```

---

### **Visualizing the Original Channel and Embedding Spaces**

If you wish to visualize how the original channel space and embedding space align with task labels before fine-tuning, or if you simply want to perform inference on raw channels:

```python
chs = lwm_inference(
    model, 
    preprocessed_data, 
    input_type="cls_emb", 
    device=device, 
    batch_size=64, 
    visualization=True, 
    labels=labels, 
    visualization_method=visualization_method
)
```

This generates embeddings or visualizations, depending on your configuration. For example, the following figures show the 2D T-SNE representations of original, embedding, and fine-tuned embedding spaces for the LoS/NLoS classification and beam prediction tasks.

### **LoS/NLoS Classification Task**

| ![Image 1](https://huggingface.co/wi-lab/lwm-v1.1/resolve/main/images/los_raw.png) | ![Image 2](https://huggingface.co/wi-lab/lwm-v1.1/resolve/main/images/los_embedding_noFT.png) | ![Image 3](https://huggingface.co/wi-lab/lwm-v1.1/resolve/main/images/los_embedding_FT.png) |
|:---------------------------------------------:|:---------------------------------------------:|:---------------------------------------------:|
| **Raw Channels**                              | **General-Purpose Embeddings**                | **Task-Specific Embeddings**                  |

### **Beam Prediction Task**

| ![Image 4](https://huggingface.co/wi-lab/lwm-v1.1/resolve/main/images/bp_raw.png) | ![Image 5](https://huggingface.co/wi-lab/lwm-v1.1/resolve/main/images/bp_embedding_noFT.png) | ![Image 6](https://huggingface.co/wi-lab/lwm-v1.1/resolve/main/images/bp_embedding_FT.png) |
|:---------------------------------------------:|:---------------------------------------------:|:---------------------------------------------:|
| **Raw Channels**                              | **General-Purpose Embeddings**                | **Task-Specific Embeddings**                  |


---

### **Fine-Tuning the Pretrained Model**

Fine-tune the **LWM-v1.1** model for your specific downstream task. You can choose to leave the pretrained model unchanged, fine-tune specific encoder layers, or fine-tune the entire model. Avoid over-parameterizing the downstream model to maintain generalization.

```python
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
            
            # Prepare data loaders
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
            
            # Fine-tune LWM
            fine_tuned_model, best_model_path, train_losses, val_losses, f1_scores, attn_maps_ft = finetune(
                base_model=model, 
                train_loader=train_loader,
                val_loader=val_loader,
                task_type=task_type,
                input_type=input_type,
                num_classes=n_beams if task == 'Beam Prediction' else 2 if task == 'LoS/NLoS Classification' else None,
                output_dim=target.shape[-1] if task_type == 'regression' else None,
                use_custom_head=True,
                fine_tune_layers=fine_tuning_stat,
                optimizer_config={"lr": 1e-3},
                epochs=15,
                device=device,
                task=task
            )
            
            results[fine_tuning_stat_idx][input_type_idx][train_ratio_idx] = f1_scores[-1]
```

---

### **Visualizing Fine-Tuning Results**

Visualize the effect of fine-tuning on performance across different training ratios, input types, and fine-tuning configurations:

```python
markers = ['o', 's', 'D']
labels = ['CLS Emb', 'CHS Emb', 'Raw']
fine_tuning_status_labels = ['No FT', 'Partial FT', 'Full FT']
line_styles = ['-', '--', ':']
colors = plt.cm.viridis(np.linspace(0, 0.8, len(labels)))

plt.figure(figsize=(12, 8), dpi=500)
for ft_idx, (ft_status_label, line_style) in enumerate(zip(fine_tuning_status_labels, line_styles)):
    for idx, (marker, label, color) in enumerate(zip(markers, labels, colors)):
        if label == "Raw" and ft_status_label != "No FT":
            continue
        plt.plot(
            train_ratios, 
            results[ft_idx, idx], 
            marker=marker, 
            linestyle=line_style, 
            label=f"{label} ({ft_status_label})", 
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
```

---

### **Comparing the Original Channel Space with Fine-Tuned Embedding Space**

After fine-tuning, compare how the embedding space has adapted to task-specific details:

```python
chs = lwm_inference(
    fine_tuned_model.model, 
    preprocessed_data, 
    input_type="cls_emb", 
    device=device, 
    batch_size=64, 
    visualization=False, 
    labels=labels, 
    visualization_method=visualization_method
)
```

---

# **2. PRE-TRAINING LWM-v1.1**

This section details the process of pre-training the **LWM 1.1** model, including data preparation, model initialization, and optimization settings. Each step has been carefully designed to enable the model to learn robust and general-purpose embeddings for wireless channel data.

---

### **Loading Required Packages and Modules**

The following packages are required to preprocess data, initialize the model, and train it effectively:

```python
import torch
import torch.nn as nn
from torch.utils.data import random_split
from input_preprocess import tokenizer, scenarios_list
from utils import create_dataloader, count_parameters
import numpy as np
import lwm_model
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.optim import AdamW
from train import train_lwm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
```

---

### **Settings**

Set the key hyperparameters for pretraining:

```python
EPOCHS = 50
BATCH_SIZE = 128 
VAL_BATCH_SIZE = 64 
WARMUP_EPOCHS = 5
BASE_LR = 5e-4
N_ROWS = 4
N_COLUMNS = 4
ELEMENT_LENGTH = N_ROWS * N_COLUMNS * 2
D_MODEL = 128 
MAX_LEN = 513
N_LAYERS = 12 
WEIGHT_DECAY = 0.05
BETA1 = 0.9
BETA2 = 0.999
MASK_PERCENT = 0.40
N_HEADS = 8
DROPOUT = 0.1
```

- **Data Parameters**:
  - **`N_ROWS` and `N_COLUMNS`**: Number of rows and columns in each channel patch (4 antennas × 4 subcarriers).
  - **`ELEMENT_LENGTH`**: Number of elements in each patch, including real and imaginary parts (4 * 4 * 2 = 32).
  - **`MAX_LEN`**: Maximum input length (including positional encoding).

- **Model Hyperparameters**:
  - **`D_MODEL`**: Embedding size (128).
  - **`N_LAYERS`**: Number of transformer layers (12).
  - **`N_HEADS`**: Number of attention heads (8).
  - **`DROPOUT`**: Dropout probability (0.1).

- **Training Hyperparameters**:
  - **`EPOCHS`**: Total number of epochs (50).
  - **`BATCH_SIZE`**: Batch size for training (128) and validation (64).
  - **`BASE_LR` and `WARMUP_EPOCHS`**: Initial learning rate (5e-4) and warmup period (5 epochs).
  - **`MASK_PERCENT`**: Percentage of masked patches during pretraining (40%).

---

### **Generating the Dataset**

The dataset is prepared by tokenizing scenarios using the `tokenizer` function:

```python
bs_idxs = [1, 2, 3] 
selected_scenario_names = scenarios_list()[:80] 
preprocessed_data = tokenizer(
    selected_scenario_names, 
    MAX_LEN, 
    masking_percent=MASK_PERCENT, 
    mask=True, 
    seed=42
)
```

- **Parameters**:
  - **`bs_idxs`**: Selects base stations 1, 2, and 3 for data generation.
  - **`selected_scenario_names`**: Uses the first 80 scenarios from the `scenarios_list`.
  - **`masking_percent`**: Masks 40% of patches in each channel during pretraining.

- **Outputs**:
  - **`preprocessed_data`**: A dictionary where keys are scenario names, and values are preprocessed samples.

---

### **Splitting the Dataset**

Split the dataset into training, validation, and test sets:

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
train_ratio = 0.8
val_ratio = 0.2
train_data = {}
val_data = {}
test_data = {}

for key, samples in preprocessed_data.items():
    total_samples = len(samples)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    
    train_data[key], val_data[key], test_data[key] = random_split(
        samples, [train_size, val_size, test_size]
    )

train_loaders = create_dataloader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loaders = create_dataloader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=False)
```

- **Data Ratios**:
  - **`train_ratio`**: 80% of the data for training.
  - **`val_ratio`**: 20% for validation.
  - Remaining samples are reserved for testing.

- **Data Loaders**:
  - `train_loaders` and `val_loaders` provide batched data for training and validation.

---

### **Initializing the Model**

Initialize **LWM 1.1** and optionally load a pretrained checkpoint:

```python
load_model = True
gpu_ids = [0]
device = torch.device("cuda:0")
model = lwm_model.lwm().to(device)

if load_model:
    model_name = "model.pth"
    state_dict = torch.load(f"models/{model_name}", map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

model = nn.DataParallel(model, gpu_ids)
print(f"Model loaded successfully on GPU {device.index}")
n_parameters = count_parameters(model)
print(f"Number of trainable parameters: {n_parameters:,}")
```

- **GPU Handling**:
  - The model runs on GPU `cuda:0`. It can also use multiple GPUs if specified.

- **Checkpoint Loading**:
  - If `load_model` is `True`, a pretrained checkpoint is loaded, ensuring the model starts with learned weights.

- **Parameter Count**:
  - Displays the number of trainable parameters for transparency.

---

### **Optimizer and Learning Rate Scheduler**

Define the optimizer and learning rate scheduler:

```python
optimizer = AdamW(
    model.parameters(),
    lr=BASE_LR,
    betas=(BETA1, BETA2),
    weight_decay=WEIGHT_DECAY
)

def lr_lambda(current_step):
    if current_step < WARMUP_STEPS:
        return current_step / WARMUP_STEPS
    else:
        scaled_progress = (current_step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * scaled_progress))
        return cosine_decay * (BASE_LR - MIN_LR) / BASE_LR + MIN_LR / BASE_LR

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
```

- **AdamW Optimizer**:
  - Includes weight decay for better generalization.
- **Learning Rate Scheduler**:
  - Combines linear warmup and cosine decay for smooth training.

---

### **Training the Model**

Train the model using the `train_lwm` function:

```python
pretrained_model = train_lwm(
    model,
    train_loaders,
    val_loaders,
    optimizer,
    scheduler,
    EPOCHS,
    device=device
)
```

- **Inputs**:
  - **`model`**: The initialized LWM model.
  - **`train_loaders` and `val_loaders`**: Data loaders for training and validation.
  - **`optimizer` and `scheduler`**: Configured optimizer and learning rate scheduler.
  - **`EPOCHS`**: Number of training epochs.
  - **`device`**: Specifies whether training occurs on GPU or CPU.

- **Output**:
  - **`pretrained_model`**: The trained LWM-v1.1 model.

---

# **Explore the Interactive Demo**

Experience **LWM** interactively via our Hugging Face Spaces demo:  
[**Try the Interactive Demo!**](https://huggingface.co/spaces/wi-lab/lwm-interactive-demo)

---

You are now ready to explore the power of **LWM** in wireless communications! Start processing datasets and generate high-quality embeddings to advance your research or applications.

If you have questions or need assistance, feel free to:
- Visit the [Hugging Face Discussions](https://huggingface.co/wi-lab/lwm-v1.1/discussions) for community support.
- Check out the [LWM website FAQ](https://lwm-wireless.net/community).
- Contact us directly via email at [lwmwireless@gmail.com](mailto:lwmwireless@gmail.com).
