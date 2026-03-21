# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:27:17 2024

This scripts performs the LWM inference on raw channel representations.

@author: Sadjad Alikhani
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import visualize_embeddings
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
#%%
def lwm_inference(model, data, input_type="cls_emb", device="cpu", batch_size=64, visualization=False, labels=None, visualization_method="t-sne"):
    
    if input_type == "raw":
        output_total = data
    else:
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        embeddings = []
        model.eval()
        with torch.no_grad():
            with tqdm(dataloader, desc="Inference", unit="batch") as t:
                for batch in t:
                    
                    input_ids = batch[0].to(device)
                    output = model(input_ids)[0]
                    
                    if input_type == "cls_emb":
                        batch_embeddings = output[:, 0, :] 
                        embeddings.append(batch_embeddings)
                    elif input_type == "channel_emb":
                        batch_embeddings = output[:, 1:, :] 
                        embeddings.append(batch_embeddings)
                        
        output_total = torch.cat(embeddings, dim=0).float()
        
        if visualization:
            visualize_embeddings(output_total.view(output_total.size(0), -1), 
                                 labels, 
                                 method=visualization_method, 
                                 label="Embedding Space")
            visualize_embeddings(data.view(data.size(0), -1), 
                                 labels, 
                                 method=visualization_method, 
                                 label="Original Space")
        
    return output_total
