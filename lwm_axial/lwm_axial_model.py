import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ELEMENT_LENGTH = 16
D_MODEL = 64
MAX_LEN = 129
N_LAYERS = 12
N_HEADS = 8
D_FF = D_MODEL * 4
D_K = D_MODEL // N_HEADS
D_V = D_MODEL // N_HEADS
DROPOUT = 0.1

# Axial Attention Config
ANTENNAS = 32
FREQ_GROUPS = 4 # 32 * 4 = 128 (closest to 129)

from .axial_linear_att import AxialSoftmaxAttention

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        return self.norm(x)

class Embedding(nn.Module):
    def __init__(self, element_length, d_model, max_len, antennas=32, freq_groups=4):
        super().__init__()
        self.element_length = element_length
        self.d_model = d_model
        self.proj = nn.Linear(element_length, d_model)
        
        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        # 2D Positional Embeddings (Registered as Buffers to avoid re-creation)
        self.antennas = antennas
        self.freq_groups = freq_groups
        
        pos_emb_ant = torch.zeros(1, antennas, d_model)
        pos_emb_freq = torch.zeros(1, freq_groups, d_model)
        pos_emb_cls = torch.zeros(1, 1, d_model)
        
        nn.init.normal_(pos_emb_ant, std=0.02)
        nn.init.normal_(pos_emb_freq, std=0.02)
        nn.init.normal_(pos_emb_cls, std=0.02)
        
        # Register as parameters? Users usually want them learnable.
        # Original code used nn.Embedding (learnable).
        # We stick to nn.Parameter BUT we precompute the indices in forward
        # Wait, user said "You are double-normalizing... Remove norm from embedding".
        # User also said "Positional embeddings are recreated every forward pass... Register as buffers".
        # Actually user meant the INDICES tensor creation (arange, repeat) is wasteful.
        # But for embeddings themselves, if they are param, they are fine.
        # I will register the INDICES as buffers if possible, or just keep it simple.
        # Re-reading: "This should be precomputed once and registered as buffers." -> Referring to INDICES.
        
        self.pos_emb_ant = nn.Parameter(pos_emb_ant)
        self.pos_emb_freq = nn.Parameter(pos_emb_freq)
        self.pos_emb_cls = nn.Parameter(pos_emb_cls)

        # Precompute indices
        ant_idx = torch.arange(antennas).repeat_interleave(freq_groups)
        freq_idx = torch.arange(freq_groups).repeat(antennas)
        self.register_buffer('ant_idx', ant_idx, persistent=False)
        self.register_buffer('freq_idx', freq_idx, persistent=False)
        
        # Remove Norm from embedding (as per user request: "Remove norm from embedding OR from first layer")
        # self.norm = LayerNormalization(d_model)

    def forward(self, x, masked_pos=None):
        # x: [B, 129, 16] (Values)
        B, Seq, _ = x.shape
        
        # 1. Project values
        tok_emb = self.proj(x.float())  # [B, 129, D]
        
        # 2. Replace CLS token (Index 0) with learnable embedding
        tok_emb[:, 0:1, :] = self.cls_token.to(dtype=tok_emb.dtype).expand(B, -1, -1)
        
        # 3. Replace MASK tokens with learnable embedding
        if masked_pos is not None:
            masked_pos = masked_pos.long()
            mask_tokens = self.mask_token.to(dtype=tok_emb.dtype).expand(B, masked_pos.shape[1], -1)
            batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand_as(masked_pos)
            tok_emb[batch_idx, masked_pos] = mask_tokens

        # 4. Add 2D Positional Embeddings
        # Lookup embeddings using precomputed buffer indices
        ant_emb = self.pos_emb_ant[:, self.ant_idx, :] # [1, 128, D]
        freq_emb = self.pos_emb_freq[:, self.freq_idx, :] # [1, 128, D]
        
        # Combine
        grid_pos_emb = ant_emb + freq_emb
        
        # Add CLS pos and concat
        full_pos_emb = torch.cat([self.pos_emb_cls, grid_pos_emb], dim=1) # [1, 129, D]
        
        embedding = tok_emb + full_pos_emb
        return embedding # No Norm here

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Correct order: Linear -> ReLU -> Dropout -> Linear -> Dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = AxialSoftmaxAttention(
            d_model=D_MODEL, 
            n_heads=N_HEADS,
            d_k=D_K,
            antennas=ANTENNAS,
            freq_groups=FREQ_GROUPS,
            dropout=DROPOUT
        )
        self.pos_ffn = PoswiseFeedForwardNet()
        self.norm1 = LayerNormalization(D_MODEL)  # For attention
        self.norm2 = LayerNormalization(D_MODEL)  # For FFN
        self.cls_scale = nn.Parameter(torch.tensor(0.5))  # Learnable CLS mixing scale

    def forward(self, x):
        # ---- Attention block (true Pre-Norm) ----
        x_norm = self.norm1(x)
        cls_out, grid_out = self.enc_self_attn(x_norm)
        
        if cls_out is not None:
            delta = torch.cat([cls_out, grid_out], dim=1)
        else:
            delta = grid_out
        
        x = x + delta
        
        # CLS mixing after attention (learnable scale)
        if cls_out is not None:
            cls = x[:, :1, :]
            grid = x[:, 1:, :]
            grid = grid + self.cls_scale * cls
            x = torch.cat([cls, grid], dim=1)
        
        # ---- FFN block (true Pre-Norm) ----
        x = x + self.pos_ffn(self.norm2(x))
        
        return x, None

class lwm(torch.nn.Module):
    def __init__(self, element_length=16, d_model=64, max_len=129, n_layers=12):
        super().__init__()
        self.embedding = Embedding(element_length, d_model, max_len, antennas=ANTENNAS, freq_groups=FREQ_GROUPS)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, d_model)
        self.norm = LayerNormalization(d_model)

        # Untied Decoder: Explicit Linear Layer
        self.element_length = element_length
        self.decoder = nn.Linear(d_model, element_length)
        self.decoder_bias = nn.Parameter(torch.zeros(element_length))
        
        # Initialize decoder with small variance for regression stability
        nn.init.xavier_uniform_(self.decoder.weight)

    @classmethod
    def from_pretrained(cls, ckpt_name='model_weights.pth', device='cuda', use_auth_token=None):
        model = cls().to(device)

        ckpt_path = ckpt_name
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Model loaded successfully from {ckpt_path} to {device}")

        return model

    def forward(self, input_ids, masked_pos=None):
        output = self.embedding(input_ids, masked_pos)
        for layer in self.layers:
            output, _ = layer(output)


        masked_pos = masked_pos.long()[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(F.relu(self.linear(h_masked)))
        
        # Untied Decoder (for regression tasks)
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        return logits_lm, output
