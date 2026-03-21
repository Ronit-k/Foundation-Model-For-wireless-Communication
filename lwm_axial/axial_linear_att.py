import torch
import torch.nn as nn
import torch.nn.functional as F
from lwm_axial.rope import RotaryEmbedding, apply_rotary_pos_emb

class AxialSoftmaxAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, antennas=32, freq_groups=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.inner_dim = self.d_k * n_heads
        
        self.antennas = antennas
        self.freq_groups = freq_groups
        self.dropout_p = dropout
        
        # RoPE 
        self.rope_ant = RotaryEmbedding(self.d_k)
        self.rope_freq = RotaryEmbedding(self.d_k)

        # --- Antigen (Spatial) Attention Projections ---
        self.w_q_ant = nn.Linear(d_model, self.inner_dim)
        self.w_k_ant = nn.Linear(d_model, self.inner_dim)
        self.w_v_ant = nn.Linear(d_model, self.inner_dim)
        self.out_ant = nn.Linear(self.inner_dim, d_model) 
        
        # --- Frequency (Spectral) Attention Projections ---
        self.w_q_freq = nn.Linear(d_model, self.inner_dim)
        self.w_k_freq = nn.Linear(d_model, self.inner_dim)
        self.w_v_freq = nn.Linear(d_model, self.inner_dim)
        self.out_freq = nn.Linear(self.inner_dim, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Norms
        self.norm_ant = nn.LayerNorm(d_model)
        self.norm_freq = nn.LayerNorm(d_model)
        
        # Initialize
        for m in [self.w_q_ant, self.w_k_ant, self.w_v_ant, self.out_ant,
                  self.w_q_freq, self.w_k_freq, self.w_v_freq, self.out_freq]:
            nn.init.xavier_uniform_(m.weight)


    def forward(self, x, mask=None):
        """
        x: [Batch, SeqLen, D_Model]
        """
        B, P, D = x.shape
        
        # 1. Separate CLS token
        cls_token = None
        grid = x
        
        if P > (self.antennas * self.freq_groups):
            cls_token = x[:, 0:1, :] 
            grid = x[:, 1:, :]       
            
        # 2. Reshape to [Batch, Ant, Freq, D]
        grid = grid.reshape(B, self.antennas, self.freq_groups, D)

        # ---------------------------------------------------------
        # AXIS 1: ANTENNA (SPATIAL) ATTENTION + RESIDUAL
        # ---------------------------------------------------------
        
        residual = grid
        
        # -- Antenna Block --
        # [B, F, A, D]
        x_ant = grid.permute(0, 2, 1, 3).contiguous().view(B * self.freq_groups, self.antennas, D)
        
        q_ant = self.w_q_ant(x_ant).view(B * self.freq_groups, self.antennas, self.n_heads, self.d_k).transpose(1, 2)
        k_ant = self.w_k_ant(x_ant).view(B * self.freq_groups, self.antennas, self.n_heads, self.d_k).transpose(1, 2)
        v_ant = self.w_v_ant(x_ant).view(B * self.freq_groups, self.antennas, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE 
        cos, sin = self.rope_ant(v_ant, seq_len=self.antennas)
        q_ant, k_ant = apply_rotary_pos_emb(q_ant, k_ant, cos, sin)
        
        out_ant = F.scaled_dot_product_attention(q_ant, k_ant, v_ant, dropout_p=self.dropout_p if self.training else 0.0)
        
        out_ant = out_ant.transpose(1, 2).contiguous().view(B * self.freq_groups, self.antennas, self.inner_dim)
        out_ant = self.out_ant(out_ant) 
        out_ant = self.dropout(out_ant)
        
        # Reshape back 
        out_ant = out_ant.view(B, self.freq_groups, self.antennas, D).permute(0, 2, 1, 3) 
        
        # Residual
        grid = grid + out_ant
        
        # ---------------------------------------------------------
        # AXIS 2: FREQUENCY (SPECTRAL) ATTENTION + RESIDUAL
        # ---------------------------------------------------------
        
        # Normalize
        grid_norm = self.norm_freq(grid)
        
        # [B*A, F, D]
        x_freq = grid_norm.contiguous().reshape(B * self.antennas, self.freq_groups, D)
        
        q_freq = self.w_q_freq(x_freq).view(B * self.antennas, self.freq_groups, self.n_heads, self.d_k).transpose(1, 2)
        k_freq = self.w_k_freq(x_freq).view(B * self.antennas, self.freq_groups, self.n_heads, self.d_k).transpose(1, 2)
        v_freq = self.w_v_freq(x_freq).view(B * self.antennas, self.freq_groups, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply RoPE 
        cos, sin = self.rope_freq(v_freq, seq_len=self.freq_groups)
        q_freq, k_freq = apply_rotary_pos_emb(q_freq, k_freq, cos, sin)
        
        out_freq = F.scaled_dot_product_attention(q_freq, k_freq, v_freq, dropout_p=self.dropout_p if self.training else 0.0)
        
        out_freq = out_freq.transpose(1, 2).contiguous().view(B * self.antennas, self.freq_groups, self.inner_dim)
        out_freq = self.out_freq(out_freq) 
        out_freq = self.dropout(out_freq)
        
        # Reshape back
        out_freq = out_freq.view(B, self.antennas, self.freq_groups, D)
        
        # Final Residual
        grid = grid + out_freq
        
        # ---------------------------------------------------------
        # Final Flatten
        # ---------------------------------------------------------
        grid_final = grid.reshape(B, self.antennas * self.freq_groups, D)
        
        if cls_token is not None:
             grid_input = x[:, 1:, :]
        else:
             grid_input = x
             
        delta = grid_final - grid_input.reshape(B, self.antennas * self.freq_groups, D)
        
        # ---------------------------------------------------------
        # CLS Token Update
        # ---------------------------------------------------------
        cls_out = None
        if cls_token is not None:
            grid_mean = delta.mean(dim=1, keepdim=True)
            cls_out = grid_mean

        return cls_out, delta
