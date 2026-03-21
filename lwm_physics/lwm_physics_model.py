import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lwm_physics.physics_priors import compute_physics_bias

# ==========================================
# Coordinate Attention (Ported from lwm_ca/coordatt.py)
# ==========================================
class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """
    Coordinate attention block for (N, C, H, W) inputs.
    Includes BatchNorm2d which is critical for input normalization.
    """
    def __init__(self, in_channels, out_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = HSwish()

        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        _, _, h, w = x.shape

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return identity * a_h * a_w

# ==========================================
# Preprocessing Helpers (Ported from torch_pipeline_axial)
# ==========================================

def ensure_ri_channels(channels):
    """
    Normalize channels to (B, 2, H, W) float tensor (real, imag).
    Accepts complex (B, H, W) or real (B, 2, H, W).
    """
    if channels.dim() == 5 and channels.size(2) == 1:
        channels = channels.squeeze(2)
    if channels.is_complex():
        real = channels.real
        imag = channels.imag
        return torch.stack([real, imag], dim=1).float()

    if channels.dim() == 4 and channels.size(1) == 2:
        return channels.float()

    if channels.dim() == 4 and channels.size(1) == 1:
        zeros = torch.zeros_like(channels)
        return torch.cat([channels, zeros], dim=1).float()

    if channels.dim() == 3:
        zeros = torch.zeros_like(channels)
        return torch.stack([channels, zeros], dim=1).float()

    raise ValueError(f"Unsupported channel shape: {tuple(channels.shape)}")

def channels_to_patches(channels_ri, patch_size=16):
    """
    Convert (B, 2, Ant, Sub) into patches ordered by [Antenna, Component, Freq].
    Input: (B, 2, 32, 32) real/imag.
    Output: (B, 128, 16) flattened patches.
    """
    batch_size, channels, n_ant, n_sub = channels_ri.shape
    
    # 1. Permute to [B, Ant, 2, Sub]
    x = channels_ri.permute(0, 2, 1, 3) # [B, 32, 2, 32]
    
    # 2. Unfold subcarriers into patches
    # Sub (32) -> 2 patches of size 16
    n_patches_per_sub = n_sub // patch_size
    x = x.reshape(batch_size, n_ant, 2, n_patches_per_sub, patch_size)
    
    # 3. Flatten dimensions 2 and 3 (Component and FreqPatch)
    x = x.reshape(batch_size, n_ant, 2 * n_patches_per_sub, patch_size)
    
    # 4. Flatten all patches
    # [B, 128, 16]
    return x.reshape(batch_size, -1, patch_size)

def mask_patches(patches, mask_ratio=0.15, gen_raw=False):
    """
    Apply MCM-style masking to patches.
    Returns input_ids, masked_tokens, masked_pos.
    """
    batch_size, n_patches, patch_size = patches.shape
    # Define CLS and Mask token values (fixed for now, model learns embedding replacement)
    cls_token_val = 0.2
    mask_token_val = 0.1
    
    cls_token = torch.full(
        (batch_size, 1, patch_size), cls_token_val, device=patches.device, dtype=patches.dtype
    )
    input_ids = torch.cat([cls_token, patches], dim=1)

    real_tokens = n_patches // 2
    n_masks_half = int(mask_ratio * real_tokens)
    if n_masks_half < 1:
         # Fallback if too small
         n_masks_half = 1
    
    mask_vec = torch.full((patch_size,), mask_token_val, device=patches.device, dtype=patches.dtype)
    n_masks = n_masks_half * 2

    # Vectorized Random Selection ensuring Real+Imag pairs
    rand = torch.rand(batch_size, real_tokens, device=patches.device) 
    selected_indices = rand.topk(n_masks_half, dim=1).indices # [B, N_Masks_Half]
    
    # Map to physical indices (Ant * 4 + SubFreq)
    pos_real = (selected_indices // 2) * 4 + (selected_indices % 2)
    pos_imag = pos_real + 2
    
    masked_pos = torch.cat([pos_real, pos_imag], dim=1) + 1  # shift for CLS
    
    # Gather ground truth
    masked_tokens = torch.gather(
        input_ids, 1, masked_pos.unsqueeze(-1).expand(-1, -1, patch_size)
    ).detach()

    # Apply formatting
    if not gen_raw:
        # 80% mask, 10% random, 10% original (BERT style)
        rand_mask = torch.rand(batch_size, n_masks, device=patches.device)
        mask_mask = (rand_mask < 0.8)
        random_mask = (rand_mask >= 0.8) & (rand_mask < 0.9)

        batch_idx = torch.arange(batch_size, device=patches.device)[:, None].expand_as(masked_pos)
        
        # Apply MASK token
        if mask_mask.any():
            input_ids[batch_idx[mask_mask], masked_pos[mask_mask]] = mask_vec
            
        # Apply Random noise
        if random_mask.any():
            random_vals = torch.rand(
                batch_size, n_masks, patch_size, device=patches.device, dtype=patches.dtype
            )
            input_ids[batch_idx[random_mask], masked_pos[random_mask]] = random_vals[random_mask]

    return input_ids, masked_tokens, masked_pos

ELEMENT_LENGTH = 16
D_MODEL = 64
MAX_LEN = 129
N_LAYERS = 12
N_HEADS = 12
D_FF = D_MODEL * 4
D_K = D_MODEL // N_HEADS
D_V = D_MODEL // N_HEADS
DROPOUT = 0.1

class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class Embedding(nn.Module):
    def __init__(self, element_length, d_model, max_len):
        super().__init__()
        self.element_length = element_length
        self.d_model = d_model
        self.proj = nn.Linear(element_length, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = LayerNormalization(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x[:, :, 0])
        tok_emb = self.proj(x.float())  
        embedding = tok_emb + self.pos_embed(pos)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(D_K)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(D_MODEL, D_K * N_HEADS)
        self.W_K = nn.Linear(D_MODEL, D_K * N_HEADS)
        self.W_V = nn.Linear(D_MODEL, D_V * N_HEADS)
        self.linear = nn.Linear(N_HEADS * D_V, D_MODEL)
        self.norm = LayerNormalization(D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        
    def forward(self, Q, K, V, physics_bias=None, lambda_scalar=0.0):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, N_HEADS, D_V).transpose(1, 2)

        # Scaled Dot Product
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(D_K)
        
        # Inject Physics Bias to SCORES
        if physics_bias is not None:
             # scores: [B, H, Seq, Seq]
             # bias: [1, 1, Seq, Seq]
             scores = scores + (lambda_scalar * physics_bias)

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_s)
        
        output = context.transpose(1, 2).contiguous().view(batch_size, -1, N_HEADS * D_V)
        output = self.linear(output)
        return residual + self.dropout(output), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = LayerNormalization(D_MODEL)

    def forward(self, x):
        output = self.fc2(self.dropout(F.relu(self.fc1(x))))
        return x + self.dropout(output)

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.norm = LayerNormalization(D_MODEL)

    def forward(self, enc_inputs, physics_bias=None, lambda_scalar=0.0):
        attn_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, physics_bias, lambda_scalar)
        attn_outputs = self.norm(attn_outputs)
        enc_outputs = self.pos_ffn(attn_outputs)
        return enc_outputs, attn

class lwm_physics(torch.nn.Module):
    def __init__(self, element_length=16, d_model=64, max_len=129, n_layers=12, 
                 lambda_init=0.3, antennas=32, freq_groups=4):
        super().__init__()
        self.embedding = Embedding(element_length, d_model, max_len)
        
        # Physics Bias
        # Register as buffer to move to device automatically
        bias = compute_physics_bias(n_antennas=antennas, n_freq_groups=freq_groups)
        self.register_buffer("physics_bias", bias)
        self.lambda_scalar = nn.Parameter(torch.tensor(lambda_init)) # Learnable weight for physics
        
        # Coordinate Attention for Preprocessing
        self.coordatt = CoordAtt(in_channels=2, out_channels=2, reduction=32)
        
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, d_model)
        self.norm = LayerNormalization(d_model)

        embed_weight = self.embedding.proj.weight
        d_model, n_dim = embed_weight.size()
        self.decoder = nn.Linear(d_model, n_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(n_dim))

    @classmethod
    def from_pretrained(cls, ckpt_name='model_weights.pth', device='cuda', use_auth_token=None):
        model = cls().to(device)

        ckpt_path = ckpt_name
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Model loaded successfully from {ckpt_path} to {device}")

        return model

    def forward(self, input_ids, masked_pos=None):
        # Handle raw channels input (if masked_pos is None, we assume input_ids is channels)
        masked_tokens = None
        if masked_pos is None:
            channels = input_ids
            # Preprocess: Normalize -> CoordAtt -> Patch -> Mask
            channels_ri = ensure_ri_channels(channels)
            channels_ri = self.coordatt(channels_ri) # Apply Coordinate Attention (includes BatchNorm)
            patches = channels_to_patches(channels_ri, patch_size=self.embedding.element_length)
            input_ids, masked_tokens, masked_pos = mask_patches(patches, mask_ratio=0.15)

        output = self.embedding(input_ids)
        for layer in self.layers:
            output, _ = layer(output, self.physics_bias, self.lambda_scalar)

        masked_pos = masked_pos.long()[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(F.relu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        return logits_lm, masked_tokens, output

if __name__ == "__main__":
    # Test
    model = lwm_physics()
    x = torch.randint(0, 100, (2, 129, 16)).float()
    logits, _ = model(x, torch.zeros(2, 5).long())
    print(logits.shape)
