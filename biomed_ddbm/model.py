import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaGN(nn.Module):
    """
    Adaptive Group Normalization.
    Modulates group norm parameters based on the input embedding (c_img).
    """
    def __init__(self, num_channels, emb_dim, num_groups=32):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        self.proj = nn.Linear(emb_dim, num_channels * 2)

    def forward(self, x, emb):
        # x: [B, C, H, W]
        # emb: [B, emb_dim]
        
        # Standard Group Norm
        x = self.group_norm(x)
        
        # Predict scale and shift
        stats = self.proj(emb) # [B, 2*C]
        scale, shift = stats.chunk(2, dim=1) # [B, C], [B, C]
        
        # Reshape for broadcast
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        
        # Modulate
        x = x * (1 + scale) + shift
        return x

class SemanticCrossAttention(nn.Module):
    """
    Cross-Attention layer.
    Query: Image Features
    Key/Value: Text Embeddings (c_text)
    """
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(0.0) # Can add dropout if needed
        )

    def forward(self, x, context):
        # x: [B, C, H, W] -> flatten -> [B, HW, C]
        # context: [B, L, context_dim] (or [B, 1, context_dim])
        
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).permute(0, 2, 1) # [B, HW, C]

        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)

        # Split heads
        q = q.view(b, -1, self.heads, q.shape[-1] // self.heads).permute(0, 2, 1, 3)
        k = k.view(b, -1, self.heads, k.shape[-1] // self.heads).permute(0, 2, 1, 3)
        v = v.view(b, -1, self.heads, v.shape[-1] // self.heads).permute(0, 2, 1, 3)

        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)

        # Merge heads
        out = out.permute(0, 2, 1, 3).reshape(b, -1, out.shape[-1] * self.heads)
        
        # Output project
        out = self.to_out(out)
        
        # Reshape back to image
        out = out.permute(0, 2, 1).view(b, c, h, w)
        
        # Residual connection usually handled outside or assume additive here?
        # Typically x + attention(x, c). Let's return just diff.
        return out + x

class BioMedScaling(nn.Module):
    """
    Predicts c_in, c_out, c_skip based on time and condition.
    Condition is typically concatenated c_img + c_text (mean).
    """
    def __init__(self, emb_dim, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmbed(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(emb_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.out_head = nn.Sequential(
            nn.Linear(time_dim * 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, 3) # c_in, c_out, c_skip
        )
        
    def forward(self, t, cond):
        # t: [B]
        # cond: [B, emb_dim]
        t_emb = self.time_mlp(t)
        c_emb = self.cond_mlp(cond)
        
        joint = torch.cat([t_emb, c_emb], dim=-1)
        scaling = self.out_head(joint)
        
        # Apply constraints? Usually softplus or sigmoid depending on definition.
        # DDBM often uses these as freely learned weights or constrained. 
        # For stability, we might leave them raw but typically they are positive.
        # Let's use Sigmoid to keep them bound [0, 1] or generic.
        # DDBM paper: c_skip ~ 1, c_out ~ 0 at t=0 etc.
        # Let's return raw for now, logic dictates usage.
        return scaling

class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, emb_dim=None, dropout=0.0):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.SiLU(),
        )
        self.adagn = AdaGN(dim_out, emb_dim) if emb_dim else nn.Identity()
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim_out, dim_out, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, emb):
        h = self.block1(x)
        if isinstance(self.adagn, AdaGN):
            h = self.adagn(h, emb)
        h = self.block2(h)
        return h + self.res_conv(x)


        
class BioMedDDBM(nn.Module):
    def __init__(
        self,
        dim=64,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        in_channels=None, # NEW ARGUMENT
        resnet_block_groups=8,
        learned_sinusoidal_cond=False, # Not using yet
        random_fourier_features=False, # Not using yet
        learned_sinusoidal_dim=16,
        biomed_img_dim=512, # BiomedCLIP vision output dim
        biomed_text_dim=512 # BiomedCLIP text output dim
    ):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels if in_channels is not None else channels
        
        input_dim = self.in_channels
        init_dim = init_dim if init_dim is not None else dim
        self.init_conv = nn.Conv2d(input_dim, init_dim, 7, padding=3)
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        
        # Time Embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmbed(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Joint Embedding Dimension for AdaGN (Time + Image)
        # We project Time + Image -> AdaGN dim
        self.adagn_dim = time_dim + biomed_img_dim
        
        # Layers
        curr_dim = init_dim
        
        # Downsample
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            
            self.downs.append(nn.ModuleList([
                ResBlock(dim_in, dim_in, emb_dim=self.adagn_dim),
                ResBlock(dim_in, dim_in, emb_dim=self.adagn_dim),
                SemanticCrossAttention(dim_in, biomed_text_dim), # Cross Attention with Text
                nn.Conv2d(dim_in, dim_out, 4, 2, 1) if not is_last else nn.Conv2d(dim_in, dim_out, 3, 1, 1)
            ]))
            
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock(mid_dim, mid_dim, emb_dim=self.adagn_dim)
        self.mid_attn = SemanticCrossAttention(mid_dim, biomed_text_dim)
        self.mid_block2 = ResBlock(mid_dim, mid_dim, emb_dim=self.adagn_dim)
        
        # Upsample
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (num_resolutions - 1)
            
            self.ups.append(nn.ModuleList([
                ResBlock(dim_out + dim_in, dim_out, emb_dim=self.adagn_dim),
                ResBlock(dim_out, dim_out, emb_dim=self.adagn_dim),
                SemanticCrossAttention(dim_out, biomed_text_dim),
                nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1) if not is_last else nn.Conv2d(dim_out, dim_in, 3, 1, 1)
            ]))
            
        self.final_res_block = ResBlock(init_dim * 2, dim, emb_dim=self.adagn_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)
        
        # Scaling Network
        self.scaling_net = BioMedScaling(emb_dim=biomed_img_dim + biomed_text_dim, time_dim=time_dim)

    def forward(self, x, t, c_img, c_text):
        """
        x: [B, C, H, W] Input Image (Noisy)
        t: [B] Time steps
        c_img: [B, 512] BioMedCLIP Image Embedding (of original image or reference)
        c_text: [B, 1, 512] BioMedCLIP Text Embedding
        """
        # 1. Scaling Factors
        # c_cond = cat(c_img, mean(c_text))
        c_text_mean = c_text.mean(dim=1) if c_text.dim() == 3 else c_text
        scaling_cond = torch.cat([c_img, c_text_mean], dim=-1) # [B, 1024]
        
        c_factors = self.scaling_net(t, scaling_cond)
        c_in, c_out, c_skip = c_factors.chunk(3, dim=-1)
        
        # Expand for broadcast
        c_in = c_in.view(-1, 1, 1, 1)
        c_out = c_out.view(-1, 1, 1, 1)
        c_skip = c_skip.view(-1, 1, 1, 1)
        
        # 2. Scale Input
        # Note: In standard diffusion, prediction is usually epsilon or x0.
        # Implementing "Preconditioning" formulation if desired, or standard epsilon prediction.
        # User requested "Scaling Function c_in, c_out, c_skip".
        # This implies: D(x, t) = c_skip * x + c_out * F(c_in * x, t)
        
        # 2. Scale Input for Model
        # If in_channels > channels, assumption is [Noisy(3) | Condition(4)]
        # We only scale the Noisy part according to Karras formulation.
        if self.in_channels > self.channels:
            x_noisy = x[:, :self.channels, :, :] # First 3 channels
            x_cond = x[:, self.channels:, :, :] # Remaining channels (Mask, Context)
            
            # Apply Preconditioning to Noisy Input
            x_noisy_in = c_in * x_noisy
            
            # Concatenate back
            x_in = torch.cat([x_noisy_in, x_cond], dim=1)
        else:
            x_noisy = x
            x_in = c_in * x
            
        # 3. U-Net Forward
        # Embeddings for AdaGN
        t_emb = self.time_mlp(t)
        adagn_emb = torch.cat([t_emb, c_img], dim=-1)
        
        h = self.init_conv(x_in)
        r = h.clone()
        
        h_s = []
        
        # Down
        for block1, block2, attn, downsample in self.downs:
            h = block1(h, adagn_emb)
            h = block2(h, adagn_emb)
            h = attn(h, c_text)
            h_s.append(h)
            h = downsample(h)
            
        # Mid
        h = self.mid_block1(h, adagn_emb)
        h = self.mid_attn(h, c_text)
        h = self.mid_block2(h, adagn_emb)
        
        # Up
        for block1, block2, attn, upsample in self.ups:
            h_skip = h_s.pop()
            h = torch.cat((h, h_skip), dim=1)
            h = block1(h, adagn_emb)
            h = block2(h, adagn_emb)
            h = attn(h, c_text)
            h = upsample(h)
            
        h = torch.cat((h, r), dim=1)
        h = self.final_res_block(h, adagn_emb)
        f_x = self.final_conv(h)
        
        # 4. Final Combination
        # Apply Skip Connection to the ORIGINAL Noisy Input (x_noisy)
        out = c_skip * x_noisy + c_out * f_x
        
        return out
