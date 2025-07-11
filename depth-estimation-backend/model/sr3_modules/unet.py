import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import numpy as np

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        # Handle both tensor and float inputs
        if isinstance(noise_level, (int, float)):
            noise_level = torch.tensor([noise_level], device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure noise_level is 1D
        if noise_level.dim() == 0:
            noise_level = noise_level.unsqueeze(0)
        
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return self.act(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)

# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout=0):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, dim), dim)
        self.block = nn.Sequential(
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level) if exists(noise_level_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out, dropout=dropout)

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.noise_func) and exists(time_emb):
            h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(min(norm_groups, in_channel), in_channel)

        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key) / math.sqrt(channel)
        attn = torch.softmax(attn.view(batch, n_head, height, width, -1), dim=-1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value)
        out = self.out(out.view(batch, channel, height, width))

        return out + input

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=min(norm_groups, dim_out), dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=min(norm_groups, dim_out))

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=2,  # Changed: 1 NIR + 1 noisy depth = 2 channels
        out_channel=1,  # Single-channel noise prediction
        inner_channel=64,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        
        # Initial convolution
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        
        # Downsampling
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
                
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
                
        self.downs = nn.ModuleList(downs)

        # Middle blocks
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups, dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups, dropout=dropout, with_attn=False)
        ])

        # Upsampling
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
                
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        # Final convolution
        self.final_conv = Block(pre_channel, out_channel)

    def forward(self, x, time):
        """
        x: [batch, 2, H, W] - concatenated NIR image and noisy depth
        time: [batch] - timesteps
        """
        # Time embedding
        t = self.noise_level_mlp(time.float()) if exists(self.noise_level_mlp) else None

        # Store features for skip connections
        feats = []
        
        # Downsampling
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        # Middle
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        # Upsampling with skip connections
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = torch.cat((x, feats.pop()), dim=1)
                x = layer(x, t)
            else:
                x = layer(x)

        # Final projection to noise prediction
        return self.final_conv(x)

# Additional helper class for compatibility
class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fcn(x)