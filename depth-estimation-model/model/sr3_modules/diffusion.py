import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import os
import utils
from PIL import Image
import torchvision.transforms as transforms

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

# gaussian diffusion trainer class

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=1,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional

        if schedule_opt is not None:
            pass

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(device)
        elif self.loss_type == 'mae':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'huber':
            self.loss_func = nn.SmoothL1Loss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_sample(self, x_start, t, noise=None):
        """Add noise to x_start (depth map) at timestep t"""
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        # Get the appropriate alpha values for the batch
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.posterior_mean_coef1[t].view(-1, 1, 1, 1) * x_start +
            self.posterior_mean_coef2[t].view(-1, 1, 1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].view(-1, 1, 1, 1)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t].view(-1, 1, 1, 1)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, condition):
        """CORRECTED: Calculate model mean and variance for p(x_{t-1}|x_t)"""
        
        # CRITICAL: Concatenate condition (NIR) with noisy depth for model input
        # condition: [B, 1, H, W] (NIR image)
        # x: [B, 1, H, W] (noisy depth)
        # model_input: [B, 2, H, W] (concatenated)
        
        model_input = torch.cat([condition, x], dim=1)
        print(f"Model input shape: {model_input.shape} (condition + noisy depth)")
        
        # Predict noise using the denoising network
        predicted_noise = self.denoise_fn(model_input, t)
        
        # CRITICAL: Ensure predicted noise has same shape as target
        if predicted_noise.shape[1] != 1:
            # If model outputs multiple channels, take only the first one
            predicted_noise = predicted_noise[:, :1, :, :]
        
        if predicted_noise.shape != x.shape:
            # Resize if needed
            predicted_noise = F.interpolate(predicted_noise, size=x.shape[-2:], 
                                        mode='bilinear', align_corners=False)
        
        # Get x_0 prediction from noise prediction
        x_recon = self.predict_start_from_noise(x, t, predicted_noise)
        
        # Clamp to reasonable range
        x_recon = torch.clamp(x_recon, -2, 2)  # Adjust range as needed
        
        # Get posterior distribution
        model_mean, model_variance, model_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        
        return model_mean, model_variance, model_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, t, condition):
        """CORRECTED: Sample x_{t-1} from x_t with proper conditioning"""
        b = x.shape[0]
        device = x.device
        
        # Get model predictions
        model_mean, model_variance, model_log_variance, x_recon = self.p_mean_variance(
            x=x, t=t, condition=condition)
        
        # No noise when t == 0
        if t[0] > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        
        # Sample from the distribution
        sample = model_mean + torch.sqrt(model_variance) * noise
        
        return sample

    @torch.no_grad()
    def p_sample_loop(self, condition, continous=False):
        """Generate depth map from NIR image through reverse diffusion"""
        print(f"Condition (NIR) shape: {condition.shape}")
        print(f"Condition range: [{condition.min():.4f}, {condition.max():.4f}]")
        
        device = self.betas.device
        b = condition.shape[0]
        
        # CRITICAL: Output should have same spatial dimensions as condition
        # but only 1 channel (depth is single-channel)
        shape = (b, 1, condition.shape[2], condition.shape[3]) 
        print(f"Target depth shape: {shape}")
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        print(f"Starting noise shape: {img.shape}")
        print(f"Starting noise range: [{img.min():.4f}, {img.max():.4f}]")
        
        imgs = []
        
        # Reverse diffusion process
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            
            # CRITICAL: Pass condition to p_sample
            img = self.p_sample(img, t, condition)
            
            if continous and i % 10 == 0:
                imgs.append(img.clone())

        print(f"Final depth shape: {img.shape}")
        print(f"Final depth range: [{img.min():.4f}, {img.max():.4f}]")
        
        if continous:
            return imgs
        return img

    def p_losses(self, x_in, noise=None):
        """
        Calculate diffusion loss for training
        x_in: dict with 'HR' (NIR image) and 'SR' (depth ground truth)
        """
        if not isinstance(x_in, dict) or 'HR' not in x_in or 'SR' not in x_in:
            raise ValueError("x_in must be a dict with 'HR' and 'SR' keys")
        
        # Extract NIR image and depth ground truth
        nir_image = x_in['HR']  # Condition (NIR image)
        depth_gt = x_in['SR']    # Target (depth map)
        
        # Ensure proper shapes (B, C, H, W)
        if nir_image.dim() == 3:
            nir_image = nir_image.unsqueeze(1)
        if depth_gt.dim() == 3:
            depth_gt = depth_gt.unsqueeze(1)
        
        # Fix channel mismatches if needed
        if nir_image.shape[1] != self.channels:
            nir_image = nir_image.repeat(1, self.channels, 1, 1) if nir_image.shape[1] < self.channels else nir_image[:, :self.channels]
        if depth_gt.shape[1] != self.channels:
            depth_gt = depth_gt.repeat(1, self.channels, 1, 1) if depth_gt.shape[1] < self.channels else depth_gt[:, :self.channels]
        
        # Fix spatial dimensions: make SR match HR
        if depth_gt.shape[-2:] != nir_image.shape[-2:]:
            depth_gt = F.interpolate(depth_gt, size=nir_image.shape[-2:], mode='bilinear', align_corners=False)
        
        b, c, h, w = depth_gt.shape
        device = depth_gt.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # Generate noise
        noise = default(noise, lambda: torch.randn_like(depth_gt))
        
        # Add noise to the depth map
        noisy_depth = self.q_sample(x_start=depth_gt, t=t, noise=noise)
        
        # Concatenate NIR image with noisy depth as condition
        model_input = torch.cat([nir_image, noisy_depth], dim=1)
        
        # Predict the noise
        predicted_noise = self.denoise_fn(model_input, t)
        
        # Ensure predicted noise has same shape as target
        if predicted_noise.shape != noise.shape:
            predicted_noise = F.interpolate(predicted_noise, size=noise.shape[-2:], mode='bilinear', align_corners=False)
        
        # Store for visualization
        self.predicted_depth = self.predict_start_from_noise(noisy_depth, t, predicted_noise)
        self.processed_HR = nir_image
        
        # Calculate loss between predicted and actual noise
        loss = self.loss_func(predicted_noise, noise)
        
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)