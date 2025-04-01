#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Diffusion model for synthetic Renaissance text generation.
This module implements a diffusion-based model for generating synthetic Renaissance-style text images.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class TimeEmbedding(nn.Module):
    """Time embedding for diffusion model."""
    
    def __init__(self, dim):
        """
        Initialize time embedding.
        
        Args:
            dim (int): Embedding dimension
        """
        super().__init__()
        self.dim = dim
        
        half_dim = dim // 2
        self.emb = nn.Sequential(
            nn.Linear(half_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        """
        Forward pass.
        
        Args:
            t (torch.Tensor): Time steps
            
        Returns:
            torch.Tensor: Time embeddings
        """
        # Create sinusoidal position embeddings
        device = t.device
        half_dim = self.dim // 2
        
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        # Transform embeddings
        emb = self.emb(emb)
        
        return emb


class ResidualBlock(nn.Module):
    """Residual block for UNet."""
    
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        """
        Initialize residual block.
        
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            time_channels (int): Time embedding channels
            dropout (float): Dropout probability
        """
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection if channel dimensions don't match
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            time_emb (torch.Tensor): Time embedding
            
        Returns:
            torch.Tensor: Output tensor
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip_connection(x)


class AttentionBlock(nn.Module):
    """Self-attention block for UNet."""
    
    def __init__(self, channels, num_heads=4):
        """
        Initialize attention block.
        
        Args:
            channels (int): Input channels
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        b, c, h, w = x.shape
        
        # Normalize input
        x_norm = self.norm(x)
        
        # Get query, key, value
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Reshape for multi-head attention
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 2, 3)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)
        
        # Compute attention
        scale = (c // self.num_heads) ** -0.5
        attn = torch.matmul(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        
        # Project output
        out = self.proj(out)
        
        return out + x


class DownBlock(nn.Module):
    """Downsampling block for UNet."""
    
    def __init__(self, in_channels, out_channels, time_channels, has_attn=False):
        """
        Initialize downsampling block.
        
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            time_channels (int): Time embedding channels
            has_attn (bool): Whether to include attention block
        """
        super().__init__()
        
        self.res1 = ResidualBlock(in_channels, out_channels, time_channels)
        self.res2 = ResidualBlock(out_channels, out_channels, time_channels)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        self.has_attn = has_attn
        if has_attn:
            self.attn = AttentionBlock(out_channels)
    
    def forward(self, x, time_emb):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            time_emb (torch.Tensor): Time embedding
            
        Returns:
            tuple: (Output tensor, Skip connection tensor)
        """
        x = self.res1(x, time_emb)
        x = self.res2(x, time_emb)
        
        if self.has_attn:
            x = self.attn(x)
        
        skip = x
        x = self.downsample(x)
        
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block for UNet."""
    
    def __init__(self, in_channels, out_channels, time_channels, has_attn=False):
        """
        Initialize upsampling block.
        
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            time_channels (int): Time embedding channels
            has_attn (bool): Whether to include attention block
        """
        super().__init__()
        
        self.res1 = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        self.res2 = ResidualBlock(out_channels + out_channels, out_channels, time_channels)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        
        self.has_attn = has_attn
        if has_attn:
            self.attn = AttentionBlock(out_channels)
    
    def forward(self, x, skip1, skip2, time_emb):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            skip1 (torch.Tensor): Skip connection from downsampling path
            skip2 (torch.Tensor): Skip connection from downsampling path
            time_emb (torch.Tensor): Time embedding
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = torch.cat([x, skip1], dim=1)
        x = self.res1(x, time_emb)
        
        x = torch.cat([x, skip2], dim=1)
        x = self.res2(x, time_emb)
        
        if self.has_attn:
            x = self.attn(x)
        
        x = self.upsample(x)
        
        return x


class UNet(nn.Module):
    """UNet architecture for diffusion model."""
    
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=256, base_channels=64, channel_mults=(1, 2, 4, 8)):
        """
        Initialize UNet.
        
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            time_emb_dim (int): Time embedding dimension
            base_channels (int): Base channel multiplier
            channel_mults (tuple): Channel multipliers for each resolution
        """
        super().__init__()
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling path
        self.downs = nn.ModuleList()
        
        in_ch = base_channels
        channels = []
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            # Add attention for deeper layers
            has_attn = i >= len(channel_mults) - 2
            
            self.downs.append(DownBlock(in_ch, out_ch, time_emb_dim, has_attn))
            in_ch = out_ch
            channels.append(out_ch)
        
        # Middle blocks
        self.mid_res1 = ResidualBlock(in_ch, in_ch, time_emb_dim)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_res2 = ResidualBlock(in_ch, in_ch, time_emb_dim)
        
        # Upsampling path
        self.ups = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            
            # Add attention for deeper layers
            has_attn = i >= len(channel_mults) - 2
            
            self.ups.append(UpBlock(in_ch, out_ch, time_emb_dim, has_attn))
            in_ch = out_ch
        
        # Final blocks
        self.final_res = ResidualBlock(in_ch, base_channels, time_emb_dim)
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, time):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            time (torch.Tensor): Time steps
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Time embedding
        time_emb = self.time_embedding(time)
        
        # Initial convolution
        h = self.init_conv(x)
        
        # Downsampling
        skips = []
        for down in self.downs:
            h, skip = down(h, time_emb)
            skips.append(skip)
        
        # Middle blocks
        h = self.mid_res1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, time_emb)
        
        # Upsampling
        for up in self.ups:
            skip2 = skips.pop()
            skip1 = skips.pop()
            h = up(h, skip1, skip2, time_emb)
        
        # Final blocks
        h = self.final_res(h, time_emb)
        h = self.final_conv(h)
        
        return h


class DiffusionModel:
    """Diffusion model for synthetic text generation."""
    
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, timesteps=1000, device='cuda'):
        """
        Initialize diffusion model.
        
        Args:
            model (nn.Module): UNet model
            beta_start (float): Start value for noise schedule
            beta_end (float): End value for noise schedule
            timesteps (int): Number of diffusion steps
            device (str): Device to use
        """
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Define noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0).
        
        Args:
            x_0 (torch.Tensor): Initial clean data
            t (torch.Tensor): Time steps
            noise (torch.Tensor, optional): Noise to add
            
        Returns:
            torch.Tensor: Noisy data at time t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_0, t, noise=None):
        """
        Training loss for diffusion model.
        
        Args:
            x_0 (torch.Tensor): Initial clean data
            t (torch.Tensor): Time steps
            noise (torch.Tensor, optional): Noise to add
            
        Returns:
            torch.Tensor: Loss
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Generate noisy sample
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Calculate loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def p_sample(self, x_t, t):
        """
        Sample from p(x_{t-1} | x_t).
        
        Args:
            x_t (torch.Tensor): Noisy data at time t
            t (torch.Tensor): Time step
            
        Returns:
            torch.Tensor: Sample from p(x_{t-1} | x_t)
        """
        with torch.no_grad():
            # Predict noise
            predicted_noise = self.model(x_t, t)
            
            # Calculate mean for posterior
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            beta_t = self.betas[t]
            
            alpha_t = alpha_t.reshape(-1, 1, 1, 1)
            alpha_cumprod_t = alpha_cumprod_t.reshape(-1, 1, 1, 1)
            beta_t = beta_t.reshape(-1, 1, 1, 1)
            
            # Compute mean of p(x_{t-1} | x_t)
            mean = (1 / torch.sqrt(alpha_t)) * (
                x_t - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            )
            
            # Add noise if t > 0, otherwise just return mean
            if t[0] > 0:
                noise = torch.randn_like(x_t)
                var_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
                return mean + torch.sqrt(var_t) * noise
            else:
                return mean
    
    def p_sample_loop(self, shape, return_intermediates=False):
        """
        Sample from the model by iterating through the diffusion process.
        
        Args:
            shape (tuple): Shape of the sample to generate
            return_intermediates (bool): Whether to return intermediate samples
            
        Returns:
            torch.Tensor or list: Generated sample or list of intermediate samples
        """
        device = self.device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        intermediates = [img] if return_intermediates else None
        
        # Iterate from t=T to t=0
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t)
            
            if return_intermediates:
                intermediates.append(img)
        
        if return_intermediates:
            return intermediates
        
        return img
    
    def train(self, dataloader, optimizer, epochs, scheduler=None, save_dir=None, save_freq=5):
        """
        Train the diffusion model.
        
        Args:
            dataloader (DataLoader): Data loader
            optimizer (Optimizer): Optimizer
            epochs (int): Number of epochs
            scheduler (lr_scheduler, optional): Learning rate scheduler
            save_dir (str, optional): Directory to save checkpoints
            save_freq (int): Frequency to save checkpoints
            
        Returns:
            list: Training losses
        """
        device = self.device
        self.model.to(device)
        self.model.train()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch in progress_bar:
                optimizer.zero_grad()
                
                # Get batch
                x = batch['image'].to(device)
                
                # Sample random timesteps
                t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()
                
                # Calculate loss
                loss = self.p_losses(x, t)
                
                # Update model
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                epoch_losses.append(loss.item())
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Calculate average loss
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
            
            # Save checkpoint
            if save_dir is not None and (epoch + 1) % save_freq == 0:
                os.makedirs(save_dir, exist_ok=True)
                checkpoint = {
                    'model': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        return losses
    
    def generate_samples(self, num_samples, img_size, batch_size=4):
        """
        Generate samples from the model.
        
        Args:
            num_samples (int): Number of samples to generate
            img_size (tuple): Image size (C, H, W)
            batch_size (int): Batch size for generation
            
        Returns:
            torch.Tensor: Generated samples
        """
        all_samples = []
        
        for i in range(0, num_samples, batch_size):
            batch_size_i = min(batch_size, num_samples - i)
            shape = (batch_size_i, *img_size)
            
            samples = self.p_sample_loop(shape)
            all_samples.append(samples)
        
        # Concatenate all samples
        samples = torch.cat(all_samples, dim=0)
        
        return samples
    
    def save(self, path):
        """
        Save model.
        
        Args:
            path (str): Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        """
        Load model.
        
        Args:
            path (str): Path to load model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
