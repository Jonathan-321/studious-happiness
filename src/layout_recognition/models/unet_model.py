#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
U-Net model for Renaissance layout recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import os
from PIL import Image
import numpy as np

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetLayoutModel(nn.Module):
    """
    U-Net model for Renaissance layout recognition.
    """
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        """
        Initialize U-Net model.
        
        Args:
            n_channels (int): Number of input channels
            n_classes (int): Number of output classes
            bilinear (bool): Whether to use bilinear upsampling
        """
        super(UNetLayoutModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Input size for the model
        self.input_size = (512, 512)

        # Initialize encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Initialize decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Segmentation logits
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for inference.
        
        Args:
            image_path (str): Path to image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Store original size
        original_size = image.size
        
        # Resize to model input size
        image = image.resize(self.input_size)
        
        # Convert to numpy array and normalize
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)
        
        return image_tensor, image

def create_unet_model(pretrained_path=None, device='cpu'):
    """
    Create and load a U-Net model.
    
    Args:
        pretrained_path (str): Path to pretrained weights
        device (torch.device): Device to load model on
        
    Returns:
        UNetLayoutModel: Loaded model
    """
    model = UNetLayoutModel(n_channels=3, n_classes=2, bilinear=False)
    
    if pretrained_path and os.path.exists(pretrained_path):
        # Load weights
        checkpoint = torch.load(pretrained_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
            
        # Try to load weights, handling potential mismatches
        try:
            model.load_state_dict(model_state_dict)
        except Exception as e:
            print(f"Warning: Could not load weights directly: {e}")
            print("Attempting to load weights with shape adaptation...")
            
            # Get model state dict
            model_dict = model.state_dict()
            
            # Filter out incompatible keys
            pretrained_dict = {k: v for k, v in model_state_dict.items() 
                              if k in model_dict and model_dict[k].shape == v.shape}
            
            # Update model dict with filtered pretrained dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")
    
    model = model.to(device)
    model.eval()
    return model
