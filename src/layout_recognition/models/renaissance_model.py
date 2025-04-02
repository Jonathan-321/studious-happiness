#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Renaissance-specific layout recognition model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np

class RenaissanceLayoutModel(nn.Module):
    """
    A simplified model for Renaissance layout recognition.
    Focuses on binary segmentation of text vs non-text regions.
    """
    
    def __init__(self, input_channels=3, backbone='resnet34'):
        # Input size for the model
        self.input_size = (512, 512)  # ResNet can handle larger input sizes
        """
        Initialize Renaissance Layout model.
        
        Args:
            input_channels (int): Number of input channels
            backbone (str): Backbone architecture
        """
        super(RenaissanceLayoutModel, self).__init__()
        
        # Initialize backbone
        if backbone == 'resnet34':
            try:
                self.backbone = models.resnet34(weights='IMAGENET1K_V1')
            except TypeError:
                # For older versions of torchvision
                self.backbone = models.resnet34(pretrained=True)
                
            if input_channels != 3:
                self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Remove the final FC layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            feature_dim = 512
        elif backbone == 'resnet18':
            try:
                self.backbone = models.resnet18(weights='IMAGENET1K_V1')
            except TypeError:
                # For older versions of torchvision
                self.backbone = models.resnet18(pretrained=True)
                
            if input_channels != 3:
                self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Segmentation head - binary classification (text vs non-text)
        self.seg_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 2, kernel_size=2, stride=2)  # 2 classes: text vs non-text
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Segmentation logits
        """
        # Extract features
        features = self.backbone(x)
        
        # Apply segmentation head
        logits = self.seg_head(features)
        
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
        
        # Resize to model input size
        image = image.resize(self.input_size)
        
        # Convert to numpy array and normalize
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)
        
        return image_tensor, image

def create_renaissance_model(pretrained_path=None, device='cpu'):
    """
    Create and load a Renaissance layout model.
    
    Args:
        pretrained_path (str): Path to pretrained weights
        device (torch.device): Device to load model on
        
    Returns:
        RenaissanceLayoutModel: Loaded model
    """
    model = RenaissanceLayoutModel(input_channels=3, backbone='resnet34')
    
    if pretrained_path:
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
    
    return model
