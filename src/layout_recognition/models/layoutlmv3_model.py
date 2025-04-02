#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LayoutLMv3-based model for Renaissance layout recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3FeatureExtractor
from transformers import LayoutLMv3Processor, LayoutLMv3Config
import numpy as np
from PIL import Image
import os
from typing import Dict, List, Optional, Union, Tuple

class LayoutLMv3SegmentationHead(nn.Module):
    """
    Segmentation head for LayoutLMv3 model.
    """
    def __init__(self, in_channels, out_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

class RenaissanceLayoutLMv3Model(nn.Module):
    """
    LayoutLMv3-based model for Renaissance layout recognition.
    """
    def __init__(self, model_name_or_path="microsoft/layoutlmv3-base", num_classes=2):
        """
        Initialize LayoutLMv3-based model.
        
        Args:
            model_name_or_path (str): Name or path of pretrained LayoutLMv3 model
            num_classes (int): Number of output classes (default: 2 for text/non-text)
        """
        super().__init__()
        
        # Load LayoutLMv3 configuration
        self.config = LayoutLMv3Config.from_pretrained(model_name_or_path)
        
        # Initialize feature extractor and processor
        self.feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained(model_name_or_path, apply_ocr=False)
        self.processor = LayoutLMv3Processor.from_pretrained(model_name_or_path, apply_ocr=False)
        
        # Create LayoutLMv3 model
        self.layoutlmv3 = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Extract hidden dimension
        hidden_size = self.config.hidden_size
        
        # Create segmentation head
        self.seg_head = LayoutLMv3SegmentationHead(hidden_size, num_classes)
        
        # Input size for the model
        self.input_size = (224, 224)
        
    def _process_image(self, image):
        """
        Process image for LayoutLMv3.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Dict: Processed inputs for LayoutLMv3
        """
        # Ensure image is a PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Resize image to expected input size
        image = image.resize(self.input_size)
        
        # Process image with LayoutLMv3 processor
        # Since we set apply_ocr=False, we need to provide dummy words and boxes
        dummy_words = [[""]]
        dummy_boxes = [[[0, 0, 1, 1]]]  # Minimal box to avoid errors
        
        encoding = self.processor(
            image,
            text=dummy_words,
            boxes=dummy_boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        
        return encoding
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Segmentation logits
        """
        batch_size, channels, height, width = x.shape
        
        # Process each image in the batch
        processed_inputs = []
        for i in range(batch_size):
            # Convert tensor to PIL Image
            img = x[i].detach().cpu().permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            
            # Process image
            encoding = self._process_image(pil_img)
            processed_inputs.append(encoding)
        
        # Combine batch inputs
        batch_inputs = {
            'pixel_values': torch.cat([inputs['pixel_values'] for inputs in processed_inputs], dim=0).to(x.device),
            'attention_mask': torch.cat([inputs['attention_mask'] for inputs in processed_inputs], dim=0).to(x.device),
        }
        
        # Forward through LayoutLMv3
        outputs = self.layoutlmv3(**batch_inputs, output_hidden_states=True)
        
        # Get hidden states from the last layer
        hidden_states = outputs.hidden_states[-1]
        
        # Reshape hidden states for segmentation head
        # We need to reshape to [batch_size, hidden_size, height/patch_size, width/patch_size]
        patch_size = 16  # LayoutLMv3 uses patch size of 16
        feature_height = height // patch_size
        feature_width = width // patch_size
        
        # Take the first token (CLS) embedding for each image
        cls_embedding = hidden_states[:, 0]
        
        # Reshape to 2D feature map
        features = cls_embedding.view(batch_size, -1, 1, 1)
        features = features.expand(-1, -1, feature_height, feature_width)
        
        # Apply segmentation head
        logits = self.seg_head(features)
        
        # Upscale to original image size
        logits = F.interpolate(logits, size=(height, width), mode='bilinear', align_corners=False)
        
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

def create_layoutlmv3_model(pretrained_path=None, device='cpu'):
    """
    Create and load a LayoutLMv3-based model.
    
    Args:
        pretrained_path (str): Path to pretrained weights
        device (torch.device): Device to load model on
        
    Returns:
        RenaissanceLayoutLMv3Model: Loaded model
    """
    model = RenaissanceLayoutLMv3Model(model_name_or_path="microsoft/layoutlmv3-base", num_classes=2)
    
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
