#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LayoutLM implementation/adaptation for document layout analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMModel, LayoutLMConfig
from torchvision.ops import roi_align
from torchvision import models

class LayoutLMForSegmentation(nn.Module):
    """LayoutLM model adapted for layout segmentation."""
    
    def __init__(self, num_classes=9, pretrained_model_name="microsoft/layoutlm-base-uncased"):
        """
        Initialize LayoutLM for segmentation.
        
        Args:
            num_classes (int): Number of layout classes
            pretrained_model_name (str): Name of the pretrained model
        """
        super(LayoutLMForSegmentation, self).__init__()
        
        # Load pretrained LayoutLM
        self.layoutlm = LayoutLMModel.from_pretrained(pretrained_model_name)
        self.config = self.layoutlm.config
        
        # Feature projection
        self.feature_proj = nn.Linear(self.config.hidden_size, 256)
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def forward(self, input_ids, bbox, attention_mask=None, token_type_ids=None, image_size=(512, 512)):
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Token IDs
            bbox (torch.Tensor): Bounding box coordinates for each token
            attention_mask (torch.Tensor): Attention mask
            token_type_ids (torch.Tensor): Token type IDs
            image_size (tuple): Size of the input image (height, width)
            
        Returns:
            torch.Tensor: Segmentation logits
        """
        # Get LayoutLM embeddings
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get sequence output
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Project features
        features = self.feature_proj(sequence_output)  # [batch_size, seq_len, 256]
        
        # Create feature map from token features and their positions
        batch_size, seq_len, feat_dim = features.size()
        height, width = image_size
        
        # Initialize feature map
        feature_map = torch.zeros(batch_size, feat_dim, height, width, device=features.device)
        
        # Fill feature map based on token positions
        for b in range(batch_size):
            for i in range(1, seq_len):  # Skip [CLS] token
                if attention_mask is None or attention_mask[b, i] == 1:
                    # Get normalized bbox coordinates
                    x1, y1, x2, y2 = bbox[b, i].float()
                    
                    # Scale to feature map size
                    x1 = int(x1 * width / 1000)
                    y1 = int(y1 * height / 1000)
                    x2 = int(x2 * width / 1000)
                    y2 = int(y2 * height / 1000)
                    
                    # Ensure valid coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width - 1, x2), min(height - 1, y2)
                    
                    if x2 > x1 and y2 > y1:
                        # Add token features to the feature map
                        feature_map[b, :, y1:y2, x1:x2] = features[b, i].view(-1, 1, 1)
        
        # Apply segmentation head
        logits = self.seg_head(feature_map)
        
        return logits

class LayoutLMForRegionClassification(nn.Module):
    """LayoutLM model for region classification."""
    
    def __init__(self, num_classes=9, pretrained_model_name="microsoft/layoutlm-base-uncased"):
        """
        Initialize LayoutLM for region classification.
        
        Args:
            num_classes (int): Number of layout classes
            pretrained_model_name (str): Name of the pretrained model
        """
        super(LayoutLMForRegionClassification, self).__init__()
        
        # Load pretrained LayoutLM
        self.layoutlm = LayoutLMModel.from_pretrained(pretrained_model_name)
        
        # Classification head
        self.classifier = nn.Linear(self.layoutlm.config.hidden_size, num_classes)
    
    def forward(self, input_ids, bbox, attention_mask=None, token_type_ids=None):
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Token IDs
            bbox (torch.Tensor): Bounding box coordinates for each token
            attention_mask (torch.Tensor): Attention mask
            token_type_ids (torch.Tensor): Token type IDs
            
        Returns:
            torch.Tensor: Classification logits
        """
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Use [CLS] token representation for classification
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0, :]
        
        # Apply classification head
        logits = self.classifier(cls_output)
        
        return logits

class LayoutLM(nn.Module):
    """Main LayoutLM model for layout recognition."""
    
    def __init__(self, input_channels=3, num_classes=9, backbone='resnet50'):
        """
        Initialize LayoutLM model.
        
        Args:
            input_channels (int): Number of input channels
            num_classes (int): Number of layout classes
            backbone (str): Backbone architecture
        """
        super(LayoutLM, self).__init__()
        
        # Initialize backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            # Modify first conv layer to accept arbitrary number of input channels
            if input_channels != 3:
                self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Remove the final FC layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            feature_dim = 2048
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            if input_channels != 3:
                self.backbone.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Segmentation head
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
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
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
        
        # Ensure output size matches input size
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return logits


class LayoutLMWithVisualFeatures(nn.Module):
    """LayoutLM model with additional visual features."""
    
    def __init__(self, num_classes=9, pretrained_model_name="microsoft/layoutlm-base-uncased"):
        """
        Initialize LayoutLM with visual features.
        
        Args:
            num_classes (int): Number of layout classes
            pretrained_model_name (str): Name of the pretrained model
        """
        super(LayoutLMWithVisualFeatures, self).__init__()
        
        # Load pretrained LayoutLM
        self.layoutlm = LayoutLMModel.from_pretrained(pretrained_model_name)
        self.config = self.layoutlm.config
        
        # Visual feature extractor (e.g., from a CNN)
        self.visual_feature_dim = 256
        self.visual_feature_proj = nn.Linear(self.visual_feature_dim, self.config.hidden_size)
        
        # Multimodal fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, input_ids, bbox, visual_features, attention_mask=None, token_type_ids=None, roi_boxes=None):
        """
        Forward pass.
        
        Args:
            input_ids (torch.Tensor): Token IDs
            bbox (torch.Tensor): Bounding box coordinates for each token
            visual_features (torch.Tensor): Visual features from CNN
            attention_mask (torch.Tensor): Attention mask
            token_type_ids (torch.Tensor): Token type IDs
            roi_boxes (torch.Tensor): ROI boxes for visual features
            
        Returns:
            torch.Tensor: Classification logits for each region
        """
        # Get LayoutLM embeddings
        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Extract visual features for each region
        batch_size, num_regions = roi_boxes.size()[:2]
        
        # Reshape visual features for ROI align
        visual_feats = roi_align(
            visual_features,
            roi_boxes.view(-1, 4),
            output_size=(7, 7),
            spatial_scale=1.0,
            sampling_ratio=-1
        )
        
        # Pool visual features
        visual_feats = F.adaptive_avg_pool2d(visual_feats, (1, 1))
        visual_feats = visual_feats.view(batch_size, num_regions, self.visual_feature_dim)
        
        # Project visual features to the same dimension as LayoutLM
        visual_feats = self.visual_feature_proj(visual_feats)
        
        # Extract text features for each region
        text_feats = []
        for b in range(batch_size):
            for r in range(num_regions):
                # Find tokens that belong to this region
                region_tokens = []
                for i in range(1, sequence_output.size(1)):  # Skip [CLS] token
                    if attention_mask is None or attention_mask[b, i] == 1:
                        # Check if token is inside the region
                        token_box = bbox[b, i]
                        region_box = roi_boxes[b, r]
                        
                        if (token_box[0] >= region_box[0] and token_box[2] <= region_box[2] and
                            token_box[1] >= region_box[1] and token_box[3] <= region_box[3]):
                            region_tokens.append(sequence_output[b, i])
                
                if region_tokens:
                    # Average token features
                    region_text_feat = torch.stack(region_tokens).mean(0)
                else:
                    # Use zero vector if no tokens in the region
                    region_text_feat = torch.zeros(self.config.hidden_size, device=sequence_output.device)
                
                text_feats.append(region_text_feat)
        
        text_feats = torch.stack(text_feats).view(batch_size, num_regions, -1)
        
        # Fuse text and visual features
        fused_feats = torch.cat([text_feats, visual_feats], dim=2)
        fused_feats = self.fusion(fused_feats)
        
        # Apply segmentation head
        logits = self.seg_head(fused_feats)
        
        return logits
