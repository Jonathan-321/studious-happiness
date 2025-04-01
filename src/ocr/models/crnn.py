#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CRNN (Convolutional Recurrent Neural Network) model for OCR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class CRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) model for OCR.
    Architecture: CNN backbone + Bidirectional RNN + CTC loss
    """
    
    def __init__(self, 
                 input_channels=3, 
                 hidden_size=256, 
                 num_layers=2, 
                 num_classes=95, 
                 dropout=0.1, 
                 backbone='resnet18',
                 bidirectional=True):
        """
        Initialize CRNN model.
        
        Args:
            input_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
            hidden_size (int): Hidden size of RNN
            num_layers (int): Number of RNN layers
            num_classes (int): Number of output classes (vocabulary size)
            dropout (float): Dropout probability
            backbone (str): CNN backbone ('resnet18', 'resnet34', 'vgg16', 'custom')
            bidirectional (bool): Whether to use bidirectional RNN
        """
        super(CRNN, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.backbone_name = backbone
        
        # CNN backbone
        self.cnn = self._build_cnn_backbone(backbone)
        
        # Calculate feature dimensions after CNN
        self.cnn_output_channels = self._get_cnn_output_channels(backbone)
        
        # RNN layers
        self.rnn = nn.GRU(
            input_size=self.cnn_output_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        fc_in_features = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_in_features, num_classes)
    
    def _build_cnn_backbone(self, backbone):
        """
        Build CNN backbone.
        
        Args:
            backbone (str): CNN backbone type
            
        Returns:
            nn.Sequential: CNN backbone
        """
        if backbone == 'resnet18':
            return self._build_resnet_backbone(models.resnet18(pretrained=True))
        elif backbone == 'resnet34':
            return self._build_resnet_backbone(models.resnet34(pretrained=True))
        elif backbone == 'vgg16':
            return self._build_vgg_backbone(models.vgg16_bn(pretrained=True))
        elif backbone == 'custom':
            return self._build_custom_backbone()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def _build_resnet_backbone(self, resnet):
        """
        Build ResNet backbone for CRNN.
        
        Args:
            resnet: ResNet model
            
        Returns:
            nn.Sequential: Modified ResNet backbone
        """
        # Remove the last two layers (avgpool and fc)
        modules = list(resnet.children())[:-2]
        
        # Modify the first conv layer if input channels != 3
        if self.input_channels != 3:
            modules[0] = nn.Conv2d(
                self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        return nn.Sequential(*modules)
    
    def _build_vgg_backbone(self, vgg):
        """
        Build VGG backbone for CRNN.
        
        Args:
            vgg: VGG model
            
        Returns:
            nn.Sequential: Modified VGG backbone
        """
        # Get the features part of VGG
        features = vgg.features
        
        # Modify the first conv layer if input channels != 3
        if self.input_channels != 3:
            features[0] = nn.Conv2d(
                self.input_channels, 64, kernel_size=3, stride=1, padding=1
            )
        
        return features
    
    def _build_custom_backbone(self):
        """
        Build custom CNN backbone.
        
        Returns:
            nn.Sequential: Custom CNN backbone
        """
        return nn.Sequential(
            # Layer 1
            nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # Layer 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Layer 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # Layer 7
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
    
    def _get_cnn_output_channels(self, backbone):
        """
        Get number of output channels from CNN backbone.
        
        Args:
            backbone (str): CNN backbone type
            
        Returns:
            int: Number of output channels
        """
        if backbone == 'resnet18' or backbone == 'resnet34':
            return 512
        elif backbone == 'vgg16':
            return 512
        elif backbone == 'custom':
            return 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, sequence_length, num_classes]
        """
        # CNN feature extraction
        conv_features = self.cnn(x)  # [batch_size, channels, height, width]
        
        # Prepare for RNN
        batch_size, channels, height, width = conv_features.size()
        
        # Reshape to [batch_size, width, height * channels]
        # Width becomes the sequence length, and each "pixel column" is a feature
        conv_features = conv_features.permute(0, 3, 1, 2)  # [batch_size, width, channels, height]
        conv_features = conv_features.reshape(batch_size, width, channels * height)
        
        # RNN sequence processing
        rnn_output, _ = self.rnn(conv_features)  # [batch_size, width, hidden_size*2]
        
        # Linear projection to class scores
        output = self.fc(rnn_output)  # [batch_size, width, num_classes]
        
        # Apply log softmax over class dimension
        output = F.log_softmax(output, dim=2)
        
        return output
    
    def predict(self, x, decoder):
        """
        Make prediction with greedy decoding.
        
        Args:
            x (torch.Tensor): Input tensor
            decoder: Decoder object with decode_greedy method
            
        Returns:
            list: List of predicted texts
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            predictions = decoder.decode_greedy(output)
        return predictions


class CRNNDecoder:
    """Decoder for CRNN model output."""
    
    def __init__(self, idx_to_char, blank_idx=0):
        """
        Initialize decoder.
        
        Args:
            idx_to_char (dict): Index to character mapping
            blank_idx (int): Index of blank token for CTC
        """
        self.idx_to_char = idx_to_char
        self.blank_idx = blank_idx
    
    def decode_greedy(self, output):
        """
        Greedy decoding of model output.
        
        Args:
            output (torch.Tensor): Model output of shape [batch_size, sequence_length, num_classes]
            
        Returns:
            list: List of decoded texts
        """
        # Get most probable class indices
        _, indices = torch.max(output, dim=2)  # [batch_size, sequence_length]
        indices = indices.cpu().numpy()
        
        # Decode each sequence in the batch
        batch_size = indices.shape[0]
        decoded_texts = []
        
        for i in range(batch_size):
            # Get sequence for this batch item
            sequence = indices[i]
            
            # Remove duplicates
            collapsed = []
            for j, idx in enumerate(sequence):
                if j == 0 or idx != sequence[j-1]:
                    collapsed.append(idx)
            
            # Remove blank tokens
            filtered = [idx for idx in collapsed if idx != self.blank_idx]
            
            # Convert indices to characters
            text = ''.join([self.idx_to_char.get(idx, '') for idx in filtered])
            decoded_texts.append(text)
        
        return decoded_texts
    
    def decode_beam_search(self, output, beam_width=5):
        """
        Beam search decoding of model output.
        
        Args:
            output (torch.Tensor): Model output of shape [batch_size, sequence_length, num_classes]
            beam_width (int): Beam width for search
            
        Returns:
            list: List of decoded texts
        """
        # Convert to log probabilities
        log_probs = output.cpu().numpy()
        batch_size = log_probs.shape[0]
        decoded_texts = []
        
        for i in range(batch_size):
            # Get log probabilities for this batch item
            sequence_log_probs = log_probs[i]  # [sequence_length, num_classes]
            
            # Initialize beam
            beam = [([], 0.0)]  # (prefix, accumulated_log_prob)
            
            # Beam search through the sequence
            for t in range(sequence_log_probs.shape[0]):
                # Get log probabilities at this time step
                t_log_probs = sequence_log_probs[t]  # [num_classes]
                
                # Extend each beam
                new_beam = []
                for prefix, accumulated_log_prob in beam:
                    # Add top-k probable characters
                    top_indices = t_log_probs.argsort()[-beam_width:]
                    
                    for idx in top_indices:
                        new_prefix = prefix + [idx]
                        new_log_prob = accumulated_log_prob + t_log_probs[idx]
                        new_beam.append((new_prefix, new_log_prob))
                
                # Keep top-k beams
                new_beam.sort(key=lambda x: x[1], reverse=True)
                beam = new_beam[:beam_width]
            
            # Get best beam
            best_prefix, _ = beam[0]
            
            # Remove duplicates
            collapsed = []
            for j, idx in enumerate(best_prefix):
                if j == 0 or idx != best_prefix[j-1]:
                    collapsed.append(idx)
            
            # Remove blank tokens
            filtered = [idx for idx in collapsed if idx != self.blank_idx]
            
            # Convert indices to characters
            text = ''.join([self.idx_to_char.get(idx, '') for idx in filtered])
            decoded_texts.append(text)
        
        return decoded_texts


class CTCLoss(nn.Module):
    """CTC Loss for CRNN model."""
    
    def __init__(self, blank_idx=0, reduction='mean'):
        """
        Initialize CTC Loss.
        
        Args:
            blank_idx (int): Index of blank token
            reduction (str): Reduction method ('none', 'mean', 'sum')
        """
        super(CTCLoss, self).__init__()
        self.criterion = nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Forward pass.
        
        Args:
            log_probs (torch.Tensor): Log probabilities from model [batch_size, sequence_length, num_classes]
            targets (torch.Tensor): Target indices [batch_size, max_target_length]
            input_lengths (torch.Tensor): Lengths of input sequences [batch_size]
            target_lengths (torch.Tensor): Lengths of target sequences [batch_size]
            
        Returns:
            torch.Tensor: CTC loss
        """
        # Permute dimensions to [sequence_length, batch_size, num_classes]
        log_probs = log_probs.permute(1, 0, 2)
        
        # Flatten targets
        targets = targets.flatten()
        
        # Compute CTC loss
        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
        
        return loss


class ResNetCRNN(CRNN):
    """ResNet-based CRNN model for OCR."""
    
    def __init__(self, 
                 input_channels=3, 
                 hidden_size=256, 
                 num_layers=2, 
                 num_classes=95, 
                 dropout=0.1,
                 resnet_type='resnet18'):
        """
        Initialize ResNet-based CRNN model.
        
        Args:
            input_channels (int): Number of input channels
            hidden_size (int): Hidden size of RNN
            num_layers (int): Number of RNN layers
            num_classes (int): Number of output classes
            dropout (float): Dropout probability
            resnet_type (str): ResNet type ('resnet18', 'resnet34', 'resnet50')
        """
        super(ResNetCRNN, self).__init__(
            input_channels=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            backbone=resnet_type,
            bidirectional=True
        )


class EfficientNetCRNN(nn.Module):
    """EfficientNet-based CRNN model for OCR."""
    
    def __init__(self, 
                 input_channels=3, 
                 hidden_size=256, 
                 num_layers=2, 
                 num_classes=95, 
                 dropout=0.1,
                 efficientnet_type='efficientnet-b0'):
        """
        Initialize EfficientNet-based CRNN model.
        
        Args:
            input_channels (int): Number of input channels
            hidden_size (int): Hidden size of RNN
            num_layers (int): Number of RNN layers
            num_classes (int): Number of output classes
            dropout (float): Dropout probability
            efficientnet_type (str): EfficientNet type ('efficientnet-b0', 'efficientnet-b1', etc.)
        """
        super(EfficientNetCRNN, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Import EfficientNet dynamically to avoid dependency issues
        try:
            from efficientnet_pytorch import EfficientNet
            self.cnn = EfficientNet.from_pretrained(efficientnet_type)
            
            # Modify the first conv layer if input channels != 3
            if input_channels != 3:
                self.cnn._conv_stem = nn.Conv2d(
                    input_channels, self.cnn._conv_stem.out_channels,
                    kernel_size=3, stride=2, bias=False
                )
            
            # Get output channels from EfficientNet
            if 'b0' in efficientnet_type:
                self.cnn_output_channels = 1280
            elif 'b1' in efficientnet_type:
                self.cnn_output_channels = 1280
            elif 'b2' in efficientnet_type:
                self.cnn_output_channels = 1408
            elif 'b3' in efficientnet_type:
                self.cnn_output_channels = 1536
            else:
                self.cnn_output_channels = 1792
        except ImportError:
            raise ImportError("Please install efficientnet_pytorch: pip install efficientnet_pytorch")
        
        # RNN layers
        self.rnn = nn.GRU(
            input_size=self.cnn_output_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, sequence_length, num_classes]
        """
        # Extract features using EfficientNet
        features = self.cnn.extract_features(x)  # [batch_size, channels, height, width]
        
        # Prepare for RNN
        batch_size, channels, height, width = features.size()
        
        # Reshape to [batch_size, width, height * channels]
        features = features.permute(0, 3, 1, 2)  # [batch_size, width, channels, height]
        features = features.reshape(batch_size, width, channels * height)
        
        # RNN sequence processing
        rnn_output, _ = self.rnn(features)  # [batch_size, width, hidden_size*2]
        
        # Linear projection to class scores
        output = self.fc(rnn_output)  # [batch_size, width, num_classes]
        
        # Apply log softmax over class dimension
        output = F.log_softmax(output, dim=2)
        
        return output
    
    def predict(self, x, decoder):
        """
        Make prediction with greedy decoding.
        
        Args:
            x (torch.Tensor): Input tensor
            decoder: Decoder object with decode_greedy method
            
        Returns:
            list: List of predicted texts
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            predictions = decoder.decode_greedy(output)
        return predictions


class TransformerCRNN(nn.Module):
    """Transformer-based CRNN model for OCR."""
    
    def __init__(self, 
                 input_channels=3, 
                 hidden_size=256, 
                 num_layers=4, 
                 num_classes=95, 
                 dropout=0.1,
                 backbone='resnet18',
                 nhead=8):
        """
        Initialize Transformer-based CRNN model.
        
        Args:
            input_channels (int): Number of input channels
            hidden_size (int): Hidden size of transformer
            num_layers (int): Number of transformer layers
            num_classes (int): Number of output classes
            dropout (float): Dropout probability
            backbone (str): CNN backbone type
            nhead (int): Number of attention heads
        """
        super(TransformerCRNN, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # CNN backbone
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
            modules = list(resnet.children())[:-2]
            if input_channels != 3:
                modules[0] = nn.Conv2d(
                    input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            self.cnn = nn.Sequential(*modules)
            self.cnn_output_channels = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
            modules = list(resnet.children())[:-2]
            if input_channels != 3:
                modules[0] = nn.Conv2d(
                    input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            self.cnn = nn.Sequential(*modules)
            self.cnn_output_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        # Feature projection
        self.feature_proj = nn.Linear(self.cnn_output_channels, hidden_size)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, sequence_length, num_classes]
        """
        # CNN feature extraction
        features = self.cnn(x)  # [batch_size, channels, height, width]
        
        # Prepare for transformer
        batch_size, channels, height, width = features.size()
        
        # Reshape to [batch_size, width, height * channels]
        features = features.permute(0, 3, 1, 2)  # [batch_size, width, channels, height]
        features = features.reshape(batch_size, width, channels * height)
        
        # Project features to hidden size
        features = self.feature_proj(features)  # [batch_size, width, hidden_size]
        
        # Add positional encoding
        features = self.pos_encoder(features)  # [batch_size, width, hidden_size]
        
        # Transformer expects [sequence_length, batch_size, hidden_size]
        features = features.permute(1, 0, 2)  # [width, batch_size, hidden_size]
        
        # Transformer encoding
        transformer_output = self.transformer_encoder(features)  # [width, batch_size, hidden_size]
        
        # Back to [batch_size, width, hidden_size]
        transformer_output = transformer_output.permute(1, 0, 2)
        
        # Output projection
        output = self.output_proj(transformer_output)  # [batch_size, width, num_classes]
        
        # Apply log softmax over class dimension
        output = F.log_softmax(output, dim=2)
        
        return output
    
    def predict(self, x, decoder):
        """
        Make prediction with greedy decoding.
        
        Args:
            x (torch.Tensor): Input tensor
            decoder: Decoder object with decode_greedy method
            
        Returns:
            list: List of predicted texts
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            predictions = decoder.decode_greedy(output)
        return predictions


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Model dimension
            dropout (float): Dropout probability
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, d_model]
            
        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)
