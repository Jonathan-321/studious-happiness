#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training configuration for layout recognition models.
"""

import os
import yaml
from datetime import datetime

class Config:
    """Configuration class for training layout recognition models."""
    
    def __init__(self, config_file=None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            config_file (str, optional): Path to YAML configuration file
            **kwargs: Additional configuration parameters
        """
        # Default configuration
        self.config = {
            # Data settings
            'data_dir': 'data/processed',
            'train_annotations': 'data/processed/layout_annotations/train_annotations.json',
            'val_annotations': 'data/processed/layout_annotations/val_annotations.json',
            'image_size': (512, 512),
            'num_classes': 9,
            
            # Model settings
            'model_type': 'unet',  # unet, layoutlm, mask_rcnn
            'pretrained': True,
            'backbone': 'resnet50',
            
            # Training settings
            'batch_size': 8,
            'num_workers': 4,
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'optimizer': 'adam',  # adam, sgd, adamw
            'scheduler': 'cosine',  # step, cosine, plateau, none
            'early_stopping': 10,
            
            # Optimizer specific settings
            'momentum': 0.9,  # For SGD
            'beta1': 0.9,  # For Adam/AdamW
            'beta2': 0.999,  # For Adam/AdamW
            
            # Scheduler specific settings
            'step_size': 10,  # For StepLR
            'gamma': 0.1,  # For StepLR
            'T_max': None,  # For CosineAnnealingLR, defaults to epochs
            'eta_min': 0,  # For CosineAnnealingLR
            'patience': 5,  # For ReduceLROnPlateau
            'factor': 0.1,  # For ReduceLROnPlateau
            
            # Loss settings
            'loss': 'cross_entropy',  # cross_entropy, bce, dice, focal, combined
            'focal_alpha': 0.25,  # For FocalLoss
            'focal_gamma': 2.0,  # For FocalLoss
            
            # Augmentation settings
            'augmentation': True,
            'aug_prob': 0.5,
            
            # Output settings
            'output_dir': 'models/layout_recognition',
            'save_freq': 5,
            'device': 'cuda',
        }
        
        # Load configuration from file if provided
        if config_file is not None:
            self.load_from_file(config_file)
        
        # Update with additional parameters
        self.config.update(kwargs)
        
        # Set output directory with timestamp
        if 'output_dir_timestamp' not in kwargs or kwargs['output_dir_timestamp']:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.config['output_dir'] = os.path.join(
                self.config['output_dir'],
                f"{self.config['model_type']}_{timestamp}"
            )
    
    def load_from_file(self, config_file):
        """
        Load configuration from YAML file.
        
        Args:
            config_file (str): Path to YAML configuration file
        """
        with open(config_file, 'r') as f:
            file_config = yaml.safe_load(f)
            self.config.update(file_config)
    
    def save_to_file(self, output_file):
        """
        Save configuration to YAML file.
        
        Args:
            output_file (str): Path to save the configuration
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def __getitem__(self, key):
        """Get configuration value."""
        return self.config.get(key)
    
    def __setitem__(self, key, value):
        """Set configuration value."""
        self.config[key] = value
    
    def get(self, key, default=None):
        """
        Get configuration value with default.
        
        Args:
            key (str): Configuration key
            default: Default value if key is not found
            
        Returns:
            Value for the key or default
        """
        return self.config.get(key, default)
    
    def update(self, config_dict):
        """
        Update configuration with dictionary.
        
        Args:
            config_dict (dict): Dictionary with configuration values
        """
        self.config.update(config_dict)
    
    def to_dict(self):
        """
        Convert configuration to dictionary.
        
        Returns:
            dict: Configuration dictionary
        """
        return self.config.copy()

# Default configurations for different models

def get_unet_config():
    """
    Get default configuration for U-Net model.
    
    Returns:
        Config: Configuration object
    """
    return Config(
        model_type='unet',
        backbone='resnet50',
        image_size=(512, 512),
        batch_size=8,
        learning_rate=1e-4,
        loss='combined',
        augmentation=True
    )

def get_layoutlm_config():
    """
    Get default configuration for LayoutLM model.
    
    Returns:
        Config: Configuration object
    """
    return Config(
        model_type='layoutlm',
        pretrained_model_name='microsoft/layoutlm-base-uncased',
        image_size=(512, 512),
        batch_size=4,
        learning_rate=5e-5,
        loss='cross_entropy',
        augmentation=False
    )

def get_mask_rcnn_config():
    """
    Get default configuration for Mask R-CNN model.
    
    Returns:
        Config: Configuration object
    """
    return Config(
        model_type='mask_rcnn',
        backbone='resnet50',
        image_size=(800, 800),
        batch_size=2,
        learning_rate=1e-3,
        optimizer='sgd',
        momentum=0.9,
        weight_decay=1e-4,
        scheduler='step',
        step_size=3,
        gamma=0.1,
        epochs=12
    )
