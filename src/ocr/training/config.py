#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration for OCR model training.
"""

import os
import yaml
from datetime import datetime

class OCRConfig:
    """Configuration class for OCR model training."""
    
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
            'train_annotations': 'data/processed/ocr_annotations/train_annotations.json',
            'val_annotations': 'data/processed/ocr_annotations/val_annotations.json',
            'test_annotations': 'data/processed/ocr_annotations/test_annotations.json',
            'image_height': 32,
            'image_width': 320,
            'max_text_length': 100,
            
            # Model settings
            'model_type': 'crnn',  # crnn, resnet_crnn, transformer_crnn
            'input_channels': 3,
            'hidden_size': 256,
            'num_layers': 2,
            'dropout': 0.1,
            'backbone': 'resnet18',  # For CNN-based models
            'bidirectional': True,
            
            # Training settings
            'batch_size': 64,
            'num_workers': 4,
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'optimizer': 'adam',  # adam, sgd, adamw
            'scheduler': 'plateau',  # step, cosine, plateau, none
            'early_stopping': 10,
            'clip_grad': 5.0,
            
            # Optimizer specific settings
            'momentum': 0.9,  # For SGD
            'beta1': 0.9,  # For Adam/AdamW
            'beta2': 0.999,  # For Adam/AdamW
            'nesterov': True,  # For SGD
            
            # Scheduler specific settings
            'step_size': 30,  # For StepLR
            'gamma': 0.1,  # For StepLR
            'T_max': None,  # For CosineAnnealingLR, defaults to epochs
            'eta_min': 0,  # For CosineAnnealingLR
            'patience': 5,  # For ReduceLROnPlateau
            'factor': 0.1,  # For ReduceLROnPlateau
            
            # Augmentation settings
            'augmentation': True,
            'aug_prob': 0.5,
            
            # Output settings
            'output_dir': 'models/ocr',
            'save_freq': 5,
            'plot_freq': 10,
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


# Default configurations for different OCR models

def get_crnn_config():
    """
    Get default configuration for CRNN model.
    
    Returns:
        OCRConfig: Configuration object
    """
    return OCRConfig(
        model_type='crnn',
        backbone='custom',
        input_channels=3,
        hidden_size=256,
        num_layers=2,
        batch_size=64,
        learning_rate=1e-4,
        augmentation=True
    )

def get_resnet_crnn_config():
    """
    Get default configuration for ResNet-CRNN model.
    
    Returns:
        OCRConfig: Configuration object
    """
    return OCRConfig(
        model_type='resnet_crnn',
        backbone='resnet18',
        input_channels=3,
        hidden_size=256,
        num_layers=2,
        batch_size=64,
        learning_rate=1e-4,
        augmentation=True
    )

def get_transformer_crnn_config():
    """
    Get default configuration for Transformer-CRNN model.
    
    Returns:
        OCRConfig: Configuration object
    """
    return OCRConfig(
        model_type='transformer_crnn',
        backbone='resnet18',
        input_channels=3,
        hidden_size=256,
        num_layers=4,
        dropout=0.2,
        batch_size=32,
        learning_rate=5e-5,
        augmentation=True
    )

def get_line_ocr_config():
    """
    Get default configuration for line-level OCR.
    
    Returns:
        OCRConfig: Configuration object
    """
    return OCRConfig(
        model_type='resnet_crnn',
        backbone='resnet18',
        input_channels=3,
        hidden_size=256,
        num_layers=2,
        image_height=32,
        image_width=320,
        max_text_length=100,
        batch_size=64,
        learning_rate=1e-4,
        augmentation=True
    )

def get_word_ocr_config():
    """
    Get default configuration for word-level OCR.
    
    Returns:
        OCRConfig: Configuration object
    """
    return OCRConfig(
        model_type='crnn',
        backbone='custom',
        input_channels=3,
        hidden_size=128,
        num_layers=2,
        image_height=32,
        image_width=128,
        max_text_length=30,
        batch_size=128,
        learning_rate=1e-3,
        augmentation=True
    )
