#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset classes for OCR (Optical Character Recognition).
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from torchvision import transforms

class OCRDataset(Dataset):
    """Dataset class for OCR tasks."""
    
    def __init__(self, 
                 image_dir, 
                 annotation_file, 
                 transform=None, 
                 max_length=100, 
                 char_to_idx=None, 
                 augment=False):
        """
        Initialize OCR dataset.
        
        Args:
            image_dir (str): Directory containing images
            annotation_file (str): Path to annotation file (JSON)
            transform (callable, optional): Optional transform to be applied on images
            max_length (int): Maximum sequence length for text
            char_to_idx (dict, optional): Character to index mapping
            augment (bool): Whether to apply data augmentation
        """
        self.image_dir = image_dir
        self.annotations = self._load_annotations(annotation_file)
        self.transform = transform
        self.max_length = max_length
        self.augment = augment
        
        # Create character to index mapping if not provided
        if char_to_idx is None:
            self.char_to_idx, self.idx_to_char = self._create_char_mappings()
        else:
            self.char_to_idx = char_to_idx
            self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        # Define augmentation transforms
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3)
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.2),
            transforms.RandomRotation(degrees=2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))
        ]) if augment else None
    
    def _load_annotations(self, annotation_file):
        """
        Load annotations from JSON file.
        
        Args:
            annotation_file (str): Path to annotation file
            
        Returns:
            list: List of annotation dictionaries
        """
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # Ensure annotations is a list
        if isinstance(annotations, dict):
            annotations = list(annotations.values())
        
        return annotations
    
    def _create_char_mappings(self):
        """
        Create character to index mappings from the dataset.
        
        Returns:
            tuple: (char_to_idx, idx_to_char) dictionaries
        """
        # Collect all unique characters in the dataset
        chars = set()
        for item in self.annotations:
            text = item.get('text', '')
            chars.update(text)
        
        # Sort characters for deterministic mapping
        chars = sorted(list(chars))
        
        # Add special tokens
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        all_chars = special_tokens + chars
        
        # Create mappings
        char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
        idx_to_char = {idx: char for idx, char in enumerate(all_chars)}
        
        return char_to_idx, idx_to_char
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample containing image and text
        """
        # Get annotation
        annotation = self.annotations[idx]
        
        # Get image path and text
        image_path = os.path.join(self.image_dir, annotation['image_file'])
        text = annotation.get('text', '')
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply augmentation if enabled
        if self.augment and self.augmentation_transforms:
            image = self.augmentation_transforms(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Encode text
        encoded_text = self.encode_text(text)
        text_length = min(len(text), self.max_length)
        
        return {
            'image': image,
            'text': text,
            'encoded_text': encoded_text,
            'text_length': text_length
        }
    
    def encode_text(self, text):
        """
        Encode text to indices.
        
        Args:
            text (str): Input text
            
        Returns:
            torch.Tensor: Encoded text as indices
        """
        # Truncate text if longer than max_length
        text = text[:self.max_length]
        
        # Encode characters to indices
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx['<unk>'])
        
        # Pad sequence to max_length
        padding_length = self.max_length - len(indices)
        if padding_length > 0:
            indices.extend([self.char_to_idx['<pad>']] * padding_length)
        
        return torch.tensor(indices, dtype=torch.long)
    
    def decode_indices(self, indices):
        """
        Decode indices to text.
        
        Args:
            indices (torch.Tensor or list): Indices to decode
            
        Returns:
            str: Decoded text
        """
        # Convert tensor to list if needed
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        
        # Decode indices to characters
        chars = []
        for idx in indices:
            if idx == self.char_to_idx['<pad>'] or idx == self.char_to_idx['<eos>']:
                break
            if idx == self.char_to_idx['<sos>']:
                continue
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
            else:
                chars.append('<unk>')
        
        return ''.join(chars)
    
    def get_vocab_size(self):
        """
        Get vocabulary size.
        
        Returns:
            int: Size of the vocabulary
        """
        return len(self.char_to_idx)
    
    def get_collate_fn(self):
        """
        Get collate function for DataLoader.
        
        Returns:
            callable: Collate function
        """
        def collate_fn(batch):
            images = torch.stack([item['image'] for item in batch])
            encoded_texts = torch.stack([item['encoded_text'] for item in batch])
            text_lengths = torch.tensor([item['text_length'] for item in batch], dtype=torch.long)
            texts = [item['text'] for item in batch]
            
            return {
                'images': images,
                'encoded_texts': encoded_texts,
                'text_lengths': text_lengths,
                'texts': texts
            }
        
        return collate_fn


class LineOCRDataset(OCRDataset):
    """Dataset class for line-level OCR."""
    
    def __init__(self, 
                 image_dir, 
                 annotation_file, 
                 transform=None, 
                 max_length=100, 
                 char_to_idx=None, 
                 augment=False, 
                 height=32, 
                 width=320):
        """
        Initialize Line OCR dataset.
        
        Args:
            image_dir (str): Directory containing images
            annotation_file (str): Path to annotation file (JSON)
            transform (callable, optional): Optional transform to be applied on images
            max_length (int): Maximum sequence length for text
            char_to_idx (dict, optional): Character to index mapping
            augment (bool): Whether to apply data augmentation
            height (int): Height to resize line images to
            width (int): Width to resize line images to
        """
        super().__init__(image_dir, annotation_file, transform, max_length, char_to_idx, augment)
        self.height = height
        self.width = width
        
        # Define default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample containing image and text
        """
        # Get annotation
        annotation = self.annotations[idx]
        
        # Get image path and text
        image_path = os.path.join(self.image_dir, annotation['image_file'])
        text = annotation.get('text', '')
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply augmentation if enabled
        if self.augment and self.augmentation_transforms:
            image = self.augmentation_transforms(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Encode text
        encoded_text = self.encode_text(text)
        text_length = min(len(text), self.max_length)
        
        return {
            'image': image,
            'text': text,
            'encoded_text': encoded_text,
            'text_length': text_length
        }


class WordOCRDataset(OCRDataset):
    """Dataset class for word-level OCR."""
    
    def __init__(self, 
                 image_dir, 
                 annotation_file, 
                 transform=None, 
                 max_length=30, 
                 char_to_idx=None, 
                 augment=False, 
                 height=32, 
                 width=128):
        """
        Initialize Word OCR dataset.
        
        Args:
            image_dir (str): Directory containing images
            annotation_file (str): Path to annotation file (JSON)
            transform (callable, optional): Optional transform to be applied on images
            max_length (int): Maximum sequence length for text
            char_to_idx (dict, optional): Character to index mapping
            augment (bool): Whether to apply data augmentation
            height (int): Height to resize word images to
            width (int): Width to resize word images to
        """
        super().__init__(image_dir, annotation_file, transform, max_length, char_to_idx, augment)
        self.height = height
        self.width = width
        
        # Define default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((height, width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


class DocumentOCRDataset(Dataset):
    """Dataset class for document-level OCR with layout information."""
    
    def __init__(self, 
                 image_dir, 
                 annotation_file, 
                 transform=None, 
                 max_length=100, 
                 char_to_idx=None, 
                 augment=False, 
                 target_size=(1024, 1024)):
        """
        Initialize Document OCR dataset.
        
        Args:
            image_dir (str): Directory containing images
            annotation_file (str): Path to annotation file (JSON)
            transform (callable, optional): Optional transform to be applied on images
            max_length (int): Maximum sequence length for text
            char_to_idx (dict, optional): Character to index mapping
            augment (bool): Whether to apply data augmentation
            target_size (tuple): Target size for document images (height, width)
        """
        self.image_dir = image_dir
        self.annotations = self._load_annotations(annotation_file)
        self.transform = transform
        self.max_length = max_length
        self.augment = augment
        self.target_size = target_size
        
        # Create character to index mapping if not provided
        if char_to_idx is None:
            self.char_to_idx, self.idx_to_char = self._create_char_mappings()
        else:
            self.char_to_idx = char_to_idx
            self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        # Define default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Define augmentation transforms
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ], p=0.3),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.1),
            transforms.RandomRotation(degrees=1),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.98, 1.02))
        ]) if augment else None
    
    def _load_annotations(self, annotation_file):
        """
        Load annotations from JSON file.
        
        Args:
            annotation_file (str): Path to annotation file
            
        Returns:
            list: List of annotation dictionaries
        """
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # Ensure annotations is a list
        if isinstance(annotations, dict):
            annotations = list(annotations.values())
        
        return annotations
    
    def _create_char_mappings(self):
        """
        Create character to index mappings from the dataset.
        
        Returns:
            tuple: (char_to_idx, idx_to_char) dictionaries
        """
        # Collect all unique characters in the dataset
        chars = set()
        for item in self.annotations:
            for text_region in item.get('text_regions', []):
                text = text_region.get('text', '')
                chars.update(text)
        
        # Sort characters for deterministic mapping
        chars = sorted(list(chars))
        
        # Add special tokens
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        all_chars = special_tokens + chars
        
        # Create mappings
        char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
        idx_to_char = {idx: char for idx, char in enumerate(all_chars)}
        
        return char_to_idx, idx_to_char
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample containing image, text regions, and layout information
        """
        # Get annotation
        annotation = self.annotations[idx]
        
        # Get image path
        image_path = os.path.join(self.image_dir, annotation['image_file'])
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Apply augmentation if enabled
        if self.augment and self.augmentation_transforms:
            image = self.augmentation_transforms(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Process text regions
        text_regions = []
        for region in annotation.get('text_regions', []):
            # Get text and bounding box
            text = region.get('text', '')
            bbox = region.get('bbox', [0, 0, 0, 0])  # [x, y, width, height]
            
            # Scale bounding box to target size
            scaled_bbox = self._scale_bbox(bbox, original_size, self.target_size)
            
            # Encode text
            encoded_text = self.encode_text(text)
            text_length = min(len(text), self.max_length)
            
            text_regions.append({
                'text': text,
                'encoded_text': encoded_text,
                'text_length': text_length,
                'bbox': scaled_bbox,
                'region_type': region.get('region_type', 'text')
            })
        
        return {
            'image': image,
            'text_regions': text_regions,
            'image_id': annotation.get('image_id', ''),
            'file_name': annotation['image_file']
        }
    
    def _scale_bbox(self, bbox, original_size, target_size):
        """
        Scale bounding box from original image size to target size.
        
        Args:
            bbox (list): Bounding box [x, y, width, height]
            original_size (tuple): Original image size (width, height)
            target_size (tuple): Target image size (height, width)
            
        Returns:
            list: Scaled bounding box [x, y, width, height]
        """
        # Extract coordinates
        x, y, width, height = bbox
        orig_width, orig_height = original_size
        target_height, target_width = target_size
        
        # Scale coordinates
        scaled_x = int(x * target_width / orig_width)
        scaled_y = int(y * target_height / orig_height)
        scaled_width = int(width * target_width / orig_width)
        scaled_height = int(height * target_height / orig_height)
        
        return [scaled_x, scaled_y, scaled_width, scaled_height]
    
    def encode_text(self, text):
        """
        Encode text to indices.
        
        Args:
            text (str): Input text
            
        Returns:
            torch.Tensor: Encoded text as indices
        """
        # Truncate text if longer than max_length
        text = text[:self.max_length]
        
        # Encode characters to indices
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx['<unk>'])
        
        # Pad sequence to max_length
        padding_length = self.max_length - len(indices)
        if padding_length > 0:
            indices.extend([self.char_to_idx['<pad>']] * padding_length)
        
        return torch.tensor(indices, dtype=torch.long)
    
    def decode_indices(self, indices):
        """
        Decode indices to text.
        
        Args:
            indices (torch.Tensor or list): Indices to decode
            
        Returns:
            str: Decoded text
        """
        # Convert tensor to list if needed
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        
        # Decode indices to characters
        chars = []
        for idx in indices:
            if idx == self.char_to_idx['<pad>'] or idx == self.char_to_idx['<eos>']:
                break
            if idx == self.char_to_idx['<sos>']:
                continue
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
            else:
                chars.append('<unk>')
        
        return ''.join(chars)
    
    def get_vocab_size(self):
        """
        Get vocabulary size.
        
        Returns:
            int: Size of the vocabulary
        """
        return len(self.char_to_idx)
    
    def get_collate_fn(self):
        """
        Get collate function for DataLoader.
        
        Returns:
            callable: Collate function
        """
        def collate_fn(batch):
            images = torch.stack([item['image'] for item in batch])
            image_ids = [item['image_id'] for item in batch]
            file_names = [item['file_name'] for item in batch]
            
            # Process text regions (variable length)
            all_text_regions = []
            for item in batch:
                all_text_regions.append(item['text_regions'])
            
            return {
                'images': images,
                'text_regions': all_text_regions,
                'image_ids': image_ids,
                'file_names': file_names
            }
        
        return collate_fn
