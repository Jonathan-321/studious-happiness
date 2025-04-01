#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset classes for layout recognition.
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LayoutDataset(Dataset):
    """Dataset for document layout analysis."""
    
    def __init__(self, image_dir, annotation_file, transform=None, target_size=(512, 512)):
        """
        Initialize the layout dataset.
        
        Args:
            image_dir (str): Directory containing document images
            annotation_file (str): Path to the annotation file
            transform (callable, optional): Optional transform to be applied on a sample
            target_size (tuple): Target size for resizing images (height, width)
        """
        self.image_dir = image_dir
        self.target_size = target_size
        
        # Load annotations
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        self.image_files = list(self.annotations.keys())
        
        # Define region types
        self.region_types = [
            "text", "title", "paragraph", "figure", "table", 
            "marginalia", "header", "footer", "decoration"
        ]
        self.region_to_idx = {region: i for i, region in enumerate(self.region_types)}
        
        # Set up transforms
        if transform is None:
            self.transform = A.Compose([
                A.Resize(height=target_size[0], width=target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample containing image and layout information
        """
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        orig_height, orig_width = image.shape[:2]
        
        # Get annotations for this image
        regions = self.annotations[image_file]
        
        # Create layout mask (one channel per region type)
        mask = np.zeros((len(self.region_types), self.target_size[0], self.target_size[1]), dtype=np.float32)
        
        # Create bounding boxes for detection models
        boxes = []
        labels = []
        
        # Scale factor for bounding boxes
        scale_x = self.target_size[1] / orig_width
        scale_y = self.target_size[0] / orig_height
        
        for region in regions:
            region_type = region["type"]
            class_idx = self.region_to_idx[region_type]
            
            # Get bounding box coordinates
            x, y, w, h = region["bbox"]
            
            # Scale coordinates
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            w_scaled = int(w * scale_x)
            h_scaled = int(h * scale_y)
            
            # Ensure coordinates are within bounds
            x_scaled = max(0, min(x_scaled, self.target_size[1] - 1))
            y_scaled = max(0, min(y_scaled, self.target_size[0] - 1))
            w_scaled = max(1, min(w_scaled, self.target_size[1] - x_scaled))
            h_scaled = max(1, min(h_scaled, self.target_size[0] - y_scaled))
            
            # Add to mask
            mask[class_idx, y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled] = 1.0
            
            # Add to boxes and labels
            boxes.append([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled])
            labels.append(class_idx)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image = transformed["image"]
        
        # Convert boxes and labels to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)
        
        # Convert mask to tensor
        mask = torch.tensor(mask, dtype=torch.float32)
        
        return {
            "image": image,
            "mask": mask,
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
            "file_name": image_file
        }

class COCOLayoutDataset(Dataset):
    """Dataset for document layout analysis using COCO format."""
    
    def __init__(self, image_dir, coco_annotation_file, transform=None, target_size=(512, 512)):
        """
        Initialize the COCO layout dataset.
        
        Args:
            image_dir (str): Directory containing document images
            coco_annotation_file (str): Path to the COCO annotation file
            transform (callable, optional): Optional transform to be applied on a sample
            target_size (tuple): Target size for resizing images (height, width)
        """
        self.image_dir = image_dir
        self.target_size = target_size
        
        # Load COCO annotations
        with open(coco_annotation_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        self.images = coco_data["images"]
        self.annotations = coco_data["annotations"]
        self.categories = coco_data["categories"]
        
        # Create mapping from image_id to annotations
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann["image_id"]
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)
        
        # Create category mapping
        self.cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(self.categories)}
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in self.categories}
        
        # Set up transforms
        if transform is None:
            self.transform = A.Compose([
                A.Resize(height=target_size[0], width=target_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Sample containing image and layout information
        """
        image_info = self.images[idx]
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_path = os.path.join(self.image_dir, file_name)
        
        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))
        orig_height, orig_width = image.shape[:2]
        
        # Get annotations for this image
        image_anns = self.image_id_to_annotations.get(image_id, [])
        
        # Create mask (one channel per category)
        mask = np.zeros((len(self.categories), self.target_size[0], self.target_size[1]), dtype=np.float32)
        
        # Create bounding boxes for detection models
        boxes = []
        labels = []
        
        # Scale factor for bounding boxes
        scale_x = self.target_size[1] / orig_width
        scale_y = self.target_size[0] / orig_height
        
        for ann in image_anns:
            cat_id = ann["category_id"]
            class_idx = self.cat_id_to_idx[cat_id]
            
            # Get bounding box coordinates (COCO format: [x, y, width, height])
            x, y, w, h = ann["bbox"]
            
            # Scale coordinates
            x_scaled = int(x * scale_x)
            y_scaled = int(y * scale_y)
            w_scaled = int(w * scale_x)
            h_scaled = int(h * scale_y)
            
            # Ensure coordinates are within bounds
            x_scaled = max(0, min(x_scaled, self.target_size[1] - 1))
            y_scaled = max(0, min(y_scaled, self.target_size[0] - 1))
            w_scaled = max(1, min(w_scaled, self.target_size[1] - x_scaled))
            h_scaled = max(1, min(h_scaled, self.target_size[0] - y_scaled))
            
            # Add to mask
            mask[class_idx, y_scaled:y_scaled+h_scaled, x_scaled:x_scaled+w_scaled] = 1.0
            
            # Add to boxes and labels (convert to [x1, y1, x2, y2] format)
            boxes.append([x_scaled, y_scaled, x_scaled + w_scaled, y_scaled + h_scaled])
            labels.append(class_idx)
        
        # Apply transforms
        transformed = self.transform(image=image)
        image = transformed["image"]
        
        # Convert boxes and labels to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.long)
        
        # Convert mask to tensor
        mask = torch.tensor(mask, dtype=torch.float32)
        
        return {
            "image": image,
            "mask": mask,
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "file_name": file_name
        }
