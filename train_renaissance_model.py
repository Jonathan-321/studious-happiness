#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train the Renaissance layout model.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import the model
from layout_recognition.models.renaissance_model import RenaissanceLayoutModel

class RenaissanceDataset(Dataset):
    """Dataset for Renaissance layout recognition."""
    
    def __init__(self, image_dir, mask_dir=None, transform=None, is_train=True):
        """
        Initialize the dataset.
        
        Args:
            image_dir (str): Directory containing images
            mask_dir (str): Directory containing masks (optional)
            transform (callable): Optional transform to be applied on a sample
            is_train (bool): Whether this is a training dataset
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train
        
        # Get image paths
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))
        
        # Sort for reproducibility
        self.image_paths.sort()
        
        print(f"Found {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Get mask path if available
        if self.mask_dir:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(self.mask_dir, f"{image_name}.png")
            
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                # Ensure mask is binary (0 or 1)
                mask = np.array(mask) > 127
                mask = mask.astype(np.int64)
            else:
                # If mask doesn't exist, create a dummy mask
                mask = np.zeros((image.height, image.width), dtype=np.int64)
        else:
            # For inference or when masks are not available
            mask = np.zeros((image.height, image.width), dtype=np.int64)
            
            # For training without masks, we can use a simple heuristic to create pseudo-masks
            if self.is_train:
                # Convert to grayscale and threshold to create a pseudo-mask
                gray = np.array(image.convert('L'))
                # Otsu's thresholding
                mask = (gray < np.mean(gray) - 0.5 * np.std(gray)).astype(np.int64)
        
        # Resize both image and mask to a fixed size
        image = image.resize((224, 224), Image.LANCZOS)
        mask = Image.fromarray(mask.astype(np.uint8) * 255).resize((224, 224), Image.NEAREST)
        mask = np.array(mask) > 127  # Convert back to binary
        mask = mask.astype(np.int64)
        
        # Convert to tensors
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask)
        
        # Apply transforms to image only
        if self.transform:
            image = self.transform(image)
            
        # Normalize image
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        
        return {'image': image, 'mask': mask, 'path': image_path}

def train_model(args):
    """Train the model."""
    # Set device
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    print("Creating model...")
    model = RenaissanceLayoutModel(input_channels=3, backbone=args.backbone)
    model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Create datasets
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])
    
    val_transform = None
    
    train_dataset = RenaissanceDataset(
        image_dir=os.path.join(args.data_dir, 'train'),
        mask_dir=os.path.join(args.data_dir, 'train_masks') if os.path.exists(os.path.join(args.data_dir, 'train_masks')) else None,
        transform=train_transform,
        is_train=True
    )
    
    val_dataset = RenaissanceDataset(
        image_dir=os.path.join(args.data_dir, 'val'),
        mask_dir=os.path.join(args.data_dir, 'val_masks') if os.path.exists(os.path.join(args.data_dir, 'val_masks')) else None,
        transform=val_transform,
        is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Resize masks to match output size
            if outputs.shape[2:] != masks.shape[1:]:
                # Resize masks to match output size
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(),
                    size=outputs.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * images.size(0)
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == masks).sum().item()
            train_total += masks.numel()
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Get data
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Resize masks to match output size
                if outputs.shape[2:] != masks.shape[1:]:
                    # Resize masks to match output size
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(1).float(),
                        size=outputs.shape[2:],
                        mode='nearest'
                    ).squeeze(1).long()
                
                # Calculate loss
                loss = criterion(outputs, masks)
                
                # Update metrics
                val_loss += loss.item() * images.size(0)
                
                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == masks).sum().item()
                val_total += masks.numel()
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving best model with val_loss: {val_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(args.output_dir, 'best_model.pth'))
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc
        }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("Training complete!")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Renaissance layout model')
    parser.add_argument('--data_dir', type=str, default='data/renaissance',
                        help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, default='models/renaissance',
                        help='Directory to save model')
    parser.add_argument('--backbone', type=str, default='resnet34',
                        help='Backbone architecture (resnet18, resnet34)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (cuda or cpu)')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    train_model(args)

if __name__ == '__main__':
    main()
