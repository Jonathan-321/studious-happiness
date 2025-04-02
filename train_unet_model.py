#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for U-Net Renaissance layout recognition model.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import random
import cv2
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import the model
from layout_recognition.models.unet_model import UNetLayoutModel

class RenaissanceDataset(Dataset):
    """Dataset for Renaissance layout recognition."""
    
    def __init__(self, image_dir, mask_dir=None, transform=None, generate_pseudo_masks=False):
        """
        Initialize dataset.
        
        Args:
            image_dir (str): Directory containing images
            mask_dir (str, optional): Directory containing masks
            transform (callable, optional): Optional transform to be applied on a sample
            generate_pseudo_masks (bool): Whether to generate pseudo masks if mask_dir is None
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.generate_pseudo_masks = generate_pseudo_masks
        
        # Get image paths
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        # Sort image paths for reproducibility
        self.image_paths.sort()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Resize image to model input size
        image = image.resize((512, 512))
        
        # Convert to numpy array and normalize
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Get mask
        if self.mask_dir:
            # Get mask path
            mask_name = os.path.basename(image_path)
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            # Load mask if it exists
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize((512, 512))
                mask_np = np.array(mask).astype(np.float32) / 255.0
                mask_np = (mask_np > 0.5).astype(np.float32)
            else:
                # Generate pseudo mask if mask doesn't exist
                mask_np = self._generate_pseudo_mask(image_np)
        elif self.generate_pseudo_masks:
            # Generate pseudo mask
            mask_np = self._generate_pseudo_mask(image_np)
        else:
            # No mask, return zeros
            mask_np = np.zeros((512, 512), dtype=np.float32)
        
        # Apply transform if specified
        if self.transform:
            image_np, mask_np = self.transform(image_np, mask_np)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'path': image_path
        }
    
    def _generate_pseudo_mask(self, image_np):
        """
        Generate pseudo mask for an image.
        
        Args:
            image_np (np.ndarray): Image as numpy array
            
        Returns:
            np.ndarray: Pseudo mask
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            (gray * 255).astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Normalize
        mask = binary.astype(np.float32) / 255.0
        
        return mask

def train_model(args):
    """
    Train the model.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = RenaissanceDataset(
        image_dir=args.train_dir,
        mask_dir=args.mask_dir,
        generate_pseudo_masks=args.generate_pseudo_masks
    )
    
    val_dataset = RenaissanceDataset(
        image_dir=args.val_dir,
        mask_dir=args.mask_dir,
        generate_pseudo_masks=args.generate_pseudo_masks
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create model
    model = UNetLayoutModel(n_channels=3, n_classes=2, bilinear=False)
    model = model.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Train)"):
            # Get batch
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Convert masks to long for CrossEntropyLoss
            masks = masks.squeeze(1).long()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Update loss
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Val)"):
                # Get batch
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Convert masks to long for CrossEntropyLoss
                masks = masks.squeeze(1).long()
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss = criterion(outputs, masks)
                
                # Update loss
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    # Save final model
    checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1]
    }
    
    torch.save(checkpoint, os.path.join(args.output_dir, 'final_model.pth'))
    print(f"Saved final model with validation loss: {val_losses[-1]:.4f}")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'loss_curves.png'))
    
    print(f"Training complete! Models saved to {args.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net Renaissance layout model')
    parser.add_argument('--train_dir', type=str, default='data/renaissance/train',
                        help='Directory containing training images')
    parser.add_argument('--val_dir', type=str, default='data/renaissance/val',
                        help='Directory containing validation images')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory containing masks (optional)')
    parser.add_argument('--output_dir', type=str, default='models/unet',
                        help='Directory to save models')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--generate_pseudo_masks', action='store_true',
                        help='Generate pseudo masks if mask_dir is None')
    return parser.parse_args()

def main():
    args = parse_args()
    train_model(args)

if __name__ == '__main__':
    main()
