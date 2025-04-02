#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to prepare training data for the Renaissance layout model.
"""

import os
import sys
import argparse
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

def prepare_data(args):
    """Prepare training data."""
    # Create directories
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'val'), exist_ok=True)
    
    if args.mask_dir:
        os.makedirs(os.path.join(args.output_dir, 'train_masks'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'val_masks'), exist_ok=True)
    
    # Get image paths
    image_paths = []
    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                image_paths.append(os.path.join(root, file))
    
    # Sort for reproducibility
    image_paths.sort()
    
    # Shuffle
    random.seed(42)
    random.shuffle(image_paths)
    
    # Split into train and val
    split_idx = int(len(image_paths) * (1 - args.val_split))
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    print(f"Total images: {len(image_paths)}")
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    
    # Copy images to train and val directories
    print("Copying training images...")
    for path in tqdm(train_paths):
        filename = os.path.basename(path)
        shutil.copy(path, os.path.join(args.output_dir, 'train', filename))
        
        # Copy mask if available
        if args.mask_dir:
            mask_path = os.path.join(args.mask_dir, os.path.splitext(filename)[0] + '.png')
            if os.path.exists(mask_path):
                shutil.copy(mask_path, os.path.join(args.output_dir, 'train_masks', os.path.splitext(filename)[0] + '.png'))
    
    print("Copying validation images...")
    for path in tqdm(val_paths):
        filename = os.path.basename(path)
        shutil.copy(path, os.path.join(args.output_dir, 'val', filename))
        
        # Copy mask if available
        if args.mask_dir:
            mask_path = os.path.join(args.mask_dir, os.path.splitext(filename)[0] + '.png')
            if os.path.exists(mask_path):
                shutil.copy(mask_path, os.path.join(args.output_dir, 'val_masks', os.path.splitext(filename)[0] + '.png'))
    
    # If no masks are provided, generate pseudo-masks using Otsu thresholding
    if not args.mask_dir and args.generate_masks:
        print("Generating pseudo-masks for training images...")
        generate_pseudo_masks(os.path.join(args.output_dir, 'train'), os.path.join(args.output_dir, 'train_masks'))
        
        print("Generating pseudo-masks for validation images...")
        generate_pseudo_masks(os.path.join(args.output_dir, 'val'), os.path.join(args.output_dir, 'val_masks'))
    
    print("Data preparation complete!")

def generate_pseudo_masks(image_dir, mask_dir):
    """Generate pseudo-masks using Otsu thresholding."""
    os.makedirs(mask_dir, exist_ok=True)
    
    # Get image paths
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                image_paths.append(os.path.join(root, file))
    
    # Generate masks
    for path in tqdm(image_paths):
        # Load image
        image = Image.open(path).convert('L')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply Otsu's thresholding
        threshold = np.mean(img_array) - 0.5 * np.std(img_array)
        mask = (img_array < threshold).astype(np.uint8) * 255
        
        # Save mask
        filename = os.path.splitext(os.path.basename(path))[0] + '.png'
        mask_path = os.path.join(mask_dir, filename)
        Image.fromarray(mask).save(mask_path)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare training data for Renaissance layout model')
    parser.add_argument('--image_dir', type=str, default='data/processed/layout/images',
                        help='Directory containing images')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='Directory containing masks (optional)')
    parser.add_argument('--output_dir', type=str, default='data/renaissance',
                        help='Directory to save prepared data')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--generate_masks', action='store_true',
                        help='Generate pseudo-masks if no masks are provided')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    prepare_data(args)

if __name__ == '__main__':
    main()
