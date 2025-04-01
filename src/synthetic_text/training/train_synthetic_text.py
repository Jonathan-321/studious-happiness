#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train diffusion model for synthetic Renaissance text generation.
This script trains a diffusion model to generate synthetic Renaissance-style text images.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image

from src.synthetic_text.models.diffusion_model import UNet, DiffusionModel
from src.synthetic_text.data_processing.dataset import TextImageDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train diffusion model for synthetic Renaissance text generation')
    parser.add_argument('--data_dir', type=str, default='data/processed/text_images',
                        help='Directory containing processed text image data')
    parser.add_argument('--output_dir', type=str, default='models/synthetic_text',
                        help='Directory to save model and results')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size for training')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Frequency to save checkpoints')
    parser.add_argument('--sample_freq', type=int, default=5,
                        help='Frequency to generate samples during training')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of samples to generate for visualization')
    return parser.parse_args()


def load_annotations(json_path):
    """
    Load annotations from JSON file.
    
    Args:
        json_path (str): Path to JSON file
        
    Returns:
        list: List of annotations
    """
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    return annotations


def save_images(images, save_dir, prefix='sample', normalize=True):
    """
    Save images to disk.
    
    Args:
        images (torch.Tensor): Batch of images
        save_dir (str): Directory to save images
        prefix (str): Prefix for image filenames
        normalize (bool): Whether to normalize images from [-1, 1] to [0, 255]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        # Convert to numpy and transpose
        img = img.detach().cpu().numpy().transpose(1, 2, 0)
        
        # Normalize if needed
        if normalize:
            img = (img + 1) / 2  # [-1, 1] -> [0, 1]
            img = (img * 255).astype(np.uint8)
        
        # Convert to PIL image and save
        img = Image.fromarray(img.squeeze(), mode='L')
        img.save(os.path.join(save_dir, f'{prefix}_{i:04d}.png'))


def plot_loss(losses, save_path):
    """
    Plot training loss.
    
    Args:
        losses (list): List of losses
        save_path (str): Path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Load annotations
    train_annotations = load_annotations(os.path.join(args.data_dir, 'train_annotations.json'))
    val_annotations = load_annotations(os.path.join(args.data_dir, 'val_annotations.json'))
    
    # Create datasets
    train_dataset = TextImageDataset(
        annotations=train_annotations,
        img_size=args.img_size,
        augment=True
    )
    
    val_dataset = TextImageDataset(
        annotations=val_annotations,
        img_size=args.img_size,
        augment=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = UNet(
        in_channels=1,
        out_channels=1,
        time_emb_dim=256,
        base_channels=64,
        channel_mults=(1, 2, 4, 8)
    )
    
    # Create diffusion model
    diffusion = DiffusionModel(
        model=model,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=args.timesteps,
        device=device
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=50,
        gamma=0.5
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume_from:
        print(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # Train model
    print(f"Starting training for {args.epochs} epochs")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        model.train()
        epoch_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Get batch
            x = batch['image'].to(device)
            
            # Sample random timesteps
            t = torch.randint(0, args.timesteps, (x.shape[0],), device=device).long()
            
            # Calculate loss
            loss = diffusion.p_losses(x, t)
            
            # Update model
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        val_epoch_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch
                x = batch['image'].to(device)
                
                # Sample random timesteps
                t = torch.randint(0, args.timesteps, (x.shape[0],), device=device).long()
                
                # Calculate loss
                loss = diffusion.p_losses(x, t)
                val_epoch_losses.append(loss.item())
        
        # Calculate average validation loss
        avg_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
        val_losses.append(avg_val_loss)
        
        # Update scheduler
        scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {avg_train_loss:.6f} - "
              f"Val Loss: {avg_val_loss:.6f} - "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Generate samples
        if (epoch + 1) % args.sample_freq == 0 or epoch == args.epochs - 1:
            print(f"Generating samples at epoch {epoch+1}")
            
            model.eval()
            samples = diffusion.generate_samples(
                num_samples=args.num_samples,
                img_size=(1, args.img_size, args.img_size),
                batch_size=4
            )
            
            # Save samples
            save_images(
                images=samples,
                save_dir=os.path.join(samples_dir, f'epoch_{epoch+1}'),
                prefix='sample',
                normalize=True
            )
        
        # Plot and save loss curves
        plot_loss(
            losses=train_losses,
            save_path=os.path.join(args.output_dir, 'train_loss.png')
        )
        
        plot_loss(
            losses=val_losses,
            save_path=os.path.join(args.output_dir, 'val_loss.png')
        )
    
    # Save final model
    diffusion.save(os.path.join(args.output_dir, 'final_model.pt'))
    
    # Generate final samples
    print("Generating final samples")
    
    model.eval()
    samples = diffusion.generate_samples(
        num_samples=args.num_samples * 2,
        img_size=(1, args.img_size, args.img_size),
        batch_size=4
    )
    
    # Save final samples
    save_images(
        images=samples,
        save_dir=os.path.join(args.output_dir, 'final_samples'),
        prefix='sample',
        normalize=True
    )
    
    print(f"Training completed. Model and samples saved to {args.output_dir}")


if __name__ == '__main__':
    main()
