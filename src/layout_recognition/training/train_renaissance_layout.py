#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train layout recognition model on Renaissance dataset for GSoC 2025 submission.
This script trains a layout recognition model on the Renaissance dataset.
"""

import os
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

from src.layout_recognition.data_processing.dataset import LayoutDataset
from src.layout_recognition.models.layoutlm import LayoutLM
from src.layout_recognition.training.config import LayoutConfig
from src.layout_recognition.evaluation.metrics import calculate_iou, calculate_dice, calculate_precision_recall_f1
from src.layout_recognition.visualization.visualizer import LayoutVisualizer


class LayoutTrainer:
    """Trainer class for layout recognition models."""
    
    def __init__(self, model, config, device=None):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): Layout recognition model
            config (dict): Training configuration
            device (torch.device, optional): Device to use for training
        """
        self.model = model
        self.config = config
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
        self.learning_rates = []
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def train(self, train_dataset, val_dataset=None, resume_from=None):
        """
        Train the model.
        
        Args:
            train_dataset (Dataset): Training dataset
            val_dataset (Dataset, optional): Validation dataset
            resume_from (str, optional): Path to checkpoint to resume from
            
        Returns:
            dict: Training history
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('batch_size', 8),
                shuffle=False,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True
            )
        
        # Resume from checkpoint if provided
        if resume_from is not None:
            self._load_checkpoint(resume_from)
        
        # Training loop
        epochs = self.config.get('epochs', 50)
        early_stopping_patience = self.config.get('early_stopping', 10)
        early_stopping_counter = 0
        
        print(f"Starting training for {epochs} epochs on {self.device}")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate if validation dataset is provided
            val_loss, val_iou = 0.0, 0.0
            if val_loader is not None:
                val_loss, val_iou = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_ious.append(val_iou)
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
            
            # Save current learning rate
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val IoU: {val_iou:.4f} - "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if epoch % self.config.get('save_freq', 5) == 0 or epoch == epochs - 1:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
            
            # Save best model
            if val_loader is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.pth")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Plot training curves
            if (epoch + 1) % self.config.get('plot_freq', 10) == 0 or epoch == epochs - 1:
                self._plot_training_curves()
        
        # Training completed
        print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self._save_checkpoint("final_model.pth")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss
        }
        
        # Save history to file
        with open(os.path.join(self.config['output_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f)
        
        return history
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        epoch_loss = 0.0
        
        for batch in train_loader:
            # Move data to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.model.loss_function(outputs, masks)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update epoch loss
            epoch_loss += loss.item()
        
        # Calculate average loss
        avg_loss = epoch_loss / len(train_loader)
        
        return avg_loss
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            tuple: (average validation loss, average IoU)
        """
        self.model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.model.loss_function(outputs, masks)
                val_loss += loss.item()
                
                # Calculate IoU
                pred_masks = torch.argmax(outputs, dim=1)
                batch_iou = calculate_iou(pred_masks.cpu().numpy(), masks.cpu().numpy())
                val_iou += batch_iou
        
        # Calculate average loss and IoU
        avg_loss = val_loss / len(val_loader)
        avg_iou = val_iou / len(val_loader)
        
        return avg_loss, avg_iou
    
    def evaluate(self, test_dataset):
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_dataset (Dataset): Test dataset
            
        Returns:
            dict: Evaluation metrics
        """
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Evaluate
        self.model.eval()
        test_loss = 0.0
        all_ious = []
        all_dices = []
        all_precisions = []
        all_recalls = []
        all_f1s = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Move data to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.model.loss_function(outputs, masks)
                test_loss += loss.item()
                
                # Calculate metrics
                pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()
                true_masks = masks.cpu().numpy()
                
                for i in range(len(pred_masks)):
                    iou = calculate_iou(pred_masks[i], true_masks[i])
                    dice = calculate_dice(pred_masks[i], true_masks[i])
                    precision, recall, f1 = calculate_precision_recall_f1(pred_masks[i], true_masks[i])
                    
                    all_ious.append(iou)
                    all_dices.append(dice)
                    all_precisions.append(precision)
                    all_recalls.append(recall)
                    all_f1s.append(f1)
        
        # Calculate average metrics
        avg_loss = test_loss / len(test_loader)
        avg_iou = np.mean(all_ious)
        avg_dice = np.mean(all_dices)
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1 = np.mean(all_f1s)
        
        # Prepare evaluation results
        results = {
            'loss': avg_loss,
            'iou': avg_iou,
            'dice': avg_dice,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'per_sample': {
                'iou': all_ious,
                'dice': all_dices,
                'precision': all_precisions,
                'recall': all_recalls,
                'f1': all_f1s
            }
        }
        
        # Save results to file
        results_path = os.path.join(self.config['output_dir'], 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation completed:")
        print(f"Loss: {avg_loss:.4f}")
        print(f"IoU: {avg_iou:.4f}")
        print(f"Dice: {avg_dice:.4f}")
        print(f"Precision: {avg_precision:.4f}")
        print(f"Recall: {avg_recall:.4f}")
        print(f"F1: {avg_f1:.4f}")
        
        return results
    
    def _save_checkpoint(self, filename):
        """
        Save model checkpoint.
        
        Args:
            filename (str): Checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_ious': self.val_ious,
            'learning_rates': self.learning_rates,
            'config': self.config
        }
        
        torch.save(checkpoint, os.path.join(self.config['output_dir'], filename))
    
    def _load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_ious = checkpoint.get('val_ious', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch-1}")
    
    def _plot_training_curves(self):
        """Plot training curves."""
        plt.figure(figsize=(15, 10))
        
        # Plot training and validation loss
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot validation IoU
        if self.val_ious:
            plt.subplot(2, 2, 2)
            plt.plot(self.val_ious, label='IoU')
            plt.xlabel('Epoch')
            plt.ylabel('IoU')
            plt.title('Validation IoU')
            plt.legend()
            plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(self.learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate')
        plt.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], f'training_curves_epoch_{self.current_epoch+1}.png'))
        plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train layout recognition model on Renaissance dataset')
    parser.add_argument('--data_dir', type=str, default='data/processed/layout',
                        help='Directory containing processed layout data')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='models/layout/renaissance',
                        help='Directory to save model and results')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training')
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


def main():
    """Main function."""
    args = parse_args()
    
    # Load annotations
    train_annotations = load_annotations(os.path.join(args.data_dir, 'train_annotations.json'))
    val_annotations = load_annotations(os.path.join(args.data_dir, 'val_annotations.json'))
    test_annotations = load_annotations(os.path.join(args.data_dir, 'test_annotations.json'))
    
    # Create datasets
    train_dataset = LayoutDataset(
        annotations=train_annotations,
        augment=True
    )
    
    val_dataset = LayoutDataset(
        annotations=val_annotations,
        augment=False
    )
    
    test_dataset = LayoutDataset(
        annotations=test_annotations,
        augment=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create configuration
    if args.config_file:
        config = LayoutConfig(args.config_file)
    else:
        config = LayoutConfig()
    
    # Update configuration with command line arguments
    config.config['epochs'] = args.epochs
    config.config['batch_size'] = args.batch_size
    config.config['learning_rate'] = args.learning_rate
    config.config['output_dir'] = args.output_dir
    
    if args.device:
        config.config['device'] = args.device
    
    # Create model
    model = LayoutLM(
        input_channels=3,
        num_classes=train_dataset.num_classes,
        backbone=config.config.get('backbone', 'resnet50')
    )
    
    # Create trainer
    trainer = LayoutTrainer(
        model=model,
        config=config.config,
        device=torch.device(config.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    )
    
    # Train model
    print(f"Starting training on {config.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')}...")
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        resume_from=args.resume_from
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Visualize results
    visualizer = LayoutVisualizer(output_dir=os.path.join(config.config['output_dir'], 'visualizations'))
    
    # Visualize test samples
    print("Visualizing test samples...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Get a batch of test samples
    batch = next(iter(test_loader))
    images = batch['image']
    masks = batch['mask']
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(trainer.device))
        pred_masks = torch.argmax(outputs, dim=1)
    
    # Visualize predictions
    for i in range(len(images)):
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        true_mask = masks[i].cpu().numpy()
        pred_mask = pred_masks[i].cpu().numpy()
        
        visualizer.visualize_prediction(
            image=image,
            true_mask=true_mask,
            pred_mask=pred_mask,
            save_path=os.path.join(config.config['output_dir'], 'visualizations', f'test_sample_{i}.png')
        )
    
    # Visualize confusion matrix
    visualizer.visualize_confusion_matrix(
        true_masks=masks.cpu().numpy(),
        pred_masks=pred_masks.cpu().numpy(),
        save_path=os.path.join(config.config['output_dir'], 'visualizations', 'confusion_matrix.png')
    )
    
    print(f"Training and evaluation completed. Results saved to {config.config['output_dir']}")


if __name__ == '__main__':
    main()
