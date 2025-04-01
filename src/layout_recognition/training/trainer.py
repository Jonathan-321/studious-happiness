#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training loop implementation for layout recognition models.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class LayoutTrainer:
    """Trainer class for layout recognition models."""
    
    def __init__(self, model, train_dataset, val_dataset, config):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): Model to train
            train_dataset (Dataset): Training dataset
            val_dataset (Dataset): Validation dataset
            config (dict): Training configuration
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Set device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 8),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Set up optimizer
        optimizer_name = config.get('optimizer', 'adam').lower()
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = config.get('momentum', 0.9)
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Set up learning rate scheduler
        scheduler_name = config.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'step':
            step_size = config.get('step_size', 10)
            gamma = config.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'cosine':
            T_max = config.get('T_max', config.get('epochs', 100))
            eta_min = config.get('eta_min', 0)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_name == 'plateau':
            patience = config.get('patience', 5)
            factor = config.get('factor', 0.1)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience, verbose=True
            )
        elif scheduler_name == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        # Set up loss function
        loss_name = config.get('loss', 'cross_entropy').lower()
        
        if loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_name == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_name == 'dice':
            from src.layout_recognition.utils.losses import DiceLoss
            self.criterion = DiceLoss()
        elif loss_name == 'focal':
            from src.layout_recognition.utils.losses import FocalLoss
            self.criterion = FocalLoss()
        elif loss_name == 'combined':
            from src.layout_recognition.utils.losses import CombinedLoss
            self.criterion = CombinedLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
        
        # Set up logging and checkpoints
        self.output_dir = config.get('output_dir', 'models/layout_recognition')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up TensorBoard
        log_dir = os.path.join(self.output_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # Add file handler
            file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch
            
        Returns:
            float: Average loss for this epoch
        """
        self.model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), global_step)
        
        # Calculate average loss
        avg_loss = epoch_loss / len(self.train_loader)
        
        # Log epoch metrics
        self.logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        
        return avg_loss
    
    def validate(self, epoch):
        """
        Validate the model.
        
        Args:
            epoch (int): Current epoch
            
        Returns:
            float: Validation loss
            float: IoU score
        """
        self.model.eval()
        val_loss = 0
        iou_scores = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate IoU
                preds = torch.sigmoid(outputs) > 0.5
                iou = self.calculate_iou(preds, masks)
                iou_scores.append(iou)
        
        # Calculate average metrics
        avg_val_loss = val_loss / len(self.val_loader)
        avg_iou = np.mean(iou_scores)
        
        # Log validation metrics
        self.logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}")
        self.writer.add_scalar('val/loss', avg_val_loss, epoch)
        self.writer.add_scalar('val/iou', avg_iou, epoch)
        
        return avg_val_loss, avg_iou
    
    def calculate_iou(self, preds, targets):
        """
        Calculate IoU score.
        
        Args:
            preds (torch.Tensor): Predicted masks
            targets (torch.Tensor): Ground truth masks
            
        Returns:
            float: IoU score
        """
        # Flatten tensors
        preds = preds.view(-1).float()
        targets = targets.view(-1).float()
        
        # Calculate intersection and union
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        
        # Calculate IoU
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        return iou.item()
    
    def save_checkpoint(self, epoch, val_loss, val_iou, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            val_loss (float): Validation loss
            val_iou (float): Validation IoU
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_iou': val_iou,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.output_dir, 'latest_checkpoint.pth'))
        
        # Save epoch checkpoint
        torch.save(checkpoint, os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'best_model.pth'))
            self.logger.info(f"Saved best model with IoU: {val_iou:.4f}")
    
    def train(self):
        """
        Train the model for the specified number of epochs.
        
        Returns:
            dict: Training history
        """
        epochs = self.config.get('epochs', 100)
        early_stopping = self.config.get('early_stopping', 0)
        
        best_val_loss = float('inf')
        best_val_iou = 0
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'lr': []
        }
        
        self.logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_iou = self.validate(epoch)
            history['val_loss'].append(val_loss)
            history['val_iou'].append(val_iou)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            self.writer.add_scalar('train/lr', current_lr, epoch)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check if this is the best model
            is_best = val_iou > best_val_iou
            if is_best:
                best_val_iou = val_iou
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, val_iou, is_best)
            
            # Early stopping
            if early_stopping > 0 and patience_counter >= early_stopping:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation IoU: {best_val_iou:.4f}")
        
        # Close TensorBoard writer
        self.writer.close()
        
        return history

class MaskRCNNTrainer:
    """Trainer class for Mask R-CNN model."""
    
    def __init__(self, model, train_dataset, val_dataset, config):
        """
        Initialize the Mask R-CNN trainer.
        
        Args:
            model (nn.Module): Mask R-CNN model
            train_dataset (Dataset): Training dataset
            val_dataset (Dataset): Validation dataset
            config (dict): Training configuration
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Set device
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = self.model.to(self.device)
        
        # Create data loaders with custom collate function
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 2),
            shuffle=True,
            num_workers=config.get('num_workers', 4),
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 2),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            collate_fn=self.collate_fn
        )
        
        # Set up optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer_name = config.get('optimizer', 'sgd').lower()
        lr = config.get('learning_rate', 1e-3)
        weight_decay = config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            momentum = config.get('momentum', 0.9)
            self.optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Set up learning rate scheduler
        scheduler_name = config.get('scheduler', 'step').lower()
        
        if scheduler_name == 'step':
            step_size = config.get('step_size', 3)
            gamma = config.get('gamma', 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'cosine':
            T_max = config.get('T_max', config.get('epochs', 12))
            eta_min = config.get('eta_min', 0)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_name == 'plateau':
            patience = config.get('patience', 2)
            factor = config.get('factor', 0.1)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience, verbose=True
            )
        elif scheduler_name == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        # Set up logging and checkpoints
        self.output_dir = config.get('output_dir', 'models/layout_recognition')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up TensorBoard
        log_dir = os.path.join(self.output_dir, 'logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # Add file handler
            file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Training samples: {len(train_dataset)}")
        self.logger.info(f"Validation samples: {len(val_dataset)}")
    
    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for Mask R-CNN.
        
        Args:
            batch (list): Batch of samples
            
        Returns:
            tuple: Tuple of images and targets
        """
        return tuple(zip(*batch))
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch
            
        Returns:
            dict: Losses for this epoch
        """
        self.model.train()
        epoch_loss = 0
        epoch_loss_classifier = 0
        epoch_loss_box_reg = 0
        epoch_loss_mask = 0
        epoch_loss_objectness = 0
        epoch_loss_rpn_box_reg = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move data to device
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            loss_dict = self.model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize
            losses.backward()
            self.optimizer.step()
            
            # Update progress bar
            epoch_loss += losses.item()
            epoch_loss_classifier += loss_dict['loss_classifier'].item()
            epoch_loss_box_reg += loss_dict['loss_box_reg'].item()
            epoch_loss_mask += loss_dict['loss_mask'].item()
            epoch_loss_objectness += loss_dict['loss_objectness'].item()
            epoch_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
            
            progress_bar.set_postfix({'loss': losses.item()})
            
            # Log to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('train/loss', losses.item(), global_step)
            self.writer.add_scalar('train/loss_classifier', loss_dict['loss_classifier'].item(), global_step)
            self.writer.add_scalar('train/loss_box_reg', loss_dict['loss_box_reg'].item(), global_step)
            self.writer.add_scalar('train/loss_mask', loss_dict['loss_mask'].item(), global_step)
            self.writer.add_scalar('train/loss_objectness', loss_dict['loss_objectness'].item(), global_step)
            self.writer.add_scalar('train/loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'].item(), global_step)
        
        # Calculate average losses
        avg_loss = epoch_loss / len(self.train_loader)
        avg_loss_classifier = epoch_loss_classifier / len(self.train_loader)
        avg_loss_box_reg = epoch_loss_box_reg / len(self.train_loader)
        avg_loss_mask = epoch_loss_mask / len(self.train_loader)
        avg_loss_objectness = epoch_loss_objectness / len(self.train_loader)
        avg_loss_rpn_box_reg = epoch_loss_rpn_box_reg / len(self.train_loader)
        
        # Log epoch metrics
        self.logger.info(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        self.writer.add_scalar('train/epoch_loss_classifier', avg_loss_classifier, epoch)
        self.writer.add_scalar('train/epoch_loss_box_reg', avg_loss_box_reg, epoch)
        self.writer.add_scalar('train/epoch_loss_mask', avg_loss_mask, epoch)
        self.writer.add_scalar('train/epoch_loss_objectness', avg_loss_objectness, epoch)
        self.writer.add_scalar('train/epoch_loss_rpn_box_reg', avg_loss_rpn_box_reg, epoch)
        
        return {
            'loss': avg_loss,
            'loss_classifier': avg_loss_classifier,
            'loss_box_reg': avg_loss_box_reg,
            'loss_mask': avg_loss_mask,
            'loss_objectness': avg_loss_objectness,
            'loss_rpn_box_reg': avg_loss_rpn_box_reg
        }
    
    def validate(self, epoch):
        """
        Validate the model.
        
        Args:
            epoch (int): Current epoch
            
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        val_loss = 0
        val_loss_classifier = 0
        val_loss_box_reg = 0
        val_loss_mask = 0
        val_loss_objectness = 0
        val_loss_rpn_box_reg = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = self.model(images, targets)
                
                # Calculate total loss
                losses = sum(loss for loss in loss_dict.values())
                
                # Update metrics
                val_loss += losses.item()
                val_loss_classifier += loss_dict['loss_classifier'].item()
                val_loss_box_reg += loss_dict['loss_box_reg'].item()
                val_loss_mask += loss_dict['loss_mask'].item()
                val_loss_objectness += loss_dict['loss_objectness'].item()
                val_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        
        # Calculate average losses
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_loss_classifier = val_loss_classifier / len(self.val_loader)
        avg_val_loss_box_reg = val_loss_box_reg / len(self.val_loader)
        avg_val_loss_mask = val_loss_mask / len(self.val_loader)
        avg_val_loss_objectness = val_loss_objectness / len(self.val_loader)
        avg_val_loss_rpn_box_reg = val_loss_rpn_box_reg / len(self.val_loader)
        
        # Log validation metrics
        self.logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
        self.writer.add_scalar('val/loss', avg_val_loss, epoch)
        self.writer.add_scalar('val/loss_classifier', avg_val_loss_classifier, epoch)
        self.writer.add_scalar('val/loss_box_reg', avg_val_loss_box_reg, epoch)
        self.writer.add_scalar('val/loss_mask', avg_val_loss_mask, epoch)
        self.writer.add_scalar('val/loss_objectness', avg_val_loss_objectness, epoch)
        self.writer.add_scalar('val/loss_rpn_box_reg', avg_val_loss_rpn_box_reg, epoch)
        
        return {
            'loss': avg_val_loss,
            'loss_classifier': avg_val_loss_classifier,
            'loss_box_reg': avg_val_loss_box_reg,
            'loss_mask': avg_val_loss_mask,
            'loss_objectness': avg_val_loss_objectness,
            'loss_rpn_box_reg': avg_val_loss_rpn_box_reg
        }
    
    def save_checkpoint(self, epoch, val_metrics, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            val_metrics (dict): Validation metrics
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(self.output_dir, 'latest_checkpoint.pth'))
        
        # Save epoch checkpoint
        torch.save(checkpoint, os.path.join(self.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'best_model.pth'))
            self.logger.info(f"Saved best model with loss: {val_metrics['loss']:.4f}")
    
    def train(self):
        """
        Train the model for the specified number of epochs.
        
        Returns:
            dict: Training history
        """
        epochs = self.config.get('epochs', 12)
        early_stopping = self.config.get('early_stopping', 0)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        self.logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            history['train_loss'].append(train_metrics['loss'])
            
            # Validate
            val_metrics = self.validate(epoch)
            history['val_loss'].append(val_metrics['loss'])
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['lr'].append(current_lr)
            self.writer.add_scalar('train/lr', current_lr, epoch)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Check if this is the best model
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if early_stopping > 0 and patience_counter >= early_stopping:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Close TensorBoard writer
        self.writer.close()
        
        return history
