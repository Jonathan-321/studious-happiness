#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trainer for OCR models.
"""

import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from ..models.crnn import CTCLoss, CRNNDecoder


class OCRTrainer:
    """Trainer class for OCR models."""
    
    def __init__(self, model, config, device=None):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): OCR model
            config (dict): Training configuration
            device (torch.device, optional): Device to use for training
        """
        self.model = model
        self.config = config
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = CTCLoss(blank_idx=0)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize decoder
        self.decoder = None  # Will be set when dataset is loaded
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_cer = float('inf')
        self.best_val_wer = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_cers = []
        self.val_wers = []
        self.learning_rates = []
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def _create_optimizer(self):
        """
        Create optimizer.
        
        Returns:
            torch.optim.Optimizer: Optimizer
        """
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.999))
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.999))
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=self.config.get('momentum', 0.9),
                weight_decay=weight_decay,
                nesterov=self.config.get('nesterov', True)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_scheduler(self):
        """
        Create learning rate scheduler.
        
        Returns:
            torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
        """
        scheduler_name = self.config.get('scheduler', 'plateau').lower()
        
        if scheduler_name == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('factor', 0.1),
                patience=self.config.get('patience', 5),
                verbose=True
            )
        elif scheduler_name == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('T_max', self.config.get('epochs', 100)),
                eta_min=self.config.get('eta_min', 0)
            )
        elif scheduler_name == 'step':
            return StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'none' or not scheduler_name:
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
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
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            collate_fn=train_dataset.get_collate_fn(),
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.get('batch_size', 32),
                shuffle=False,
                num_workers=self.config.get('num_workers', 4),
                collate_fn=val_dataset.get_collate_fn(),
                pin_memory=True
            )
        
        # Initialize decoder
        self.decoder = CRNNDecoder(
            idx_to_char=train_dataset.idx_to_char,
            blank_idx=0
        )
        
        # Resume from checkpoint if provided
        if resume_from is not None:
            self._load_checkpoint(resume_from)
        
        # Training loop
        epochs = self.config.get('epochs', 100)
        early_stopping_patience = self.config.get('early_stopping', 10)
        early_stopping_counter = 0
        
        print(f"Starting training for {epochs} epochs on {self.device}")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self._train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate if validation dataset is provided
            val_loss, val_cer, val_wer = 0.0, 0.0, 0.0
            if val_loader is not None:
                val_loss, val_cer, val_wer = self._validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_cers.append(val_cer)
                self.val_wers.append(val_wer)
                
                # Update learning rate scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
            else:
                # Update learning rate scheduler
                if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step()
            
            # Save current learning rate
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val CER: {val_cer:.4f} - "
                  f"Val WER: {val_wer:.4f} - "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if epoch % self.config.get('save_freq', 5) == 0 or epoch == epochs - 1:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
            
            # Save best model
            is_best = False
            if val_loader is not None:
                # Check if this is the best model based on validation loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    is_best = True
                    early_stopping_counter = 0
                    self._save_checkpoint("best_model_loss.pth")
                else:
                    early_stopping_counter += 1
                
                # Check if this is the best model based on CER
                if val_cer < self.best_val_cer:
                    self.best_val_cer = val_cer
                    self._save_checkpoint("best_model_cer.pth")
                
                # Check if this is the best model based on WER
                if val_wer < self.best_val_wer:
                    self.best_val_wer = val_wer
                    self._save_checkpoint("best_model_wer.pth")
                
                # Early stopping
                if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Plot training curves
            if (epoch + 1) % self.config.get('plot_freq', 10) == 0 or epoch == epochs - 1:
                self._plot_training_curves()
        
        # Training completed
        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation CER: {self.best_val_cer:.4f}")
        print(f"Best validation WER: {self.best_val_wer:.4f}")
        
        # Save final model
        self._save_checkpoint("final_model.pth")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_cers': self.val_cers,
            'val_wers': self.val_wers,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_val_cer': self.best_val_cer,
            'best_val_wer': self.best_val_wer,
            'total_time': total_time
        }
        
        # Save history to file
        with open(os.path.join(self.config['output_dir'], 'training_history.json'), 'w') as f:
            json.dump(history, f)
        
        return history
    
    def _train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        epoch_loss = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch in pbar:
            # Move data to device
            images = batch['images'].to(self.device)
            encoded_texts = batch['encoded_texts'].to(self.device)
            text_lengths = batch['text_lengths'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            batch_size, seq_length, _ = outputs.size()
            input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long, device=self.device)
            
            loss = self.criterion(outputs, encoded_texts, input_lengths, text_lengths)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('clip_grad', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('clip_grad'))
            
            self.optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        avg_loss = epoch_loss / len(train_loader)
        
        return avg_loss
    
    def _validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            tuple: (average validation loss, character error rate, word error rate)
        """
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                images = batch['images'].to(self.device)
                encoded_texts = batch['encoded_texts'].to(self.device)
                text_lengths = batch['text_lengths'].to(self.device)
                texts = batch['texts']
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                batch_size, seq_length, _ = outputs.size()
                input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long, device=self.device)
                
                loss = self.criterion(outputs, encoded_texts, input_lengths, text_lengths)
                val_loss += loss.item()
                
                # Decode predictions
                predictions = self.decoder.decode_greedy(outputs)
                all_predictions.extend(predictions)
                all_targets.extend(texts)
        
        # Calculate average loss
        avg_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        cer = self._calculate_cer(all_predictions, all_targets)
        wer = self._calculate_wer(all_predictions, all_targets)
        
        return avg_loss, cer, wer
    
    def _calculate_cer(self, predictions, targets):
        """
        Calculate Character Error Rate (CER).
        
        Args:
            predictions (list): List of predicted texts
            targets (list): List of target texts
            
        Returns:
            float: Character Error Rate
        """
        total_distance = 0
        total_length = 0
        
        for pred, target in zip(predictions, targets):
            # Calculate Levenshtein distance
            distance = self._levenshtein_distance(pred, target)
            total_distance += distance
            total_length += len(target)
        
        # Calculate CER
        if total_length == 0:
            return 0.0
        
        cer = total_distance / total_length
        
        return cer
    
    def _calculate_wer(self, predictions, targets):
        """
        Calculate Word Error Rate (WER).
        
        Args:
            predictions (list): List of predicted texts
            targets (list): List of target texts
            
        Returns:
            float: Word Error Rate
        """
        total_distance = 0
        total_length = 0
        
        for pred, target in zip(predictions, targets):
            # Split into words
            pred_words = pred.split()
            target_words = target.split()
            
            # Calculate Levenshtein distance
            distance = self._levenshtein_distance(pred_words, target_words)
            total_distance += distance
            total_length += len(target_words)
        
        # Calculate WER
        if total_length == 0:
            return 0.0
        
        wer = total_distance / total_length
        
        return wer
    
    def _levenshtein_distance(self, s1, s2):
        """
        Calculate Levenshtein distance between two sequences.
        
        Args:
            s1: First sequence
            s2: Second sequence
            
        Returns:
            int: Levenshtein distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
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
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val_loss': self.best_val_loss,
            'best_val_cer': self.best_val_cer,
            'best_val_wer': self.best_val_wer,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_cers': self.val_cers,
            'val_wers': self.val_wers,
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
        
        if checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_cer = checkpoint.get('best_val_cer', float('inf'))
        self.best_val_wer = checkpoint.get('best_val_wer', float('inf'))
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_cers = checkpoint.get('val_cers', [])
        self.val_wers = checkpoint.get('val_wers', [])
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
        
        # Plot validation CER and WER
        if self.val_cers and self.val_wers:
            plt.subplot(2, 2, 2)
            plt.plot(self.val_cers, label='CER')
            plt.plot(self.val_wers, label='WER')
            plt.xlabel('Epoch')
            plt.ylabel('Error Rate')
            plt.title('Validation Error Rates')
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
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            collate_fn=test_dataset.get_collate_fn(),
            pin_memory=True
        )
        
        # Initialize decoder if not already initialized
        if self.decoder is None:
            self.decoder = CRNNDecoder(
                idx_to_char=test_dataset.idx_to_char,
                blank_idx=0
            )
        
        # Evaluate
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []
        all_image_ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation"):
                # Move data to device
                images = batch['images'].to(self.device)
                encoded_texts = batch['encoded_texts'].to(self.device)
                text_lengths = batch['text_lengths'].to(self.device)
                texts = batch['texts']
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                batch_size, seq_length, _ = outputs.size()
                input_lengths = torch.full((batch_size,), seq_length, dtype=torch.long, device=self.device)
                
                loss = self.criterion(outputs, encoded_texts, input_lengths, text_lengths)
                test_loss += loss.item()
                
                # Decode predictions
                predictions = self.decoder.decode_greedy(outputs)
                all_predictions.extend(predictions)
                all_targets.extend(texts)
                
                # Store image IDs if available
                if 'image_ids' in batch:
                    all_image_ids.extend(batch['image_ids'])
        
        # Calculate average loss
        avg_loss = test_loss / len(test_loader)
        
        # Calculate metrics
        cer = self._calculate_cer(all_predictions, all_targets)
        wer = self._calculate_wer(all_predictions, all_targets)
        
        # Calculate per-sample metrics
        per_sample_metrics = []
        for i, (pred, target) in enumerate(zip(all_predictions, all_targets)):
            sample_cer = self._calculate_cer([pred], [target])
            sample_wer = self._calculate_wer([pred], [target])
            
            sample_metric = {
                'prediction': pred,
                'target': target,
                'cer': sample_cer,
                'wer': sample_wer
            }
            
            if all_image_ids:
                sample_metric['image_id'] = all_image_ids[i]
            
            per_sample_metrics.append(sample_metric)
        
        # Sort samples by error rate
        sorted_by_cer = sorted(per_sample_metrics, key=lambda x: x['cer'], reverse=True)
        
        # Prepare evaluation results
        results = {
            'loss': avg_loss,
            'cer': cer,
            'wer': wer,
            'per_sample': per_sample_metrics,
            'worst_samples': sorted_by_cer[:10],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save results to file
        results_path = os.path.join(self.config['output_dir'], 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation completed:")
        print(f"Loss: {avg_loss:.4f}")
        print(f"CER: {cer:.4f}")
        print(f"WER: {wer:.4f}")
        
        return results
