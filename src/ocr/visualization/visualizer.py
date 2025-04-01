#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for OCR models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import seaborn as sns
from sklearn.metrics import confusion_matrix


class OCRVisualizer:
    """Class for visualizing OCR model outputs."""
    
    def __init__(self, output_dir='visualizations/ocr'):
        """
        Initialize visualizer.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_prediction(self, image, prediction, target=None, save_path=None):
        """
        Visualize OCR prediction on an image.
        
        Args:
            image (PIL.Image or np.ndarray): Input image
            prediction (str): Predicted text
            target (str, optional): Target text
            save_path (str, optional): Path to save visualization
            
        Returns:
            PIL.Image: Visualization image
        """
        # Convert image to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Create a copy for drawing
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw prediction
        text_y = 10
        draw.rectangle([(10, text_y - 5), (image.width - 10, text_y + 20)], fill=(0, 0, 0, 128))
        draw.text((15, text_y), f"Pred: {prediction}", fill=(255, 255, 255), font=font)
        
        # Draw target if provided
        if target is not None:
            text_y += 25
            draw.rectangle([(10, text_y - 5), (image.width - 10, text_y + 20)], fill=(0, 0, 0, 128))
            draw.text((15, text_y), f"GT: {target}", fill=(255, 255, 255), font=font)
            
            # Indicate if prediction matches target
            text_y += 25
            match = prediction == target
            color = (0, 255, 0) if match else (255, 0, 0)
            draw.rectangle([(10, text_y - 5), (image.width - 10, text_y + 20)], fill=(0, 0, 0, 128))
            draw.text((15, text_y), f"Match: {match}", fill=color, font=font)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vis_image.save(save_path)
        
        return vis_image
    
    def visualize_attention(self, image, attention_map, prediction, target=None, save_path=None):
        """
        Visualize attention map on an image.
        
        Args:
            image (PIL.Image or np.ndarray): Input image
            attention_map (np.ndarray): Attention map
            prediction (str): Predicted text
            target (str, optional): Target text
            save_path (str, optional): Path to save visualization
            
        Returns:
            PIL.Image: Visualization image
        """
        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize attention map to match image size
        attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
        
        # Normalize attention map
        attention_norm = Normalize()(attention_map)
        attention_heatmap = cm.jet(attention_norm)
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        # Create overlay
        alpha = 0.4
        overlay = image.copy()
        overlay = (attention_heatmap[:, :, :3] * 255).astype(np.uint8)
        
        # Blend images
        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        # Convert back to PIL
        vis_image = Image.fromarray(blended)
        draw = ImageDraw.Draw(vis_image)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw prediction
        text_y = 10
        draw.rectangle([(10, text_y - 5), (image.shape[1] - 10, text_y + 20)], fill=(0, 0, 0, 128))
        draw.text((15, text_y), f"Pred: {prediction}", fill=(255, 255, 255), font=font)
        
        # Draw target if provided
        if target is not None:
            text_y += 25
            draw.rectangle([(10, text_y - 5), (image.shape[1] - 10, text_y + 20)], fill=(0, 0, 0, 128))
            draw.text((15, text_y), f"GT: {target}", fill=(255, 255, 255), font=font)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vis_image.save(save_path)
        
        return vis_image
    
    def visualize_batch(self, images, predictions, targets=None, attention_maps=None, save_dir=None):
        """
        Visualize a batch of OCR predictions.
        
        Args:
            images (list): List of images
            predictions (list): List of predicted texts
            targets (list, optional): List of target texts
            attention_maps (list, optional): List of attention maps
            save_dir (str, optional): Directory to save visualizations
            
        Returns:
            list: List of visualization images
        """
        vis_images = []
        
        for i, (image, pred) in enumerate(zip(images, predictions)):
            target = targets[i] if targets is not None else None
            attention_map = attention_maps[i] if attention_maps is not None else None
            
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"sample_{i}.png")
            
            if attention_map is not None:
                vis_image = self.visualize_attention(image, attention_map, pred, target, save_path)
            else:
                vis_image = self.visualize_prediction(image, pred, target, save_path)
            
            vis_images.append(vis_image)
        
        return vis_images
    
    def visualize_confusion_matrix(self, predictions, targets, vocab, save_path=None):
        """
        Visualize character confusion matrix.
        
        Args:
            predictions (list): List of predicted texts
            targets (list): List of target texts
            vocab (list): List of characters in vocabulary
            save_path (str, optional): Path to save visualization
            
        Returns:
            matplotlib.figure.Figure: Confusion matrix figure
        """
        # Create character-level confusion matrix
        all_pred_chars = ''.join(predictions)
        all_target_chars = ''.join(targets)
        
        # Get unique characters
        unique_chars = sorted(set(all_pred_chars + all_target_chars))
        
        # Create confusion matrix
        y_true = []
        y_pred = []
        
        for target, pred in zip(targets, predictions):
            # Align sequences using dynamic programming
            i, j = 0, 0
            while i < len(target) or j < len(pred):
                if i < len(target) and j < len(pred):
                    if target[i] == pred[j]:
                        y_true.append(target[i])
                        y_pred.append(pred[j])
                        i += 1
                        j += 1
                    elif i + 1 < len(target) and target[i + 1] == pred[j]:
                        y_true.append(target[i])
                        y_pred.append('')  # Deletion
                        i += 1
                    elif j + 1 < len(pred) and target[i] == pred[j + 1]:
                        y_true.append('')  # Insertion
                        y_pred.append(pred[j])
                        j += 1
                    else:
                        y_true.append(target[i])
                        y_pred.append(pred[j])
                        i += 1
                        j += 1
                elif i < len(target):
                    y_true.append(target[i])
                    y_pred.append('')
                    i += 1
                elif j < len(pred):
                    y_true.append('')
                    y_pred.append(pred[j])
                    j += 1
        
        # Filter out empty characters
        filtered_y_true = []
        filtered_y_pred = []
        for t, p in zip(y_true, y_pred):
            if t and p:
                filtered_y_true.append(t)
                filtered_y_pred.append(p)
        
        # Create confusion matrix
        cm = confusion_matrix(filtered_y_true, filtered_y_pred, labels=vocab)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        # Create figure
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                   xticklabels=vocab, yticklabels=vocab)
        plt.title('Character Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        
        return plt.gcf()
    
    def visualize_error_distribution(self, predictions, targets, save_path=None):
        """
        Visualize error distribution.
        
        Args:
            predictions (list): List of predicted texts
            targets (list): List of target texts
            save_path (str, optional): Path to save visualization
            
        Returns:
            matplotlib.figure.Figure: Error distribution figure
        """
        # Calculate character error rate for each sample
        cers = []
        for pred, target in zip(predictions, targets):
            # Calculate edit distance
            distance = self._levenshtein_distance(pred, target)
            cer = distance / len(target) if len(target) > 0 else 0
            cers.append(cer)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.hist(cers, bins=20, alpha=0.7, color='blue')
        plt.axvline(np.mean(cers), color='red', linestyle='dashed', linewidth=2, label=f'Mean CER: {np.mean(cers):.4f}')
        plt.title('Character Error Rate Distribution')
        plt.xlabel('Character Error Rate')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        
        return plt.gcf()
    
    def visualize_worst_cases(self, images, predictions, targets, save_dir=None, top_n=5):
        """
        Visualize worst prediction cases.
        
        Args:
            images (list): List of images
            predictions (list): List of predicted texts
            targets (list): List of target texts
            save_dir (str, optional): Directory to save visualizations
            top_n (int): Number of worst cases to visualize
            
        Returns:
            list: List of visualization images
        """
        # Calculate error for each sample
        errors = []
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            distance = self._levenshtein_distance(pred, target)
            error = distance / len(target) if len(target) > 0 else 0
            errors.append((i, error))
        
        # Sort by error
        errors.sort(key=lambda x: x[1], reverse=True)
        
        # Get worst cases
        worst_indices = [idx for idx, _ in errors[:top_n]]
        
        # Visualize worst cases
        vis_images = []
        for i, idx in enumerate(worst_indices):
            image = images[idx]
            pred = predictions[idx]
            target = targets[idx]
            
            save_path = None
            if save_dir:
                save_path = os.path.join(save_dir, f"worst_case_{i}.png")
            
            vis_image = self.visualize_prediction(image, pred, target, save_path)
            vis_images.append(vis_image)
        
        return vis_images
    
    def visualize_metrics(self, metrics, save_path=None):
        """
        Visualize OCR evaluation metrics.
        
        Args:
            metrics (dict): Dictionary of evaluation metrics
            save_path (str, optional): Path to save visualization
            
        Returns:
            matplotlib.figure.Figure: Metrics figure
        """
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot basic metrics
        plt.subplot(2, 2, 1)
        basic_metrics = [metrics['cer'], metrics['wer'], 1 - metrics['accuracy']]
        plt.bar(['CER', 'WER', 'Error Rate'], basic_metrics, color=['blue', 'orange', 'green'])
        plt.title('Basic Error Metrics')
        plt.ylabel('Error Rate')
        plt.grid(True, alpha=0.3)
        
        # Plot metrics by complexity
        plt.subplot(2, 2, 2)
        if 'by_complexity' in metrics:
            complexities = []
            complexity_cers = []
            for complexity, values in metrics['by_complexity'].items():
                complexities.append(complexity)
                complexity_cers.append(values['cer'])
            
            plt.bar(complexities, complexity_cers)
            plt.title('CER by Text Complexity')
            plt.ylabel('Character Error Rate')
            plt.grid(True, alpha=0.3)
        
        # Plot metrics by length
        plt.subplot(2, 2, 3)
        if 'by_length' in metrics:
            lengths = []
            length_cers = []
            for length, values in sorted(metrics['by_length'].items()):
                if isinstance(length, int) and values['total'] >= 5:  # Only plot if enough samples
                    lengths.append(length)
                    length_cers.append(values['cer'])
            
            if lengths:
                plt.plot(lengths, length_cers, marker='o')
                plt.title('CER by Text Length')
                plt.xlabel('Text Length')
                plt.ylabel('Character Error Rate')
                plt.grid(True, alpha=0.3)
        
        # Plot alternative metrics
        plt.subplot(2, 2, 4)
        alt_metrics = [
            metrics['normalized_edit_distance'],
            1 - metrics['normalized_similarity'],
            1 - metrics['case_insensitive_accuracy'],
            1 - metrics['partial_match_accuracy']
        ]
        plt.bar(['Norm. Edit Dist.', 'Dissimilarity', 'Case-Sensitive Error', 'Partial Match Error'], 
                alt_metrics, color=['blue', 'orange', 'green', 'red'])
        plt.title('Alternative Error Metrics')
        plt.ylabel('Error Rate')
        plt.xticks(rotation=15)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        
        return plt.gcf()
    
    def _levenshtein_distance(self, s1, s2):
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            s1 (str): First string
            s2 (str): Second string
            
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
