#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization utilities for layout recognition.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
from PIL import Image, ImageDraw, ImageFont

class LayoutVisualizer:
    """Class for visualizing layout recognition results."""
    
    def __init__(self, class_names=None, colors=None):
        """
        Initialize the visualizer.
        
        Args:
            class_names (list, optional): List of class names
            colors (list, optional): List of colors for each class
        """
        # Default class names for document layout
        self.class_names = class_names or [
            'Background', 'Title', 'Text', 'List', 'Table', 
            'Figure', 'Caption', 'Header', 'Footer'
        ]
        
        # Default colors for visualization
        self.colors = colors or [
            [0, 0, 0],       # Background - Black
            [255, 0, 0],     # Title - Red
            [0, 0, 255],     # Text - Blue
            [0, 255, 0],     # List - Green
            [255, 255, 0],   # Table - Yellow
            [255, 0, 255],   # Figure - Magenta
            [0, 255, 255],   # Caption - Cyan
            [128, 0, 0],     # Header - Maroon
            [0, 128, 0]      # Footer - Dark Green
        ]
        
        # Create colormap for segmentation masks
        self.cmap = ListedColormap(np.array(self.colors) / 255.0)
    
    def visualize_mask(self, image, mask, alpha=0.5, save_path=None, show=True):
        """
        Visualize segmentation mask overlaid on the image.
        
        Args:
            image (np.ndarray): Input image
            mask (np.ndarray): Segmentation mask of shape [H, W] with class indices
            alpha (float): Transparency of the mask overlay
            save_path (str, optional): Path to save the visualization
            show (bool): Whether to display the visualization
            
        Returns:
            np.ndarray: Visualization image
        """
        # Convert image to RGB if grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot image with mask overlay
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(mask, alpha=alpha, cmap=self.cmap, vmin=0, vmax=len(self.class_names)-1)
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        # Add colorbar with class names
        cbar = plt.colorbar(ticks=np.arange(len(self.class_names)))
        cbar.set_ticklabels(self.class_names)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        # Create and return visualization image
        vis_image = image.copy()
        colored_mask = self.apply_mask_color(mask)
        vis_image = cv2.addWeighted(vis_image, 1-alpha, colored_mask, alpha, 0)
        
        return vis_image
    
    def apply_mask_color(self, mask):
        """
        Apply colors to segmentation mask.
        
        Args:
            mask (np.ndarray): Segmentation mask of shape [H, W] with class indices
            
        Returns:
            np.ndarray: Colored mask of shape [H, W, 3]
        """
        # Initialize colored mask
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply color for each class
        for i, color in enumerate(self.colors):
            colored_mask[mask == i] = color
        
        return colored_mask
    
    def visualize_comparison(self, image, pred_mask, gt_mask, save_path=None, show=True):
        """
        Visualize comparison between predicted and ground truth masks.
        
        Args:
            image (np.ndarray): Input image
            pred_mask (np.ndarray): Predicted segmentation mask
            gt_mask (np.ndarray): Ground truth segmentation mask
            save_path (str, optional): Path to save the visualization
            show (bool): Whether to display the visualization
            
        Returns:
            np.ndarray: Visualization image
        """
        # Convert image to RGB if grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create figure
        plt.figure(figsize=(16, 8))
        
        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot ground truth mask
        plt.subplot(1, 3, 2)
        plt.imshow(image)
        plt.imshow(gt_mask, alpha=0.5, cmap=self.cmap, vmin=0, vmax=len(self.class_names)-1)
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Plot predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.imshow(pred_mask, alpha=0.5, cmap=self.cmap, vmin=0, vmax=len(self.class_names)-1)
        plt.title('Prediction')
        plt.axis('off')
        
        # Add colorbar with class names
        cbar = plt.colorbar(ticks=np.arange(len(self.class_names)))
        cbar.set_ticklabels(self.class_names)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        # Create and return visualization image (side-by-side comparison)
        h, w = image.shape[:2]
        vis_image = np.zeros((h, w*3, 3), dtype=np.uint8)
        
        # Original image
        vis_image[:, :w] = image
        
        # Ground truth overlay
        gt_colored = image.copy()
        gt_mask_colored = self.apply_mask_color(gt_mask)
        gt_overlay = cv2.addWeighted(gt_colored, 0.5, gt_mask_colored, 0.5, 0)
        vis_image[:, w:2*w] = gt_overlay
        
        # Prediction overlay
        pred_colored = image.copy()
        pred_mask_colored = self.apply_mask_color(pred_mask)
        pred_overlay = cv2.addWeighted(pred_colored, 0.5, pred_mask_colored, 0.5, 0)
        vis_image[:, 2*w:] = pred_overlay
        
        return vis_image
    
    def visualize_error(self, image, pred_mask, gt_mask, save_path=None, show=True):
        """
        Visualize error between predicted and ground truth masks.
        
        Args:
            image (np.ndarray): Input image
            pred_mask (np.ndarray): Predicted segmentation mask
            gt_mask (np.ndarray): Ground truth segmentation mask
            save_path (str, optional): Path to save the visualization
            show (bool): Whether to display the visualization
            
        Returns:
            np.ndarray: Visualization image
        """
        # Convert image to RGB if grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create error mask
        error_mask = (pred_mask != gt_mask).astype(np.uint8)
        
        # Create figure
        plt.figure(figsize=(16, 8))
        
        # Plot original image
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot ground truth mask
        plt.subplot(2, 2, 2)
        plt.imshow(image)
        plt.imshow(gt_mask, alpha=0.5, cmap=self.cmap, vmin=0, vmax=len(self.class_names)-1)
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Plot predicted mask
        plt.subplot(2, 2, 3)
        plt.imshow(image)
        plt.imshow(pred_mask, alpha=0.5, cmap=self.cmap, vmin=0, vmax=len(self.class_names)-1)
        plt.title('Prediction')
        plt.axis('off')
        
        # Plot error mask
        plt.subplot(2, 2, 4)
        plt.imshow(image)
        plt.imshow(error_mask, alpha=0.5, cmap='Reds')
        plt.title('Error (Red)')
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        # Create and return visualization image
        error_colored = np.zeros_like(image)
        error_colored[error_mask > 0] = [255, 0, 0]  # Red for errors
        vis_image = cv2.addWeighted(image, 0.7, error_colored, 0.3, 0)
        
        return vis_image
    
    def visualize_bounding_boxes(self, image, boxes, labels=None, scores=None, save_path=None, show=True):
        """
        Visualize bounding boxes on the image.
        
        Args:
            image (np.ndarray): Input image
            boxes (list): List of bounding boxes in format [x1, y1, x2, y2]
            labels (list, optional): List of class labels for each box
            scores (list, optional): List of confidence scores for each box
            save_path (str, optional): Path to save the visualization
            show (bool): Whether to display the visualization
            
        Returns:
            np.ndarray: Visualization image
        """
        # Convert image to RGB if grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        # Create axis
        ax = plt.gca()
        
        # Plot each bounding box
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Get color based on class label
            if labels is not None and i < len(labels):
                color = np.array(self.colors[labels[i]]) / 255.0
            else:
                color = np.array(self.colors[0]) / 255.0
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label and score if available
            if labels is not None and i < len(labels):
                label_text = self.class_names[labels[i]]
                if scores is not None and i < len(scores):
                    label_text += f' {scores[i]:.2f}'
                
                plt.text(
                    x1, y1-5, label_text,
                    color='white', fontsize=10, 
                    bbox=dict(facecolor=color, alpha=0.8, pad=2)
                )
        
        plt.title('Bounding Box Visualization')
        plt.axis('off')
        
        # Save figure if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        # Create and return visualization image
        vis_image = image.copy()
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Get color based on class label
            if labels is not None and i < len(labels):
                color = self.colors[labels[i]]
            else:
                color = self.colors[0]
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label and score if available
            if labels is not None and i < len(labels):
                label_text = self.class_names[labels[i]]
                if scores is not None and i < len(scores):
                    label_text += f' {scores[i]:.2f}'
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw text background
                cv2.rectangle(
                    vis_image, 
                    (x1, y1-text_height-5), 
                    (x1+text_width, y1), 
                    color, 
                    -1
                )
                
                # Draw text
                cv2.putText(
                    vis_image, 
                    label_text, 
                    (x1, y1-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1
                )
        
        return vis_image
    
    def create_confusion_matrix_plot(self, confusion_matrix, save_path=None, show=True):
        """
        Create confusion matrix visualization.
        
        Args:
            confusion_matrix (np.ndarray): Confusion matrix of shape [num_classes, num_classes]
            save_path (str, optional): Path to save the visualization
            show (bool): Whether to display the visualization
            
        Returns:
            np.ndarray: Visualization image
        """
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_norm = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        # Plot confusion matrix
        plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add class labels
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)
        
        # Add text annotations
        thresh = cm_norm.max() / 2.0
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                plt.text(
                    j, i, f'{cm_norm[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > thresh else 'black'
                )
        
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save figure if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        # Convert plot to image
        fig = plt.gcf()
        fig.canvas.draw()
        vis_image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()
        
        return vis_image
    
    def create_metrics_summary(self, metrics, save_path=None, show=True):
        """
        Create metrics summary visualization.
        
        Args:
            metrics (dict): Dictionary of evaluation metrics
            save_path (str, optional): Path to save the visualization
            show (bool): Whether to display the visualization
            
        Returns:
            np.ndarray: Visualization image
        """
        # Extract metrics
        mean_iou = metrics.get('mean_iou', 0)
        mean_dice = metrics.get('mean_dice', 0)
        pixel_accuracy = metrics.get('pixel_accuracy', 0)
        class_ious = metrics.get('class_ious', np.zeros(len(self.class_names)))
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot overall metrics
        plt.subplot(1, 2, 1)
        overall_metrics = [mean_iou, mean_dice, pixel_accuracy]
        metric_names = ['Mean IoU', 'Mean Dice', 'Pixel Accuracy']
        
        plt.bar(metric_names, overall_metrics, color='skyblue')
        plt.title('Overall Metrics')
        plt.ylim(0, 1)
        
        # Add values on bars
        for i, v in enumerate(overall_metrics):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Plot per-class IoU
        plt.subplot(1, 2, 2)
        plt.bar(self.class_names, class_ious, color='lightgreen')
        plt.title('Per-Class IoU')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add values on bars
        for i, v in enumerate(class_ious):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Show figure if requested
        if show:
            plt.show()
        else:
            plt.close()
        
        # Convert plot to image
        fig = plt.gcf()
        fig.canvas.draw()
        vis_image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()
        
        return vis_image
    
    def create_report(self, image, pred_mask, gt_mask, metrics, save_dir=None):
        """
        Create a comprehensive evaluation report.
        
        Args:
            image (np.ndarray): Input image
            pred_mask (np.ndarray): Predicted segmentation mask
            gt_mask (np.ndarray): Ground truth segmentation mask
            metrics (dict): Dictionary of evaluation metrics
            save_dir (str, optional): Directory to save the report
            
        Returns:
            dict: Dictionary of visualization images
        """
        # Create visualizations
        comparison_vis = self.visualize_comparison(image, pred_mask, gt_mask, show=False)
        error_vis = self.visualize_error(image, pred_mask, gt_mask, show=False)
        metrics_vis = self.create_metrics_summary(metrics, show=False)
        confusion_matrix_vis = self.create_confusion_matrix_plot(metrics.get('confusion_matrix', np.zeros((len(self.class_names), len(self.class_names)))), show=False)
        
        # Save visualizations if directory is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(save_dir, 'comparison.png'), cv2.cvtColor(comparison_vis, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_dir, 'error.png'), cv2.cvtColor(error_vis, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_dir, 'metrics.png'), cv2.cvtColor(metrics_vis, cv2.COLOR_RGBA2BGR))
            cv2.imwrite(os.path.join(save_dir, 'confusion_matrix.png'), cv2.cvtColor(confusion_matrix_vis, cv2.COLOR_RGBA2BGR))
        
        # Return visualizations
        return {
            'comparison': comparison_vis,
            'error': error_vis,
            'metrics': metrics_vis,
            'confusion_matrix': confusion_matrix_vis
        }
