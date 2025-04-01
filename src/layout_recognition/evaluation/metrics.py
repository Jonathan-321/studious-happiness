#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for layout recognition.
"""

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def calculate_iou(pred_mask, gt_mask):
    """
    Calculate Intersection over Union (IoU) between predicted and ground truth masks.
    
    Args:
        pred_mask (torch.Tensor or np.ndarray): Predicted binary mask
        gt_mask (torch.Tensor or np.ndarray): Ground truth binary mask
        
    Returns:
        float: IoU score
    """
    # Convert to numpy if tensors
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    # Ensure binary masks
    pred_mask = pred_mask > 0.5
    gt_mask = gt_mask > 0.5
    
    # Calculate intersection and union
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    # Calculate IoU
    iou = intersection / (union + 1e-10)
    
    return iou

def calculate_dice(pred_mask, gt_mask):
    """
    Calculate Dice coefficient between predicted and ground truth masks.
    
    Args:
        pred_mask (torch.Tensor or np.ndarray): Predicted binary mask
        gt_mask (torch.Tensor or np.ndarray): Ground truth binary mask
        
    Returns:
        float: Dice coefficient
    """
    # Convert to numpy if tensors
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    # Ensure binary masks
    pred_mask = pred_mask > 0.5
    gt_mask = gt_mask > 0.5
    
    # Calculate intersection and sum
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()
    
    # Calculate Dice
    dice = 2 * intersection / (total + 1e-10)
    
    return dice

def calculate_precision_recall_f1(pred_mask, gt_mask):
    """
    Calculate precision, recall, and F1 score between predicted and ground truth masks.
    
    Args:
        pred_mask (torch.Tensor or np.ndarray): Predicted binary mask
        gt_mask (torch.Tensor or np.ndarray): Ground truth binary mask
        
    Returns:
        tuple: (precision, recall, f1)
    """
    # Convert to numpy if tensors
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    # Ensure binary masks
    pred_mask = pred_mask > 0.5
    gt_mask = gt_mask > 0.5
    
    # Flatten masks
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_flat, pred_flat, average='binary', zero_division=0
    )
    
    return precision, recall, f1

def calculate_mean_iou(pred_masks, gt_masks, num_classes):
    """
    Calculate mean IoU across all classes.
    
    Args:
        pred_masks (torch.Tensor or np.ndarray): Predicted masks of shape [N, C, H, W]
        gt_masks (torch.Tensor or np.ndarray): Ground truth masks of shape [N, C, H, W]
        num_classes (int): Number of classes
        
    Returns:
        tuple: (mean_iou, class_ious)
    """
    # Convert to numpy if tensors
    if isinstance(pred_masks, torch.Tensor):
        pred_masks = pred_masks.cpu().numpy()
    if isinstance(gt_masks, torch.Tensor):
        gt_masks = gt_masks.cpu().numpy()
    
    # Initialize IoU for each class
    class_ious = []
    
    # Calculate IoU for each class
    for c in range(num_classes):
        pred_c = pred_masks[:, c, :, :]
        gt_c = gt_masks[:, c, :, :]
        
        # Calculate IoU for this class
        iou_c = calculate_iou(pred_c, gt_c)
        class_ious.append(iou_c)
    
    # Calculate mean IoU
    mean_iou = np.mean(class_ious)
    
    return mean_iou, class_ious

def calculate_pixel_accuracy(pred_mask, gt_mask):
    """
    Calculate pixel accuracy between predicted and ground truth masks.
    
    Args:
        pred_mask (torch.Tensor or np.ndarray): Predicted mask
        gt_mask (torch.Tensor or np.ndarray): Ground truth mask
        
    Returns:
        float: Pixel accuracy
    """
    # Convert to numpy if tensors
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    # Ensure binary masks
    pred_mask = pred_mask > 0.5
    gt_mask = gt_mask > 0.5
    
    # Calculate accuracy
    correct = (pred_mask == gt_mask).sum()
    total = pred_mask.size
    
    accuracy = correct / total
    
    return accuracy

def calculate_confusion_matrix(pred_mask, gt_mask, num_classes):
    """
    Calculate confusion matrix for multi-class segmentation.
    
    Args:
        pred_mask (torch.Tensor or np.ndarray): Predicted mask of shape [H, W]
        gt_mask (torch.Tensor or np.ndarray): Ground truth mask of shape [H, W]
        num_classes (int): Number of classes
        
    Returns:
        np.ndarray: Confusion matrix of shape [num_classes, num_classes]
    """
    # Convert to numpy if tensors
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    # Flatten masks
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Calculate confusion matrix
    cm = confusion_matrix(gt_flat, pred_flat, labels=range(num_classes))
    
    return cm

def calculate_detection_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Calculate detection metrics (precision, recall, F1) for bounding boxes.
    
    Args:
        pred_boxes (list): List of predicted bounding boxes [x1, y1, x2, y2]
        gt_boxes (list): List of ground truth bounding boxes [x1, y1, x2, y2]
        iou_threshold (float): IoU threshold for considering a detection as correct
        
    Returns:
        tuple: (precision, recall, f1)
    """
    if not gt_boxes:
        return 0.0, 0.0, 0.0
    
    if not pred_boxes:
        return 0.0, 0.0, 0.0
    
    # Calculate IoU for each pair of boxes
    ious = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            ious[i, j] = calculate_box_iou(pred_box, gt_box)
    
    # Find matches
    matches = ious >= iou_threshold
    
    # Calculate metrics
    tp = np.sum(np.any(matches, axis=1))  # True positives
    fp = len(pred_boxes) - tp  # False positives
    fn = len(gt_boxes) - np.sum(np.any(matches, axis=0))  # False negatives
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    return precision, recall, f1

def calculate_box_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    
    Args:
        box1 (list or np.ndarray): First box [x1, y1, x2, y2]
        box2 (list or np.ndarray): Second box [x1, y1, x2, y2]
        
    Returns:
        float: IoU between the boxes
    """
    # Convert to numpy arrays
    box1 = np.array(box1)
    box2 = np.array(box2)
    
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0
    
    return iou

def calculate_map(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_thresholds=[0.5]):
    """
    Calculate mean Average Precision (mAP) for object detection.
    
    Args:
        pred_boxes (list): List of lists of predicted boxes for each image
        pred_scores (list): List of lists of confidence scores for each image
        pred_labels (list): List of lists of predicted class labels for each image
        gt_boxes (list): List of lists of ground truth boxes for each image
        gt_labels (list): List of lists of ground truth class labels for each image
        iou_thresholds (list): List of IoU thresholds for evaluation
        
    Returns:
        float: mAP score
    """
    aps = []
    
    # Calculate AP for each IoU threshold
    for iou_threshold in iou_thresholds:
        # Calculate AP for each class
        class_aps = []
        unique_classes = set()
        
        # Collect all unique classes
        for labels in gt_labels:
            unique_classes.update(labels)
        
        for cls in unique_classes:
            # Collect all predictions and ground truths for this class
            all_preds = []
            all_matched_gt = []
            
            for i in range(len(pred_boxes)):
                # Get predictions for this class in this image
                cls_indices = [j for j, label in enumerate(pred_labels[i]) if label == cls]
                cls_boxes = [pred_boxes[i][j] for j in cls_indices]
                cls_scores = [pred_scores[i][j] for j in cls_indices]
                
                # Get ground truths for this class in this image
                gt_indices = [j for j, label in enumerate(gt_labels[i]) if label == cls]
                cls_gt_boxes = [gt_boxes[i][j] for j in gt_indices]
                
                # Sort predictions by score
                sorted_indices = np.argsort(cls_scores)[::-1]
                cls_boxes = [cls_boxes[j] for j in sorted_indices]
                cls_scores = [cls_scores[j] for j in sorted_indices]
                
                # Mark ground truths as matched or not
                matched_gt = [False] * len(cls_gt_boxes)
                
                # Add each prediction to the list
                for box, score in zip(cls_boxes, cls_scores):
                    # Check if this prediction matches any ground truth
                    match_idx = -1
                    max_iou = iou_threshold
                    
                    for j, gt_box in enumerate(cls_gt_boxes):
                        if matched_gt[j]:
                            continue
                        
                        iou = calculate_box_iou(box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                            match_idx = j
                    
                    # If there's a match, mark it
                    if match_idx >= 0:
                        matched_gt[match_idx] = True
                        all_preds.append((score, True))
                    else:
                        all_preds.append((score, False))
                
                # Add unmatched ground truths to the count
                all_matched_gt.extend(matched_gt)
            
            # Calculate precision-recall curve
            all_preds.sort(reverse=True)
            tp = 0
            fp = 0
            precision_values = []
            recall_values = []
            
            for score, matched in all_preds:
                if matched:
                    tp += 1
                else:
                    fp += 1
                
                precision = tp / (tp + fp)
                recall = tp / sum(1 for m in all_matched_gt if m)
                
                precision_values.append(precision)
                recall_values.append(recall)
            
            # Calculate AP using precision-recall curve
            if not precision_values:
                ap = 0.0
            else:
                # Use all points interpolation
                ap = 0.0
                for r in np.arange(0, 1.1, 0.1):
                    prec_at_rec = [p for p, rec in zip(precision_values, recall_values) if rec >= r]
                    if prec_at_rec:
                        ap += max(prec_at_rec) / 11
            
            class_aps.append(ap)
        
        # Calculate mAP for this IoU threshold
        if class_aps:
            aps.append(np.mean(class_aps))
    
    # Calculate mAP across all IoU thresholds
    mAP = np.mean(aps) if aps else 0.0
    
    return mAP

class LayoutEvaluator:
    """Class for evaluating layout recognition models."""
    
    def __init__(self, num_classes=9):
        """
        Initialize the evaluator.
        
        Args:
            num_classes (int): Number of classes
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset evaluation metrics."""
        self.total_iou = 0.0
        self.total_dice = 0.0
        self.total_pixel_accuracy = 0.0
        self.class_ious = np.zeros(self.num_classes)
        self.class_dices = np.zeros(self.num_classes)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.num_samples = 0
    
    def update(self, pred_masks, gt_masks):
        """
        Update evaluation metrics with new predictions.
        
        Args:
            pred_masks (torch.Tensor or np.ndarray): Predicted masks of shape [N, C, H, W]
            gt_masks (torch.Tensor or np.ndarray): Ground truth masks of shape [N, C, H, W]
        """
        # Convert to numpy if tensors
        if isinstance(pred_masks, torch.Tensor):
            pred_masks = pred_masks.cpu().numpy()
        if isinstance(gt_masks, torch.Tensor):
            gt_masks = gt_masks.cpu().numpy()
        
        batch_size = pred_masks.shape[0]
        self.num_samples += batch_size
        
        # Calculate metrics for each sample
        for i in range(batch_size):
            pred_mask = pred_masks[i]
            gt_mask = gt_masks[i]
            
            # Calculate overall metrics
            iou = calculate_iou(pred_mask, gt_mask)
            dice = calculate_dice(pred_mask, gt_mask)
            pixel_accuracy = calculate_pixel_accuracy(pred_mask, gt_mask)
            
            self.total_iou += iou
            self.total_dice += dice
            self.total_pixel_accuracy += pixel_accuracy
            
            # Calculate per-class metrics
            for c in range(self.num_classes):
                pred_c = pred_mask[c]
                gt_c = gt_mask[c]
                
                iou_c = calculate_iou(pred_c, gt_c)
                dice_c = calculate_dice(pred_c, gt_c)
                
                self.class_ious[c] += iou_c
                self.class_dices[c] += dice_c
            
            # Update confusion matrix
            pred_class = np.argmax(pred_mask, axis=0)
            gt_class = np.argmax(gt_mask, axis=0)
            
            cm = confusion_matrix(
                gt_class.flatten(), pred_class.flatten(),
                labels=range(self.num_classes)
            )
            self.confusion_matrix += cm
    
    def get_results(self):
        """
        Get evaluation results.
        
        Returns:
            dict: Evaluation metrics
        """
        if self.num_samples == 0:
            return {
                'mean_iou': 0.0,
                'mean_dice': 0.0,
                'pixel_accuracy': 0.0,
                'class_ious': np.zeros(self.num_classes),
                'class_dices': np.zeros(self.num_classes),
                'confusion_matrix': self.confusion_matrix
            }
        
        # Calculate mean metrics
        mean_iou = self.total_iou / self.num_samples
        mean_dice = self.total_dice / self.num_samples
        pixel_accuracy = self.total_pixel_accuracy / self.num_samples
        
        # Calculate mean per-class metrics
        class_ious = self.class_ious / self.num_samples
        class_dices = self.class_dices / self.num_samples
        
        return {
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'pixel_accuracy': pixel_accuracy,
            'class_ious': class_ious,
            'class_dices': class_dices,
            'confusion_matrix': self.confusion_matrix
        }
