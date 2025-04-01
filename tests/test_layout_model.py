#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for layout recognition model.
This script tests the layout recognition model on the Renaissance dataset.
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.layout_recognition.models.layoutlm import LayoutLM
from src.layout_recognition.data_processing.dataset import LayoutDataset
from src.layout_recognition.evaluation.metrics import calculate_iou, calculate_dice, calculate_precision_recall_f1
from src.layout_recognition.visualization.visualizer import LayoutVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test layout recognition model on Renaissance dataset')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Directory containing processed data')
    parser.add_argument('--layout_dir', type=str, default='layout_annotations',
                        help='Subdirectory containing layout annotations')
    parser.add_argument('--images_dir', type=str, default='images',
                        help='Subdirectory containing images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='results/layout',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for testing')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
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


def test_model(model, test_dataset, device, batch_size=8):
    """
    Test the model on a dataset.
    
    Args:
        model (nn.Module): Layout recognition model
        test_dataset (Dataset): Test dataset
        device (torch.device): Device to use
        batch_size (int): Batch size
        
    Returns:
        dict: Test results
    """
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Test model
    model.eval()
    all_ious = []
    all_dices = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_pred_masks = []
    all_true_masks = []
    all_images = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # Get batch
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()
            true_masks = masks.cpu().numpy()
            
            # Store images and masks for visualization
            all_images.extend(images.cpu().numpy())
            all_pred_masks.extend(pred_masks)
            all_true_masks.extend(true_masks)
            
            # Calculate metrics for each sample
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
    avg_iou = np.mean(all_ious)
    avg_dice = np.mean(all_dices)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)
    
    # Find worst cases
    worst_samples = []
    for i in range(len(all_ious)):
        if all_ious[i] < 0.5:  # Low IoU indicates poor performance
            worst_samples.append({
                'image_id': i,
                'iou': all_ious[i],
                'dice': all_dices[i],
                'precision': all_precisions[i],
                'recall': all_recalls[i],
                'f1': all_f1s[i]
            })
    
    # Sort worst samples by IoU
    worst_samples = sorted(worst_samples, key=lambda x: x['iou'])
    
    # Prepare results
    results = {
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
        },
        'images': all_images,
        'pred_masks': all_pred_masks,
        'true_masks': all_true_masks,
        'worst_samples': worst_samples
    }
    
    return results


def visualize_results(results, output_dir, num_samples=10):
    """
    Visualize test results.
    
    Args:
        results (dict): Test results
        output_dir (str): Directory to save visualizations
        num_samples (int): Number of samples to visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = LayoutVisualizer(output_dir=output_dir)
    
    # Visualize random samples
    indices = np.random.choice(len(results['images']), min(num_samples, len(results['images'])), replace=False)
    
    for i, idx in enumerate(indices):
        # Get sample
        image = results['images'][idx].transpose(1, 2, 0)
        true_mask = results['true_masks'][idx]
        pred_mask = results['pred_masks'][idx]
        
        # Visualize
        visualizer.visualize_prediction(
            image=image,
            true_mask=true_mask,
            pred_mask=pred_mask,
            save_path=os.path.join(output_dir, f'sample_{i+1}.png')
        )
    
    # Visualize worst cases
    for i, sample in enumerate(results['worst_samples'][:min(5, len(results['worst_samples']))]):
        # Get sample
        idx = sample['image_id']
        image = results['images'][idx].transpose(1, 2, 0)
        true_mask = results['true_masks'][idx]
        pred_mask = results['pred_masks'][idx]
        
        # Visualize
        visualizer.visualize_prediction(
            image=image,
            true_mask=true_mask,
            pred_mask=pred_mask,
            save_path=os.path.join(output_dir, f'worst_case_{i+1}.png')
        )
    
    # Visualize confusion matrix
    visualizer.visualize_confusion_matrix(
        true_masks=np.concatenate(results['true_masks']),
        pred_masks=np.concatenate(results['pred_masks']),
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Visualize metrics distribution
    plt.figure(figsize=(15, 10))
    
    # IoU distribution
    plt.subplot(2, 3, 1)
    plt.hist(results['per_sample']['iou'], bins=20)
    plt.xlabel('IoU')
    plt.ylabel('Count')
    plt.title(f'IoU Distribution (Avg: {results["iou"]:.4f})')
    
    # Dice distribution
    plt.subplot(2, 3, 2)
    plt.hist(results['per_sample']['dice'], bins=20)
    plt.xlabel('Dice')
    plt.ylabel('Count')
    plt.title(f'Dice Distribution (Avg: {results["dice"]:.4f})')
    
    # Precision distribution
    plt.subplot(2, 3, 3)
    plt.hist(results['per_sample']['precision'], bins=20)
    plt.xlabel('Precision')
    plt.ylabel('Count')
    plt.title(f'Precision Distribution (Avg: {results["precision"]:.4f})')
    
    # Recall distribution
    plt.subplot(2, 3, 4)
    plt.hist(results['per_sample']['recall'], bins=20)
    plt.xlabel('Recall')
    plt.ylabel('Count')
    plt.title(f'Recall Distribution (Avg: {results["recall"]:.4f})')
    
    # F1 distribution
    plt.subplot(2, 3, 5)
    plt.hist(results['per_sample']['f1'], bins=20)
    plt.xlabel('F1')
    plt.ylabel('Count')
    plt.title(f'F1 Distribution (Avg: {results["f1"]:.4f})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if layout annotations exist
    layout_dir = os.path.join(args.data_dir, args.layout_dir)
    images_dir = os.path.join(args.data_dir, args.images_dir)
    
    if not os.path.exists(layout_dir):
        print(f"Error: Layout annotations directory not found at {layout_dir}")
        print("Available directories in data/processed:")
        for item in os.listdir(args.data_dir):
            if os.path.isdir(os.path.join(args.data_dir, item)):
                print(f"  - {item}")
        return
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return
    
    # Create dummy test annotations for demonstration
    print("Note: Creating dummy test annotations for demonstration purposes")
    print("In a real scenario, you would load actual annotations from JSON files")
    
    # Find some images to use for testing
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"Error: No images found in {images_dir}")
        return
    
    # Create dummy annotations
    test_annotations = []
    for i, img_file in enumerate(image_files[:min(10, len(image_files))]):
        img_path = os.path.join(images_dir, img_file)
        try:
            # Get image dimensions
            with Image.open(img_path) as img:
                width, height = img.size
                
            # Create dummy annotation
            annotation = {
                'image_path': img_path,
                'width': width,
                'height': height,
                'regions': [
                    {'label': 'text', 'bbox': [0, 0, width//2, height//2]},
                    {'label': 'figure', 'bbox': [width//2, 0, width, height//2]},
                    {'label': 'table', 'bbox': [0, height//2, width//2, height]},
                    {'label': 'margin', 'bbox': [width//2, height//2, width, height]}
                ]
            }
            test_annotations.append(annotation)
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
    
    if not test_annotations:
        print("Error: Could not create test annotations")
        return
    
    # Create a simple dataset class for testing
    class SimpleLayoutDataset(torch.utils.data.Dataset):
        def __init__(self, annotations):
            self.annotations = annotations
            self.num_classes = 4  # text, figure, table, margin
            
        def __len__(self):
            return len(self.annotations)
            
        def __getitem__(self, idx):
            # Get annotation
            ann = self.annotations[idx]
            
            # Load image
            image = Image.open(ann['image_path']).convert('RGB')
            
            # Create mask
            mask = np.zeros((ann['height'], ann['width']), dtype=np.int64)
            
            # Fill mask with region labels
            for i, region in enumerate(ann['regions']):
                x1, y1, x2, y2 = region['bbox']
                label_idx = {'text': 1, 'figure': 2, 'table': 3, 'margin': 0}.get(region['label'], 0)
                mask[y1:y2, x1:x2] = label_idx
            
            # Convert to tensor
            image = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
            
            return {'image': image, 'mask': mask}
    
    # Create dataset
    test_dataset = SimpleLayoutDataset(test_annotations)
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    
    # Create model
    model = LayoutLM(
        input_channels=3,
        num_classes=test_dataset.num_classes,
        backbone='resnet50'
    )
    
    # Load weights
    model.load_state_dict(model_state_dict)
    model.to(device)
    
    # Test model
    print("Testing model...")
    results = test_model(
        model=model,
        test_dataset=test_dataset,
        device=device,
        batch_size=args.batch_size
    )
    
    # Print results
    print(f"IoU: {results['iou']:.4f}")
    print(f"Dice: {results['dice']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(
        results=results,
        output_dir=os.path.join(args.output_dir, 'visualizations'),
        num_samples=args.num_samples
    )
    
    # Save results
    results_json = {
        'iou': float(results['iou']),
        'dice': float(results['dice']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1': float(results['f1']),
        'per_sample': {
            'iou': [float(x) for x in results['per_sample']['iou']],
            'dice': [float(x) for x in results['per_sample']['dice']],
            'precision': [float(x) for x in results['per_sample']['precision']],
            'recall': [float(x) for x in results['per_sample']['recall']],
            'f1': [float(x) for x in results['per_sample']['f1']]
        },
        'worst_samples': [{
            'image_id': int(sample['image_id']),
            'iou': float(sample['iou']),
            'dice': float(sample['dice']),
            'precision': float(sample['precision']),
            'recall': float(sample['recall']),
            'f1': float(sample['f1'])
        } for sample in results['worst_samples']]
    }
    
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
