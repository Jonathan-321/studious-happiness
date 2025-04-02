#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Renaissance Layout Recognition Application

This script applies the layout recognition model specifically to the Renaissance dataset,
focusing on detecting main text regions and ignoring embellishments and marginalia.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from layout_recognition.models.layoutlm import LayoutLM
from layout_recognition.evaluation.metrics import calculate_iou, calculate_dice, calculate_precision_recall_f1

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Renaissance Layout Recognition Application')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                      help='Directory containing processed data')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='results/renaissance_layout',
                      help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for inference')
    return parser.parse_args()

def load_images(data_dir):
    """
    Load images from the data directory.
    
    Args:
        data_dir (str): Directory containing images
        
    Returns:
        list: List of image paths
    """
    image_dir = os.path.join(data_dir, 'layout', 'images')
    if not os.path.exists(image_dir):
        print(f"Warning: Image directory {image_dir} not found, using the data directory directly")
        image_dir = data_dir
    
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    return image_paths

def preprocess_image(image_path):
    """
    Preprocess image for model input.
    
    Args:
        image_path (str): Path to image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize to a standard size
    image = image.resize((512, 512))
    
    # Convert to tensor and normalize
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
    
    # Normalize with ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor

def run_inference(model, image_paths, device, batch_size=4):
    """
    Run inference on images.
    
    Args:
        model (nn.Module): Layout recognition model
        image_paths (list): List of image paths
        device (torch.device): Device to use for inference
        batch_size (int): Batch size for inference
        
    Returns:
        list: List of prediction results
    """
    model.eval()
    results = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc='Processing images'):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        # Preprocess images
        for image_path in batch_paths:
            image_tensor = preprocess_image(image_path)
            batch_images.append(image_tensor)
        
        # Stack images into batch
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(batch_tensor)
        
        # Process outputs
        _, preds = torch.max(outputs, dim=1)
        
        # Add results
        for j, image_path in enumerate(batch_paths):
            pred = preds[j].cpu().numpy()
            
            # Create result dictionary
            result = {
                'image_path': image_path,
                'prediction': pred,
                'text_regions': get_text_regions(pred),
                'original_size': Image.open(image_path).size
            }
            
            results.append(result)
    
    return results

def get_text_regions(prediction):
    """
    Extract text regions from prediction.
    
    Args:
        prediction (numpy.ndarray): Prediction mask
        
    Returns:
        list: List of text region bounding boxes
    """
    # In our model, class 1 represents text
    text_mask = (prediction == 1).astype(np.uint8)
    
    # For simplicity, we'll just get bounding boxes of connected components
    # In a production system, you might want to use more sophisticated methods
    from scipy.ndimage import label, find_objects
    
    # Label connected components
    labeled, num_features = label(text_mask)
    
    # Extract bounding boxes
    text_regions = []
    for i in range(1, num_features + 1):
        obj = find_objects(labeled == i)[0]
        y_start, y_end = obj[0].start, obj[0].stop
        x_start, x_end = obj[1].start, obj[1].stop
        
        # Add region
        text_regions.append({
            'bbox': [x_start, y_start, x_end, y_end],
            'label': 'text'
        })
    
    return text_regions

def visualize_results(results, output_dir):
    """
    Visualize layout recognition results.
    
    Args:
        results (list): List of prediction results
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize each result
    for i, result in enumerate(results):
        # Load original image
        image = Image.open(result['image_path'])
        image = image.resize((512, 512))
        image_np = np.array(image)
        
        # Create visualization
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        ax[0].imshow(image_np)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        # Prediction with text regions highlighted
        ax[1].imshow(image_np)
        for region in result['text_regions']:
            x1, y1, x2, y2 = region['bbox']
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                 linewidth=2, edgecolor='r', facecolor='none')
            ax[1].add_patch(rect)
        ax[1].set_title('Text Regions')
        ax[1].axis('off')
        
        # Set title for the figure
        fig.suptitle(f"Layout Recognition: {os.path.basename(result['image_path'])}")
        
        # Save figure
        plt.savefig(os.path.join(vis_dir, f'result_{i:03d}.png'), bbox_inches='tight')
        plt.close()
    
    # Create summary visualization
    num_images = min(5, len(results))
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 4 * num_images))
    
    for i in range(num_images):
        # Load original image
        image = Image.open(results[i]['image_path'])
        image = image.resize((512, 512))
        image_np = np.array(image)
        
        # Original image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # Prediction with text regions highlighted
        axes[i, 1].imshow(image_np)
        for region in results[i]['text_regions']:
            x1, y1, x2, y2 = region['bbox']
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                 linewidth=2, edgecolor='r', facecolor='none')
            axes[i, 1].add_patch(rect)
        axes[i, 1].set_title(f'Text Regions {i+1}')
        axes[i, 1].axis('off')
    
    # Save summary
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'summary.png'), bbox_inches='tight')
    plt.close()

def save_results(results, output_dir):
    """
    Save layout recognition results.
    
    Args:
        results (list): List of prediction results
        output_dir (str): Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to JSON
    results_data = []
    for result in results:
        # Convert numpy arrays to lists for JSON serialization
        result_copy = result.copy()
        result_copy['prediction'] = result_copy['prediction'].tolist()
        results_data.append(result_copy)
    
    with open(os.path.join(output_dir, 'layout_results.json'), 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Count text regions
    total_regions = sum(len(r['text_regions']) for r in results)
    avg_regions = total_regions / len(results) if results else 0
    
    # Generate report
    report = {
        'num_images': len(results),
        'total_text_regions': total_regions,
        'avg_text_regions_per_image': avg_regions
    }
    
    with open(os.path.join(output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create a text report
    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write('Renaissance Layout Recognition Report\n')
        f.write('====================================\n\n')
        f.write(f'Total images processed: {len(results)}\n')
        f.write(f'Total text regions detected: {total_regions}\n')
        f.write(f'Average text regions per image: {avg_regions:.2f}\n\n')
        f.write('Note: This layout recognition model focuses on detecting main text regions\n')
        f.write('in early modern printed sources, ignoring marginalia and embellishments.\n')

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load images
    image_paths = load_images(args.data_dir)
    
    if not image_paths:
        print(f"Error: No images found in {args.data_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Load model
    print("Loading model...")
    model = LayoutLM(
        input_channels=3,
        num_classes=4,  # background, text, figure, margin
        backbone='resnet34'
    )
    
    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    
    model.load_state_dict(model_state_dict)
    model.to(device)
    
    # Run inference
    print("Running inference...")
    results = run_inference(
        model=model,
        image_paths=image_paths,
        device=device,
        batch_size=args.batch_size
    )
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(results, args.output_dir)
    
    # Save results
    print("Saving results...")
    save_results(results, args.output_dir)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
