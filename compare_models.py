#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compare the performance of different layout recognition models.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import cv2
from tqdm import tqdm
from scipy.ndimage import zoom
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import models
from layout_recognition.models.renaissance_model import create_renaissance_model
from layout_recognition.models.unet_model import create_unet_model
try:
    from layout_recognition.models.layoutlmv3_model import create_layoutlmv3_model
    LAYOUTLMV3_AVAILABLE = True
except ImportError:
    print("LayoutLMv3 model not available, skipping...")
    LAYOUTLMV3_AVAILABLE = False

# Import post-processing utilities
from layout_recognition.postprocessing.refinement import refine_text_mask

def load_models(args):
    """
    Load the specified models.
    
    Args:
        args: Command line arguments
        
    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    device = torch.device(args.device)
    
    # Load ResNet34 model if specified
    if 'resnet34' in args.models:
        try:
            print("Loading ResNet34 model...")
            models['resnet34'] = create_renaissance_model(
                pretrained_path=args.resnet34_path,
                device=device
            )
            print("ResNet34 model loaded successfully")
        except Exception as e:
            print(f"Error loading ResNet34 model: {e}")
    
    # Load U-Net model if specified
    if 'unet' in args.models:
        try:
            print("Loading U-Net model...")
            models['unet'] = create_unet_model(
                pretrained_path=args.unet_path,
                device=device
            )
            print("U-Net model loaded successfully")
        except Exception as e:
            print(f"Error loading U-Net model: {e}")
    
    # Load LayoutLMv3 model if specified and available
    if 'layoutlmv3' in args.models and LAYOUTLMV3_AVAILABLE:
        try:
            print("Loading LayoutLMv3 model...")
            models['layoutlmv3'] = create_layoutlmv3_model(
                pretrained_path=args.layoutlmv3_path,
                device=device
            )
            print("LayoutLMv3 model loaded successfully")
        except Exception as e:
            print(f"Error loading LayoutLMv3 model: {e}")
    
    return models

def process_image(image_path, models, device, args):
    """
    Process an image with multiple models.
    
    Args:
        image_path: Path to image
        models: Dictionary of models
        device: Device to run inference on
        args: Command line arguments
        
    Returns:
        dict: Dictionary of results for each model
    """
    results = {}
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size
    
    # Process with each model
    for model_name, model in models.items():
        print(f"Processing with {model_name} model...")
        start_time = time.time()
        
        # Preprocess image
        image_tensor, _ = model.preprocess_image(image_path)
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
        
        # Get prediction
        probs = torch.nn.functional.softmax(output, dim=1)
        text_prob = probs[0, 1].cpu().numpy()  # Class 1 is text
        
        # Apply adaptive thresholding based on the histogram of probabilities
        if args.adaptive_threshold:
            # Find a good threshold using percentile method
            threshold = max(0.3, min(0.7, np.percentile(text_prob, 70)))
        else:
            threshold = args.threshold
        
        pred = (text_prob > threshold).astype(np.uint8)
        
        # Resize back to original size with higher quality interpolation
        zoom_factors = (original_size[1] / pred.shape[0], original_size[0] / pred.shape[1])
        pred_resized = zoom(pred, zoom_factors, order=0)
        prob_resized = zoom(text_prob, zoom_factors, order=1)
        
        # Apply post-processing refinement
        if args.refine:
            # Convert PIL image to numpy array for OpenCV
            image_np = np.array(original_image)
            
            # Refine the mask
            pred_refined = refine_text_mask(
                pred_resized, 
                original_image=image_np,
                min_region_size=args.min_region_size,
                morph_kernel_size=args.morph_kernel_size
            )
        else:
            pred_refined = pred_resized
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Store results
        results[model_name] = {
            'prediction': pred_refined,
            'probability': prob_resized,
            'threshold': threshold,
            'inference_time': inference_time
        }
    
    return results, original_image

def visualize_comparison(results, original_image, output_path):
    """
    Visualize the comparison of different models.
    
    Args:
        results: Dictionary of results for each model
        original_image: Original image
        output_path: Path to save visualization
    """
    # Convert original image to numpy array
    image_np = np.array(original_image)
    
    # Create figure
    n_models = len(results)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(5 * (n_models + 1), 10))
    
    # Plot original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Empty plot for alignment
    axes[1, 0].axis('off')
    
    # Plot results for each model
    for i, (model_name, result) in enumerate(results.items()):
        # Plot probability map
        im = axes[0, i + 1].imshow(result['probability'], cmap='viridis')
        axes[0, i + 1].set_title(f'{model_name} Probability\nThreshold: {result["threshold"]:.2f}')
        axes[0, i + 1].axis('off')
        plt.colorbar(im, ax=axes[0, i + 1])
        
        # Plot mask
        axes[1, i + 1].imshow(result['prediction'], cmap='gray')
        axes[1, i + 1].set_title(f'{model_name} Mask\nInference Time: {result["inference_time"]:.2f}s')
        axes[1, i + 1].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def calculate_metrics(results, ground_truth=None):
    """
    Calculate metrics for each model.
    
    Args:
        results: Dictionary of results for each model
        ground_truth: Ground truth mask (optional)
        
    Returns:
        dict: Dictionary of metrics for each model
    """
    metrics = {}
    
    # Calculate metrics for each model
    for model_name, result in results.items():
        # Initialize metrics
        metrics[model_name] = {
            'inference_time': result['inference_time'],
            'text_percentage': np.mean(result['prediction']) * 100,
        }
        
        # Calculate IoU with ground truth if available
        if ground_truth is not None:
            intersection = np.logical_and(result['prediction'], ground_truth)
            union = np.logical_or(result['prediction'], ground_truth)
            iou = np.sum(intersection) / np.sum(union)
            metrics[model_name]['iou'] = iou
    
    return metrics

def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    models = load_models(args)
    
    if not models:
        print("No models loaded, exiting...")
        return
    
    # Get image paths
    image_paths = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images")
    
    # Process images
    all_metrics = []
    
    for i, image_path in enumerate(tqdm(image_paths)):
        # Process image with all models
        results, original_image = process_image(image_path, models, device, args)
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        metrics['image_path'] = image_path
        all_metrics.append(metrics)
        
        # Visualize comparison
        output_path = os.path.join(args.output_dir, f"comparison_{i:03d}.png")
        visualize_comparison(results, original_image, output_path)
    
    # Calculate average metrics
    avg_metrics = {}
    for model_name in models.keys():
        avg_metrics[model_name] = {
            'avg_inference_time': np.mean([m[model_name]['inference_time'] for m in all_metrics]),
            'avg_text_percentage': np.mean([m[model_name]['text_percentage'] for m in all_metrics]),
        }
    
    # Print average metrics
    print("\nAverage Metrics:")
    for model_name, metrics in avg_metrics.items():
        print(f"  {model_name}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")
    
    # Create comparison table
    create_comparison_table(all_metrics, avg_metrics, os.path.join(args.output_dir, "comparison_table.html"))
    
    # Create comparison gallery
    create_comparison_gallery(os.path.join(args.output_dir, "comparison_gallery.html"), len(image_paths))
    
    print(f"Results saved to {args.output_dir}")

def create_comparison_table(all_metrics, avg_metrics, output_path):
    """
    Create an HTML table with comparison metrics.
    
    Args:
        all_metrics: List of metrics for each image
        avg_metrics: Dictionary of average metrics for each model
        output_path: Path to save HTML file
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison - Metrics</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2 {
            color: #333;
            text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
        }
        th, td {
            padding: 10px;
            border: 1px solid #ddd;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f0f0f0;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>Model Comparison - Metrics</h1>
    
    <h2>Average Metrics</h2>
    <table>
        <tr>
            <th>Model</th>
            <th>Average Inference Time (s)</th>
            <th>Average Text Percentage (%)</th>
        </tr>
    """
    
    # Add average metrics
    for model_name, metrics in avg_metrics.items():
        html_content += f"""
        <tr>
            <td>{model_name}</td>
            <td>{metrics['avg_inference_time']:.4f}</td>
            <td>{metrics['avg_text_percentage']:.2f}</td>
        </tr>
        """
    
    html_content += """
    </table>
    
    <h2>Individual Image Metrics</h2>
    <table>
        <tr>
            <th>Image</th>
    """
    
    # Add model headers
    for model_name in avg_metrics.keys():
        html_content += f"""
            <th colspan="2">{model_name}</th>
        """
    
    html_content += """
        </tr>
        <tr>
            <th>Path</th>
    """
    
    # Add metric headers
    for _ in avg_metrics.keys():
        html_content += """
            <th>Inference Time (s)</th>
            <th>Text Percentage (%)</th>
        """
    
    html_content += """
        </tr>
    """
    
    # Add metrics for each image
    for metrics in all_metrics:
        html_content += f"""
        <tr>
            <td>{os.path.basename(metrics['image_path'])}</td>
        """
        
        for model_name in avg_metrics.keys():
            html_content += f"""
            <td>{metrics[model_name]['inference_time']:.4f}</td>
            <td>{metrics[model_name]['text_percentage']:.2f}</td>
            """
        
        html_content += """
        </tr>
        """
    
    html_content += """
    </table>
    
    <div class="footer">
        <p>Model Comparison - Generated on <span id="date"></span></p>
        <script>
            document.getElementById('date').textContent = new Date().toLocaleDateString();
        </script>
    </div>
</body>
</html>
    """
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

def create_comparison_gallery(output_path, num_images):
    """
    Create an HTML gallery with comparison visualizations.
    
    Args:
        output_path: Path to save HTML file
        num_images: Number of images
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison - Gallery</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2 {
            color: #333;
            text-align: center;
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(600px, 1fr));
            grid-gap: 20px;
            margin-top: 20px;
        }
        .gallery-item {
            background-color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 5px;
            overflow: hidden;
            transition: transform 0.3s;
        }
        .gallery-item:hover {
            transform: scale(1.02);
        }
        .gallery-item img {
            width: 100%;
            height: auto;
            display: block;
        }
        .gallery-item .caption {
            padding: 10px;
            text-align: center;
            font-size: 14px;
            color: #666;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>Model Comparison - Gallery</h1>
    <p style="text-align: center;">This page shows the comparison of different layout recognition models.</p>
    
    <div class="gallery" id="gallery">
    """
    
    # Add gallery items
    for i in range(num_images):
        html_content += f"""
        <div class="gallery-item">
            <img src="comparison_{i:03d}.png" alt="Comparison {i+1}">
            <div class="caption">Comparison {i+1}</div>
        </div>
        """
    
    html_content += """
    </div>
    
    <div class="footer">
        <p>Model Comparison - Generated on <span id="date"></span></p>
        <script>
            document.getElementById('date').textContent = new Date().toLocaleDateString();
        </script>
    </div>
</body>
</html>
    """
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare different layout recognition models')
    parser.add_argument('--input_dir', type=str, default='data/renaissance/val',
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='results/model_comparison',
                        help='Directory to save results')
    parser.add_argument('--models', nargs='+', default=['resnet34', 'unet', 'layoutlmv3'],
                        help='Models to compare (resnet34, unet, layoutlmv3)')
    parser.add_argument('--resnet34_path', type=str, default='models/renaissance/best_model.pth',
                        help='Path to ResNet34 model weights')
    parser.add_argument('--unet_path', type=str, default='models/unet/best_model.pth',
                        help='Path to U-Net model weights')
    parser.add_argument('--layoutlmv3_path', type=str, default='models/layoutlmv3/best_model.pth',
                        help='Path to LayoutLMv3 model weights')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for text detection')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--adaptive_threshold', action='store_true',
                        help='Use adaptive thresholding based on image content')
    parser.add_argument('--refine', action='store_true',
                        help='Apply post-processing refinement')
    parser.add_argument('--min_region_size', type=int, default=100,
                        help='Minimum size of text regions to keep')
    parser.add_argument('--morph_kernel_size', type=int, default=5,
                        help='Kernel size for morphological operations')
    return parser.parse_args()

if __name__ == '__main__':
    main()
