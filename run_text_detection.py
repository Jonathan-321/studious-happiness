#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced text detection script using ResNet34 model with post-processing refinements.

This script processes images using a pre-trained ResNet34 model to detect text regions.
It includes adaptive thresholding, post-processing refinements, and visualization capabilities.

Features:
- Adaptive thresholding based on image content
- Post-processing refinements to improve accuracy
- Option to focus on dark areas (text is usually dark on light background)
- Detailed visualizations and metrics for evaluation
- HTML report generation

Usage:
    python run_text_detection.py --input_path data/renaissance/val \
                               --output_dir results/text_detection \
                               --model_path models/renaissance/best_model.pth \
                               --device cpu \
                               --adaptive_threshold \
                               --refine \
                               --fill_holes
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
from scipy.ndimage import label, binary_fill_holes, zoom
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import models and utilities
from layout_recognition.models.renaissance_model import create_renaissance_model
from layout_recognition.postprocessing.refinement import refine_text_mask

def process_image(model, image_path, args):
    """
    Process a single image with the model.
    
    Args:
        model: The text detection model
        image_path: Path to the image
        args: Command line arguments
        
    Returns:
        dict: Results including mask, probabilities, and metrics
    """
    # Load and preprocess image
    image_tensor, original_image = model.preprocess_image(image_path)
    image_tensor = image_tensor.to(args.device)
    
    # Get original image size
    original_size = original_image.size
    
    # Run inference
    start_time = time.time()
    with torch.no_grad():
        output = model(image_tensor)
    inference_time = time.time() - start_time
    
    # Get prediction probabilities
    probs = torch.nn.functional.softmax(output, dim=1)
    text_prob = probs[0, 1].cpu().numpy()  # Class 1 is text
    
    # Apply adaptive thresholding if enabled
    if args.adaptive_threshold:
        # Use a moderate threshold for text detection
        # We can see from the probability map that text has higher values
        threshold = max(0.4, min(0.6, np.percentile(text_prob, 80)))
    else:
        threshold = args.threshold
    
    # Get binary prediction
    pred = (text_prob > threshold).astype(np.uint8)
    
    # Resize prediction and probability map to original size
    zoom_factors = (original_size[1] / pred.shape[0], original_size[0] / pred.shape[1])
    pred_resized = zoom(pred, zoom_factors, order=0)
    prob_resized = zoom(text_prob, zoom_factors, order=1)
    
    # Apply post-processing refinements
    if args.refine:
        # Convert PIL image to numpy array
        image_np = np.array(original_image)
        
        # Get grayscale image for analysis
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Basic morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        pred_refined = cv2.morphologyEx(pred_resized, cv2.MORPH_OPEN, kernel)
        pred_refined = cv2.morphologyEx(pred_refined, cv2.MORPH_CLOSE, kernel)
        
        # Remove very small regions
        labeled_mask, num_features = label(pred_refined)
        clean_mask = np.zeros_like(pred_refined)
        for i in range(1, num_features + 1):
            region_size = np.sum(labeled_mask == i)
            if region_size >= args.min_region_size:
                clean_mask[labeled_mask == i] = 1
        
        # Use a more balanced approach based on what we've seen in the images
        # The model is detecting text correctly but we need to refine it
        
        # Keep the original prediction but remove very small noise
        pred_refined = clean_mask
        
        # If we want to focus on dark areas (text is usually dark on light background)
        if args.focus_dark_areas:
            # Create a mask of dark areas with a moderate threshold
            _, dark_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Apply morphological operations to clean up the dark mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
            
            # Combine with our prediction - we want areas that are both
            # predicted as text AND are dark
            pred_refined = pred_refined * (dark_mask > 0).astype(np.uint8)
    else:
        pred_refined = pred_resized
    
    # Calculate metrics
    text_percentage = np.mean(pred_refined) * 100
    num_regions, labels = label(pred_refined)
    
    # Fill holes in text regions
    if args.fill_holes:
        pred_refined = binary_fill_holes(pred_refined).astype(np.uint8)
    
    # Store both original and resized predictions
    return {
        'prediction': pred_refined,  # Full size prediction
        'prediction_small': pred,     # Model output size
        'probability': prob_resized,  # Full size probability
        'probability_small': text_prob,  # Model output size
        'threshold': threshold,
        'inference_time': inference_time,
        'text_percentage': text_percentage,
        'num_regions': num_regions,
        'labels': labels
    }

def visualize_results(image_path, results, output_path):
    """
    Visualize the text detection results.
    
    Args:
        image_path: Path to original image
        results: Dictionary of results
        output_path: Path to save visualization
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Ensure prediction mask is the same size as the image
    mask = results['prediction']
    if mask.shape[:2] != image_np.shape[:2]:
        # Resize mask to match image dimensions
        mask = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Plot probability map (resized to match display)
    prob_display = cv2.resize(results['probability'], (512, 512), interpolation=cv2.INTER_LINEAR)
    im = axes[0, 1].imshow(prob_display, cmap='viridis')
    axes[0, 1].set_title(f'Text Probability Map\nThreshold: {results["threshold"]:.2f}')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Plot binary mask
    mask_display = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    axes[1, 0].imshow(mask_display, cmap='gray')
    axes[1, 0].set_title(f'Text Mask\nRegions: {results["num_regions"]}')
    axes[1, 0].axis('off')
    
    # Plot overlay
    overlay = image_np.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = [255, 0, 0]  # Red overlay for text
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f'Text Overlay\nText %: {results["text_percentage"]:.1f}%')
    axes[1, 1].axis('off')
    
    # Add metrics text
    plt.figtext(0.02, 0.02, f'Inference Time: {results["inference_time"]:.3f}s', 
                fontsize=10, ha='left')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading ResNet34 model...")
    model = create_renaissance_model(args.model_path, device)
    print("Model loaded successfully")
    
    # Get image paths
    if os.path.isfile(args.input_path):
        image_paths = [args.input_path]
    else:
        image_paths = []
        for root, _, files in os.walk(args.input_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images")
    
    # Process images
    all_results = []
    
    for i, image_path in enumerate(tqdm(image_paths)):
        # Process image
        results = process_image(model, image_path, args)
        results['image_path'] = image_path
        all_results.append(results)
        
        # Save visualization
        output_path = os.path.join(args.output_dir, f"result_{i:03d}.png")
        visualize_results(image_path, results, output_path)
    
    # Calculate and display average metrics
    avg_metrics = {
        'inference_time': np.mean([r['inference_time'] for r in all_results]),
        'text_percentage': np.mean([r['text_percentage'] for r in all_results]),
        'num_regions': np.mean([r['num_regions'] for r in all_results])
    }
    
    print("\nAverage Metrics:")
    for metric_name, value in avg_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Create HTML report
    create_html_report(all_results, avg_metrics, args.output_dir)
    
    print(f"Results saved to {args.output_dir}")

def create_html_report(results, avg_metrics, output_dir):
    """
    Create an HTML report of the results.
    
    Args:
        results: List of results for each image
        avg_metrics: Dictionary of average metrics
        output_dir: Directory to save the report
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Detection Results</title>
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
        .metrics {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 20px 0;
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
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            grid-gap: 20px;
            margin-top: 20px;
        }
        .gallery-item {
            background-color: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .gallery-item img {
            width: 100%;
            height: auto;
            border-radius: 3px;
        }
        .footer {
            margin-top: 40px;
            text-align: center;
            color: #666;
        }
    </style>
</head>
<body>
    <h1>Text Detection Results</h1>
    
    <div class="metrics">
        <h2>Average Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
    """
    
    # Add average metrics
    for metric_name, value in avg_metrics.items():
        html_content += f"""
            <tr>
                <td>{metric_name}</td>
                <td>{value:.4f}</td>
            </tr>
        """
    
    html_content += """
        </table>
    </div>
    
    <h2>Individual Results</h2>
    <div class="gallery">
    """
    
    # Add gallery items
    for i, result in enumerate(results):
        html_content += f"""
        <div class="gallery-item">
            <img src="result_{i:03d}.png" alt="Result {i+1}">
            <table>
                <tr>
                    <th>Image</th>
                    <td>{os.path.basename(result['image_path'])}</td>
                </tr>
                <tr>
                    <th>Inference Time</th>
                    <td>{result['inference_time']:.4f}s</td>
                </tr>
                <tr>
                    <th>Text Percentage</th>
                    <td>{result['text_percentage']:.2f}%</td>
                </tr>
                <tr>
                    <th>Text Regions</th>
                    <td>{result['num_regions']}</td>
                </tr>
            </table>
        </div>
        """
    
    html_content += """
    </div>
    
    <div class="footer">
        <p>Report generated on <span id="date"></span></p>
        <script>
            document.getElementById('date').textContent = new Date().toLocaleDateString();
        </script>
    </div>
</body>
</html>
    """
    
    # Write HTML file
    with open(os.path.join(output_dir, 'report.html'), 'w') as f:
        f.write(html_content)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced text detection using ResNet34')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='results/text_detection',
                        help='Directory to save results')
    parser.add_argument('--model_path', type=str, default='models/renaissance/best_model.pth',
                        help='Path to model weights')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run inference on (cpu or cuda)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for text detection')
    parser.add_argument('--adaptive_threshold', action='store_true',
                        help='Use adaptive thresholding based on image content')
    parser.add_argument('--refine', action='store_true',
                        help='Apply post-processing refinement')
    parser.add_argument('--min_region_size', type=int, default=100,
                        help='Minimum size of text regions to keep')
    parser.add_argument('--morph_kernel_size', type=int, default=2,
                        help='Kernel size for morphological operations')
    parser.add_argument('--fill_holes', action='store_true',
                        help='Fill holes in text regions')
    parser.add_argument('--focus_dark_areas', action='store_true',
                        help='Focus detection on dark areas (text is usually dark)')
    return parser.parse_args()

if __name__ == '__main__':
    main()
