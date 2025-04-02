#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper script to run the Renaissance layout recognition application with U-Net.
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from scipy.ndimage import zoom

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import the model and post-processing utilities
from layout_recognition.models.unet_model import create_unet_model
from layout_recognition.postprocessing.refinement import refine_text_mask, enhance_text_regions

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Renaissance layout recognition with U-Net')
    parser.add_argument('--input_dir', type=str, default='data/renaissance/val',
                        help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='results/unet_recognition',
                        help='Directory to save results')
    parser.add_argument('--model_path', type=str, default='models/unet/best_model.pth',
                        help='Path to pretrained model')
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

def process_image(image_path, model, device, args):
    """Process a single image."""
    # Load and preprocess image
    image_tensor, image = model.preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Get original size
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size
    
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
    
    return {
        'image': original_image,
        'prediction': pred_refined,
        'probability': prob_resized,
        'threshold': threshold
    }

def visualize_result(result, output_path):
    """Visualize the result."""
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(np.array(result['image']))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot text probability
    im = axes[1].imshow(result['probability'], cmap='viridis')
    axes[1].set_title('Text Probability')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Plot text mask
    axes[2].imshow(result['prediction'], cmap='gray')
    axes[2].set_title('Text Mask')
    axes[2].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # Load model
    print("Loading model...")
    try:
        model = create_unet_model(pretrained_path=args.model_path, device=device)
        print("Loaded U-Net model successfully")
    except Exception as e:
        print(f"Error loading U-Net model: {e}")
        print("Exiting...")
        return
    
    # Get image paths
    image_paths = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images")
    
    # Process images
    results = []
    for i, image_path in enumerate(tqdm(image_paths)):
        result = process_image(image_path, model, device, args)
        result['path'] = image_path
        results.append(result)
        
        # Save results
        filename = f"result_{i:03d}"
        
        # Save mask
        mask_path = os.path.join(args.output_dir, 'masks', f"{filename}_mask.png")
        mask_image = Image.fromarray((result['prediction'] * 255).astype(np.uint8))
        mask_image.save(mask_path)
        
        # Save visualization
        visualize_result(result, os.path.join(args.output_dir, 'visualizations', f"{filename}.png"))
    
    # Create an HTML page to view results
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>U-Net Recognition Results</title>
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
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
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
    <h1>U-Net Recognition Results</h1>
    <p style="text-align: center;">This page shows the results of the U-Net model on the Renaissance dataset.</p>
    
    <h2>Visualizations</h2>
    <div class="gallery" id="gallery">
    """
    
    # Add gallery items
    for i in range(len(results)):
        filename = f"result_{i:03d}"
        image_path = os.path.join('visualizations', f"{filename}.png")
        html_content += f"""
        <div class="gallery-item">
            <img src="{image_path}" alt="Result {i+1}">
            <div class="caption">Result {i+1}</div>
        </div>
        """
    
    # Close HTML
    html_content += """
    </div>
    
    <div class="footer">
        <p>U-Net Recognition - Generated on <span id="date"></span></p>
        <script>
            document.getElementById('date').textContent = new Date().toLocaleDateString();
        </script>
    </div>
</body>
</html>
    """
    
    # Write HTML file
    with open(os.path.join(args.output_dir, 'index.html'), 'w') as f:
        f.write(html_content)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
