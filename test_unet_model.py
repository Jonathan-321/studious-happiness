#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for U-Net Renaissance layout recognition model.
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

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import the model
from layout_recognition.models.unet_model import create_unet_model
from layout_recognition.postprocessing.refinement import refine_text_mask, enhance_text_regions

def test_model(args):
    """
    Test the U-Net model on a single image.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = create_unet_model(pretrained_path=args.model_path, device=device)
    model.eval()
    
    # Load image
    print(f"Processing image: {args.image_path}")
    image_tensor, image = model.preprocess_image(args.image_path)
    image_tensor = image_tensor.to(device)
    
    # Get original size
    original_image = Image.open(args.image_path).convert('RGB')
    original_size = original_image.size
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get prediction
    probs = torch.nn.functional.softmax(output, dim=1)
    text_prob = probs[0, 1].cpu().numpy()  # Class 1 is text
    
    # Apply adaptive thresholding based on the histogram of probabilities
    if args.adaptive_threshold:
        # Find a good threshold using Otsu's method or percentile
        threshold = max(0.3, min(0.7, np.percentile(text_prob, 70)))
        print(f"Using adaptive threshold: {threshold:.2f}")
    else:
        threshold = args.threshold
        print(f"Using fixed threshold: {threshold:.2f}")
    
    pred = (text_prob > threshold).astype(np.uint8)
    
    # Resize back to original size with higher quality interpolation
    zoom_factors = (original_size[1] / pred.shape[0], original_size[0] / pred.shape[1])
    pred_resized = zoom(pred, zoom_factors, order=0)
    prob_resized = zoom(text_prob, zoom_factors, order=1)
    
    # Apply post-processing refinement
    if args.refine:
        print("Applying post-processing refinement...")
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
    
    # Create color mask for visualization
    color_mask = np.zeros((pred_refined.shape[0], pred_refined.shape[1], 3), dtype=np.uint8)
    color_mask[pred_refined == 0] = [0, 0, 0]  # non-text - black
    color_mask[pred_refined == 1] = [255, 0, 0]  # text - red
    
    # Create overlay
    image_np = np.array(original_image)
    alpha = 0.5
    overlay = image_np.copy()
    overlay[pred_refined == 1] = (
        alpha * np.array([255, 0, 0]) + 
        (1 - alpha) * image_np[pred_refined == 1]
    ).astype(np.uint8)
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Text probability
    im = axes[1].imshow(prob_resized, cmap='viridis')
    axes[1].set_title('Text Probability')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Text mask
    axes[2].imshow(color_mask)
    axes[2].set_title('Text Mask')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', edgecolor='black', label='Non-text'),
        Patch(facecolor='red', edgecolor='red', label='Text')
    ]
    axes[2].legend(handles=legend_elements, loc='lower right')
    
    # Save result
    os.makedirs(args.output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'test_result.png'), dpi=150)
    print(f"Test complete! Result saved to {os.path.join(args.output_dir, 'test_result.png')}")
    
    # Save individual results
    np.save(os.path.join(args.output_dir, 'text_probability.npy'), prob_resized)
    np.save(os.path.join(args.output_dir, 'text_mask.npy'), pred_refined)
    
    # Save mask as image
    mask_image = Image.fromarray((pred_refined * 255).astype(np.uint8))
    mask_image.save(os.path.join(args.output_dir, 'text_mask.png'))
    
    # Save overlay as image
    overlay_image = Image.fromarray(overlay)
    overlay_image.save(os.path.join(args.output_dir, 'overlay.png'))

def parse_args():
    parser = argparse.ArgumentParser(description='Test U-Net Renaissance layout model')
    parser.add_argument('--model_path', type=str, default='models/unet/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--image_path', type=str, default='data/renaissance/val/1.png',
                        help='Path to test image')
    parser.add_argument('--output_dir', type=str, default='results/unet_test',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for text detection')
    parser.add_argument('--adaptive_threshold', action='store_true',
                        help='Use adaptive thresholding based on image content')
    parser.add_argument('--refine', action='store_true',
                        help='Apply post-processing refinement')
    parser.add_argument('--min_region_size', type=int, default=100,
                        help='Minimum size of text regions to keep')
    parser.add_argument('--morph_kernel_size', type=int, default=5,
                        help='Kernel size for morphological operations')
    return parser.parse_args()

def main():
    args = parse_args()
    test_model(args)

if __name__ == '__main__':
    main()
