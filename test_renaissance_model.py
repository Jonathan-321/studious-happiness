#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Renaissance layout model.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import cv2

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import the model and post-processing utilities
from layout_recognition.models.renaissance_model import RenaissanceLayoutModel
from layout_recognition.postprocessing import refine_text_mask, enhance_text_regions

def test_model(args):
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create model
    model = RenaissanceLayoutModel(input_channels=3, backbone='resnet34')
    
    # Load model weights
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load a test image
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return
    
    # Load image
    image = Image.open(args.image_path).convert('RGB')
    original_size = image.size
    
    # Resize to a standard size
    image_resized = image.resize((512, 512))
    
    # Convert to tensor and normalize
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
    
    # Normalize with ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get prediction
    probs = torch.nn.functional.softmax(output, dim=1)
    text_prob = probs[0, 1].cpu().numpy()  # Class 1 is text
    
    # Apply adaptive thresholding based on the histogram of probabilities
    if args.adaptive_threshold:
        # Find a good threshold using Otsu's method
        threshold = max(0.3, min(0.7, np.percentile(text_prob, 70)))
        print(f"Using adaptive threshold: {threshold:.2f}")
    else:
        threshold = args.threshold
        print(f"Using fixed threshold: {threshold:.2f}")
    
    pred = (text_prob > threshold).astype(np.uint8)
    
    # Resize back to original size with higher quality interpolation
    from scipy.ndimage import zoom
    zoom_factors = (original_size[1] / pred.shape[0], original_size[0] / pred.shape[1])
    pred_resized = zoom(pred, zoom_factors, order=0)
    prob_resized = zoom(text_prob, zoom_factors, order=1)
    
    # Apply post-processing refinement
    if args.refine:
        print("Applying post-processing refinement...")
        # Convert PIL image to numpy array for OpenCV
        image_np = np.array(image)
        
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
    image_np_full = np.array(image)
    alpha = 0.5
    overlay = image_np_full.copy()
    overlay[pred_refined == 1] = (
        alpha * np.array([255, 0, 0]) + 
        (1 - alpha) * image_np_full[pred_refined == 1]
    ).astype(np.uint8)
    
    # Visualize
    plt.figure(figsize=(16, 4))
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Text probability
    plt.subplot(1, 4, 2)
    im = plt.imshow(prob_resized, cmap='viridis', vmin=0, vmax=1)
    plt.title('Text Probability')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    
    # Mask
    plt.subplot(1, 4, 3)
    plt.imshow(color_mask)
    plt.title('Text Mask')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 4, 4)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', label='Non-text'),
        Patch(facecolor='red', label='Text')
    ]
    plt.legend(handles=legend_elements, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.2))
    
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, 'test_result.png')
    plt.savefig(output_path, bbox_inches='tight')
    
    print(f"Test complete! Result saved to {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Test Renaissance layout model')
    parser.add_argument('--model_path', type=str, default='models/renaissance/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--image_path', type=str, default='data/renaissance/val/1.png',
                        help='Path to test image')
    parser.add_argument('--output_dir', type=str, default='results/test',
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
