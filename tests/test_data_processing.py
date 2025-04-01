#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for Renaissance dataset processing pipeline.
This script tests the data processing pipeline for the Renaissance dataset.
"""

import os
import sys
import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.scripts.process_renaissance_dataset import process_dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Renaissance dataset processing pipeline')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                        help='Directory containing raw data')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory to save processed data')
    parser.add_argument('--sample_size', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize processed samples')
    return parser.parse_args()


def check_dataset_structure(output_dir):
    """
    Check if the dataset structure is correct.
    
    Args:
        output_dir (str): Directory containing processed data
        
    Returns:
        bool: True if structure is correct, False otherwise
    """
    # Check if directories exist
    ocr_dir = os.path.join(output_dir, 'ocr')
    layout_dir = os.path.join(output_dir, 'layout')
    text_images_dir = os.path.join(output_dir, 'text_images')
    
    if not os.path.exists(ocr_dir):
        print(f"ERROR: OCR directory {ocr_dir} does not exist")
        return False
    
    if not os.path.exists(layout_dir):
        print(f"ERROR: Layout directory {layout_dir} does not exist")
        return False
    
    if not os.path.exists(text_images_dir):
        print(f"ERROR: Text images directory {text_images_dir} does not exist")
        return False
    
    # Check if annotation files exist
    for dataset_type in ['train', 'val', 'test']:
        ocr_annotations = os.path.join(ocr_dir, f'{dataset_type}_annotations.json')
        layout_annotations = os.path.join(layout_dir, f'{dataset_type}_annotations.json')
        text_images_annotations = os.path.join(text_images_dir, f'{dataset_type}_annotations.json')
        
        if not os.path.exists(ocr_annotations):
            print(f"ERROR: OCR annotations {ocr_annotations} do not exist")
            return False
        
        if not os.path.exists(layout_annotations):
            print(f"ERROR: Layout annotations {layout_annotations} do not exist")
            return False
        
        if not os.path.exists(text_images_annotations):
            print(f"ERROR: Text images annotations {text_images_annotations} do not exist")
            return False
    
    return True


def check_annotation_format(output_dir):
    """
    Check if the annotation format is correct.
    
    Args:
        output_dir (str): Directory containing processed data
        
    Returns:
        bool: True if format is correct, False otherwise
    """
    # Check OCR annotations
    ocr_dir = os.path.join(output_dir, 'ocr')
    for dataset_type in ['train', 'val', 'test']:
        annotations_path = os.path.join(ocr_dir, f'{dataset_type}_annotations.json')
        
        try:
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
            
            if not isinstance(annotations, list):
                print(f"ERROR: OCR annotations {annotations_path} should be a list")
                return False
            
            if len(annotations) == 0:
                print(f"WARNING: OCR annotations {annotations_path} is empty")
                continue
            
            # Check first annotation
            first_anno = annotations[0]
            required_keys = ['image_path', 'text']
            for key in required_keys:
                if key not in first_anno:
                    print(f"ERROR: OCR annotation missing required key '{key}'")
                    return False
            
            # Check if image exists
            image_path = first_anno['image_path']
            if not os.path.exists(image_path):
                print(f"ERROR: Image {image_path} does not exist")
                return False
        
        except Exception as e:
            print(f"ERROR: Failed to load OCR annotations {annotations_path}: {e}")
            return False
    
    # Check layout annotations
    layout_dir = os.path.join(output_dir, 'layout')
    for dataset_type in ['train', 'val', 'test']:
        annotations_path = os.path.join(layout_dir, f'{dataset_type}_annotations.json')
        
        try:
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
            
            if not isinstance(annotations, list):
                print(f"ERROR: Layout annotations {annotations_path} should be a list")
                return False
            
            if len(annotations) == 0:
                print(f"WARNING: Layout annotations {annotations_path} is empty")
                continue
            
            # Check first annotation
            first_anno = annotations[0]
            required_keys = ['image_path', 'regions']
            for key in required_keys:
                if key not in first_anno:
                    print(f"ERROR: Layout annotation missing required key '{key}'")
                    return False
            
            # Check if image exists
            image_path = first_anno['image_path']
            if not os.path.exists(image_path):
                print(f"ERROR: Image {image_path} does not exist")
                return False
            
            # Check regions
            regions = first_anno['regions']
            if not isinstance(regions, list):
                print(f"ERROR: Regions should be a list")
                return False
            
            if len(regions) > 0:
                first_region = regions[0]
                region_keys = ['type', 'bbox']
                for key in region_keys:
                    if key not in first_region:
                        print(f"ERROR: Region missing required key '{key}'")
                        return False
        
        except Exception as e:
            print(f"ERROR: Failed to load layout annotations {annotations_path}: {e}")
            return False
    
    # Check text images annotations
    text_images_dir = os.path.join(output_dir, 'text_images')
    for dataset_type in ['train', 'val', 'test']:
        annotations_path = os.path.join(text_images_dir, f'{dataset_type}_annotations.json')
        
        try:
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)
            
            if not isinstance(annotations, list):
                print(f"ERROR: Text images annotations {annotations_path} should be a list")
                return False
            
            if len(annotations) == 0:
                print(f"WARNING: Text images annotations {annotations_path} is empty")
                continue
            
            # Check first annotation
            first_anno = annotations[0]
            required_keys = ['image_path', 'text']
            for key in required_keys:
                if key not in first_anno:
                    print(f"ERROR: Text image annotation missing required key '{key}'")
                    return False
            
            # Check if image exists
            image_path = first_anno['image_path']
            if not os.path.exists(image_path):
                print(f"ERROR: Image {image_path} does not exist")
                return False
        
        except Exception as e:
            print(f"ERROR: Failed to load text images annotations {annotations_path}: {e}")
            return False
    
    return True


def visualize_samples(output_dir, sample_size=5):
    """
    Visualize processed samples.
    
    Args:
        output_dir (str): Directory containing processed data
        sample_size (int): Number of samples to visualize
    """
    # Create output directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize OCR samples
    ocr_dir = os.path.join(output_dir, 'ocr')
    train_annotations_path = os.path.join(ocr_dir, 'train_annotations.json')
    
    with open(train_annotations_path, 'r') as f:
        ocr_annotations = json.load(f)
    
    # Sample random annotations
    if len(ocr_annotations) > sample_size:
        import random
        ocr_samples = random.sample(ocr_annotations, sample_size)
    else:
        ocr_samples = ocr_annotations
    
    # Visualize OCR samples
    fig, axes = plt.subplots(len(ocr_samples), 1, figsize=(10, 4 * len(ocr_samples)))
    if len(ocr_samples) == 1:
        axes = [axes]
    
    for i, anno in enumerate(ocr_samples):
        image_path = anno['image_path']
        text = anno['text']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Display image and text
        axes[i].imshow(image)
        axes[i].set_title(f"OCR Text: {text}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'ocr_samples.png'))
    plt.close()
    
    # Visualize layout samples
    layout_dir = os.path.join(output_dir, 'layout')
    train_annotations_path = os.path.join(layout_dir, 'train_annotations.json')
    
    with open(train_annotations_path, 'r') as f:
        layout_annotations = json.load(f)
    
    # Sample random annotations
    if len(layout_annotations) > sample_size:
        import random
        layout_samples = random.sample(layout_annotations, sample_size)
    else:
        layout_samples = layout_annotations
    
    # Visualize layout samples
    for i, anno in enumerate(layout_samples):
        image_path = anno['image_path']
        regions = anno['regions']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)
        
        # Draw bounding boxes
        for region in regions:
            bbox = region['bbox']
            region_type = region['type']
            
            x, y, w, h = bbox
            
            # Draw rectangle
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(x, y, region_type, color='white', fontsize=12, 
                     bbox=dict(facecolor='red', alpha=0.5))
        
        plt.axis('off')
        plt.title(f"Layout Sample {i+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'layout_sample_{i+1}.png'))
        plt.close()
    
    # Visualize text image samples
    text_images_dir = os.path.join(output_dir, 'text_images')
    train_annotations_path = os.path.join(text_images_dir, 'train_annotations.json')
    
    with open(train_annotations_path, 'r') as f:
        text_images_annotations = json.load(f)
    
    # Sample random annotations
    if len(text_images_annotations) > sample_size:
        import random
        text_images_samples = random.sample(text_images_annotations, sample_size)
    else:
        text_images_samples = text_images_annotations
    
    # Visualize text image samples
    fig, axes = plt.subplots(len(text_images_samples), 1, figsize=(10, 4 * len(text_images_samples)))
    if len(text_images_samples) == 1:
        axes = [axes]
    
    for i, anno in enumerate(text_images_samples):
        image_path = anno['image_path']
        text = anno['text']
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Display image and text
        axes[i].imshow(image)
        axes[i].set_title(f"Text: {text}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'text_images_samples.png'))
    plt.close()
    
    print(f"Visualizations saved to {vis_dir}")


def main():
    """Main function."""
    args = parse_args()
    
    # Process dataset if needed
    if not os.path.exists(args.output_dir):
        print("Processing dataset...")
        process_dataset(args.data_dir, args.output_dir)
    
    # Check dataset structure
    print("Checking dataset structure...")
    structure_ok = check_dataset_structure(args.output_dir)
    
    if not structure_ok:
        print("Dataset structure check failed")
        return
    
    print("Dataset structure check passed")
    
    # Check annotation format
    print("Checking annotation format...")
    format_ok = check_annotation_format(args.output_dir)
    
    if not format_ok:
        print("Annotation format check failed")
        return
    
    print("Annotation format check passed")
    
    # Visualize samples
    if args.visualize:
        print("Visualizing samples...")
        visualize_samples(args.output_dir, args.sample_size)
    
    print("All tests passed!")


if __name__ == '__main__':
    main()
