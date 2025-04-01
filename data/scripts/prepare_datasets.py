#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to prepare datasets for training and evaluation.
Handles splitting data into train/val/test sets and creating appropriate data formats.
"""

import os
import json
import random
import shutil
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def create_dataset_splits(data_dir, annotations_file, output_dir, val_size=0.15, test_size=0.15, seed=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        data_dir (str): Directory containing the image files
        annotations_file (str): Path to the annotations file
        output_dir (str): Directory to save the split datasets
        val_size (float): Proportion of data for validation
        test_size (float): Proportion of data for testing
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary with train, val, test splits
    """
    # Load annotations
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Get list of image files
    image_files = list(annotations.keys())
    
    # Split into train, validation, test
    train_val_files, test_files = train_test_split(
        image_files, test_size=test_size, random_state=seed
    )
    
    train_files, val_files = train_test_split(
        train_val_files, test_size=val_size/(1-test_size), random_state=seed
    )
    
    # Create directories
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    # Create split annotations
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    # Save split information
    for split_name, files in splits.items():
        # Create split annotation file
        split_annotations = {f: annotations[f] for f in files}
        with open(os.path.join(output_dir, f'{split_name}_annotations.json'), 'w', encoding='utf-8') as f:
            json.dump(split_annotations, f, indent=2)
        
        # Copy images to split directories
        for img_file in tqdm(files, desc=f"Copying {split_name} images"):
            src_path = os.path.join(data_dir, img_file)
            dst_path = os.path.join(output_dir, split_name, img_file)
            shutil.copy2(src_path, dst_path)
    
    # Create split summary
    summary = {
        'train': len(train_files),
        'val': len(val_files),
        'test': len(test_files),
        'total': len(image_files)
    }
    
    print(f"Dataset split summary:")
    print(f"  Train: {summary['train']} images ({summary['train']/summary['total']:.1%})")
    print(f"  Validation: {summary['val']} images ({summary['val']/summary['total']:.1%})")
    print(f"  Test: {summary['test']} images ({summary['test']/summary['total']:.1%})")
    
    return splits

def prepare_ocr_dataset(image_dir, transcription_dir, output_dir, aligned=True):
    """
    Prepare dataset for OCR training.
    
    Args:
        image_dir (str): Directory containing text line images
        transcription_dir (str): Directory containing transcription files
        output_dir (str): Directory to save the prepared dataset
        aligned (bool): Whether to create aligned data format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Create dataset metadata
    dataset = []
    
    for img_file in tqdm(image_files, desc="Processing OCR data"):
        img_path = os.path.join(image_dir, img_file)
        
        # Find corresponding transcription file
        base_name = Path(img_file).stem
        txt_file = f"{base_name}.txt"
        txt_path = os.path.join(transcription_dir, txt_file)
        
        if not os.path.exists(txt_path):
            print(f"Warning: No transcription found for {img_file}")
            continue
        
        # Read transcription
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Add to dataset
        dataset.append({
            'image_path': img_path,
            'text': text,
            'id': base_name
        })
    
    # Save dataset metadata
    df = pd.DataFrame(dataset)
    df.to_csv(os.path.join(output_dir, 'ocr_dataset.csv'), index=False)
    
    # If aligned format is requested, create it
    if aligned:
        aligned_dir = os.path.join(output_dir, 'aligned')
        os.makedirs(aligned_dir, exist_ok=True)
        
        for item in tqdm(dataset, desc="Creating aligned data"):
            # Copy image
            shutil.copy2(item['image_path'], os.path.join(aligned_dir, f"{item['id']}.png"))
            
            # Create text file
            with open(os.path.join(aligned_dir, f"{item['id']}.txt"), 'w', encoding='utf-8') as f:
                f.write(item['text'])
    
    print(f"Prepared OCR dataset with {len(dataset)} samples")
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for training and evaluation")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Split dataset command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument("--data_dir", required=True, help="Directory containing image files")
    split_parser.add_argument("--annotations", required=True, help="Path to annotations file")
    split_parser.add_argument("--output_dir", required=True, help="Directory to save split datasets")
    split_parser.add_argument("--val_size", type=float, default=0.15, help="Proportion for validation")
    split_parser.add_argument("--test_size", type=float, default=0.15, help="Proportion for testing")
    split_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # OCR dataset command
    ocr_parser = subparsers.add_parser('ocr', help='Prepare OCR dataset')
    ocr_parser.add_argument("--image_dir", required=True, help="Directory with text line images")
    ocr_parser.add_argument("--transcription_dir", required=True, help="Directory with transcriptions")
    ocr_parser.add_argument("--output_dir", required=True, help="Directory to save prepared dataset")
    ocr_parser.add_argument("--aligned", action='store_true', help="Create aligned data format")
    
    args = parser.parse_args()
    
    if args.command == 'split':
        create_dataset_splits(
            args.data_dir, args.annotations, args.output_dir,
            args.val_size, args.test_size, args.seed
        )
    elif args.command == 'ocr':
        prepare_ocr_dataset(
            args.image_dir, args.transcription_dir, args.output_dir, args.aligned
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
