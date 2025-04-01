#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process Renaissance dataset for GSoC 2025 submission.
This script processes the 6 scanned early modern printed sources and prepares
them for layout recognition and OCR tasks.
"""

import os
import json
import argparse
import numpy as np
from PIL import Image
import cv2
import pytesseract
from tqdm import tqdm
import fitz  # PyMuPDF

from src.layout_recognition.data_processing.pdf_converter import PDFConverter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process Renaissance dataset')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing PDF files')
    parser.add_argument('--transcription_dir', type=str, required=True,
                        help='Directory containing transcription files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save processed data')
    parser.add_argument('--task', type=str, choices=['layout', 'ocr', 'both'], default='both',
                        help='Task to prepare data for')
    parser.add_argument('--split_ratio', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                        help='Train/val/test split ratio')
    return parser.parse_args()


def extract_layout_from_pdf(pdf_path, output_dir):
    """
    Extract layout information from PDF.
    
    Args:
        pdf_path (str): Path to PDF file
        output_dir (str): Directory to save layout annotations
        
    Returns:
        list: List of layout annotations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PDF to images
    pdf_converter = PDFConverter()
    image_dir = os.path.join(output_dir, 'images')
    pdf_converter.convert_pdf(pdf_path, image_dir)
    
    # Extract layout information
    doc = fitz.open(pdf_path)
    layout_annotations = []
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        
        # Get text blocks
        blocks = page.get_text("blocks")
        
        # Create image filename
        image_filename = f"page_{page_idx+1}.png"
        image_path = os.path.join(image_dir, image_filename)
        
        # Get page dimensions
        page_width, page_height = page.rect.width, page.rect.height
        
        # Create annotation for this page
        page_annotation = {
            'image_path': image_path,
            'width': page_width,
            'height': page_height,
            'regions': []
        }
        
        # Process each text block
        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            
            # Skip empty blocks
            if not text.strip():
                continue
            
            # Create region annotation
            region = {
                'x': x0,
                'y': y0,
                'width': x1 - x0,
                'height': y1 - y0,
                'type': 'text',
                'text': text
            }
            
            page_annotation['regions'].append(region)
        
        layout_annotations.append(page_annotation)
    
    return layout_annotations


def match_transcription_to_pages(pdf_path, transcription_path, num_pages=3):
    """
    Match transcription to PDF pages.
    
    Args:
        pdf_path (str): Path to PDF file
        transcription_path (str): Path to transcription file
        num_pages (int): Number of pages with transcription
        
    Returns:
        dict: Dictionary mapping page index to transcription
    """
    # Read transcription file
    with open(transcription_path, 'r', encoding='utf-8') as f:
        transcription_text = f.read()
    
    # Split transcription into pages (assuming pages are separated by specific markers)
    # This is a simplification - actual implementation would depend on transcription format
    page_texts = transcription_text.split('--- PAGE ---')
    
    # Clean up page texts
    page_texts = [text.strip() for text in page_texts if text.strip()]
    
    # Ensure we have the expected number of pages
    if len(page_texts) < num_pages:
        print(f"Warning: Expected {num_pages} pages in transcription, but found {len(page_texts)}")
    
    # Create mapping
    transcription_map = {}
    for i in range(min(num_pages, len(page_texts))):
        transcription_map[i] = page_texts[i]
    
    return transcription_map


def prepare_ocr_dataset(pdf_paths, transcription_paths, output_dir):
    """
    Prepare dataset for OCR task.
    
    Args:
        pdf_paths (list): List of PDF file paths
        transcription_paths (list): List of transcription file paths
        output_dir (str): Directory to save OCR dataset
        
    Returns:
        dict: OCR dataset information
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    ocr_annotations = []
    pdf_converter = PDFConverter()
    
    for pdf_idx, (pdf_path, transcription_path) in enumerate(zip(pdf_paths, transcription_paths)):
        print(f"Processing {os.path.basename(pdf_path)} for OCR...")
        
        # Convert PDF to images
        source_name = os.path.splitext(os.path.basename(pdf_path))[0]
        source_images_dir = os.path.join(images_dir, source_name)
        os.makedirs(source_images_dir, exist_ok=True)
        
        # Convert PDF to images
        pdf_converter.convert_pdf(pdf_path, source_images_dir)
        
        # Match transcription to pages
        transcription_map = match_transcription_to_pages(pdf_path, transcription_path)
        
        # Process each page with transcription
        doc = fitz.open(pdf_path)
        
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            
            # Create image filename
            image_filename = f"page_{page_idx+1}.png"
            image_path = os.path.join(source_images_dir, image_filename)
            
            # Check if we have transcription for this page
            if page_idx in transcription_map:
                # Get text blocks
                blocks = page.get_text("blocks")
                
                # Process each text block
                for block_idx, block in enumerate(blocks):
                    x0, y0, x1, y1, text, block_no, block_type = block
                    
                    # Skip empty blocks
                    if not text.strip():
                        continue
                    
                    # Create block image
                    block_image_filename = f"{source_name}_page_{page_idx+1}_block_{block_idx+1}.png"
                    block_image_path = os.path.join(source_images_dir, block_image_filename)
                    
                    # Extract block image
                    pix = page.get_pixmap(clip=(x0, y0, x1, y1))
                    pix.save(block_image_path)
                    
                    # Create annotation
                    annotation = {
                        'image_path': block_image_path,
                        'text': text.strip(),
                        'source': source_name,
                        'page': page_idx,
                        'block': block_idx,
                        'has_ground_truth': True
                    }
                    
                    ocr_annotations.append(annotation)
            else:
                # For pages without transcription, we'll use them for inference testing
                blocks = page.get_text("blocks")
                
                for block_idx, block in enumerate(blocks):
                    x0, y0, x1, y1, text, block_no, block_type = block
                    
                    # Skip empty blocks
                    if not text.strip():
                        continue
                    
                    # Create block image
                    block_image_filename = f"{source_name}_page_{page_idx+1}_block_{block_idx+1}.png"
                    block_image_path = os.path.join(source_images_dir, block_image_filename)
                    
                    # Extract block image
                    pix = page.get_pixmap(clip=(x0, y0, x1, y1))
                    pix.save(block_image_path)
                    
                    # Create annotation
                    annotation = {
                        'image_path': block_image_path,
                        'text': text.strip(),  # This is the OCR text from PyMuPDF, not ground truth
                        'source': source_name,
                        'page': page_idx,
                        'block': block_idx,
                        'has_ground_truth': False
                    }
                    
                    ocr_annotations.append(annotation)
    
    return ocr_annotations


def prepare_layout_dataset(pdf_paths, output_dir):
    """
    Prepare dataset for layout recognition task.
    
    Args:
        pdf_paths (list): List of PDF file paths
        output_dir (str): Directory to save layout dataset
        
    Returns:
        dict: Layout dataset information
    """
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    layout_annotations = []
    
    for pdf_idx, pdf_path in enumerate(pdf_paths):
        print(f"Processing {os.path.basename(pdf_path)} for layout recognition...")
        
        # Extract layout information
        source_name = os.path.splitext(os.path.basename(pdf_path))[0]
        source_images_dir = os.path.join(images_dir, source_name)
        os.makedirs(source_images_dir, exist_ok=True)
        
        source_annotations = extract_layout_from_pdf(pdf_path, source_images_dir)
        
        # Add source information
        for annotation in source_annotations:
            annotation['source'] = source_name
            layout_annotations.append(annotation)
    
    return layout_annotations


def split_dataset(annotations, split_ratio):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        annotations (list): List of annotations
        split_ratio (list): Train/val/test split ratio
        
    Returns:
        tuple: (train_annotations, val_annotations, test_annotations)
    """
    # Shuffle annotations
    np.random.shuffle(annotations)
    
    # Calculate split indices
    n = len(annotations)
    train_idx = int(n * split_ratio[0])
    val_idx = train_idx + int(n * split_ratio[1])
    
    # Split annotations
    train_annotations = annotations[:train_idx]
    val_annotations = annotations[train_idx:val_idx]
    test_annotations = annotations[val_idx:]
    
    return train_annotations, val_annotations, test_annotations


def main():
    """Main function."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get PDF and transcription files
    pdf_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                 if f.lower().endswith('.pdf')]
    
    transcription_files = [os.path.join(args.transcription_dir, f) for f in os.listdir(args.transcription_dir)
                           if f.lower().endswith('.txt')]
    
    # Sort files to ensure consistent pairing
    pdf_files.sort()
    transcription_files.sort()
    
    # Check if we have matching files
    if len(transcription_files) < len(pdf_files):
        print(f"Warning: Found {len(pdf_files)} PDF files but only {len(transcription_files)} transcription files")
    
    # Prepare datasets
    if args.task in ['layout', 'both']:
        layout_dir = os.path.join(args.output_dir, 'layout')
        layout_annotations = prepare_layout_dataset(pdf_files, layout_dir)
        
        # Split dataset
        train_layout, val_layout, test_layout = split_dataset(layout_annotations, args.split_ratio)
        
        # Save annotations
        with open(os.path.join(layout_dir, 'train_annotations.json'), 'w') as f:
            json.dump(train_layout, f, indent=2)
        
        with open(os.path.join(layout_dir, 'val_annotations.json'), 'w') as f:
            json.dump(val_layout, f, indent=2)
        
        with open(os.path.join(layout_dir, 'test_annotations.json'), 'w') as f:
            json.dump(test_layout, f, indent=2)
        
        print(f"Layout dataset prepared: {len(train_layout)} train, {len(val_layout)} val, {len(test_layout)} test")
    
    if args.task in ['ocr', 'both']:
        ocr_dir = os.path.join(args.output_dir, 'ocr')
        ocr_annotations = prepare_ocr_dataset(pdf_files, transcription_files, ocr_dir)
        
        # Split dataset
        train_ocr, val_ocr, test_ocr = split_dataset(ocr_annotations, args.split_ratio)
        
        # Save annotations
        with open(os.path.join(ocr_dir, 'train_annotations.json'), 'w') as f:
            json.dump(train_ocr, f, indent=2)
        
        with open(os.path.join(ocr_dir, 'val_annotations.json'), 'w') as f:
            json.dump(val_ocr, f, indent=2)
        
        with open(os.path.join(ocr_dir, 'test_annotations.json'), 'w') as f:
            json.dump(test_ocr, f, indent=2)
        
        print(f"OCR dataset prepared: {len(train_ocr)} train, {len(val_ocr)} val, {len(test_ocr)} test")


if __name__ == '__main__':
    main()
