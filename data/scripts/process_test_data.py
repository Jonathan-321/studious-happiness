#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process the Renaissance test data for layout recognition and OCR tasks.
"""

import os
import argparse
import shutil
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import PyPDF2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(output_dir):
    """
    Create necessary output directories.
    
    Args:
        output_dir (str): Base output directory
    """
    # Create main directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each task
    layout_dir = os.path.join(output_dir, 'layout')
    ocr_dir = os.path.join(output_dir, 'ocr')
    
    os.makedirs(layout_dir, exist_ok=True)
    os.makedirs(ocr_dir, exist_ok=True)
    
    # Create subdirectories for layout task
    os.makedirs(os.path.join(layout_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(layout_dir, 'annotations'), exist_ok=True)
    
    # Create subdirectories for OCR task
    os.makedirs(os.path.join(ocr_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(ocr_dir, 'transcriptions'), exist_ok=True)
    
    logger.info(f"Created directory structure in {output_dir}")

def extract_images_from_pdf(pdf_path, output_dir):
    """
    Extract images from a PDF file using PyPDF2 and PIL.
    This creates a visualization of the PDF page with text content.
    
    Args:
        pdf_path (str): Path to PDF file
        output_dir (str): Output directory for extracted images
    
    Returns:
        list: List of paths to extracted images
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get PDF filename without extension
        pdf_filename = os.path.basename(pdf_path)
        pdf_name = os.path.splitext(pdf_filename)[0]
        
        # Open PDF file
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            
            # Get number of pages
            num_pages = len(pdf.pages)
            
            # Create a list to store image paths
            image_paths = []
            
            # Process each page
            for i in range(num_pages):
                # Get page
                page = pdf.pages[i]
                
                # Extract text from page
                text = page.extract_text()
                
                # Create image path
                img_path = os.path.join(output_dir, f"{pdf_name}_page_{i+1}.png")
                
                # Create a visualization of the page
                # Use a larger size for better visibility
                width, height = 1200, 1600
                img = Image.new('RGB', (width, height), color='white')
                
                # Create a drawing context
                draw = ImageDraw.Draw(img)
                
                # Try to load a font
                try:
                    # Try to use a system font
                    font = ImageFont.truetype("Arial", 16)
                except IOError:
                    # Fall back to default font
                    font = ImageFont.load_default()
                
                # Split text into lines
                lines = text.split('\n')
                
                # Draw text on image
                y_position = 50
                for line in lines:
                    # Skip empty lines
                    if not line.strip():
                        y_position += 20
                        continue
                        
                    # Draw line of text
                    draw.text((50, y_position), line, fill='black', font=font)
                    y_position += 20
                
                # Draw a border to simulate page edges
                draw.rectangle([(20, 20), (width-20, height-20)], outline='gray', width=2)
                
                # Add some visual elements to simulate layout
                # Header
                draw.rectangle([(50, 50), (width-50, 100)], outline='lightgray', width=1)
                
                # Columns (if text suggests multiple columns)
                if len(lines) > 10 and any(len(line) < 40 for line in lines):
                    mid = width // 2
                    draw.line([(mid, 150), (mid, height-150)], fill='lightgray', width=1)
                
                # Save the image
                img.save(img_path)
                
                # Add to list of image paths
                image_paths.append(img_path)
            
            logger.info(f"Extracted {len(image_paths)} images from {pdf_path}")
            return image_paths
    
    except Exception as e:
        logger.error(f"Error extracting images from {pdf_path}: {e}")
        return []

def process_source_images(source_dir, output_dir):
    """
    Process source images for layout recognition and OCR.
    
    Args:
        source_dir (str): Directory containing source images or PDFs
        output_dir (str): Output directory for processed data
    
    Returns:
        list: List of processed image paths
    """
    # Get all image files in the source directory
    image_files = list(Path(source_dir).glob('*.jpg')) + \
                 list(Path(source_dir).glob('*.jpeg')) + \
                 list(Path(source_dir).glob('*.png'))
    
    # Get all PDF files in the source directory
    pdf_files = list(Path(source_dir).glob('*.pdf'))
    
    logger.info(f"Found {len(image_files)} images and {len(pdf_files)} PDFs in {source_dir}")
    
    # Process each image
    processed_images = []
    
    # First, process regular image files
    for img_path in image_files:
        # Get image filename
        img_filename = os.path.basename(img_path)
        
        # Create paths for layout and OCR tasks
        layout_img_path = os.path.join(output_dir, 'layout', 'images', img_filename)
        ocr_img_path = os.path.join(output_dir, 'ocr', 'images', img_filename)
        
        # Copy image to layout and OCR directories
        shutil.copy(img_path, layout_img_path)
        shutil.copy(img_path, ocr_img_path)
        
        processed_images.append({
            'original_path': str(img_path),
            'layout_path': layout_img_path,
            'ocr_path': ocr_img_path,
            'filename': img_filename
        })
        
        logger.info(f"Processed image {img_filename}")
    
    # Then, process PDF files
    for pdf_path in pdf_files:
        # Skip directories with 'HANDWRITTEN' in the name for now
        if 'HANDWRITTEN' in str(pdf_path):
            logger.info(f"Skipping handwritten document: {pdf_path}")
            continue
            
        # Get PDF filename
        pdf_filename = os.path.basename(pdf_path)
        
        # Extract images from PDF
        temp_dir = os.path.join(output_dir, 'temp', os.path.splitext(pdf_filename)[0])
        os.makedirs(temp_dir, exist_ok=True)
        
        extracted_images = extract_images_from_pdf(pdf_path, temp_dir)
        
        # Process each extracted image
        for img_path in extracted_images:
            # Get image filename
            img_filename = os.path.basename(img_path)
            
            # Create paths for layout and OCR tasks
            layout_img_path = os.path.join(output_dir, 'layout', 'images', img_filename)
            ocr_img_path = os.path.join(output_dir, 'ocr', 'images', img_filename)
            
            # Copy image to layout and OCR directories
            shutil.copy(img_path, layout_img_path)
            shutil.copy(img_path, ocr_img_path)
            
            processed_images.append({
                'original_path': img_path,
                'layout_path': layout_img_path,
                'ocr_path': ocr_img_path,
                'filename': img_filename,
                'source_pdf': str(pdf_path)
            })
            
            logger.info(f"Processed PDF page {img_filename}")
    
    return processed_images

def create_dummy_layout_annotations(processed_images, output_dir):
    """
    Create dummy layout annotations for the processed images.
    In a real scenario, these would be created from actual annotations.
    
    Args:
        processed_images (list): List of dictionaries containing processed image information
        output_dir (str): Output directory for processed data
    """
    annotations_dir = os.path.join(output_dir, 'layout', 'annotations')
    
    for img_info in processed_images:
        # Get the layout image path
        img_path = img_info['layout_path']
        img_filename = img_info['filename']
        base_name = os.path.splitext(img_filename)[0]
        json_filename = f"{base_name}.json"
        json_path = os.path.join(annotations_dir, json_filename)
        
        # Read image to get dimensions
        img = Image.open(img_path)
        width, height = img.size
        
        # Create a dummy annotation with common layout regions
        annotation = {
            "image_path": img_path,
            "width": width,
            "height": height,
            "regions": [
                {
                    "id": 1,
                    "label": "Title",
                    "points": [
                        [int(width * 0.1), int(height * 0.05)],
                        [int(width * 0.9), int(height * 0.05)],
                        [int(width * 0.9), int(height * 0.15)],
                        [int(width * 0.1), int(height * 0.15)]
                    ],
                    "type": "polygon"
                },
                {
                    "id": 2,
                    "label": "Text",
                    "points": [
                        [int(width * 0.1), int(height * 0.2)],
                        [int(width * 0.9), int(height * 0.2)],
                        [int(width * 0.9), int(height * 0.7)],
                        [int(width * 0.1), int(height * 0.7)]
                    ],
                    "type": "polygon"
                },
                {
                    "id": 3,
                    "label": "Figure",
                    "points": [
                        [int(width * 0.3), int(height * 0.75)],
                        [int(width * 0.7), int(height * 0.75)],
                        [int(width * 0.7), int(height * 0.9)],
                        [int(width * 0.3), int(height * 0.9)]
                    ],
                    "type": "polygon"
                }
            ]
        }
        
        # Save the annotation as JSON
        with open(json_path, 'w') as f:
            json.dump(annotation, f, indent=4)
        
        logger.info(f"Created dummy annotation: {json_filename}")

def process_transcriptions(transcription_dir, output_dir, processed_images):
    """
    Process transcription files for OCR.
    
    Args:
        transcription_dir (str): Directory containing transcription files
        output_dir (str): Output directory for processed data
        processed_images (list): List of dictionaries containing processed image information
    """
    ocr_transcriptions_dir = os.path.join(output_dir, 'ocr', 'transcriptions')
    
    # Get all transcription files (including docx files)
    transcription_files = []
    for ext in ['*.txt', '*.xml', '*.json', '*.docx']:
        transcription_files.extend(list(Path(transcription_dir).glob(ext)))
    
    logger.info(f"Found {len(transcription_files)} transcription files in {transcription_dir}")
    
    # Process each transcription file
    for trans_path in transcription_files:
        trans_filename = os.path.basename(trans_path)
        base_name = os.path.splitext(trans_filename)[0]
        
        # Find corresponding images by matching part of the filename
        # This is a simple approach - in a real scenario, you might need a more sophisticated matching
        matching_images = [img_info for img_info in processed_images 
                          if any(part in img_info['filename'] for part in base_name.split())]
        
        if matching_images:
            # For simplicity, we'll just use the first matching image
            img_info = matching_images[0]
            
            # Create a text file with the transcription content if it's a docx file
            if trans_path.suffix.lower() == '.docx':
                try:
                    import docx
                    doc = docx.Document(trans_path)
                    content = '\n'.join([para.text for para in doc.paragraphs])
                    
                    # Save as text file
                    txt_filename = f"{base_name}.txt"
                    txt_path = os.path.join(ocr_transcriptions_dir, txt_filename)
                    
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    logger.info(f"Converted DOCX to text: {txt_filename}")
                except Exception as e:
                    logger.error(f"Error processing DOCX file {trans_path}: {e}")
                    # Copy the original file as fallback
                    output_trans_path = os.path.join(ocr_transcriptions_dir, trans_filename)
                    shutil.copy(trans_path, output_trans_path)
            else:
                # Copy the transcription file to OCR transcriptions directory
                output_trans_path = os.path.join(ocr_transcriptions_dir, trans_filename)
                shutil.copy(trans_path, output_trans_path)
            
            logger.info(f"Processed transcription: {trans_filename}")
        else:
            logger.warning(f"No matching image found for transcription: {trans_filename}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Process Renaissance test data')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Directory containing source images')
    parser.add_argument('--transcription_dir', type=str, required=True,
                        help='Directory containing transcription files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    args = parser.parse_args()
    
    # Setup output directories
    setup_directories(args.output_dir)
    
    # Process source images
    processed_images = process_source_images(args.source_dir, args.output_dir)
    
    # Create dummy layout annotations
    create_dummy_layout_annotations(processed_images, args.output_dir)
    
    # Process transcriptions
    process_transcriptions(args.transcription_dir, args.output_dir, processed_images)
    
    logger.info("Data processing completed!")

if __name__ == '__main__':
    main()
