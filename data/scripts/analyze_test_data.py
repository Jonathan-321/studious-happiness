#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze the Renaissance test data.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import PyPDF2
import docx
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_pdf(pdf_path):
    """
    Analyze a PDF file.
    
    Args:
        pdf_path (str): Path to PDF file
    
    Returns:
        dict: PDF analysis results
    """
    try:
        # Open PDF file
        with open(pdf_path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            
            # Get number of pages
            num_pages = len(pdf.pages)
            
            # Extract text from first few pages
            sample_text = []
            for i in range(min(3, num_pages)):
                page = pdf.pages[i]
                text = page.extract_text()
                sample_text.append(f"Page {i+1}:\n{text[:500]}...")
            
            # Get PDF info if available
            info = {}
            if pdf.metadata:
                for key, value in pdf.metadata.items():
                    if key.startswith('/'):
                        info[key[1:]] = value
                    else:
                        info[key] = value
            
            return {
                'path': pdf_path,
                'num_pages': num_pages,
                'sample_text': sample_text,
                'info': info
            }
    
    except Exception as e:
        logger.error(f"Error analyzing {pdf_path}: {e}")
        return None


def read_docx(docx_path):
    """
    Read content from a DOCX file.
    
    Args:
        docx_path (str): Path to DOCX file
    
    Returns:
        str: Content of the DOCX file
    """
    try:
        doc = docx.Document(docx_path)
        content = []
        
        for para in doc.paragraphs:
            content.append(para.text)
        
        return '\n'.join(content)
    
    except Exception as e:
        logger.error(f"Error reading {docx_path}: {e}")
        return ""


def analyze_image(image_path):
    """
    Analyze an image.
    
    Args:
        image_path (str): Path to image
    
    Returns:
        dict: Image analysis results
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Get image dimensions
        width, height = img.size
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Calculate mean and std of pixel values
        mean_r = np.mean(img_array[:, :, 0])
        mean_g = np.mean(img_array[:, :, 1])
        mean_b = np.mean(img_array[:, :, 2])
        std_r = np.std(img_array[:, :, 0])
        std_g = np.std(img_array[:, :, 1])
        std_b = np.std(img_array[:, :, 2])
        
        # Calculate histogram
        hist_r, _ = np.histogram(img_array[:, :, 0], bins=256, range=(0, 256))
        hist_g, _ = np.histogram(img_array[:, :, 1], bins=256, range=(0, 256))
        hist_b, _ = np.histogram(img_array[:, :, 2], bins=256, range=(0, 256))
        
        return {
            'path': image_path,
            'width': width,
            'height': height,
            'mean': (mean_r, mean_g, mean_b),
            'std': (std_r, std_g, std_b),
            'hist': (hist_r, hist_g, hist_b)
        }
    
    except Exception as e:
        logger.error(f"Error analyzing {image_path}: {e}")
        return None


def visualize_image(image_path, output_path=None):
    """
    Visualize an image.
    
    Args:
        image_path (str): Path to image
        output_path (str, optional): Path to save visualization
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot image
        plt.imshow(img)
        plt.title(f"Image: {os.path.basename(image_path)}")
        plt.axis('off')
        
        # Save or show
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    except Exception as e:
        logger.error(f"Error visualizing {image_path}: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze Renaissance test data')
    parser.add_argument('--source_dir', type=str, default='data/raw/test_sources',
                        help='Directory containing source PDFs')
    parser.add_argument('--transcription_dir', type=str, default='data/raw/test_transcriptions',
                        help='Directory containing transcription files')
    parser.add_argument('--output_dir', type=str, default='data/processed/analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--num_samples', type=int, default=2,
                        help='Number of pages to analyze per PDF')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectories
    images_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    visualizations_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Get all PDF files
    pdf_files = list(Path(args.source_dir).glob('*.pdf'))
    logger.info(f"Found {len(pdf_files)} PDF files in {args.source_dir}")
    
    # Process each PDF
    for pdf_path in pdf_files:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        logger.info(f"Processing {pdf_name}")
        
        # Analyze PDF
        analysis = analyze_pdf(pdf_path)
        if analysis:
            logger.info(f"Analyzed {pdf_name}: {analysis['num_pages']} pages")
            
            # Save analysis to text file
            analysis_path = os.path.join(args.output_dir, f"{pdf_name}_analysis.txt")
            with open(analysis_path, 'w', encoding='utf-8') as f:
                f.write(f"PDF: {pdf_name}\n")
                f.write(f"Number of pages: {analysis['num_pages']}\n")
                f.write(f"Info: {analysis['info']}\n\n")
                f.write("Sample text:\n")
                for text in analysis['sample_text']:
                    f.write(f"{text}\n\n")
            
            logger.info(f"Saved analysis to {analysis_path}")
    
    # Get all transcription files
    transcription_files = list(Path(args.transcription_dir).glob('*.docx'))
    logger.info(f"Found {len(transcription_files)} transcription files in {args.transcription_dir}")
    
    # Process each transcription file
    for docx_path in transcription_files:
        docx_name = os.path.splitext(os.path.basename(docx_path))[0]
        logger.info(f"Processing {docx_name}")
        
        # Read transcription
        content = read_docx(docx_path)
        
        # Save content to text file
        txt_path = os.path.join(args.output_dir, f"{docx_name}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Saved transcription to {txt_path}")
    
    logger.info("Analysis completed!")


if __name__ == '__main__':
    main()
