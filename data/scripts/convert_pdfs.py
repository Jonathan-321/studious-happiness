#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to convert PDF documents to images for further processing.
"""

import os
import argparse
from pathlib import Path
from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def convert_pdf_to_images(pdf_path, output_dir, dpi=300, fmt='png'):
    """
    Convert a PDF file to a series of images.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Directory to save the output images
        dpi (int): DPI resolution for the output images
        fmt (str): Output format (png, jpg, etc.)
    
    Returns:
        list: List of paths to the generated images
    """
    pdf_name = Path(pdf_path).stem
    output_paths = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=dpi)
    
    # Save images
    for i, image in enumerate(images):
        output_path = os.path.join(output_dir, f"{pdf_name}_page_{i+1:03d}.{fmt}")
        image.save(output_path, fmt.upper())
        output_paths.append(output_path)
    
    return output_paths

def process_pdfs(pdf_dir, output_dir, dpi=300, fmt='png', max_workers=4):
    """
    Process all PDFs in a directory.
    
    Args:
        pdf_dir (str): Directory containing PDF files
        output_dir (str): Directory to save the output images
        dpi (int): DPI resolution for the output images
        fmt (str): Output format (png, jpg, etc.)
        max_workers (int): Maximum number of parallel workers
    """
    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for pdf_path in pdf_files:
            futures.append(
                executor.submit(convert_pdf_to_images, pdf_path, output_dir, dpi, fmt)
            )
        
        for future in tqdm(futures, desc="Converting PDFs"):
            future.result()

def main():
    parser = argparse.ArgumentParser(description="Convert PDF documents to images")
    parser.add_argument("--pdf_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", required=True, help="Directory to save output images")
    parser.add_argument("--dpi", type=int, default=300, help="DPI resolution for output images")
    parser.add_argument("--format", default="png", help="Output image format (png, jpg, etc.)")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of parallel workers")
    
    args = parser.parse_args()
    
    process_pdfs(args.pdf_dir, args.output_dir, args.dpi, args.format, args.max_workers)
    
if __name__ == "__main__":
    main()
