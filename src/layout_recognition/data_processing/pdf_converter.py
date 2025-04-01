#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for converting PDFs to images for layout analysis.
"""

import os
import logging
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)

class PDFConverter:
    """Class for converting PDF documents to images."""
    
    def __init__(self, dpi=300, output_format='png', grayscale=False):
        """
        Initialize the PDF converter.
        
        Args:
            dpi (int): Resolution for the output images
            output_format (str): Output image format (png, jpg)
            grayscale (bool): Whether to convert images to grayscale
        """
        self.dpi = dpi
        self.output_format = output_format
        self.grayscale = grayscale
    
    def convert_pdf(self, pdf_path, output_dir, start_page=1, end_page=None):
        """
        Convert a PDF file to images.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str): Directory to save the output images
            start_page (int): First page to convert (1-indexed)
            end_page (int): Last page to convert (inclusive)
            
        Returns:
            list: Paths to the generated image files
        """
        pdf_name = Path(pdf_path).stem
        os.makedirs(output_dir, exist_ok=True)
        
        # Adjust for pdf2image 0-indexing
        first_page = start_page
        last_page = end_page
        
        # Convert PDF pages to images
        try:
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=first_page,
                last_page=last_page
            )
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path}: {e}")
            return []
        
        # Save images
        output_paths = []
        for i, image in enumerate(images):
            page_num = i + start_page
            output_path = os.path.join(output_dir, f"{pdf_name}_page_{page_num:03d}.{self.output_format}")
            
            # Convert to grayscale if requested
            if self.grayscale:
                image = image.convert('L')
            
            # Save the image
            image.save(output_path)
            output_paths.append(output_path)
            
            logger.info(f"Saved page {page_num} to {output_path}")
        
        return output_paths
    
    def convert_directory(self, pdf_dir, output_dir):
        """
        Convert all PDFs in a directory.
        
        Args:
            pdf_dir (str): Directory containing PDF files
            output_dir (str): Directory to save the output images
            
        Returns:
            dict: Mapping of PDF filenames to lists of generated image paths
        """
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        results = {}
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            logger.info(f"Converting {pdf_path}")
            
            # Create subdirectory for each PDF
            pdf_name = Path(pdf_file).stem
            pdf_output_dir = os.path.join(output_dir, pdf_name)
            
            # Convert the PDF
            image_paths = self.convert_pdf(pdf_path, pdf_output_dir)
            results[pdf_file] = image_paths
        
        return results
