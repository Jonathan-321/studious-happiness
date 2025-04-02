#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Post-processing utilities for refining layout recognition results.
"""

import numpy as np
import cv2
from scipy import ndimage

def apply_morphological_operations(mask, kernel_size=5):
    """
    Apply morphological operations to refine the mask.
    
    Args:
        mask (numpy.ndarray): Binary mask
        kernel_size (int): Size of the kernel for morphological operations
        
    Returns:
        numpy.ndarray: Refined mask
    """
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply closing to fill small holes
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # Apply opening to remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def remove_small_regions(mask, min_size=100):
    """
    Remove small connected components from the mask.
    
    Args:
        mask (numpy.ndarray): Binary mask
        min_size (int): Minimum size of connected components to keep
        
    Returns:
        numpy.ndarray: Refined mask
    """
    # Label connected components
    labeled_mask, num_features = ndimage.label(mask)
    
    # Get component sizes
    component_sizes = np.bincount(labeled_mask.ravel())
    
    # Set small components to 0
    too_small = component_sizes < min_size
    too_small[0] = False  # Keep background
    mask_out = np.logical_not(too_small[labeled_mask])
    
    return mask_out.astype(np.uint8)

def refine_text_mask(mask, original_image=None, min_region_size=100, morph_kernel_size=5):
    """
    Refine text mask using various post-processing techniques.
    
    Args:
        mask (numpy.ndarray): Binary mask
        original_image (numpy.ndarray, optional): Original image for reference
        min_region_size (int): Minimum size of connected components to keep
        morph_kernel_size (int): Size of the kernel for morphological operations
        
    Returns:
        numpy.ndarray: Refined mask
    """
    # Apply morphological operations
    refined_mask = apply_morphological_operations(mask, kernel_size=morph_kernel_size)
    
    # Remove small regions
    refined_mask = remove_small_regions(refined_mask, min_size=min_region_size)
    
    # If original image is provided, we can use it for additional refinement
    if original_image is not None:
        # Convert to grayscale if needed
        if len(original_image.shape) == 3:
            gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = original_image
            
        # Use edge detection to refine text regions
        edges = cv2.Canny(gray_image, 50, 150)
        
        # Dilate edges to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Combine with refined mask
        combined_mask = np.logical_and(refined_mask, dilated_edges > 0)
        
        # Apply morphological operations again to clean up
        combined_mask = apply_morphological_operations(combined_mask, kernel_size=3)
        
        # Remove small regions again
        refined_mask = remove_small_regions(combined_mask, min_size=min_region_size // 2)
    
    return refined_mask

def enhance_text_regions(image, mask, alpha=0.7, color=(255, 0, 0)):
    """
    Enhance text regions in the image.
    
    Args:
        image (numpy.ndarray): Original image
        mask (numpy.ndarray): Binary mask
        alpha (float): Transparency factor
        color (tuple): Color for text regions (BGR)
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    # Create a copy of the image
    enhanced_image = image.copy()
    
    # Create color overlay
    overlay = np.zeros_like(image)
    overlay[mask > 0] = color
    
    # Apply overlay with transparency
    cv2.addWeighted(overlay, alpha, enhanced_image, 1 - alpha, 0, enhanced_image)
    
    return enhanced_image
