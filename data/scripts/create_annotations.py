#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to assist in creating annotations for layout regions and text.
This tool helps generate annotation files in a format suitable for training.
"""

import os
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Define region types for layout annotation
REGION_TYPES = [
    "text", "title", "paragraph", "figure", "table", 
    "marginalia", "header", "footer", "decoration"
]

class AnnotationTool:
    def __init__(self, image_dir, output_dir):
        """
        Initialize the annotation tool.
        
        Args:
            image_dir (str): Directory containing document images
            output_dir (str): Directory to save annotation files
        """
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.annotations = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_existing_annotations(self, annotation_file):
        """
        Load existing annotations from a file.
        
        Args:
            annotation_file (str): Path to the annotation file
        """
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
            print(f"Loaded {len(self.annotations)} existing annotations")
    
    def save_annotations(self, annotation_file):
        """
        Save annotations to a file.
        
        Args:
            annotation_file (str): Path to save the annotation file
        """
        with open(annotation_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"Saved {len(self.annotations)} annotations to {annotation_file}")
    
    def convert_annotations_to_coco(self, output_file):
        """
        Convert annotations to COCO format.
        
        Args:
            output_file (str): Path to save the COCO format file
        """
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Create categories
        for i, category in enumerate(REGION_TYPES):
            coco_data["categories"].append({
                "id": i + 1,
                "name": category,
                "supercategory": "layout"
            })
        
        # Convert annotations
        annotation_id = 1
        for image_id, (image_name, regions) in enumerate(self.annotations.items()):
            # Add image info
            image_path = os.path.join(self.image_dir, image_name)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            height, width = img.shape[:2]
            
            coco_data["images"].append({
                "id": image_id + 1,
                "file_name": image_name,
                "width": width,
                "height": height
            })
            
            # Add annotations
            for region in regions:
                x, y, w, h = region["bbox"]
                category_id = REGION_TYPES.index(region["type"]) + 1
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id + 1,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "segmentation": [],
                    "iscrowd": 0
                })
                annotation_id += 1
        
        # Save COCO format
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
        print(f"Converted annotations to COCO format and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Create and manage annotations for document layout")
    parser.add_argument("--image_dir", required=True, help="Directory containing document images")
    parser.add_argument("--output_dir", required=True, help="Directory to save annotation files")
    parser.add_argument("--load", help="Load existing annotation file")
    parser.add_argument("--save", required=True, help="Output annotation file name")
    parser.add_argument("--coco", help="Convert and save annotations in COCO format")
    
    args = parser.parse_args()
    
    tool = AnnotationTool(args.image_dir, args.output_dir)
    
    if args.load:
        tool.load_existing_annotations(args.load)
    
    # Here would be interactive annotation code if needed
    # For now, we just save the loaded annotations
    
    tool.save_annotations(os.path.join(args.output_dir, args.save))
    
    if args.coco:
        tool.convert_annotations_to_coco(os.path.join(args.output_dir, args.coco))

if __name__ == "__main__":
    main()
