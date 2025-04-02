#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Renaissance Layout Recognition Application

A simplified version focusing on detecting main text regions in the Renaissance dataset.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

def process_images(model_path, data_dir, output_dir, batch_size=4):
    """
    Process images using the layout recognition model.
    
    Args:
        model_path (str): Path to trained model
        data_dir (str): Directory containing processed data
        output_dir (str): Directory to save results
        batch_size (int): Batch size for inference
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    image_paths = _load_images(data_dir)
    
    if not image_paths:
        print(f"Error: No images found in {data_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Import model here to avoid circular imports
    from layout_recognition.models.renaissance_model import create_renaissance_model
    
    # Load model
    print("Loading model...")
    try:
        # Try to load with the Renaissance-specific model
        model = create_renaissance_model(pretrained_path=model_path, device=device)
        print("Using Renaissance-specific layout model")
    except Exception as e:
        print(f"Warning: Could not load Renaissance model: {e}")
        print("Falling back to generic LayoutLM model")
        
        # Fall back to the generic model
        from layout_recognition.models.layoutlm import LayoutLM
        model = LayoutLM(
            input_channels=3,
            num_classes=2,  # binary classification: text vs non-text
            backbone='resnet34'
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
        
        # Try to load weights, handling potential mismatches
        try:
            model.load_state_dict(model_state_dict)
        except Exception as e:
            print(f"Warning: Could not load weights directly: {e}")
            print("Attempting to load weights with shape adaptation...")
            
            # Get model state dict
            model_dict = model.state_dict()
            
            # Filter out incompatible keys
            pretrained_dict = {k: v for k, v in model_state_dict.items() 
                              if k in model_dict and model_dict[k].shape == v.shape}
            
            # Update model dict with filtered pretrained dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")
    
    model.to(device)
    
    # Run inference
    print("Running inference...")
    results = _run_inference(
        model=model,
        image_paths=image_paths,
        device=device,
        batch_size=batch_size
    )
    
    # Visualize results
    print("Visualizing results...")
    _visualize_results(results, output_dir)
    
    # Save results
    print("Saving results...")
    _save_results(results, output_dir)
    
    print(f"Results saved to {output_dir}")

def _load_images(data_dir):
    """
    Load images from the data directory.
    
    Args:
        data_dir (str): Directory containing images
        
    Returns:
        list: List of image paths
    """
    image_dir = os.path.join(data_dir, 'layout', 'images')
    if not os.path.exists(image_dir):
        print(f"Warning: Image directory {image_dir} not found, using the processed data directory")
        # Try to find images directly in the processed data directory
        image_paths = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths
    
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    return image_paths

def _preprocess_image(image_path):
    """
    Preprocess image for model input.
    
    Args:
        image_path (str): Path to image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize to a standard size
    image = image.resize((512, 512))
    
    # Convert to tensor and normalize
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
    
    # Normalize with ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    return image_tensor, image

def _run_inference(model, image_paths, device, batch_size=4):
    """
    Run inference on images.
    
    Args:
        model (nn.Module): Layout recognition model
        image_paths (list): List of image paths
        device (torch.device): Device to use for inference
        batch_size (int): Batch size for inference
        
    Returns:
        list: List of prediction results
    """
    model.eval()
    results = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc='Processing images'):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        original_images = []
        
        # Preprocess images
        for image_path in batch_paths:
            try:
                image_tensor, original_image = _preprocess_image(image_path)
                batch_images.append(image_tensor)
                original_images.append(original_image)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Stack images into batch
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(batch_tensor)
        
        # Process outputs - use softmax to get probabilities
        probs = F.softmax(outputs, dim=1)
        
        # For binary classification, we can use a threshold on the text class probability
        # This gives us more control than just argmax
        text_probs = probs[:, 1]  # Class 1 is text
        preds = (text_probs > 0.5).long()  # Threshold at 0.5
        
        # Add results
        for j, (image_path, original_image) in enumerate(zip(batch_paths[:len(batch_images)], original_images)):
            pred = preds[j].cpu().numpy()
            
            # Resize prediction back to original image size if needed
            if pred.shape != (512, 512):
                from scipy.ndimage import zoom
                zoom_factors = (original_image.size[1] / pred.shape[0], original_image.size[0] / pred.shape[1])
                pred = zoom(pred, zoom_factors, order=0)
            
            # Get probability map for visualization
            prob_map = text_probs[j].cpu().numpy()
            
            # Resize probability map to original image size if needed
            if prob_map.shape != (512, 512):
                from scipy.ndimage import zoom
                zoom_factors = (original_image.size[1] / prob_map.shape[0], original_image.size[0] / prob_map.shape[1])
                prob_map = zoom(prob_map, zoom_factors, order=1)  # Use order=1 for smoother interpolation
            
            # Create result dictionary
            result = {
                'image_path': image_path,
                'prediction': pred,
                'probability_map': prob_map,
                'original_image': original_image
            }
            
            results.append(result)
    
    return results

def _visualize_results(results, output_dir):
    """
    Visualize layout recognition results.
    
    Args:
        results (list): List of prediction results
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Class colors for visualization (non-text, text)
    class_colors = [
        [0, 0, 0],      # non-text - black
        [255, 0, 0],    # text - red
    ]
    
    # Class names
    class_names = ['Non-text', 'Text']
    
    # Visualize each result
    for i, result in enumerate(results):
        # Get original image and prediction
        original_image = result['original_image']
        pred = result['prediction']
        
        # Get probability map if available
        prob_map = result.get('probability_map', None)
        
        # Convert to numpy arrays for visualization
        image_np = np.array(original_image)
        
        # Create color mask for visualization
        color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for c in range(len(class_colors)):
            color_mask[pred == c] = class_colors[c]
        
        # Create overlay
        alpha = 0.5
        overlay = image_np.copy()
        for c in range(1, len(class_colors)):  # Skip background
            overlay[pred == c] = (
                alpha * np.array(class_colors[c]) + 
                (1 - alpha) * image_np[pred == c]
            ).astype(np.uint8)
        
        # Determine number of subplots based on available data
        n_plots = 4 if prob_map is not None else 3
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        
        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # If probability map is available, show it
        if prob_map is not None:
            # Show text probability map (class 1)
            text_prob = prob_map[1] if len(prob_map.shape) > 2 else prob_map
            im = axes[1].imshow(text_prob, cmap='viridis', vmin=0, vmax=1)
            axes[1].set_title('Text Probability')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Class segmentation
            axes[2].imshow(color_mask)
            axes[2].set_title('Text Mask')
            axes[2].axis('off')
            
            # Overlay
            axes[3].imshow(overlay)
            axes[3].set_title('Overlay')
            axes[3].axis('off')
        else:
            # Class segmentation
            axes[1].imshow(color_mask)
            axes[1].set_title('Text Mask')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=np.array(class_colors[c])/255, 
                  label=class_names[c])
            for c in range(len(class_colors))
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=len(class_colors))
        
        # Set title for the figure
        fig.suptitle(f"Layout Recognition: {os.path.basename(result['image_path'])}")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'result_{i:03d}.png'), bbox_inches='tight')
        plt.close()
    
    # Create summary with metrics
    class_proportions = {}
    for c in range(len(class_names)):
        class_proportions[class_names[c]] = []
    
    for result in results:
        pred = result['prediction']
        for c in range(len(class_names)):
            proportion = np.sum(pred == c) / pred.size
            class_proportions[class_names[c]].append(proportion)
    
    # Create bar chart of average class proportions
    plt.figure(figsize=(10, 6))
    avg_proportions = [np.mean(class_proportions[name]) for name in class_names]
    plt.bar(class_names, avg_proportions, color=[np.array(c)/255 for c in class_colors])
    plt.ylim(0, 1)
    plt.ylabel('Average Proportion')
    plt.title('Average Class Distribution')
    plt.savefig(os.path.join(vis_dir, 'class_distribution.png'))
    plt.close()

def _save_results(results, output_dir):
    """
    Save layout recognition results.
    
    Args:
        results (list): List of prediction results
        output_dir (str): Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    class_names = ['Background', 'Text', 'Figure', 'Margin']
    class_counts = {name: 0 for name in class_names}
    total_pixels = 0
    
    for result in results:
        pred = result['prediction']
        total_pixels += pred.size
        for c, name in enumerate(class_names):
            class_counts[name] += np.sum(pred == c)
    
    # Calculate proportions
    class_proportions = {name: count / total_pixels for name, count in class_counts.items()}
    
    # Generate report - convert NumPy types to Python native types for JSON serialization
    report = {
        'num_images': len(results),
        'class_counts': {name: int(count) for name, count in class_counts.items()},
        'class_proportions': {name: float(prop) for name, prop in class_proportions.items()}
    }
    
    # Save report to JSON
    with open(os.path.join(output_dir, 'layout_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create a text report
    with open(os.path.join(output_dir, 'layout_report.txt'), 'w') as f:
        f.write('Renaissance Layout Recognition Report\n')
        f.write('====================================\n\n')
        f.write(f'Total images processed: {len(results)}\n\n')
        f.write('Class Distribution:\n')
        for name, count in class_counts.items():
            f.write(f'  - {name}: {count} pixels ({class_proportions[name]:.2%})\n')
        f.write('\nNote: This layout recognition model focuses on detecting main text regions\n')
        f.write('in early modern printed sources, ignoring marginalia and embellishments.\n')
