#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for synthetic Renaissance text generation.
This script tests the diffusion model for generating synthetic Renaissance-style text images.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthetic_text.models.diffusion_model import UNet, DiffusionModel
from src.ocr.models.crnn import CRNN
from src.ocr.data_processing.dataset import LineOCRDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test synthetic Renaissance text generation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained diffusion model')
    parser.add_argument('--ocr_model_path', type=str, default=None,
                        help='Path to trained OCR model for evaluation')
    parser.add_argument('--output_dir', type=str, default='results/synthetic_text',
                        help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for generation')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size for generation')
    parser.add_argument('--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for generation')
    parser.add_argument('--reference_dir', type=str, default=None,
                        help='Directory containing reference images for comparison')
    return parser.parse_args()


class CTCDecoder:
    """CTC decoder for OCR model."""
    
    def __init__(self, idx_to_char):
        """
        Initialize CTC decoder.
        
        Args:
            idx_to_char (dict): Mapping from index to character
        """
        self.idx_to_char = idx_to_char
    
    def decode_greedy(self, logits):
        """
        Greedy decoding.
        
        Args:
            logits (torch.Tensor): Model output logits
            
        Returns:
            list: List of decoded texts
        """
        # Get predictions
        preds = torch.argmax(logits, dim=2).detach().cpu().numpy()
        
        # Decode predictions
        texts = []
        for pred in preds:
            text = self._decode_sequence(pred)
            texts.append(text)
        
        return texts
    
    def _decode_sequence(self, sequence):
        """
        Decode a sequence of indices.
        
        Args:
            sequence (numpy.ndarray): Sequence of indices
            
        Returns:
            str: Decoded text
        """
        # Remove duplicates
        previous = -1
        result = []
        
        for i in sequence:
            if i != previous and i != 0:  # Skip blank
                result.append(self.idx_to_char[i])
            previous = i
        
        return ''.join(result)


def evaluate_with_ocr(samples, ocr_model, device, idx_to_char):
    """
    Evaluate generated samples with OCR model.
    
    Args:
        samples (torch.Tensor): Generated samples
        ocr_model (nn.Module): OCR model
        device (torch.device): Device to use
        idx_to_char (dict): Mapping from index to character
        
    Returns:
        list: List of recognized texts
    """
    # Create decoder
    decoder = CTCDecoder(idx_to_char)
    
    # Evaluate samples
    ocr_model.eval()
    recognized_texts = []
    
    with torch.no_grad():
        # Process in batches
        batch_size = 8
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            
            # Convert to RGB if needed (OCR model might expect 3 channels)
            if batch.shape[1] == 1 and ocr_model.cnn.conv1.in_channels == 3:
                batch = batch.repeat(1, 3, 1, 1)
            
            # Forward pass
            outputs = ocr_model(batch.to(device))
            
            # Decode predictions
            texts = decoder.decode_greedy(outputs)
            recognized_texts.extend(texts)
    
    return recognized_texts


def calculate_image_metrics(generated_samples, reference_samples=None):
    """
    Calculate image quality metrics.
    
    Args:
        generated_samples (torch.Tensor): Generated samples
        reference_samples (torch.Tensor, optional): Reference samples
        
    Returns:
        dict: Image quality metrics
    """
    # Convert to numpy and normalize to [0, 1]
    generated_np = generated_samples.cpu().numpy()
    generated_np = (generated_np + 1) / 2  # [-1, 1] -> [0, 1]
    
    # Calculate internal diversity (mean pairwise SSIM)
    num_samples = len(generated_np)
    pairwise_ssims = []
    
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            img1 = generated_np[i].squeeze()
            img2 = generated_np[j].squeeze()
            
            # Calculate SSIM
            ssim_val = ssim(img1, img2, data_range=1.0)
            pairwise_ssims.append(ssim_val)
    
    diversity = 1.0 - np.mean(pairwise_ssims) if pairwise_ssims else 0.0
    
    # Calculate reference metrics if reference samples are provided
    reference_metrics = {}
    if reference_samples is not None:
        reference_np = reference_samples.cpu().numpy()
        reference_np = (reference_np + 1) / 2  # [-1, 1] -> [0, 1]
        
        # Calculate FID (simplified version using SSIM and PSNR)
        ssim_values = []
        psnr_values = []
        
        for i in range(min(num_samples, len(reference_np))):
            gen_img = generated_np[i].squeeze()
            ref_img = reference_np[i].squeeze()
            
            # Calculate SSIM
            ssim_val = ssim(gen_img, ref_img, data_range=1.0)
            ssim_values.append(ssim_val)
            
            # Calculate PSNR
            psnr_val = psnr(ref_img, gen_img, data_range=1.0)
            psnr_values.append(psnr_val)
        
        reference_metrics = {
            'ssim': np.mean(ssim_values),
            'psnr': np.mean(psnr_values)
        }
    
    # Prepare metrics
    metrics = {
        'diversity': diversity,
        'reference': reference_metrics
    }
    
    return metrics


def visualize_samples(samples, output_dir, texts=None):
    """
    Visualize generated samples.
    
    Args:
        samples (torch.Tensor): Generated samples
        output_dir (str): Directory to save visualizations
        texts (list, optional): Recognized texts
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy and normalize to [0, 255]
    samples_np = samples.cpu().numpy()
    samples_np = (samples_np + 1) / 2  # [-1, 1] -> [0, 1]
    samples_np = (samples_np * 255).astype(np.uint8)
    
    # Visualize individual samples
    for i, sample in enumerate(samples_np):
        # Convert to PIL image
        img = Image.fromarray(sample.squeeze(), mode='L')
        
        # Save image
        if texts is not None:
            img.save(os.path.join(output_dir, f'sample_{i+1}_{texts[i]}.png'))
        else:
            img.save(os.path.join(output_dir, f'sample_{i+1}.png'))
    
    # Create grid visualization
    num_samples = len(samples_np)
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    plt.figure(figsize=(15, 15))
    
    for i, sample in enumerate(samples_np):
        if i >= num_samples:
            break
        
        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(sample.squeeze(), cmap='gray')
        
        if texts is not None:
            plt.title(texts[i], fontsize=8)
        
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples_grid.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    model = UNet(
        in_channels=1,
        out_channels=1,
        time_emb_dim=256,
        base_channels=64,
        channel_mults=(1, 2, 4, 8)
    )
    
    # Create diffusion model
    diffusion = DiffusionModel(
        model=model,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=args.timesteps,
        device=device
    )
    
    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load OCR model if provided
    ocr_model = None
    idx_to_char = None
    
    if args.ocr_model_path:
        print(f"Loading OCR model from {args.ocr_model_path}")
        ocr_checkpoint = torch.load(args.ocr_model_path, map_location=device)
        
        # Get character map
        if 'idx_to_char' in ocr_checkpoint:
            idx_to_char = ocr_checkpoint['idx_to_char']
        else:
            # Create a simple character map
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?()-'\""
            idx_to_char = {i+1: char for i, char in enumerate(chars)}
            idx_to_char[0] = '<blank>'
        
        # Create OCR model
        ocr_model = CRNN(
            input_channels=3,
            output_classes=len(idx_to_char),
            hidden_size=256,
            num_layers=2,
            dropout=0.1,
            backbone='resnet18',
            bidirectional=True
        )
        
        # Load weights
        if 'model_state_dict' in ocr_checkpoint:
            ocr_model.load_state_dict(ocr_checkpoint['model_state_dict'])
        else:
            ocr_model.load_state_dict(ocr_checkpoint)
        
        ocr_model.to(device)
    
    # Load reference samples if provided
    reference_samples = None
    if args.reference_dir:
        print(f"Loading reference samples from {args.reference_dir}")
        reference_files = [f for f in os.listdir(args.reference_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if reference_files:
            reference_samples = []
            
            for file in reference_files[:args.num_samples]:
                img_path = os.path.join(args.reference_dir, file)
                img = Image.open(img_path).convert('L')
                img = img.resize((args.img_size, args.img_size))
                
                # Convert to tensor and normalize to [-1, 1]
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                img_tensor = img_tensor * 2.0 - 1.0
                img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                
                reference_samples.append(img_tensor)
            
            reference_samples = torch.cat(reference_samples, dim=0)
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    samples = diffusion.generate_samples(
        num_samples=args.num_samples,
        img_size=(1, args.img_size, args.img_size),
        batch_size=args.batch_size
    )
    
    # Evaluate with OCR if model is provided
    recognized_texts = None
    if ocr_model is not None and idx_to_char is not None:
        print("Evaluating samples with OCR model...")
        recognized_texts = evaluate_with_ocr(samples, ocr_model, device, idx_to_char)
        
        # Print recognized texts
        print("Recognized texts:")
        for i, text in enumerate(recognized_texts):
            print(f"Sample {i+1}: {text}")
    
    # Calculate image metrics
    print("Calculating image metrics...")
    metrics = calculate_image_metrics(samples, reference_samples)
    
    # Print metrics
    print(f"Diversity: {metrics['diversity']:.4f}")
    
    if metrics['reference']:
        print(f"SSIM: {metrics['reference']['ssim']:.4f}")
        print(f"PSNR: {metrics['reference']['psnr']:.4f}")
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'diversity': float(metrics['diversity']),
            'reference': {
                'ssim': float(metrics['reference']['ssim']) if metrics['reference'] else None,
                'psnr': float(metrics['reference']['psnr']) if metrics['reference'] else None
            }
        }, f, indent=2)
    
    # Visualize samples
    print("Visualizing samples...")
    visualize_samples(
        samples=samples,
        output_dir=os.path.join(args.output_dir, 'samples'),
        texts=recognized_texts
    )
    
    # Save samples
    torch.save(samples, os.path.join(args.output_dir, 'generated_samples.pt'))
    
    print(f"Testing completed. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
