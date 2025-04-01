#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for OCR model.
This script tests the OCR model on the Renaissance dataset.
"""

import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr.models.crnn import CRNN
from src.ocr.data_processing.dataset import LineOCRDataset
from src.ocr.evaluation.metrics import calculate_cer, calculate_wer
from src.ocr.visualization.visualizer import OCRVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test OCR model on Renaissance dataset')
    parser.add_argument('--data_dir', type=str, default='data/processed/ocr',
                        help='Directory containing processed OCR data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='results/ocr',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for testing')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for testing')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    return parser.parse_args()


def load_annotations(json_path):
    """
    Load annotations from JSON file.
    
    Args:
        json_path (str): Path to JSON file
        
    Returns:
        list: List of annotations
    """
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    return annotations


def create_char_map(annotations):
    """
    Create character map from annotations.
    
    Args:
        annotations (list): List of annotations
        
    Returns:
        tuple: (char_to_idx, idx_to_char)
    """
    # Collect all characters
    chars = set()
    for anno in annotations:
        text = anno['text']
        chars.update(text)
    
    # Create mapping
    chars = sorted(list(chars))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}  # Reserve 0 for blank
    char_to_idx['<blank>'] = 0
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    return char_to_idx, idx_to_char


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


def test_model(model, test_dataset, device, batch_size=16):
    """
    Test the model on a dataset.
    
    Args:
        model (nn.Module): OCR model
        test_dataset (Dataset): Test dataset
        device (torch.device): Device to use
        batch_size (int): Batch size
        
    Returns:
        dict: Test results
    """
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=test_dataset.get_collate_fn()
    )
    
    # Create decoder
    decoder = CTCDecoder(test_dataset.idx_to_char)
    
    # Test model
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # Get batch
            images = batch['images'].to(device)
            texts = batch['texts']
            
            # Forward pass
            outputs = model(images)
            
            # Decode predictions
            predictions = decoder.decode_greedy(outputs)
            
            # Store predictions and targets
            all_predictions.extend(predictions)
            all_targets.extend(texts)
    
    # Calculate metrics
    cer = calculate_cer(all_predictions, all_targets)
    wer = calculate_wer(all_predictions, all_targets)
    
    # Find worst cases
    worst_samples = []
    for i, (pred, target) in enumerate(zip(all_predictions, all_targets)):
        sample_cer = calculate_cer([pred], [target])
        
        if sample_cer > 0.5:  # High CER indicates poor performance
            worst_samples.append({
                'image_id': i,
                'prediction': pred,
                'target': target,
                'cer': sample_cer
            })
    
    # Sort worst samples by CER
    worst_samples = sorted(worst_samples, key=lambda x: x['cer'], reverse=True)
    
    # Prepare results
    results = {
        'cer': cer,
        'wer': wer,
        'predictions': all_predictions,
        'targets': all_targets,
        'worst_samples': worst_samples
    }
    
    return results


def visualize_results(results, test_dataset, output_dir, num_samples=10):
    """
    Visualize test results.
    
    Args:
        results (dict): Test results
        test_dataset (Dataset): Test dataset
        output_dir (str): Directory to save visualizations
        num_samples (int): Number of samples to visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = OCRVisualizer(output_dir=output_dir)
    
    # Visualize random samples
    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        # Get sample
        sample = test_dataset[idx]
        image = sample['image']
        target = sample['text']
        
        # Get prediction
        prediction = results['predictions'][idx]
        
        # Visualize
        visualizer.visualize_prediction(
            image=image.permute(1, 2, 0).cpu().numpy(),
            prediction=prediction,
            target=target,
            save_path=os.path.join(output_dir, f'sample_{i+1}.png')
        )
    
    # Visualize worst cases
    for i, sample in enumerate(results['worst_samples'][:min(5, len(results['worst_samples']))]):
        # Get image
        idx = sample['image_id']
        image = test_dataset[idx]['image']
        
        # Visualize
        visualizer.visualize_prediction(
            image=image.permute(1, 2, 0).cpu().numpy(),
            prediction=sample['prediction'],
            target=sample['target'],
            save_path=os.path.join(output_dir, f'worst_case_{i+1}.png')
        )
    
    # Visualize metrics
    visualizer.visualize_metrics(
        metrics=results,
        save_path=os.path.join(output_dir, 'metrics.png')
    )
    
    # Create confusion matrix
    visualizer.visualize_confusion_matrix(
        predictions=results['predictions'],
        targets=results['targets'],
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
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
    
    # Load annotations
    test_annotations = load_annotations(os.path.join(args.data_dir, 'test_annotations.json'))
    train_annotations = load_annotations(os.path.join(args.data_dir, 'train_annotations.json'))
    
    # Create character map
    char_to_idx, idx_to_char = create_char_map(train_annotations + test_annotations)
    
    # Create dataset
    test_dataset = LineOCRDataset(
        annotations=test_annotations,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        augment=False
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    
    # Create model
    model = CRNN(
        input_channels=3,
        output_classes=len(char_to_idx),
        hidden_size=256,
        num_layers=2,
        dropout=0.1,
        backbone='resnet18',
        bidirectional=True
    )
    
    # Load weights
    model.load_state_dict(model_state_dict)
    model.to(device)
    
    # Test model
    print("Testing model...")
    results = test_model(
        model=model,
        test_dataset=test_dataset,
        device=device,
        batch_size=args.batch_size
    )
    
    # Print results
    print(f"Character Error Rate (CER): {results['cer']:.4f}")
    print(f"Word Error Rate (WER): {results['wer']:.4f}")
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(
        results=results,
        test_dataset=test_dataset,
        output_dir=os.path.join(args.output_dir, 'visualizations'),
        num_samples=args.num_samples
    )
    
    # Save results
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        # Convert numpy values to Python types
        results_json = {
            'cer': float(results['cer']),
            'wer': float(results['wer']),
            'predictions': results['predictions'],
            'targets': results['targets'],
            'worst_samples': results['worst_samples']
        }
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
