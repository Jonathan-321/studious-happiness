#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train OCR model on Renaissance dataset for GSoC 2025 submission.
This script trains a CRNN model on the Renaissance dataset for the OCR task.
"""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

from src.ocr.data_processing.dataset import LineOCRDataset
from src.ocr.models.crnn import CRNN
from src.ocr.training.trainer import OCRTrainer
from src.ocr.training.config import get_resnet_crnn_config
from src.ocr.evaluation.metrics import OCREvaluator
from src.ocr.visualization.visualizer import OCRVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train OCR model on Renaissance dataset')
    parser.add_argument('--data_dir', type=str, default='data/processed/ocr',
                        help='Directory containing processed OCR data')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='models/ocr/renaissance',
                        help='Directory to save model and results')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training')
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


def main():
    """Main function."""
    args = parse_args()
    
    # Load annotations
    train_annotations = load_annotations(os.path.join(args.data_dir, 'train_annotations.json'))
    val_annotations = load_annotations(os.path.join(args.data_dir, 'val_annotations.json'))
    test_annotations = load_annotations(os.path.join(args.data_dir, 'test_annotations.json'))
    
    # Create character map
    char_to_idx, idx_to_char = create_char_map(train_annotations + val_annotations)
    print(f"Character set size: {len(char_to_idx)}")
    
    # Create datasets
    train_dataset = LineOCRDataset(
        annotations=train_annotations,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        augment=True
    )
    
    val_dataset = LineOCRDataset(
        annotations=val_annotations,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        augment=False
    )
    
    test_dataset = LineOCRDataset(
        annotations=test_annotations,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        augment=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create configuration
    if args.config_file:
        config = get_resnet_crnn_config()
        config.load_from_file(args.config_file)
    else:
        config = get_resnet_crnn_config()
    
    # Update configuration with command line arguments
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['output_dir'] = args.output_dir
    
    if args.device:
        config['device'] = args.device
    
    # Create model
    model = CRNN(
        input_channels=config['input_channels'],
        output_classes=len(char_to_idx),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        backbone=config['backbone'],
        bidirectional=config['bidirectional']
    )
    
    # Create trainer
    trainer = OCRTrainer(
        model=model,
        config=config.to_dict(),
        device=torch.device(config['device'])
    )
    
    # Train model
    print(f"Starting training on {config['device']}...")
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        resume_from=args.resume_from
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Print test results
    print(f"Test CER: {test_results['cer']:.4f}")
    print(f"Test WER: {test_results['wer']:.4f}")
    
    # Visualize results
    visualizer = OCRVisualizer(output_dir=os.path.join(config['output_dir'], 'visualizations'))
    
    # Visualize test samples
    print("Visualizing test samples...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=test_dataset.get_collate_fn()
    )
    
    # Get a batch of test samples
    batch = next(iter(test_loader))
    images = batch['images']
    texts = batch['texts']
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(trainer.device))
        predictions = trainer.decoder.decode_greedy(outputs)
    
    # Visualize predictions
    for i in range(min(8, len(images))):
        image = test_dataset.denormalize(images[i]).cpu().numpy().transpose(1, 2, 0)
        prediction = predictions[i]
        target = texts[i]
        
        visualizer.visualize_prediction(
            image=image,
            prediction=prediction,
            target=target,
            save_path=os.path.join(config['output_dir'], 'visualizations', f'test_sample_{i}.png')
        )
    
    # Visualize worst cases
    worst_samples = test_results['worst_samples']
    print("Visualizing worst cases...")
    
    for i, sample in enumerate(worst_samples[:5]):
        image_path = test_dataset.annotations[sample['image_id']]['image_path']
        image = Image.open(image_path).convert('RGB')
        
        visualizer.visualize_prediction(
            image=image,
            prediction=sample['prediction'],
            target=sample['target'],
            save_path=os.path.join(config['output_dir'], 'visualizations', f'worst_case_{i}.png')
        )
    
    # Visualize metrics
    visualizer.visualize_metrics(
        metrics=test_results,
        save_path=os.path.join(config['output_dir'], 'visualizations', 'metrics.png')
    )
    
    print(f"Training and evaluation completed. Results saved to {config['output_dir']}")


if __name__ == '__main__':
    main()
