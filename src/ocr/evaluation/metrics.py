#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation metrics for OCR models.
"""

import numpy as np
import torch
import editdistance
from collections import defaultdict
import re
import string


def calculate_cer(predictions, targets):
    """
    Calculate Character Error Rate (CER).
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        
    Returns:
        float: Character Error Rate
    """
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(predictions, targets):
        # Calculate edit distance
        distance = editdistance.eval(pred, target)
        total_distance += distance
        total_length += len(target)
    
    # Calculate CER
    if total_length == 0:
        return 0.0
    
    cer = total_distance / total_length
    
    return cer


def calculate_wer(predictions, targets):
    """
    Calculate Word Error Rate (WER).
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        
    Returns:
        float: Word Error Rate
    """
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(predictions, targets):
        # Split into words
        pred_words = pred.split()
        target_words = target.split()
        
        # Calculate edit distance
        distance = editdistance.eval(pred_words, target_words)
        total_distance += distance
        total_length += len(target_words)
    
    # Calculate WER
    if total_length == 0:
        return 0.0
    
    wer = total_distance / total_length
    
    return wer


def calculate_accuracy(predictions, targets):
    """
    Calculate exact match accuracy.
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        
    Returns:
        float: Accuracy
    """
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    total = len(predictions)
    
    if total == 0:
        return 0.0
    
    accuracy = correct / total
    
    return accuracy


def calculate_normalized_edit_distance(predictions, targets, normalization='length'):
    """
    Calculate normalized edit distance.
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        normalization (str): Normalization method ('length' or 'max')
        
    Returns:
        float: Normalized edit distance
    """
    total_normalized_distance = 0.0
    
    for pred, target in zip(predictions, targets):
        distance = editdistance.eval(pred, target)
        
        if normalization == 'length':
            # Normalize by target length
            normalizer = max(len(target), 1)
        elif normalization == 'max':
            # Normalize by max length
            normalizer = max(len(pred), len(target), 1)
        else:
            raise ValueError(f"Unsupported normalization method: {normalization}")
        
        normalized_distance = distance / normalizer
        total_normalized_distance += normalized_distance
    
    # Calculate average
    avg_normalized_distance = total_normalized_distance / len(predictions) if predictions else 0.0
    
    return avg_normalized_distance


def calculate_per_class_metrics(predictions, targets, classes):
    """
    Calculate metrics per character class.
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        classes (list): List of character classes to evaluate
        
    Returns:
        dict: Dictionary of per-class metrics
    """
    class_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, target in zip(predictions, targets):
        # Align sequences
        alignment = needleman_wunsch_alignment(pred, target)
        
        for pred_char, target_char in alignment:
            if target_char in classes:
                class_metrics[target_char]['total'] += 1
                if pred_char == target_char:
                    class_metrics[target_char]['correct'] += 1
    
    # Calculate accuracy for each class
    results = {}
    for char, metrics in class_metrics.items():
        accuracy = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0.0
        results[char] = {
            'accuracy': accuracy,
            'correct': metrics['correct'],
            'total': metrics['total']
        }
    
    return results


def needleman_wunsch_alignment(seq1, seq2, match_score=1, mismatch_score=-1, gap_score=-1):
    """
    Perform Needleman-Wunsch alignment of two sequences.
    
    Args:
        seq1 (str): First sequence
        seq2 (str): Second sequence
        match_score (int): Score for matching characters
        mismatch_score (int): Score for mismatching characters
        gap_score (int): Score for gaps
        
    Returns:
        list: List of aligned character pairs
    """
    # Initialize score matrix
    n, m = len(seq1), len(seq2)
    score_matrix = np.zeros((n+1, m+1))
    
    # Initialize first row and column
    for i in range(n+1):
        score_matrix[i, 0] = i * gap_score
    for j in range(m+1):
        score_matrix[0, j] = j * gap_score
    
    # Fill score matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            match = score_matrix[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
            delete = score_matrix[i-1, j] + gap_score
            insert = score_matrix[i, j-1] + gap_score
            score_matrix[i, j] = max(match, delete, insert)
    
    # Traceback
    alignment = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score_matrix[i, j] == score_matrix[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score):
            alignment.append((seq1[i-1], seq2[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and score_matrix[i, j] == score_matrix[i-1, j] + gap_score:
            alignment.append((seq1[i-1], '-'))
            i -= 1
        else:
            alignment.append(('-', seq2[j-1]))
            j -= 1
    
    # Reverse alignment
    alignment.reverse()
    
    return alignment


def calculate_confusion_matrix(predictions, targets, vocab):
    """
    Calculate confusion matrix for character recognition.
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        vocab (list): List of characters in vocabulary
        
    Returns:
        np.ndarray: Confusion matrix
    """
    # Create mapping from character to index
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    
    # Initialize confusion matrix
    n_classes = len(vocab)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for pred, target in zip(predictions, targets):
        # Align sequences
        alignment = needleman_wunsch_alignment(pred, target)
        
        for pred_char, target_char in alignment:
            if target_char in char_to_idx and pred_char in char_to_idx:
                confusion_matrix[char_to_idx[target_char], char_to_idx[pred_char]] += 1
    
    return confusion_matrix


def calculate_precision_recall_f1(predictions, targets):
    """
    Calculate precision, recall, and F1 score for text recognition.
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        
    Returns:
        tuple: (precision, recall, f1)
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred, target in zip(predictions, targets):
        # Convert to sets of characters
        pred_chars = set(pred)
        target_chars = set(target)
        
        # Calculate metrics
        true_positives += len(pred_chars.intersection(target_chars))
        false_positives += len(pred_chars - target_chars)
        false_negatives += len(target_chars - pred_chars)
    
    # Calculate precision, recall, and F1
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    return precision, recall, f1


def calculate_normalized_levenshtein_similarity(predictions, targets):
    """
    Calculate normalized Levenshtein similarity.
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        
    Returns:
        float: Normalized Levenshtein similarity
    """
    total_similarity = 0.0
    
    for pred, target in zip(predictions, targets):
        distance = editdistance.eval(pred, target)
        max_length = max(len(pred), len(target))
        
        if max_length == 0:
            similarity = 1.0
        else:
            similarity = 1.0 - (distance / max_length)
        
        total_similarity += similarity
    
    # Calculate average
    avg_similarity = total_similarity / len(predictions) if predictions else 0.0
    
    return avg_similarity


def calculate_sequence_accuracy(predictions, targets):
    """
    Calculate sequence accuracy (exact match).
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        
    Returns:
        float: Sequence accuracy
    """
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    total = len(predictions)
    
    accuracy = correct / total if total > 0 else 0.0
    
    return accuracy


def calculate_partial_match_accuracy(predictions, targets, threshold=0.8):
    """
    Calculate partial match accuracy.
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        threshold (float): Similarity threshold for considering a match
        
    Returns:
        float: Partial match accuracy
    """
    matches = 0
    
    for pred, target in zip(predictions, targets):
        max_length = max(len(pred), len(target))
        
        if max_length == 0:
            similarity = 1.0
        else:
            distance = editdistance.eval(pred, target)
            similarity = 1.0 - (distance / max_length)
        
        if similarity >= threshold:
            matches += 1
    
    accuracy = matches / len(predictions) if predictions else 0.0
    
    return accuracy


def calculate_case_insensitive_accuracy(predictions, targets):
    """
    Calculate case-insensitive accuracy.
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        
    Returns:
        float: Case-insensitive accuracy
    """
    correct = sum(1 for pred, target in zip(predictions, targets) if pred.lower() == target.lower())
    total = len(predictions)
    
    accuracy = correct / total if total > 0 else 0.0
    
    return accuracy


def calculate_alphanumeric_accuracy(predictions, targets):
    """
    Calculate accuracy considering only alphanumeric characters.
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        
    Returns:
        float: Alphanumeric accuracy
    """
    def keep_alphanumeric(text):
        return ''.join(c for c in text if c.isalnum())
    
    pred_filtered = [keep_alphanumeric(pred) for pred in predictions]
    target_filtered = [keep_alphanumeric(target) for target in targets]
    
    correct = sum(1 for pred, target in zip(pred_filtered, target_filtered) if pred == target)
    total = len(predictions)
    
    accuracy = correct / total if total > 0 else 0.0
    
    return accuracy


def calculate_metrics_by_length(predictions, targets):
    """
    Calculate metrics grouped by text length.
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        
    Returns:
        dict: Dictionary of metrics by length
    """
    length_metrics = defaultdict(lambda: {'correct': 0, 'total': 0, 'cer': 0.0, 'wer': 0.0})
    
    for pred, target in zip(predictions, targets):
        length = len(target)
        length_metrics[length]['total'] += 1
        
        if pred == target:
            length_metrics[length]['correct'] += 1
        
        # Calculate CER and WER
        cer = calculate_cer([pred], [target])
        wer = calculate_wer([pred], [target])
        
        length_metrics[length]['cer'] += cer
        length_metrics[length]['wer'] += wer
    
    # Calculate averages
    results = {}
    for length, metrics in length_metrics.items():
        accuracy = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0.0
        avg_cer = metrics['cer'] / metrics['total'] if metrics['total'] > 0 else 0.0
        avg_wer = metrics['wer'] / metrics['total'] if metrics['total'] > 0 else 0.0
        
        results[length] = {
            'accuracy': accuracy,
            'cer': avg_cer,
            'wer': avg_wer,
            'correct': metrics['correct'],
            'total': metrics['total']
        }
    
    return results


def calculate_metrics_by_complexity(predictions, targets):
    """
    Calculate metrics grouped by text complexity.
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        
    Returns:
        dict: Dictionary of metrics by complexity
    """
    # Define complexity categories
    def get_complexity(text):
        has_uppercase = any(c.isupper() for c in text)
        has_digit = any(c.isdigit() for c in text)
        has_special = any(not c.isalnum() for c in text)
        
        if has_special:
            return 'complex'
        elif has_uppercase and has_digit:
            return 'medium'
        else:
            return 'simple'
    
    complexity_metrics = defaultdict(lambda: {'correct': 0, 'total': 0, 'cer': 0.0, 'wer': 0.0})
    
    for pred, target in zip(predictions, targets):
        complexity = get_complexity(target)
        complexity_metrics[complexity]['total'] += 1
        
        if pred == target:
            complexity_metrics[complexity]['correct'] += 1
        
        # Calculate CER and WER
        cer = calculate_cer([pred], [target])
        wer = calculate_wer([pred], [target])
        
        complexity_metrics[complexity]['cer'] += cer
        complexity_metrics[complexity]['wer'] += wer
    
    # Calculate averages
    results = {}
    for complexity, metrics in complexity_metrics.items():
        accuracy = metrics['correct'] / metrics['total'] if metrics['total'] > 0 else 0.0
        avg_cer = metrics['cer'] / metrics['total'] if metrics['total'] > 0 else 0.0
        avg_wer = metrics['wer'] / metrics['total'] if metrics['total'] > 0 else 0.0
        
        results[complexity] = {
            'accuracy': accuracy,
            'cer': avg_cer,
            'wer': avg_wer,
            'correct': metrics['correct'],
            'total': metrics['total']
        }
    
    return results


def calculate_detailed_metrics(predictions, targets):
    """
    Calculate detailed metrics for OCR evaluation.
    
    Args:
        predictions (list): List of predicted texts
        targets (list): List of target texts
        
    Returns:
        dict: Dictionary of detailed metrics
    """
    # Basic metrics
    cer = calculate_cer(predictions, targets)
    wer = calculate_wer(predictions, targets)
    accuracy = calculate_accuracy(predictions, targets)
    
    # Normalized metrics
    normalized_edit_distance = calculate_normalized_edit_distance(predictions, targets)
    normalized_similarity = calculate_normalized_levenshtein_similarity(predictions, targets)
    
    # Precision, recall, F1
    precision, recall, f1 = calculate_precision_recall_f1(predictions, targets)
    
    # Alternative accuracy metrics
    case_insensitive_accuracy = calculate_case_insensitive_accuracy(predictions, targets)
    alphanumeric_accuracy = calculate_alphanumeric_accuracy(predictions, targets)
    partial_match_accuracy = calculate_partial_match_accuracy(predictions, targets)
    
    # Metrics by text properties
    metrics_by_length = calculate_metrics_by_length(predictions, targets)
    metrics_by_complexity = calculate_metrics_by_complexity(predictions, targets)
    
    # Compile all metrics
    metrics = {
        'cer': cer,
        'wer': wer,
        'accuracy': accuracy,
        'normalized_edit_distance': normalized_edit_distance,
        'normalized_similarity': normalized_similarity,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'case_insensitive_accuracy': case_insensitive_accuracy,
        'alphanumeric_accuracy': alphanumeric_accuracy,
        'partial_match_accuracy': partial_match_accuracy,
        'by_length': metrics_by_length,
        'by_complexity': metrics_by_complexity
    }
    
    return metrics


class OCREvaluator:
    """Class for evaluating OCR models."""
    
    def __init__(self, idx_to_char=None):
        """
        Initialize evaluator.
        
        Args:
            idx_to_char (dict, optional): Index to character mapping
        """
        self.idx_to_char = idx_to_char
        self.reset()
    
    def reset(self):
        """Reset evaluation metrics."""
        self.predictions = []
        self.targets = []
        self.image_ids = []
    
    def update(self, predictions, targets, image_ids=None):
        """
        Update evaluation with new predictions.
        
        Args:
            predictions (list): List of predicted texts
            targets (list): List of target texts
            image_ids (list, optional): List of image IDs
        """
        self.predictions.extend(predictions)
        self.targets.extend(targets)
        
        if image_ids is not None:
            self.image_ids.extend(image_ids)
    
    def get_results(self):
        """
        Get evaluation results.
        
        Returns:
            dict: Evaluation metrics
        """
        if not self.predictions:
            return {
                'cer': 0.0,
                'wer': 0.0,
                'accuracy': 0.0
            }
        
        # Calculate detailed metrics
        metrics = calculate_detailed_metrics(self.predictions, self.targets)
        
        # Add per-sample metrics
        per_sample_metrics = []
        for i, (pred, target) in enumerate(zip(self.predictions, self.targets)):
            sample_cer = calculate_cer([pred], [target])
            sample_wer = calculate_wer([pred], [target])
            
            sample_metric = {
                'prediction': pred,
                'target': target,
                'cer': sample_cer,
                'wer': sample_wer,
                'correct': pred == target
            }
            
            if self.image_ids and i < len(self.image_ids):
                sample_metric['image_id'] = self.image_ids[i]
            
            per_sample_metrics.append(sample_metric)
        
        metrics['per_sample'] = per_sample_metrics
        
        # Sort samples by error rate
        sorted_by_cer = sorted(per_sample_metrics, key=lambda x: x['cer'], reverse=True)
        metrics['worst_samples'] = sorted_by_cer[:10]
        
        return metrics
    
    def print_results(self, detailed=False):
        """
        Print evaluation results.
        
        Args:
            detailed (bool): Whether to print detailed metrics
        """
        results = self.get_results()
        
        print(f"OCR Evaluation Results:")
        print(f"Character Error Rate (CER): {results['cer']:.4f}")
        print(f"Word Error Rate (WER): {results['wer']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        
        if detailed:
            print(f"\nDetailed Metrics:")
            print(f"Normalized Edit Distance: {results['normalized_edit_distance']:.4f}")
            print(f"Normalized Similarity: {results['normalized_similarity']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1 Score: {results['f1']:.4f}")
            print(f"Case-Insensitive Accuracy: {results['case_insensitive_accuracy']:.4f}")
            print(f"Alphanumeric Accuracy: {results['alphanumeric_accuracy']:.4f}")
            print(f"Partial Match Accuracy: {results['partial_match_accuracy']:.4f}")
            
            print(f"\nWorst Samples:")
            for i, sample in enumerate(results['worst_samples'][:5]):
                print(f"  {i+1}. Target: '{sample['target']}' | Prediction: '{sample['prediction']}' | CER: {sample['cer']:.4f}")
        
        return results
