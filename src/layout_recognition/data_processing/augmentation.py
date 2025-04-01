#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data augmentation techniques for layout recognition.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LayoutAugmentation:
    """Class for data augmentation specific to document layout analysis."""
    
    @staticmethod
    def get_train_transforms(target_size=(512, 512), p=0.5):
        """
        Get augmentation transforms for training.
        
        Args:
            target_size (tuple): Target size for resizing (height, width)
            p (float): Probability of applying each augmentation
            
        Returns:
            A.Compose: Composition of transforms
        """
        return A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p),
                A.RandomGamma(gamma_limit=(80, 120), p=p),
            ], p=p),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=p),
                A.MotionBlur(blur_limit=(3, 5), p=p),
            ], p=p),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=p),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=p),
            ], p=p),
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=p),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=p),
                A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=p),
            ], p=p),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=p,
                               border_mode=0, value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_val_transforms(target_size=(512, 512)):
        """
        Get transforms for validation/testing.
        
        Args:
            target_size (tuple): Target size for resizing (height, width)
            
        Returns:
            A.Compose: Composition of transforms
        """
        return A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_historical_document_transforms(target_size=(512, 512), p=0.5):
        """
        Get transforms specific to historical documents.
        
        Args:
            target_size (tuple): Target size for resizing (height, width)
            p (float): Probability of applying each augmentation
            
        Returns:
            A.Compose: Composition of transforms
        """
        return A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            # Simulate aging and degradation
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.1), contrast_limit=(-0.2, 0.1), p=p),
                A.RandomGamma(gamma_limit=(80, 120), p=p),
            ], p=p),
            # Simulate ink bleeding and fading
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=p),
                A.MotionBlur(blur_limit=(3, 7), p=p),
                A.MedianBlur(blur_limit=5, p=p),
            ], p=p),
            # Simulate noise and stains
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=p),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=p),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=p),
            ], p=p),
            # Simulate page warping and distortions
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=p),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=p),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=p),
            ], p=p),
            # Simulate slight rotation and misalignment
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=p,
                               border_mode=0, value=0),
            # Simulate color variations in old documents
            A.OneOf([
                A.ToSepia(p=p),
                A.ToGray(p=p),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=p),
            ], p=p),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_cutmix_transform(dataset, alpha=1.0):
        """
        Implement CutMix augmentation for layout segmentation.
        
        Args:
            dataset: The dataset to sample from
            alpha (float): Parameter for beta distribution
            
        Returns:
            function: CutMix function
        """
        def cutmix(sample1):
            """
            Apply CutMix to a sample.
            
            Args:
                sample1 (dict): Original sample
                
            Returns:
                dict: Augmented sample
            """
            # Randomly select another sample
            idx2 = np.random.randint(0, len(dataset))
            sample2 = dataset[idx2]
            
            # Get images and masks
            image1, mask1 = sample1["image"], sample1["mask"]
            image2, mask2 = sample2["image"], sample2["mask"]
            
            # Generate random box
            lam = np.random.beta(alpha, alpha)
            
            _, h, w = image1.shape
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(w * cut_rat)
            cut_h = int(h * cut_rat)
            
            cx = np.random.randint(w)
            cy = np.random.randint(h)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby2 = np.clip(cy + cut_h // 2, 0, h)
            
            # Apply CutMix
            image_mixed = image1.clone()
            mask_mixed = mask1.clone()
            
            image_mixed[:, bby1:bby2, bbx1:bbx2] = image2[:, bby1:bby2, bbx1:bbx2]
            mask_mixed[:, bby1:bby2, bbx1:bbx2] = mask2[:, bby1:bby2, bbx1:bbx2]
            
            # Update sample
            sample1["image"] = image_mixed
            sample1["mask"] = mask_mixed
            
            return sample1
        
        return cutmix
