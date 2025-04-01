#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mask R-CNN implementation for document layout analysis.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNNForLayoutAnalysis(nn.Module):
    """Mask R-CNN model for document layout analysis."""
    
    def __init__(self, num_classes=10, pretrained=True, min_size=800, max_size=1333):
        """
        Initialize Mask R-CNN for layout analysis.
        
        Args:
            num_classes (int): Number of classes (including background)
            pretrained (bool): Whether to use pretrained weights
            min_size (int): Minimum size of the image to be rescaled
            max_size (int): Maximum size of the image to be rescaled
        """
        super(MaskRCNNForLayoutAnalysis, self).__init__()
        
        # Load pre-trained model
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=pretrained,
            min_size=min_size,
            max_size=max_size
        )
        
        # Replace the pre-trained head with a new one
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
    
    def forward(self, images, targets=None):
        """
        Forward pass.
        
        Args:
            images (list): List of images
            targets (list, optional): List of targets
            
        Returns:
            dict or list: During training, returns a dict of losses.
                         During inference, returns a list of predicted boxes, labels, and masks.
        """
        return self.model(images, targets)

class DocumentLayoutDetector:
    """Wrapper class for document layout detection using Mask R-CNN."""
    
    def __init__(self, model, device, confidence_threshold=0.7):
        """
        Initialize the document layout detector.
        
        Args:
            model (nn.Module): Trained Mask R-CNN model
            device (torch.device): Device to run the model on
            confidence_threshold (float): Confidence threshold for detections
        """
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Set model to evaluation mode
        self.model.eval()
        self.model.to(device)
        
        # Define class names (adjust based on your dataset)
        self.class_names = [
            "background",
            "text",
            "title",
            "paragraph",
            "figure",
            "table",
            "marginalia",
            "header",
            "footer",
            "decoration"
        ]
    
    def predict(self, image):
        """
        Predict layout regions in an image.
        
        Args:
            image (PIL.Image or torch.Tensor): Input image
            
        Returns:
            dict: Predicted regions with boxes, masks, labels, and scores
        """
        # Convert PIL image to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = torchvision.transforms.functional.to_tensor(image)
        
        # Move image to device
        image = image.to(self.device)
        
        # Add batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model([image])[0]
        
        # Filter predictions by confidence
        keep = prediction['scores'] > self.confidence_threshold
        boxes = prediction['boxes'][keep].cpu().numpy()
        masks = prediction['masks'][keep].squeeze().cpu().numpy()
        labels = prediction['labels'][keep].cpu().numpy()
        scores = prediction['scores'][keep].cpu().numpy()
        
        # Convert labels to class names
        label_names = [self.class_names[label] for label in labels]
        
        return {
            'boxes': boxes,
            'masks': masks,
            'labels': labels,
            'label_names': label_names,
            'scores': scores
        }
    
    def visualize(self, image, prediction, output_path=None):
        """
        Visualize layout predictions.
        
        Args:
            image (PIL.Image): Input image
            prediction (dict): Prediction from the predict method
            output_path (str, optional): Path to save the visualization
            
        Returns:
            PIL.Image: Visualization image
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        from PIL import Image, ImageDraw
        
        # Convert PIL image to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Create figure and axes
        fig, ax = plt.subplots(1, figsize=(12, 12))
        
        # Display the image
        ax.imshow(image_np)
        
        # Define colors for different classes
        colors = [
            'red', 'green', 'blue', 'yellow', 'purple',
            'orange', 'cyan', 'magenta', 'lime', 'pink'
        ]
        
        # Draw bounding boxes and labels
        for i, (box, label_name, score) in enumerate(zip(
            prediction['boxes'], prediction['label_names'], prediction['scores']
        )):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=colors[i % len(colors)],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label and score
            ax.text(
                x1, y1 - 5,
                f"{label_name}: {score:.2f}",
                color=colors[i % len(colors)],
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7)
            )
        
        # Remove axes
        plt.axis('off')
        
        # Save visualization if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        
        plt.close()
        
        # Create a mask overlay
        overlay = Image.fromarray(image_np.copy())
        draw = ImageDraw.Draw(overlay, 'RGBA')
        
        for i, (mask, label) in enumerate(zip(prediction['masks'], prediction['labels'])):
            color = colors[i % len(colors)]
            
            # Convert color to RGBA with alpha
            r, g, b = ImageColor.getrgb(color)
            rgba = (r, g, b, 128)  # 50% opacity
            
            # Create binary mask
            binary_mask = mask > 0.5
            
            # Draw mask
            for y in range(binary_mask.shape[0]):
                for x in range(binary_mask.shape[1]):
                    if binary_mask[y, x]:
                        draw.point((x, y), fill=rgba)
        
        # Blend original image with overlay
        result = Image.blend(Image.fromarray(image_np), overlay, alpha=0.5)
        
        if output_path:
            result.save(output_path.replace('.png', '_mask.png'))
        
        return result

class MaskRCNNWithResNeXt(nn.Module):
    """Mask R-CNN with ResNeXt backbone for document layout analysis."""
    
    def __init__(self, num_classes=10, pretrained=True):
        """
        Initialize Mask R-CNN with ResNeXt backbone.
        
        Args:
            num_classes (int): Number of classes (including background)
            pretrained (bool): Whether to use pretrained weights
        """
        super(MaskRCNNWithResNeXt, self).__init__()
        
        # Load Mask R-CNN with ResNeXt-101-32x8d-FPN backbone
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=pretrained,
            pretrained_backbone=pretrained
        )
        
        # Replace the backbone with ResNeXt
        backbone = torchvision.models.resnet.resnext101_32x8d(pretrained=pretrained)
        backbone_layers = list(backbone.children())[:-2]  # Remove avg pool and fc
        self.model.backbone.body.conv1 = backbone.conv1
        self.model.backbone.body.bn1 = backbone.bn1
        self.model.backbone.body.relu = backbone.relu
        self.model.backbone.body.maxpool = backbone.maxpool
        self.model.backbone.body.layer1 = backbone.layer1
        self.model.backbone.body.layer2 = backbone.layer2
        self.model.backbone.body.layer3 = backbone.layer3
        self.model.backbone.body.layer4 = backbone.layer4
        
        # Replace the pre-trained head with a new one
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
    
    def forward(self, images, targets=None):
        """
        Forward pass.
        
        Args:
            images (list): List of images
            targets (list, optional): List of targets
            
        Returns:
            dict or list: During training, returns a dict of losses.
                         During inference, returns a list of predicted boxes, labels, and masks.
        """
        return self.model(images, targets)
