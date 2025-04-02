# RenAIssance GSoC 2025

## Project Overview
This repository contains the implementation for the RenAIssance GSoC 2025 test, focusing on processing and analyzing Renaissance-era texts using modern AI techniques. The project is structured around three main tasks:

1. **Layout Organization Recognition**: Identifying and segmenting different regions in historical document scans
2. **Optical Character Recognition**: Transcribing historical text from images to machine-readable format
3. **Synthetic Renaissance Text Generation**: Generating realistic historical text images with period-appropriate degradation effects

## Setup Instructions

### Requirements
- Python 3.8+
- PyTorch 1.9+
- Other dependencies listed in `requirements.txt`

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/renaissance-gsoc-2025.git
cd renaissance-gsoc-2025

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
renaissance-gsoc-2025/
├── data/
│   ├── raw/                  # Raw Renaissance dataset
│   ├── processed/            # Processed data for each task
│   └── scripts/              # Data processing scripts
├── src/
│   ├── layout_recognition/   # Layout recognition module
│   │   ├── data_processing/  # Data processing for layout recognition
│   │   ├── models/           # Layout recognition models
│   │   │   ├── renaissance_model.py  # Original ResNet34-based model
│   │   │   ├── layoutlmv3_model.py   # LayoutLMv3-based model
│   │   │   └── unet_model.py         # U-Net-based model
│   │   ├── postprocessing/   # Post-processing utilities
│   │   ├── applications/     # Application scripts
│   │   ├── evaluation/       # Evaluation metrics and utilities
│   │   └── visualization/    # Visualization utilities
│   ├── ocr/                  # OCR module
│   │   ├── data_processing/  # Data processing for OCR
│   │   ├── models/           # OCR models (CRNN)
│   │   ├── training/         # Training scripts and utilities
│   │   ├── evaluation/       # Evaluation metrics (CER, WER)
│   │   └── visualization/    # Visualization utilities
│   └── synthetic_text/       # Synthetic text generation module
│       ├── data_processing/  # Data processing for text generation
│       ├── models/           # Diffusion models
│       ├── training/         # Training scripts and utilities
│       └── visualization/    # Visualization utilities
├── models/                   # Saved model checkpoints
├── results/                  # Evaluation results and visualizations
├── tests/                    # Test scripts for each module
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Data Processing

### Processing the Full Dataset

The first step is to process the Renaissance dataset for all three tasks:

```bash
# Process the Renaissance dataset
python data/scripts/process_renaissance_dataset.py --data_dir data/raw --output_dir data/processed

# Test the data processing pipeline
python tests/test_data_processing.py --data_dir data/raw --output_dir data/processed --visualize
```

This will:
1. Convert PDFs to images
2. Extract layout information
3. Extract text regions for OCR
4. Prepare text images for synthetic generation
5. Split the dataset into training, validation, and testing sets

### Processing the Test Data

For testing purposes, you can process just the test data:

```bash
# Process the Renaissance test data
python data/scripts/process_test_data.py \
    --source_dir data/raw/test_sources \
    --transcription_dir data/raw/test_transcriptions \
    --output_dir data/processed
```

This will:
1. Process source images for layout recognition
2. Create layout annotations (or use dummy annotations if none are available)
3. Process transcription files for OCR
4. Organize the data in the appropriate directory structure

## Training Models

### 1. Layout Recognition

```bash
# Train layout recognition model
python src/layout_recognition/training/train_renaissance_layout.py \
    --data_dir data/processed/layout \
    --output_dir models/layout/renaissance \
    --epochs 50 \
    --batch_size 8 \
    --learning_rate 1e-4
```

### 2. OCR

```bash
# Train OCR model
python src/ocr/training/train_renaissance_ocr.py \
    --data_dir data/processed/ocr \
    --output_dir models/ocr/renaissance \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4
```

### 3. Synthetic Text Generation

```bash
# Train diffusion model for synthetic text generation
python src/synthetic_text/training/train_synthetic_text.py \
    --data_dir data/processed/text_images \
    --output_dir models/synthetic_text \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --img_size 128
```

## Testing and Evaluation

## Layout Recognition Models

This project implements three different layout recognition models for text segmentation:

### 1. ResNet34-based Model (Original)

A binary segmentation model using ResNet34 as the backbone with a simple decoder head.

```bash
# Train the ResNet34 model
python train_renaissance_model.py --train_dir data/renaissance/train --val_dir data/renaissance/val --output_dir models/renaissance

# Test the ResNet34 model
python test_renaissance_model.py --model_path models/renaissance/best_model.pth --image_path data/renaissance/val/sample.png

# Run layout recognition on a directory
python run_layout_recognition.py --input_dir data/renaissance/val --output_dir results/renaissance_recognition
```

### 2. LayoutLMv3-based Model

A state-of-the-art model based on LayoutLMv3, which is specifically designed for document layout analysis.

```bash
# Test the LayoutLMv3 model
python test_layoutlmv3_model.py --image_path data/renaissance/val/sample.png --refine --adaptive_threshold

# Run layout recognition on a directory
python run_layoutlmv3_recognition.py --input_dir data/renaissance/val --output_dir results/layoutlmv3_recognition --refine --adaptive_threshold
```

### 3. U-Net Model

A U-Net-based model for binary segmentation, which is a popular architecture for image segmentation tasks.

```bash
# Train the U-Net model
python train_unet_model.py --train_dir data/renaissance/train --val_dir data/renaissance/val --output_dir models/unet --generate_pseudo_masks

# Test the U-Net model
python test_unet_model.py --model_path models/unet/best_model.pth --image_path data/renaissance/val/sample.png --refine --adaptive_threshold

# Run layout recognition on a directory
python run_unet_recognition.py --input_dir data/renaissance/val --output_dir results/unet_recognition --refine --adaptive_threshold
```

## Text Detection

The project includes an enhanced text detection implementation using the ResNet34 model with adaptive thresholding and post-processing refinements.

### Features

- **Adaptive Thresholding**: Automatically adjusts threshold values based on image content
- **Post-Processing Refinements**: Applies morphological operations to improve detection accuracy
- **Dark Area Focus**: Option to focus detection on dark areas (text is usually dark on light background)
- **Region Filtering**: Removes small noise regions while preserving text
- **Hole Filling**: Option to fill holes in text regions for better segmentation
- **Detailed Visualizations**: Generates visualizations of original images, probability maps, text masks, and overlays
- **HTML Report Generation**: Creates an HTML report with metrics and visualizations

### Usage

```bash
# Run text detection with default parameters
python run_text_detection.py --input_path data/renaissance/val --output_dir results/text_detection --model_path models/renaissance/best_model.pth

# Run text detection with enhanced features
python run_text_detection.py --input_path data/renaissance/val --output_dir results/text_detection_enhanced \
    --model_path models/renaissance/best_model.pth --device cpu --adaptive_threshold --refine --fill_holes

# Run text detection with focus on dark areas
python run_text_detection.py --input_path data/renaissance/val --output_dir results/text_detection_dark \
    --model_path models/renaissance/best_model.pth --device cpu --adaptive_threshold --refine --fill_holes --focus_dark_areas
```

## Testing and Evaluation

### 1. Layout Recognition Testing

```bash
# Test layout recognition model
python tests/test_layout_model.py \
    --data_dir data/processed/layout \
    --model_path models/layout/renaissance/best_model.pth \
    --output_dir results/layout \
    --num_samples 10
```

This will:
- Evaluate the layout recognition model on the test set
- Calculate IoU, Dice coefficient, precision, recall, and F1 score
- Visualize predictions and ground truth
- Generate a confusion matrix
- Save all results to the output directory

### 2. OCR Testing

```bash
# Test OCR model
python tests/test_ocr_model.py \
    --data_dir data/processed/ocr \
    --model_path models/ocr/renaissance/best_model.pth \
    --output_dir results/ocr \
    --num_samples 10
```

This will:
- Evaluate the OCR model on the test set
- Calculate Character Error Rate (CER) and Word Error Rate (WER)
- Visualize predictions and ground truth
- Identify worst-case examples
- Save all results to the output directory

### 3. Synthetic Text Generation Testing

```bash
# Test synthetic text generation
python tests/test_synthetic_text.py \
    --model_path models/synthetic_text/final_model.pt \
    --ocr_model_path models/ocr/renaissance/best_model.pth \
    --output_dir results/synthetic_text \
    --num_samples 20 \
    --img_size 128
```

This will:
- Generate synthetic Renaissance text images
- Evaluate image quality using diversity metrics
- If an OCR model is provided, test readability of generated images
- Visualize generated samples
- Save all results to the output directory

## Key Features

### Layout Recognition
- Implements a LayoutLM-based model for document layout analysis
- Segments Renaissance documents into regions (text, figures, margins, etc.)
- Provides comprehensive evaluation metrics (IoU, Dice, precision, recall, F1)
- Visualizes layout predictions with color-coded regions

### OCR
- Implements a CRNN (CNN + RNN) model for text recognition
- Handles historical fonts and printing imperfections
- Provides character-level and word-level accuracy metrics
- Includes attention visualization for model interpretability

### Synthetic Text Generation
- Implements a diffusion model for generating Renaissance-style text images
- Produces realistic historical printing effects and degradations
- Generates diverse samples that maintain period-appropriate characteristics
- Evaluates generation quality using both visual metrics and OCR readability

## License
[MIT License](LICENSE)

## Acknowledgements
This project is part of the Google Summer of Code 2025 application for the RenAIssance organization.
