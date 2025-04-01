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
│   │   ├── training/         # Training scripts and utilities
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
