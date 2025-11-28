# Fruit and Vegetable Recognition

A multimodal machine learning project for classifying fruits and vegetables using logistic regression with three feature types: images, text descriptions, and categorical/numerical metadata.

## Features

- **Image Features**: Color histograms, LBP texture, GLCM texture, and statistical features
- **Text Features**: Sentence embeddings using Sentence-BERT (all-MiniLM-L6-v2)
- **Metadata Features**: Weight, color, season, and origin (one-hot encoded)
- **Multimodal Fusion**: Combined feature representation with weighted fusion
- **Custom Implementation**: One-vs-All logistic regression from scratch
- **Comparison**: Performance comparison with scikit-learn

## Dataset Structure

```
dataset/
├── images/
│   ├── original/          # Original images by category
│   └── generated/         # Augmented images by category
├── raw/
│   ├── metadata.csv       # Categorical/numerical attributes
│   └── description.csv    # Text descriptions
└── processed/
    ├── X_final.npy        # Feature vectors (499 dimensions)
    └── y_final.npy        # Labels
```

**Categories**: Banana, Cucumber, Mandarin, Potato, Tomato

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Feature Extraction
```bash
cd src
python raw_to_vector.py
```
Processes all modalities and creates feature vectors in `dataset/processed/`.

### 2. Train Models
```bash
cd src/logistic_regression
python train_logistic_regression.py
```
Trains logistic regression models for each feature type and saves to `results/`.

### 3. Evaluate Models
```bash
python evaluate_metrics.py
```
Calculates accuracy, precision, recall, F1-score, and AUC metrics.

### 4. Compare with Scikit-learn
```bash
python compare_with_sklearn.py
```
Compares custom implementation with scikit-learn and generates visualization plots.

## Results

All results are saved in the `results/` directory:
- `trained_models.pkl` - Trained model objects
- `metrics/` - Performance metrics (CSV files)
- `loss_plots/` - Training/validation loss curves
- `comparison/` - Comparison results and plots

## Model Configuration

- **Learning Rate**: 0.1
- **Max Iterations**: 1000
- **Regularization**: L2 (λ=0.1)
- **Feature Dimensions**: 499 (18 metadata + 280 text + 201 image)
- **Train/Val/Test Split**: ~72% / 14% / 14%

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Project Structure

```
FruitAndVegetableRecognition/
├── dataset/              # Data files
├── src/                  # Source code
│   ├── models/          # Logistic regression implementation
│   └── logistic_regression/  # Training and evaluation scripts
├── results/             # Output files and plots
└── requirements.txt     # Dependencies
```

