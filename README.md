# Fruit and Vegetable Recognition Project - CMPE 462 Assignment 1

This project aims to develop a logistic regression classifier using a multi-modal dataset for fruit and vegetable recognition.

## Project Categories

- Banana
- Tomato
- Cucumber
- Mandarin
- Potato

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download SpaCy model (for text feature extraction):
```bash
python -m spacy download tr_core_news_sm
```
or for English:
```bash
python -m spacy download en_core_web_sm
```

## Project Structure

```
462/
├── data/                    # Datasets
│   ├── raw/                # Raw data
│   ├── processed/          # Processed data
│   ├── train/              # Training set
│   ├── test/               # Test set
│   └── val/                # Validation set
├── src/                    # Source code
│   ├── data_collection/    # Data collection modules
│   ├── feature_extraction/ # Feature extraction modules
│   ├── models/             # Model implementations
│   ├── evaluation/         # Evaluation modules
│   └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks
└── reports/                # Report files
```

## Usage

### Data Collection

```python
from src.data_collection.image_collector import ImageCollector
from src.data_collection.metadata_collector import MetadataCollector
from src.data_collection.text_collector import TextCollector

# Image collection
image_collector = ImageCollector()
image_collector.collect_images(categories=['banana', 'tomato', 'cucumber', 'mandarin', 'potato'])

# Metadata collection
metadata_collector = MetadataCollector()
metadata_collector.collect_metadata()

# Text data collection
text_collector = TextCollector()
text_collector.collect_descriptions()
```

### Feature Extraction

```python
from src.feature_extraction.image_features import extract_image_features
from src.feature_extraction.text_features import extract_text_features
from src.feature_extraction.feature_fusion import fuse_features

# Image features
image_features = extract_image_features(image_paths)

# Text features
text_features = extract_text_features(text_descriptions)

# Feature fusion
fused_features = fuse_features(image_features, text_features, metadata_features)
```

### Model Training

```python
from src.models.one_vs_all import OneVsAllClassifier
from src.models.logistic_regression import LogisticRegression

# Create model
model = OneVsAllClassifier(LogisticRegression, n_classes=5)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

## Reproducing Results

1. Prepare the dataset (place in data/ folder)
2. Run the `src/main.py` script:
```bash
python src/main.py
```

3. Or run notebooks sequentially:
   - `notebooks/01_data_exploration.ipynb`
   - `notebooks/02_feature_extraction.ipynb`
   - `notebooks/03_model_training.ipynb`

## Notes

- At least 50 samples per category should be collected manually
- Total dataset: 3000 samples (600 per category)
- Training set: 2500 samples
- Test set: 500 samples
- Validation set: 500 samples from training set

## License

This project was developed as part of the CMPE 462 course.
