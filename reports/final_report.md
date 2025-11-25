# Fruit and Vegetable Recognition Project - Final Report
## CMPE 462 Assignment 1

---

## 1. (a) Dataset Description (5 points)

### Dataset Overview

This project aims to recognize 5 different fruit and vegetable categories:

1. **Banana**
2. **Tomato**
3. **Cucumber**
4. **Mandarin**
5. **Potato**

- **Total number of samples:** 3000
- **Number of samples per category:** 600
- **Training set:** 2500 samples
- **Test set:** 500 samples
- **Validation set:** 500 samples

### Example from Each Category

#### Banana

- **Sample ID:** banana_0000
- **Weight:** 218.54 grams
- **Color:** white
- **Season:** summer
- **Origin:** local
- **Description:** Vitamin ve mineral açısından zengin ürün.
- **Image Path:** data/raw/images/banana/banana_1.jpg

#### Tomato

- **Sample ID:** tomato_0000
- **Weight:** 234.2 grams
- **Color:** orange
- **Season:** winter
- **Origin:** tropical
- **Description:** Doğal koşullarda yetiştirilmiş ürün.
- **Image Path:** data/raw/images/tomato/tomato_1.jpg

#### Cucumber

- **Sample ID:** cucumber_0000
- **Weight:** 331.81 grams
- **Color:** white
- **Season:** winter
- **Origin:** imported
- **Description:** Taze, sağlıklı ve besleyici bir ürün.
- **Image Path:** data/raw/images/cucumber/cucumber_1.jpg

#### Mandarin

- **Sample ID:** mandarin_0000
- **Weight:** 58.62 grams
- **Color:** red
- **Season:** autumn
- **Origin:** imported
- **Description:** Sağlıklı beslenme için önemli bir gıda maddesi.
- **Image Path:** data/raw/images/mandarin/mandarin_1.jpg

#### Potato

- **Sample ID:** potato_0000
- **Weight:** 197.53 grams
- **Color:** yellow
- **Season:** spring
- **Origin:** local
- **Description:** Besleyici değeri yüksek, lezzetli bir gıda maddesi.
- **Image Path:** data/raw/images/potato/potato_1.jpg

---

## 1. (b) Data Collection and Pre-processing (5 points)

### Data Collection Procedure

The data collection process consists of three main components:

1. **Metadata Collection (Categorical and Numerical Features):**
   - 600 samples were generated for each category
   - Categorical features: color, season, origin
   - Numerical features: weight
   - Category-specific value ranges were used for each feature
   - 5% of outliers were intentionally added

2. **Text Description Generation:**
   - Natural language descriptions were generated for each sample
   - Descriptions contain category-specific characteristics
   - Used as training data for the Word2Vec model

3. **Image Collection:**
   - Images are stored in category folders under `data/raw/images/`
   - Total 125 images found
   - Images are optional; if not available, only text and metadata features are used

### Data Pre-processing Steps

1. **Image Pre-processing:**
   - Images were converted from BGR to RGB format
   - Images were resized to 224x224 pixels
   - Pixel values were normalized to the range [0, 1]

2. **Text Pre-processing:**
   - Descriptions were tokenized
   - Word2Vec model was trained on all descriptions

3. **Categorical Features:**
   - One-hot encoding was applied
   - Categories: color, season, origin

4. **Numerical Features:**
   - Standardization (z-score normalization) was applied
   - Feature: weight

5. **Data Splitting:**
   - Stratified splitting was used
   - Random state: 42 (for reproducibility)

---

## 1. (c) Feature Extraction Procedure (30 points)

### Feature Selection Rationale

Features from different modalities were selected:

- **Image features:** Capture visual appearance of objects
- **Text features:** Capture semantic information
- **Categorical features:** Capture structural information
- **Numerical features:** Capture physical properties

### Image Feature Extraction

**Constraint:** Deep learning models were not used.

Classical computer vision methods were used:

1. **HOG (Histogram of Oriented Gradients):**
   - Histogram of oriented gradients
   - Parameters: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)
   - Captures shape and edge information of objects

2. **LBP (Local Binary Pattern):**
   - Local binary patterns
   - Parameters: radius=3, n_points=24, method='uniform'
   - Captures texture features

3. **Color Histogram:**
   - Color histograms
   - Parameters: bins=32, color_space='rgb'
   - Captures color distribution

**Image feature dimension:** 26366 dimensions

### Text Feature Extraction

**Allowed:** Deep learning-based word embedding models can be used.

**Method used:** Word2Vec (Gensim library)

- **Model type:** Continuous Bag of Words (CBOW)
- **Embedding dimension:** 100
- **Window size:** 5
- **Min count:** 2
- **Model training:** Trained on all descriptions
- **Feature extraction:** Average of word embeddings for each description

**Text feature dimension:** 100 dimensions

### Categorical Feature Encoding

- **Method:** One-hot encoding
- **Features:** color, season, origin
- **Feature dimension:** 13 dimensions

### Numerical Feature Normalization

- **Method:** Standardization (z-score normalization)
- **Feature:** weight
- **Feature dimension:** 1 dimensions

### Feature Fusion Strategy

Features from different modalities were combined:

**Method:** Concatenation (Simple concatenation)

1. All feature matrices must have the same number of samples
2. Features are concatenated horizontally (hstack)
3. Result: A single feature vector

**Alternative approaches (can be tried in the future):**
- Weighted fusion
- Dimensionality reduction with PCA
- Late fusion

**Fused feature dimension:** 26480 dimensions
**Total number of samples:** 3000 samples

---

## 1. (d) Similarity Reporting and Outlier Detection (20 points)

### Similarity Reporting Strategy

The following strategy was used for similarity analysis:

**Metric Used:** Cosine Similarity

Reasons for choosing cosine similarity:
- Works well in high-dimensional feature spaces
- Focuses on the direction rather than magnitude of features
- Suitable for normalized features

**Alternative metric:** Euclidean distance (normalized)

### Intra-Class Similarity Results

Intra-class similarity measures how similar samples within the same class are to each other.

- **Banana:** 0.7639
- **Tomato:** 0.5506
- **Cucumber:** 0.8589
- **Mandarin:** 0.8455
- **Potato:** 0.5788

**Average Intra-Class Similarity:** 0.7195

### Inter-Class Similarity Results

Inter-class similarity matrix:

| Class | Banana | Tomato | Cucumber | Mandarin | Potato |
|-------|---|---|---|---|---|
| Banana | 1.0000 | 0.7218 | 0.8237 | 0.8166 | 0.5923 |
| Tomato | 0.7218 | 1.0000 | 0.7213 | 0.7161 | 0.5692 |
| Cucumber | 0.8237 | 0.7213 | 1.0000 | 0.8089 | 0.6117 |
| Mandarin | 0.8166 | 0.7161 | 0.8089 | 1.0000 | 0.6243 |
| Potato | 0.5923 | 0.5692 | 0.6117 | 0.6243 | 1.0000 |

### Dataset Difficulty Analysis

- **Average Intra-Class Similarity:** 0.7195
- **Average Inter-Class Similarity:** 0.7006
- **Separability Score:** 0.0189
- **Difficulty Level:** zor

**Comment:**
The dataset is difficult. Inter-class similarity is high, which makes distinction difficult. More advanced feature extraction or model architecture may be required.

### Outlier Detection Strategy

The following approach was used for outlier detection:

**Method Used:** IQR (Interquartile Range) Method

Reasons for choosing IQR method:
- Simple and understandable
- Applicable to multidimensional data
- Parameter: factor=1.5 (standard value)

**Alternative methods:**
- Z-score method (threshold=3.0)
- Isolation Forest (contamination=0.1)
- Elliptic Envelope (contamination=0.1)

### Outlier Detection Results

**General Statistics:**
- Total number of samples: 2000
- Number of detected outliers: 2000
- Outlier ratio: 100.00%

**Per-Class Outlier Analysis:**

- **Banana:**
  - Total samples: 400
  - Number of outliers: 400
  - Outlier ratio: 100.00%

- **Tomato:**
  - Total samples: 400
  - Number of outliers: 400
  - Outlier ratio: 100.00%

- **Cucumber:**
  - Total samples: 400
  - Number of outliers: 400
  - Outlier ratio: 100.00%

- **Mandarin:**
  - Total samples: 400
  - Number of outliers: 400
  - Outlier ratio: 100.00%

- **Potato:**
  - Total samples: 400
  - Number of outliers: 400
  - Outlier ratio: 100.00%

### Impact of Outliers

Outliers can affect model performance:
- During model training, outliers may reduce the model's generalization ability
- However, in some cases, outliers may represent real data diversity
- In this project, outliers were intentionally added at a rate of 5%
- The model is expected to be robust against these outliers

---

# Task 2: Logistic Regression Classifier

## Task 2(a): Implementation from Scratch (25 points)

### Implementation Details

We implemented logistic regression classifier from scratch using the one-vs-all approach for multiclass classification.

**Key Features:**
- Binary logistic regression with sigmoid activation
- Gradient descent optimization
- L2 regularization support
- One-vs-all extension for 5 classes

### Training and Validation Loss Plots

Training and validation loss plots have been generated for each feature set:

- `loss_history_image_only.png` - Image features only
- `loss_history_categorical_numerical_only.png` - Categorical and numerical features only
- `loss_history_text_only.png` - Text features only
- `loss_history_fused.png` - Fused features

### Model Training Results

**Image Only:**
- Training time: 97.5003 seconds
- Test accuracy: 0.1900

**Categorical Numerical Only:**
- Training time: 2.0909 seconds
- Test accuracy: 0.2020

**Text Only:**
- Training time: 8.4497 seconds
- Test accuracy: 0.2180

**Fused:**
- Training time: 101.3385 seconds
- Test accuracy: 1.0000

## Task 2(b): Classification Metrics (10 points)

### Performance Comparison

The following table compares the performance of classifiers trained on different feature sets:

| Feature Set | Accuracy | Precision | Recall | F1-Score | AUC |
|-------------|----------|-----------|--------|----------|-----|
| Image Only | 0.1900 | 0.0722 | 0.1900 | 0.0920 | 0.4972 |
| Categorical Numerical Only | 0.2020 | 0.2015 | 0.2020 | 0.2010 | 0.4991 |
| Text Only | 0.2180 | 0.2285 | 0.2180 | 0.1886 | 0.5423 |
| Fused | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Detailed Metrics

**Image Only:**
- Accuracy: 0.1900
- Precision: 0.0722
- Recall: 0.1900
- F1_score: 0.0920
- Auc: 0.4972

**Categorical Numerical Only:**
- Accuracy: 0.2020
- Precision: 0.2015
- Recall: 0.2020
- F1_score: 0.2010
- Auc: 0.4991

**Text Only:**
- Accuracy: 0.2180
- Precision: 0.2285
- Recall: 0.2180
- F1_score: 0.1886
- Auc: 0.5423

**Fused:**
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1_score: 1.0000
- Auc: 1.0000

## Task 2(c): Comparison with Sklearn (5 points)

### Performance Comparison

Our implementation was compared with Scikit-learn's LogisticRegression:

**Note:** Detailed comparison metrics are shown in the console output during execution.
The comparison includes:
- Accuracy, Precision, Recall, F1-Score, and AUC metrics
- Runtime performance comparison
- Model behavior analysis

---

## Conclusion

This report covers dataset description, data collection and preprocessing, feature extraction, similarity analysis, outlier detection, and logistic regression classifier implementation and evaluation.
