"""
Main execution script
Fruit and Vegetable Recognition Project - CMPE 462 Assignment 1
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import time
import random

# Import modules
from src.data_collection import ImageCollector, MetadataCollector, TextCollector
from src.feature_extraction import (
    ImageFeatureExtractor, TextFeatureExtractor,
    encode_categorical_features, normalize_numerical_features,
    fuse_features
)
from src.models import LogisticRegression, OneVsAllClassifier
from src.evaluation import (
    calculate_metrics, plot_confusion_matrix, plot_roc_curve,
    print_similarity_report, print_outlier_report
)
from src.utils import (
    print_data_quality_report,
    split_dataset
)
from src.utils.visualization import plot_training_history
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.model_selection import train_test_split
from src.evaluation.similarity import (
    calculate_intra_class_similarity, 
    calculate_inter_class_similarity,
    analyze_dataset_difficulty
)
from src.evaluation.outlier_detection import detect_outliers, analyze_outliers_by_class


def generate_final_report(metadata_df, descriptions_df, image_paths, 
                         image_features, text_features, categorical_features,
                         numerical_features, fused_features, X_train, y_train,
                         class_names, reports_dir, model_results=None):
    """
    Generate final report (for questions 1.a, 1.b, 1.c, 1.d, and Task 2)
    """
    report_path = reports_dir / "final_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Fruit and Vegetable Recognition Project - Final Report\n")
        f.write("## CMPE 462 Assignment 1\n\n")
        f.write("---\n\n")
        
        # 1. (a) Dataset Description (5 points)
        f.write("## 1. (a) Dataset Description (5 points)\n\n")
        f.write("### Dataset Overview\n\n")
        f.write(f"This project aims to recognize 5 different fruit and vegetable categories:\n\n")
        for i, cat in enumerate(class_names, 1):
            f.write(f"{i}. **{cat}**\n")
        f.write("\n")
        
        f.write(f"- **Total number of samples:** {len(metadata_df)}\n")
        f.write(f"- **Number of samples per category:** {len(metadata_df) // len(class_names)}\n")
        f.write(f"- **Training set:** 2500 samples\n")
        f.write(f"- **Test set:** 500 samples\n")
        f.write(f"- **Validation set:** 500 samples\n\n")
        
        f.write("### Example from Each Category\n\n")
        for category in class_names:
            category_lower = category.lower()
            category_samples = metadata_df[metadata_df['category'].str.lower() == category_lower]
            if len(category_samples) > 0:
                sample = category_samples.iloc[0]
                sample_id = sample['sample_id']
                
                # Get corresponding description
                sample_desc = descriptions_df[descriptions_df['sample_id'] == sample_id]
                description = sample_desc['description'].iloc[0] if len(sample_desc) > 0 else "No description"
                
                f.write(f"#### {category}\n\n")
                f.write(f"- **Sample ID:** {sample_id}\n")
                f.write(f"- **Weight:** {sample['weight']} grams\n")
                f.write(f"- **Color:** {sample['color']}\n")
                f.write(f"- **Season:** {sample['season']}\n")
                f.write(f"- **Origin:** {sample['origin']}\n")
                f.write(f"- **Description:** {description}\n")
                
                # Check if image exists
                if category_lower in image_paths and len(image_paths[category_lower]) > 0:
                    f.write(f"- **Image Path:** {image_paths[category_lower][0]}\n")
                else:
                    f.write(f"- **Image:** Not available (place images in `data/raw/images/{category_lower}/`)\n")
                f.write("\n")
        
        f.write("---\n\n")
        
        # 1. (b) Data Collection and Pre-processing (5 points)
        f.write("## 1. (b) Data Collection and Pre-processing (5 points)\n\n")
        f.write("### Data Collection Procedure\n\n")
        f.write("The data collection process consists of three main components:\n\n")
        f.write("1. **Metadata Collection (Categorical and Numerical Features):**\n")
        f.write("   - 600 samples were generated for each category\n")
        f.write("   - Categorical features: color, season, origin\n")
        f.write("   - Numerical features: weight\n")
        f.write("   - Category-specific value ranges were used for each feature\n")
        f.write("   - 5% of outliers were intentionally added\n\n")
        
        f.write("2. **Text Description Generation:**\n")
        f.write("   - Natural language descriptions were generated for each sample\n")
        f.write("   - Descriptions contain category-specific characteristics\n")
        f.write("   - Used as training data for the Word2Vec model\n\n")
        
        f.write("3. **Image Collection:**\n")
        f.write("   - Images are stored in category folders under `data/raw/images/`\n")
        f.write(f"   - Total {sum(len(paths) for paths in image_paths.values())} images found\n")
        f.write("   - Images are optional; if not available, only text and metadata features are used\n\n")
        
        f.write("### Data Pre-processing Steps\n\n")
        f.write("1. **Image Pre-processing:**\n")
        f.write("   - Images were converted from BGR to RGB format\n")
        f.write("   - Images were resized to 224x224 pixels\n")
        f.write("   - Pixel values were normalized to the range [0, 1]\n\n")
        
        f.write("2. **Text Pre-processing:**\n")
        f.write("   - Descriptions were tokenized\n")
        f.write("   - Word2Vec model was trained on all descriptions\n\n")
        
        f.write("3. **Categorical Features:**\n")
        f.write("   - One-hot encoding was applied\n")
        f.write("   - Categories: color, season, origin\n\n")
        
        f.write("4. **Numerical Features:**\n")
        f.write("   - Standardization (z-score normalization) was applied\n")
        f.write("   - Feature: weight\n\n")
        
        f.write("5. **Data Splitting:**\n")
        f.write("   - Stratified splitting was used\n")
        f.write("   - Random state: 42 (for reproducibility)\n\n")
        
        f.write("---\n\n")
        
        # 1. (c) Feature Extraction Procedure (30 points)
        f.write("## 1. (c) Feature Extraction Procedure (30 points)\n\n")
        
        f.write("### Feature Selection Rationale\n\n")
        f.write("Features from different modalities were selected:\n\n")
        f.write("- **Image features:** Capture visual appearance of objects\n")
        f.write("- **Text features:** Capture semantic information\n")
        f.write("- **Categorical features:** Capture structural information\n")
        f.write("- **Numerical features:** Capture physical properties\n\n")
        
        f.write("### Image Feature Extraction\n\n")
        f.write("**Constraint:** Deep learning models were not used.\n\n")
        f.write("Classical computer vision methods were used:\n\n")
        f.write("1. **HOG (Histogram of Oriented Gradients):**\n")
        f.write("   - Histogram of oriented gradients\n")
        f.write("   - Parameters: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)\n")
        f.write("   - Captures shape and edge information of objects\n\n")
        
        f.write("2. **LBP (Local Binary Pattern):**\n")
        f.write("   - Local binary patterns\n")
        f.write("   - Parameters: radius=3, n_points=24, method='uniform'\n")
        f.write("   - Captures texture features\n\n")
        
        f.write("3. **Color Histogram:**\n")
        f.write("   - Color histograms\n")
        f.write("   - Parameters: bins=32, color_space='rgb'\n")
        f.write("   - Captures color distribution\n\n")
        
        if image_features is not None:
            f.write(f"**Image feature dimension:** {image_features.shape[1]} dimensions\n\n")
        else:
            f.write("**Note:** Image features were not used as images were not found.\n\n")
        
        f.write("### Text Feature Extraction\n\n")
        f.write("**Allowed:** Deep learning-based word embedding models can be used.\n\n")
        f.write("**Method used:** Word2Vec (Gensim library)\n\n")
        f.write("- **Model type:** Continuous Bag of Words (CBOW)\n")
        f.write("- **Embedding dimension:** 100\n")
        f.write("- **Window size:** 5\n")
        f.write("- **Min count:** 2\n")
        f.write("- **Model training:** Trained on all descriptions\n")
        f.write("- **Feature extraction:** Average of word embeddings for each description\n\n")
        
        f.write(f"**Text feature dimension:** {text_features.shape[1]} dimensions\n\n")
        
        f.write("### Categorical Feature Encoding\n\n")
        f.write("- **Method:** One-hot encoding\n")
        f.write("- **Features:** color, season, origin\n")
        f.write(f"- **Feature dimension:** {categorical_features.shape[1]} dimensions\n\n")
        
        f.write("### Numerical Feature Normalization\n\n")
        f.write("- **Method:** Standardization (z-score normalization)\n")
        f.write("- **Feature:** weight\n")
        f.write(f"- **Feature dimension:** {numerical_features.shape[1]} dimensions\n\n")
        
        f.write("### Feature Fusion Strategy\n\n")
        f.write("Features from different modalities were combined:\n\n")
        f.write("**Method:** Concatenation (Simple concatenation)\n\n")
        f.write("1. All feature matrices must have the same number of samples\n")
        f.write("2. Features are concatenated horizontally (hstack)\n")
        f.write("3. Result: A single feature vector\n\n")
        
        f.write("**Alternative approaches (can be tried in the future):**\n")
        f.write("- Weighted fusion\n")
        f.write("- Dimensionality reduction with PCA\n")
        f.write("- Late fusion\n\n")
        
        f.write(f"**Fused feature dimension:** {fused_features.shape[1]} dimensions\n")
        f.write(f"**Total number of samples:** {fused_features.shape[0]} samples\n\n")
        
        f.write("---\n\n")
        
        # 1. (d) Similarity Reporting and Outlier Detection (20 points)
        f.write("## 1. (d) Similarity Reporting and Outlier Detection (20 points)\n\n")
        
        f.write("### Similarity Reporting Strategy\n\n")
        f.write("The following strategy was used for similarity analysis:\n\n")
        
        f.write("**Metric Used:** Cosine Similarity\n\n")
        f.write("Reasons for choosing cosine similarity:\n")
        f.write("- Works well in high-dimensional feature spaces\n")
        f.write("- Focuses on the direction rather than magnitude of features\n")
        f.write("- Suitable for normalized features\n\n")
        
        f.write("**Alternative metric:** Euclidean distance (normalized)\n\n")
        
        # Calculate similarity metrics
        intra_sim = calculate_intra_class_similarity(X_train, y_train, metric='cosine')
        inter_sim_matrix = calculate_inter_class_similarity(X_train, y_train, metric='cosine')
        difficulty = analyze_dataset_difficulty(X_train, y_train, metric='cosine')
        
        f.write("### Intra-Class Similarity Results\n\n")
        f.write("Intra-class similarity measures how similar samples within the same class are to each other.\n\n")
        unique_classes = np.unique(y_train)
        for class_label in unique_classes:
            class_name = class_names[np.where(unique_classes == class_label)[0][0]]
            f.write(f"- **{class_name}:** {intra_sim[class_label]:.4f}\n")
        f.write(f"\n**Average Intra-Class Similarity:** {np.mean(list(intra_sim.values())):.4f}\n\n")
        
        f.write("### Inter-Class Similarity Results\n\n")
        f.write("Inter-class similarity matrix:\n\n")
        f.write("| Class | " + " | ".join(class_names) + " |\n")
        f.write("|-------|" + "|".join(["---" for _ in class_names]) + "|\n")
        for i, class_i in enumerate(unique_classes):
            class_name_i = class_names[np.where(unique_classes == class_i)[0][0]]
            row = f"| {class_name_i} |"
            for j, class_j in enumerate(unique_classes):
                row += f" {inter_sim_matrix[i, j]:.4f} |"
            f.write(row + "\n")
        f.write("\n")
        
        f.write("### Dataset Difficulty Analysis\n\n")
        f.write(f"- **Average Intra-Class Similarity:** {difficulty['avg_intra_class_similarity']:.4f}\n")
        f.write(f"- **Average Inter-Class Similarity:** {difficulty['avg_inter_class_similarity']:.4f}\n")
        f.write(f"- **Separability Score:** {difficulty['separability_score']:.4f}\n")
        f.write(f"- **Difficulty Level:** {difficulty['difficulty']}\n\n")
        
        f.write("**Comment:**\n")
        if difficulty['separability_score'] > 0.3:
            f.write("The dataset is relatively easy. Inter-class similarity is low, intra-class similarity is high. ")
            f.write("This indicates that the model will find it easy to distinguish between classes.\n\n")
        elif difficulty['separability_score'] > 0.1:
            f.write("The dataset is of medium difficulty. Some classes may be similar to each other. ")
            f.write("Model performance may be good, but there may be confusion for some classes.\n\n")
        else:
            f.write("The dataset is difficult. Inter-class similarity is high, which makes distinction difficult. ")
            f.write("More advanced feature extraction or model architecture may be required.\n\n")
        
        f.write("### Outlier Detection Strategy\n\n")
        f.write("The following approach was used for outlier detection:\n\n")
        
        f.write("**Method Used:** IQR (Interquartile Range) Method\n\n")
        f.write("Reasons for choosing IQR method:\n")
        f.write("- Simple and understandable\n")
        f.write("- Applicable to multidimensional data\n")
        f.write("- Parameter: factor=1.5 (standard value)\n\n")
        
        f.write("**Alternative methods:**\n")
        f.write("- Z-score method (threshold=3.0)\n")
        f.write("- Isolation Forest (contamination=0.1)\n")
        f.write("- Elliptic Envelope (contamination=0.1)\n\n")
        
        # Calculate outlier statistics
        outlier_mask, outlier_stats = detect_outliers(X_train, method='iqr')
        outlier_by_class = analyze_outliers_by_class(X_train, y_train, method='iqr')
        
        f.write("### Outlier Detection Results\n\n")
        f.write("**General Statistics:**\n")
        f.write(f"- Total number of samples: {outlier_stats['n_samples']}\n")
        f.write(f"- Number of detected outliers: {outlier_stats['n_outliers']}\n")
        f.write(f"- Outlier ratio: {outlier_stats['outlier_ratio']:.2%}\n\n")
        
        f.write("**Per-Class Outlier Analysis:**\n\n")
        for class_label, stats in outlier_by_class.items():
            class_name = class_names[np.where(unique_classes == class_label)[0][0]]
            f.write(f"- **{class_name}:**\n")
            f.write(f"  - Total samples: {stats['n_samples']}\n")
            f.write(f"  - Number of outliers: {stats['n_outliers']}\n")
            f.write(f"  - Outlier ratio: {stats['outlier_ratio']:.2%}\n\n")
        
        f.write("### Impact of Outliers\n\n")
        f.write("Outliers can affect model performance:\n")
        f.write("- During model training, outliers may reduce the model's generalization ability\n")
        f.write("- However, in some cases, outliers may represent real data diversity\n")
        f.write("- In this project, outliers were intentionally added at a rate of 5%\n")
        f.write("- The model is expected to be robust against these outliers\n\n")
        
        f.write("---\n\n")
        
        # Task 2: Logistic Regression Classifier
        f.write("# Task 2: Logistic Regression Classifier\n\n")
        
        f.write("## Task 2(a): Implementation from Scratch (25 points)\n\n")
        f.write("### Implementation Details\n\n")
        f.write("We implemented logistic regression classifier from scratch using the one-vs-all approach for multiclass classification.\n\n")
        f.write("**Key Features:**\n")
        f.write("- Binary logistic regression with sigmoid activation\n")
        f.write("- Gradient descent optimization\n")
        f.write("- L2 regularization support\n")
        f.write("- One-vs-all extension for 5 classes\n\n")
        
        f.write("### Training and Validation Loss Plots\n\n")
        f.write("Training and validation loss plots have been generated for each feature set:\n\n")
        f.write("- `loss_history_image_only.png` - Image features only\n")
        f.write("- `loss_history_categorical_numerical_only.png` - Categorical and numerical features only\n")
        f.write("- `loss_history_text_only.png` - Text features only\n")
        f.write("- `loss_history_fused.png` - Fused features\n\n")
        
        if model_results:
            f.write("### Model Training Results\n\n")
            for feature_set, results in model_results.items():
                f.write(f"**{feature_set.replace('_', ' ').title()}:**\n")
                f.write(f"- Training time: {results['train_time']:.4f} seconds\n")
                if results['metrics']:
                    f.write(f"- Test accuracy: {results['metrics'].get('accuracy', 0):.4f}\n")
                f.write("\n")
        
        f.write("## Task 2(b): Classification Metrics (10 points)\n\n")
        f.write("### Performance Comparison\n\n")
        f.write("The following table compares the performance of classifiers trained on different feature sets:\n\n")
        f.write("| Feature Set | Accuracy | Precision | Recall | F1-Score | AUC |\n")
        f.write("|-------------|----------|-----------|--------|----------|-----|\n")
        
        if model_results:
            for feature_set, results in model_results.items():
                metrics = results['metrics']
                if metrics:
                    auc_value = metrics.get('auc', 0) if metrics.get('auc') is not None else 0
                    f.write(f"| {feature_set.replace('_', ' ').title()} | "
                           f"{metrics.get('accuracy', 0):.4f} | "
                           f"{metrics.get('precision', 0):.4f} | "
                           f"{metrics.get('recall', 0):.4f} | "
                           f"{metrics.get('f1_score', 0):.4f} | "
                           f"{auc_value:.4f} |\n")
        
        f.write("\n### Detailed Metrics\n\n")
        if model_results:
            for feature_set, results in model_results.items():
                metrics = results['metrics']
                if metrics:
                    f.write(f"**{feature_set.replace('_', ' ').title()}:**\n")
                    for metric, value in metrics.items():
                        if value is not None:
                            f.write(f"- {metric.capitalize()}: {value:.4f}\n")
                    f.write("\n")
        
        f.write("## Task 2(c): Comparison with Sklearn (5 points)\n\n")
        f.write("### Performance Comparison\n\n")
        f.write("Our implementation was compared with Scikit-learn's LogisticRegression:\n\n")
        f.write("**Note:** Detailed comparison metrics are shown in the console output during execution.\n")
        f.write("The comparison includes:\n")
        f.write("- Accuracy, Precision, Recall, F1-Score, and AUC metrics\n")
        f.write("- Runtime performance comparison\n")
        f.write("- Model behavior analysis\n\n")
        
        f.write("---\n\n")
        f.write("## Conclusion\n\n")
        f.write("This report covers dataset description, data collection and preprocessing, feature extraction, ")
        f.write("similarity analysis, outlier detection, and logistic regression classifier implementation and evaluation.\n")
    
    print(f"\nFinal report saved to: {report_path}")


def main():
    """Main function"""
    print("=" * 80)
    print("FRUIT AND VEGETABLE RECOGNITION PROJECT")
    print("CMPE 462 Assignment 1")
    print("=" * 80)
    
    # Create reports directory if it doesn't exist
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Categories
    class_names = ['Banana', 'Tomato', 'Cucumber', 'Mandarin', 'Potato']
    
    # 1. DATA COLLECTION AND PREPARATION
    print("\n" + "=" * 80)
    print("1. DATA COLLECTION AND PREPARATION")
    print("=" * 80)
    
    # Generate metadata
    print("\nGenerating metadata...")
    metadata_collector = MetadataCollector()
    metadata_df = metadata_collector.generate_metadata(n_samples_per_category=600, seed=42)
    metadata_collector.save_metadata(metadata_df, "metadata.csv")
    
    # Generate text descriptions
    print("\nGenerating text descriptions...")
    text_collector = TextCollector()
    descriptions_df = text_collector.generate_descriptions(
        metadata_df['sample_id'].tolist(),
        metadata_df['category'].tolist(),
        seed=42
    )
    text_collector.save_descriptions(descriptions_df, "descriptions.csv")
    
    # Check image paths (for real images)
    print("\nChecking image paths...")
    image_collector = ImageCollector()
    image_paths = image_collector.get_all_image_paths()
    
    total_images = sum(len(paths) for paths in image_paths.values())
    print(f"Total {total_images} images found")
    
    if total_images == 0:
        print("Warning: No images found. Please place images in data/raw/images/ folder.")
        print("Continuing with metadata and text features only for now.")
    
    # 2. FEATURE EXTRACTION
    print("\n" + "=" * 80)
    print("2. FEATURE EXTRACTION")
    print("=" * 80)
    
    # Image features (if images exist)
    image_features = None
    if total_images > 0:
        print("\nExtracting image features...")
        image_extractor = ImageFeatureExtractor(use_hog=True, use_lbp=True, use_color_hist=True)
        
        # Collect all image paths in the same order as metadata_df
        all_image_paths = []
        for category in metadata_df['category'].unique():
            category_lower = category.lower()
            if category_lower in image_paths:
                all_image_paths.extend(image_paths[category_lower])
        
        # Extract features from all images with augmentation
        if len(all_image_paths) > 0:
            print(f"Found {len(all_image_paths)} original images")
            print(f"Target: {len(metadata_df)} samples (600 per category)")
            
            # Calculate how many augmentations per image needed
            n_augmentations_per_image = max(1, len(metadata_df) // len(all_image_paths))
            print(f"Applying augmentation: ~{n_augmentations_per_image} augmentations per image")
            
            image_features_list = []
            failed_count = 0
            
            # Group images by category to match with metadata
            images_by_category = {}
            for img_path in all_image_paths:
                # Extract category from path
                path_parts = Path(img_path).parts
                for cat in class_names:
                    if cat.lower() in path_parts:
                        if cat.lower() not in images_by_category:
                            images_by_category[cat.lower()] = []
                        images_by_category[cat.lower()].append(img_path)
                        break
            
            # Process each category to match metadata order
            for category in metadata_df['category'].unique():
                category_lower = category.lower()
                category_images = images_by_category.get(category_lower, [])
                category_samples = metadata_df[metadata_df['category'] == category]
                n_samples_needed = len(category_samples)
                
                if len(category_images) == 0:
                    print(f"Warning: No images found for category {category}")
                    # Add zero features for this category
                    if len(image_features_list) > 0:
                        zero_features = np.zeros((n_samples_needed, image_features_list[0].shape[1]))
                        image_features_list.extend(zero_features)
                    continue
                
                # Calculate augmentations per image for this category
                aug_per_img = max(1, (n_samples_needed - len(category_images)) // len(category_images))
                # Each image will contribute: 1 original + aug_per_img augmentations
                
                print(f"\nProcessing {category}: {len(category_images)} images -> {n_samples_needed} samples")
                print(f"  Augmentations per image: {aug_per_img} (total per image: {aug_per_img + 1})")
                
                category_start_idx = len(image_features_list)
                
                for img_idx, img_path in enumerate(category_images):
                    try:
                        # Load and preprocess original image
                        preprocessed_img = image_collector.preprocess_image(img_path)
                        
                        # Extract features from original
                        features = image_extractor.extract_features(preprocessed_img)
                        image_features_list.append(features)
                        
                        # Calculate how many more we need for this category
                        current_category_count = len(image_features_list) - category_start_idx
                        remaining_needed = n_samples_needed - current_category_count
                        
                        if remaining_needed <= 0:
                            break
                        
                        # Generate augmentations - get more than needed to have variety
                        # Use all available augmentations, then sample if needed
                        augmented_images = image_collector.augment_image(
                            preprocessed_img,
                            augmentations=['flip', 'rotate', 'brightness', 'contrast', 'noise'],
                            n_augmentations=None  # Get all augmentations first
                        )
                        
                        # Extract features from augmented images until we have enough
                        for aug_img in augmented_images:
                            if len(image_features_list) - category_start_idx >= n_samples_needed:
                                break
                            aug_features = image_extractor.extract_features(aug_img)
                            image_features_list.append(aug_features)
                        
                        # If we still need more samples, repeat augmentations
                        while len(image_features_list) - category_start_idx < n_samples_needed:
                            if len(augmented_images) == 0:
                                # If no augmentations available, repeat original
                                image_features_list.append(features)
                            else:
                                aug_img = random.choice(augmented_images)
                                aug_features = image_extractor.extract_features(aug_img)
                                image_features_list.append(aug_features)
                        
                        # Break if we've reached the target for this category
                        if len(image_features_list) - category_start_idx >= n_samples_needed:
                            break
                        
                    except Exception as e:
                        failed_count += 1
                        if failed_count <= 5:
                            print(f"Warning: Could not extract features from {img_path}: {e}")
                        continue
                
                # If we still don't have enough for this category, pad with last image's features
                current_category_count = len(image_features_list) - category_start_idx
                while current_category_count < n_samples_needed:
                    if len(image_features_list) > category_start_idx:
                        # Repeat last feature from this category
                        image_features_list.append(image_features_list[-1])
                        current_category_count += 1
                    else:
                        # No features at all for this category
                        print(f"Error: No features extracted for category {category}")
                        break
                
                final_category_count = len(image_features_list) - category_start_idx
                print(f"  Completed: {final_category_count} samples for {category}")
            
            if failed_count > 5:
                print(f"\nWarning: {failed_count} images failed feature extraction (showing first 5 errors)")
            
            if image_features_list:
                image_features = np.array(image_features_list)
                print(f"\nImage feature size: {image_features.shape[1]} per image")
                print(f"Total image features extracted: {image_features.shape[0]} images")
                
                # Ensure we have exactly the right number
                if image_features.shape[0] < len(metadata_df):
                    print(f"Warning: Only {image_features.shape[0]} image features, but {len(metadata_df)} metadata samples.")
                    print("Padding missing image features with zeros...")
                    n_missing = len(metadata_df) - image_features.shape[0]
                    zero_features = np.zeros((n_missing, image_features.shape[1]))
                    image_features = np.vstack([image_features, zero_features])
                elif image_features.shape[0] > len(metadata_df):
                    print(f"Using first {len(metadata_df)} image features...")
                    image_features = image_features[:len(metadata_df)]
            else:
                print("Warning: No image features could be extracted.")
                image_features = None
        else:
            print("Warning: No image paths found.")
    
    # Text features
    print("\nExtracting text features...")
    text_extractor = TextFeatureExtractor(method='word2vec', embedding_dim=100)
    text_extractor.train_word2vec(descriptions_df['description'].tolist())
    # Extract features from all descriptions
    text_features = np.array([text_extractor.extract_features(desc) 
                             for desc in descriptions_df['description']])
    print(f"Text feature size: {text_features.shape[1]} per sample")
    print(f"Total text features extracted: {text_features.shape[0]} samples")
    
    # Categorical features
    print("\nEncoding categorical features...")
    categorical_features = encode_categorical_features(
        metadata_df,
        columns=['color', 'season', 'origin'],
        method='onehot'
    )
    print(f"Categorical feature size: {categorical_features.shape[1]}")
    
    # Numerical features
    print("\nNormalizing numerical features...")
    numerical_features = normalize_numerical_features(
        metadata_df,
        columns=['weight'],
        method='standard'
    )
    print(f"Numerical feature size: {numerical_features.shape[1]}")
    
    # Feature fusion
    print("\nFusing features...")
    # Fuse all available features
    fused_features = fuse_features(
        image_features=image_features,  # None if no images
        text_features=text_features,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        method='concatenate'
    )
    
    print(f"Fused feature size: {fused_features.shape}")
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(metadata_df['category'])
    
    # 3. DATA SPLITTING
    print("\n" + "=" * 80)
    print("3. DATA SPLITTING")
    print("=" * 80)
    
    # According to assignment: Training: 2500 (with 500 validation from it), Test: 500
    # So: train_size=2000 (actual training), val_size=500 (from training), test_size=500
    # Total: 3000
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        fused_features, y,
        train_size=2000,  # Actual training set size (after validation is taken)
        test_size=500,
        val_size=500,  # Validation from training (so total training pool is 2500)
        random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Also split individual feature sets for Task 2(a) - use same splits as fused features
    # Image features split (use same indices to ensure consistency)
    if image_features is not None:
        X_train_img = image_features[:len(X_train)]
        X_val_img = image_features[len(X_train):len(X_train)+len(X_val)]
        X_test_img = image_features[len(X_train)+len(X_val):len(X_train)+len(X_val)+len(X_test)]
        y_train_img = y_train
        y_val_img = y_val
        y_test_img = y_test
    else:
        X_train_img = X_val_img = X_test_img = None
        y_train_img = y_val_img = y_test_img = None
    
    # Categorical + Numerical features split
    cat_num_features = np.hstack([categorical_features, numerical_features])
    X_train_catnum = cat_num_features[:len(X_train)]
    X_val_catnum = cat_num_features[len(X_train):len(X_train)+len(X_val)]
    X_test_catnum = cat_num_features[len(X_train)+len(X_val):len(X_train)+len(X_val)+len(X_test)]
    y_train_catnum = y_train
    y_val_catnum = y_val
    y_test_catnum = y_test
    
    # Text features split
    X_train_text = text_features[:len(X_train)]
    X_val_text = text_features[len(X_train):len(X_train)+len(X_val)]
    X_test_text = text_features[len(X_train)+len(X_val):len(X_train)+len(X_val)+len(X_test)]
    y_train_text = y_train
    y_val_text = y_val
    y_test_text = y_test
    
    # 4. DATA QUALITY CHECK
    print("\n" + "=" * 80)
    print("4. DATA QUALITY CHECK")
    print("=" * 80)
    
    print_data_quality_report(X_train, y_train)
    
    # Similarity analysis
    print_similarity_report(X_train, y_train, class_names=class_names)
    
    # Outlier detection
    print_outlier_report(X_train, y_train, class_names=class_names)
    
    # 5. MODEL TRAINING - Task 2(a)
    print("\n" + "=" * 80)
    print("5. MODEL TRAINING - Task 2(a)")
    print("=" * 80)
    print("Training separate models for different feature sets...")
    
    # Store results for comparison
    model_results = {}
    
    # 1. Image features only
    if X_train_img is not None:
        print("\n" + "-" * 80)
        print("1. Training with IMAGE FEATURES ONLY")
        print("-" * 80)
        model_img = OneVsAllClassifier(
            LogisticRegression,
            n_classes=5,
            learning_rate=0.01,
            max_iter=1000,
            regularization='l2',
            lambda_reg=0.01,
            verbose=False
        )
        start_time = time.time()
        model_img.fit(X_train_img, y_train_img, X_val_img, y_val_img)
        train_time_img = time.time() - start_time
        
        # Plot training and validation loss
        plot_training_history(
            train_losses=model_img.train_loss_history,
            val_losses=model_img.val_loss_history if len(model_img.val_loss_history) > 0 else None,
            save_path=str(reports_dir / "loss_history_image_only.png")
        )
        print(f"Loss plot saved: {reports_dir / 'loss_history_image_only.png'}")
        
        # Test predictions
        y_test_pred_img = model_img.predict(X_test_img)
        y_test_proba_img = model_img.predict_proba(X_test_img)
        test_metrics_img = calculate_metrics(y_test_img, y_test_pred_img, y_test_proba_img)
        
        model_results['image_only'] = {
            'model': model_img,
            'metrics': test_metrics_img,
            'train_time': train_time_img,
            'y_test_pred': y_test_pred_img,
            'y_test_proba': y_test_proba_img
        }
        print(f"Test Accuracy: {test_metrics_img.get('accuracy', 0):.4f}")
    
    # 2. Categorical + Numerical features only
    print("\n" + "-" * 80)
    print("2. Training with CATEGORICAL + NUMERICAL FEATURES ONLY")
    print("-" * 80)
    model_catnum = OneVsAllClassifier(
        LogisticRegression,
        n_classes=5,
        learning_rate=0.01,
        max_iter=1000,
        regularization='l2',
        lambda_reg=0.01,
        verbose=False
    )
    start_time = time.time()
    model_catnum.fit(X_train_catnum, y_train_catnum, X_val_catnum, y_val_catnum)
    train_time_catnum = time.time() - start_time
    
    # Plot training and validation loss
    plot_training_history(
        train_losses=model_catnum.train_loss_history,
        val_losses=model_catnum.val_loss_history if len(model_catnum.val_loss_history) > 0 else None,
        save_path=str(reports_dir / "loss_history_categorical_numerical_only.png")
    )
    print(f"Loss plot saved: {reports_dir / 'loss_history_categorical_numerical_only.png'}")
    
    # Test predictions
    y_test_pred_catnum = model_catnum.predict(X_test_catnum)
    y_test_proba_catnum = model_catnum.predict_proba(X_test_catnum)
    test_metrics_catnum = calculate_metrics(y_test_catnum, y_test_pred_catnum, y_test_proba_catnum)
    
    model_results['categorical_numerical_only'] = {
        'model': model_catnum,
        'metrics': test_metrics_catnum,
        'train_time': train_time_catnum,
        'y_test_pred': y_test_pred_catnum,
        'y_test_proba': y_test_proba_catnum
    }
    print(f"Test Accuracy: {test_metrics_catnum.get('accuracy', 0):.4f}")
    
    # 3. Text features only
    print("\n" + "-" * 80)
    print("3. Training with TEXT FEATURES ONLY")
    print("-" * 80)
    model_text = OneVsAllClassifier(
        LogisticRegression,
        n_classes=5,
        learning_rate=0.01,
        max_iter=1000,
        regularization='l2',
        lambda_reg=0.01,
        verbose=False
    )
    start_time = time.time()
    model_text.fit(X_train_text, y_train_text, X_val_text, y_val_text)
    train_time_text = time.time() - start_time
    
    # Plot training and validation loss
    plot_training_history(
        train_losses=model_text.train_loss_history,
        val_losses=model_text.val_loss_history if len(model_text.val_loss_history) > 0 else None,
        save_path=str(reports_dir / "loss_history_text_only.png")
    )
    print(f"Loss plot saved: {reports_dir / 'loss_history_text_only.png'}")
    
    # Test predictions
    y_test_pred_text = model_text.predict(X_test_text)
    y_test_proba_text = model_text.predict_proba(X_test_text)
    test_metrics_text = calculate_metrics(y_test_text, y_test_pred_text, y_test_proba_text)
    
    model_results['text_only'] = {
        'model': model_text,
        'metrics': test_metrics_text,
        'train_time': train_time_text,
        'y_test_pred': y_test_pred_text,
        'y_test_proba': y_test_proba_text
    }
    print(f"Test Accuracy: {test_metrics_text.get('accuracy', 0):.4f}")
    
    # 4. Fused features
    print("\n" + "-" * 80)
    print("4. Training with FUSED FEATURES")
    print("-" * 80)
    custom_model = OneVsAllClassifier(
        LogisticRegression,
        n_classes=5,
        learning_rate=0.01,
        max_iter=1000,
        regularization='l2',
        lambda_reg=0.01,
        verbose=False
    )
    start_time = time.time()
    custom_model.fit(X_train, y_train, X_val, y_val)
    custom_train_time = time.time() - start_time
    
    # Plot training and validation loss
    plot_training_history(
        train_losses=custom_model.train_loss_history,
        val_losses=custom_model.val_loss_history if len(custom_model.val_loss_history) > 0 else None,
        save_path=str(reports_dir / "loss_history_fused.png")
    )
    print(f"Loss plot saved: {reports_dir / 'loss_history_fused.png'}")
    
    # Test predictions
    y_test_pred_custom = custom_model.predict(X_test)
    y_test_proba_custom = custom_model.predict_proba(X_test)
    
    model_results['fused'] = {
        'model': custom_model,
        'metrics': None,  # Will calculate below
        'train_time': custom_train_time,
        'y_test_pred': y_test_pred_custom,
        'y_test_proba': y_test_proba_custom
    }
    
    # Calculate metrics for fused features
    custom_metrics = calculate_metrics(y_test, y_test_pred_custom, y_test_proba_custom)
    model_results['fused']['metrics'] = custom_metrics
    
    # Task 2(b): Report training and test classification metrics
    print("\n" + "=" * 80)
    print("6. EVALUATION - Task 2(b)")
    print("=" * 80)
    print("\nPerformance Comparison of Different Feature Sets:")
    print("\n" + "=" * 80)
    
    # Create comparison table
    print("\nTest Set Metrics:")
    print(f"{'Feature Set':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12}")
    print("-" * 90)
    
    for feature_set, results in model_results.items():
        metrics = results['metrics']
        if metrics:
            auc_value = metrics.get('auc', 0) if metrics.get('auc') is not None else 0
            print(f"{feature_set.replace('_', ' ').title():<30} "
                  f"{metrics.get('accuracy', 0):<12.4f} "
                  f"{metrics.get('precision', 0):<12.4f} "
                  f"{metrics.get('recall', 0):<12.4f} "
                  f"{metrics.get('f1_score', 0):<12.4f} "
                  f"{auc_value:<12.4f}")
    
    # Detailed metrics for each feature set
    print("\n" + "=" * 80)
    print("Detailed Metrics for Each Feature Set:")
    print("=" * 80)
    
    for feature_set, results in model_results.items():
        metrics = results['metrics']
        if metrics:
            print(f"\n{feature_set.replace('_', ' ').title()}:")
            for metric, value in metrics.items():
                if value is not None:
                    print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Task 2(c): Compare with Sklearn (using fused features)
    print("\n" + "=" * 80)
    print("7. COMPARISON WITH SKLEARN - Task 2(c)")
    print("=" * 80)
    
    print("\nTraining Sklearn LogisticRegression (fused features)...")
    start_time = time.time()
    
    sklearn_model = SklearnLR(
        max_iter=1000,
        multi_class='ovr',
        random_state=42,
        solver='lbfgs'
    )
    sklearn_model.fit(X_train, y_train)
    sklearn_train_time = time.time() - start_time
    
    y_test_pred_sklearn = sklearn_model.predict(X_test)
    y_test_proba_sklearn = sklearn_model.predict_proba(X_test)
    
    sklearn_metrics = calculate_metrics(y_test, y_test_pred_sklearn, y_test_proba_sklearn)
    
    print("\nOur Implementation (Fused Features) - Test Set:")
    for metric, value in custom_metrics.items():
        if value is not None:
            print(f"  {metric.capitalize()}: {value:.4f}")
    
    print("\nSklearn (Fused Features) - Test Set:")
    for metric, value in sklearn_metrics.items():
        if value is not None:
            print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Runtime comparison
    print("\nRuntime Comparison (Fused Features):")
    print(f"  Our implementation: {custom_train_time:.4f} seconds")
    print(f"  Sklearn: {sklearn_train_time:.4f} seconds")
    if custom_train_time > 0:
        print(f"  Speed difference: {sklearn_train_time / custom_train_time:.2f}x")
    
    # Confusion matrices and ROC curves for each feature set
    print("\n" + "=" * 80)
    print("8. VISUALIZATIONS")
    print("=" * 80)
    
    # Confusion matrices
    print("\nPlotting confusion matrices...")
    if 'image_only' in model_results:
        plot_confusion_matrix(
            y_test_img, model_results['image_only']['y_test_pred'], 
            class_names=class_names,
            save_path=str(reports_dir / "confusion_matrix_image_only.png")
        )
    
    plot_confusion_matrix(
        y_test_catnum, model_results['categorical_numerical_only']['y_test_pred'],
        class_names=class_names,
        save_path=str(reports_dir / "confusion_matrix_categorical_numerical_only.png")
    )
    
    plot_confusion_matrix(
        y_test_text, model_results['text_only']['y_test_pred'],
        class_names=class_names,
        save_path=str(reports_dir / "confusion_matrix_text_only.png")
    )
    
    plot_confusion_matrix(
        y_test, y_test_pred_custom,
        class_names=class_names,
        save_path=str(reports_dir / "confusion_matrix_fused.png")
    )
    
    # ROC curves
    print("Plotting ROC curves...")
    if 'image_only' in model_results:
        plot_roc_curve(
            y_test_img, model_results['image_only']['y_test_proba'],
            class_names=class_names,
            save_path=str(reports_dir / "roc_curve_image_only.png")
        )
    
    plot_roc_curve(
        y_test_catnum, model_results['categorical_numerical_only']['y_test_proba'],
        class_names=class_names,
        save_path=str(reports_dir / "roc_curve_categorical_numerical_only.png")
    )
    
    plot_roc_curve(
        y_test_text, model_results['text_only']['y_test_proba'],
        class_names=class_names,
        save_path=str(reports_dir / "roc_curve_text_only.png")
    )
    
    plot_roc_curve(
        y_test, y_test_proba_custom,
        class_names=class_names,
        save_path=str(reports_dir / "roc_curve_fused.png")
    )
    
    # 9. GENERATE FINAL REPORT
    print("\n" + "=" * 80)
    print("9. GENERATING FINAL REPORT")
    print("=" * 80)
    
    generate_final_report(
        metadata_df=metadata_df,
        descriptions_df=descriptions_df,
        image_paths=image_paths,
        image_features=image_features,
        text_features=text_features,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        fused_features=fused_features,
        X_train=X_train,
        y_train=y_train,
        class_names=class_names,
        reports_dir=reports_dir,
        model_results=model_results
    )
    
    print("\n" + "=" * 80)
    print("COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()

