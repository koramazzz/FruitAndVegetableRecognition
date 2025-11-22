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
from sklearn.linear_model import LogisticRegression as SklearnLR


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
        
        all_image_paths = []
        for category_paths in image_paths.values():
            all_image_paths.extend(category_paths)
        
        # Extract features from first few images (for demo)
        # In real usage, extract from all images
        if len(all_image_paths) > 0:
            sample_size = min(100, len(all_image_paths))
            sample_paths = all_image_paths[:sample_size]
            # Preprocess image using ImageCollector
            preprocessed_img = image_collector.preprocess_image(sample_paths[0])
            image_features = image_extractor.extract_features(preprocessed_img)
            print(f"Image feature size: {len(image_features)}")
    
    # Text features
    print("\nExtracting text features...")
    text_extractor = TextFeatureExtractor(method='word2vec', embedding_dim=100)
    text_extractor.train_word2vec(descriptions_df['description'].tolist())
    text_features = text_extractor.extract_features(descriptions_df['description'].iloc[0])
    print(f"Text feature size: {len(text_features)}")
    
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
    # For demo, fuse only text, categorical and numerical features
    # In real usage, image features should also be added
    fused_features = fuse_features(
        image_features=None,  # None if no images
        text_features=np.array([text_extractor.extract_features(desc) 
                               for desc in descriptions_df['description']]),
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
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        fused_features, y,
        train_size=2500,
        test_size=500,
        val_size=500,
        random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # 4. DATA QUALITY CHECK
    print("\n" + "=" * 80)
    print("4. DATA QUALITY CHECK")
    print("=" * 80)
    
    print_data_quality_report(X_train, y_train)
    
    # Similarity analysis
    print_similarity_report(X_train, y_train, class_names=class_names)
    
    # Outlier detection
    print_outlier_report(X_train, y_train, class_names=class_names)
    
    # 5. MODEL TRAINING
    print("\n" + "=" * 80)
    print("5. MODEL TRAINING")
    print("=" * 80)
    
    # Our own implementation
    print("\nTraining our own Logistic Regression implementation...")
    start_time = time.time()
    
    custom_model = OneVsAllClassifier(
        LogisticRegression,
        n_classes=5,
        learning_rate=0.01,
        max_iter=1000,
        regularization='l2',
        lambda_reg=0.01,
        verbose=True
    )
    
    custom_model.fit(X_train, y_train)
    custom_train_time = time.time() - start_time
    
    # Test predictions
    y_test_pred_custom = custom_model.predict(X_test)
    y_test_proba_custom = custom_model.predict_proba(X_test)
    
    # Sklearn comparison
    print("\nTraining Sklearn LogisticRegression...")
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
    
    # 6. EVALUATION
    print("\n" + "=" * 80)
    print("6. EVALUATION")
    print("=" * 80)
    
    # Our own implementation
    print("\nOur Implementation - Test Set:")
    custom_metrics = calculate_metrics(y_test, y_test_pred_custom, y_test_proba_custom)
    for metric, value in custom_metrics.items():
        if value is not None:
            print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Sklearn
    print("\nSklearn - Test Set:")
    sklearn_metrics = calculate_metrics(y_test, y_test_pred_sklearn, y_test_proba_sklearn)
    for metric, value in sklearn_metrics.items():
        if value is not None:
            print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Runtime comparison
    print("\nRuntime Comparison:")
    print(f"  Our implementation: {custom_train_time:.4f} seconds")
    print(f"  Sklearn: {sklearn_train_time:.4f} seconds")
    print(f"  Speed difference: {sklearn_train_time / custom_train_time:.2f}x")
    
    # Confusion matrix
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(y_test, y_test_pred_custom, class_names=class_names,
                         save_path=str(reports_dir / "confusion_matrix.png"))
    
    # ROC curve
    print("Plotting ROC curve...")
    plot_roc_curve(y_test, y_test_proba_custom, class_names=class_names,
                   save_path=str(reports_dir / "roc_curve.png"))
    
    print("\n" + "=" * 80)
    print("COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()

