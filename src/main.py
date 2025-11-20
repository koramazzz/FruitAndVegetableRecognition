"""
Ana çalıştırma scripti
Meyve ve Sebze Tanıma Projesi - CMPE 462 Assignment 1
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
import sys

# Modülleri import et
from src.data_collection import ImageCollector, MetadataCollector, TextCollector
from src.feature_extraction import (
    ImageFeatureExtractor, TextFeatureExtractor,
    encode_categorical_features, normalize_numerical_features,
    fuse_features
)
from src.models import LogisticRegression, OneVsAllClassifier
from src.evaluation import (
    calculate_metrics, plot_confusion_matrix, plot_roc_curve,
    calculate_intra_class_similarity, calculate_inter_class_similarity,
    print_similarity_report, print_outlier_report
)
from src.utils import (
    load_dataset, split_dataset, preprocess_features,
    print_data_quality_report
)
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import accuracy_score


def main():
    """Ana fonksiyon"""
    print("=" * 80)
    print("MEYVE VE SEBZE TANIMA PROJESİ")
    print("CMPE 462 Assignment 1")
    print("=" * 80)
    
    # Kategoriler
    categories = ['muz', 'domates', 'salatalik', 'mandalina', 'patates']
    class_names = ['Muz', 'Domates', 'Salatalık', 'Mandalina', 'Patates']
    
    # 1. VERİ TOPLAMA VE HAZIRLAMA
    print("\n" + "=" * 80)
    print("1. VERİ TOPLAMA VE HAZIRLAMA")
    print("=" * 80)
    
    # Metadata oluştur
    print("\nMetadata oluşturuluyor...")
    metadata_collector = MetadataCollector()
    metadata_df = metadata_collector.generate_metadata(n_samples_per_category=600, seed=42)
    metadata_collector.save_metadata(metadata_df, "metadata.csv")
    
    # Metin açıklamaları oluştur
    print("\nMetin açıklamaları oluşturuluyor...")
    text_collector = TextCollector()
    descriptions_df = text_collector.generate_descriptions(
        metadata_df['sample_id'].tolist(),
        metadata_df['category'].tolist(),
        seed=42
    )
    text_collector.save_descriptions(descriptions_df, "descriptions.csv")
    
    # Görüntü yollarını kontrol et (gerçek görüntüler için)
    print("\nGörüntü yolları kontrol ediliyor...")
    image_collector = ImageCollector()
    image_paths = image_collector.get_all_image_paths()
    
    total_images = sum(len(paths) for paths in image_paths.values())
    print(f"Toplam {total_images} görüntü bulundu")
    
    if total_images == 0:
        print("Uyarı: Görüntü bulunamadı. Lütfen görüntüleri data/raw/images/ klasörüne yerleştirin.")
        print("Şimdilik sadece metadata ve metin özellikleriyle devam edilecek.")
    
    # 2. ÖZELLİK ÇIKARIMI
    print("\n" + "=" * 80)
    print("2. ÖZELLİK ÇIKARIMI")
    print("=" * 80)
    
    # Görüntü özellikleri (eğer görüntüler varsa)
    image_features = None
    if total_images > 0:
        print("\nGörüntü özellikleri çıkarılıyor...")
        image_extractor = ImageFeatureExtractor(use_hog=True, use_lbp=True, use_color_hist=True)
        
        all_image_paths = []
        for category_paths in image_paths.values():
            all_image_paths.extend(category_paths)
        
        # İlk birkaç görüntüden özellik çıkar (demo için)
        # Gerçek kullanımda tüm görüntülerden çıkarılmalı
        if len(all_image_paths) > 0:
            sample_size = min(100, len(all_image_paths))
            sample_paths = all_image_paths[:sample_size]
            image_features = image_extractor.extract_features(
                image_extractor.preprocess_image(sample_paths[0])
            )
            print(f"Görüntü özellik boyutu: {len(image_features)}")
    
    # Metin özellikleri
    print("\nMetin özellikleri çıkarılıyor...")
    text_extractor = TextFeatureExtractor(method='word2vec', embedding_dim=100)
    text_extractor.train_word2vec(descriptions_df['description'].tolist())
    text_features = text_extractor.extract_features(descriptions_df['description'].iloc[0])
    print(f"Metin özellik boyutu: {len(text_features)}")
    
    # Kategorik özellikler
    print("\nKategorik özellikler encode ediliyor...")
    categorical_features = encode_categorical_features(
        metadata_df,
        columns=['color', 'season', 'origin'],
        method='onehot'
    )
    print(f"Kategorik özellik boyutu: {categorical_features.shape[1]}")
    
    # Numerik özellikler
    print("\nNumerik özellikler normalize ediliyor...")
    numerical_features = normalize_numerical_features(
        metadata_df,
        columns=['weight'],
        method='standard'
    )
    print(f"Numerik özellik boyutu: {numerical_features.shape[1]}")
    
    # Özellik birleştirme
    print("\nÖzellikler birleştiriliyor...")
    # Demo için sadece metin, kategorik ve numerik özellikleri birleştir
    # Gerçek kullanımda görüntü özellikleri de eklenmeli
    fused_features = fuse_features(
        image_features=None,  # Görüntüler yoksa None
        text_features=np.array([text_extractor.extract_features(desc) 
                               for desc in descriptions_df['description']]),
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        method='concatenate'
    )
    
    print(f"Birleştirilmiş özellik boyutu: {fused_features.shape}")
    
    # Etiketleri encode et
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(metadata_df['category'])
    
    # 3. VERİ BÖLME
    print("\n" + "=" * 80)
    print("3. VERİ BÖLME")
    print("=" * 80)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        fused_features, y,
        train_size=2500,
        test_size=500,
        val_size=500,
        random_state=42
    )
    
    print(f"Eğitim seti: {X_train.shape[0]} örnek")
    print(f"Validasyon seti: {X_val.shape[0]} örnek")
    print(f"Test seti: {X_test.shape[0]} örnek")
    
    # 4. VERİ KALİTESİ KONTROLÜ
    print("\n" + "=" * 80)
    print("4. VERİ KALİTESİ KONTROLÜ")
    print("=" * 80)
    
    print_data_quality_report(X_train, y_train)
    
    # Benzerlik analizi
    print_similarity_report(X_train, y_train, class_names=class_names)
    
    # Outlier tespiti
    print_outlier_report(X_train, y_train, class_names=class_names)
    
    # 5. MODEL EĞİTİMİ
    print("\n" + "=" * 80)
    print("5. MODEL EĞİTİMİ")
    print("=" * 80)
    
    # Kendi implementasyonumuz
    print("\nKendi Logistic Regression implementasyonumuz eğitiliyor...")
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
    
    # Validasyon tahminleri
    y_val_pred_custom = custom_model.predict(X_val)
    y_val_proba_custom = custom_model.predict_proba(X_val)
    
    # Test tahminleri
    y_test_pred_custom = custom_model.predict(X_test)
    y_test_proba_custom = custom_model.predict_proba(X_test)
    
    # Sklearn karşılaştırması
    print("\nSklearn LogisticRegression eğitiliyor...")
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
    
    # 6. DEĞERLENDİRME
    print("\n" + "=" * 80)
    print("6. DEĞERLENDİRME")
    print("=" * 80)
    
    # Kendi implementasyonumuz
    print("\nKendi Implementasyonumuz - Test Seti:")
    custom_metrics = calculate_metrics(y_test, y_test_pred_custom, y_test_proba_custom)
    for metric, value in custom_metrics.items():
        if value is not None:
            print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Sklearn
    print("\nSklearn - Test Seti:")
    sklearn_metrics = calculate_metrics(y_test, y_test_pred_sklearn, y_test_proba_sklearn)
    for metric, value in sklearn_metrics.items():
        if value is not None:
            print(f"  {metric.capitalize()}: {value:.4f}")
    
    # Runtime karşılaştırması
    print("\nRuntime Karşılaştırması:")
    print(f"  Kendi implementasyonumuz: {custom_train_time:.4f} saniye")
    print(f"  Sklearn: {sklearn_train_time:.4f} saniye")
    print(f"  Hız farkı: {sklearn_train_time / custom_train_time:.2f}x")
    
    # Confusion matrix
    print("\nConfusion Matrix çiziliyor...")
    plot_confusion_matrix(y_test, y_test_pred_custom, class_names=class_names,
                         save_path="reports/confusion_matrix.png")
    
    # ROC curve
    print("ROC eğrisi çiziliyor...")
    plot_roc_curve(y_test, y_test_proba_custom, class_names=class_names,
                   save_path="reports/roc_curve.png")
    
    print("\n" + "=" * 80)
    print("TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()

