"""
Veri ön işleme modülü
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import Optional, Union, Tuple


def handle_missing_values(X: Union[np.ndarray, pd.DataFrame],
                         strategy: str = 'mean',
                         fill_value: Optional[float] = None) -> np.ndarray:
    """
    Eksik değerleri işle
    
    Args:
        X: Veri matrisi
        strategy: Doldurma stratejisi ('mean', 'median', 'most_frequent', 'constant')
        fill_value: Sabit değer (strategy='constant' için)
        
    Returns:
        Eksik değerleri doldurulmuş veri matrisi
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Eksik değer kontrolü
    if not np.isnan(X).any():
        return X
    
    # Imputer oluştur
    if strategy == 'constant':
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    else:
        imputer = SimpleImputer(strategy=strategy)
    
    X_imputed = imputer.fit_transform(X)
    
    return X_imputed


def preprocess_features(X: Union[np.ndarray, pd.DataFrame],
                       method: str = 'standard',
                       handle_missing: bool = True,
                       missing_strategy: str = 'mean') -> Tuple[np.ndarray, object]:
    """
    Özellikleri ön işle
    
    Args:
        X: Özellik matrisi
        method: Normalizasyon yöntemi ('standard', 'minmax', None)
        handle_missing: Eksik değerleri işle
        missing_strategy: Eksik değer stratejisi
        
    Returns:
        (işlenmiş_özellikler, scaler) tuple'ı
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Eksik değerleri işle
    if handle_missing:
        X = handle_missing_values(X, strategy=missing_strategy)
    
    # Normalizasyon
    scaler = None
    if method == 'standard':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif method == 'minmax':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif method is None:
        pass
    else:
        raise ValueError(f"Bilinmeyen normalizasyon yöntemi: {method}")
    
    return X, scaler


def check_data_quality(X: np.ndarray,
                      y: Optional[np.ndarray] = None) -> dict:
    """
    Veri kalitesini kontrol et
    
    Args:
        X: Özellik matrisi
        y: Etiketler (opsiyonel)
        
    Returns:
        Kalite kontrol sonuçları
    """
    results = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'has_missing_values': np.isnan(X).any(),
        'n_missing_values': np.isnan(X).sum(),
        'missing_ratio': np.isnan(X).sum() / X.size,
        'has_infinite_values': np.isinf(X).any(),
        'feature_range': {
            'min': np.nanmin(X),
            'max': np.nanmax(X),
            'mean': np.nanmean(X),
            'std': np.nanstd(X)
        }
    }
    
    if y is not None:
        unique_classes = np.unique(y)
        results['n_classes'] = len(unique_classes)
        results['class_distribution'] = {int(c): int(np.sum(y == c)) for c in unique_classes}
    
    return results


def print_data_quality_report(X: np.ndarray, y: Optional[np.ndarray] = None):
    """
    Veri kalitesi raporu yazdır
    
    Args:
        X: Özellik matrisi
        y: Etiketler (opsiyonel)
    """
    quality = check_data_quality(X, y)
    
    print("=" * 60)
    print("VERİ KALİTESİ RAPORU")
    print("=" * 60)
    print(f"Örnek Sayısı: {quality['n_samples']}")
    print(f"Özellik Sayısı: {quality['n_features']}")
    print(f"Eksik Değer Var mı: {quality['has_missing_values']}")
    if quality['has_missing_values']:
        print(f"  Eksik Değer Sayısı: {quality['n_missing_values']}")
        print(f"  Eksik Değer Oranı: {quality['missing_ratio']:.2%}")
    print(f"Sonsuz Değer Var mı: {quality['has_infinite_values']}")
    print(f"\nÖzellik İstatistikleri:")
    print(f"  Min: {quality['feature_range']['min']:.4f}")
    print(f"  Max: {quality['feature_range']['max']:.4f}")
    print(f"  Ortalama: {quality['feature_range']['mean']:.4f}")
    print(f"  Std Sapma: {quality['feature_range']['std']:.4f}")
    
    if y is not None:
        print(f"\nSınıf Dağılımı:")
        for class_label, count in quality['class_distribution'].items():
            print(f"  Sınıf {class_label}: {count} örnek")
    
    print("=" * 60)

