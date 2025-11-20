"""
Özellik birleştirme modülü
Farklı modalitelerden gelen özellikleri birleştirme stratejileri
"""

import numpy as np
from typing import List, Optional, Union
from sklearn.preprocessing import StandardScaler


class FeatureFusion:
    """Özellik birleştirme sınıfı"""
    
    def __init__(self, method: str = 'concatenate', weights: Optional[List[float]] = None):
        """
        Args:
            method: Birleştirme yöntemi ('concatenate', 'weighted', 'pca')
            weights: Ağırlıklar (weighted method için)
        """
        self.method = method
        self.weights = weights
        self.scaler = StandardScaler()
        self._fitted = False
    
    def concatenate(self, feature_list: List[np.ndarray]) -> np.ndarray:
        """
        Özellikleri basitçe birleştir
        
        Args:
            feature_list: Özellik matrisleri listesi
            
        Returns:
            Birleştirilmiş özellik matrisi
        """
        # Tüm özellikleri yatay olarak birleştir
        return np.hstack(feature_list)
    
    def weighted_fusion(self, feature_list: List[np.ndarray]) -> np.ndarray:
        """
        Ağırlıklı özellik birleştirme
        
        Args:
            feature_list: Özellik matrisleri listesi
            
        Returns:
            Birleştirilmiş özellik matrisi
        """
        if self.weights is None:
            # Eşit ağırlıklar
            self.weights = [1.0 / len(feature_list)] * len(feature_list)
        
        if len(self.weights) != len(feature_list):
            raise ValueError("Ağırlık sayısı özellik sayısıyla eşleşmiyor")
        
        # Her özellik setini normalize et ve ağırlıkla çarp
        weighted_features = []
        for features, weight in zip(feature_list, self.weights):
            # Normalize et
            features_normalized = self.scaler.fit_transform(features)
            weighted_features.append(features_normalized * weight)
        
        # Birleştir
        return np.hstack(weighted_features)
    
    def fuse(self, feature_list: List[np.ndarray]) -> np.ndarray:
        """
        Özellikleri birleştir
        
        Args:
            feature_list: Özellik matrisleri listesi
            
        Returns:
            Birleştirilmiş özellik matrisi
        """
        # Boş özellikleri filtrele
        feature_list = [f for f in feature_list if f.size > 0]
        
        if not feature_list:
            raise ValueError("Birleştirilecek özellik yok")
        
        # Örnek sayılarını kontrol et
        n_samples = feature_list[0].shape[0]
        for features in feature_list[1:]:
            if features.shape[0] != n_samples:
                raise ValueError("Tüm özellik matrisleri aynı sayıda örneğe sahip olmalıdır")
        
        if self.method == 'concatenate':
            return self.concatenate(feature_list)
        elif self.method == 'weighted':
            return self.weighted_fusion(feature_list)
        else:
            raise ValueError(f"Bilinmeyen birleştirme yöntemi: {self.method}")


def fuse_features(image_features: Optional[np.ndarray] = None,
                 text_features: Optional[np.ndarray] = None,
                 categorical_features: Optional[np.ndarray] = None,
                 numerical_features: Optional[np.ndarray] = None,
                 method: str = 'concatenate',
                 weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Farklı modalitelerden gelen özellikleri birleştir
    
    Args:
        image_features: Görüntü özellikleri (n_samples, n_image_features)
        text_features: Metin özellikleri (n_samples, n_text_features)
        categorical_features: Kategorik özellikler (n_samples, n_cat_features)
        numerical_features: Numerik özellikler (n_samples, n_num_features)
        method: Birleştirme yöntemi ('concatenate', 'weighted')
        weights: Ağırlıklar (weighted method için)
        
    Returns:
        Birleştirilmiş özellik matrisi (n_samples, n_total_features)
    """
    fusion = FeatureFusion(method=method, weights=weights)
    
    feature_list = []
    
    if image_features is not None and image_features.size > 0:
        feature_list.append(image_features)
    
    if text_features is not None and text_features.size > 0:
        feature_list.append(text_features)
    
    if categorical_features is not None and categorical_features.size > 0:
        feature_list.append(categorical_features)
    
    if numerical_features is not None and numerical_features.size > 0:
        feature_list.append(numerical_features)
    
    if not feature_list:
        raise ValueError("En az bir özellik seti sağlanmalıdır")
    
    fused = fusion.fuse(feature_list)
    
    # Özellik boyutunu kontrol et (10-500 arası olmalı)
    if fused.shape[1] < 10:
        raise ValueError(f"Özellik boyutu çok küçük: {fused.shape[1]}. En az 10 olmalı.")
    elif fused.shape[1] > 500:
        # PCA ile boyut azaltma önerisi
        import warnings
        warnings.warn(f"Özellik boyutu çok büyük: {fused.shape[1]}. "
                     f"PCA ile boyut azaltma önerilir.")
    
    return fused

