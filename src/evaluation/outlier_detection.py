"""
Outlier tespiti modülü
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from scipy import stats
import pandas as pd


class OutlierDetector:
    """Outlier tespiti sınıfı"""
    
    def __init__(self, method: str = 'iqr'):
        """
        Args:
            method: Outlier tespit yöntemi ('iqr', 'zscore', 'isolation_forest', 'elliptic')
        """
        self.method = method
        self.detector = None
        self.thresholds = {}
    
    def detect_iqr(self, X: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """
        IQR (Interquartile Range) yöntemiyle outlier tespit et
        
        Args:
            X: Özellik matrisi (n_samples, n_features)
            factor: IQR çarpanı
            
        Returns:
            Outlier maskesi (True = outlier)
        """
        outlier_mask = np.zeros(X.shape[0], dtype=bool)
        
        for i in range(X.shape[1]):
            Q1 = np.percentile(X[:, i], 25)
            Q3 = np.percentile(X[:, i], 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            feature_outliers = (X[:, i] < lower_bound) | (X[:, i] > upper_bound)
            outlier_mask |= feature_outliers
        
        return outlier_mask
    
    def detect_zscore(self, X: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Z-score yöntemiyle outlier tespit et
        
        Args:
            X: Özellik matrisi
            threshold: Z-score eşiği
            
        Returns:
            Outlier maskesi
        """
        z_scores = np.abs(stats.zscore(X, axis=0))
        outlier_mask = np.any(z_scores > threshold, axis=1)
        return outlier_mask
    
    def detect_isolation_forest(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """
        Isolation Forest yöntemiyle outlier tespit et
        
        Args:
            X: Özellik matrisi
            contamination: Beklenen outlier oranı
            
        Returns:
            Outlier maskesi
        """
        self.detector = IsolationForest(contamination=contamination, random_state=42)
        predictions = self.detector.fit_predict(X)
        # -1 = outlier, 1 = normal
        outlier_mask = predictions == -1
        return outlier_mask
    
    def detect_elliptic_envelope(self, X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
        """
        Elliptic Envelope yöntemiyle outlier tespit et
        
        Args:
            X: Özellik matrisi
            contamination: Beklenen outlier oranı
            
        Returns:
            Outlier maskesi
        """
        self.detector = EllipticEnvelope(contamination=contamination, random_state=42)
        predictions = self.detector.fit_predict(X)
        # -1 = outlier, 1 = normal
        outlier_mask = predictions == -1
        return outlier_mask
    
    def detect(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Outlier tespit et
        
        Args:
            X: Özellik matrisi
            **kwargs: Yönteme özel parametreler
            
        Returns:
            Outlier maskesi
        """
        if self.method == 'iqr':
            factor = kwargs.get('factor', 1.5)
            return self.detect_iqr(X, factor=factor)
        elif self.method == 'zscore':
            threshold = kwargs.get('threshold', 3.0)
            return self.detect_zscore(X, threshold=threshold)
        elif self.method == 'isolation_forest':
            contamination = kwargs.get('contamination', 0.1)
            return self.detect_isolation_forest(X, contamination=contamination)
        elif self.method == 'elliptic':
            contamination = kwargs.get('contamination', 0.1)
            return self.detect_elliptic_envelope(X, contamination=contamination)
        else:
            raise ValueError(f"Bilinmeyen yöntem: {self.method}")


def detect_outliers(X: np.ndarray,
                   method: str = 'iqr',
                   **kwargs) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Outlier tespit et
    
    Args:
        X: Özellik matrisi
        method: Tespit yöntemi
        **kwargs: Yönteme özel parametreler
        
    Returns:
        (outlier_maskesi, istatistikler) tuple'ı
    """
    detector = OutlierDetector(method=method)
    outlier_mask = detector.detect(X, **kwargs)
    
    n_outliers = np.sum(outlier_mask)
    n_samples = len(outlier_mask)
    outlier_ratio = n_outliers / n_samples
    
    stats_dict = {
        'n_outliers': n_outliers,
        'n_samples': n_samples,
        'outlier_ratio': outlier_ratio,
        'method': method
    }
    
    return outlier_mask, stats_dict


def analyze_outliers_by_class(X: np.ndarray,
                              y: np.ndarray,
                              method: str = 'iqr',
                              **kwargs) -> Dict[str, any]:
    """
    Sınıf bazında outlier analizi
    
    Args:
        X: Özellik matrisi
        y: Sınıf etiketleri
        method: Tespit yöntemi
        **kwargs: Yönteme özel parametreler
        
    Returns:
        Sınıf bazında outlier istatistikleri
    """
    unique_classes = np.unique(y)
    results = {}
    
    for class_label in unique_classes:
        class_mask = y == class_label
        X_class = X[class_mask]
        
        outlier_mask, stats = detect_outliers(X_class, method=method, **kwargs)
        
        results[class_label] = {
            'n_outliers': stats['n_outliers'],
            'n_samples': stats['n_samples'],
            'outlier_ratio': stats['outlier_ratio']
        }
    
    return results


def print_outlier_report(X: np.ndarray,
                        y: Optional[np.ndarray] = None,
                        method: str = 'iqr',
                        class_names: Optional[List[str]] = None,
                        **kwargs):
    """
    Outlier analizi raporu yazdır
    
    Args:
        X: Özellik matrisi
        y: Sınıf etiketleri (opsiyonel)
        method: Tespit yöntemi
        class_names: Sınıf isimleri
        **kwargs: Yönteme özel parametreler
    """
    print("=" * 60)
    print("OUTLIER TESPİTİ RAPORU")
    print("=" * 60)
    print(f"Yöntem: {method.upper()}")
    
    if y is not None:
        print("\nSınıf Bazında Outlier Analizi:")
        results = analyze_outliers_by_class(X, y, method=method, **kwargs)
        
        unique_classes = np.unique(y)
        if class_names is None:
            class_names = [str(c) for c in unique_classes]
        
        for class_label, stats in results.items():
            class_name = class_names[np.where(unique_classes == class_label)[0][0]]
            print(f"\n{class_name}:")
            print(f"  Toplam Örnek: {stats['n_samples']}")
            print(f"  Outlier Sayısı: {stats['n_outliers']}")
            print(f"  Outlier Oranı: {stats['outlier_ratio']:.2%}")
    else:
        outlier_mask, stats = detect_outliers(X, method=method, **kwargs)
        print(f"\nGenel Outlier Analizi:")
        print(f"  Toplam Örnek: {stats['n_samples']}")
        print(f"  Outlier Sayısı: {stats['n_outliers']}")
        print(f"  Outlier Oranı: {stats['outlier_ratio']:.2%}")
    
    print("=" * 60)

