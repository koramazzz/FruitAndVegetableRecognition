"""
Benzerlik hesaplama modülü
İntra-sınıf ve inter-sınıf benzerlik analizi
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import pdist, squareform
import pandas as pd


def calculate_intra_class_similarity(X: np.ndarray,
                                    y: np.ndarray,
                                    metric: str = 'cosine') -> Dict[str, float]:
    """
    İntra-sınıf (sınıf içi) benzerlik hesapla
    
    Args:
        X: Özellik matrisi (n_samples, n_features)
        y: Sınıf etiketleri (n_samples,)
        metric: Benzerlik metriği ('cosine', 'euclidean')
        
    Returns:
        Sınıf bazında ortalama benzerlik dictionary'si
    """
    unique_classes = np.unique(y)
    similarities = {}
    
    for class_label in unique_classes:
        # Bu sınıfa ait örnekleri al
        class_mask = y == class_label
        X_class = X[class_mask]
        
        if len(X_class) < 2:
            similarities[class_label] = 1.0
            continue
        
        # Benzerlik matrisini hesapla
        if metric == 'cosine':
            sim_matrix = cosine_similarity(X_class)
        elif metric == 'euclidean':
            # Euclidean distance'ı benzerliğe çevir
            dist_matrix = euclidean_distances(X_class)
            # Normalize et ve benzerliğe çevir
            max_dist = dist_matrix.max()
            if max_dist > 0:
                sim_matrix = 1 - (dist_matrix / max_dist)
            else:
                sim_matrix = np.ones_like(dist_matrix)
        else:
            raise ValueError(f"Bilinmeyen metrik: {metric}")
        
        # Üst üçgen matrisini al (diagonal hariç)
        n = len(X_class)
        mask = np.triu(np.ones((n, n)), k=1).astype(bool)
        sim_values = sim_matrix[mask]
        
        # Ortalama benzerlik
        avg_similarity = np.mean(sim_values)
        similarities[class_label] = avg_similarity
    
    return similarities


def calculate_inter_class_similarity(X: np.ndarray,
                                    y: np.ndarray,
                                    metric: str = 'cosine') -> np.ndarray:
    """
    Inter-sınıf (sınıflar arası) benzerlik hesapla
    
    Args:
        X: Özellik matrisi (n_samples, n_features)
        y: Sınıf etiketleri (n_samples,)
        metric: Benzerlik metriği ('cosine', 'euclidean')
        
    Returns:
        Sınıf çiftleri arası benzerlik matrisi (n_classes, n_classes)
    """
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    similarity_matrix = np.zeros((n_classes, n_classes))
    
    # Her sınıf için centroid hesapla
    centroids = {}
    for class_label in unique_classes:
        class_mask = y == class_label
        X_class = X[class_mask]
        centroids[class_label] = np.mean(X_class, axis=0)
    
    # Centroid'ler arası benzerlik hesapla
    for i, class_i in enumerate(unique_classes):
        for j, class_j in enumerate(unique_classes):
            centroid_i = centroids[class_i].reshape(1, -1)
            centroid_j = centroids[class_j].reshape(1, -1)
            
            if metric == 'cosine':
                sim = cosine_similarity(centroid_i, centroid_j)[0, 0]
            elif metric == 'euclidean':
                dist = euclidean_distances(centroid_i, centroid_j)[0, 0]
                # Normalize et ve benzerliğe çevir
                max_dist = np.max([np.linalg.norm(centroid_i), np.linalg.norm(centroid_j)])
                if max_dist > 0:
                    sim = 1 - (dist / max_dist)
                else:
                    sim = 1.0
            else:
                raise ValueError(f"Bilinmeyen metrik: {metric}")
            
            similarity_matrix[i, j] = sim
    
    return similarity_matrix


def analyze_dataset_difficulty(X: np.ndarray,
                               y: np.ndarray,
                               metric: str = 'cosine') -> Dict[str, float]:
    """
    Veri setinin zorluğunu analiz et
    
    Args:
        X: Özellik matrisi
        y: Sınıf etiketleri
        metric: Benzerlik metriği
        
    Returns:
        Zorluk analizi sonuçları
    """
    # İntra-sınıf benzerlik
    intra_sim = calculate_intra_class_similarity(X, y, metric=metric)
    avg_intra_sim = np.mean(list(intra_sim.values()))
    
    # Inter-sınıf benzerlik
    inter_sim_matrix = calculate_inter_class_similarity(X, y, metric=metric)
    # Diagonal hariç ortalamayı al
    mask = ~np.eye(inter_sim_matrix.shape[0], dtype=bool)
    avg_inter_sim = np.mean(inter_sim_matrix[mask])
    
    # Separability skoru (ne kadar yüksekse o kadar kolay)
    separability = avg_intra_sim - avg_inter_sim
    
    results = {
        'avg_intra_class_similarity': avg_intra_sim,
        'avg_inter_class_similarity': avg_inter_sim,
        'separability_score': separability,
        'difficulty': 'kolay' if separability > 0.3 else 'orta' if separability > 0.1 else 'zor'
    }
    
    return results


def print_similarity_report(X: np.ndarray,
                           y: np.ndarray,
                           class_names: Optional[List[str]] = None,
                           metric: str = 'cosine'):
    """
    Benzerlik analizi raporu yazdır
    
    Args:
        X: Özellik matrisi
        y: Sınıf etiketleri
        class_names: Sınıf isimleri
        metric: Benzerlik metriği
    """
    unique_classes = np.unique(y)
    if class_names is None:
        class_names = [str(c) for c in unique_classes]
    
    print("=" * 60)
    print("BENZERLİK ANALİZİ RAPORU")
    print("=" * 60)
    
    # İntra-sınıf benzerlik
    print("\nİntra-Sınıf (Sınıf İçi) Benzerlik:")
    intra_sim = calculate_intra_class_similarity(X, y, metric=metric)
    for class_label, sim in intra_sim.items():
        class_name = class_names[np.where(unique_classes == class_label)[0][0]]
        print(f"  {class_name}: {sim:.4f}")
    
    print(f"\nOrtalama İntra-Sınıf Benzerlik: {np.mean(list(intra_sim.values())):.4f}")
    
    # Inter-sınıf benzerlik
    print("\nInter-Sınıf (Sınıflar Arası) Benzerlik Matrisi:")
    inter_sim = calculate_inter_class_similarity(X, y, metric=metric)
    inter_df = pd.DataFrame(inter_sim, index=class_names, columns=class_names)
    print(inter_df.round(4))
    
    # Zorluk analizi
    print("\nVeri Seti Zorluk Analizi:")
    difficulty = analyze_dataset_difficulty(X, y, metric=metric)
    print(f"  Ortalama İntra-Sınıf Benzerlik: {difficulty['avg_intra_class_similarity']:.4f}")
    print(f"  Ortalama Inter-Sınıf Benzerlik: {difficulty['avg_inter_class_similarity']:.4f}")
    print(f"  Separability Skoru: {difficulty['separability_score']:.4f}")
    print(f"  Zorluk Seviyesi: {difficulty['difficulty']}")
    
    print("=" * 60)

