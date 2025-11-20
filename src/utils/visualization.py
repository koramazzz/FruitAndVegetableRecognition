"""
Görselleştirme modülü
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import pandas as pd


def plot_training_history(train_losses: List[float],
                         val_losses: Optional[List[float]] = None,
                         train_accuracies: Optional[List[float]] = None,
                         val_accuracies: Optional[List[float]] = None,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 5)):
    """
    Eğitim geçmişini çiz
    
    Args:
        train_losses: Eğitim loss değerleri
        val_losses: Validasyon loss değerleri (opsiyonel)
        train_accuracies: Eğitim accuracy değerleri (opsiyonel)
        val_accuracies: Validasyon accuracy değerleri (opsiyonel)
        save_path: Kayıt yolu
        figsize: Figür boyutu
    """
    n_plots = 1
    if train_accuracies is not None or val_accuracies is not None:
        n_plots = 2
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Loss grafiği
    ax = axes[0]
    ax.plot(train_losses, label='Eğitim Loss', marker='o')
    if val_losses is not None:
        ax.plot(val_losses, label='Validasyon Loss', marker='s')
    ax.set_xlabel('Iterasyon')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Geçmişi')
    ax.legend()
    ax.grid(True)
    
    # Accuracy grafiği (varsa)
    if n_plots == 2 and (train_accuracies is not None or val_accuracies is not None):
        ax = axes[1]
        if train_accuracies is not None:
            ax.plot(train_accuracies, label='Eğitim Accuracy', marker='o')
        if val_accuracies is not None:
            ax.plot(val_accuracies, label='Validasyon Accuracy', marker='s')
        ax.set_xlabel('Iterasyon')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Geçmişi')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_feature_distributions(X: np.ndarray,
                              y: Optional[np.ndarray] = None,
                              feature_indices: Optional[List[int]] = None,
                              n_features: int = 9,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (15, 10)):
    """
    Özellik dağılımlarını çiz
    
    Args:
        X: Özellik matrisi
        y: Etiketler (opsiyonel, sınıf bazında gösterim için)
        feature_indices: Gösterilecek özellik indeksleri
        n_features: Gösterilecek özellik sayısı
        save_path: Kayıt yolu
        figsize: Figür boyutu
    """
    if feature_indices is None:
        # İlk n_features özelliğini seç
        feature_indices = list(range(min(n_features, X.shape[1])))
    
    n_features_plot = len(feature_indices)
    n_cols = 3
    n_rows = (n_features_plot + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features_plot > 1 else [axes]
    
    for idx, feat_idx in enumerate(feature_indices):
        ax = axes[idx]
        
        if y is not None:
            # Sınıf bazında gösterim
            unique_classes = np.unique(y)
            for class_label in unique_classes:
                class_mask = y == class_label
                ax.hist(X[class_mask, feat_idx], alpha=0.5, label=f'Sınıf {class_label}', bins=30)
            ax.legend()
        else:
            ax.hist(X[:, feat_idx], bins=30)
        
        ax.set_xlabel(f'Özellik {feat_idx}')
        ax.set_ylabel('Frekans')
        ax.set_title(f'Özellik {feat_idx} Dağılımı')
        ax.grid(True, alpha=0.3)
    
    # Kullanılmayan subplot'ları kaldır
    for idx in range(n_features_plot, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_class_distribution(y: np.ndarray,
                           class_names: Optional[List[str]] = None,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 6)):
    """
    Sınıf dağılımını çiz
    
    Args:
        y: Etiketler
        class_names: Sınıf isimleri
        save_path: Kayıt yolu
        figsize: Figür boyutu
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    
    if class_names is None:
        class_names = [str(c) for c in unique_classes]
    
    plt.figure(figsize=figsize)
    plt.bar(class_names, counts)
    plt.xlabel('Sınıf')
    plt.ylabel('Örnek Sayısı')
    plt.title('Sınıf Dağılımı')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_correlation_matrix(X: np.ndarray,
                          feature_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 10)):
    """
    Korelasyon matrisini çiz
    
    Args:
        X: Özellik matrisi
        feature_names: Özellik isimleri
        save_path: Kayıt yolu
        figsize: Figür boyutu
    """
    # Korelasyon matrisini hesapla
    corr_matrix = np.corrcoef(X.T)
    
    plt.figure(figsize=figsize)
    
    if feature_names is None:
        feature_names = [f'Özellik {i}' for i in range(X.shape[1])]
    
    # Sadece ilk 20 özelliği göster (çok fazla özellik varsa)
    max_features = 20
    if len(feature_names) > max_features:
        corr_matrix = corr_matrix[:max_features, :max_features]
        feature_names = feature_names[:max_features]
    
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                xticklabels=feature_names, yticklabels=feature_names,
                square=True, linewidths=0.5)
    plt.title('Özellik Korelasyon Matrisi')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

