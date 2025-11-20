"""
Performans metrikleri modülü
Accuracy, Precision, Recall, F1-score, AUC hesaplama
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_proba: Optional[np.ndarray] = None,
                     average: str = 'weighted') -> Dict[str, float]:
    """
    Sınıflandırma metriklerini hesapla
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        y_proba: Tahmin edilen olasılıklar (AUC için)
        average: Metrik ortalaması ('weighted', 'macro', 'micro')
        
    Returns:
        Metrikler dictionary'si
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # AUC hesapla (multiclass için)
    if y_proba is not None:
        try:
            # Multiclass AUC için one-vs-rest yaklaşımı
            if len(np.unique(y_true)) > 2:
                metrics['auc'] = roc_auc_score(
                    y_true, y_proba, 
                    multi_class='ovr', 
                    average=average
                )
            else:
                # Binary classification
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba)
        except Exception as e:
            print(f"AUC hesaplanamadı: {e}")
            metrics['auc'] = None
    else:
        metrics['auc'] = None
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          class_names: Optional[list] = None,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8)):
    """
    Confusion matrix çiz
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        class_names: Sınıf isimleri listesi
        save_path: Kayıt yolu
        figsize: Figür boyutu
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def plot_roc_curve(y_true: np.ndarray,
                  y_proba: np.ndarray,
                  class_names: Optional[list] = None,
                  save_path: Optional[str] = None,
                  figsize: Tuple[int, int] = (10, 8)):
    """
    ROC eğrisi çiz
    
    Args:
        y_true: Gerçek etiketler
        y_proba: Tahmin edilen olasılıklar (n_samples, n_classes)
        class_names: Sınıf isimleri listesi
        save_path: Kayıt yolu
        figsize: Figür boyutu
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    n_classes = y_proba.shape[1]
    unique_classes = np.unique(y_true)
    
    # Etiketleri binarize et
    y_true_bin = label_binarize(y_true, classes=unique_classes)
    
    if n_classes == 2:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    
    plt.figure(figsize=figsize)
    
    # Her sınıf için ROC eğrisi
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        class_name = class_names[i] if class_names else f'Sınıf {unique_classes[i]}'
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Rastgele')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Eğrisi')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def print_classification_report(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               class_names: Optional[list] = None):
    """
    Detaylı sınıflandırma raporu yazdır
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        class_names: Sınıf isimleri listesi
    """
    if class_names is not None:
        target_names = [str(name) for name in class_names]
    else:
        target_names = None
    
    print(classification_report(y_true, y_pred, target_names=target_names))

