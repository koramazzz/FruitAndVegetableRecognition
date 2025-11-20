"""
Veri yükleme ve bölme modülü
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_dataset(data_dir: str = "data/processed",
                metadata_file: str = "metadata.csv",
                descriptions_file: str = "descriptions.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Veri setini yükle
    
    Args:
        data_dir: Veri dizini
        metadata_file: Metadata dosya adı
        descriptions_file: Açıklamalar dosya adı
        
    Returns:
        (metadata_df, descriptions_df) tuple'ı
    """
    data_path = Path(data_dir)
    
    # Metadata'yı yükle
    metadata_path = data_path / metadata_file
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata dosyası bulunamadı: {metadata_path}")
    
    metadata_df = pd.read_csv(metadata_path)
    
    # Açıklamaları yükle
    descriptions_path = data_path / descriptions_file
    if descriptions_path.exists():
        descriptions_df = pd.read_csv(descriptions_path, encoding='utf-8')
    else:
        print(f"Uyarı: Açıklamalar dosyası bulunamadı: {descriptions_path}")
        descriptions_df = pd.DataFrame()
    
    return metadata_df, descriptions_df


def split_dataset(X: np.ndarray,
                 y: np.ndarray,
                 train_size: int = 2500,
                 test_size: int = 500,
                 val_size: int = 500,
                 random_state: int = 42,
                 stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray, np.ndarray]:
    """
    Veri setini eğitim, test ve validasyon setlerine böl
    
    Args:
        X: Özellik matrisi
        y: Etiketler
        train_size: Eğitim seti boyutu
        test_size: Test seti boyutu
        val_size: Validasyon seti boyutu
        random_state: Rastgele sayı üreteci seed'i
        stratify: Stratified splitting kullan
        
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test) tuple'ı
    """
    n_samples = len(X)
    total_requested = train_size + test_size + val_size
    
    if n_samples < total_requested:
        print(f"Uyarı: Toplam örnek sayısı ({n_samples}) istenen toplamdan ({total_requested}) az. "
              f"Tüm veri kullanılacak.")
        # Mevcut veriyi orantılı olarak böl
        train_ratio = train_size / total_requested
        test_ratio = test_size / total_requested
        val_ratio = val_size / total_requested
        
        # Önce test setini ayır
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=random_state,
            stratify=y if stratify else None
        )
        
        # Kalanı train ve val'e böl
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=random_state,
            stratify=y_temp if stratify else None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # Önce test setini ayır
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )
    
    # Kalanı train ve val'e böl
    # Validasyon seti eğitim setinden alınacak
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp if stratify else None
    )
    
    # Eğitim seti boyutunu kontrol et ve gerekirse ayarla
    if len(X_train) > train_size:
        # Fazla örnekleri çıkar
        indices = np.random.RandomState(random_state).choice(
            len(X_train), size=train_size, replace=False
        )
        X_train = X_train[indices]
        y_train = y_train[indices]
    elif len(X_train) < train_size:
        print(f"Uyarı: Eğitim seti boyutu ({len(X_train)}) istenen boyuttan ({train_size}) az.")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_splits(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
               y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
               save_dir: str = "data"):
    """
    Bölünmüş veri setlerini kaydet
    
    Args:
        X_train, X_val, X_test: Özellik matrisleri
        y_train, y_val, y_test: Etiketler
        save_dir: Kayıt dizini
    """
    save_path = Path(save_dir)
    
    # Dizinleri oluştur
    (save_path / "train").mkdir(parents=True, exist_ok=True)
    (save_path / "test").mkdir(parents=True, exist_ok=True)
    (save_path / "val").mkdir(parents=True, exist_ok=True)
    
    # Kaydet
    np.save(save_path / "train" / "X_train.npy", X_train)
    np.save(save_path / "train" / "y_train.npy", y_train)
    np.save(save_path / "val" / "X_val.npy", X_val)
    np.save(save_path / "val" / "y_val.npy", y_val)
    np.save(save_path / "test" / "X_test.npy", X_test)
    np.save(save_path / "test" / "y_test.npy", y_test)
    
    print(f"Veri setleri kaydedildi: {save_path}")


def load_splits(load_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray, np.ndarray]:
    """
    Kaydedilmiş veri setlerini yükle
    
    Args:
        load_dir: Yükleme dizini
        
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test) tuple'ı
    """
    load_path = Path(load_dir)
    
    X_train = np.load(load_path / "train" / "X_train.npy")
    y_train = np.load(load_path / "train" / "y_train.npy")
    X_val = np.load(load_path / "val" / "X_val.npy")
    y_val = np.load(load_path / "val" / "y_val.npy")
    X_test = np.load(load_path / "test" / "X_test.npy")
    y_test = np.load(load_path / "test" / "y_test.npy")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

