"""
Kategorik özellik encoding modülü
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import List, Union, Optional


def encode_categorical_features(data: Union[pd.DataFrame, np.ndarray],
                               columns: Optional[List[str]] = None,
                               method: str = 'onehot',
                               handle_unknown: str = 'ignore') -> np.ndarray:
    """
    Kategorik özellikleri encode et
    
    Args:
        data: Kategorik veri (DataFrame veya array)
        columns: Encode edilecek sütunlar (DataFrame için)
        method: Encoding yöntemi ('onehot', 'label')
        handle_unknown: Bilinmeyen değerler için strateji ('ignore', 'error')
        
    Returns:
        Encoded özellik matrisi
    """
    if isinstance(data, pd.DataFrame):
        if columns is None:
            # Tüm kategorik sütunları seç
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not columns:
            return np.array([]).reshape(len(data), 0)
        
        encoded_features = []
        
        for col in columns:
            if method == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
                encoded = encoder.fit_transform(data[[col]])
                encoded_features.append(encoded)
            elif method == 'label':
                encoder = LabelEncoder()
                encoded = encoder.fit_transform(data[col].values)
                encoded_features.append(encoded.reshape(-1, 1))
            else:
                raise ValueError(f"Bilinmeyen encoding yöntemi: {method}")
        
        if encoded_features:
            return np.hstack(encoded_features)
        else:
            return np.array([]).reshape(len(data), 0)
    
    else:
        # NumPy array için
        if method == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown=handle_unknown)
            data_2d = data.reshape(-1, 1) if data.ndim == 1 else data
            encoded = encoder.fit_transform(data_2d)
            return encoded
        elif method == 'label':
            encoder = LabelEncoder()
            encoded = encoder.fit_transform(data.flatten())
            return encoded.reshape(-1, 1)
        else:
            raise ValueError(f"Bilinmeyen encoding yöntemi: {method}")


def encode_metadata(metadata_df: pd.DataFrame,
                   categorical_cols: List[str] = ['color', 'season', 'origin'],
                   method: str = 'onehot') -> np.ndarray:
    """
    Metadata DataFrame'inden kategorik özellikleri encode et
    
    Args:
        metadata_df: Metadata DataFrame'i
        categorical_cols: Kategorik sütunlar listesi
        method: Encoding yöntemi
        
    Returns:
        Encoded özellik matrisi
    """
    return encode_categorical_features(metadata_df, columns=categorical_cols, method=method)


def normalize_numerical_features(data: Union[pd.DataFrame, np.ndarray],
                                 columns: Optional[List[str]] = None,
                                 method: str = 'standard') -> np.ndarray:
    """
    Numerik özellikleri normalize et
    
    Args:
        data: Numerik veri
        columns: Normalize edilecek sütunlar (DataFrame için)
        method: Normalizasyon yöntemi ('standard', 'minmax')
        
    Returns:
        Normalize edilmiş özellik matrisi
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    if isinstance(data, pd.DataFrame):
        if columns is None:
            # Tüm numerik sütunları seç
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            return np.array([]).reshape(len(data), 0)
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Bilinmeyen normalizasyon yöntemi: {method}")
        
        normalized = scaler.fit_transform(data[columns])
        return normalized
    
    else:
        # NumPy array için
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Bilinmeyen normalizasyon yöntemi: {method}")
        
        data_2d = data.reshape(-1, 1) if data.ndim == 1 else data
        normalized = scaler.fit_transform(data_2d)
        return normalized

