"""
Kategorik ve numerik veri toplama modülü
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import random


class MetadataCollector:
    """Metadata (kategorik ve numerik özellikler) toplama sınıfı"""
    
    def __init__(self, base_dir: str = "data/raw"):
        """
        Args:
            base_dir: Metadata'nın kaydedileceği temel dizin
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Kategoriler
        self.categories = ['muz', 'domates', 'salatalik', 'mandalina', 'patates']
        
        # Kategorik özellikler için olası değerler
        self.color_options = {
            'muz': ['sari', 'yesil', 'kahverengi'],
            'domates': ['kirmizi', 'yesil', 'turuncu'],
            'salatalik': ['yesil', 'beyaz', 'sari'],
            'mandalina': ['turuncu', 'yesil', 'sari'],
            'patates': ['kahverengi', 'sari', 'beyaz']
        }
        
        self.season_options = {
            'muz': ['yaz', 'kis', 'sonbahar', 'ilkbahar'],
            'domates': ['yaz', 'sonbahar'],
            'salatalik': ['yaz', 'ilkbahar', 'sonbahar'],
            'mandalina': ['kis', 'sonbahar'],
            'patates': ['yaz', 'sonbahar', 'ilkbahar']
        }
        
        self.origin_options = {
            'muz': ['tropik', 'ithal'],
            'domates': ['yerli', 'ithal'],
            'salatalik': ['yerli', 'ithal'],
            'mandalina': ['yerli', 'ithal'],
            'patates': ['yerli', 'ithal']
        }
        
        # Numerik özellikler için aralıklar (gram cinsinden)
        self.weight_ranges = {
            'muz': (80, 200),
            'domates': (50, 300),
            'salatalik': (100, 400),
            'mandalina': (50, 150),
            'patates': (100, 500)
        }
    
    def generate_metadata(self, 
                         n_samples_per_category: int = 600,
                         seed: Optional[int] = None) -> pd.DataFrame:
        """
        Metadata oluştur
        
        Args:
            n_samples_per_category: Kategori başına örnek sayısı
            seed: Rastgele sayı üreteci seed'i
            
        Returns:
            Metadata DataFrame'i
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        data = []
        
        for category in self.categories:
            for i in range(n_samples_per_category):
                # Kategorik özellikler
                color = random.choice(self.color_options[category])
                season = random.choice(self.season_options[category])
                origin = random.choice(self.origin_options[category])
                
                # Numerik özellikler
                weight_min, weight_max = self.weight_ranges[category]
                weight = np.random.uniform(weight_min, weight_max)
                
                # Biraz gürültü ekle (outlier ve çeşitlilik için)
                if np.random.random() < 0.05:  # %5 outlier
                    weight = np.random.uniform(weight_max, weight_max * 1.5)
                
                data.append({
                    'sample_id': f"{category}_{i:04d}",
                    'category': category,
                    'weight': round(weight, 2),
                    'color': color,
                    'season': season,
                    'origin': origin
                })
        
        df = pd.DataFrame(data)
        return df
    
    def save_metadata(self, 
                     df: pd.DataFrame,
                     filename: str = "metadata.csv"):
        """
        Metadata'yı CSV olarak kaydet
        
        Args:
            df: Metadata DataFrame'i
            filename: Dosya adı
        """
        filepath = self.base_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Metadata kaydedildi: {filepath}")
    
    def load_metadata(self, filename: str = "metadata.csv") -> pd.DataFrame:
        """
        Metadata'yı CSV'den yükle
        
        Args:
            filename: Dosya adı
            
        Returns:
            Metadata DataFrame'i
        """
        filepath = self.base_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Metadata dosyası bulunamadı: {filepath}")
        
        df = pd.read_csv(filepath)
        return df
    
    def add_metadata_for_images(self, 
                               image_paths: Dict[str, List[str]],
                               existing_metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Görüntü yollarına göre metadata oluştur
        
        Args:
            image_paths: Kategori -> görüntü yolları dictionary'si
            existing_metadata: Mevcut metadata (varsa)
            
        Returns:
            Güncellenmiş metadata DataFrame'i
        """
        if existing_metadata is None:
            existing_metadata = pd.DataFrame()
        
        new_data = []
        
        for category, paths in image_paths.items():
            for i, path in enumerate(paths):
                # Kategorik özellikler
                color = random.choice(self.color_options.get(category, ['bilinmiyor']))
                season = random.choice(self.season_options.get(category, ['bilinmiyor']))
                origin = random.choice(self.origin_options.get(category, ['bilinmiyor']))
                
                # Numerik özellikler
                weight_min, weight_max = self.weight_ranges.get(category, (100, 300))
                weight = np.random.uniform(weight_min, weight_max)
                
                new_data.append({
                    'sample_id': Path(path).stem,
                    'image_path': path,
                    'category': category,
                    'weight': round(weight, 2),
                    'color': color,
                    'season': season,
                    'origin': origin
                })
        
        new_df = pd.DataFrame(new_data)
        
        if not existing_metadata.empty:
            return pd.concat([existing_metadata, new_df], ignore_index=True)
        return new_df
    
    def validate_metadata(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Metadata'yı doğrula
        
        Args:
            df: Metadata DataFrame'i
            
        Returns:
            Doğrulama sonuçları dictionary'si
        """
        results = {
            'no_missing_values': df.isnull().sum().sum() == 0,
            'all_categories_present': set(df['category'].unique()) == set(self.categories),
            'weight_positive': (df['weight'] > 0).all(),
            'valid_colors': True,
            'valid_seasons': True,
            'valid_origins': True
        }
        
        # Kategori bazında renk kontrolü
        for category in self.categories:
            category_df = df[df['category'] == category]
            if not category_df.empty:
                valid_colors = set(self.color_options[category])
                actual_colors = set(category_df['color'].unique())
                if not actual_colors.issubset(valid_colors):
                    results['valid_colors'] = False
        
        return results

