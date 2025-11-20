"""
Categorical and numerical data collection module
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import random


class MetadataCollector:
    """Metadata (categorical and numerical features) collection class"""
    
    def __init__(self, base_dir: str = "data/raw"):
        """
        Args:
            base_dir: Base directory where metadata will be saved
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Categories
        self.categories = ['banana', 'tomato', 'cucumber', 'mandarin', 'potato']
        
        # Possible values for categorical features
        self.color_options = {
            'banana': ['yellow', 'green', 'brown'],
            'tomato': ['red', 'green', 'orange'],
            'cucumber': ['green', 'white', 'yellow'],
            'mandarin': ['orange', 'green', 'yellow'],
            'potato': ['brown', 'yellow', 'white']
        }
        
        self.season_options = {
            'banana': ['summer', 'winter', 'autumn', 'spring'],
            'tomato': ['summer', 'autumn'],
            'cucumber': ['summer', 'spring', 'autumn'],
            'mandarin': ['winter', 'autumn'],
            'potato': ['summer', 'autumn', 'spring']
        }
        
        self.origin_options = {
            'banana': ['tropical', 'imported'],
            'tomato': ['local', 'imported'],
            'cucumber': ['local', 'imported'],
            'mandarin': ['local', 'imported'],
            'potato': ['local', 'imported']
        }
        
        # Ranges for numerical features (in grams)
        self.weight_ranges = {
            'banana': (80, 200),
            'tomato': (50, 300),
            'cucumber': (100, 400),
            'mandarin': (50, 150),
            'potato': (100, 500)
        }
    
    def generate_metadata(self, 
                         n_samples_per_category: int = 600,
                         seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate metadata
        
        Args:
            n_samples_per_category: Number of samples per category
            seed: Random seed
            
        Returns:
            Metadata DataFrame
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        data = []
        
        for category in self.categories:
            for i in range(n_samples_per_category):
                # Categorical features
                color = random.choice(self.color_options[category])
                season = random.choice(self.season_options[category])
                origin = random.choice(self.origin_options[category])
                
                # Numerical features
                weight_min, weight_max = self.weight_ranges[category]
                weight = np.random.uniform(weight_min, weight_max)
                
                # Add some noise (for outliers and diversity)
                if np.random.random() < 0.05:  # 5% outliers
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
        Save metadata as CSV
        
        Args:
            df: Metadata DataFrame
            filename: File name
        """
        filepath = self.base_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Metadata saved: {filepath}")
    
    def load_metadata(self, filename: str = "metadata.csv") -> pd.DataFrame:
        """
        Load metadata from CSV
        
        Args:
            filename: File name
            
        Returns:
            Metadata DataFrame
        """
        filepath = self.base_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        return df
    
    def add_metadata_for_images(self, 
                               image_paths: Dict[str, List[str]],
                               existing_metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create metadata based on image paths
        
        Args:
            image_paths: Category -> image paths dictionary
            existing_metadata: Existing metadata (if any)
            
        Returns:
            Updated metadata DataFrame
        """
        if existing_metadata is None:
            existing_metadata = pd.DataFrame()
        
        new_data = []
        
        for category, paths in image_paths.items():
            for i, path in enumerate(paths):
                # Categorical features
                color = random.choice(self.color_options.get(category, ['unknown']))
                season = random.choice(self.season_options.get(category, ['unknown']))
                origin = random.choice(self.origin_options.get(category, ['unknown']))
                
                # Numerical features
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
        Validate metadata
        
        Args:
            df: Metadata DataFrame
            
        Returns:
            Validation results dictionary
        """
        results = {
            'no_missing_values': df.isnull().sum().sum() == 0,
            'all_categories_present': set(df['category'].unique()) == set(self.categories),
            'weight_positive': (df['weight'] > 0).all(),
            'valid_colors': True,
            'valid_seasons': True,
            'valid_origins': True
        }
        
        # Color check by category
        for category in self.categories:
            category_df = df[df['category'] == category]
            if not category_df.empty:
                valid_colors = set(self.color_options[category])
                actual_colors = set(category_df['color'].unique())
                if not actual_colors.issubset(valid_colors):
                    results['valid_colors'] = False
        
        return results

