"""
Metin açıklama toplama modülü
"""

import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import random


class TextCollector:
    """Metin açıklamaları toplama sınıfı"""
    
    def __init__(self, base_dir: str = "data/raw"):
        """
        Args:
            base_dir: Metin verilerinin kaydedileceği temel dizin
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Kategoriler
        self.categories = ['muz', 'domates', 'salatalik', 'mandalina', 'patates']
        
        # Kategori mapping (İngilizce -> Türkçe)
        self.category_mapping = {
            'banana': 'muz',
            'tomato': 'domates',
            'cucumber': 'salatalik',
            'mandarin': 'mandalina',
            'potato': 'patates'
        }
        
        # Generic açıklamalar - tüm kategoriler için ortak havuz
        # IMPORTANT: Using generic descriptions to avoid data leakage
        # All categories share the same description pool
        self.generic_descriptions = [
            "Renkli, çeşitli şekillerde olabilen gıda maddesi.",
            "Taze, sağlıklı ve besleyici bir ürün.",
            "Doğal, organik ve lezzetli bir gıda.",
            "Vitamin ve mineral açısından zengin ürün.",
            "Mutfakta çeşitli şekillerde kullanılabilen gıda.",
            "Taze ve kaliteli bir ürün.",
            "Sağlıklı beslenme için önemli bir gıda maddesi.",
            "Doğal koşullarda yetiştirilmiş ürün.",
            "Besin değeri yüksek, lezzetli bir gıda.",
            "Taze, sulu ve lezzetli bir ürün.",
            "Sağlıklı ve dengeli beslenme için ideal gıda.",
            "Doğal renk ve dokuda bir ürün.",
            "Vitamin açısından zengin, sağlıklı gıda.",
            "Taze ve kaliteli, mutfakta çok kullanılan ürün.",
            "Besleyici değeri yüksek, lezzetli bir gıda maddesi.",
            "Doğal şekil ve renkte bir ürün.",
            "Taze, sağlıklı ve besleyici gıda.",
            "Vitamin ve mineral deposu bir ürün.",
            "Mutfakta çeşitli yemeklerde kullanılabilen gıda.",
            "Doğal, organik ve kaliteli bir ürün."
        ]
        
        # Her kategori için aynı generic açıklamaları kullan
        self.descriptions = {
            'muz': self.generic_descriptions,
            'domates': self.generic_descriptions,
            'salatalik': self.generic_descriptions,
            'mandalina': self.generic_descriptions,
            'patates': self.generic_descriptions
        }
    
    def generate_descriptions(self, 
                            sample_ids: List[str],
                            categories: List[str],
                            seed: Optional[int] = None) -> pd.DataFrame:
        """
        Örnek ID'leri için açıklamalar oluştur
        
        Args:
            sample_ids: Örnek ID'leri listesi
            categories: Kategori listesi (sample_ids ile aynı uzunlukta)
            seed: Rastgele sayı üreteci seed'i
            
        Returns:
            Açıklamalar DataFrame'i
        """
        if seed is not None:
            random.seed(seed)
        
        if len(sample_ids) != len(categories):
            raise ValueError("sample_ids ve categories aynı uzunlukta olmalıdır")
        
        descriptions_list = []
        
        for sample_id, category in zip(sample_ids, categories):
            # İngilizce kategori isimlerini Türkçe'ye çevir
            category_key = self.category_mapping.get(category, category)
            
            if category_key not in self.descriptions:
                description = f"{category} kategorisinde bir örnek."
            else:
                description = random.choice(self.descriptions[category_key])
            
            descriptions_list.append({
                'sample_id': sample_id,
                'category': category,
                'description': description
            })
        
        df = pd.DataFrame(descriptions_list)
        return df
    
    def save_descriptions(self, 
                         df: pd.DataFrame,
                         filename: str = "descriptions.csv"):
        """
        Açıklamaları CSV olarak kaydet
        
        Args:
            df: Açıklamalar DataFrame'i
            filename: Dosya adı
        """
        filepath = self.base_dir / filename
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Açıklamalar kaydedildi: {filepath}")
    
    def load_descriptions(self, filename: str = "descriptions.csv") -> pd.DataFrame:
        """
        Açıklamaları CSV'den yükle
        
        Args:
            filename: Dosya adı
            
        Returns:
            Açıklamalar DataFrame'i
        """
        filepath = self.base_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Açıklamalar dosyası bulunamadı: {filepath}")
        
        df = pd.read_csv(filepath, encoding='utf-8')
        return df
    
    def add_custom_description(self, 
                              category: str,
                              description: str):
        """
        Belirli bir kategori için özel açıklama ekle
        
        Args:
            category: Kategori adı
            description: Açıklama metni
        """
        if category not in self.descriptions:
            self.descriptions[category] = []
        
        self.descriptions[category].append(description)
    
    def get_descriptions_for_category(self, category: str) -> List[str]:
        """
        Belirli bir kategori için mevcut açıklamaları getir
        
        Args:
            category: Kategori adı
            
        Returns:
            Açıklama listesi
        """
        return self.descriptions.get(category, [])

