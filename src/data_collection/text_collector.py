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
        
        # Her kategori için örnek açıklamalar
        self.descriptions = {
            'muz': [
                "Sarı renkli, uzun ve kavisli bir meyve.",
                "Tropikal bir meyve, sarı kabuklu ve tatlı.",
                "Uzun, sarı renkli, yumuşak dokulu meyve.",
                "Sarı kabuklu, içi beyaz, tatlı bir meyve.",
                "Kavisli şekilli, sarı renkli tropikal meyve.",
                "Yumuşak, sarı renkli, potasyum açısından zengin meyve.",
                "Uzun ve kavisli, sarı veya yeşil renkli meyve.",
                "Tropik bölgelerden gelen sarı renkli meyve.",
                "Kabuğu sarı, içi yumuşak ve tatlı meyve.",
                "Sarı renkli, uzun şekilli, enerji veren meyve."
            ],
            'domates': [
                "Kırmızı renkli, yuvarlak şekilli sebze.",
                "Parlak kırmızı renkli, sulu ve lezzetli sebze.",
                "Yuvarlak veya oval, kırmızı renkli sebze.",
                "Kırmızı renkli, içi sulu, çekirdekli sebze.",
                "Parlak kırmızı, yuvarlak şekilli, C vitamini açısından zengin.",
                "Kırmızı renkli, yumuşak dokulu, salata için ideal sebze.",
                "Yuvarlak, kırmızı renkli, mutfakta çok kullanılan sebze.",
                "Parlak kırmızı, sulu, lezzetli bir sebze.",
                "Kırmızı renkli, yuvarlak, içi çekirdekli sebze.",
                "Yuvarlak şekilli, kırmızı renkli, salata ve yemek için sebze."
            ],
            'salatalik': [
                "Yeşil renkli, uzun ve silindirik şekilli sebze.",
                "Açık yeşil renkli, sulu ve ferahlatıcı sebze.",
                "Uzun, yeşil renkli, çıtır dokulu sebze.",
                "Yeşil kabuklu, içi sulu, serinletici sebze.",
                "Uzun ve silindirik, yeşil renkli, düşük kalorili sebze.",
                "Açık yeşil, uzun şekilli, salata için ideal sebze.",
                "Yeşil renkli, uzun, sulu ve ferahlatıcı sebze.",
                "Uzun şekilli, yeşil renkli, çıtır dokulu sebze.",
                "Yeşil kabuklu, içi sulu, serinletici bir sebze.",
                "Uzun ve silindirik, yeşil renkli, sağlıklı sebze."
            ],
            'mandalina': [
                "Turuncu renkli, yuvarlak, küçük turunçgiller meyvesi.",
                "Turuncu kabuklu, kolay soyulabilen, tatlı meyve.",
                "Küçük, turuncu renkli, portakala benzer meyve.",
                "Turuncu renkli, yuvarlak, C vitamini açısından zengin.",
                "Küçük ve yuvarlak, turuncu renkli, tatlı meyve.",
                "Turuncu kabuklu, kolay soyulabilen, sulu meyve.",
                "Yuvarlak şekilli, turuncu renkli, kış meyvesi.",
                "Turuncu renkli, küçük, portakala benzer meyve.",
                "Küçük ve yuvarlak, turuncu renkli, tatlı ve sulu.",
                "Turuncu kabuklu, kolay soyulabilen, C vitamini deposu."
            ],
            'patates': [
                "Kahverengi kabuklu, yuvarlak veya oval şekilli sebze.",
                "Kahverengi renkli, sert dokulu, nişastalı sebze.",
                "Yuvarlak veya oval, kahverengi kabuklu sebze.",
                "Kahverengi renkli, içi beyaz, pişirilerek tüketilen sebze.",
                "Yuvarlak şekilli, kahverengi kabuklu, karbonhidrat açısından zengin.",
                "Kahverengi renkli, sert dokulu, mutfakta çok kullanılan sebze.",
                "Yuvarlak veya oval, kahverengi renkli, doyurucu sebze.",
                "Kahverengi kabuklu, içi beyaz, pişirilerek tüketilen sebze.",
                "Yuvarlak şekilli, kahverengi renkli, nişastalı sebze.",
                "Kahverengi renkli, sert dokulu, temel gıda maddesi."
            ]
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

