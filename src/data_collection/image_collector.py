"""
Görüntü toplama ve ön işleme modülü
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import shutil
from pathlib import Path


class ImageCollector:
    """Görüntü toplama ve ön işleme sınıfı"""
    
    def __init__(self, base_dir: str = "data/raw/images"):
        """
        Args:
            base_dir: Görüntülerin kaydedileceği temel dizin
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Kategoriler
        self.categories = ['muz', 'domates', 'salatalik', 'mandalina', 'patates']
        
    def create_category_dirs(self):
        """Her kategori için dizin oluştur"""
        for category in self.categories:
            category_dir = self.base_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_images(self, 
                      source_dir: Optional[str] = None,
                      min_samples_per_category: int = 50,
                      target_samples_per_category: int = 600):
        """
        Görüntüleri topla ve organize et
        
        Args:
            source_dir: Kaynak görüntü dizini (None ise manuel toplama beklenir)
            min_samples_per_category: Her kategoriden minimum örnek sayısı
            target_samples_per_category: Her kategoriden hedef örnek sayısı
        """
        self.create_category_dirs()
        
        if source_dir:
            self._copy_from_source(source_dir)
        else:
            print(f"Lütfen her kategoriden en az {min_samples_per_category} görüntüyü "
                  f"şu dizinlere yerleştirin:")
            for category in self.categories:
                print(f"  - {self.base_dir / category}")
    
    def _copy_from_source(self, source_dir: str):
        """Kaynak dizinden görüntüleri kopyala"""
        source_path = Path(source_dir)
        if not source_path.exists():
            raise ValueError(f"Kaynak dizin bulunamadı: {source_dir}")
        
        # Kategori bazında görüntüleri kopyala
        for category in self.categories:
            source_category_dir = source_path / category
            if source_category_dir.exists():
                target_category_dir = self.base_dir / category
                for img_file in source_category_dir.glob("*.*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        shutil.copy2(img_file, target_category_dir / img_file.name)
    
    def preprocess_image(self, 
                        image_path: str,
                        target_size: Tuple[int, int] = (224, 224),
                        normalize: bool = True) -> np.ndarray:
        """
        Görüntüyü ön işle
        
        Args:
            image_path: Görüntü dosya yolu
            target_size: Hedef boyut (genişlik, yükseklik)
            normalize: Normalizasyon uygula mı
            
        Returns:
            İşlenmiş görüntü array'i
        """
        # Görüntüyü oku
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Görüntü okunamadı: {image_path}")
        
        # BGR'den RGB'ye çevir
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Boyutlandır
        img = cv2.resize(img, target_size)
        
        # Normalizasyon
        if normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
    
    def augment_image(self, 
                     image: np.ndarray,
                     augmentations: List[str] = ['flip', 'rotate', 'brightness']) -> List[np.ndarray]:
        """
        Görüntü augmentasyonu uygula
        
        Args:
            image: Görüntü array'i
            augmentations: Uygulanacak augmentasyonlar
            
        Returns:
            Augment edilmiş görüntü listesi
        """
        augmented_images = []
        
        if 'flip' in augmentations:
            # Yatay çevir
            flipped = np.fliplr(image)
            augmented_images.append(flipped)
        
        if 'rotate' in augmentations:
            # 90 derece döndür
            rotated = np.rot90(image)
            augmented_images.append(rotated)
        
        if 'brightness' in augmentations:
            # Parlaklık ayarla
            bright = np.clip(image * 1.2, 0, 1)
            dark = np.clip(image * 0.8, 0, 1)
            augmented_images.extend([bright, dark])
        
        return augmented_images
    
    def get_image_paths(self, category: str) -> List[str]:
        """
        Belirli bir kategoriye ait görüntü yollarını getir
        
        Args:
            category: Kategori adı
            
        Returns:
            Görüntü dosya yolları listesi
        """
        category_dir = self.base_dir / category
        if not category_dir.exists():
            return []
        
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend([str(p) for p in category_dir.glob(ext)])
            image_paths.extend([str(p) for p in category_dir.glob(ext.upper())])
        
        return sorted(image_paths)
    
    def get_all_image_paths(self) -> dict:
        """
        Tüm kategorilere ait görüntü yollarını getir
        
        Returns:
            Kategori -> görüntü yolları dictionary'si
        """
        all_paths = {}
        for category in self.categories:
            all_paths[category] = self.get_image_paths(category)
        return all_paths

