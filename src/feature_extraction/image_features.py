"""
Görüntü özellik çıkarımı modülü
Derin öğrenme kullanılmadan klasik yöntemlerle özellik çıkarımı
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage import color
from typing import List, Union, Tuple
import os


class ImageFeatureExtractor:
    """Görüntü özellik çıkarımı sınıfı"""
    
    def __init__(self, 
                 use_hog: bool = True,
                 use_lbp: bool = True,
                 use_color_hist: bool = True,
                 use_sift: bool = False):
        """
        Args:
            use_hog: HOG özelliklerini kullan
            use_lbp: LBP özelliklerini kullan
            use_color_hist: Renk histogramlarını kullan
            use_sift: SIFT özelliklerini kullan (daha yavaş)
        """
        self.use_hog = use_hog
        self.use_lbp = use_lbp
        self.use_color_hist = use_color_hist
        self.use_sift = use_sift
    
    def extract_hog_features(self, 
                            image: np.ndarray,
                            orientations: int = 9,
                            pixels_per_cell: Tuple[int, int] = (8, 8),
                            cells_per_block: Tuple[int, int] = (2, 2)) -> np.ndarray:
        """
        HOG (Histogram of Oriented Gradients) özelliklerini çıkar
        
        Args:
            image: Görüntü array'i (RGB veya grayscale)
            orientations: Yön sayısı
            pixels_per_cell: Hücre başına piksel sayısı
            cells_per_block: Blok başına hücre sayısı
            
        Returns:
            HOG özellik vektörü
        """
        # Grayscale'e çevir
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image
        
        # HOG özelliklerini çıkar
        features = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            feature_vector=True
        )
        
        return features
    
    def extract_lbp_features(self, 
                           image: np.ndarray,
                           radius: int = 3,
                           n_points: int = 24,
                           method: str = 'uniform') -> np.ndarray:
        """
        LBP (Local Binary Pattern) özelliklerini çıkar
        
        Args:
            image: Görüntü array'i
            radius: LBP yarıçapı
            n_points: Komşu nokta sayısı
            method: LBP metodu ('uniform', 'default', vb.)
            
        Returns:
            LBP histogram özellik vektörü
        """
        # Grayscale'e çevir
        if len(image.shape) == 3:
            gray = color.rgb2gray(image)
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = image.astype(np.uint8) if image.dtype != np.uint8 else image
        
        # LBP hesapla
        lbp = local_binary_pattern(gray, n_points, radius, method=method)
        
        # Histogram oluştur
        n_bins = lbp.max() + 1
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist
    
    def extract_color_histogram(self, 
                               image: np.ndarray,
                               bins: int = 32,
                               color_space: str = 'rgb') -> np.ndarray:
        """
        Renk histogramı özelliklerini çıkar
        
        Args:
            image: Görüntü array'i (RGB)
            bins: Histogram bin sayısı
            color_space: Renk uzayı ('rgb', 'hsv')
            
        Returns:
            Renk histogram özellik vektörü
        """
        if color_space == 'hsv':
            # HSV'ye çevir
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            channels = [0, 1, 2]  # H, S, V
        else:
            # RGB kullan
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            hsv = image
            channels = [0, 1, 2]  # R, G, B
        
        # Her kanal için histogram hesapla
        hist_features = []
        for channel in channels:
            hist = cv2.calcHist([hsv], [channel], None, [bins], [0, 256])
            hist_features.extend(hist.flatten())
        
        # Normalize et
        hist_features = np.array(hist_features)
        hist_features = hist_features / (hist_features.sum() + 1e-7)
        
        return hist_features
    
    def extract_sift_features(self, 
                             image: np.ndarray,
                             max_features: int = 100) -> np.ndarray:
        """
        SIFT özelliklerini çıkar (bag-of-words benzeri)
        
        Args:
            image: Görüntü array'i
            max_features: Maksimum özellik sayısı
            
        Returns:
            SIFT özellik vektörü (histogram)
        """
        # Grayscale'e çevir
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        if image.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)
        
        # SIFT detector oluştur
        sift = cv2.SIFT_create(nfeatures=max_features)
        
        # Keypoint ve descriptor'ları bul
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(descriptors) == 0:
            # Özellik bulunamazsa sıfır vektör döndür
            return np.zeros(128)
        
        # Descriptor'ların ortalamasını al (basit bir yaklaşım)
        feature_vector = descriptors.mean(axis=0)
        
        return feature_vector
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Tüm özellikleri çıkar ve birleştir
        
        Args:
            image: Görüntü array'i (RGB, 0-1 arası normalize edilmiş veya 0-255 uint8)
            
        Returns:
            Birleştirilmiş özellik vektörü
        """
        features = []
        
        # Görüntüyü normalize et (0-1 arası)
        if image.dtype == np.uint8:
            image_normalized = image.astype(np.float32) / 255.0
        else:
            image_normalized = image.copy()
        
        # HOG özellikleri
        if self.use_hog:
            hog_feat = self.extract_hog_features(image_normalized)
            features.append(hog_feat)
        
        # LBP özellikleri
        if self.use_lbp:
            lbp_feat = self.extract_lbp_features(image_normalized)
            features.append(lbp_feat)
        
        # Renk histogramı
        if self.use_color_hist:
            color_feat = self.extract_color_histogram(image_normalized)
            features.append(color_feat)
        
        # SIFT özellikleri (opsiyonel, daha yavaş)
        if self.use_sift:
            sift_feat = self.extract_sift_features(image_normalized)
            features.append(sift_feat)
        
        # Tüm özellikleri birleştir
        if features:
            combined = np.concatenate(features)
        else:
            raise ValueError("En az bir özellik çıkarım yöntemi seçilmelidir")
        
        return combined


def extract_image_features(image_paths: List[str],
                          extractor: ImageFeatureExtractor = None) -> np.ndarray:
    """
    Birden fazla görüntüden özellik çıkar
    
    Args:
        image_paths: Görüntü dosya yolları listesi
        extractor: ImageFeatureExtractor instance (None ise varsayılan oluşturulur)
        
    Returns:
        Özellik matrisi (n_samples, n_features)
    """
    if extractor is None:
        extractor = ImageFeatureExtractor()
    
    features_list = []
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Uyarı: Görüntü bulunamadı: {image_path}")
            continue
        
        # Görüntüyü oku
        image = cv2.imread(image_path)
        if image is None:
            print(f"Uyarı: Görüntü okunamadı: {image_path}")
            continue
        
        # BGR'den RGB'ye çevir
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Özellikleri çıkar
        features = extractor.extract_features(image)
        features_list.append(features)
    
    if not features_list:
        raise ValueError("Hiçbir görüntüden özellik çıkarılamadı")
    
    return np.array(features_list)

