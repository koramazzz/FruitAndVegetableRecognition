"""
Görüntü özellik çıkarımı modülü
Derin öğrenme kullanılmadan klasik yöntemlerle özellik çıkarımı
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage import color
from typing import List, Tuple
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
        lbp_max = lbp.max()
        n_bins = lbp_max + 1
        
        # Ensure n_bins is a Python int
        if isinstance(n_bins, (np.integer, np.ndarray)):
            n_bins = int(n_bins.item() if hasattr(n_bins, 'item') else n_bins)
        else:
            n_bins = int(n_bins)
        
        try:
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        except Exception as e:
            print(f"[ERROR extract_lbp_features] np.histogram failed: {type(e).__name__}: {e}")
            print(f"[ERROR extract_lbp_features] n_bins value: {n_bins}, type: {type(n_bins)}")
            import traceback
            print(f"[ERROR extract_lbp_features] Traceback:\n{traceback.format_exc()}")
            raise
        
        return hist
    
    def extract_color_histogram(self, 
                               image: np.ndarray,
                               bins: int = 32,
                               color_space: str = 'rgb') -> np.ndarray:
        """
        Extract color histogram features
        
        Args:
            image: Image array (RGB, normalized 0-1 or uint8 0-255)
            bins: Number of histogram bins
            color_space: Color space ('rgb', 'hsv')
            
        Returns:
            Color histogram feature vector
        """
        # Ensure bins is a pure Python integer (not numpy scalar)
        # Handle all possible types that bins might be
        if bins is None:
            bins = 32
        elif isinstance(bins, np.integer):
            bins = int(bins.item())
        elif isinstance(bins, np.ndarray):
            bins = int(bins.item() if bins.size > 0 else 32)
        elif isinstance(bins, (float, np.floating)):
            bins = int(round(bins))
        else:
            try:
                bins = int(bins)
            except (TypeError, ValueError):
                bins = 32  # Fallback to default
        
        # Final safety check
        if not isinstance(bins, int) or bins <= 0:
            bins = 32
        
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            # Clamp values to [0, 1] range before conversion
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
        
        # Ensure image is contiguous and has correct shape
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        if color_space == 'hsv':
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            channels = [0, 1, 2]  # H, S, V
        else:
            # Use RGB
            hsv = image
            channels = [0, 1, 2]  # R, G, B
        
        # Calculate histogram for each channel
        hist_features = []
        
        for channel in channels:
            # Extract single channel
            channel_data = hsv[:, :, channel].flatten()
            
            # Use numpy histogram with explicit bin edges (most reliable method)
            # This avoids any type issues with the bins parameter
            try:
                bin_edges = np.linspace(0, 256, bins + 1)
                hist, _ = np.histogram(channel_data, bins=bin_edges, density=False)
                hist = hist.astype(np.float32)
                hist_features.extend(hist)
            except Exception as e:
                print(f"[ERROR extract_color_histogram] Error in channel {channel}: {type(e).__name__}: {e}")
                print(f"[ERROR extract_color_histogram] bins value: {bins}, type: {type(bins)}")
                import traceback
                print(f"[ERROR extract_color_histogram] Traceback:\n{traceback.format_exc()}")
                raise
        
        # Normalize
        hist_features = np.array(hist_features, dtype=np.float32)
        hist_sum = hist_features.sum()
        if hist_sum > 0:
            hist_features = hist_features / hist_sum
        
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

