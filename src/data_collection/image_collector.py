"""
Image collection and preprocessing module
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import shutil
from pathlib import Path


class ImageCollector:
    """Image collection and preprocessing class"""
    
    def __init__(self, base_dir: str = "data/raw/images"):
        """
        Args:
            base_dir: Base directory where images will be saved
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Categories
        self.categories = ['banana', 'tomato', 'cucumber', 'mandarin', 'potato']
        
    def create_category_dirs(self):
        """Create directories for each category"""
        for category in self.categories:
            category_dir = self.base_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_images(self, 
                      source_dir: Optional[str] = None,
                      min_samples_per_category: int = 50,
                      target_samples_per_category: int = 600):
        """
        Collect and organize images
        
        Args:
            source_dir: Source image directory (None if manual collection is expected)
            min_samples_per_category: Minimum number of samples per category
            target_samples_per_category: Target number of samples per category
        """
        self.create_category_dirs()
        
        if source_dir:
            self._copy_from_source(source_dir)
        else:
            print(f"Please place at least {min_samples_per_category} images from each category "
                  f"in the following directories:")
            for category in self.categories:
                print(f"  - {self.base_dir / category}")
    
    def _copy_from_source(self, source_dir: str):
        """Copy images from source directory"""
        source_path = Path(source_dir)
        if not source_path.exists():
            raise ValueError(f"Source directory not found: {source_dir}")
        
        # Copy images by category
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
        Preprocess image
        
        Args:
            image_path: Image file path
            target_size: Target size (width, height)
            normalize: Apply normalization
            
        Returns:
            Preprocessed image array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image could not be read: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, target_size)
        
        # Normalization
        if normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
    
    def augment_image(self, 
                     image: np.ndarray,
                     augmentations: List[str] = ['flip', 'rotate', 'brightness']) -> List[np.ndarray]:
        """
        Apply image augmentation
        
        Args:
            image: Image array
            augmentations: Augmentations to apply
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        if 'flip' in augmentations:
            # Horizontal flip
            flipped = np.fliplr(image)
            augmented_images.append(flipped)
        
        if 'rotate' in augmentations:
            # Rotate 90 degrees
            rotated = np.rot90(image)
            augmented_images.append(rotated)
        
        if 'brightness' in augmentations:
            # Adjust brightness
            bright = np.clip(image * 1.2, 0, 1)
            dark = np.clip(image * 0.8, 0, 1)
            augmented_images.extend([bright, dark])
        
        return augmented_images
    
    def get_image_paths(self, category: str) -> List[str]:
        """
        Get image paths for a specific category
        
        Args:
            category: Category name
            
        Returns:
            List of image file paths
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
        Get image paths for all categories
        
        Returns:
            Category -> image paths dictionary
        """
        all_paths = {}
        for category in self.categories:
            all_paths[category] = self.get_image_paths(category)
        return all_paths

