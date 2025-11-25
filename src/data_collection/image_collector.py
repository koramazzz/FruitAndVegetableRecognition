"""
Image collection and preprocessing module
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import shutil
from pathlib import Path
import random


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
                     augmentations: List[str] = ['flip', 'rotate', 'brightness'],
                     n_augmentations: int = None) -> List[np.ndarray]:
        """
        Apply image augmentation
        
        Args:
            image: Image array (normalized 0-1)
            augmentations: Augmentations to apply
            n_augmentations: Number of augmentations to generate (if None, uses all available)
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        if 'flip' in augmentations:
            # Horizontal flip
            flipped = np.fliplr(image)
            augmented_images.append(flipped)
        
        if 'rotate' in augmentations:
            # Rotate 90, 180, 270 degrees
            for angle in [90, 180, 270]:
                rotated = np.rot90(image, k=angle//90)
                augmented_images.append(rotated)
        
        if 'brightness' in augmentations:
            # Adjust brightness (multiple levels)
            for factor in [0.7, 0.85, 1.15, 1.3]:
                adjusted = np.clip(image * factor, 0, 1)
                augmented_images.append(adjusted)
        
        if 'contrast' in augmentations:
            # Adjust contrast
            for factor in [0.8, 1.2]:
                mean = image.mean()
                contrasted = np.clip((image - mean) * factor + mean, 0, 1)
                augmented_images.append(contrasted)
        
        if 'noise' in augmentations:
            # Add Gaussian noise
            for std in [0.01, 0.02]:
                noise = np.random.normal(0, std, image.shape).astype(np.float32)
                noisy = np.clip(image + noise, 0, 1)
                augmented_images.append(noisy)
        
        if 'crop' in augmentations:
            # Random crop (center crop with slight offset)
            h, w = image.shape[:2]
            crop_size = int(min(h, w) * 0.9)
            start_h = (h - crop_size) // 2
            start_w = (w - crop_size) // 2
            cropped = image[start_h:start_h+crop_size, start_w:start_w+crop_size]
            # Resize back to original size
            cropped_resized = cv2.resize(cropped, (w, h))
            augmented_images.append(cropped_resized)
        
        # If n_augmentations is specified, randomly sample that many
        if n_augmentations is not None and n_augmentations < len(augmented_images):
            augmented_images = random.sample(augmented_images, n_augmentations)
        elif n_augmentations is not None and n_augmentations > len(augmented_images):
            # If we need more, repeat some augmentations
            while len(augmented_images) < n_augmentations:
                augmented_images.append(random.choice(augmented_images))
        
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

