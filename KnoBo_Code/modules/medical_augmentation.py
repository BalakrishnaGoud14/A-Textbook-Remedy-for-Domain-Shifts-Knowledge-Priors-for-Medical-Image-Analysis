"""
Medical Image Augmentation with CLAHE and Histogram Equalization
Applies medical-specific preprocessing and data augmentation
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Optional, Tuple
import torchvision.transforms as transforms


class CLAHETransform:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE)
    Particularly effective for medical images to enhance local contrast
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Args:
            clip_limit: Threshold for contrast limiting (1.0-4.0 typical)
            tile_grid_size: Size of grid for histogram equalization
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        
    def __call__(self, img):
        """
        Apply CLAHE to image
        
        Args:
            img: PIL Image or numpy array
        
        Returns:
            CLAHE-processed PIL Image
        """
        # Convert PIL to numpy if needed
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            was_pil = True
        else:
            img_array = img
            was_pil = False
        
        # Handle different image formats
        if len(img_array.shape) == 2:
            # Grayscale
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, 
                                   tileGridSize=self.tile_grid_size)
            enhanced = clahe.apply(img_array.astype(np.uint8))
        elif len(img_array.shape) == 3:
            # Color image - apply to luminance channel
            if img_array.shape[2] == 3:
                # Convert RGB to LAB
                lab = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2LAB)
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                       tileGridSize=self.tile_grid_size)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                # Convert back to RGB
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                enhanced = img_array
        else:
            enhanced = img_array
        
        # Convert back to PIL if input was PIL
        if was_pil:
            return Image.fromarray(enhanced)
        return enhanced


class HistogramEqualization:
    """
    Standard histogram equalization for global contrast enhancement
    """
    def __init__(self, adaptive=False):
        """
        Args:
            adaptive: If True, use adaptive histogram equalization (CLAHE)
        """
        self.adaptive = adaptive
        
    def __call__(self, img):
        """
        Apply histogram equalization
        
        Args:
            img: PIL Image or numpy array
            
        Returns:
            Equalized PIL Image
        """
        # Convert PIL to numpy if needed
        if isinstance(img, Image.Image):
            img_array = np.array(img)
            was_pil = True
        else:
            img_array = img
            was_pil = False
        
        # Handle different image formats
        if len(img_array.shape) == 2:
            # Grayscale
            equalized = cv2.equalizeHist(img_array.astype(np.uint8))
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # Color image - convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        else:
            equalized = img_array
        
        # Convert back to PIL if input was PIL
        if was_pil:
            return Image.fromarray(equalized)
        return equalized


class MedicalAugmentation:
    """
    Complete medical image augmentation pipeline
    Combines CLAHE, histogram equalization, and standard augmentations
    """
    def __init__(self, 
                 use_clahe=True,
                 use_hist_eq=False,
                 clip_limit=2.0,
                 tile_grid_size=(8, 8),
                 rotation_degrees=10,
                 horizontal_flip_prob=0.5,
                 vertical_flip_prob=0.0,
                 brightness=0.1,
                 contrast=0.1,
                 training=True):
        """
        Args:
            use_clahe: Apply CLAHE preprocessing
            use_hist_eq: Apply histogram equalization (alternative to CLAHE)
            clip_limit: CLAHE clip limit
            tile_grid_size: CLAHE grid size
            rotation_degrees: Random rotation range
            horizontal_flip_prob: Probability of horizontal flip
            vertical_flip_prob: Probability of vertical flip
            brightness: Brightness jitter factor
            contrast: Contrast jitter factor
            training: If True, apply augmentations; if False, only preprocessing
        """
        self.use_clahe = use_clahe
        self.use_hist_eq = use_hist_eq
        self.training = training
        
        # Preprocessing transforms
        preprocess_list = []
        
        if use_clahe:
            preprocess_list.append(CLAHETransform(clip_limit, tile_grid_size))
        elif use_hist_eq:
            preprocess_list.append(HistogramEqualization())
        
        self.preprocess = transforms.Compose(preprocess_list) if preprocess_list else None
        
        # Training augmentations
        if training:
            aug_list = []
            
            if rotation_degrees > 0:
                aug_list.append(transforms.RandomRotation(rotation_degrees))
            
            if horizontal_flip_prob > 0:
                aug_list.append(transforms.RandomHorizontalFlip(horizontal_flip_prob))
            
            if vertical_flip_prob > 0:
                aug_list.append(transforms.RandomVerticalFlip(vertical_flip_prob))
            
            if brightness > 0 or contrast > 0:
                aug_list.append(transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast
                ))
            
            self.augmentation = transforms.Compose(aug_list) if aug_list else None
        else:
            self.augmentation = None
    
    def __call__(self, img):
        """
        Apply augmentation pipeline
        
        Args:
            img: PIL Image
            
        Returns:
            Augmented PIL Image
        """
        # Apply preprocessing (CLAHE or histogram equalization)
        if self.preprocess is not None:
            img = self.preprocess(img)
        
        # Apply training augmentations if in training mode
        if self.training and self.augmentation is not None:
            img = self.augmentation(img)
        
        return img


def get_medical_augmentation(modality='xray', 
                             training=True,
                             augmentation_strength='medium'):
    """
    Get recommended augmentation pipeline for medical imaging modality
    
    Args:
        modality: 'xray' or 'skin'
        training: Training mode (with augmentation) or inference mode
        augmentation_strength: 'none', 'light', 'medium', 'strong'
    
    Returns:
        MedicalAugmentation instance
    """
    if augmentation_strength == 'none':
        return MedicalAugmentation(
            use_clahe=False,
            use_hist_eq=False,
            training=False
        )
    
    # X-ray specific augmentation
    if modality == 'xray':
        if augmentation_strength == 'light':
            return MedicalAugmentation(
                use_clahe=True,
                use_hist_eq=False,
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                rotation_degrees=5,
                horizontal_flip_prob=0.3,
                vertical_flip_prob=0.0,
                brightness=0.05,
                contrast=0.05,
                training=training
            )
        elif augmentation_strength == 'medium':
            return MedicalAugmentation(
                use_clahe=True,
                use_hist_eq=False,
                clip_limit=3.0,
                tile_grid_size=(8, 8),
                rotation_degrees=10,
                horizontal_flip_prob=0.5,
                vertical_flip_prob=0.0,
                brightness=0.1,
                contrast=0.1,
                training=training
            )
        else:  # strong
            return MedicalAugmentation(
                use_clahe=True,
                use_hist_eq=False,
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                rotation_degrees=15,
                horizontal_flip_prob=0.5,
                vertical_flip_prob=0.2,
                brightness=0.2,
                contrast=0.2,
                training=training
            )
    
    # Skin lesion specific augmentation
    elif modality == 'skin':
        if augmentation_strength == 'light':
            return MedicalAugmentation(
                use_clahe=True,
                use_hist_eq=False,
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                rotation_degrees=10,
                horizontal_flip_prob=0.5,
                vertical_flip_prob=0.5,
                brightness=0.1,
                contrast=0.1,
                training=training
            )
        elif augmentation_strength == 'medium':
            return MedicalAugmentation(
                use_clahe=True,
                use_hist_eq=False,
                clip_limit=3.0,
                tile_grid_size=(8, 8),
                rotation_degrees=20,
                horizontal_flip_prob=0.5,
                vertical_flip_prob=0.5,
                brightness=0.15,
                contrast=0.15,
                training=training
            )
        else:  # strong
            return MedicalAugmentation(
                use_clahe=True,
                use_hist_eq=False,
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                rotation_degrees=30,
                horizontal_flip_prob=0.5,
                vertical_flip_prob=0.5,
                brightness=0.2,
                contrast=0.2,
                training=training
            )
    
    # Default: medium augmentation
    return MedicalAugmentation(training=training)


if __name__ == "__main__":
    print("Medical Image Augmentation Module")
    print("=" * 80)
    print("\nFeatures:")
    print("  [CHECK] CLAHE (Contrast Limited Adaptive Histogram Equalization)")
    print("  [CHECK] Histogram Equalization")
    print("  [CHECK] Random Rotation")
    print("  [CHECK] Random Horizontal/Vertical Flip")
    print("  [CHECK] Color Jitter (Brightness/Contrast)")
    print("\nUsage:")
    print("""
    from modules.medical_augmentation import get_medical_augmentation
    
    # X-ray augmentation
    xray_aug = get_medical_augmentation(modality='xray', 
                                        training=True,
                                        augmentation_strength='medium')
    
    # Skin lesion augmentation
    skin_aug = get_medical_augmentation(modality='skin',
                                        training=True,
                                        augmentation_strength='medium')
    
    # Apply to image
    from PIL import Image
    img = Image.open('medical_image.jpg')
    augmented_img = xray_aug(img)
    """)
    print("\nAugmentation Strengths:")
    print("  - 'none': No augmentation (baseline)")
    print("  - 'light': Minimal augmentation (clip_limit=2.0, rotation=5-10 degrees)")
    print("  - 'medium': Moderate augmentation (clip_limit=3.0, rotation=10-20 degrees)")
    print("  - 'strong': Aggressive augmentation (clip_limit=4.0, rotation=15-30 degrees)")
