"""
Data augmentation for HSI patches
"""

import torch
import numpy as np


class RandomFlip:
    """Random horizontal and vertical flips."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, patch):
        """
        Args:
            patch: (H, W, C) numpy array or tensor

        Returns:
            patch: Flipped patch
        """
        if np.random.random() < self.p:
            # Horizontal flip
            patch = np.flip(patch, axis=1).copy()

        if np.random.random() < self.p:
            # Vertical flip
            patch = np.flip(patch, axis=0).copy()

        return patch


class RandomRotate90:
    """Random 90-degree rotations."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, patch):
        """
        Args:
            patch: (H, W, C) numpy array

        Returns:
            patch: Rotated patch
        """
        if np.random.random() < self.p:
            # Random 90-degree rotation
            k = np.random.randint(1, 4)  # 1, 2, or 3 times 90°
            patch = np.rot90(patch, k=k, axes=(0, 1)).copy()

        return patch


class SpectralNoise:
    """Add Gaussian noise to spectral dimension."""

    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std

    def __call__(self, patch):
        """
        Args:
            patch: (H, W, C) numpy array

        Returns:
            patch: Patch with added noise
        """
        noise = np.random.normal(0, self.noise_std, patch.shape)
        patch = patch + noise.astype(patch.dtype)
        return patch


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, patch):
        for t in self.transforms:
            patch = t(patch)
        return patch


# ==============================================================================
# STRONGER CONTRASTIVE AUGMENTATIONS (HSI-Safe)
# ==============================================================================

class RandomCrop:
    """Random crop and resize back to original size."""
    
    def __init__(self, crop_ratio_range=(0.6, 1.0)):
        self.crop_ratio_range = crop_ratio_range
    
    def __call__(self, patch):
        """
        Args:
            patch: (H, W, C) numpy array
        Returns:
            patch: Cropped and resized patch
        """
        H, W, C = patch.shape
        
        # Random crop ratio
        ratio = np.random.uniform(*self.crop_ratio_range)
        crop_h = int(H * ratio)
        crop_w = int(W * ratio)
        
        # Random crop position
        top = np.random.randint(0, H - crop_h + 1)
        left = np.random.randint(0, W - crop_w + 1)
        
        # Crop
        cropped = patch[top:top+crop_h, left:left+crop_w, :]
        
        # Resize back using simple interpolation (nearest neighbor for HSI)
        # Using zoom from scipy or simple repeat
        from scipy.ndimage import zoom
        zoom_factors = (H / crop_h, W / crop_w, 1)
        resized = zoom(cropped, zoom_factors, order=1)  # order=1 for bilinear
        
        # Ensure exact size
        resized = resized[:H, :W, :C]
        
        return resized.astype(np.float32)


class SpectralBandDropout:
    """Randomly drop (zero out) spectral bands - HSI-specific augmentation."""
    
    def __init__(self, drop_ratio=0.1, p=0.5):
        self.drop_ratio = drop_ratio
        self.p = p
    
    def __call__(self, patch):
        """
        Args:
            patch: (H, W, C) numpy array
        Returns:
            patch: With some bands zeroed
        """
        if np.random.random() > self.p:
            return patch
        
        H, W, C = patch.shape
        n_drop = int(C * self.drop_ratio)
        
        if n_drop > 0:
            drop_indices = np.random.choice(C, n_drop, replace=False)
            patch = patch.copy()
            patch[:, :, drop_indices] = 0
        
        return patch


class SpectralShift:
    """Shift spectral bands (simulates wavelength calibration differences)."""
    
    def __init__(self, max_shift=3, p=0.5):
        self.max_shift = max_shift
        self.p = p
    
    def __call__(self, patch):
        if np.random.random() > self.p:
            return patch
        
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        if shift == 0:
            return patch
        
        patch = patch.copy()
        if shift > 0:
            patch[:, :, shift:] = patch[:, :, :-shift]
            patch[:, :, :shift] = 0
        else:
            patch[:, :, :shift] = patch[:, :, -shift:]
            patch[:, :, shift:] = 0
        
        return patch


class IntensityScale:
    """Random intensity scaling (simulates illumination changes)."""
    
    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p
    
    def __call__(self, patch):
        if np.random.random() > self.p:
            return patch
        
        scale = np.random.uniform(*self.scale_range)
        return (patch * scale).astype(np.float32)


class GaussianBlur:
    """Apply Gaussian blur spatially."""
    
    def __init__(self, sigma_range=(0.1, 2.0), p=0.3):
        self.sigma_range = sigma_range
        self.p = p
    
    def __call__(self, patch):
        if np.random.random() > self.p:
            return patch
        
        from scipy.ndimage import gaussian_filter
        sigma = np.random.uniform(*self.sigma_range)
        
        # Apply blur to each band
        blurred = np.zeros_like(patch)
        for c in range(patch.shape[2]):
            blurred[:, :, c] = gaussian_filter(patch[:, :, c], sigma=sigma)
        
        return blurred.astype(np.float32)


# ==============================================================================
# PRE-DEFINED AUGMENTATION PIPELINES
# ==============================================================================

def get_train_transforms():
    """Get standard training augmentations (for MAE mode)."""
    return Compose([
        RandomFlip(p=0.5),
        RandomRotate90(p=0.5),
        SpectralNoise(noise_std=0.02)
    ])


def get_contrastive_transforms():
    """
    Get STRONG augmentations for contrastive learning.
    Creates diverse views while preserving spectral identity.
    """
    return Compose([
        RandomCrop(crop_ratio_range=(0.7, 1.0)),
        RandomFlip(p=0.5),
        RandomRotate90(p=0.5),
        IntensityScale(scale_range=(0.8, 1.2), p=0.5),
        SpectralBandDropout(drop_ratio=0.05, p=0.3),
        SpectralShift(max_shift=2, p=0.3),
        SpectralNoise(noise_std=0.05),
        GaussianBlur(sigma_range=(0.1, 1.5), p=0.3),
    ])


def get_val_transforms():
    """No augmentation for validation."""
    return None
