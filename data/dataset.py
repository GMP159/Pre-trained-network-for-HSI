"""
Dataset for loading HSI patches from TIFF files
TIFF version adapted for 64x64x257 hyperspectral images
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
import os
import glob
import logging
from pathlib import Path
import rasterio
from rasterio.windows import Window
from src2.data.transforms import Compose

logger = logging.getLogger(__name__)


class TIFFHSIDataset(Dataset):
    """
    HSI dataset loading from TIFF files.
    
    Your data structure:
    - TIFF files shape: (64, 64, 257)
    - Data type: float32
    - Bands 1-256: Spectral data
    - Band 257: Mask (separate, not normalized with data)
    - Located in D:\Thesis_new\new_data
    
    Returns:
    - Normalized spectral patch: (64, 64, 256) or (32, 32, 256)
    - Corresponding mask: (64, 64) or (32, 32)
    
    Args:
        tiff_paths: List of paths to TIFF files
        split: 'train' or 'val'
        train_ratio: Ratio of train split (0.8 = 80% train, 20% val)
        normalize_per_patch: If True, normalize spectral bands independently (does NOT normalize mask)
        patch_size: Size of patches to extract (e.g., 32 for 32x32 patches)
    """

    def __init__(
        self,
        tiff_paths: List[str],
        split: str = 'train',
        train_ratio: float = 0.8,
        normalize_per_patch: bool = True,
        patch_size: int = 32,
        seed: int = 42,
        transform: Optional[Compose] = None
    ):
        self.tiff_paths = tiff_paths
        self.split = split
        self.normalize_per_patch = normalize_per_patch
        self.patch_size = patch_size
        self.transform = transform
        
        np.random.seed(seed)

        # Build index of all patches
        self.samples = []
        self.file_info = {}

        logger.info(f"Loading {split} split from {len(tiff_paths)} TIFF files...")
        logger.info(f"Patch size: {patch_size}x{patch_size}")

        for file_idx, tiff_path in enumerate(tiff_paths):
            try:
                # Read TIFF metadata
                with rasterio.open(tiff_path) as src:
                    height, width = src.height, src.width
                    num_bands = src.count
                    
                logger.info(f"  {os.path.basename(tiff_path)}: "
                          f"{height}x{width}x{num_bands}")

                # Verify expected shape
                if num_bands != 257:
                    logger.warning(f"  Warning: Expected 257 bands (256 data + 1 mask) but got {num_bands}")

                if height != 64 or width != 64:
                    logger.warning(f"  Warning: Expected 64x64 but got {height}x{width}")

                self.file_info[file_idx] = {
                    'tiff_path': tiff_path,
                    'height': height,
                    'width': width,
                    'num_bands': num_bands
                }

                # Generate patch coordinates for this file
                n_patches_h = (height - patch_size) // patch_size + 1
                n_patches_w = (width - patch_size) // patch_size + 1
                n_patches = n_patches_h * n_patches_w

                # Split into train/val using indices
                indices = np.arange(n_patches)
                np.random.shuffle(indices)
                
                n_train = int(n_patches * train_ratio)
                if split == 'train':
                    split_indices = indices[:n_train]
                else:
                    split_indices = indices[n_train:]

                # Add to samples list
                for patch_idx in split_indices:
                    # Convert flat patch index to 2D coordinates
                    patch_h = (patch_idx // n_patches_w) * patch_size
                    patch_w = (patch_idx % n_patches_w) * patch_size
                    
                    self.samples.append({
                        'file_idx': file_idx,
                        'patch_h': patch_h,
                        'patch_w': patch_w,
                        'tiff_path': tiff_path,
                        'file_id': f"file_{file_idx}"
                    })

            except Exception as e:
                logger.error(f"Error loading {tiff_path}: {e}")
                continue

        logger.info(f"Total {split} samples: {len(self.samples)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]

        try:
            # Load patch from TIFF
            tiff_path = sample['tiff_path']
            patch_h = sample['patch_h']
            patch_w = sample['patch_w']
            
            with rasterio.open(tiff_path) as src:
                # Read all bands for the patch window
                window = Window(patch_w, patch_h, self.patch_size, self.patch_size)
                
                # Read patch from all bands: (bands, height, width)
                patch_data = src.read(window=window)  # (257, H, W)
                
                # Separate spectral data (bands 1-256) and mask (band 257)
                spectral_data = patch_data[:256]  # (256, H, W) - spectral bands
                mask_data = patch_data[256]        # (H, W) - mask band
                
                # Convert to (H, W, C) format for spectral data
                patch = np.transpose(spectral_data, (1, 2, 0))  # (H, W, 256)
                mask = mask_data  # (H, W)
            
            # Convert to float32
            patch = patch.astype(np.float32)
            mask = mask.astype(np.float32)

            # Normalize ONLY the spectral data (NOT the mask band)
            if self.normalize_per_patch:
                # Per-patch normalization for spectral bands
                patch_mean = patch.mean()
                patch_std = patch.std()
                if patch_std < 1e-6:
                    patch_std = 1.0
                patch = (patch - patch_mean) / patch_std
            else:
                # Per-band normalization for spectral data
                patch_mean = patch.mean(axis=(0, 1), keepdims=True)
                patch_std = patch.std(axis=(0, 1), keepdims=True)
                patch_std = np.where(patch_std < 1e-6, 1.0, patch_std)
                patch = (patch - patch_mean) / patch_std
            
            # Mask is NOT normalized - keep as is
            
            # Apply transforms if any (to spectral data only)
            if self.transform is not None:
                patch = self.transform(patch)

            # Convert to tensor (keep as H, W, C for model input)
            patch = torch.from_numpy(patch).float()
            mask = torch.from_numpy(mask).float()

            # Return patch, mask, and file_id
            label = sample['file_idx']
            return patch, mask, label
                
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            # Return a random valid sample instead
            return self.__getitem__(np.random.randint(0, len(self)))


class SimpleTIFFDataset(Dataset):
    """
    Simplified TIFF dataset that loads entire images without patching.
    For when you want to work with full 64x64x256 images + mask.
    
    Returns:
    - Normalized spectral patch: (64, 64, 256)
    - Mask: (64, 64)
    """

    def __init__(
        self,
        tiff_paths: List[str],
        split: str = 'train',
        train_ratio: float = 0.8,
        normalize_per_patch: bool = True,
        seed: int = 42,
        transform: Optional[Compose] = None
    ):
        self.tiff_paths = tiff_paths
        self.split = split
        self.normalize_per_patch = normalize_per_patch
        self.transform = transform
        
        np.random.seed(seed)

        # Split files into train/val WITHIN each domain (subfolder)
        # Group files by their parent directory (domain)
        domain_files = {}
        for tiff_path in tiff_paths:
            domain = os.path.dirname(tiff_path)  # Get parent directory (domain)
            if domain not in domain_files:
                domain_files[domain] = []
            domain_files[domain].append(tiff_path)
        
        logger.info(f"Found {len(domain_files)} domains (subfolders)")
        for domain, files in domain_files.items():
            logger.info(f"  Domain: {os.path.basename(domain)} - {len(files)} files")
        
        # Split each domain's files into train/val
        self.file_indices = []
        for domain, files in domain_files.items():
            n_files = len(files)
            indices = np.arange(n_files)
            np.random.shuffle(indices)
            
            n_train = int(n_files * train_ratio)
            if split == 'train':
                split_indices = indices[:n_train]
            else:
                split_indices = indices[n_train:]
            
            # Store the actual file paths for this split
            for idx in split_indices:
                self.file_indices.append(files[idx])
        
        logger.info(f"SimpleTIFFDataset - {split} split: {len(self.file_indices)} files ({train_ratio*100:.0f}/{(1-train_ratio)*100:.0f}% ratio per domain)")

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        tiff_path = self.file_indices[idx]

        try:
            # Load entire TIFF image
            with rasterio.open(tiff_path) as src:
                patch_data = src.read()  # (257, 64, 64)
                
                # Separate spectral data and mask
                spectral_data = patch_data[:256]  # (256, 64, 64)
                mask_data = patch_data[256]        # (64, 64)
                
                # Convert to (H, W, C) format
                patch = np.transpose(spectral_data, (1, 2, 0))  # (64, 64, 256)
                mask = mask_data  # (64, 64)
            
            # Convert to float32
            patch = patch.astype(np.float32)
            mask = mask.astype(np.float32)

            # Normalize ONLY spectral data
            if self.normalize_per_patch:
                patch_mean = patch.mean()
                patch_std = patch.std()
                if patch_std < 1e-6:
                    patch_std = 1.0
                patch = (patch - patch_mean) / patch_std
            else:
                patch_mean = patch.mean(axis=(0, 1), keepdims=True)
                patch_std = patch.std(axis=(0, 1), keepdims=True)
                patch_std = np.where(patch_std < 1e-6, 1.0, patch_std)
                patch = (patch - patch_mean) / patch_std
            
            # Apply transforms if any
            if self.transform is not None:
                patch = self.transform(patch)

            # Convert to tensor
            patch = torch.from_numpy(patch).float()
            mask = torch.from_numpy(mask).float()
            label = idx  # Use index as label
            
            return patch, mask, label
                
        except Exception as e:
            logger.error(f"Error loading {tiff_path}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))


def find_tiff_files(data_root: str) -> List[str]:
    """
    Find all TIFF files in directory (recursively searches subdirectories).
    
    Args:
        data_root: Root directory with TIFF files
        
    Returns:
        List of TIFF file paths, sorted
    """
    tiff_paths = []
    
    # Search for TIFF files recursively
    patterns = ['**/*.tif', '**/*.tiff', '**/*.TIF', '**/*.TIFF']
    
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(data_root, pattern), recursive=True))
        tiff_paths.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in tiff_paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    logger.info(f"Found {len(unique_paths)} TIFF files in {data_root} (including subdirectories)")
    
    return unique_paths


def create_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    seed: int = 42,
    patch_size: int = 32,
    simple_mode: bool = False,
    train_transforms=None,
    val_transforms=None,
    distributed: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """
    Create train and validation dataloaders for TIFF files.
    
    Note: Returns (patch, mask, label) tuples.
    - patch: Normalized spectral data (H, W, 256)
    - mask: Unnormalized mask (H, W)
    - label: File index

    Args:
        data_root: Root directory containing TIFF files
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_ratio: Train/val split ratio
        seed: Random seed for reproducibility
        patch_size: Size of patches to extract (32 for 32x32 patches)
        simple_mode: If True, use SimpleTIFFDataset (full images), else use patched dataset
        train_transforms: Transforms for training data
        val_transforms: Transforms for validation data
        distributed: If True, use DistributedSampler for DDP training

    Returns:
        train_loader, val_loader, dataset_info
    """
    # Find TIFF files
    tiff_paths = find_tiff_files(data_root)

    if len(tiff_paths) == 0:
        raise ValueError(f"No TIFF files found in: {data_root}")

    logger.info(f"Found {len(tiff_paths)} TIFF files")

    # Create datasets
    if simple_mode:
        # Use full images
        train_dataset = SimpleTIFFDataset(
            tiff_paths=tiff_paths,
            split='train',
            train_ratio=train_ratio,
            normalize_per_patch=True,
            seed=seed,
            transform=train_transforms
        )

        val_dataset = SimpleTIFFDataset(
            tiff_paths=tiff_paths,
            split='val',
            train_ratio=train_ratio,
            normalize_per_patch=True,
            seed=seed,
            transform=val_transforms
        )
    else:
        # Use patched version
        train_dataset = TIFFHSIDataset(
            tiff_paths=tiff_paths,
            split='train',
            train_ratio=train_ratio,
            normalize_per_patch=True,
            patch_size=patch_size,
            seed=seed,
            transform=train_transforms
        )

        val_dataset = TIFFHSIDataset(
            tiff_paths=tiff_paths,
            split='val',
            train_ratio=train_ratio,
            normalize_per_patch=True,
            patch_size=patch_size,
            seed=seed,
            transform=val_transforms
        )

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using DistributedSampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
        persistent_workers=False if num_workers == 0 else True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=False if num_workers == 0 else True
    )

    dataset_info = {
        'num_files': len(tiff_paths),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'num_bands': 256,  # Spectral bands (mask is separate)
        'img_size': 64,
        'has_mask': True
    }

    return train_loader, val_loader, dataset_info


if __name__ == "__main__":
    # Test dataset
    logging.basicConfig(level=logging.INFO)
    
    data_root = r"D:\Thesis_new\new_data"
    
    print("Testing dataset creation...")
    try:
        train_loader, val_loader, info = create_dataloaders(
            data_root=data_root,
            batch_size=4,
            num_workers=0,
            patch_size=32,
            simple_mode=False  # Use patched mode
        )
        
        print(f"\nDataset info: {info}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test loading a batch
        print("\nTesting batch loading...")
        patches, labels = next(iter(train_loader))
        print(f"Patches shape: {patches.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Patches dtype: {patches.dtype}")
        print(f"Patches range: [{patches.min():.3f}, {patches.max():.3f}]")
        print(f"Labels: {labels}")
        print("\n  Dataset test passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
