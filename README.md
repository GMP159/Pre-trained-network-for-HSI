README.md
# src2 - Masked Spatial-Spectral Transformer for TIFF HSI

## Overview

`src2` is an adapted version of `src1` designed specifically for **TIFF hyperspectral images** instead of zarr files.

### Key Differences from src1

| Aspect | src1 | src2 |
|--------|------|------|
| **Input Format** | Zarr files | TIFF files (.tif, .tiff) |
| **Image Size** | 32×32×256 | **64×64×257** (256 spectral + 1 mask) |
| **Spectral Bands** | 256 | **256** (band 257 is mask) |
| **Patch Embedding** | 8×8 spatial patches | **16×16 spatial patches** (for 64×64 images) |
| **Data Location** | Various zarr sources | D:\Thesis_new\new_data |

### Directory Structure

```
src2/
├── data/
│   ├── dataset.py          # TIFF dataset loaders
│   ├── transforms.py       # Data augmentation
│   └── __init__.py
├── models/
│   ├── masked_sst.py       # Main model (256 spectral bands)
│   ├── patch_embedding.py  # Adapted for 64×64×256 spectral
│   ├── positional_encoding.py
│   ├── transformer_block.py
│   ├── masking.py
│   ├── heads.py
│   └── __init__.py
├── training/
│   ├── pretrain_trainer.py # Pre-training loop
│   ├── losses.py          # Loss functions
│   └── __init__.py
├── evaluation/
│   └── __init__.py
├── scripts/
│   └── __init__.py
└── __init__.py
```

## Installation

### Required Packages

```bash
pip install torch torchvision
pip install rasterio  # For reading TIFF files
pip install tqdm
pip install matplotlib numpy
pip install wandb  # Optional, for experiment tracking
```

## Usage

### 1. Loading TIFF Data

```python
from src2.data.dataset import create_dataloaders

# Create dataloaders
train_loader, val_loader, dataset_info = create_dataloaders(
    data_root=r"D:\Thesis_new\new_data",
    batch_size=32,
    num_workers=0,
    patch_size=32,  # Extract 32×32 patches from 64×64 images
    simple_mode=False  # Use patched mode
)

print(f"Train samples: {dataset_info['train_samples']}")
print(f"Val samples: {dataset_info['val_samples']}")
print(f"Image size: {dataset_info['img_size']}×{dataset_info['img_size']}")
print(f"Num bands: {dataset_info['num_bands']}")
```

#### Dataset Modes

**Patched Mode** (`simple_mode=False`):
- Extracts 32×32 patches from 64×64 images
- Useful for more training samples
- Default patch_size=32

**Simple Mode** (`simple_mode=True`):
- Uses full 64×64×256 images (spectral data only)
- Mask band (257) handled separately
- Fewer training samples but full spatial context

### 2. Creating a Model

```python
from src2.models.masked_sst import create_model

# Create model for pre-training
model = create_model(num_classes=1)  # 1 class since using single TIFF file

# For multiple TIFF files (as domains):
model = create_model(num_classes=10)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 3. Pre-training

```python
from src2.training.pretrain_trainer import PretrainTrainer

trainer = PretrainTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    lr=1e-3,
    epochs=200,
    save_dir='outputs/checkpoints/pretrain_tiff',
    warmup_epochs=10,
    use_wandb=False  # Set to True if using Weights & Biases
)

trainer.train()
```

### 4. Key Model Architecture Changes for TIFF

#### Input Dimensions
- **src1**: (B, 32, 32, 256)
- **src2**: (B, 64, 64, 256) - 256 spectral bands (band 257 is mask, handled separately)

#### Patch Embedding Output
- **src1**: (B, 64, 16, 128) - 8×8 spatial × 16 spectral
- **src2**: (B, 256, 16, 128) - 16×16 spatial × 16 spectral

**Note**: 256 spectral bands ÷ 16 = 16 groups exactly (clean division)

#### Masking Strategy
- **src1**: Random masking (75%)
- **src2**: **Tube masking (85%)** - masks entire spatial tubes for coherence

## Implementation Details

### 1. TIFF Data Loading (`src2/data/dataset.py`)

```python
class TIFFHSIDataset(Dataset):
    """
    Loads TIFF files with shape (64, 64, 257).
    Supports both full image and patched extraction.
    """
```

Features:
- Uses `rasterio` for efficient TIFF reading
- Per-patch normalization (better for pre-training)
- Automatic train/val split
- Supports data augmentation

### 2. Patch Embedding for 257 Bands

```python
class PatchEmbedding3D(nn.Module):
    # For 257 bands with 16-band groups:
    n_spectral = (257 + 16 - 1) // 16 = 17
```

Handles ceiling division for non-divisible band counts.

### 3. Loss Functions

```python
def patchify_target(x, patch_h=4, patch_w=4, patch_c=16):
    """Patchifies 64×64×257 to match model output (256, 17, 256)"""

def masked_l1_loss(pred, target, mask):
    """L1 loss on masked tokens only"""

def reconstruction_loss(model_output, original_patches, mask):
    """Complete reconstruction loss for pre-training"""
```

## Differences in Configuration

### Model Parameters
```python
# src1 (zarr, 32×32×256)
MaskedSST(
    img_size=32,
    in_channels=256,
    patch_h=4, patch_w=4, patch_c=16,
)

# src2 (TIFF, 64×64×257)
MaskedSST(
    img_size=64,
    in_channels=257,
    patch_h=4, patch_w=4, patch_c=16,
)
```

### Number of Patches
```
src1: (64 spatial patches) × (16 spectral groups) = 1,024 tokens
src2: (256 spatial patches) × (17 spectral groups) = 4,352 tokens
```

Computational complexity:
- **src1**: O(1024²) ≈ 1M ops
- **src2**: O(256² + 17²) ≈ 65K ops (factorized)

### Masking Strategy
```python
# src1
TubeMasking(mask_ratio=0.90)  # 90% of spatial tubes masked

# src2
TubeMasking(mask_ratio=0.85)  # 85% of spatial tubes masked
```

## Testing

### Test Dataset Loading
```python
python -m src2.data.dataset
```

### Test Model
```python
python -m src2.models.masked_sst
```

Expected output:
```
Reconstruction output: torch.Size([2, 256, 17, 256])
Classification output: torch.Size([2, 10])
Total parameters: 42,789,120
```

### Test Losses
```python
python -m src2.training.losses
```

## Important Notes

1. **TIFF File Requirements**:
   - Shape: (64, 64, 257)
   - Data type: float32
   - Location: D:\Thesis_new\new_data

2. **Memory Considerations**:
   - Full 64×64×257 images are ~67 MB each
   - Batch size of 4 ≈ 268 MB GPU memory
   - Use smaller batch sizes if memory constrained

3. **Rasterio Installation**:
   If you encounter issues with rasterio:
   ```bash
   pip install rasterio  # Windows pre-compiled wheels available
   ```

4. **No Changes to src1**:
   src1 remains unchanged for zarr-based processing

## Troubleshooting

### "No TIFF files found"
- Check path: D:\Thesis_new\new_data
- Verify file extensions (.tif, .tiff case-insensitive)

### "Input channels 257 doesn't match expected 256"
- Your TIFF files have 257 bands (expected for src2)
- Ensure using src2, not src1

### Memory errors
- Reduce batch_size in create_dataloaders()
- Enable use_amp=True in trainer

### Rasterio ImportError
```bash
pip install --upgrade rasterio
```

## Next Steps

1. Verify TIFF files exist in D:\Thesis_new\new_data
2. Test dataset loading with the example code
3. Start pre-training with small epochs (test_epochs=2)
4. Monitor training curves in outputs/checkpoints/pretrain_tiff/

## References

See src1/ for the original zarr-based implementation.
