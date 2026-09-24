# Mask Band Separation - Summary of Changes

## User Question
"the band 257 which is the last band is the mask actually, did you normalize the data before u use it in the training"

## Understanding
- **TIFF Structure**: Each file has 257 bands total
  - Bands 1-256: Spectral/hyperspectral data
  - Band 257: Mask channel (not spectral data)
- **Normalization**: Only spectral data should be normalized, NOT the mask

## Changes Made to src2

### 1. **Data Loading (src2/data/dataset.py)**

#### Before
- Treated all 257 bands as spectral data
- Normalized entire 257-band input

#### After
- Separates band 257 immediately upon TIFF read:
  ```python
  spectral_data = patch_data[:256]     # Bands 1-256 (spectral)
  mask_data = patch_data[256]          # Band 257 (mask)
  ```
- Normalizes ONLY spectral data (mean=0, std=1):
  ```python
  patch = (spectral_data - patch_mean) / patch_std  # Spectral normalized
  mask = mask_data.astype(np.float32)  # Mask NOT normalized
  ```
- Returns 3-tuple: `(spectral_patch, mask_patch, file_label)`

### 2. **Model Architecture (src2/models/)**

#### Updated Files
- **patch_embedding.py**: Now expects 256 input channels (not 257)
  - Input shape: (B, 64, 64, 256) instead of (B, 64, 64, 257)
  - Spectral groups: 256 / 16 = 16 exactly (was 257 / 16 = 17 with ceiling)
  - Cleaner architecture with exact division

- **masked_sst.py**: Model updated to work with 256-band input
  - Output tokens: (B, 256, 16, 128) meaning:
    - 256 spatial patches (16×16 from 64×64)
    - 16 spectral groups (256 bands / 16)
    - 128-dimensional embeddings

### 3. **Loss Functions (src2/training/losses.py)**

#### Updated
- **patchify_target()**: Simplified from ceiling division to exact division
  - Changed: `n_spectral = ceil(C / patch_c)` → `n_spectral = C // patch_c`
  - Now: 256 / 16 = 16 spectral patch groups (no padding)
  - Reduced complexity and edge cases

- **reconstruction_loss()**: Updated documentation for (B, 64, 64, 256) input

### 4. **Training Loop (src2/training/pretrain_trainer.py)**

#### Updated Methods
- **train_epoch()**: Unpacks 3-tuple from dataloader
  ```python
  patches, mask_unused, _ = batch_data  # New format
  # Uses only 'patches' for forward pass
  ```

- **validate()**: Unpacks 3-tuple from dataloader (same pattern)

- **visualize_reconstruction()**: Handles 3-tuple unpacking safely
  ```python
  if len(batch_data) == 3:
      patches, mask_batch, _ = batch_data
  else:
      patches = batch_data[0]
  ```

## Verification: Data Normalization

**YES, data IS normalized before training:**

1. **Location**: `src2/data/dataset.py` in `TIFFHSIDataset.__getitem__`
2. **When**: During data loading, BEFORE yielding to model
3. **What**: Only spectral data (bands 1-256)
4. **How**: Per-patch z-score normalization: `(x - mean) / std`
5. **Mask**: Kept completely separate, NOT normalized

### Normalization Code
```python
# In TIFFHSIDataset.__getitem__ (lines 155-162 in dataset.py)
if self.normalize_per_patch:
    # Per-patch normalization
    patch_mean = patch.mean()
    patch_std = patch.std()
    if patch_std < 1e-6:
        patch_std = 1.0
    patch = (patch - patch_mean) / patch_std

# Mask is NOT normalized:
# mask = mask_data.astype(np.float32)  # Kept as-is
```

## Data Flow Summary

```
TIFF File (64, 64, 257)
    ↓
[Load with rasterio]
    ↓
Separate:
  - Spectral: (64, 64, 256)  ← Normalize here
  - Mask:     (64, 64)       ← NOT normalized
    ↓
[Model receives ONLY spectral]
    ↓
Input to Model: (B, 64, 64, 256) normalized spectral data
    ↓
Model Processing:
  1. Patch Embedding → (B, 256, 16, 128) tokens
  2. Transformer Blocks → Factorized attention
  3. Reconstruction Head → (B, 64, 64, 256)
    ↓
Output: Reconstructed spectral data
```

## Files Modified (7 total)

1. ✓ src2/data/dataset.py (TIFFHSIDataset, SimpleTIFFDataset, create_dataloaders)
2. ✓ src2/models/patch_embedding.py (256 channels, 16 spectral groups)
3. ✓ src2/models/masked_sst.py (model architecture, dimensions)
4. ✓ src2/training/losses.py (patchify_target simplified)
5. ✓ src2/training/pretrain_trainer.py (train_epoch, validate, visualize_reconstruction)
6. ✓ src2/README.md (updated documentation)
7. + Created: DATA_PIPELINE_EXPLANATION.md (this explanation)

## Key Takeaway

- **Spectral Data (256 bands)**: Normalized per-patch (mean=0, std=1)
- **Mask Data (band 257)**: Kept as-is, not normalized
- **Architecture**: Simplified to 256 channels → 16 spectral groups (exact division)
- **Model Input**: Only normalized spectral data, mask kept separate
- **Training**: Proceeds with normalized spectral data for stable convergence

Everything is ready to train with proper data normalization!
