# src2 Mask Band Separation - Completion Checklist

## ✓ COMPLETED TASKS

### Data Layer (src2/data/dataset.py)
- [x] TIFFHSIDataset class docstring updated to document 256+1 band structure
- [x] TIFFHSIDataset warning message updated for 257 bands (256 data + 1 mask)
- [x] TIFFHSIDataset.__getitem__() modified to:
  - [x] Separate spectral_data = patch_data[:256]
  - [x] Separate mask_data = patch_data[256]
  - [x] Normalize only spectral data
  - [x] Keep mask as-is (not normalized)
  - [x] Return 3-tuple: (patch, mask, label)
- [x] SimpleTIFFDataset updated with same logic
- [x] create_dataloaders() updated:
  - [x] Dataset info dict reflects num_bands=256, has_mask=True
  - [x] Documentation clarifies band separation

### Model Architecture Layer (src2/models/)
- [x] patch_embedding.py:
  - [x] Docstring updated to clarify 256 spectral bands
  - [x] Default in_channels changed 257 → 256
  - [x] n_spectral calculation: 256 // 16 = 16 (exact division, no ceiling)
  - [x] Forward docstring updated for (B, 64, 64, 256) input

- [x] masked_sst.py:
  - [x] Class docstring updated (256 bands, 16 spectral groups)
  - [x] Model init updated: in_channels=256
  - [x] forward_encoder docstring updated
  - [x] forward_reconstruction docstring updated
  - [x] forward_classification docstring updated
  - [x] forward() docstring updated
  - [x] create_model() updated: in_channels=256
  - [x] __main__ test updated: input (2, 64, 64, 256)
  - [x] Output shape now (B, 256, 16, 128) instead of (B, 256, 17, 128)

### Loss Functions Layer (src2/training/losses.py)
- [x] patchify_target() function:
  - [x] Removed ceiling division logic
  - [x] n_spectral = C // patch_c (256 // 16 = 16)
  - [x] Docstring updated: output shape (B, 256, 16, 256)
  - [x] Output patches calculation simplified
  
- [x] reconstruction_loss() docstring updated
  - [x] Input shape (B, 64, 64, 256) documented

### Training Layer (src2/training/pretrain_trainer.py)
- [x] train_epoch() method:
  - [x] Unpacks 3-tuple: patches, mask_unused, _ = batch_data
  - [x] Handles both 2-tuple and 3-tuple formats for safety

- [x] validate() method:
  - [x] Unpacks 3-tuple: patches, mask_unused, _ = batch_data
  - [x] Handles both 2-tuple and 3-tuple formats for safety

- [x] visualize_reconstruction() method:
  - [x] Unpacks batch_data as 3-tuple
  - [x] Conditional handling for both formats
  - [x] Safely extracts patches for visualization

### Documentation
- [x] src2/README.md:
  - [x] Updated comparison table: "64×64×257 (256 spectral + 1 mask)"
  - [x] Updated spectral bands column: "256 (band 257 is mask)"
  - [x] Updated model description: "256 spectral bands"
  - [x] Updated patch_embedding.py reference: "64×64×256 spectral"

- [x] Created src2/DATA_PIPELINE_EXPLANATION.md:
  - [x] Complete TIFF file structure explanation
  - [x] Data loading pipeline (5 steps)
  - [x] Band separation logic
  - [x] Normalization (spectral only)
  - [x] Mask handling (not normalized)
  - [x] Model input/output shapes
  - [x] Training pipeline flow diagram

- [x] Created src2/MASK_BAND_SEPARATION_SUMMARY.md:
  - [x] User question documented
  - [x] Understanding of mask band clarified
  - [x] All changes summarized by file
  - [x] Normalization verification included
  - [x] Data flow diagram
  - [x] Completion checklist

## ✓ CODE VERIFICATION

### Dataset Behavior
- [x] Reads TIFF with rasterio.open()
- [x] Separates bands 1-256 (spectral) and 257 (mask) at load time
- [x] Applies z-score normalization: (data - mean) / std
- [x] Normalization applied ONLY to spectral, NOT to mask
- [x] Returns tuple: (spectral_normalized, mask_raw, file_id)

### Model Behavior
- [x] Accepts input shape: (B, 64, 64, 256) spectral only
- [x] Patch embedding: 4×4 spatial, 16 spectral groups
- [x] Transformer processing: (B, 256, 16, 128) tokens
- [x] Output: (B, 64, 64, 256) reconstructed spectral data
- [x] Mask generation: Internal masking (85% tubes), independent of file mask

### Training Behavior
- [x] train_epoch() unpacks 3-tuple batch format
- [x] validate() unpacks 3-tuple batch format
- [x] visualize_reconstruction() handles 3-tuple format
- [x] Loss computed on normalized spectral data
- [x] Only masked regions contribute to loss

## ✓ NORMALIZATION VERIFICATION

**CONFIRMED**: Data is normalized before training
- [x] Location: src2/data/dataset.py, TIFFHSIDataset.__getitem__()
- [x] Timing: During data loading, before yielding to model
- [x] What: Spectral data (bands 1-256) only
- [x] How: Per-patch z-score: (x - mean) / std
- [x] Mask: Kept raw/separate, NOT normalized
- [x] Result: Model receives normalized spectral data (mean ≈ 0, std ≈ 1)

## ✓ ARCHITECTURE SIMPLIFICATION

Before (with 257 bands):
- Input: (B, 64, 64, 257)
- Spectral groups: ceil(257 / 16) = 17
- Output: (B, 256, 17, 128)
- Issue: Remainder band 257 requires padding

After (with 256 spectral + 1 mask):
- Input: (B, 64, 64, 256) spectral only
- Spectral groups: 256 / 16 = 16 (exact)
- Output: (B, 256, 16, 128)
- Benefit: Clean division, no padding needed, simpler loss computation

## ✓ FILES STATUS

| File | Status | Changes |
|------|--------|---------|
| src2/data/dataset.py | ✓ Complete | Band separation, 3-tuple return, normalization logic |
| src2/models/patch_embedding.py | ✓ Complete | 256 channels, 16 spectral groups |
| src2/models/masked_sst.py | ✓ Complete | 256-band model, (B,256,16,128) output |
| src2/models/masking.py | ✓ Unchanged | No changes needed |
| src2/models/heads.py | ✓ Unchanged | No changes needed |
| src2/models/positional_encoding.py | ✓ Unchanged | No changes needed |
| src2/models/transformer_block.py | ✓ Unchanged | No changes needed |
| src2/training/losses.py | ✓ Complete | Simplified patchify_target for 16 groups |
| src2/training/pretrain_trainer.py | ✓ Complete | 3-tuple unpacking in all methods |
| src2/README.md | ✓ Complete | Updated documentation |
| + NEW: DATA_PIPELINE_EXPLANATION.md | ✓ Created | Complete pipeline documentation |
| + NEW: MASK_BAND_SEPARATION_SUMMARY.md | ✓ Created | Summary of all changes |

## ✓ READY FOR TRAINING

All components updated and consistent:
- [x] Data loading properly separates and normalizes spectral data
- [x] Mask handled separately (not normalized)
- [x] Model expects 256-channel input
- [x] Training loop unpacks 3-tuple format
- [x] Documentation explains new structure
- [x] Architecture simplified (256→16, not 257→17)

**Status**: src2 is ready for training with proper mask band separation and data normalization!
