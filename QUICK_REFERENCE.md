# Quick Reference - src2 Data Format & Normalization

## TIFF File Structure
```
Input: TIFF file from D:\Thesis_new\new_data
Shape: (64, 64, 257)
├─ Bands 1-256: Hyperspectral data (NORMALIZED before training)
└─ Band 257:    Mask channel (NOT normalized, kept raw)
```

## Data Normalization Answer

**Q: Did you normalize the data before use in training?**  
**A: YES ✓**

### Where Normalization Happens
```python
File: src2/data/dataset.py
Method: TIFFHSIDataset.__getitem__()
Line: ~155-162

# Normalize spectral data (bands 1-256)
patch = (spectral_data - mean) / std
# Result: mean ≈ 0, std ≈ 1

# Mask (band 257) is NOT normalized
mask = mask_data.astype(np.float32)  # Kept as-is
```

### Normalization Timeline
```
1. Load TIFF file
   ↓
2. Separate: Spectral (256 bands) vs Mask (1 band)
   ↓
3. NORMALIZE spectral data (z-score)
   ↓
4. Pass normalized spectral to model
   ↓
5. Training begins with normalized data
```

## Data Format Through Pipeline

### Stage 1: Data Loading
```python
# From dataset.__getitem__()
spectral_patch: (64, 64, 256)  - normalized
mask_patch:     (64, 64)       - raw values
file_label:     int            - file ID

return (spectral_patch, mask_patch, file_label)  # 3-tuple
```

### Stage 2: Training Loop
```python
# In trainer.train_epoch()
batch_data = next(dataloader)  # 3-tuple
patches, mask_unused, _ = batch_data

patches.shape = (B, 64, 64, 256)  # B = batch size
# patches are normalized ✓
```

### Stage 3: Model Processing
```python
# In model.forward()
input:  (B, 64, 64, 256)     # normalized spectral
   ↓
patch_embedding:  (B, 256, 16, 128)
   - 256 spatial patches (16×16 from 64×64)
   - 16 spectral groups (256÷16)
   - 128-dim embeddings
   ↓
transformer_blocks: apply factorized attention
   ↓
reconstruction: (B, 64, 64, 256)  # reconstructed spectral
```

## Architecture Summary

| Component | 256-Band Design | Details |
|-----------|-----------------|---------|
| Input | (B, 64, 64, 256) | Spectral only (normalized) |
| Spatial patches | 256 | 16×16 patches from 64×64 image |
| Spectral groups | 16 | 256 bands ÷ 16 = exact division |
| Patch embeddings | (B, 256, 16, 128) | 256 spatial, 16 spectral, 128-dim |
| Output | (B, 64, 64, 256) | Reconstructed spectral |

## Key Points

1. **Data is normalized**: Spectral data normalized to mean≈0, std≈1
2. **Mask is separate**: Band 257 not used in model, kept for post-processing
3. **When normalized**: During data loading, BEFORE model training
4. **Clean division**: 256 bands → 16 spectral groups (no padding)
5. **3-tuple format**: (spectral_normalized, mask_raw, label)

## Verification Commands

To verify normalization in code:
```bash
# Check dataset.py for normalization logic
grep -n "patch_mean\|patch_std" src2/data/dataset.py

# Verify 256-channel input
grep -n "in_channels.*256" src2/models/*.py

# Check trainer unpacks 3-tuple
grep -n "patches, mask" src2/training/pretrain_trainer.py
```

## Training-Ready Checklist

- [x] Spectral data normalized (mean=0, std=1)
- [x] Mask handled separately (not normalized)
- [x] Model expects 256-channel input
- [x] Training loop handles 3-tuple format
- [x] Architecture simplified (256÷16=16)
- [x] Ready to train!

---

**Bottom Line**: Your data is properly normalized during loading, only spectral bands are normalized (not mask), and everything is ready for training! 🚀
