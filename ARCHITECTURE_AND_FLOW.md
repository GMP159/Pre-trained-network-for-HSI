# Visual Architecture Diagram

## Data Pipeline Flow

```
╔═══════════════════════════════════════════════════════════════════╗
║                      TIFF FILE LOADING                            ║
║                    (D:\Thesis_new\new_data)                       ║
╚═══════════════════════════════════════════════════════════════════╝
                              │
                              ▼
                    ┌─────────────────────┐
                    │   TIFF File Shape   │
                    │     (64, 64, 257)   │
                    └─────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌──────────────────┐      ┌──────────────────┐
        │  Bands 1-256     │      │   Band 257       │
        │  SPECTRAL DATA   │      │   MASK CHANNEL   │
        │  (64, 64, 256)   │      │   (64, 64)       │
        └──────────────────┘      └──────────────────┘
                │                           │
                ▼                           ▼
        ┌──────────────────┐      ┌──────────────────┐
        │  NORMALIZE ✓     │      │  KEEP RAW ✗      │
        │  (z-score)       │      │  (no change)     │
        │                  │      │                  │
        │ mean = 0         │      │ values preserved │
        │ std = 1          │      │                  │
        └──────────────────┘      └──────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                              ▼
                ┌─────────────────────────────────┐
                │   DATALOADER 3-TUPLE FORMAT     │
                │   (spectral, mask, label)       │
                │   ✓ normalized  ✗ raw  │ int    │
                └─────────────────────────────────┘
                              │
                              ▼
                ┌─────────────────────────────────┐
                │      MODEL INPUT LAYER          │
                │  Accepts ONLY spectral data     │
                │      (B, 64, 64, 256)           │
                │      ✓ NORMALIZED               │
                └─────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Patch Embedding  │      │  Positional      │
        │                  │      │  Encoding        │
        │ (B,64,64,256)→   │      │                  │
        │ (B,256,16,128)   │      │ Add learnable    │
        │                  │      │ position embeds  │
        │ Spatial: 16×16   │      │                  │
        │ Spectral: 16     │      │                  │
        │ Embed: 128-dim   │      │                  │
        └──────────────────┘      └──────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                              ▼
                ┌─────────────────────────────────┐
                │  Transformer Blocks with        │
                │  Factorized Spatial-Spectral    │
                │  Attention                      │
                │                                 │
                │  Input:  (B, 256, 16, 128)     │
                │  Output: (B, 256, 16, 128)     │
                └─────────────────────────────────┘
                              │
                              ▼
                ┌─────────────────────────────────┐
                │    Reconstruction Head          │
                │                                 │
                │  Input:  (B, 256, 16, 128)     │
                │  Output: (B, 64, 64, 256)      │
                └─────────────────────────────────┘
                              │
                              ▼
                ┌─────────────────────────────────┐
                │   RECONSTRUCTION LOSS           │
                │   Compared with normalized      │
                │   input spectral data           │
                │                                 │
                │   L1 loss on masked regions     │
                └─────────────────────────────────┘
```

## Normalization Verification

```
┌─────────────────────────────────────────────────────┐
│           NORMALIZATION CHECK POINTS                 │
└─────────────────────────────────────────────────────┘

1. DATA LOADING PHASE
   ├─ Location: src2/data/dataset.py
   ├─ Method:  TIFFHSIDataset.__getitem__()
   ├─ When:    During dataset iteration
   ├─ What:    Bands 1-256 (spectral only)
   ├─ How:     z-score: (x - mean) / std
   └─ Result:  ✓ NORMALIZED

2. BAND 257 (MASK) HANDLING
   ├─ Status:  Separated at load time
   ├─ Process: mask = patch_data[256]
   ├─ Normalize: NO (kept raw)
   └─ Result:  ✗ NOT NORMALIZED

3. MODEL INPUT
   ├─ Data:    (B, 64, 64, 256)
   ├─ Property: mean ≈ 0, std ≈ 1
   ├─ Format:  Normalized spectral ✓
   └─ Mask:    Separate (not used directly)

4. TRAINING
   ├─ Input:   Normalized spectral data
   ├─ Process: Forward pass → reconstruction
   ├─ Loss:    L1 on normalized spectral
   └─ Status:  Training on normalized data ✓
```

## Model Architecture Dimensions

```
┌────────────────────────────────────────────────────────┐
│          SPATIAL-SPECTRAL TRANSFORMER FLOW             │
└────────────────────────────────────────────────────────┘

Input Layer
│
├─ Tensor Shape:      (B, 64, 64, 256)
├─ B:                 batch size
├─ 64×64:             spatial dimensions
├─ 256:               spectral bands (normalized)
└─ Status:            ✓ Ready for embedding


Patch Embedding Layer
│
├─ Spatial Patches
│  ├─ 4×4 spatial patches from 64×64 image
│  ├─ Result: 16×16 = 256 spatial patches
│  └─ Tensor: (B, 256, D)
│
├─ Spectral Groups
│  ├─ 256 bands ÷ 16 = 16 spectral groups
│  ├─ Result: Clean division (no padding needed)
│  └─ Tensor: (B, 256, 16, D)
│
└─ Embedding Dimension
   ├─ 4×4 spatial × 16 spectral = 256 patch features
   ├─ Projected to 128-dim embeddings
   └─ Final: (B, 256, 16, 128)


Transformer Blocks (N layers)
│
├─ Factorized Attention
│  ├─ Spatial attention: across 256 patches
│  ├─ Spectral attention: across 16 groups
│  └─ Independent processing for efficiency
│
└─ Output: (B, 256, 16, 128)


Reconstruction Head
│
├─ Input:  (B, 256, 16, 128)
├─ Process: Dense layers to recover patch features
├─ Reshape: Back to (B, 64, 64, 256)
└─ Output: (B, 64, 64, 256) reconstructed spectral


Loss Computation
│
├─ Target:    (B, 64, 64, 256) - normalized input
├─ Prediction: (B, 64, 64, 256) - model output
├─ Mask:      85% of spatial regions (internal)
└─ Loss:      L1 on masked regions only
```

## File Organization

```
src2/
│
├─ data/
│  ├─ dataset.py ✓ UPDATED
│  │  ├─ TIFFHSIDataset
│  │  │  ├─ Load TIFF
│  │  │  ├─ Separate bands 1-256 (spectral) and 257 (mask)
│  │  │  ├─ Normalize spectral only
│  │  │  └─ Return (spectral, mask, label) ✓ 3-tuple
│  │  │
│  │  ├─ SimpleTIFFDataset (same updates)
│  │  │
│  │  └─ create_dataloaders()
│  │     └─ num_bands=256, has_mask=True
│  │
│  └─ transforms.py ✓ UNCHANGED
│
├─ models/
│  ├─ patch_embedding.py ✓ UPDATED
│  │  └─ in_channels: 256 (not 257)
│  │  └─ n_spectral: 256 // 16 = 16
│  │
│  ├─ masked_sst.py ✓ UPDATED
│  │  ├─ Input: (B, 64, 64, 256)
│  │  ├─ Output: (B, 256, 16, 128) ✓ not (B, 256, 17, 128)
│  │  └─ in_channels: 256
│  │
│  ├─ positional_encoding.py ✓ UNCHANGED
│  ├─ transformer_block.py ✓ UNCHANGED
│  ├─ masking.py ✓ UNCHANGED
│  └─ heads.py ✓ UNCHANGED
│
├─ training/
│  ├─ losses.py ✓ UPDATED
│  │  ├─ patchify_target: 256 // 16 = 16 (simplified)
│  │  └─ reconstruction_loss updated
│  │
│  └─ pretrain_trainer.py ✓ UPDATED
│     ├─ train_epoch(): unpacks 3-tuple ✓
│     ├─ validate(): unpacks 3-tuple ✓
│     └─ visualize_reconstruction(): handles 3-tuple ✓
│
└─ DOCUMENTATION ✓ ADDED
   ├─ README.md (updated)
   ├─ QUICK_REFERENCE.md (NEW)
   ├─ DATA_PIPELINE_EXPLANATION.md (NEW)
   ├─ MASK_BAND_SEPARATION_SUMMARY.md (NEW)
   ├─ COMPLETION_CHECKLIST.md (NEW)
   ├─ FINAL_SUMMARY.md (NEW)
   └─ This file: Architecture & Flow (NEW)
```

---

**Key Takeaway**: Your data flows through a complete pipeline where spectral data is normalized early (at load time), mask is kept separate, and the model processes only normalized spectral information for training! ✅
