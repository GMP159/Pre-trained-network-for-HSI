# Pre-trained Deep Learning Networks for Hyperspectral Image Data

> **Master Thesis** — Otto von Guericke University Magdeburg, 2025
>
> **Author:** Gowtham Premkumar
>
> **Supervisor:** Prof. Dr. rer. nat. Frank Ortmeier
>
> **Advisers:** Dr.-Ing. Andreas Herzog, M.Sc. Konstantin Kirchheim
>
> Fraunhofer Institute for Factory Operation and Automation (IFF)

---

## Overview

This repository contains the implementation of a **Masked Spatial-Spectral Transformer (MaskedSST)** for self-supervised pre-training on hyperspectral image (HSI) data. The architecture integrates blockwise 3D patch embedding, factorized spatial-spectral attention, and two self-supervised pre-training strategies — **Masked Autoencoding (MAE)** and **Contrastive Learning (SimCLR)** — to learn transferable representations from unlabeled hyperspectral data.

The key research questions addressed:
1. Can self-supervised pre-training improve HSI classification in small-data regimes vs. training from scratch?
2. Which pre-training approach is more effective: reconstruction-based (MAE) or discriminative (SimCLR)?
3. Do the learned representations capture meaningful spatial-spectral patterns that transfer to downstream tasks?

### Key Results

| Method             | 2×2 Patch Acc. | 4×4 Patch Acc. | Δ vs Random (2×2) |
|--------------------|:--------------:|:--------------:|:------------------:|
| MAE Pre-trained    | 88.45%         | 82.10%         | +26.15%            |
| SimCLR Pre-trained | 84.20%         | 78.50%         | +21.90%            |
| Random Baseline    | 62.30%         | 59.15%         | —                  |

---

## Architecture

```
Input HSI Patch (B, 64, 64, 256)
        │
        ▼
┌─────────────────────────────┐
│  Blockwise 3D Patch Embedding│  16 separate linear projections
│  (B, 64, 64, 256) → (B, N_s,│  (one per spectral block)
│   N_λ, 128)                 │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  3D Positional Encoding     │  Sinusoidal (height, width, spectral)
└──────────────┬──────────────┘
               ▼
        ┌──────┴──────┐
   MAE  │  Select Model│  Contrastive
        └──────┬──────┘
               │
    ┌──────────┼──────────┐
    ▼                     ▼
┌──────────┐       ┌───────────────┐
│ Random   │       │ Augmentation  │
│ Masking  │       │ (View 1 & 2)  │
│ (85%)    │       │               │
└────┬─────┘       └──────┬────────┘
     ▼                    ▼
┌─────────────────────────────┐
│  Factorized Transformer     │  4× blocks
│  Encoder                    │
│  ├─ Spatial Attention       │  O(N_spatial²)
│  └─ Spectral Attention      │  O(N_spectral²)
└──────────────┬──────────────┘
               ▼
    ┌──────────┼──────────┐
    ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌──────────┐
│Recon   │ │Classif.│ │Projection│
│Head    │ │Head    │ │Head      │
│(MAE)   │ │(Finetune)│(SimCLR)  │
└────────┘ └────────┘ └──────────┘
```

### Model Configuration

| Component            | Configuration                    |
|----------------------|----------------------------------|
| Input Size           | 64 × 64 × 256                   |
| Spatial Patch Size   | 2×2 or 4×4                      |
| Spectral Patch Size  | 16 bands per block               |
| Embedding Dimension  | 128                              |
| Encoder Depth        | 4 layers                         |
| Attention Heads      | 8 per layer                      |
| MLP Expansion Ratio  | 4                                |
| Masking Ratio (MAE)  | 85%                              |
| Total Parameters     | ~1.65M                           |

### Factorized Attention Speedup

Standard self-attention on 4,096 tokens: $O(N^2) \approx 16.8\text{M}$ operations

Factorized (spatial + spectral): $O(N_{\text{spatial}}^2 + N_{\text{spectral}}^2) = O(256^2 + 16^2) \approx 65.8\text{K}$ operations — a **256× speedup**

---

## Project Structure

```
├── models/
│   ├── masked_sst.py            # Main MaskedSST model (encoder + heads)
│   ├── patch_embedding.py       # Blockwise 3D spectral patch embedding
│   ├── positional_encoding.py   # 3D sinusoidal positional encoding
│   ├── transformer_block.py     # Factorized spatial-spectral transformer block
│   ├── masking.py               # Tube masking & random masking strategies
│   └── heads.py                 # Reconstruction, classification & projection heads
│
├── data/
│   ├── dataset.py               # TIFF dataset loaders (64×64×257 GeoTIFF)
│   └── transforms.py           # HSI-aware augmentations (spatial + spectral)
│
├── training/
│   ├── losses.py                # MAE reconstruction loss, InfoNCE, CrossEntropy
│   └── pretrain_trainer.py      # Pre-training orchestration (MAE & SimCLR)
│
├── evaluation/
│   ├── apple_dataset.py         # Apple disease classification dataset (4 classes)
│   ├── coffee_dataset.py        # Coffee variety classification dataset (3 classes)
│   ├── finetune_mae.py          # Full fine-tuning with MAE pre-trained weights
│   ├── finetune_contrastive.py  # Full fine-tuning with SimCLR pre-trained weights
│   ├── train_from_scratch.py    # Baseline: training without pre-training
│   ├── train_linear_probe_mae.py        # Linear probe on MAE encoder (apple)
│   ├── train_linear_probe_contrastive.py# Linear probe on SimCLR encoder (apple)
│   └── linear_probe_coffee_mae.py       # Linear probe on MAE encoder (coffee)
│
├── outputs/
│   ├── mae/                     # MAE pre-training loss curves
│   └── contrastive/             # SimCLR pre-training loss curves
│
├── run_pretrain_enhanced.py     # Main pre-training entry point (single/multi-GPU)
├── run_pretrain_cluster.slurm   # SLURM script for HPC pre-training
├── run_finetune_all_mae.slurm   # SLURM script for MAE fine-tuning experiments
└── run_finetune_all_con.slurm   # SLURM script for SimCLR fine-tuning experiments
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.x with CUDA support
- rasterio (GeoTIFF I/O)
- numpy, scipy, tqdm, matplotlib, scikit-learn

```bash
pip install torch torchvision rasterio numpy scipy tqdm matplotlib scikit-learn
```

Optional (experiment tracking):
```bash
pip install wandb
```

---

## Dataset

### Pre-training Dataset (Unlabeled)

A multi-domain collection of **4,512 hyperspectral patches** (64×64 spatial, 256 spectral bands) spanning six application domains:

| Domain                | Description                                       |
|-----------------------|---------------------------------------------------|
| Coffee Beans          | Arabica & Canephora from Costa Rica and Ecuador   |
| Apple Leaves          | VNIR and SWIR scans with disease variability       |
| Paper Materials       | Industrial material signatures                     |
| Sugar Compounds       | Pure chemical endmembers (Galactose, Glucose, etc.)|
| Pottery (Scherben)    | Mineralogical spectral signatures                  |
| Grapes (JKI Bluestar) | Vegetation-specific features                       |

### Downstream Datasets (Labeled)

**Apple Disease Classification (4 classes):**

| Class | Disease     | Train | Test |
|-------|-------------|------:|-----:|
| 0     | Healthy     | 2,599 | 650  |
| 1     | Scab        | 561   | 141  |
| 2     | Rust        | 1,957 | 489  |
| 3     | Fire Blight | 381   | 95   |

**Coffee Variety Classification (3 classes):** Arabica, Immature, Robusta

### Data Format

- GeoTIFF files with shape `(64, 64, 257)`
- Bands 1–256: spectral reflectance data
- Band 257: validity mask (foreground/background)
- Per-patch normalization: $\hat{X} = (X - \mu) / (\sigma + \epsilon)$

---

## Usage

### Pre-training

**Single GPU (MAE):**
```bash
python run_pretrain_enhanced.py \
    --data_root data/patches_64x64_geotiff \
    --mode mae \
    --batch_size 32 \
    --epochs 200 \
    --lr 1e-4 \
    --patch_h 4 --patch_w 4 --patch_c 16
```

**Single GPU (SimCLR):**
```bash
python run_pretrain_enhanced.py \
    --data_root data/patches_64x64_geotiff \
    --mode contrastive \
    --batch_size 32 \
    --epochs 200 \
    --lr 1e-4
```

**Multi-GPU (HPC Cluster):**
```bash
torchrun --nproc_per_node=8 run_pretrain_enhanced.py \
    --data_root /scratch/data/patches_64x64_geotiff \
    --mode mae \
    --batch_size 8 \
    --epochs 200 \
    --lr 1e-4
```

**SLURM Submission:**
```bash
sbatch run_pretrain_cluster.slurm
```

### Fine-tuning

**With MAE pre-trained weights:**
```bash
python evaluation/finetune_mae.py --use_pretrained
```

**With SimCLR pre-trained weights:**
```bash
python evaluation/finetune_contrastive.py --use_pretrained
```

**From scratch (baseline):**
```bash
python evaluation/train_from_scratch.py
```

### Linear Probe Evaluation

**Apple dataset with MAE encoder:**
```bash
python evaluation/train_linear_probe_mae.py --use_pretrained
```

**Apple dataset with SimCLR encoder:**
```bash
python evaluation/train_linear_probe_contrastive.py --use_pretrained
```

**Coffee dataset with MAE encoder:**
```bash
python evaluation/linear_probe_coffee_mae.py --use_pretrained
```

---

## Training Configuration

| Parameter              | Pre-training       | Fine-tuning       |
|------------------------|:------------------:|:-----------------:|
| Optimizer              | AdamW              | AdamW             |
| Learning Rate          | 1×10⁻⁴             | 1×10⁻⁴            |
| Weight Decay           | 0.05               | 0.05              |
| LR Schedule            | Cosine + Warmup    | Cosine Annealing  |
| Warmup Epochs          | 10                 | —                 |
| Gradient Clip Norm     | 1.0                | 1.0               |
| Batch Size (effective) | 64 (8×8 GPUs)      | 8–32              |
| Max Epochs             | 200                | 100               |
| Mixed Precision        | Yes (AMP)          | No                |
| Seed                   | 42                 | 42                |

---

## Pre-training Convergence

### MAE Reconstruction Loss

| Patch Size | Initial Loss | Final Train Loss | Final Val Loss | Epochs |
|:----------:|:------------:|:----------------:|:--------------:|:------:|
| 2×2        | 0.80         | 0.1162           | 0.1085         | 60     |
| 4×4        | 1.50         | 0.3779           | 0.3832         | 110    |

### SimCLR Contrastive Loss

| Patch Size | Initial Loss | Final Train Loss | Final Val Loss | Epochs |
|:----------:|:------------:|:----------------:|:--------------:|:------:|
| 2×2        | 1.20         | 0.3346           | 0.5420         | 30     |
| 4×4        | 2.25         | 0.4503           | 0.6422         | 80     |

---

## Hardware

Pre-training was conducted on the HPC cluster of Otto von Guericke University Magdeburg:

- **GPUs:** 8× NVIDIA Tesla V100-SXM2 (32 GB HBM2 each)
- **CPU:** 32 cores Intel Xeon
- **RAM:** 64 GB
- **CUDA:** 12.2
- **Distributed Training:** PyTorch DDP with NCCL backend

---

## Citation

```
Premkumar, Gowtham.
Pre-trained Deep Learning Networks for Hyperspectral Image Data.
Master Thesis, Otto von Guericke University Magdeburg, 2025.
```

## References

1. Adão et al. *Hyperspectral imaging: A review on UAV-based sensors.* Remote Sensing, 2017.
2. Chen et al. *A simple framework for contrastive learning of visual representations (SimCLR).* ICML, 2020.
3. Chen et al. *Deep feature extraction and classification of hyperspectral images based on CNNs.* IEEE TGRS, 2016.
4. Dosovitskiy et al. *An image is worth 16x16 words: Transformers for image recognition at scale (ViT).* ICLR, 2021.
5. He et al. *Masked autoencoders are scalable vision learners (MAE).* CVPR, 2022.
6. Hong et al. *SpectralFormer: Rethinking hyperspectral image classification with transformers.* IEEE TGRS, 2022.
7. Scheibenreif et al. *Masked vision transformers for hyperspectral image classification.* CVPRW, 2023.

---

## License

This project was developed as part of a Master Thesis at Otto von Guericke University Magdeburg in collaboration with Fraunhofer IFF.
