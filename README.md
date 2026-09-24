# Pre-trained Networks for Hyperspectral Image Analysis

This repository contains the PyTorch implementation developed for the thesis **Pre-trained Deep Learning Networks for Analyzing Hyperspectral Image Data**. It implements a Masked Spatial-Spectral Transformer (MaskedSST) for learning representations from unlabeled hyperspectral image patches and transferring those representations to downstream classification tasks.

The project compares two self-supervised pre-training objectives using the same transformer backbone:

- **Masked Autoencoding (MAE):** reconstruct masked spatial-spectral patches.
- **Contrastive learning (SimCLR):** bring augmented views of the same sample together in a learned representation space.

## Project Workflow

```text
Hyperspectral TIFF files
        |
        v
Load spectral bands and optional validity mask
        |
        v
Normalize spectral data and create data loaders
        |
        v
3D blockwise patch embedding
        |
        v
Learnable spatial-spectral positional encoding
        |
        v
Factorized spatial attention + spectral attention
        |
        +--------------------+----------------------+
        |                    |                      |
        v                    v                      v
   MAE head            Contrastive head       Classification head
   masked              augmented views        fine-tuning or
   reconstruction      and InfoNCE loss       linear probing
```

Spatial attention models relationships between image locations, while spectral attention models relationships between groups of wavelength bands. This avoids constructing one full attention matrix over every spatial-spectral token.

## Repository Layout

```text
.
├── data/
│   ├── dataset.py       TIFF dataset classes and dataloader creation
│   └── transforms.py    Spatial and spectral data augmentations
├── models/
│   ├── masked_sst.py    Main MaskedSST model and model factory
│   ├── patch_embedding.py
│   ├── positional_encoding.py
│   ├── transformer_block.py
│   ├── masking.py
│   └── heads.py         Reconstruction and classification heads
├── training/
│   ├── pretrain_trainer.py  MAE and contrastive pre-training loop
│   └── losses.py            Reconstruction and InfoNCE losses
├── evaluation/
│   ├── apple_dataset.py     Apple disease dataset utilities
│   ├── coffee_dataset.py    Coffee classification dataset utilities
│   ├── finetune_mae.py      Fine-tuning from MAE weights
│   ├── finetune_contrastive.py
│   ├── train_from_scratch.py
│   ├── train_linear_probe_mae.py
│   ├── train_linear_probe_contrastive.py
│   └── linear_probe_coffee_mae.py
├── scripts/             Reserved for runnable project scripts
└── *.md                 Design notes, pipeline explanations, and checklists
```

## Data Format

The intended input is a hyperspectral TIFF file with 257 bands:

```text
Bands 1-256: hyperspectral measurements
Band 257:    validity or foreground mask
```

The dataset code separates these components before returning a sample:

```text
spectral data: (height, width, 256), normalized and passed to the model
mask data:     (height, width), kept separately in its original value range
label:         integer file or dataset label
```

The model receives only the 256 spectral bands. The TIFF mask band is not the same as the model's internal pre-training mask. During MAE pre-training, `RandomMasking` creates a new token mask and the reconstruction loss is applied to masked tokens.

The thesis experiments standardized samples to 64 x 64 pixels and 256 spectral bands. Depending on the dataset loader configuration, samples can be used as full images or extracted spatial patches.

## MaskedSST Architecture

The main model is defined in `models/masked_sst.py` and supports three modes.

### Reconstruction mode

1. Embed the input into 3D spatial-spectral tokens.
2. Add learnable positional information.
3. Mask tokens using an 85% masking ratio.
4. Apply factorized transformer blocks.
5. Reconstruct the original patch values with the reconstruction head.
6. Calculate masked L1 reconstruction loss.

### Contrastive mode

1. Create two augmented views of each hyperspectral sample.
2. Encode both views without token masking.
3. Pool spatial and spectral tokens into one feature vector.
4. Project the vector into a 128-dimensional contrastive space.
5. Optimize the representations with the InfoNCE/NT-Xent loss.

### Classification mode

1. Encode the sample without masking.
2. Apply global average pooling over spatial and spectral tokens.
3. Use a linear classification head to produce class logits.

For the thesis configuration, a 64 x 64 x 256 input is divided into 4 x 4 x 16 blocks. This produces 256 spatial locations and 16 spectral groups, or 4,096 spatial-spectral tokens. The architecture uses 128-dimensional embeddings, eight attention heads, four transformer blocks, and approximately 1.65 million trainable parameters.

The code also supports finer spatial patch sizes, including the 2 x 2 setup used in the thesis ablation study. Smaller spatial patches preserve more local detail but require more computation.

## Training

`training/pretrain_trainer.py` contains the training loop for both MAE and contrastive pre-training. It includes:

- AdamW optimization
- linear learning-rate warmup
- cosine learning-rate decay
- gradient clipping and gradient monitoring
- validation loss tracking
- checkpoint saving
- optional Weights & Biases logging
- optional distributed training support
- training-curve and diagnostic figure generation

The thesis training environment used PyTorch on an HPC cluster with NVIDIA V100 GPUs and PyTorch Distributed Data Parallel. A typical thesis setup used an effective batch size of 64, a learning rate of `1e-4`, weight decay of `0.05`, a warmup period of 10 epochs, and a fixed random seed of 42.

The exact command-line entry points are not centralized in this repository. The files under `evaluation/` are Python experiment modules and are intended to be configured for local dataset paths and available hardware.

## Installation

Create a Python environment and install the main libraries:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

```bash
pip install torch torchvision numpy scipy matplotlib tqdm rasterio
```

`wandb` is optional:

```bash
pip install wandb
```

The repository currently does not contain a `requirements.txt` or packaging configuration, so dependencies must be installed manually or added to a local environment file.

## Basic Usage

The following example shows the intended Python API. Adjust the dataset path and loader arguments to match the local data layout.

```python
from data.dataset import create_dataloaders
from models.masked_sst import create_model
from training.pretrain_trainer import PretrainTrainer

train_loader, val_loader, dataset_info = create_dataloaders(
    data_root="path/to/hyperspectral/tiffs",
    batch_size=4,
    num_workers=0,
)

model = create_model(
    num_classes=4,
    depth=4,
    patch_h=4,
    patch_w=4,
    patch_c=16,
)

trainer = PretrainTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device="cuda",
    mode="mae",
    epochs=200,
    save_dir="checkpoints/mae",
)

trainer.train()
```

For contrastive pre-training, change `mode="mae"` to `mode="contrastive"`. For downstream work, load the encoder checkpoint and use the evaluation modules to fine-tune or train a linear probe.

## Thesis Dataset and Experiments

The thesis pre-training corpus contained 4,512 unlabeled hyperspectral patches from six domains:

| Domain | Samples |
| --- | ---: |
| Coffee beans | 2,000 |
| Apple leaves | 1,500 |
| Sugar compounds | 560 |
| Paper materials | 200 |
| Archaeological artifacts | 150 |
| Grape samples | 102 |
| **Total** | **4,512** |

The main downstream experiments evaluated apple disease classification and transfer to the University of Houston 2018 hyperspectral benchmark. Comparisons included MAE versus contrastive pre-training, pre-trained versus random initialization, 2 x 2 versus 4 x 4 spatial patches, different unlabeled-data fractions, cumulative domain addition, fine-tuning, and linear probing.

## Reported Thesis Results

| Experiment | Result |
| --- | ---: |
| Apple disease, MAE, 2 x 2 patches | 88.45% accuracy |
| Apple disease, contrastive, 2 x 2 patches | 85.10% accuracy |
| Apple disease, random initialization, 2 x 2 patches | 62.30% accuracy |
| Houston 2018, multi-domain MAE pre-training | 88.00% OA |
| Houston 2018, multi-domain contrastive pre-training | 86.00% OA |
| Houston 2018, random initialization | 85.00% OA |

The thesis reported that MAE produced the strongest transfer in the evaluated experiments, fine spatial tokens preserved useful disease-related detail, and much of the benefit from unlabeled data appeared in the first 20%-40% of the pre-training corpus. These values are thesis results and are not automatically reproduced by importing the package; reproduction requires the original datasets, checkpoints, preprocessing settings, and hardware.

## Additional Documentation

- [Architecture and data flow](ARCHITECTURE_AND_FLOW.md)
- [Data pipeline explanation](DATA_PIPELINE_EXPLANATION.md)
- [Mask and band separation notes](MASK_BAND_SEPARATION_SUMMARY.md)
- [Quick reference](QUICK_REFERENCE.md)
- [Documentation index](DOCUMENTATION_INDEX.md)
- [Completion checklist](COMPLETION_CHECKLIST.md)
- [Final project summary](FINAL_SUMMARY.md)

## Limitations and Future Work

The project depends on substantial GPU memory for high-resolution token sequences and on consistent spectral calibration across sensors and domains. The thesis identifies these future directions:

- validation on airborne and satellite sensors such as AVIRIS, PRISMA, and EnMAP;
- larger contrastive batches and multi-node training;
- hybrid reconstruction and contrastive objectives;
- physics-informed spectral masking; and
- explicit cross-sensor and cross-domain adaptation.

## Citation and Attribution

This code accompanies the master's thesis by Gowtham Premkumar at Otto von Guericke University Magdeburg and the Fraunhofer Institute for Factory Operation and Automation IFF. Please cite the thesis and the relevant upstream methods, including Masked Autoencoders, SimCLR, Vision Transformers, and Masked Vision Transformers for hyperspectral image classification, when using this repository in academic work.
