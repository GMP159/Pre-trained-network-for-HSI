#!/usr/bin/env python3
"""
Quick Start Training Script for src2 (TIFF Format)
Runs masked reconstruction (MAE) OR Contrastive Learning (SimCLR) pre-training.
Supports single-GPU and multi-GPU (DDP via torchrun).

Usage:
    # Local single-GPU (MAE default):
    python run_pretrain_enhanced.py

    # Local single-GPU (Contrastive):
    python run_pretrain_enhanced.py --mode contrastive
    
    # Multi-GPU with DDP (cluster):
    torchrun --nproc_per_node=8 run_pretrain_enhanced.py --data_root /scratch/data --batch_size 64 --epochs 100
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import argparse

# Add project to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src2.data.dataset import create_dataloaders
from src2.models.masked_sst import create_model
from src2.training.pretrain_trainer import PretrainTrainer
from src2.data.transforms import get_train_transforms, get_contrastive_transforms, get_val_transforms

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_ddp():
    """Initialize DDP if launched via torchrun. Returns (rank, local_rank, world_size, is_distributed)."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='TIFF Hyperspectral Pretraining',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local training (MAE default):
  python run_pretrain_enhanced.py
  
  # Contrastive Learning:
  python run_pretrain_enhanced.py --mode contrastive

  # Cluster training:
  python run_pretrain_enhanced.py \
    --data_root /scratch/data \
    --mode contrastive \
    --batch_size 64 
        """
    )
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default=r"data",
                        help='Path to TIFF data root directory (default: local Windows path)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers (default: 4)')
    
    # NEW: Mode Argument
    parser.add_argument('--mode', type=str, default='mae', choices=['mae', 'contrastive'],
                        help='Training mode: mae (reconstruction) or contrastive (SimCLR)')

    # Model arguments
    parser.add_argument('--depth', type=int, default=4,
                        help='Transformer depth (number of blocks, default: 4)')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--patch_h', type=int, default=4,
                        help='Spatial patch height (default: 4, use 2 for finer resolution)')
    parser.add_argument('--patch_w', type=int, default=4,
                        help='Spatial patch width (default: 4, use 2 for finer resolution)')
    parser.add_argument('--patch_c', type=int, default=16,
                        help='Spectral patch size (default: 16)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs (default: 10)')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='outputs/checkpoints/pretrain_tiff_enhanced',
                        help='Directory to save checkpoints and visualizations')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    return parser.parse_args()

def main():
    """Main training function. Supports single-GPU and multi-GPU DDP."""
    
    # ============================================================================
    # DDP SETUP (auto-detected from torchrun environment)
    # ============================================================================
    rank, local_rank, world_size, is_distributed = setup_ddp()
    is_main = (rank == 0)  # Only rank 0 logs, saves, visualizes
    
    # Suppress logging on non-main ranks
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Parse arguments
    args = parse_arguments()
    
    if is_main:
        logger.info("="*80)
        logger.info("SRC2 - TIFF HYPERSPECTRAL PRETRAINING")
        logger.info("With Rich Visualizations, Gradient Monitoring & Analysis")
        if is_distributed:
            logger.info(f"DDP: {world_size} GPUs across {os.environ.get('SLURM_NODELIST', 'local')}")
        logger.info("="*80)
    
    # ============================================================================
    # CONFIGURATION (from arguments)
    # ============================================================================
    
    # Data
    DATA_ROOT = args.data_root
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    MODE = args.mode
    PATCH_SIZE = 32
    
    # Model
    NUM_CLASSES = 1
    EMBED_DIM = args.embed_dim
    DEPTH = args.depth
    NUM_HEADS = 8
    PATCH_H = args.patch_h
    PATCH_W = args.patch_w
    PATCH_C = args.patch_c
    
    # Training
    EPOCHS = args.epochs
    LR = args.lr
    WEIGHT_DECAY = 0.05
    WARMUP_EPOCHS = args.warmup_epochs
    
    # Logging & Checkpoints
    SAVE_DIR = os.path.join(args.save_dir, MODE)
    USE_WANDB = args.use_wandb and is_main  # Only rank 0 uses wandb
    WANDB_PROJECT = 'hsi-pretrain'
    
    # Device
    if is_distributed:
        DEVICE = f'cuda:{local_rank}'
    else:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if is_main:
        logger.info(f"Training Mode: {MODE.upper()}")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Distributed: {is_distributed} (world_size={world_size})")
        logger.info(f"Data Root: {DATA_ROOT}")
        logger.info(f"Epochs: {EPOCHS}")
        logger.info(f"Batch Size: {BATCH_SIZE} per GPU ({BATCH_SIZE * world_size} effective)")
        logger.info(f"Num Workers: {NUM_WORKERS}")
        logger.info(f"Learning Rate: {LR}")
        logger.info(f"Model Depth: {DEPTH}")
        logger.info(f"Patch Size: {PATCH_H}x{PATCH_W} spatial, {PATCH_C} spectral")
        logger.info(f"Save Directory: {SAVE_DIR}")
        logger.info("="*80)
    
    # ============================================================================
    # LOAD DATA
    # ============================================================================
    
    logger.info("Loading data...")
    try:
        # TIFF files are in subdirectories of data_root
        # The dataset loader will search recursively
        import glob
        tiff_count = len(glob.glob(f"{DATA_ROOT}/**/*.tif*", recursive=True))
        logger.info(f"Found {tiff_count} TIFF files in {DATA_ROOT} (including subdirs)")
        
        # Select transforms based on training mode
        if MODE == 'contrastive':
            train_transforms = get_contrastive_transforms()
            logger.info("Using STRONG contrastive augmentations")
        else:
            train_transforms = get_train_transforms()
            logger.info("Using standard MAE augmentations")
        
        val_transforms = get_val_transforms()
        
        train_loader, val_loader, dataset_info = create_dataloaders(
            data_root=DATA_ROOT,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            train_ratio=0.8,
            simple_mode=True,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            distributed=is_distributed
        )
        
        if is_main:
            logger.info(f" Data loaded successfully!")
            logger.info(f"  - Train samples: {dataset_info['train_samples']}")
            logger.info(f"  - Val samples: {dataset_info['val_samples']}")
            logger.info(f"  - Image size: {dataset_info['img_size']}x{dataset_info['img_size']}")
            logger.info(f"  - Spectral bands: {dataset_info['num_bands']} (256 data + 1 mask)")
            logger.info(f"  - Has mask: {dataset_info.get('has_mask', False)}")
        
    except Exception as e:
        logger.error(f" Error loading data: {e}")
        logger.error(f"  Make sure TIFF files are in: {DATA_ROOT}")
        return
    
    # ============================================================================
    # CREATE MODEL
    # ============================================================================
    
    if is_main:
        logger.info("Creating model...")
    try:
        model = create_model(
            num_classes=NUM_CLASSES, 
            depth=DEPTH,
            patch_h=PATCH_H,
            patch_w=PATCH_W,
            patch_c=PATCH_C,
            embed_dim=EMBED_DIM
        )
        model = model.to(DEVICE)
        
        # Wrap model in DDP if distributed
        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=True)
        
        # Calculate spatial tokens for info
        n_spatial = (64 // PATCH_H) * (64 // PATCH_W)
        n_spectral = 256 // PATCH_C
        total_tokens = n_spatial * n_spectral
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if is_main:
            logger.info(f" Model created successfully!")
            if is_distributed:
                logger.info(f"  - Wrapped in DistributedDataParallel")
            logger.info(f"  - Patch size: {PATCH_H}x{PATCH_W}x{PATCH_C}")
            logger.info(f"  - Spatial tokens: {n_spatial} ({64//PATCH_H}x{64//PATCH_W})")
            logger.info(f"  - Spectral groups: {n_spectral}")
            logger.info(f"  - Total tokens per image: {total_tokens}")
            logger.info(f"  - Total parameters: {total_params:,}")
            logger.info(f"  - Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f" Error creating model: {e}")
        cleanup_ddp()
        return
    
    # ============================================================================
    # CREATE TRAINER
    # ============================================================================
    
    if is_main:
        logger.info(f"Creating trainer (Mode: {MODE})...")
    try:
        trainer = PretrainTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            mode=MODE,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            epochs=EPOCHS,
            save_dir=SAVE_DIR,
            warmup_epochs=WARMUP_EPOCHS,
            save_freq=20,
            use_wandb=USE_WANDB,
            wandb_project=WANDB_PROJECT,
            wandb_run_name=f'hsi-{MODE}-tiff-v1',
            log_gradients_every=10,
            patch_h=PATCH_H,
            patch_w=PATCH_W,
            patch_c=PATCH_C,
            rank=rank,
            world_size=world_size,
            is_distributed=is_distributed
        )
        
        if is_main:
            logger.info(f" Trainer created successfully!")
            logger.info(f"  - Save directory: {SAVE_DIR}")
            logger.info(f"  - Mixed precision (AMP): {trainer.use_amp}")
            logger.info("="*80)
        
    except Exception as e:
        logger.error(f" Error creating trainer: {e}")
        cleanup_ddp()
        return
    
    # ============================================================================
    # START TRAINING
    # ============================================================================
    
    if is_main:
        logger.info("Starting training with rich visualizations...")
        logger.info("Visualizations generated:")
        logger.info("  - Every 5 epochs: Reconstruction comparison + Training curves")
        logger.info("  - Every 10 epochs: Gradient flow analysis")
        logger.info("  - Every 15 epochs: Feature activations")
        logger.info("  - Every 20 epochs: Weight distributions")
        logger.info("  - After best model: All visualizations")
        logger.info("="*80)
    
    try:
        trainer.train()
        if is_main:
            logger.info("="*80)
            logger.info(" TRAINING COMPLETE!")
            logger.info(f" Outputs saved to: {SAVE_DIR}")
            logger.info("="*80)
        
    except KeyboardInterrupt:
        if is_main:
            logger.warning("\n Training interrupted by user")
            logger.info(f"  Partial outputs saved to: {SAVE_DIR}")
        
    except Exception as e:
        logger.error(f" Error during training on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cleanup_ddp()

if __name__ == '__main__':
    main()
