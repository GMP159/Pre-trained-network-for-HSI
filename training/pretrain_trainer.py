"""
Trainer for masked reconstruction (MAE) AND Contrastive Learning (SimCLR).
Enhanced with gradient monitoring and visualizations.
Adapted for TIFF hyperspectral patches (64x64x256).
"""

import os
import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src2.training.losses import reconstruction_loss, compute_reconstruction_accuracy, info_nce_loss
from src2.data.transforms import get_contrastive_transforms

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class GradientMonitor:
    def compute_stats(self, model) -> Dict[str, float]:
        norms = []
        for _, p in model.named_parameters():
            if p.grad is not None:
                norms.append(p.grad.norm(2).item())

        if not norms:
            return {}

        return {
            "grad/norm_mean": float(np.mean(norms)),
            "grad/norm_max": float(np.max(norms)),
            "grad/norm_min": float(np.min(norms)),
            "grad/norm_std": float(np.std(norms)),
        }

class PretrainTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device="cuda",
        mode="mae",
        lr=1e-3,
        weight_decay=0.05,
        epochs=200,
        save_dir="checkpoints/pretrain",
        warmup_epochs=10,
        save_freq=10,
        use_wandb=False,
        wandb_project="hsi-pretraining",
        wandb_run_name=None,
        log_gradients_every=10,
        patch_h=1,
        patch_w=1,
        patch_c=16,
        rank=0,
        world_size=1,
        is_distributed=False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.mode = mode
        
        # DDP state
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = is_distributed
        self.is_main = (rank == 0)
        
        # Store patch configuration (must match model!)
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_c = patch_c

        self.epochs = epochs
        self.save_dir = save_dir
        self.warmup_epochs = warmup_epochs
        self.save_freq = save_freq

        self.log_gradients_every = log_gradients_every
        self.grad_monitor = GradientMonitor()
        self.gradient_norms = []
        
        # Contrastive augmentation (applied on GPU tensors)
        self.contrastive_aug = get_contrastive_transforms() if mode == 'contrastive' else None

        self.use_wandb = bool(use_wandb and WANDB_AVAILABLE and self.is_main)
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "mode": mode,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "warmup_epochs": warmup_epochs,
                    "epochs": epochs,
                    "batch_size": train_loader.batch_size,
                    "patch_h": patch_h,
                    "patch_w": patch_w,
                    "patch_c": patch_c,
                    "world_size": world_size,
                },
            )
            wandb.watch(model, log="all", log_freq=100)

        if self.is_main:
            os.makedirs(self.save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "figures"), exist_ok=True)

        # Use unwrapped model for optimizer (DDP wraps the module)
        param_model = self._unwrap_model()
        self.optimizer = optim.AdamW(
            param_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        self.base_lr = lr
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=1e-6,
        )

        self.current_epoch = 0
        self.best_val_loss = float("inf")

        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        self._fixed_vis_batch_cpu: Optional[torch.Tensor] = None
        self._init_fixed_vis_batch()

    def _unwrap_model(self):
        """Get the underlying model (unwrapped from DDP if necessary)."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    def _init_fixed_vis_batch(self) -> None:
        try:
            batch = next(iter(self.val_loader))
        except StopIteration:
            self._fixed_vis_batch_cpu = None
            return

        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            x = batch[0]
        else:
            x = batch

        if torch.is_tensor(x):
            self._fixed_vis_batch_cpu = x[:1].detach().cpu()
        else:
            self._fixed_vis_batch_cpu = None

    def _get_fixed_vis_batch(self) -> Optional[torch.Tensor]:
        if self._fixed_vis_batch_cpu is None:
            return None
        return self._fixed_vis_batch_cpu.to(self.device, non_blocking=True)

    def warmup_lr(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def _apply_contrastive_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply strong augmentations for contrastive learning.
        Works on GPU tensors for efficiency.
        
        Args:
            x: (B, H, W, C) tensor on GPU
        Returns:
            augmented: (B, H, W, C) augmented tensor
        """
        B, H, W, C = x.shape
        augmented = x.clone()
        
        for i in range(B):
            # Convert to numpy for augmentation
            patch_np = augmented[i].cpu().numpy()
            
            # Apply strong augmentation pipeline
            if self.contrastive_aug is not None:
                patch_np = self.contrastive_aug(patch_np)
            
            # Convert back to tensor
            augmented[i] = torch.from_numpy(patch_np).to(x.device)
        
        return augmented

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        # Set epoch for DistributedSampler to ensure proper shuffling
        if self.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        epoch_grad_norms = []
        self.warmup_lr(epoch)

        # Only show progress bar on main rank
        loader = self.train_loader
        if self.is_main:
            loader = tqdm(loader, desc=f"Epoch {epoch+1}/{self.epochs} Train [{self.mode}]")
        for batch_idx, batch_data in enumerate(loader):
            # Unpack data - always expect (patches, data_mask, label)
            if len(batch_data) == 3:
                patches, data_mask, _ = batch_data
            else:
                patches = batch_data[0]
                data_mask = None

            patches = patches.to(self.device, non_blocking=True)
            if data_mask is not None:
                data_mask = data_mask.to(self.device, non_blocking=True)
            
            # --- FORWARD PASS SWITCH ---
            self.optimizer.zero_grad()
            
            if self.mode == 'mae':
                # --- MAE MODE (with data mask weighting) ---
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        reconstruction, mask = self.model(patches, mode="reconstruction")
                        loss = reconstruction_loss(reconstruction, patches, mask, data_mask=data_mask,
                                                   patch_h=self.patch_h, patch_w=self.patch_w, patch_c=self.patch_c)
                else:
                    reconstruction, mask = self.model(patches, mode="reconstruction")
                    loss = reconstruction_loss(reconstruction, patches, mask, data_mask=data_mask,
                                               patch_h=self.patch_h, patch_w=self.patch_w, patch_c=self.patch_c)
                    
                # Accuracy calculation for MAE
                with torch.no_grad():
                    from src2.training.losses import patchify_target
                    target = patchify_target(patches)
                    acc = compute_reconstruction_accuracy(reconstruction, target, mask)

            elif self.mode == 'contrastive':
                # --- CONTRASTIVE MODE (with strong augmentations) ---
                # Apply strong augmentations to create two views
                view1 = self._apply_contrastive_augmentation(patches)
                view2 = self._apply_contrastive_augmentation(patches)
                
                # Forward both views
                inputs = torch.cat([view1, view2], dim=0)
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        features = self.model(inputs, mode="contrastive")
                        loss = info_nce_loss(features, batch_size=patches.shape[0])
                else:
                    features = self.model(inputs, mode="contrastive")
                    loss = info_nce_loss(features, batch_size=patches.shape[0])
                
                acc = 0.0 # No accuracy concept in unsupervised contrastive

            # --- BACKWARD ---
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            if batch_idx % self.log_gradients_every == 0:
                grad_stats = self.grad_monitor.compute_stats(self.model)
                if grad_stats:
                    epoch_grad_norms.append(grad_stats["grad/norm_mean"])
                if self.use_wandb and grad_stats:
                    wandb.log({
                        "batch": epoch * len(self.train_loader) + batch_idx,
                        **grad_stats,
                        "batch_loss": float(loss.item()),
                    })

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            total_loss += float(loss.item())
            total_acc += float(acc)
            n_batches += 1

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.is_main and hasattr(loader, 'set_postfix'):
                loader.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{acc:.2f}" if self.mode == 'mae' else "N/A",
                    "avg_loss": f"{total_loss/n_batches:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

        avg_grad_norm = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        return {
            "loss": total_loss / max(1, n_batches),
            "accuracy": total_acc / max(1, n_batches),
            "lr": self.optimizer.param_groups[0]["lr"],
            "grad_norm": avg_grad_norm,
        }

    def validate(self) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        with torch.no_grad():
            val_loader = self.val_loader
            if self.is_main:
                val_loader = tqdm(val_loader, desc="Validation")
            for batch_data in val_loader:
                # Unpack data - always expect (patches, data_mask, label)
                if len(batch_data) == 3:
                    patches, data_mask, _ = batch_data
                else:
                    patches = batch_data[0]
                    data_mask = None

                patches = patches.to(self.device, non_blocking=True)
                if data_mask is not None:
                    data_mask = data_mask.to(self.device, non_blocking=True)

                if self.mode == 'mae':
                    if self.use_amp:
                        with torch.amp.autocast('cuda'):
                            reconstruction, mask = self.model(patches, mode="reconstruction")
                            loss = reconstruction_loss(reconstruction, patches, mask, data_mask=data_mask,
                                                       patch_h=self.patch_h, patch_w=self.patch_w, patch_c=self.patch_c)
                    else:
                        reconstruction, mask = self.model(patches, mode="reconstruction")
                        loss = reconstruction_loss(reconstruction, patches, mask, data_mask=data_mask,
                                                   patch_h=self.patch_h, patch_w=self.patch_w, patch_c=self.patch_c)
                    
                    from src2.training.losses import patchify_target
                    target = patchify_target(patches, patch_h=self.patch_h, patch_w=self.patch_w, patch_c=self.patch_c)
                    acc = compute_reconstruction_accuracy(reconstruction, target, mask)
                
                elif self.mode == 'contrastive':
                    # Validation for contrastive - use simple flip (no strong augmentation)
                    view1 = patches
                    view2 = torch.flip(patches, dims=[2])
                    inputs = torch.cat([view1, view2], dim=0)
                    
                    if self.use_amp:
                        with torch.amp.autocast('cuda'):
                            features = self.model(inputs, mode="contrastive")
                            loss = info_nce_loss(features, batch_size=patches.shape[0])
                    else:
                        features = self.model(inputs, mode="contrastive")
                        loss = info_nce_loss(features, batch_size=patches.shape[0])
                    acc = 0.0

                total_loss += float(loss.item())
                total_acc += float(acc)
                n_batches += 1

        return {
            "loss": total_loss / max(1, n_batches),
            "accuracy": total_acc / max(1, n_batches),
        }

    def visualize_reconstruction(self, epoch: int) -> None:
        # Only meaningful for MAE
        if self.mode != 'mae':
            return

        self.model.eval()
        batch_data = next(iter(self.val_loader))
        if len(batch_data) == 3:
            patches, _, _ = batch_data
        else:
            patches = batch_data[0]

        patches = patches[:4].to(self.device)

        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    _reconstruction, _mask = self.model(patches, mode="reconstruction")
            else:
                _reconstruction, _mask = self.model(patches, mode="reconstruction")

        patches_cpu = patches.cpu().numpy()

        fig = plt.figure(figsize=(12, 5))

        ax1 = fig.add_subplot(1, 2, 1)
        if len(self.train_losses) > 0:
            ax1.plot(self.train_losses, label="Train", linewidth=2, alpha=0.8)
            ax1.plot(self.val_losses, label="Val", linewidth=2, alpha=0.8)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss curves")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2 = fig.add_subplot(1, 2, 2)
        spectrum = patches_cpu[0, 32, 32, :]
        ax2.plot(spectrum, linewidth=2, alpha=0.8)
        ax2.set_xlabel("Spectral band")
        ax2.set_ylabel("Value")
        ax2.set_title(f"Center pixel spectrum (epoch {epoch})")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, "figures", f"reconstruction_epoch{epoch:03d}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    # ... [Keep visualize_gradient_flow, visualize_weight_distribution, etc. unchanged] ...
    # They are generic and work for both modes. 
    # Just copy the rest of the methods (visualize_gradient_flow onwards) from your original file.
    
    def visualize_gradient_flow(self, epoch: int) -> None:
       self.model.eval()
       grads = {}
       for name, p in self.model.named_parameters():
           if p.grad is not None:
               grads[name] = p.grad.detach().cpu().numpy().flatten()
       if not grads:
           return
       fig, axes = plt.subplots(2, 1, figsize=(14, 8))
       ax = axes[0]
       all_grads = np.concatenate(list(grads.values()))
       ax.hist(all_grads, bins=100, alpha=0.7, edgecolor="black")
       ax.set_xlabel("Gradient value")
       ax.set_ylabel("Frequency")
       ax.set_title("Gradient histogram")
       ax.set_yscale("log")
       ax.grid(True, alpha=0.3)
       ax = axes[1]
       layer_names = list(grads.keys())
       grad_norms = [np.linalg.norm(g) for g in grads.values()]
       ax.bar(range(len(layer_names)), grad_norms, alpha=0.8)
       ax.set_xlabel("Layer")
       ax.set_ylabel("Gradient norm")
       ax.set_title("Gradient norm per layer")
       ax.set_yscale("log")
       ax.grid(True, alpha=0.3, axis="y")
       ax.set_xticks(range(len(layer_names)))
       ax.set_xticklabels(
           [str(n.split(".")[-2:]) for n in layer_names], rotation=45, ha="right", fontsize=8
       )
       plt.tight_layout()
       plt.savefig(
           os.path.join(self.save_dir, "figures", f"gradients_epoch{epoch:03d}.png"),
           dpi=150,
           bbox_inches="tight",
       )
       plt.close()

    def visualize_weight_distribution(self, epoch: int) -> None:
        self.model.eval()
        weights_data = {}
        for name, p in self.model.named_parameters():
            if "weight" in name:
                weights_data[name] = p.detach().cpu().numpy().flatten()
        if not weights_data:
            return
        n_layers = min(len(weights_data), 9)
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        axes = axes.flatten()
        for idx, (name, w) in enumerate(list(weights_data.items())[:n_layers]):
            ax = axes[idx]
            ax.hist(w, bins=50, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Weight value")
            ax.set_ylabel("Frequency")
            short = name.split(".")[-2] if len(name.split(".")) >= 2 else name
            ax.set_title(f"{short} (mean={w.mean():.4f}, std={w.std():.4f})", fontsize=9)
            ax.grid(True, alpha=0.3)
        for idx in range(n_layers, 9):
            axes[idx].axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, "figures", f"weights_epoch{epoch:03d}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def visualize_activations(self, epoch: int) -> None:
        if self.mode != "mae":
            return

        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from src2.training.losses import patchify_target, unpatchify_tokens

        self.model.eval()
        x = self._get_fixed_vis_batch()
        if x is None:
            return

        # Use same device as trainer
        x = x.to(self.device)

        # Get patch params from model, with defaults
        ph = self.patch_h
        pw = self.patch_w
        pc = self.patch_c

        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    pred_tokens, mask = self.model(x, mode="reconstruction")
            else:
                pred_tokens, mask = self.model(x, mode="reconstruction")

            # x is (B, H, W, C) in your MAE pipeline
            # If it's (B, C, H, W), permute once
            if x.dim() == 4 and x.shape[1] > x.shape[2]:
                # assume (B, C, H, W) -> (B, H, W, C)
                x_hw = x.permute(0, 2, 3, 1)
            else:
                x_hw = x

            # Patchify target to match pred_tokens
            target_tokens = patchify_target(
                x_hw, patch_h=ph, patch_w=pw, patch_c=pc
            )  # (B, n_spatial, n_spectral, patch_dim)

            # Reconstruct full images from pred_tokens
            B, H, W, C = x_hw.shape
            recon_full = unpatchify_tokens(
                pred_tokens, H=H, W=W, C=C, patch_h=ph, patch_w=pw, patch_c=pc
            )  # (B, H, W, C)

        # Pick first sample for token-space visualization
        pred_np = pred_tokens[0].float().cpu().numpy()      # (n_spatial, n_spectral, patch_dim)
        tgt_np = target_tokens[0].float().cpu().numpy()     # same shape
        mask_np = mask[0].float().cpu().numpy()             # (n_spatial, n_spectral)

        n_spatial = pred_np.shape[0]
        # Compute grid size assuming square spatial grid
        grid_size = int(np.sqrt(n_spatial))

        # Token-wise MSE, then average over spectral and patch dimensions
        token_mse = np.mean((pred_np - tgt_np) ** 2, axis=-1)          # (n_spatial, n_spectral)
        spatial_err = token_mse.mean(axis=1).reshape(grid_size, grid_size)
        spatial_mask = mask_np.mean(axis=1).reshape(grid_size, grid_size)
        token_energy = np.linalg.norm(pred_np, axis=-1).mean(axis=1).reshape(grid_size, grid_size)

        # === FIGURE 1: Token-space analysis ===
        num_random_patches = 4
        random_indices = np.random.choice(n_spatial, size=num_random_patches, replace=False)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Mask Map
        im0 = axes[0, 0].imshow(spatial_mask, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, 0].set_title(f"Mask Ratio ({grid_size}x{grid_size})")
        plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        # 2. Error Map
        im1 = axes[0, 1].imshow(spatial_err, cmap="hot")
        axes[0, 1].set_title("Reconstruction MSE")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # 3. Energy Map
        im2 = axes[1, 0].imshow(token_energy, cmap="viridis")
        axes[1, 0].set_title("Token Energy")
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # 4. Multi-patch spectral comparison (average over spectral blocks)
        for idx in random_indices:
            r = idx // grid_size
            c = idx % grid_size

            # Average over spectral blocks: shape (patch_dim,)
            orig = tgt_np[idx].mean(axis=0)
            recon = pred_np[idx].mean(axis=0)

            line = axes[1, 1].plot(orig, alpha=0.6, label=f"Patch({r},{c}) Target")
            axes[1, 1].plot(
                recon, "--", color=line[0].get_color(), alpha=0.8, label=f"Patch({r},{c}) Recon"
            )

        axes[1, 1].set_title(f"Spectral Comparison ({num_random_patches} Random Patches)")
        axes[1, 1].legend(fontsize="x-small", ncol=2)
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].set_xlabel("Patch Feature Index")
        axes[1, 1].set_ylabel("Value")

        plt.tight_layout()
        fig_dir = os.path.join(self.save_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, f"activations_tokens_epoch{epoch:03d}.png"), dpi=150)
        plt.close()

        # === FIGURE 2: Full image view (similar to your simpler function) ===
        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 6))
        band = 50
        ax2[0].imshow(x_hw[0, :, :, band].cpu(), cmap="gray")
        ax2[0].set_title(f"Original Band {band}")
        ax2[0].axis("off")

        ax2[1].imshow(recon_full[0, :, :, band].cpu(), cmap="gray")
        ax2[1].set_title(f"Reconstructed Band {band}")
        ax2[1].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"full_recon_epoch{epoch:03d}.png"), dpi=150)
        plt.close()





    def plot_training_curves(self) -> None:
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        epochs = range(1, len(self.train_losses) + 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(epochs, self.train_losses, label="Train", linewidth=2, alpha=0.8)
        ax.plot(epochs, self.val_losses, label="Val", linewidth=2, alpha=0.8)
        ax.fill_between(epochs, self.train_losses, self.val_losses, alpha=0.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(epochs, self.learning_rates, linewidth=2, marker="o", markersize=3, alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_title("Learning rate schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax = fig.add_subplot(gs[0, 2])
        if len(self.gradient_norms) > 0:
            ax.plot(epochs[: len(self.gradient_norms)], self.gradient_norms, linewidth=2, alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Grad norm")
        ax.set_title("Gradient norm")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax = fig.add_subplot(gs[1, 0])
        recent_epochs = list(epochs)[-min(50, len(self.train_losses)) :]
        recent_train = self.train_losses[-min(50, len(self.train_losses)) :]
        recent_val = self.val_losses[-min(50, len(self.val_losses)) :]
        ax.plot(recent_epochs, recent_train, label="Train", linewidth=2)
        ax.plot(recent_epochs, recent_val, label="Val", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss (recent)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax = fig.add_subplot(gs[1, 1])
        if len(self.val_losses) > 1:
            improvement = np.array(self.val_losses[:-1]) - np.array(self.val_losses[1:])
            ax.bar(list(epochs)[:-1], improvement, alpha=0.7)
            ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val loss improvement")
        ax.set_title("Val improvement per epoch")
        ax.grid(True, alpha=0.3, axis="y")
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(epochs, self.val_losses, linewidth=2, alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val loss")
        ax.set_title("Validation loss")
        ax.grid(True, alpha=0.3)
        ax = fig.add_subplot(gs[2, 0])
        if len(self.val_losses) > 1:
            cum = np.cumsum(np.array(self.val_losses[0]) - np.array(self.val_losses))
            ax.plot(epochs, cum, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cumulative improvement")
        ax.set_title("Cumulative improvement")
        ax.grid(True, alpha=0.3)
        ax = fig.add_subplot(gs[2, 1])
        if len(self.val_losses) > 0:
            ratio = np.array(self.train_losses) / (np.array(self.val_losses) + 1e-8)
            ax.plot(epochs, ratio, linewidth=2)
            ax.axhline(y=1.0, color="black", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train/Val ratio")
        ax.set_title("Train/Val loss ratio")
        ax.grid(True, alpha=0.3)
        ax = fig.add_subplot(gs[2, 2])
        ax.axis("off")
        if len(self.train_losses) > 0 and len(self.val_losses) > 0:
            text = (
                f"Mode: {self.mode}\n"
                f"Total epochs: {len(self.train_losses)}\n"
                f"Best val loss: {self.best_val_loss:.6f}\n"
                f"Final val loss: {self.val_losses[-1]:.6f}\n"
                f"Best train loss: {float(np.min(self.train_losses)):.6f}\n"
                f"Final train loss: {self.train_losses[-1]:.6f}\n"
                f"LR max: {float(np.max(self.learning_rates)):.2e}\n"
                f"LR min: {float(np.min(self.learning_rates)):.2e}\n"
            )
            ax.text(0.05, 0.5, text, fontsize=10, va="center", family="monospace")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, "training_curves_comprehensive.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        # Only save on main rank
        if not self.is_main:
            return
        
        # Always save the unwrapped model state_dict (without DDP wrapper)
        unwrapped = self._unwrap_model()
        checkpoint = {
            "epoch": epoch,
            "mode": self.mode,
            "model_state_dict": unwrapped.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "gradient_norms": self.gradient_norms,
            "patch_h": self.patch_h,
            "patch_w": self.patch_w,
            "patch_c": self.patch_c,
        }

        if epoch % self.save_freq == 0:
            path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch:03d}.pth")
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = os.path.join(self.save_dir, "checkpoint_best.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint (val loss {self.best_val_loss:.4f})")

    def train(self) -> None:
        if self.is_main:
            logger.info(f"Starting pretraining in {self.mode.upper()} mode")
            logger.info(f"Device: {self.device}")
            logger.info(f"Distributed: {self.is_distributed} (world_size={self.world_size})")
            logger.info(f"Epochs: {self.epochs}")
            logger.info(f"Use wandb: {self.use_wandb}")
            logger.info(f"Gradient logging every: {self.log_gradients_every} batches")

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            self.train_losses.append(train_metrics["loss"])
            self.val_losses.append(val_metrics["loss"])
            self.learning_rates.append(train_metrics["lr"])
            self.gradient_norms.append(train_metrics["grad_norm"])

            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train/loss": train_metrics["loss"],
                        "train/accuracy": train_metrics["accuracy"],
                        "val/loss": val_metrics["loss"],
                        "val/accuracy": val_metrics["accuracy"],
                        "lr": train_metrics["lr"],
                        "grad_norm": train_metrics["grad_norm"],
                    }
                )

            if self.is_main:
                logger.info(
                    f"Epoch {epoch+1}/{self.epochs} "
                    f"train_loss={train_metrics['loss']:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"lr={train_metrics['lr']:.2e}"
                )

            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            self.save_checkpoint(epoch + 1, is_best=is_best)

            # Visualizations only on main rank
            if self.is_main:
                if (epoch + 1) % 5 == 0 or is_best:
                    self.visualize_reconstruction(epoch + 1)

                if (epoch + 1) % 10 == 0 or is_best:
                    self.visualize_gradient_flow(epoch + 1)

                if (epoch + 1) % 20 == 0 or is_best:
                    self.visualize_weight_distribution(epoch + 1)

                if (epoch + 1) % 15 == 0 or is_best:
                    self.visualize_activations(epoch + 1)

                if (epoch + 1) % 5 == 0:
                    self.plot_training_curves()
            
            # Synchronize all processes at epoch boundary
            if self.is_distributed:
                dist.barrier()

        self.save_checkpoint(self.epochs, is_best=False)
        if self.is_main:
            self.plot_training_curves()

        if self.use_wandb:
            wandb.finish()

        if self.is_main:
            logger.info(f"Finished pretraining, best_val_loss={self.best_val_loss:.4f}")
