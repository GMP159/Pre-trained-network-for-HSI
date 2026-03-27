"""
Loss functions for pre-training and fine-tuning
ADAPTED FOR TIFF FILES: 64x64x257
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(features, batch_size, n_views=2, temperature=0.2):
    """
    Computes SimCLR / InfoNCE loss.
    Args:
        features: (n_views * batch_size, dim) - Concatenated features from all views
        batch_size: int
        n_views: int (usually 2)
        temperature: float (scale factor)
    """
    # Normalize features (Cosine Similarity requires normalized vectors)
    features = F.normalize(features, dim=1)

    # Similarity matrix: (2N, 2N)
    similarity_matrix = torch.matmul(features, features.T)

    # Create labels: The "positive" for image i is image i+batch_size
    labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    # Mask out self-similarity (diagonal)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # Select positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    
    # Select negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # Concatenate: [positives, negatives]
    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature

    # Labels are all 0 (because the positive is now at index 0)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    return F.cross_entropy(logits, labels)

def patchify_target(x, patch_h=2, patch_w=2, patch_c=16):
    """
    Patchify input to match model's patch embedding.
    Must match EXACTLY how PatchEmbedding3D works!

    Args:
        x: (B, H, W, C) = (B, 64, 64, 256) - spectral data only
        patch_h: 4
        patch_w: 4
        patch_c: 16

    Returns:
        patches: (B, n_spatial, n_spectral, patch_dim)
                 (B, 256, 16, 256) for TIFF format
    """
    B, H, W, C = x.shape

    n_spatial_h = H // patch_h
    n_spatial_w = W // patch_w
    n_spectral = C // patch_c  # 256 / 16 = 16 (clean division)
    patch_dim = patch_h * patch_w * patch_c  # 256

    # Reshape to patches - SAME as PatchEmbedding3D
    x = x.reshape(
        B,
        n_spatial_h, patch_h,
        n_spatial_w, patch_w,
        n_spectral, patch_c
    )

    # Rearrange dimensions
    x = x.permute(0, 1, 3, 5, 2, 4, 6)

    # Flatten each patch
    n_spatial = n_spatial_h * n_spatial_w
    x = x.reshape(B, n_spatial, n_spectral, patch_dim)

    return x

def unpatchify_tokens(tokens, H=64, W=64, C=256, patch_h=2, patch_w=2, patch_c=16):
    B, n_spatial, n_spectral, patch_dim = tokens.shape
    n_h = H // patch_h
    n_w = W // patch_w

    x = tokens.reshape(B, n_h, n_w, n_spectral, patch_h, patch_w, patch_c)
    x = x.permute(0, 1, 4, 2, 5, 3, 6)
    x = x.reshape(B, H, W, C)
    return x

def masked_l1_loss(pred, target, mask):
    """
    L1 loss on masked tokens only.

    Args:
        pred: (B, n_spatial, n_spectral, patch_dim)
        target: (B, n_spatial, n_spectral, patch_dim)
        mask: (B, n_spatial, n_spectral) binary, 1=masked, 0=visible

    Returns:
        loss: Scalar loss
    """
    B, n_spatial, n_spectral, patch_dim = pred.shape

    # Expand mask for all features
    mask_expanded = mask.unsqueeze(-1)  # (B, n_spatial, n_spectral, 1)

    # Compute L1 loss per element
    loss = torch.abs(pred - target)  # (B, n_spatial, n_spectral, patch_dim)

    # Apply mask (broadcasting happens here!)
    masked_loss = loss * mask_expanded  # (B, n_spatial, n_spectral, patch_dim)

    # Count masked elements correctly
    n_masked_tokens = mask.sum()  # Number of masked tokens
    n_masked_elements = n_masked_tokens * patch_dim  # Total masked elements

    if n_masked_tokens == 0:
        return torch.tensor(0.0, device=pred.device)

    # Sum all losses and divide by total masked elements
    total_loss = masked_loss.sum()
    avg_loss = total_loss / n_masked_elements

    return avg_loss


def patchify_mask(data_mask, patch_h=1, patch_w=1):
    """
    Convert (B, H, W) data mask to (B, n_spatial) patch-level mask.
    A patch is valid if the MAJORITY of its pixels are valid.
    
    Args:
        data_mask: (B, H, W) - pixel-level validity mask (1=valid, 0=invalid)
        patch_h: patch height
        patch_w: patch width
    
    Returns:
        patch_mask: (B, n_spatial) - patch-level validity (avg validity per patch)
    """
    B, H, W = data_mask.shape
    n_h = H // patch_h
    n_w = W // patch_w
    
    # Reshape to patches: (B, n_h, patch_h, n_w, patch_w)
    reshaped = data_mask.reshape(B, n_h, patch_h, n_w, patch_w)
    
    # Permute: (B, n_h, n_w, patch_h, patch_w)
    reshaped = reshaped.permute(0, 1, 3, 2, 4)
    
    # Average validity per patch: (B, n_h, n_w)
    patch_validity = reshaped.float().mean(dim=(3, 4))
    
    # Flatten spatial dims: (B, n_spatial)
    patch_mask = patch_validity.reshape(B, -1)
    
    return patch_mask


def reconstruction_loss(model_output, original_patches, mask, data_mask=None,
                        patch_h=2, patch_w=2, patch_c=16):
    """
    Complete reconstruction loss for pre-training.
    Optionally weighted by data validity mask.

    Args:
        model_output: (B, n_spatial, n_spectral, patch_dim) from decoder
        original_patches: (B, 64, 64, 256) original spectral input (mask is separate)
        mask: (B, n_spatial, n_spectral) binary mask (1=masked by model, 0=visible)
        data_mask: (B, H, W) optional data validity mask (1=valid pixels, 0=invalid)
        patch_h: Spatial patch height (must match model)
        patch_w: Spatial patch width (must match model)
        patch_c: Spectral patch size (must match model)

    Returns:
        loss: Scalar loss
    """
    # Patchify original input to match model output
    target_patches = patchify_target(
        original_patches,
        patch_h=patch_h,
        patch_w=patch_w,
        patch_c=patch_c
    )

    B, n_spatial, n_spectral, patch_dim = model_output.shape
    
    # If data_mask provided, weight the loss by data validity
    if data_mask is not None:
        # Convert pixel mask to patch-level validity weights
        patch_validity = patchify_mask(data_mask, patch_h=patch_h, patch_w=patch_w)  # (B, n_spatial)
        
        # Expand to (B, n_spatial, n_spectral, 1) to match model output
        validity_weight = patch_validity.unsqueeze(-1).unsqueeze(-1)  # (B, n_spatial, 1, 1)
        validity_weight = validity_weight.expand(-1, -1, n_spectral, 1)  # (B, n_spatial, n_spectral, 1)
        
        # Compute weighted L1 loss
        mask_expanded = mask.unsqueeze(-1)  # (B, n_spatial, n_spectral, 1)
        
        # Combined mask: model mask AND data validity
        combined_weight = mask_expanded * validity_weight  # (B, n_spatial, n_spectral, 1)
        
        # L1 difference
        diff = torch.abs(model_output - target_patches)  # (B, n_spatial, n_spectral, patch_dim)
        
        # Weighted loss
        weighted_loss = diff * combined_weight
        
        # Normalize by sum of weights
        total_weight = combined_weight.sum() * patch_dim
        if total_weight < 1e-6:
            return torch.tensor(0.0, device=model_output.device)
        
        loss = weighted_loss.sum() / total_weight
    else:
        # Original unweighted loss
        loss = masked_l1_loss(model_output, target_patches, mask)

    return loss


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss for classification with label smoothing."""

    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        """
        Args:
            logits: (B, num_classes)
            labels: (B,) class indices

        Returns:
            loss: Scalar loss
        """
        return self.ce_loss(logits, labels)


def compute_reconstruction_accuracy(pred, target, mask, threshold=0.05):
    """
    Compute "accuracy" of reconstruction (% of features within threshold).

    Args:
        pred: (B, n_spatial, n_spectral, patch_dim)
        target: (B, n_spatial, n_spectral, patch_dim)
        mask: (B, n_spatial, n_spectral)
        threshold: Consider "correct" if |pred - target| < threshold

    Returns:
        accuracy: Scalar in [0, 100] (percentage)
    """
    B, n_spatial, n_spectral, patch_dim = pred.shape

    mask_expanded = mask.unsqueeze(-1)  # (B, n_spatial, n_spectral, 1)

    # Check if within threshold
    diff = torch.abs(pred - target)
    correct = (diff < threshold).float()  # (B, n_spatial, n_spectral, patch_dim)

    # Apply mask (broadcasting!)
    masked_correct = correct * mask_expanded

    # Count correctly with patch_dim
    n_correct = masked_correct.sum()  # Total correct elements
    n_masked_tokens = mask.sum()  # Number of masked tokens
    n_total_elements = n_masked_tokens * patch_dim  # Total masked elements

    if n_masked_tokens == 0:
        return 0.0

    # Return as percentage
    accuracy = 100.0 * n_correct / n_total_elements

    return accuracy.item()


if __name__ == "__main__":
    # Test with known values
    print("Testing loss functions for TIFF format...")
    print("="*80)

    # Create test data
    B, n_spatial, n_spectral, patch_dim = 2, 256, 17, 256

    # Prediction and target that are close
    pred = torch.randn(B, n_spatial, n_spectral, patch_dim) * 0.5
    target = pred + 0.2 * torch.randn_like(pred)  # Small noise

    # Create mask (85% masked for TIFF)
    mask = torch.rand(B, n_spatial, n_spectral) < 0.85
    mask = mask.float()

    print(f"Test setup:")
    print(f"  Pred shape: {pred.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Mask ratio: {mask.mean():.2%}")

    # Calculate expected values manually
    n_masked_tokens = int(mask.sum().item())
    n_masked_elements = n_masked_tokens * patch_dim
    print(f"  Masked tokens: {n_masked_tokens}")
    print(f"  Masked elements: {n_masked_elements}")
    print()

    # Test loss
    loss = masked_l1_loss(pred, target, mask)
    print(f"Masked L1 loss: {loss.item():.4f}")
    print(f"  Expected range: 0.1-0.3 (small noise added)")
    print(f"    PASS" if 0.05 < loss.item() < 0.5 else "    FAIL")
    print()

    # Test accuracy
    acc = compute_reconstruction_accuracy(pred, target, mask, threshold=1.0)
    print(f"Reconstruction accuracy: {acc:.2f}%")
    print(f"  Expected range: 60-95% (small noise)")
    print(f"    PASS" if 50 < acc < 100 else "    FAIL")
    print()

    print("="*80)
