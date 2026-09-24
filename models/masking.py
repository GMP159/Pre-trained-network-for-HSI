"""
Random masking for self-supervised pre-training
"""

import torch
import torch.nn as nn


class TubeMasking(nn.Module):
    """
    Tube masking: masks entire spatial regions across all spectral bands.
    
    Args:
        mask_ratio: Ratio of spatial patches to mask (0.75 = 75%)
    """
    def __init__(self, mask_ratio: float = 0.85):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x, mask_token):
        """
        Args:
            x: (B, n_spatial, n_spectral, embed_dim)
            mask_token: (1, 1, 1, embed_dim) learnable mask token

        Returns:
            x_masked: Masked version of x
            mask: Binary mask (1=masked, 0=visible)
        """
        B, n_spatial, n_spectral, D = x.shape
        
        # 1. Decide which spatial patches to mask
        noise = torch.rand(B, n_spatial, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # 2. Create spatial mask
        num_masked = int(n_spatial * self.mask_ratio)
        mask_spatial = torch.zeros(B, n_spatial, device=x.device)
        mask_spatial[:, :num_masked] = 1  # 1 = masked
        
        # 3. Unshuffle to original order
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask_spatial = torch.gather(mask_spatial, dim=1, index=ids_restore)
        
        # 4. Expand to "Tubes": (B, n_spatial) -> (B, n_spatial, n_spectral)
        mask = mask_spatial.unsqueeze(-1).repeat(1, 1, n_spectral)
        
        # 5. Apply to tokens
        mask_expanded = mask.unsqueeze(-1)  # (B, n_spatial, n_spectral, 1)
        x_masked = x * (1 - mask_expanded) + mask_token * mask_expanded
        
        return x_masked, mask


class RandomMasking(nn.Module):
    """
    Random masking of tokens for masked reconstruction pre-training.
    Masks individual tokens randomly across both spatial and spectral dimensions.

    Args:
        mask_ratio: Ratio of tokens to mask (0.75 = 75%)
    """

    def __init__(self, mask_ratio: float = 0.75):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, x, mask_token):
        B, n_spatial, n_spectral, embed_dim = x.shape
        N = n_spatial * n_spectral

        x_flat = x.reshape(B, N, embed_dim)

        n_masked = int(N * self.mask_ratio)
        mask = torch.zeros(B, N, device=x.device)

        for i in range(B):
            perm = torch.randperm(N, device=x.device)
            mask[i, perm[:n_masked]] = 1

        mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)

        # FIX: reshape 4D mask_token -> 3D before expand
        mask_token_flat = mask_token.reshape(1, 1, embed_dim)
        mask_token_expanded = mask_token_flat.expand(B, N, embed_dim)

        x_masked_flat = x_flat * (1 - mask_expanded) + mask_token_expanded * mask_expanded

        x_masked = x_masked_flat.reshape(B, n_spatial, n_spectral, embed_dim)
        mask_reshaped = mask.reshape(B, n_spatial, n_spectral)

        return x_masked, mask_reshaped