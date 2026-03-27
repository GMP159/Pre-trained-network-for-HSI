"""
3D Patch Embedding for Hyperspectral Images
Converts 64x64x257 patches (or 32x32x257) into tokens for transformer
Adapted for TIFF files with 257 bands
"""

import torch
import torch.nn as nn

class PatchEmbedding3D(nn.Module):
    """
    3D patchification with BLOCKWISE spectral embedding (paper's key contribution).
    
    KEY DIFFERENCE from standard ViT:
    - Each spectral block gets its OWN linear projection
    - This accounts for different wavelength characteristics (VIS, NIR, SWIR)
    
    Input: (B, H, W, C) = (B, 64, 64, 256)
    Output: (B, n_spatial, n_spectral, embed_dim)
    """
    
    def __init__(
        self,
        img_size: int = 64,
        patch_h: int = 2,
        patch_w: int = 2,
        patch_c: int = 16,
        in_channels: int = 256,
        embed_dim: int = 128
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_c = patch_c
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.n_spatial_h = img_size // patch_h # 64 // 4 = 16
        self.n_spatial_w = img_size // patch_w # 64 // 4 = 16
        self.n_spatial = self.n_spatial_h * self.n_spatial_w # 16 * 16 = 256
        self.n_spectral = in_channels // patch_c  # 256 // 16 = 16
        
        # Patch dimension
        self.patch_dim = patch_h * patch_w * patch_c  # 4 * 4 * 16 = 256
        
        # ===== BLOCKWISE SPECTRAL EMBEDDING (Paper's Key Contribution) =====
        # CHANGED: Separate projection for EACH spectral block
        self.projections = nn.ModuleList([
            nn.Linear(self.patch_dim, embed_dim)
            for _ in range(self.n_spectral)
        ])
        
    def forward(self, x):
        """
        Args:
            x: (B, H, W, C) = (B, 64, 64, 256) or (B, 32, 32, 256)
        Returns:
            patches: (B, n_spatial, n_spectral, embed_dim)
        """
        B, H, W, C = x.shape
        
        assert H == self.img_size and W == self.img_size, \
            f"Input size {H}x{W} doesn't match expected {self.img_size}x{self.img_size}"
        assert C == self.in_channels, \
            f"Input channels {C} doesn't match expected {self.in_channels}"
        
        # Reshape to patches
        # (B, H, W, C) -> (B, n_h, patch_h, n_w, patch_w, n_c, patch_c)
        x = x.reshape(
            B,
            self.n_spatial_h, self.patch_h,
            self.n_spatial_w, self.patch_w,
            self.n_spectral, self.patch_c
        )
        
        # Rearrange dimensions
        # -> (B, n_h, n_w, n_c, patch_h, patch_w, patch_c)
        #(B, 16, 16, 16, 4, 4, 16)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)
        
        # Flatten each patch
        # -> (B, n_spatial, n_spectral, patch_h*patch_w*patch_c)
        #(B, 256, 16, 256)
        x = x.reshape(B, self.n_spatial, self.n_spectral, self.patch_dim)
        
        # ===== BLOCKWISE PROJECTION (NEW) =====
        # Apply separate projection for each spectral block
        embedded_patches = []
        for spectral_idx in range(self.n_spectral):
            # Extract patches for this spectral block
            spectral_block = x[:, :, spectral_idx, :]  # (B, n_spatial, patch_dim)
            
            # Apply block-specific projection
            embedded = self.projections[spectral_idx](spectral_block)  # (B, n_spatial, embed_dim)
            
            embedded_patches.append(embedded)
        
        # Stack back: (B, n_spatial, n_spectral, embed_dim)
        x = torch.stack(embedded_patches, dim=2)
        
        return x
    
    def get_num_patches(self):
        """Returns (n_spatial, n_spectral)"""
        return self.n_spatial, self.n_spectral



class SimplePatchEmbedding(nn.Module):
    """
    Simpler 3D patch embedding that treats the full image as one big patch.
    Useful when image size is already the patch size (e.g., 64x64).
    """

    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 257,
        embed_dim: int = 128
    ):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Flatten the entire image
        self.patch_dim = img_size * img_size * in_channels
        
        # Single projection from flattened image to embed_dim
        self.projection = nn.Linear(self.patch_dim, embed_dim)
        
        # We'll return this as a single token
        self.n_spatial = 1
        self.n_spectral = 1

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)

        Returns:
            patches: (B, 1, 1, embed_dim)
        """
        B, H, W, C = x.shape

        assert H == self.img_size and W == self.img_size, \
            f"Input size {H}x{W} doesn't match expected {self.img_size}x{self.img_size}"
        assert C == self.in_channels, \
            f"Input channels {C} doesn't match expected {self.in_channels}"

        # Flatten
        x = x.reshape(B, -1)  # (B, H*W*C)

        # Project
        x = self.projection(x)  # (B, embed_dim)

        # Reshape to (B, 1, 1, embed_dim) for consistency with 3D version
        x = x.unsqueeze(1).unsqueeze(1)

        return x

    def get_num_patches(self):
        """Returns (n_spatial, n_spectral)"""
        return self.n_spatial, self.n_spectral
