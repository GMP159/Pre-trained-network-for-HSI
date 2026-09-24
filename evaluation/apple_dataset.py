import torch
import rasterio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from collections import Counter


class ApplePatchDataset(Dataset):
    def __init__(self, patch_dir, split='train', test_size=0.2,
                 random_state=42, transform=None):
        self.patch_dir = Path(patch_dir)
        self.transform = transform

        split_dir = self.patch_dir / split  # .../tiff_patches/train or /test

        all_samples = []
        for tif_file in sorted(split_dir.rglob("*.tif")):
            folder_name = tif_file.parent.name.lower()  # "class_03"
            try:
                class_label = int(folder_name.split('_')[-1]) - 1  # 0-indexed
            except ValueError:
                class_label = -1

            filename = tif_file.stem
            parts = filename.split('_')
            tree_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

            all_samples.append({
                'path': str(tif_file),
                'tree_id': tree_id,
                'class': class_label
            })

        self.samples = all_samples

        unique_trees = len(set(s['tree_id'] for s in self.samples))
        print(f"{split.upper()} set: {len(self.samples)} patches from {unique_trees} trees")
        class_counts = Counter(s['class'] for s in self.samples)
        print(f"  Class distribution: {dict(sorted(class_counts.items()))}")

        if -1 in class_counts:
            raise RuntimeError(
                f"Unknown class folders found! Check folder names in {split_dir}. "
                f"Expected: class_01 ... class_15"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with rasterio.open(sample['path']) as src:
            data = src.read()  # (257, 64, 64)

        spectral = data[:256].astype(np.float32)  # (256, 64, 64)
        spectral = (spectral - spectral.mean()) / (spectral.std() + 1e-8)

        spectral = torch.from_numpy(spectral)
        label = torch.tensor(sample['class'], dtype=torch.long)

        if self.transform:
            spectral = self.transform(spectral)

        return spectral, label