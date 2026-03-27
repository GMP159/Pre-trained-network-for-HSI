import torch
import rasterio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter

class ApplePatchDataset(Dataset):
    """
    Dataset for labeled apple HSI patches.
    Uses stratified patch-level split to ensure all classes appear in
    both train and test. Note: patches from the same tree may appear
    in both splits (acceptable trade-off for class coverage).
    """
    def __init__(self, patch_dir, split='train', test_size=0.2, random_state=42, transform=None):
        self.patch_dir = Path(patch_dir)
        self.transform = transform

        # Load all patches with metadata
        all_samples = []
        for tif_file in sorted(self.patch_dir.rglob("*.tif")):
            with rasterio.open(tif_file) as src:
                meta = src.tags(ns='METADATA')
                filename = tif_file.stem
                parts = filename.split('_')
                tree_id = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                class_label = int(meta.get('class_label', 0))
                all_samples.append({
                    'path': str(tif_file),
                    'tree_id': tree_id,
                    'class': class_label - 1  # Convert 1-4 to 0-3
                })

        all_labels = [s['class'] for s in all_samples]
        train_idx, test_idx = train_test_split(
            range(len(all_samples)),
            test_size=test_size,
            random_state=random_state,
            stratify=all_labels
        )

        if split == 'train':
            self.samples = [all_samples[i] for i in train_idx]
        else:
            self.samples = [all_samples[i] for i in test_idx]

        unique_trees = len(set(s['tree_id'] for s in self.samples))
        print(f"{split.upper()} set: {len(self.samples)} patches from {unique_trees} trees")
        class_counts = Counter(s['class'] for s in self.samples)
        print(f"  Class distribution: {dict(sorted(class_counts.items()))}")

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
