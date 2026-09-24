import torch
import rasterio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter


class CoffeePatchDataset(Dataset):
    """
    Dataset for labeled coffee-bean HSI patches.

    Expects:
      data/patches_64x64_geotiff_2/
        Arabica_Robusta_Immature_WTQ2_SWIR_arabica/
        Arabica_Robusta_Immature_WTQ2_SWIR_immature/
        Arabica_Robusta_Immature_WTQ2_SWIR_robusta/
    """

    def __init__(self, patch_dir, split="train", test_size=0.2,
                 random_state=42, transform=None):
        self.patch_dir = Path(patch_dir)
        self.transform = transform

        self.class_map = {
            "Arabica_Robusta_Immature_WTQ2_SWIR_arabica": 0,
            "Arabica_Robusta_Immature_WTQ2_SWIR_immature": 1,
            "Arabica_Robusta_Immature_WTQ2_SWIR_robusta": 2,
        }

        all_samples = []
        for class_name, class_idx in self.class_map.items():
            class_dir = self.patch_dir / class_name
            if not class_dir.is_dir():
                print(f"WARNING: class dir not found: {class_dir}")
                continue

            for tif_file in class_dir.rglob("*.tif*"):
                all_samples.append({
                    "path": str(tif_file),
                    "class": class_idx,
                })

        if len(all_samples) == 0:
            raise RuntimeError(f"No TIFF files found under {self.patch_dir}")

        labels = np.array([s["class"] for s in all_samples])
        train_idx, test_idx = train_test_split(
            np.arange(len(all_samples)),
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        if split == "train":
            self.samples = [all_samples[i] for i in train_idx]
        elif split == "test":
            self.samples = [all_samples[i] for i in test_idx]
        else:
            raise ValueError(f"Unknown split: {split}")

        class_counts = Counter([s["class"] for s in self.samples])
        print(f"{split.upper()} set: {len(self.samples)} patches")
        print(f"  Class distribution: {dict(sorted(class_counts.items()))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with rasterio.open(sample["path"]) as src:
            data = src.read()  # (257, 64, 64)

        spectral = data[:256].astype(np.float32)  # (256, 64, 64)
        spectral = (spectral - spectral.mean()) / (spectral.std() + 1e-8)

        spectral = torch.from_numpy(spectral)
        label = torch.tensor(sample["class"], dtype=torch.long)

        if self.transform:
            spectral = self.transform(spectral)

        return spectral, label
