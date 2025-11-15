# =============================================================
#  HSI File Visual Inspector: Choose Cropping and Mask Settings
# =============================================================

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.data.data_utils import load_hsi_envi
from src.data.background_filter import filter_background
from src.data.band_alignment import align_spectral_bands, normalize_data_type

# --------- EDIT THIS: List all folders containing HSI files ----------
input_folders = [
    r"E:\Data_Master\01Rohdaten",
    r"E:\Data_Master\03Kalibrierte_Rohdaten",
    r"E:\Data_Master\PaperTrails",
    r"E:\Data_Master\SCherben",
    r"E:\Data_Master\Sugar",
]

target_bands = 256  # What you want in the end
normalize_uint16 = True

# --------- List all files ----------
image_list = []
for data_root in input_folders:
    for fn in os.listdir(data_root):
        if fn.endswith(".hdr"):
            image_list.append(os.path.join(data_root, fn))
print(f"Found {len(image_list)} HSI images.")


# --------- Helper for showing HSI statistics and slices ----------
def show_hsi_visuals(data, bands=None, title=""):
    H, W, C = data.shape
    median_band = C // 2
    plt.figure(figsize=(16, 6))

    # Mean image (all bands)
    plt.subplot(1, 3, 1)
    plt.imshow(data.mean(axis=2), cmap="gray")
    plt.title(f"{title}\nMean (all bands)")

    # Visualize a single mid band
    plt.subplot(1, 3, 2)
    plt.imshow(data[:, :, median_band], cmap="gray")
    plt.title(f"Band #{median_band}")

    # Spectral signature at center pixel
    plt.subplot(1, 3, 3)
    spectrum = data[H//2, W//2, :]
    plt.plot(np.arange(C), spectrum, marker='.')
    plt.title("Center Pixel Spectrum")
    plt.xlabel("Band")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()


# --------- Main inspection loop ----------
for idx, img_path in enumerate(image_list):
    print("-"*80)
    print(f"Inspecting file {idx + 1} / {len(image_list)}: {img_path}")
    hdr_path = img_path
    img_path_explicit = img_path.replace('.hdr', '.img')
    data, meta = load_hsi_envi(hdr_path, img_path_explicit)

    print(f"Shape: {data.shape}, dtype: {data.dtype}, range: [{data.min()}, {data.max()}]")
    print(f"Metadata keys: {list(meta.keys())}")

    # Normalize if needed
    if data.dtype == np.uint16 or (normalize_uint16 and data.max() > 1000):
        print("Normalizing uint16 HSI to [0, 1]")
        data = data / 65535.0

    # Show stats before cropping
    show_hsi_visuals(data, title="Original full image")

    # --------- YOU: Choose Crop Here, then Uncomment ----------
    # Example:
    # top, bottom, left, right = 150, 650, 10, 290
    crop_region = None  # <- Set to None to skip cropping for first visual check
    # crop_region = (top, bottom, left, right)  # Fill in and uncomment after first visual!

    if crop_region is not None:
        t, b, l, r = crop_region
        data_cropped = data[t:b, l:r, :]
        print(f"Cropped to: {data_cropped.shape} from region top={t}, bottom={b}, left={l}, right={r}")
        show_hsi_visuals(data_cropped, title=f"Cropped, region {crop_region}")
    else:
        data_cropped = data

    # Align bands if needed
    if data_cropped.shape[2] != target_bands:
        print(f"Aligning bands: {data_cropped.shape[2]} → {target_bands}")
        data_cropped = align_spectral_bands(data_cropped, data_cropped.shape[2], target_bands, method='crop')

    # --------- Try Different Mask Thresholds -----
    print("Try several variance thresholds on the mask (typical: 0.01~0.15)")
    for vthresh in [0.01, 0.03, 0.05, 0.1, 0.15]:
        filtered, mask, stats = filter_background(
            data_cropped, crop_region=None,
            variance_threshold=vthresh,
            min_useful_ratio=0.3,
            fill_background=True
        )
        print(f"  Threshold {vthresh:.3f}: useful pixels: {mask.sum()/mask.size*100:.2f}%")
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask: variance_threshold={vthresh}")
        plt.show()

    print("NOTE: Write down (top,bottom,left,right) and best variance_threshold for this file/domain.")
    input("Press Enter for the next image...")
