import numpy as np
import tifffile
from pathlib import Path

datasets = {
    "Nucleus": Path("data/processed/inspection/nucleus"),
    "Membrane": Path("data/processed/inspection/membrane"),
}

for name, folder in datasets.items():

    mask_path = next(folder.glob("*_masks.tif"))
    mask = tifffile.imread(mask_path)

    labels, counts = np.unique(mask, return_counts=True)

    # Exclude background (label 0)
    object_counts = counts[labels != 0]

    print(f"\n{'=' * 40}")
    print(f"{name.upper()} MASK ANALYSIS")
    print("=" * 40)

    print("Number of objects:", len(object_counts))
    print("Total foreground pixels:", object_counts.sum())

    if len(object_counts):
        print("Smallest object:", object_counts.min(), "pixels")
        print("Largest object:", object_counts.max(), "pixels")
        print("Average object size:", round(object_counts.mean(), 2))
        print("Median object size:", np.median(object_counts))