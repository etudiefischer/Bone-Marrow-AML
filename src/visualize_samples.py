import numpy as np
import matplotlib.pyplot as plt
import tifffile
from PIL import Image
from pathlib import Path

datasets = {
    "Nucleus": Path("data/processed/inspection/nucleus"),
    "Membrane": Path("data/processed/inspection/membrane"),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for row, (name, folder) in enumerate(datasets.items()):

    image_path = next(folder.glob("*.png"))
    mask_path = next(folder.glob("*_masks.tif"))

    image = np.array(Image.open(image_path))
    mask = tifffile.imread(mask_path)

    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"{name} - Original Image")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(mask, cmap="nipy_spectral")
    axes[row, 1].set_title(f"{name} - Ground Truth Mask")
    axes[row, 1].axis("off")

plt.tight_layout()
plt.savefig("data/processed/inspection/dataset_overview.png", dpi=150)
plt.show()