import numpy as np
import tifffile
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from skimage.segmentation import find_boundaries


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

    boundaries = find_boundaries(mask, mode="outer")

    overlay = image.copy()
    overlay[boundaries] = [255, 0, 0]

    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"{name}: Original Image")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(overlay)
    axes[row, 1].set_title(f"{name}: Ground Truth Overlay")
    axes[row, 1].axis("off")


plt.tight_layout()

output = Path("data/processed/inspection/boundary_overlay.png")
plt.savefig(output, dpi=180, bbox_inches="tight")

print(f"Saved visualization: {output}")

plt.show()