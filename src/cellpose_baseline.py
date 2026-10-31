
import time
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt

from PIL import Image
from cellpose import models
from skimage.segmentation import find_boundaries


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

IMAGE_PATH = Path(
    "data/processed/inspection/nucleus/P51_FOV01_0_0.png"
)

GROUND_TRUTH_PATH = Path(
    "data/processed/inspection/nucleus/P51_FOV01_0_0_masks.tif"
)

OUTPUT_DIR = Path("results/cellpose/nucleus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cpsam_v2"


# --------------------------------------------------
# 2. Load input image and ground truth
# --------------------------------------------------

print("Loading input image...")

image = np.array(Image.open(IMAGE_PATH))
ground_truth = tifffile.imread(GROUND_TRUTH_PATH)

print("Image shape:", image.shape)
print("Ground truth shape:", ground_truth.shape)


# --------------------------------------------------
# 3. Load pretrained Cellpose model
# --------------------------------------------------

print("\nLoading pretrained Cellpose model...")

model = models.CellposeModel(
    gpu=False,
    pretrained_model=MODEL_NAME
)

print("Model loaded:", MODEL_NAME)


# --------------------------------------------------
# 4. Run inference
# --------------------------------------------------

print("\nRunning Cellpose inference...")

start_time = time.perf_counter()

predicted_mask, flows, styles = model.eval(
    image,
    channel_axis=-1,
    normalize=True,
    batch_size=8
)

inference_time = time.perf_counter() - start_time

print("Inference completed.")
print(f"Inference time: {inference_time:.2f} seconds")


# --------------------------------------------------
# 5. Calculate basic prediction statistics
# --------------------------------------------------

ground_truth_labels = np.unique(ground_truth)
predicted_labels = np.unique(predicted_mask)

gt_objects = len(ground_truth_labels) - 1
pred_objects = len(predicted_labels) - 1

print("\nSegmentation statistics")
print("-----------------------")
print("Ground truth objects:", gt_objects)
print("Predicted objects:", pred_objects)


# --------------------------------------------------
# 6. Save predicted instance mask
# --------------------------------------------------

mask_path = OUTPUT_DIR / "predicted_mask.tif"

tifffile.imwrite(
    mask_path,
    predicted_mask.astype(np.uint32)
)

print("\nSaved predicted mask:", mask_path)


# --------------------------------------------------
# 7. Generate prediction overlay
# --------------------------------------------------

boundaries = find_boundaries(
    predicted_mask,
    mode="outer"
)

overlay = image.copy()
overlay[boundaries] = [255, 0, 0]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image)
axes[0].set_title("Original Image")

axes[1].imshow(ground_truth, cmap="nipy_spectral")
axes[1].set_title("Ground Truth")

axes[2].imshow(overlay)
axes[2].set_title("Cellpose Prediction")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

figure_path = OUTPUT_DIR / "prediction_overlay.png"

plt.savefig(
    figure_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close()

print("Saved visualization:", figure_path)


# --------------------------------------------------
# 8. Save experiment metadata
# --------------------------------------------------

metadata_path = OUTPUT_DIR / "experiment.txt"

with open(metadata_path, "w") as file:
    file.write("Cellpose Baseline Experiment\n")
    file.write("============================\n")
    file.write(f"Model: {MODEL_NAME}\n")
    file.write("Version: 4.2.1.1\n")
    file.write("Device: CPU\n")
    file.write(f"Inference time: {inference_time:.2f} seconds\n")
    file.write(f"Ground truth objects: {gt_objects}\n")
    file.write(f"Predicted objects: {pred_objects}\n")
    file.write("Fine-tuning: No\n")

print("Experiment metadata saved.")
print("\nCellpose baseline experiment finished.")
