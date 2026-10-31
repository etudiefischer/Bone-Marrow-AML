
import time
import json
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from cellpose import models
from skimage.segmentation import find_boundaries
from sklearn.metrics import jaccard_score


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

IMAGE_PATH = Path(
    "data/processed/inspection/nucleus/P51_FOV01_0_0.png"
)

MASK_PATH = Path(
    "data/processed/inspection/nucleus/P51_FOV01_0_0_masks.tif"
)

COORDINATES_PATH = Path(
    "results/spatial/nucleus_coordinates.csv"
)

OUTPUT_DIR = Path("results/cellpose/crop512")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CROP_SIZE = 512

# Use Apple Silicon GPU through MPS where supported.
USE_GPU = True


# ============================================================
# 2. LOAD IMAGE, GROUND TRUTH AND SPATIAL DATA
# ============================================================

pipeline = tqdm(total=7, desc="Cellpose pipeline", unit="stage")

print("\nLoading original image and ground-truth mask...")

from PIL import Image

# Original microscopy image is PNG
image = np.array(
    Image.open(IMAGE_PATH).convert("RGB")
)

# Ground-truth mask is TIFF
ground_truth = tifffile.imread(MASK_PATH)

coordinates = pd.read_csv(COORDINATES_PATH)

assert image.shape[:2] == ground_truth.shape, (
    "Image and ground-truth dimensions do not match."
)

height, width = ground_truth.shape

print("Original image shape:", image.shape)
print("Ground-truth shape:", ground_truth.shape)
print("Annotated cells:", len(coordinates))

pipeline.update(1)


# ============================================================
# 3. EXTRACT CENTRAL 512 x 512 CROP
# ============================================================

print("\nExtracting central region...")

if height < CROP_SIZE or width < CROP_SIZE:
    raise ValueError("Crop size is larger than the original image.")

# Calculate crop offsets.
y_offset = (height - CROP_SIZE) // 2
x_offset = (width - CROP_SIZE) // 2

# Apply exactly the same crop to image and mask.
image_crop = image[
    y_offset:y_offset + CROP_SIZE,
    x_offset:x_offset + CROP_SIZE
]

gt_crop = ground_truth[
    y_offset:y_offset + CROP_SIZE,
    x_offset:x_offset + CROP_SIZE
]

print("Crop shape:", image_crop.shape)
print("X offset:", x_offset)
print("Y offset:", y_offset)

# Save crop files.
tifffile.imwrite(
    OUTPUT_DIR / "input_crop.tif",
    image_crop
)

tifffile.imwrite(
    OUTPUT_DIR / "ground_truth_crop.tif",
    gt_crop
)

pipeline.update(1)


# ============================================================
# 4. FILTER SPATIAL COORDINATES FOR THE CROP
# ============================================================

print("\nFiltering spatial coordinates...")

crop_coordinates = coordinates[
    (coordinates["centroid_x"] >= x_offset) &
    (coordinates["centroid_x"] < x_offset + CROP_SIZE) &
    (coordinates["centroid_y"] >= y_offset) &
    (coordinates["centroid_y"] < y_offset + CROP_SIZE)
].copy()

# Preserve original image coordinates.
crop_coordinates["original_x"] = crop_coordinates["centroid_x"]
crop_coordinates["original_y"] = crop_coordinates["centroid_y"]

# Add crop-relative coordinates.
crop_coordinates["centroid_x"] -= x_offset
crop_coordinates["centroid_y"] -= y_offset

crop_coordinates.to_csv(
    OUTPUT_DIR / "crop_ground_truth_coordinates.csv",
    index=False
)

print("Ground-truth cells inside crop:", len(crop_coordinates))

pipeline.update(1)


# ============================================================
# 5. LOAD PRETRAINED CELLPOSE MODEL
# ============================================================

print("\nLoading pretrained Cellpose model...")

model = models.CellposeModel(
    gpu=USE_GPU,
    pretrained_model="cpsam_v2"
)

print("Pretrained model loaded: cpsam_v2")

pipeline.update(1)


# ============================================================
# 6. RUN CELLPOSE INFERENCE
# ============================================================

print("\nStarting Cellpose inference...")
print("Processing one 512 x 512 crop.")

inference_start = time.time()

predicted_mask, flows, styles = model.eval(
    image_crop,
    channel_axis=-1,
    normalize=True,
    batch_size=4
)

inference_time = time.time() - inference_start

print("\nCellpose inference completed!")
print(f"Inference time: {inference_time:.2f} seconds")

tifffile.imwrite(
    OUTPUT_DIR / "predicted_mask.tif",
    predicted_mask.astype(np.uint32)
)

pipeline.update(1)


# ============================================================
# 7. EVALUATION AND VISUALIZATION
# ============================================================

print("\nCalculating segmentation metrics...")

# Convert instance masks to binary foreground masks.
gt_binary = gt_crop > 0
pred_binary = predicted_mask > 0

intersection = np.logical_and(gt_binary, pred_binary).sum()
union = np.logical_or(gt_binary, pred_binary).sum()

pixel_iou = (
    intersection / union if union > 0 else 0.0
)

pixel_dice = (
    2 * intersection / (gt_binary.sum() + pred_binary.sum())
    if (gt_binary.sum() + pred_binary.sum()) > 0
    else 0.0
)

gt_count = len(np.unique(gt_crop)) - 1
pred_count = len(np.unique(predicted_mask)) - 1

print("\n========== RESULTS ==========")
print("Ground-truth objects:", gt_count)
print("Predicted objects:", pred_count)
print(f"Pixel IoU: {pixel_iou:.4f}")
print(f"Pixel Dice: {pixel_dice:.4f}")
print(f"Inference time: {inference_time:.2f} seconds")

# Save a visual comparison.
gt_boundaries = find_boundaries(gt_crop, mode="outer")
pred_boundaries = find_boundaries(predicted_mask, mode="outer")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image_crop)
axes[0].set_title("Original Crop")

axes[1].imshow(image_crop)
axes[1].contour(
    gt_boundaries,
    levels=[0.5],
    colors="lime",
    linewidths=0.5
)
axes[1].set_title("Ground Truth Boundaries")

axes[2].imshow(image_crop)
axes[2].contour(
    pred_boundaries,
    levels=[0.5],
    colors="red",
    linewidths=0.5
)
axes[2].set_title("Cellpose Predictions")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "comparison_overlay.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. SAVE EXPERIMENT SUMMARY
# ============================================================

results = {
    "model": "Cellpose cpsam_v2",
    "crop_size": CROP_SIZE,
    "x_offset": x_offset,
    "y_offset": y_offset,
    "ground_truth_objects": int(gt_count),
    "predicted_objects": int(pred_count),
    "pixel_iou": float(pixel_iou),
    "pixel_dice": float(pixel_dice),
    "inference_time_seconds": float(inference_time),
    "gpu_requested": USE_GPU
}

with open(OUTPUT_DIR / "experiment_results.json", "w") as f:
    json.dump(results, f, indent=4)

pipeline.update(1)
pipeline.close()

print("\n========== EXPERIMENT COMPLETE ==========")
print("Results saved in:", OUTPUT_DIR)
print("Overlay: comparison_overlay.png")
print("Predicted mask: predicted_mask.tif")
print("Metrics: experiment_results.json")
print("Spatial CSV: crop_ground_truth_coordinates.csv")
