
import time
import json
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from cellpose import models
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import find_boundaries


# CONFIGURATION
SIZE = 1024
IMAGE_PATH = Path("data/processed/inspection/nucleus/P51_FOV01_0_0.png")
GT_PATH = Path("data/processed/inspection/nucleus/P51_FOV01_0_0_masks.tif")
CSV_PATH = Path("results/spatial/nucleus_coordinates.csv")

OUT = Path("results/cellpose/crop1024")
OUT.mkdir(parents=True, exist_ok=True)

progress = tqdm(total=6, desc="Cellpose 1024", unit="stage")


# LOAD DATA
image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
gt = tifffile.imread(GT_PATH)
coords = pd.read_csv(CSV_PATH)

assert image.shape[:2] == gt.shape

progress.update(1)


# CENTRAL CROP
h, w = gt.shape
y0 = (h - SIZE) // 2
x0 = (w - SIZE) // 2

img_crop = image[y0:y0+SIZE, x0:x0+SIZE]
gt_crop = gt[y0:y0+SIZE, x0:x0+SIZE]

assert img_crop.shape[:2] == (SIZE, SIZE)

tifffile.imwrite(OUT / "input_crop.tif", img_crop)
tifffile.imwrite(OUT / "ground_truth_crop.tif", gt_crop)

print("Crop:", img_crop.shape)
print("Offsets:", x0, y0)

progress.update(1)


# SPATIAL COORDINATES
crop_coords = coords[
    (coords.centroid_x >= x0) &
    (coords.centroid_x < x0 + SIZE) &
    (coords.centroid_y >= y0) &
    (coords.centroid_y < y0 + SIZE)
].copy()

crop_coords["original_x"] = crop_coords["centroid_x"]
crop_coords["original_y"] = crop_coords["centroid_y"]

crop_coords["centroid_x"] -= x0
crop_coords["centroid_y"] -= y0

crop_coords.to_csv(
    OUT / "ground_truth_coordinates.csv",
    index=False
)

print("Ground-truth cells in crop:", len(crop_coords))

progress.update(1)


# LOAD PRETRAINED CELLPOSE
print("\nLoading Cellpose...")
model = models.CellposeModel(
    gpu=True,
    pretrained_model="cpsam_v2"
)

progress.update(1)


# INFERENCE
print("\nRunning Cellpose inference...")
start = time.time()

pred, flows, styles = model.eval(
    img_crop,
    channel_axis=-1,
    normalize=True,
    batch_size=4
)

inference_time = time.time() - start

tifffile.imwrite(
    OUT / "predicted_mask.tif",
    pred.astype(np.uint32)
)

print("Inference time:", round(inference_time, 2), "seconds")

progress.update(1)


# PIXEL METRICS
gt_binary = gt_crop > 0
pred_binary = pred > 0

intersection = np.logical_and(gt_binary, pred_binary).sum()
union = np.logical_or(gt_binary, pred_binary).sum()

pixel_iou = intersection / union if union else 0

pixel_dice = (
    2 * intersection / (gt_binary.sum() + pred_binary.sum())
    if (gt_binary.sum() + pred_binary.sum()) else 0
)


# INSTANCE METRICS
gt_labels = np.unique(gt_crop)
gt_labels = gt_labels[gt_labels != 0]

pred_labels = np.unique(pred)
pred_labels = pred_labels[pred_labels != 0]

iou_matrix = np.zeros(
    (len(gt_labels), len(pred_labels)),
    dtype=np.float32
)

for i, gid in enumerate(gt_labels):
    g = gt_crop == gid

    for j, pid in enumerate(pred_labels):
        p = pred == pid
        inter = np.logical_and(g, p).sum()
        uni = np.logical_or(g, p).sum()

        if uni:
            iou_matrix[i, j] = inter / uni

if iou_matrix.size:
    rows, cols = linear_sum_assignment(-iou_matrix)
    matched = iou_matrix[rows, cols]
    valid = matched >= 0.5
    tp = int(valid.sum())
    mean_iou = float(matched[valid].mean()) if tp else 0.0
else:
    tp = 0
    mean_iou = 0.0

fp = len(pred_labels) - tp
fn = len(gt_labels) - tp

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)


# VISUALIZATION
gt_bound = find_boundaries(gt_crop, mode="outer")
pred_bound = find_boundaries(pred, mode="outer")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

axes[0].imshow(img_crop)
axes[0].set_title("Original 1024 Crop")

axes[1].imshow(img_crop)
axes[1].contour(gt_bound, levels=[0.5], colors="lime", linewidths=0.5)
axes[1].set_title("Ground Truth")

axes[2].imshow(img_crop)
axes[2].contour(pred_bound, levels=[0.5], colors="red", linewidths=0.5)
axes[2].set_title("Cellpose Predictions")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT / "comparison_overlay.png", dpi=200)
plt.close()


# SAVE METRICS
results = {
    "model": "Cellpose cpsam_v2",
    "crop_size": SIZE,
    "x_offset": x0,
    "y_offset": y0,
    "ground_truth_objects": int(len(gt_labels)),
    "predicted_objects": int(len(pred_labels)),
    "pixel_iou": float(pixel_iou),
    "pixel_dice": float(pixel_dice),
    "instance_tp": tp,
    "instance_fp": fp,
    "instance_fn": fn,
    "instance_precision": float(precision),
    "instance_recall": float(recall),
    "instance_f1": float(f1),
    "mean_matched_iou": mean_iou,
    "inference_time_seconds": inference_time
}

with open(OUT / "experiment_results.json", "w") as f:
    json.dump(results, f, indent=4)

progress.update(1)
progress.close()

print("\n========== CELLPOSE 1024 RESULTS ==========")
for key, value in results.items():
    print(f"{key}: {value}")

print("\nSaved in:", OUT)
