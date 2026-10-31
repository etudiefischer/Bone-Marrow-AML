
import time
import json
import numpy as np
import tifffile
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import find_boundaries


# CONFIGURATION
SIZE = 1024

INPUT = Path("results/cellpose/crop1024/input_crop.tif")
GT_PATH = Path("results/cellpose/crop1024/ground_truth_crop.tif")

OUT = Path("results/stardist/crop1024")
OUT.mkdir(parents=True, exist_ok=True)

CHANNEL = 2

progress = tqdm(total=5, desc="StarDist 1024", unit="stage")


# LOAD SAME CROP
image = tifffile.imread(INPUT)
gt = tifffile.imread(GT_PATH)

assert image.shape[:2] == gt.shape
assert gt.shape == (SIZE, SIZE)

print("Image shape:", image.shape)
print("Ground truth shape:", gt.shape)

progress.update(1)


# PREPARE BLUE CHANNEL
image_channel = image[:, :, CHANNEL].astype(np.float32)

image_normalized = normalize(
    image_channel,
    1,
    99.8,
    axis=None
)

progress.update(1)


# LOAD PRETRAINED MODEL
print("\nLoading pretrained StarDist...")

model = StarDist2D.from_pretrained(
    "2D_versatile_fluo"
)

progress.update(1)


# INFERENCE
print("\nRunning StarDist inference...")

start = time.time()

pred, details = model.predict_instances(
    image_normalized,
    prob_thresh=0.479071,
    nms_thresh=0.3
)

inference_time = time.time() - start

tifffile.imwrite(
    OUT / "predicted_mask.tif",
    pred.astype(np.uint32)
)

print("Inference time:", round(inference_time, 2), "seconds")

progress.update(1)


# PIXEL METRICS
gt_binary = gt > 0
pred_binary = pred > 0

intersection = np.logical_and(gt_binary, pred_binary).sum()
union = np.logical_or(gt_binary, pred_binary).sum()

pixel_iou = intersection / union if union else 0

pixel_dice = (
    2 * intersection / (gt_binary.sum() + pred_binary.sum())
    if (gt_binary.sum() + pred_binary.sum()) else 0
)


# INSTANCE METRICS
gt_labels = np.unique(gt)
gt_labels = gt_labels[gt_labels != 0]

pred_labels = np.unique(pred)
pred_labels = pred_labels[pred_labels != 0]

iou_matrix = np.zeros(
    (len(gt_labels), len(pred_labels)),
    dtype=np.float32
)

for i, gid in enumerate(gt_labels):
    g = gt == gid

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
gt_bound = find_boundaries(gt, mode="outer")
pred_bound = find_boundaries(pred, mode="outer")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

axes[0].imshow(image)
axes[0].set_title("Original 1024 Crop")

axes[1].imshow(image)
axes[1].contour(gt_bound, levels=[0.5], colors="lime", linewidths=0.5)
axes[1].set_title("Ground Truth")

axes[2].imshow(image)
axes[2].contour(pred_bound, levels=[0.5], colors="red", linewidths=0.5)
axes[2].set_title("StarDist Predictions")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT / "comparison_overlay.png", dpi=200)
plt.close()


# SAVE METRICS
results = {
    "model": "StarDist 2D_versatile_fluo",
    "crop_size": SIZE,
    "channel": CHANNEL,
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

print("\n========== STARDIST 1024 RESULTS ==========")
for key, value in results.items():
    print(f"{key}: {value}")

print("\nSaved in:", OUT)
