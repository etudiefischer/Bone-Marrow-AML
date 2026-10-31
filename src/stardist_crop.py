
import time
import json
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.segmentation import find_boundaries
from scipy.optimize import linear_sum_assignment


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_PATH = Path("results/cellpose/crop512/input_crop.tif")
GT_PATH = Path("results/cellpose/crop512/ground_truth_crop.tif")

COORD_PATH = Path(
    "results/cellpose/crop512/crop_ground_truth_coordinates.csv"
)

OUTPUT_DIR = Path("results/stardist/crop512")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Blue channel contains the visible nuclear fluorescence.
CHANNEL = 2

pipeline = tqdm(total=6, desc="StarDist pipeline", unit="stage")


# ============================================================
# 2. LOAD EXISTING CROP AND GROUND TRUTH
# ============================================================

print("\nLoading existing crop and ground truth...")

image = tifffile.imread(INPUT_PATH)
gt_mask = tifffile.imread(GT_PATH)

coordinates = pd.read_csv(COORD_PATH)

assert image.shape[:2] == gt_mask.shape

print("Image shape:", image.shape)
print("Ground truth shape:", gt_mask.shape)
print("Annotated cells:", len(coordinates))

pipeline.update(1)


# ============================================================
# 3. PREPARE SINGLE-CHANNEL IMAGE
# ============================================================

print("\nPreparing fluorescence channel...")

if image.ndim == 3:
    image_channel = image[:, :, CHANNEL]
else:
    image_channel = image

image_channel = image_channel.astype(np.float32)

# Normalize image intensities.
image_normalized = normalize(
    image_channel,
    1,
    99.8,
    axis=None
)

print("Selected channel:", CHANNEL)
print("Normalized image shape:", image_normalized.shape)

pipeline.update(1)


# ============================================================
# 4. LOAD PRETRAINED STARDIST MODEL
# ============================================================

print("\nLoading pretrained StarDist model...")

model = StarDist2D.from_pretrained(
    "2D_versatile_fluo"
)

print("Pretrained model loaded successfully.")

pipeline.update(1)


# ============================================================
# 5. RUN STARDIST INFERENCE
# ============================================================

print("\nStarting StarDist inference...")

start_time = time.time()

predicted_mask, details = model.predict_instances(
    image_normalized,
    prob_thresh=0.5,
    nms_thresh=0.4
)

inference_time = time.time() - start_time

print("\nStarDist inference completed!")
print(f"Inference time: {inference_time:.2f} seconds")

tifffile.imwrite(
    OUTPUT_DIR / "predicted_mask.tif",
    predicted_mask.astype(np.uint32)
)

pipeline.update(1)


# ============================================================
# 6. EVALUATE PIXEL AND INSTANCE SEGMENTATION
# ============================================================

print("\nCalculating evaluation metrics...")

gt_binary = gt_mask > 0
pred_binary = predicted_mask > 0

intersection = np.logical_and(
    gt_binary, pred_binary
).sum()

union = np.logical_or(
    gt_binary, pred_binary
).sum()

pixel_iou = intersection / union if union else 0.0

pixel_dice = (
    2 * intersection / (gt_binary.sum() + pred_binary.sum())
    if (gt_binary.sum() + pred_binary.sum()) > 0
    else 0.0
)

# Instance-level matching using IoU and Hungarian assignment.
gt_labels = np.unique(gt_mask)
gt_labels = gt_labels[gt_labels != 0]

pred_labels = np.unique(predicted_mask)
pred_labels = pred_labels[pred_labels != 0]

iou_matrix = np.zeros(
    (len(gt_labels), len(pred_labels)),
    dtype=np.float32
)

for i, gt_id in enumerate(gt_labels):
    gt_object = gt_mask == gt_id

    for j, pred_id in enumerate(pred_labels):
        pred_object = predicted_mask == pred_id

        inter = np.logical_and(gt_object, pred_object).sum()
        union_obj = np.logical_or(gt_object, pred_object).sum()

        if union_obj > 0:
            iou_matrix[i, j] = inter / union_obj

if iou_matrix.size:
    rows, cols = linear_sum_assignment(-iou_matrix)

    matched_ious = iou_matrix[rows, cols]
    valid_matches = matched_ious >= 0.5

    tp = int(valid_matches.sum())
    fp = len(pred_labels) - tp
    fn = len(gt_labels) - tp

    mean_matched_iou = (
        float(matched_ious[valid_matches].mean())
        if tp > 0 else 0.0
    )
else:
    tp = 0
    fp = len(pred_labels)
    fn = len(gt_labels)
    mean_matched_iou = 0.0

precision = tp / (tp + fp) if (tp + fp) else 0.0
recall = tp / (tp + fn) if (tp + fn) else 0.0

gt_count = len(gt_labels)
pred_count = len(pred_labels)

print("\n========== STARDIST RESULTS ==========")
print("Ground-truth objects:", gt_count)
print("Predicted objects:", pred_count)
print(f"Pixel IoU: {pixel_iou:.4f}")
print(f"Pixel Dice: {pixel_dice:.4f}")
print("Instance TP:", tp)
print("Instance FP:", fp)
print("Instance FN:", fn)
print(f"Instance Precision: {precision:.4f}")
print(f"Instance Recall: {recall:.4f}")
print(f"Mean matched IoU: {mean_matched_iou:.4f}")
print(f"Inference time: {inference_time:.2f} seconds")


# ============================================================
# 7. VISUALIZATION
# ============================================================

gt_boundaries = find_boundaries(gt_mask, mode="outer")
pred_boundaries = find_boundaries(predicted_mask, mode="outer")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image)
axes[0].set_title("Original Crop")

axes[1].imshow(image)
axes[1].contour(
    gt_boundaries,
    levels=[0.5],
    colors="lime",
    linewidths=0.5
)
axes[1].set_title("Ground Truth Boundaries")

axes[2].imshow(image)
axes[2].contour(
    pred_boundaries,
    levels=[0.5],
    colors="red",
    linewidths=0.5
)
axes[2].set_title("StarDist Predictions")

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
# 8. SAVE RESULTS
# ============================================================

results = {
    "model": "StarDist 2D_versatile_fluo",
    "input_crop": "512x512",
    "channel": CHANNEL,
    "ground_truth_objects": int(gt_count),
    "predicted_objects": int(pred_count),
    "pixel_iou": float(pixel_iou),
    "pixel_dice": float(pixel_dice),
    "instance_tp": tp,
    "instance_fp": fp,
    "instance_fn": fn,
    "instance_precision": float(precision),
    "instance_recall": float(recall),
    "mean_matched_iou": mean_matched_iou,
    "inference_time_seconds": float(inference_time)
}

with open(OUTPUT_DIR / "experiment_results.json", "w") as f:
    json.dump(results, f, indent=4)

pipeline.update(1)
pipeline.close()

print("\n========== EXPERIMENT COMPLETE ==========")
print("Results saved in:", OUTPUT_DIR)
print("Overlay: comparison_overlay.png")
print("Mask: predicted_mask.tif")
print("Metrics: experiment_results.json")
