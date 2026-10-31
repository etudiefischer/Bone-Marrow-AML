
import json
import numpy as np
import tifffile

from pathlib import Path
from scipy.optimize import linear_sum_assignment


# ============================================================
# 1. LOAD MASKS
# ============================================================

GT_PATH = Path(
    "results/cellpose/crop512/ground_truth_crop.tif"
)

PRED_PATH = Path(
    "results/cellpose/crop512/predicted_mask.tif"
)

OUTPUT_PATH = Path(
    "results/cellpose/crop512/instance_metrics.json"
)

gt_mask = tifffile.imread(GT_PATH)
pred_mask = tifffile.imread(PRED_PATH)

assert gt_mask.shape == pred_mask.shape, (
    "Ground truth and prediction dimensions differ."
)

print("Ground truth shape:", gt_mask.shape)
print("Prediction shape:", pred_mask.shape)


# ============================================================
# 2. EXTRACT INSTANCE LABELS
# ============================================================

gt_labels = np.unique(gt_mask)
gt_labels = gt_labels[gt_labels != 0]

pred_labels = np.unique(pred_mask)
pred_labels = pred_labels[pred_labels != 0]

print("\nGround-truth objects:", len(gt_labels))
print("Predicted objects:", len(pred_labels))


# ============================================================
# 3. CALCULATE INSTANCE IoU MATRIX
# ============================================================

iou_matrix = np.zeros(
    (len(gt_labels), len(pred_labels)),
    dtype=np.float32
)

for i, gt_id in enumerate(gt_labels):

    gt_object = gt_mask == gt_id

    for j, pred_id in enumerate(pred_labels):

        pred_object = pred_mask == pred_id

        intersection = np.logical_and(
            gt_object, pred_object
        ).sum()

        union = np.logical_or(
            gt_object, pred_object
        ).sum()

        if union > 0:
            iou_matrix[i, j] = intersection / union


# ============================================================
# 4. HUNGARIAN MATCHING
# ============================================================

if iou_matrix.size > 0:

    rows, cols = linear_sum_assignment(-iou_matrix)

    matched_ious = iou_matrix[rows, cols]

    # A match is valid only when IoU >= 0.5.
    valid_matches = matched_ious >= 0.5

    tp = int(valid_matches.sum())

else:
    matched_ious = np.array([])
    valid_matches = np.array([], dtype=bool)
    tp = 0

fp = len(pred_labels) - tp
fn = len(gt_labels) - tp


# ============================================================
# 5. CALCULATE METRICS
# ============================================================

precision = tp / (tp + fp) if tp + fp else 0.0

recall = tp / (tp + fn) if tp + fn else 0.0

f1_score = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

mean_matched_iou = (
    float(matched_ious[valid_matches].mean())
    if tp > 0 else 0.0
)


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n========== CELLPOSE INSTANCE METRICS ==========")

print("True Positives:", tp)
print("False Positives:", fp)
print("False Negatives:", fn)

print(f"Instance Precision: {precision:.4f}")
print(f"Instance Recall: {recall:.4f}")
print(f"Instance F1-score: {f1_score:.4f}")
print(f"Mean Matched IoU: {mean_matched_iou:.4f}")


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results = {
    "model": "Cellpose cpsam_v2",
    "iou_matching_threshold": 0.5,
    "ground_truth_objects": int(len(gt_labels)),
    "predicted_objects": int(len(pred_labels)),
    "true_positives": tp,
    "false_positives": fp,
    "false_negatives": fn,
    "instance_precision": float(precision),
    "instance_recall": float(recall),
    "instance_f1_score": float(f1_score),
    "mean_matched_iou": mean_matched_iou
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=4)

print("\nMetrics saved to:", OUTPUT_PATH)
