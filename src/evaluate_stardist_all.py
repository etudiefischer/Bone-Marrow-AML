
import io
import json
import time
import zipfile
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tifffile

from PIL import Image
from scipy.optimize import linear_sum_assignment
from stardist.models import StarDist2D
from csbdeep.utils import normalize


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = ROOT / "data/raw/cosmx/archives/Nucleus_train.zip"

OUTPUT_DIR = ROOT / "results/stardist_evaluation"

MASK_DIR = OUTPUT_DIR / "predicted_masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

MAX_SIZE = 1024
IOU_THRESHOLD = 0.50

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Archive not found: {ZIP_PATH}")


# ============================================================
# 2. LOAD PRETRAINED STARDIST MODEL
# ============================================================

print("\nLoading pretrained StarDist model...")

model = StarDist2D.from_pretrained("2D_versatile_fluo")

print("StarDist model loaded successfully.")


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def patient_from_name(filename):

    import re

    match = re.match(r"^(P\d+)_", filename)

    return match.group(1) if match else "Unknown"


def get_mask_stem(path):

    name = Path(path).name

    suffix = "_masks.tif"

    if name.lower().endswith(suffix):
        return name[:-len(suffix)]

    return None


def resize_image(image, max_size=1024):

    h, w = image.shape[:2]

    scale = min(1.0, max_size / max(h, w))

    if scale == 1.0:
        return image, scale

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


def resize_mask(mask, shape):

    h, w = shape

    if mask.shape == (h, w):
        return mask

    return cv2.resize(
        mask.astype(np.int32),
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )


def pixel_metrics(gt, pred):

    gt_binary = gt > 0
    pred_binary = pred > 0

    intersection = np.logical_and(
        gt_binary, pred_binary
    ).sum()

    union = np.logical_or(
        gt_binary, pred_binary
    ).sum()

    gt_pixels = gt_binary.sum()
    pred_pixels = pred_binary.sum()

    iou = intersection / union if union else 1.0

    dice_denominator = gt_pixels + pred_pixels

    dice = (
        2 * intersection / dice_denominator
        if dice_denominator else 1.0
    )

    return float(iou), float(dice)



def instance_metrics(gt, pred, threshold=0.50):

    # Flatten both segmentation masks
    gt_flat = np.asarray(gt).ravel()
    pred_flat = np.asarray(pred).ravel()

    if gt_flat.shape != pred_flat.shape:
        raise ValueError("Ground-truth and prediction shapes differ")

    # Convert labels to compact consecutive indices
    gt_values, gt_inverse = np.unique(
        gt_flat, return_inverse=True
    )

    pred_values, pred_inverse = np.unique(
        pred_flat, return_inverse=True
    )

    n_gt = len(gt_values)
    n_pred = len(pred_values)

    # Calculate pixel intersections between every label pair
    flat_indices = (
        gt_inverse.astype(np.int64) * n_pred
        + pred_inverse.astype(np.int64)
    )

    intersections = np.bincount(
        flat_indices,
        minlength=n_gt * n_pred
    ).reshape(n_gt, n_pred)

    gt_areas = np.bincount(
        gt_inverse, minlength=n_gt
    )

    pred_areas = np.bincount(
        pred_inverse, minlength=n_pred
    )

    # Remove background (label 0)
    intersections = intersections[1:, 1:]

    gt_areas = gt_areas[1:]
    pred_areas = pred_areas[1:]

    gt_count = len(gt_areas)
    pred_count = len(pred_areas)

    if gt_count == 0 or pred_count == 0:

        tp = 0
        mean_matched_iou = 0.0

    else:

        unions = (
            gt_areas[:, None]
            + pred_areas[None, :]
            - intersections
        )

        iou_matrix = np.divide(
            intersections,
            unions,
            out=np.zeros_like(
                intersections, dtype=float
            ),
            where=unions > 0
        )

        # Hungarian one-to-one matching
        rows, cols = linear_sum_assignment(
            1.0 - iou_matrix
        )

        matched_ious = iou_matrix[rows, cols]

        valid_matches = matched_ious >= threshold

        tp = int(valid_matches.sum())

        mean_matched_iou = (
            float(matched_ious[valid_matches].mean())
            if tp else 0.0
        )

    fp = pred_count - tp
    fn = gt_count - tp

    precision = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "gt_objects": int(gt_count),
        "pred_objects": int(pred_count),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "instance_precision": precision,
        "instance_recall": recall,
        "instance_F1": f1,
        "mean_matched_IoU": mean_matched_iou
    }


def create_overlay(image_rgb, gt, pred):

    # Work on a copy of the original image
    overlay = image_rgb.copy()

    # Extract boundaries
    gt_boundary = cv2.morphologyEx(
        gt.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8)
    ) > 0

    pred_boundary = cv2.morphologyEx(
        pred.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8)
    ) > 0

    # Ground truth: green
    overlay[gt_boundary] = [0, 255, 0]

    # Prediction: red
    overlay[pred_boundary] = [255, 0, 0]

    # Overlap: yellow
    overlap = gt_boundary & pred_boundary
    overlay[overlap] = [255, 255, 0]

    return overlay


# ============================================================
# 4. SCAN ARCHIVE AND MATCH IMAGES WITH MASKS
# ============================================================

print("\nScanning ZIP archive...")

with zipfile.ZipFile(ZIP_PATH, "r") as archive:

    all_files = [
        f for f in archive.namelist()
        if not f.endswith("/")
    ]

    images = [
        f for f in all_files
        if f.lower().endswith(".png")
    ]

    masks = [
        f for f in all_files
        if f.lower().endswith("_masks.tif")
    ]

    masks_by_location = defaultdict(list)
    masks_by_stem = defaultdict(list)

    for mask_path in masks:

        stem = get_mask_stem(mask_path)

        if stem is None:
            continue

        parent = Path(mask_path).parent.as_posix()

        masks_by_location[(parent, stem)].append(mask_path)
        masks_by_stem[stem].append(mask_path)

    matched_pairs = []

    for image_path in images:

        stem = Path(image_path).stem
        parent = Path(image_path).parent.as_posix()

        candidates = masks_by_location.get(
            (parent, stem), []
        )

        if not candidates:
            candidates = masks_by_stem.get(stem, [])

        if len(candidates) == 1:

            matched_pairs.append({
                "stem": stem,
                "image": image_path,
                "mask": candidates[0]
            })

        elif len(candidates) > 1:

            raise RuntimeError(
                f"Ambiguous mask match: {image_path}"
            )

    matched_pairs.sort(key=lambda x: x["stem"])

    print("\n========== ARCHIVE SUMMARY ==========")
    print("PNG images:", len(images))
    print("Ground-truth masks:", len(masks))
    print("Matched pairs:", len(matched_pairs))

    if len(matched_pairs) != 298:
        raise RuntimeError(
            f"Expected 298 matched pairs, "
            f"found {len(matched_pairs)}"
        )


    # ========================================================
    # 5. RUN STARDIST AND EVALUATE ALL FOVS
    # ========================================================

    results = []

    print("\n========== STARTING STARDIST EVALUATION ==========")

    for index, sample in enumerate(matched_pairs, 1):

        stem = sample["stem"]
        patient = patient_from_name(stem)

        try:

            # ------------------------------------------------
            # READ ORIGINAL IMAGE
            # ------------------------------------------------

            image_bytes = archive.read(sample["image"])

            image_rgb = np.array(
                Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")
            )

            # ------------------------------------------------
            # READ GROUND-TRUTH MASK
            # ------------------------------------------------

            mask_bytes = archive.read(sample["mask"])

            gt_original = tifffile.imread(
                io.BytesIO(mask_bytes)
            )

            if gt_original.ndim != 2:
                raise ValueError("Ground-truth mask is not 2D")

            if image_rgb.shape[:2] != gt_original.shape:
                raise ValueError("Image-mask dimensions differ")

            # ------------------------------------------------
            # RESIZE FOR INFERENCE
            # ------------------------------------------------

            image_resized, scale = resize_image(
                image_rgb, MAX_SIZE
            )

            gt_resized = resize_mask(
                gt_original,
                image_resized.shape[:2]
            )

            # ------------------------------------------------
            # EXTRACT BLUE CHANNEL
            # ------------------------------------------------

            blue_channel = image_resized[:, :, 2]

            normalized_image = normalize(
                blue_channel,
                1,
                99.8
            )

            # ------------------------------------------------
            # STARDIST INFERENCE
            # ------------------------------------------------

            start_time = time.perf_counter()

            pred_resized, details = model.predict_instances(
                normalized_image
            )

            inference_time = (
                time.perf_counter() - start_time
            )

            # ------------------------------------------------
            # CALCULATE METRICS
            # ------------------------------------------------

            pixel_iou, pixel_dice = pixel_metrics(
                gt_resized,
                pred_resized
            )

            instance_result = instance_metrics(
                gt_resized,
                pred_resized,
                IOU_THRESHOLD
            )

            # ------------------------------------------------
            # RESIZE PREDICTION TO ORIGINAL DIMENSIONS
            # ------------------------------------------------

            pred_original = resize_mask(
                pred_resized,
                gt_original.shape
            )

            # ------------------------------------------------
            # SAVE PREDICTED MASK
            # ------------------------------------------------

            mask_output = MASK_DIR / f"{stem}_stardist.tif"

            tifffile.imwrite(
                mask_output,
                pred_original.astype(np.uint32)
            )

            # ------------------------------------------------
            # SAVE OVERLAY
            # ------------------------------------------------

            overlay = create_overlay(
                image_rgb,
                gt_original,
                pred_original
            )

            overlay_output = OVERLAY_DIR / f"{stem}_overlay.png"

            Image.fromarray(overlay).save(
                overlay_output
            )

            # ------------------------------------------------
            # STORE RESULTS
            # ------------------------------------------------

            row = {

                "patient_id": patient,
                "fov_id": stem,

                "image_height": image_rgb.shape[0],
                "image_width": image_rgb.shape[1],

                "resize_scale": scale,

                "GT_nuclei": instance_result["gt_objects"],
                "Predicted_nuclei": instance_result["pred_objects"],

                "pixel_IoU": pixel_iou,
                "pixel_Dice": pixel_dice,

                "TP": instance_result["TP"],
                "FP": instance_result["FP"],
                "FN": instance_result["FN"],

                "instance_precision": instance_result["instance_precision"],
                "instance_recall": instance_result["instance_recall"],
                "instance_F1": instance_result["instance_F1"],

                "mean_matched_IoU": instance_result["mean_matched_IoU"],

                "inference_time_seconds": inference_time

            }

            results.append(row)

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if index % 10 == 0 or index == len(matched_pairs):

                print(
                    f"Processed {index}/{len(matched_pairs)} | "
                    f"Patient: {patient} | "
                    f"IoU: {pixel_iou:.4f} | "
                    f"F1: {instance_result['instance_F1']:.4f} | "
                    f"Time: {inference_time:.2f}s"
                )

                pd.DataFrame(results).to_csv(
                    OUTPUT_DIR / "per_fov_metrics.csv",
                    index=False
                )

        except Exception as error:

            print(
                f"\nERROR processing {stem}: {error}"
            )

            raise


# ============================================================
# 6. SAVE PER-FOV RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "per_fov_metrics.csv",
    index=False
)


# ============================================================
# 7. PATIENT-LEVEL SUMMARY
# ============================================================

metric_columns = [

    "pixel_IoU",
    "pixel_Dice",
    "instance_precision",
    "instance_recall",
    "instance_F1",
    "mean_matched_IoU",
    "inference_time_seconds"

]

patient_summary = (
    results_df.groupby("patient_id")[metric_columns]
    .agg(["mean", "std"])
    .round(4)
)

patient_summary.to_csv(
    OUTPUT_DIR / "patient_summary.csv"
)


# ============================================================
# 8. OVERALL SUMMARY
# ============================================================

overall_summary = results_df[metric_columns].agg(
    ["mean", "std"]
).round(4)

overall_summary.to_csv(
    OUTPUT_DIR / "overall_summary.csv"
)

# Save simple JSON summary too
summary_json = {

    "total_FOVs": int(len(results_df)),
    "total_patients": int(results_df["patient_id"].nunique()),

    "total_GT_nuclei": int(results_df["GT_nuclei"].sum()),
    "total_predicted_nuclei": int(
        results_df["Predicted_nuclei"].sum()
    ),

    "metrics_mean": results_df[metric_columns].mean().to_dict(),
    "metrics_std": results_df[metric_columns].std().to_dict()

}

with open(
    OUTPUT_DIR / "summary.json",
    "w"
) as file:

    json.dump(summary_json, file, indent=4)


# ============================================================
# 9. FINAL REPORT
# ============================================================

print("\n==============================================")
print("STARDIST EVALUATION COMPLETED")
print("==============================================")

print("Successfully evaluated FOVs:", len(results_df))
print("Patients:", results_df["patient_id"].nunique())

print("\nOverall mean metrics:")
print(results_df[metric_columns].mean().round(4))

print("\nOverall standard deviations:")
print(results_df[metric_columns].std().round(4))

print("\nPatient-level mean metrics:")

print(
    results_df.groupby("patient_id")[metric_columns]
    .mean()
    .round(4)
)

print("\nTotal ground-truth nuclei:", results_df["GT_nuclei"].sum())

print(
    "Total predicted nuclei:",
    results_df["Predicted_nuclei"].sum()
)

print("\nOutput directory:", OUTPUT_DIR)

print("\n==============================================")
