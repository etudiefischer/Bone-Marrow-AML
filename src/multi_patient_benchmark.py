
import re
import io
import json
import time
import zipfile
import random

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import cv2

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import find_boundaries

from cellpose import models
from stardist.models import StarDist2D
from csbdeep.utils import normalize


# ============================================================
# 1. CONFIGURATION
# ============================================================

ZIP_PATH = Path(
    "data/raw/cosmx/archives/Nucleus_train.zip"
)

OUTPUT_DIR = Path("results/multi_patient")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SIZE = 1024
IOU_THRESHOLD = 0.5
NUM_PATIENTS = 3

RANDOM_SEED = 42
EXCLUDED_PATIENTS = {"P51"}

USE_GPU = True
STARDIST_CHANNEL = 2

rng = random.Random(RANDOM_SEED)


# ============================================================
# 2. EVALUATION FUNCTION
# ============================================================

def evaluate_masks(gt, pred):

    gt_binary = gt > 0
    pred_binary = pred > 0

    intersection = np.logical_and(
        gt_binary, pred_binary
    ).sum()

    union = np.logical_or(
        gt_binary, pred_binary
    ).sum()

    pixel_iou = intersection / union if union else 0.0

    total_pixels = gt_binary.sum() + pred_binary.sum()

    pixel_dice = (
        2 * intersection / total_pixels
        if total_pixels else 0.0
    )

    # Extract unique instance labels.
    gt_labels = np.unique(gt)
    gt_labels = gt_labels[gt_labels != 0]

    pred_labels = np.unique(pred)
    pred_labels = pred_labels[pred_labels != 0]

    # Construct instance IoU matrix.
    iou_matrix = np.zeros(
        (len(gt_labels), len(pred_labels)),
        dtype=np.float32
    )

    for i, gt_id in enumerate(gt_labels):

        gt_object = gt == gt_id

        for j, pred_id in enumerate(pred_labels):

            pred_object = pred == pred_id

            inter = np.logical_and(
                gt_object, pred_object
            ).sum()

            uni = np.logical_or(
                gt_object, pred_object
            ).sum()

            if uni:
                iou_matrix[i, j] = inter / uni

    # Hungarian one-to-one matching.
    if iou_matrix.size:

        rows, cols = linear_sum_assignment(-iou_matrix)

        matched = iou_matrix[rows, cols]
        valid = matched >= IOU_THRESHOLD

        tp = int(valid.sum())

        mean_matched_iou = (
            float(matched[valid].mean())
            if tp else 0.0
        )

    else:
        tp = 0
        mean_matched_iou = 0.0

    fp = len(pred_labels) - tp
    fn = len(gt_labels) - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
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
        "mean_matched_iou": float(mean_matched_iou)
    }


# ============================================================
# 3. SCAN ZIP AND FIND MATCHED IMAGE-MASK PAIRS
# ============================================================

print("\nScanning Nucleus_train.zip...")

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"Archive not found: {ZIP_PATH}"
    )

with zipfile.ZipFile(ZIP_PATH, "r") as archive:

    files = archive.namelist()

    image_files = {}
    mask_files = {}

    for name in files:

        lower_name = name.lower()

        if lower_name.endswith(".png"):

            stem = Path(name).stem
            image_files[stem] = name

        elif lower_name.endswith("_masks.tif"):

            stem = Path(name).name[:-len("_masks.tif")]
            mask_files[stem] = name

    matched_stems = sorted(
        set(image_files) & set(mask_files)
    )

    print("Matched image-mask pairs:", len(matched_stems))

    if not matched_stems:
        raise RuntimeError(
            "No matching image-mask pairs found."
        )

    # --------------------------------------------------------
    # GROUP MATCHED PAIRS BY PATIENT
    # --------------------------------------------------------

    patient_pairs = {}

    for stem in matched_stems:

        match = re.match(r"^(P\d+)_", stem)

        if not match:
            continue

        patient = match.group(1)

        patient_pairs.setdefault(
            patient, []
        ).append(stem)

    # Exclude previously explored patient P51.
    eligible_patients = sorted(
        patient
        for patient in patient_pairs
        if patient not in EXCLUDED_PATIENTS
    )

    print("Eligible patients:", len(eligible_patients))

    if len(eligible_patients) < NUM_PATIENTS:
        raise RuntimeError(
            f"Only {len(eligible_patients)} eligible patients "
            f"found after excluding {EXCLUDED_PATIENTS}."
        )

    # ========================================================
    # 4. RANDOMLY SELECT PATIENTS AND FOVS
    # ========================================================

    selected_patients = rng.sample(
        eligible_patients,
        NUM_PATIENTS
    )

    selected_pairs = []

    for patient in selected_patients:

        # Randomly select one FOV for this patient.
        selected_stem = rng.choice(
            patient_pairs[patient]
        )

        selected_pairs.append({
            "patient": patient,
            "stem": selected_stem,
            "image_path": image_files[selected_stem],
            "mask_path": mask_files[selected_stem]
        })

    print("\n========== RANDOMLY SELECTED SAMPLES ==========")

    for sample in selected_pairs:

        print(
            f"Patient: {sample['patient']} | "
            f"FOV: {sample['stem']}"
        )

    # Save selection for reproducibility.
    selection_info = {
        "random_seed": RANDOM_SEED,
        "excluded_patients": sorted(EXCLUDED_PATIENTS),
        "number_of_patients": NUM_PATIENTS,
        "target_max_dimension": TARGET_SIZE,
        "selected_samples": selected_pairs
    }

    with open(
        OUTPUT_DIR / "selected_samples.json", "w"
    ) as f:

        json.dump(selection_info, f, indent=4)

    # ========================================================
    # 5. LOAD PRETRAINED MODELS ONCE
    # ========================================================

    print("\nLoading pretrained Cellpose...")

    cellpose_model = models.CellposeModel(
        gpu=USE_GPU,
        pretrained_model="cpsam_v2"
    )

    print("\nLoading pretrained StarDist...")

    stardist_model = StarDist2D.from_pretrained(
        "2D_versatile_fluo"
    )

    # ========================================================
    # 6. PROCESS EACH PATIENT
    # ========================================================

    all_results = []

    for sample in tqdm(
        selected_pairs,
        desc="Evaluating patients"
    ):

        patient = sample["patient"]
        stem = sample["stem"]

        print(f"\n\n========== {patient} ==========")

        patient_dir = OUTPUT_DIR / patient
        patient_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # READ IMAGE AND MASK DIRECTLY FROM ZIP
        # ----------------------------------------------------

        image_bytes = archive.read(
            sample["image_path"]
        )

        mask_bytes = archive.read(
            sample["mask_path"]
        )

        image = np.array(
            Image.open(
                io.BytesIO(image_bytes)
            ).convert("RGB")
        )

        gt = tifffile.imread(
            io.BytesIO(mask_bytes)
        )

        if gt.ndim != 2:
            raise ValueError(
                f"Expected a 2D mask for {stem}, "
                f"received {gt.shape}"
            )

        if image.shape[:2] != gt.shape:
            raise ValueError(
                f"Image-mask dimensions differ for {stem}: "
                f"{image.shape[:2]} vs {gt.shape}"
            )

        original_height, original_width = gt.shape

        # ----------------------------------------------------
        # RESIZE LONGEST DIMENSION TO 1024
        # ----------------------------------------------------

        scale = TARGET_SIZE / max(
            original_height, original_width
        )

        new_width = max(
            1, round(original_width * scale)
        )

        new_height = max(
            1, round(original_height * scale)
        )

        image_resized = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        # Nearest-neighbour preserves discrete instance IDs.
        gt_resized = cv2.resize(
            gt,
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST
        )

        gt_resized = gt_resized.astype(gt.dtype)

        print("Original shape:", image.shape)
        print("Resized shape:", image_resized.shape)
        print("Resize factor:", scale)

        # Save resized input and ground truth.
        tifffile.imwrite(
            patient_dir / "input_image.tif",
            image_resized
        )

        tifffile.imwrite(
            patient_dir / "ground_truth.tif",
            gt_resized
        )

        # ----------------------------------------------------
        # CELLPPOSE INFERENCE
        # ----------------------------------------------------

        print("\nRunning Cellpose...")

        start = time.time()

        cp_pred, cp_flows, cp_styles = cellpose_model.eval(
            image_resized,
            channel_axis=-1,
            normalize=True,
            batch_size=4
        )

        cp_time = time.time() - start

        cp_pred = cp_pred.astype(np.uint32)

        tifffile.imwrite(
            patient_dir / "cellpose_mask.tif",
            cp_pred
        )

        cp_metrics = evaluate_masks(
            gt_resized,
            cp_pred
        )

        print(
            "Cellpose inference:",
            round(cp_time, 3),
            "seconds"
        )

        # ----------------------------------------------------
        # STARDIST INFERENCE
        # ----------------------------------------------------

        print("\nRunning StarDist...")

        blue_channel = image_resized[
            :, :, STARDIST_CHANNEL
        ].astype(np.float32)

        blue_normalized = normalize(
            blue_channel,
            1,
            99.8,
            axis=None
        )

        start = time.time()

        sd_pred, sd_details = stardist_model.predict_instances(
            blue_normalized,
            prob_thresh=0.479071,
            nms_thresh=0.3
        )

        sd_time = time.time() - start

        sd_pred = sd_pred.astype(np.uint32)

        tifffile.imwrite(
            patient_dir / "stardist_mask.tif",
            sd_pred
        )

        sd_metrics = evaluate_masks(
            gt_resized,
            sd_pred
        )

        print(
            "StarDist inference:",
            round(sd_time, 3),
            "seconds"
        )

        # ----------------------------------------------------
        # CREATE VISUAL OVERLAYS
        # ----------------------------------------------------

        gt_bound = find_boundaries(
            gt_resized, mode="outer"
        )

        cp_bound = find_boundaries(
            cp_pred, mode="outer"
        )

        sd_bound = find_boundaries(
            sd_pred, mode="outer"
        )

        # Cellpose overlay.
        fig, axes = plt.subplots(
            1, 3, figsize=(18, 6)
        )

        axes[0].imshow(image_resized)
        axes[0].set_title(f"{patient} Original")

        axes[1].imshow(image_resized)
        axes[1].contour(
            gt_bound,
            levels=[0.5],
            colors="lime",
            linewidths=0.5
        )
        axes[1].set_title("Ground Truth")

        axes[2].imshow(image_resized)
        axes[2].contour(
            cp_bound,
            levels=[0.5],
            colors="red",
            linewidths=0.5
        )
        axes[2].set_title("Cellpose")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(
            patient_dir / "cellpose_overlay.png",
            dpi=160
        )

        plt.close()

        # StarDist overlay.
        fig, axes = plt.subplots(
            1, 3, figsize=(18, 6)
        )

        axes[0].imshow(image_resized)
        axes[0].set_title(f"{patient} Original")

        axes[1].imshow(image_resized)
        axes[1].contour(
            gt_bound,
            levels=[0.5],
            colors="lime",
            linewidths=0.5
        )
        axes[1].set_title("Ground Truth")

        axes[2].imshow(image_resized)
        axes[2].contour(
            sd_bound,
            levels=[0.5],
            colors="red",
            linewidths=0.5
        )
        axes[2].set_title("StarDist")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        plt.savefig(
            patient_dir / "stardist_overlay.png",
            dpi=160
        )

        plt.close()

        # ----------------------------------------------------
        # STORE METRICS
        # ----------------------------------------------------

        for model_name, metrics, inference_time in [
            ("Cellpose", cp_metrics, cp_time),
            ("StarDist", sd_metrics, sd_time)
        ]:

            row = {
                "patient": patient,
                "sample": stem,
                "model": model_name,
                "original_height": original_height,
                "original_width": original_width,
                "resized_height": new_height,
                "resized_width": new_width,
                "resize_factor": scale,
                "inference_time_seconds": inference_time,
                **metrics
            }

            all_results.append(row)

            print(f"\n{model_name} metrics:")

            for key, value in metrics.items():
                print(f"{key}: {value}")

        # Save partial results after every patient.
        pd.DataFrame(all_results).to_csv(
            OUTPUT_DIR / "per_patient_results.csv",
            index=False
        )


# ============================================================
# 7. PER-PATIENT RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(all_results)

display_columns = [
    "patient",
    "model",
    "ground_truth_objects",
    "predicted_objects",
    "pixel_iou",
    "pixel_dice",
    "instance_precision",
    "instance_recall",
    "instance_f1",
    "mean_matched_iou",
    "inference_time_seconds"
]

print("\n\n========== PER-PATIENT RESULTS ==========")

print(
    results_df[display_columns]
    .round(4)
    .to_string(index=False)
)

results_df.to_csv(
    OUTPUT_DIR / "per_patient_results.csv",
    index=False
)


# ============================================================
# 8. AGGREGATE RESULTS ACROSS PATIENTS
# ============================================================

metric_columns = [
    "pixel_iou",
    "pixel_dice",
    "instance_precision",
    "instance_recall",
    "instance_f1",
    "mean_matched_iou",
    "inference_time_seconds"
]

aggregate = (
    results_df.groupby("model")[metric_columns]
    .agg(["mean", "std"])
    .round(4)
)

print("\n\n========== AGGREGATE RESULTS ==========")

print(aggregate.to_string())

aggregate.to_csv(
    OUTPUT_DIR / "aggregate_results.csv"
)

# Mean-only table.
mean_table = (
    results_df.groupby("model")[metric_columns]
    .mean()
    .round(4)
)

print("\n\n========== MEAN COMPARISON ==========")

print(mean_table.to_string())

mean_table.to_csv(
    OUTPUT_DIR / "aggregate_means.csv"
)

print("\n========================================")
print("MULTI-PATIENT EXPERIMENT COMPLETED")
print("========================================")
print("Selected samples:", OUTPUT_DIR / "selected_samples.json")
print("Per-patient results:", OUTPUT_DIR / "per_patient_results.csv")
print("Aggregate results:", OUTPUT_DIR / "aggregate_results.csv")
print("Mean comparison:", OUTPUT_DIR / "aggregate_means.csv")
print("All outputs:", OUTPUT_DIR)
