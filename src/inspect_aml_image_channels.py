# src/inspect_aml_image_channels.py

import re
import json
import zipfile
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = PROJECT_ROOT / "data/raw/cosmx/AML_images.zip"

OUTPUT_DIR = (
    PROJECT_ROOT / "data/processed/inspection/channel_identity"
)
PLOT_DIR = OUTPUT_DIR / "channel_visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE_FOVS_PER_PATIENT = 2

CHANNELS = {
    0: "RED",
    1: "GREEN",
    2: "BLUE",
}

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def extract_patient_id(path):
    match = re.search(r"(P\d+)", path, re.IGNORECASE)
    return match.group(1).upper() if match else "UNKNOWN"


def extract_fov_id(path):
    """
    Accepts variations such as:
    FOV1, FOV01, FOV_01, FOV-001, FOV 01.
    """
    match = re.search(
        r"FOV[\s_-]*0*(\d+)",
        path,
        re.IGNORECASE
    )

    if match:
        return f"FOV{int(match.group(1)):02d}"

    # Keep the filename instead of discarding the image.
    return Path(path).stem


def load_rgb_image(zip_file, image_path):
    with zip_file.open(image_path) as file:
        return np.asarray(
            Image.open(file).convert("RGB")
        )


def scale_channel_for_display(channel):
    low = np.percentile(channel, 1)
    high = np.percentile(channel, 99)

    if high <= low:
        return np.zeros_like(channel, dtype=np.uint8)

    scaled = (
        channel.astype(np.float32) - low
    ) / (high - low)

    return (
        np.clip(scaled, 0, 1) * 255
    ).astype(np.uint8)


def calculate_channel_statistics(image, patient, fov, image_path):
    rows = []

    for index, channel_name in CHANNELS.items():

        channel = image[:, :, index].astype(np.float32)

        rows.append({
            "patient_id": patient,
            "fov_id": fov,
            "image_path": image_path,
            "channel_index": index,
            "channel_name": channel_name,
            "height": image.shape[0],
            "width": image.shape[1],
            "mean_intensity": float(channel.mean()),
            "median_intensity": float(np.median(channel)),
            "std_intensity": float(channel.std()),
            "minimum_intensity": float(channel.min()),
            "maximum_intensity": float(channel.max()),
            "p90_intensity": float(np.percentile(channel, 90)),
            "p95_intensity": float(np.percentile(channel, 95)),
            "p99_intensity": float(np.percentile(channel, 99)),
            "fraction_above_25": float(np.mean(channel > 25)),
            "fraction_above_50": float(np.mean(channel > 50)),
            "fraction_above_100": float(np.mean(channel > 100)),
            "fraction_above_150": float(np.mean(channel > 150)),
            "fraction_above_200": float(np.mean(channel > 200)),
        })

    return rows


# ============================================================
# 3. VALIDATE ARCHIVE
# ============================================================

print("\n" + "=" * 70)
print("AML IMAGE CHANNEL IDENTITY AUDIT")
print("=" * 70)

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"Archive not found: {ZIP_PATH}"
    )

print(f"\nArchive: {ZIP_PATH}")

# ============================================================
# 4. DISCOVER MARKER IMAGES RECURSIVELY
# ============================================================

with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:

    all_pngs = [
        name for name in zip_file.namelist()
        if name.lower().endswith(".png")
    ]

    print(f"\nPNG files in archive: {len(all_pngs)}")

    image_records = []
    marker_directories = set()

    for image_path in all_pngs:

        path_parts = image_path.split("/")

        # Find the marker directory anywhere in the path.
        marker_positions = [
            i for i, part in enumerate(path_parts)
            if "DAPI_B2M_CD34" in part.upper()
        ]

        if not marker_positions:
            continue

        marker_index = marker_positions[-1]

        marker_directory = "/".join(
            path_parts[:marker_index + 1]
        )

        marker_directories.add(marker_directory)

        patient = extract_patient_id(image_path)
        fov = extract_fov_id(image_path)

        image_records.append({
            "patient_id": patient,
            "fov_id": fov,
            "image_path": image_path,
            "marker_directory": marker_directory,
        })

    print(
        f"Marker directories discovered: "
        f"{len(marker_directories)}"
    )

    print(
        f"Marker PNG images discovered: "
        f"{len(image_records)}"
    )

    print("\nMarker directories:")

    for directory in sorted(marker_directories):
        print(" ", directory)

    if not image_records:

        print("\nExample PNG paths from archive:")

        for path in all_pngs[:20]:
            print(" ", path)

        raise RuntimeError(
            "No PNG images were found inside marker directories. "
            "Inspect the example paths printed above."
        )

    # Save an inventory so filenames and paths can be inspected.
    inventory_df = pd.DataFrame(image_records)

    inventory_df.to_csv(
        OUTPUT_DIR / "marker_image_inventory.csv",
        index=False
    )

    # ========================================================
    # 5. PRINT PATIENT AND FOV COUNTS
    # ========================================================

    print("\nImages per patient:")

    patient_counts = (
        inventory_df.groupby("patient_id")
        .size()
        .sort_index()
    )

    print(patient_counts.to_string())

    print("\nFirst 15 discovered images:")

    for record in image_records[:15]:
        print(
            record["patient_id"],
            record["fov_id"],
            record["image_path"]
        )

    # ========================================================
    # 6. CALCULATE CHANNEL STATISTICS
    # ========================================================

    print("\nCalculating channel statistics...")

    all_statistics = []
    representative_images = defaultdict(list)
    dimensions = defaultdict(set)

    image_records = sorted(
        image_records,
        key=lambda x: (
            x["patient_id"],
            x["fov_id"],
            x["image_path"]
        )
    )

    for index, record in enumerate(image_records, start=1):

        image = load_rgb_image(
            zip_file,
            record["image_path"]
        )

        patient = record["patient_id"]
        fov = record["fov_id"]

        dimensions[patient].add(image.shape[:2])

        all_statistics.extend(
            calculate_channel_statistics(
                image,
                patient,
                fov,
                record["image_path"]
            )
        )

        if (
            len(representative_images[patient])
            < REPRESENTATIVE_FOVS_PER_PATIENT
        ):
            representative_images[patient].append({
                "record": record,
                "image": image.copy(),
            })

        if index % 10 == 0 or index == len(image_records):
            print(
                f"Processed {index}/{len(image_records)} images"
            )

    stats_df = pd.DataFrame(all_statistics)

    stats_df.to_csv(
        OUTPUT_DIR / "rgb_channel_statistics.csv",
        index=False
    )

    # ========================================================
    # 7. OVERALL CHANNEL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("OVERALL RGB CHANNEL STATISTICS")
    print("=" * 70)

    overall_summary = (
        stats_df.groupby("channel_name")
        .agg({
            "mean_intensity": ["mean", "std"],
            "median_intensity": ["mean", "std"],
            "p95_intensity": ["mean", "std"],
            "p99_intensity": ["mean", "std"],
            "fraction_above_50": ["mean", "std"],
            "fraction_above_100": ["mean", "std"],
        })
        .round(3)
    )

    print(overall_summary.to_string())

    overall_summary.to_csv(
        OUTPUT_DIR / "overall_channel_summary.csv"
    )

    # ========================================================
    # 8. PATIENT-WISE CHANNEL SUMMARY
    # ========================================================

    patient_summary = (
        stats_df.groupby(["patient_id", "channel_name"])
        .agg({
            "mean_intensity": "mean",
            "median_intensity": "mean",
            "p95_intensity": "mean",
            "p99_intensity": "mean",
            "fraction_above_50": "mean",
            "fraction_above_100": "mean",
        })
        .round(3)
        .reset_index()
    )

    patient_summary.to_csv(
        OUTPUT_DIR / "patient_channel_summary.csv",
        index=False
    )

    print("\nPATIENT-WISE CHANNEL SUMMARY")
    print(patient_summary.to_string(index=False))

    # ========================================================
    # 9. REPRESENTATIVE CHANNEL VISUALIZATIONS
    # ========================================================

    print("\nGenerating channel visualizations...")

    visualization_records = []

    for patient, samples in representative_images.items():

        for sample in samples:

            record = sample["record"]
            image = sample["image"]
            fov = record["fov_id"]

            fig, axes = plt.subplots(
                2, 3,
                figsize=(18, 11)
            )

            fig.suptitle(
                f"{patient} {fov} - RGB Channel Audit",
                fontsize=16
            )

            axes[0, 0].imshow(image)
            axes[0, 0].set_title("Original RGB image")
            axes[0, 0].axis("off")

            for channel_index, channel_name in CHANNELS.items():

                channel = image[:, :, channel_index]

                display_image = scale_channel_for_display(
                    channel
                )

                axes[0, channel_index].imshow(
                    display_image,
                    cmap="gray",
                    vmin=0,
                    vmax=255
                )

                axes[0, channel_index].set_title(
                    f"{channel_name} channel "
                    f"(RGB index {channel_index})"
                )

                axes[0, channel_index].axis("off")

            # Histograms
            for channel_index, channel_name in CHANNELS.items():

                channel = image[:, :, channel_index]

                axes[1, channel_index].hist(
                    channel.ravel(),
                    bins=100,
                    range=(0, 255),
                    log=True
                )

                axes[1, channel_index].set_title(
                    f"{channel_name} intensity histogram"
                )

                axes[1, channel_index].set_xlabel(
                    "Pixel intensity"
                )

                axes[1, channel_index].set_ylabel(
                    "Pixel count (log scale)"
                )

            plt.tight_layout()

            output_path = (
                PLOT_DIR /
                f"{patient}_{fov}_channel_audit.png"
            )

            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close(fig)

            visualization_records.append({
                "patient_id": patient,
                "fov_id": fov,
                "visualization": str(output_path),
            })

    pd.DataFrame(visualization_records).to_csv(
        OUTPUT_DIR / "visualization_index.csv",
        index=False
    )

    # ========================================================
    # 10. CHANNEL CORRELATIONS
    # ========================================================

    print("\nCalculating RGB channel correlations...")

    correlation_rows = []

    for record in image_records:

        image = load_rgb_image(
            zip_file,
            record["image_path"]
        ).astype(np.float32)

        red = image[:, :, 0].ravel()
        green = image[:, :, 1].ravel()
        blue = image[:, :, 2].ravel()

        correlation_rows.append({
            "patient_id": record["patient_id"],
            "fov_id": record["fov_id"],
            "red_green_correlation": float(
                np.corrcoef(red, green)[0, 1]
            ),
            "red_blue_correlation": float(
                np.corrcoef(red, blue)[0, 1]
            ),
            "green_blue_correlation": float(
                np.corrcoef(green, blue)[0, 1]
            ),
        })

    correlation_df = pd.DataFrame(correlation_rows)

    correlation_df.to_csv(
        OUTPUT_DIR / "rgb_channel_correlations.csv",
        index=False
    )

    # ========================================================
    # 11. SAVE SUMMARY
    # ========================================================

    summary = {
        "archive": str(ZIP_PATH),
        "total_png_files_in_archive": len(all_pngs),
        "marker_directories": sorted(marker_directories),
        "total_marker_images": len(image_records),
        "patients": sorted(
            inventory_df["patient_id"].unique().tolist()
        ),
        "patient_fov_counts": {
            str(key): int(value)
            for key, value in patient_counts.items()
        },
        "channel_indices": {
            "RED": 0,
            "GREEN": 1,
            "BLUE": 2,
        },
        "channel_identity_status": (
            "NOT_CONFIRMED: biological marker identities "
            "must be verified using dataset documentation "
            "or trusted reference images."
        ),
        "image_dimensions_by_patient": {
            patient: [
                list(shape)
                for shape in sorted(patient_shapes)
            ]
            for patient, patient_shapes in dimensions.items()
        },
        "output_directory": str(OUTPUT_DIR),
    }

    with open(
        OUTPUT_DIR / "channel_audit_summary.json",
        "w"
    ) as file:
        json.dump(summary, file, indent=4)

# ============================================================
# 12. FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("CHANNEL IDENTITY AUDIT COMPLETED")
print("=" * 70)

print(f"\nMarker images processed: {len(image_records)}")

print(f"\nOutput directory:\n{OUTPUT_DIR}")

print("\nGenerated outputs:")
print("1. marker_image_inventory.csv")
print("2. rgb_channel_statistics.csv")
print("3. overall_channel_summary.csv")
print("4. patient_channel_summary.csv")
print("5. rgb_channel_correlations.csv")
print("6. visualization_index.csv")
print("7. channel_audit_summary.json")
print("8. channel_visualizations/")

print("\nBiological channel mapping has NOT been automatically assigned.")
print("Verify marker identities before measuring CD34.")

print("\nDone.")