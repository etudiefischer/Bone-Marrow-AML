
import re
import io
import zipfile
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import tifffile

from skimage.measure import regionprops


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

ZIP_PATH = ROOT / "data/raw/cosmx/archives/Nucleus_train.zip"

OUTPUT_DIR = ROOT / "data/processed/spatial"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    raise FileNotFoundError(f"Nucleus archive not found: {ZIP_PATH}")


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def get_patient_id(filename):

    match = re.match(r"^(P\d+)_", filename)

    return match.group(1) if match else "Unknown"


def get_mask_stem(filename):

    name = Path(filename).name

    suffix = "_masks.tif"

    if name.lower().endswith(suffix):
        return name[:-len(suffix)]

    return None


# ============================================================
# 3. SCAN ZIP AND MATCH IMAGE-MASK PAIRS
# ============================================================

print("\nReading archive:", ZIP_PATH)

with zipfile.ZipFile(ZIP_PATH, "r") as archive:

    all_files = [
        name for name in archive.namelist()
        if not name.endswith("/")
    ]

    # Find all PNG images, regardless of folder structure
    image_files = [
        name for name in all_files
        if name.lower().endswith(".png")
    ]

    # Find all ground-truth instance masks
    mask_files = [
        name for name in all_files
        if name.lower().endswith("_masks.tif")
    ]

    print("\n========== ARCHIVE INVENTORY ==========")
    print("Total files:", len(all_files))
    print("PNG images:", len(image_files))
    print("Mask TIFFs:", len(mask_files))

    # Build mask lookup using both parent directory and basename
    masks_by_location = defaultdict(list)
    masks_by_stem = defaultdict(list)

    for mask_path in mask_files:

        stem = get_mask_stem(mask_path)

        if stem is None:
            continue

        parent = Path(mask_path).parent.as_posix()

        masks_by_location[(parent, stem)].append(mask_path)
        masks_by_stem[stem].append(mask_path)

    matched_pairs = []
    unmatched_images = []

    for image_path in image_files:

        stem = Path(image_path).stem

        parent = Path(image_path).parent.as_posix()

        # First try matching within the same directory
        candidates = masks_by_location.get(
            (parent, stem), []
        )

        # If directory matching fails, use unique basename match
        if not candidates:

            candidates = masks_by_stem.get(stem, [])

        if len(candidates) == 1:

            matched_pairs.append({
                "stem": stem,
                "image_path": image_path,
                "mask_path": candidates[0]
            })

        elif len(candidates) > 1:

            raise RuntimeError(
                f"Ambiguous mask match for {image_path}: "
                f"{candidates}"
            )

        else:
            unmatched_images.append(image_path)

    matched_mask_paths = {
        sample["mask_path"] for sample in matched_pairs
    }

    unmatched_masks = [
        path for path in mask_files
        if path not in matched_mask_paths
    ]

    print("\n========== MATCHING RESULTS ==========")
    print("Matched image-mask pairs:", len(matched_pairs))
    print("Unmatched images:", len(unmatched_images))
    print("Unmatched masks:", len(unmatched_masks))

    if not matched_pairs:

        print("\nSample image paths:")
        print(image_files[:10])

        print("\nSample mask paths:")
        print(mask_files[:10])

        raise RuntimeError(
            "No matched image-mask pairs found. "
            "Inspect the archive paths above."
        )

    # ========================================================
    # 4. EXTRACT CELL COORDINATES AND MORPHOLOGY
    # ========================================================

    cell_rows = []
    fov_rows = []

    print("\n========== EXTRACTING SPATIAL INFORMATION ==========")

    for index, sample in enumerate(matched_pairs, 1):

        stem = sample["stem"]

        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        patient = get_patient_id(stem)

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image_bytes = archive.read(image_path)

        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:

            print("Skipping unreadable image:", stem)
            continue

        image_height, image_width = image.shape[:2]

        # ----------------------------------------------------
        # READ GROUND-TRUTH MASK
        # ----------------------------------------------------

        mask_bytes = archive.read(mask_path)

        mask = tifffile.imread(
            io.BytesIO(mask_bytes)
        )

        if mask.ndim != 2:

            print("Skipping non-2D mask:", stem)
            continue

        mask_height, mask_width = mask.shape

        # ----------------------------------------------------
        # CHECK IMAGE-MASK ALIGNMENT
        # ----------------------------------------------------

        if (
            image_height != mask_height
            or image_width != mask_width
        ):

            print(
                f"Skipping dimension mismatch: {stem} | "
                f"Image: {image.shape} | Mask: {mask.shape}"
            )

            continue

        # ----------------------------------------------------
        # EXTRACT INSTANCE PROPERTIES
        # ----------------------------------------------------

        props = regionprops(
            mask.astype(np.int32)
        )

        fov_rows.append({

            "patient_id": patient,
            "fov_id": stem,

            "image_path_in_zip": image_path,
            "mask_path_in_zip": mask_path,

            "image_height": image_height,
            "image_width": image_width,

            "number_of_nuclei": len(props)

        })

        for region in props:

            min_row, min_col, max_row, max_col = region.bbox

            cell_rows.append({

                # Identification
                "patient_id": patient,
                "fov_id": stem,
                "cell_id": int(region.label),

                # Spatial coordinates in original image pixels
                "centroid_x": float(region.centroid[1]),
                "centroid_y": float(region.centroid[0]),

                # Nucleus morphology
                "area_pixels": int(region.area),

                "major_axis_length": float(
                    region.axis_major_length
                ),

                "minor_axis_length": float(
                    region.axis_minor_length
                ),

                "eccentricity": float(
                    region.eccentricity
                ),

                "solidity": float(
                    region.solidity
                ),

                # Bounding box
                "min_row": int(min_row),
                "min_col": int(min_col),
                "max_row": int(max_row),
                "max_col": int(max_col),

                # Original image dimensions
                "image_height": image_height,
                "image_width": image_width

            })

        # ----------------------------------------------------
        # PROGRESS AND CHECKPOINT SAVING
        # ----------------------------------------------------

        if index % 25 == 0 or index == len(matched_pairs):

            print(
                f"Processed {index}/{len(matched_pairs)} FOVs | "
                f"Extracted nuclei: {len(cell_rows)}"
            )

            pd.DataFrame(cell_rows).to_csv(
                OUTPUT_DIR / "cell_coordinates.csv",
                index=False
            )

            pd.DataFrame(fov_rows).to_csv(
                OUTPUT_DIR / "fov_metadata.csv",
                index=False
            )


# ============================================================
# 5. VALIDATE EXTRACTION
# ============================================================

if not cell_rows:

    raise RuntimeError(
        "No nuclei were extracted. "
        "Check image-mask alignment and mask contents."
    )


# ============================================================
# 6. SAVE FINAL DATASETS
# ============================================================

cells_df = pd.DataFrame(cell_rows)

fovs_df = pd.DataFrame(fov_rows)

cells_df.to_csv(
    OUTPUT_DIR / "cell_coordinates.csv",
    index=False
)

cells_df.to_csv(
    OUTPUT_DIR / "cell_morphology.csv",
    index=False
)

fovs_df.to_csv(
    OUTPUT_DIR / "fov_metadata.csv",
    index=False
)


# ============================================================
# 7. GENERATE SUMMARY
# ============================================================

print("\n==============================================")
print("SPATIAL DATASET PREPARATION COMPLETED")
print("==============================================")

print("Matched image-mask pairs:", len(matched_pairs))
print("Successfully processed FOVs:", len(fovs_df))

print(
    "Unique patients:",
    fovs_df["patient_id"].nunique()
)

print(
    "Total extracted nuclei:",
    len(cells_df)
)

print("\nNuclei per patient:")

print(
    cells_df.groupby("patient_id").size().to_string()
)

print("\nAverage nuclei per FOV:")

print(
    round(fovs_df["number_of_nuclei"].mean(), 2)
)

print("\n========== OUTPUT FILES ==========")

print(OUTPUT_DIR / "cell_coordinates.csv")
print(OUTPUT_DIR / "cell_morphology.csv")
print(OUTPUT_DIR / "fov_metadata.csv")

print("\n==============================================")
