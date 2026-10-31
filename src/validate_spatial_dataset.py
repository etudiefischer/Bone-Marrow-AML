
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/spatial"

cells = pd.read_csv(DATA / "cell_coordinates.csv")
fovs = pd.read_csv(DATA / "fov_metadata.csv")

print("\n========== DATASET VALIDATION ==========")

# 1. Basic counts
print("\nTotal nuclei:", len(cells))
print("Total FOVs:", len(fovs))
print("Patients:", cells["patient_id"].nunique())

# 2. Missing values
print("\nMissing values:")
print(cells.isnull().sum())

# 3. Duplicate cell identifiers within each FOV
duplicates = cells.duplicated(
    subset=["patient_id", "fov_id", "cell_id"]
).sum()

print("\nDuplicate cell IDs:", duplicates)

# 4. Coordinate boundary validation
invalid_x = (
    (cells["centroid_x"] < 0) |
    (cells["centroid_x"] >= cells["image_width"])
).sum()

invalid_y = (
    (cells["centroid_y"] < 0) |
    (cells["centroid_y"] >= cells["image_height"])
).sum()

print("Invalid X coordinates:", invalid_x)
print("Invalid Y coordinates:", invalid_y)

# 5. Compare extracted nuclei against FOV metadata
actual_counts = (
    cells.groupby(["patient_id", "fov_id"])
    .size()
    .reset_index(name="extracted_nuclei")
)

comparison = fovs.merge(
    actual_counts,
    on=["patient_id", "fov_id"],
    how="left"
)

comparison["extracted_nuclei"] = (
    comparison["extracted_nuclei"].fillna(0).astype(int)
)

comparison["count_matches"] = (
    comparison["number_of_nuclei"] ==
    comparison["extracted_nuclei"]
)

print(
    "\nFOV count mismatches:",
    (~comparison["count_matches"]).sum()
)

# 6. Nucleus area validation
print("\nNucleus area statistics:")
print(cells["area_pixels"].describe())

# 7. Patient-level summary
print("\nPatient-level nucleus counts:")
print(cells.groupby("patient_id").size())

# 8. Final status
passed = (
    len(cells) > 0
    and len(fovs) == 298
    and duplicates == 0
    and invalid_x == 0
    and invalid_y == 0
    and comparison["count_matches"].all()
)

print("\n========== VALIDATION RESULT ==========")

if passed:
    print("PASS: Spatial dataset passed these validation checks.")
else:
    print("CHECK REQUIRED: Some validation checks failed.")

# Save validation comparison
comparison.to_csv(
    DATA / "validation_fov_counts.csv",
    index=False
)

print("\nSaved:", DATA / "validation_fov_counts.csv")
