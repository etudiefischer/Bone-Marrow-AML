import numpy as np
import pandas as pd
import tifffile

from pathlib import Path
from skimage.measure import regionprops

# Ground-truth instance segmentation mask
mask_path = Path(
    "data/processed/inspection/nucleus/"
    "P51_FOV01_0_0_masks.tif"
)

mask = tifffile.imread(mask_path)

# Extract individual cell properties
regions = regionprops(mask)

cells = []

for region in regions:
    y, x = region.centroid

    cells.append({
        "cell_id": region.label,
        "centroid_x": x,
        "centroid_y": y,
        "area_pixels": region.area,
        "min_row": region.bbox[0],
        "min_col": region.bbox[1],
        "max_row": region.bbox[2],
        "max_col": region.bbox[3]
    })

# Create dataframe
df = pd.DataFrame(cells)

# Save spatial information
output_dir = Path("results/spatial")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "nucleus_coordinates.csv"

df.to_csv(output_path, index=False)

print("Spatial coordinate extraction completed.")
print("Total annotated cells:", len(df))
print("CSV saved to:", output_path)

print("\nFirst five cells:")
print(df.head())