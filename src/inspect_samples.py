from pathlib import Path
import numpy as np
import tifffile
from PIL import Image

samples = {
    "Nucleus": Path("data/processed/inspection/nucleus"),
    "Membrane": Path("data/processed/inspection/membrane"),
}

for dataset, folder in samples.items():
    print(f"\n{'=' * 50}")
    print(f"{dataset.upper()} DATASET")
    print("=" * 50)

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() == ".png":
            image = np.array(Image.open(file))
            kind = "INPUT IMAGE"

        elif file.suffix.lower() in [".tif", ".tiff"]:
            image = tifffile.imread(file)
            kind = "GROUND-TRUTH MASK"

        else:
            continue

        print(f"\nFile: {file.name}")
        print(f"Type: {kind}")
        print(f"Shape: {image.shape}")
        print(f"Data type: {image.dtype}")
        print(f"Minimum: {image.min()}")
        print(f"Maximum: {image.max()}")
        print(f"Unique values: {len(np.unique(image))}")

        if kind == "GROUND-TRUTH MASK":
            print(f"Unique labels (first 20): {np.unique(image)[:20]}")