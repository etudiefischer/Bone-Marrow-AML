from pathlib import Path
from zipfile import ZipFile
from collections import Counter

archive = Path("data/raw/cosmx/AML_images.zip")

with ZipFile(archive, "r") as z:

    files = [f for f in z.infolist() if not f.is_dir()]

    print("\n=== FILE EXTENSION SUMMARY ===")

    extensions = Counter(
        Path(f.filename).suffix.lower()
        for f in files
    )

    for ext, count in extensions.most_common():
        print(f"{ext or '[none]'}: {count}")

    print("\n=== NON-IMAGE FILES ===")

    image_extensions = {
        ".png", ".jpg", ".jpeg", ".tif",
        ".tiff", ".bmp"
    }

    non_images = [
        f for f in files
        if Path(f.filename).suffix.lower() not in image_extensions
    ]

    print("Non-image file count:", len(non_images))

    for f in non_images[:200]:
        print(
            f"{f.filename} | "
            f"{f.file_size / (1024**2):.2f} MB"
        )

    print("\n=== DIRECTORY SUMMARY ===")

    directories = Counter()

    for f in files:
        parts = Path(f.filename).parts

        if len(parts) >= 2:
            directories["/".join(parts[:2])] += 1

    for directory, count in directories.most_common(50):
        print(directory, ":", count)