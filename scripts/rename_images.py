from pathlib import Path

DATASET_PATH = Path("../Restaurant_food_datasets")

for folder in DATASET_PATH.iterdir():
    if not folder.is_dir():
        continue

    prefix = folder.name.lower()

    images = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])

    for i, img in enumerate(images, start=1):
        new_name = f"{prefix}_{i:03d}{img.suffix.lower()}"
        img.rename(folder / new_name)

print("Renaming completed.")