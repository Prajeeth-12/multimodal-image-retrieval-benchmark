from pathlib import Path
from PIL import Image

DATASET_PATH = Path("../Restaurant_food_datasets")

deleted = 0

for img_path in DATASET_PATH.rglob("*"):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        continue

    try:
        with Image.open(img_path) as img:
            w, h = img.size

        if w < 300 or h < 300:
            print(f"Deleting: {img_path}")
            img_path.unlink()
            deleted += 1

    except:
        pass

print(f"\nDeleted {deleted} low-resolution images.")