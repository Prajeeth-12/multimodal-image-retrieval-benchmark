from pathlib import Path
import csv

DATASET_PATH = Path("../Restaurant_food_datasets")

rows = []

for folder in DATASET_PATH.iterdir():
    if not folder.is_dir():
        continue

    restaurant = folder.name

    for img in folder.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue

        rows.append([img.name, restaurant])

with open("metadata.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_name", "restaurant"])
    writer.writerows(rows)

print(f"Created metadata.csv with {len(rows)} records")