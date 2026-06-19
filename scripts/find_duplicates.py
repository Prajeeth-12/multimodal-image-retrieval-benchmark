from pathlib import Path
import hashlib

DATASET_PATH = Path("../Restaurant_food_datasets")

hashes = {}
duplicates = []

for img_path in DATASET_PATH.rglob("*"):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        continue

    with open(img_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    if file_hash in hashes:
        duplicates.append((img_path, hashes[file_hash]))
    else:
        hashes[file_hash] = img_path

print(f"Duplicate Images Found: {len(duplicates)}")

for dup, original in duplicates:
    print(f"\nDuplicate: {dup}")
    print(f"Original : {original}")