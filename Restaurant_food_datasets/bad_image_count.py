from pathlib import Path
from PIL import Image

bad_by_folder = {}

for img_path in Path(".").rglob("*"):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        continue

    try:
        with Image.open(img_path) as img:
            w, h = img.size

        if w < 300 or h < 300:
            folder = img_path.parent.name
            bad_by_folder[folder] = bad_by_folder.get(folder, 0) + 1

    except:
        pass

print("\nBAD IMAGE COUNT BY FOLDER\n")

for folder, count in bad_by_folder.items():
    print(f"{folder}: {count}")