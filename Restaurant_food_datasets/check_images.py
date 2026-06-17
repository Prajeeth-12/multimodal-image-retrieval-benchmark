from pathlib import Path
from PIL import Image

DATASET_PATH = r"."

valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

total = 0
good = 0
bad = 0
print("STARTED")
for img_path in Path(DATASET_PATH).rglob("*"):
    if img_path.suffix.lower() not in valid_exts:
        continue

    total += 1

    try:
        with Image.open(img_path) as img:
            width, height = img.size

        if width >= 300 and height >= 300:
            print(f"✅ {img_path.name} -> {width}x{height}")
            good += 1
        else:
            print(f"❌ LOW RES: {img_path.name} -> {width}x{height}")
            bad += 1

    except Exception as e:
        print(f"⚠️ ERROR: {img_path}")

print("\nSUMMARY")
print("Total:", total)
print("Good:", good)
print("Bad:", bad)