from pathlib import Path
import hashlib

hashes = {}
duplicates = []

for file in Path(".").rglob("*"):
    if file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        continue

    with open(file, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    if file_hash in hashes:
        duplicates.append((file, hashes[file_hash]))
    else:
        hashes[file_hash] = file

print(f"Duplicate Images Found: {len(duplicates)}")

for dup, original in duplicates:
    print(f"\nDuplicate: {dup}")
    print(f"Original : {original}")