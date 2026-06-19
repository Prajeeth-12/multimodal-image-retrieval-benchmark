from pathlib import Path
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ---------------------------
# Paths
# ---------------------------
DATASET_PATH = Path("../Restaurant_food_datasets")
OUTPUT_DIR = Path("../embeddings/clip")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Device
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------
# Load CLIP
# ---------------------------
print("Loading CLIP model...")

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

print("CLIP loaded successfully")

# ---------------------------
# Collect Images
# ---------------------------
image_paths = []

for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
    image_paths.extend(DATASET_PATH.rglob(ext))

image_paths = sorted(image_paths)

print(f"Found {len(image_paths)} images")

# ---------------------------
# Generate Embeddings
# ---------------------------
embeddings = []
image_names = []

for idx, img_path in enumerate(image_paths, start=1):

    try:

        image = Image.open(img_path).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            features = model.get_image_features(
                **inputs
            )

            if idx == 1:
                print("Feature type:", type(features))

            # Handle transformers version differences
            if not isinstance(features, torch.Tensor):

                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                elif hasattr(features, "image_embeds"):
                    features = features.image_embeds
                else:
                    raise ValueError(
                        f"Unexpected output type: {type(features)}"
                    )

        features = features / torch.norm(
            features,
            dim=-1,
            keepdim=True
        )

        embeddings.append(
            features.cpu().numpy()[0]
        )

        image_names.append(
            str(img_path)
        )

        if idx % 10 == 0:
            print(
                f"Processed {idx}/{len(image_paths)}"
            )

    except Exception as e:

        print(f"\nError: {img_path}")
        print(e)

# ---------------------------
# Save Results
# ---------------------------
embeddings = np.array(embeddings)

np.save(
    OUTPUT_DIR / "image_embeddings.npy",
    embeddings
)

pd.DataFrame({
    "image_path": image_names
}).to_csv(
    OUTPUT_DIR / "image_paths.csv",
    index=False
)

print("\nDone")
print("Embeddings Shape:", embeddings.shape)
print(f"Saved to: {OUTPUT_DIR}")