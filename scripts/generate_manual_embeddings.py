from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------
# Paths
# ---------------------------

CAPTIONS_FILE = Path(
    "../metadata/manual_captions.csv"
)

OUTPUT_DIR = Path(
    "../embeddings/manual"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ---------------------------
# Load Captions
# ---------------------------

df = pd.read_csv(CAPTIONS_FILE)

captions = df["caption"].tolist()

print(
    f"Loaded {len(captions)} captions"
)

# ---------------------------
# Load Model
# ---------------------------

print(
    "Loading Sentence Transformer..."
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print(
    "Model loaded"
)

# ---------------------------
# Generate Embeddings
# ---------------------------

embeddings = model.encode(
    captions,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(
    "Embedding Shape:",
    embeddings.shape
)

# ---------------------------
# Save
# ---------------------------

np.save(
    OUTPUT_DIR /
    "manual_embeddings.npy",
    embeddings
)

df.to_csv(
    OUTPUT_DIR /
    "manual_paths.csv",
    index=False
)

print("\nDone")
print(
    f"Saved to {OUTPUT_DIR}"
)