from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CAPTIONS_FILE = Path("../metadata/vlm_captions.csv")
OUTPUT_DIR = Path("../embeddings/vlm")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df = pd.read_csv(CAPTIONS_FILE)

captions = df["caption"].tolist()

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(
    captions,
    batch_size=32,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

np.save(
    OUTPUT_DIR / "vlm_embeddings.npy",
    embeddings
)

df.to_csv(
    OUTPUT_DIR / "vlm_paths.csv",
    index=False
)

print("Done")
print("Shape:", embeddings.shape)