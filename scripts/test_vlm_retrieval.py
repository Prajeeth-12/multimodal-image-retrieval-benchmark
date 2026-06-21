import os
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    local_files_only=True,
)

embeddings = np.load(
    "../embeddings/vlm/vlm_embeddings.npy"
)

metadata = pd.read_csv(
    "../embeddings/vlm/vlm_paths.csv"
)

query = "idli with sambar"

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)

scores = cosine_similarity(
    query_embedding,
    embeddings
)[0]

top_idx = scores.argsort()[::-1][:5]

print(f"\nQuery: {query}\n")

for rank, idx in enumerate(top_idx, start=1):
    print(
        f"{rank}. "
        f"{metadata.iloc[idx]['image_path']} "
        f"(score={scores[idx]:.4f})"
    )
