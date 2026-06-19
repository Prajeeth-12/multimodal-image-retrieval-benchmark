import os
import torch
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    local_files_only=True,
    use_safetensors=False,
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    local_files_only=True,
)

# Load embeddings
embeddings = np.load(
    "../embeddings/clip/image_embeddings.npy"
)

paths = pd.read_csv(
    "../embeddings/clip/image_paths.csv"
)

query = "Dindigul style mutton biryani"

inputs = processor(
    text=[query],
    return_tensors="pt",
    padding=True
)

inputs = {
    k: v.to(device)
    for k, v in inputs.items()
}

with torch.no_grad():

    text_features = model.get_text_features(**inputs)
    if not isinstance(text_features, torch.Tensor):
        query_embedding = text_features.pooler_output
    else:
        query_embedding = text_features

query_embedding = query_embedding / torch.norm(
    query_embedding,
    dim=-1,
    keepdim=True
)

query_embedding = query_embedding.cpu().numpy()

scores = embeddings @ query_embedding.T
scores = scores.squeeze()

top_k = 5
top_idx = np.argsort(scores)[::-1][:top_k]

print(f"\nQuery: {query}\n")

for rank, idx in enumerate(top_idx, start=1):
    print(
        f"{rank}. {paths.iloc[idx]['image_path']} "
        f"(score={scores[idx]:.4f})"
    )
