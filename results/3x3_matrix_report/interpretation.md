# Interpretation

Easy reading guide:
- Rows compare methods: CLIP vs Manual vs VLM.
- Columns compare similarity metrics: cosine vs euclidean vs dot_product.
- Higher value means better retrieval.

Key findings:
- Best Top-1: Manual + cosine = 0.6800
- Best Top-3: Manual + cosine = 0.8000
- Best Top-5: CLIP + cosine = 0.8000
- Best MRR: Manual + cosine = 0.7419
- Best overall method trend: Manual (Top-1), Manual (MRR).

Note: In this run, each method has the same score across cosine/euclidean/dot_product.
