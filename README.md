# Restaurant Image Retrieval Benchmark

## Project Overview
This project benchmarks three retrieval approaches on a restaurant image corpus:
1. CLIP Image Embeddings
2. Manual Human-Written Caption Embeddings
3. VLM (GPT-generated) Caption Embeddings

## Repository Architecture
- `Restaurant_food_datasets/`: image corpus by restaurant
- `metadata/`: metadata, captions, and benchmark queries
- `embeddings/clip|manual|vlm/`: precomputed vector indexes
- `scripts/`: data/embedding generation and evaluation scripts
- `results/`: generated metrics, charts, and reports

## Setup Instructions
1. Install Python 3.10+.
2. Install dependencies:
   - `pip install numpy pandas matplotlib scikit-learn transformers sentence-transformers torch`
3. Ensure model cache exists locally for offline-safe runs (`openai/clip-vit-base-patch32`, `all-MiniLM-L6-v2`).

## Execution Instructions
Run the full evaluation:

```bash
python scripts/evaluate_methods.py
```

Generated outputs:
- `results/metrics.csv`
- `results/per_query_results.csv`
- `results/charts/*.png`
- `results/final_report.md`
- `results/final_report.html`
- `results/executive_summary.md`

## Benchmark Results
| method | top1_accuracy | top3_accuracy | top5_accuracy | mrr | top1_accuracy_pct | top3_accuracy_pct | top5_accuracy_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP | 0.32 | 0.6 | 0.8 | 0.5 | 32.0 | 60.0 | 80.0 |
| Manual | 0.68 | 0.8 | 0.8 | 0.7419 | 68.0 | 80.0 | 80.0 |
| VLM | 0.64 | 0.72 | 0.8 | 0.7062 | 64.0 | 72.0 | 80.0 |

## Conclusions
- Strongest method: **Manual**
- Weakest method: **CLIP**
- Detailed discussion, failure analysis, and future work are documented in `results/final_report.md`.
