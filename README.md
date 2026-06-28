# Restaurant Image Retrieval Benchmark

## Project Overview
This project benchmarks three retrieval approaches on a restaurant image corpus:
1. CLIP Image Embeddings
2. Manual Human-Written Caption Embeddings
3. VLM (GPT-generated) Caption Embeddings

Each method is evaluated with three similarity metrics:
- cosine
- euclidean
- dot product

## Primary Deliverable: 3x3 Matrix Report
Main output folder:
- `results/3x3_matrix_report/`

Inside this folder:
- `matrices/matrix_top1_accuracy.csv`
- `matrices/matrix_top3_accuracy.csv`
- `matrices/matrix_top5_accuracy.csv`
- `matrices/matrix_mrr.csv`
- `diagrams/matrix_top1_accuracy_heatmap.png`
- `diagrams/matrix_top3_accuracy_heatmap.png`
- `diagrams/matrix_top5_accuracy_heatmap.png`
- `diagrams/matrix_mrr_heatmap.png`

## Execution Instructions
```bash
python scripts/evaluate_methods.py
```

## Additional Outputs
- `results/metrics.csv` (all 9 method-metric combinations)
- `results/per_query_results.csv`
- `results/final_report.md`
- `results/final_report.html`
