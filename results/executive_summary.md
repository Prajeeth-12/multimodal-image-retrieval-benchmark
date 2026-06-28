# Executive Summary

## Objective
Benchmark CLIP image retrieval, manual caption retrieval, and VLM-caption retrieval on the restaurant dataset.

## What Was Completed
1. Verified dataset/embedding integrity end-to-end.
2. Implemented `scripts/evaluate_methods.py` for automated benchmarking.
3. Computed Top-1, Top-3, Top-5, and MRR for all 25 queries and 3 methods.
4. Generated `results/metrics.csv` and `results/per_query_results.csv`.
5. Produced publication charts in `results/charts/`.
6. Authored research report (`results/final_report.md`) and PDF-ready HTML report (`results/final_report.html`).

## Key Findings
- Best method by MRR: **Manual**
- Weakest method by MRR: **CLIP**
- Manual vs VLM MRR delta (VLM - Manual): **-0.0357**
- CLIP MRR gap vs best method: **0.2419**

## Submission Readiness
All required evaluation artifacts and analysis outputs are generated and stored under `results/`, ready for mentor review and presentation.
