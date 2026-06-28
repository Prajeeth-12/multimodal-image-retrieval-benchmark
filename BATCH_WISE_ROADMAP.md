# Batch-Wise Push File List

Only these file groups should be pushed per batch.

## Batch 1 - Project Setup + Dataset Tooling
- `.gitignore`
- `README.md`
- `requirements.txt`
- `scripts/bad_image_count.py`
- `scripts/check_images.py`
- `scripts/delete_low_res.py`
- `scripts/find_duplicates.py`
- `scripts/generate_metadata.py`
- `scripts/rename_images.py`
- `metadata/metadata.csv`

## Batch 2 - CLIP End-to-End
- `scripts/generate_clip_embeddings.py`
- `scripts/test_clip_retrieval.py`
- `embeddings/clip/image_embeddings.npy`
- `embeddings/clip/image_paths.csv`

## Batch 3 - Manual End-to-End
- `scripts/generate_manual_captions.py`
- `scripts/generate_manual_embeddings.py`
- `scripts/test_manual_retrieval.py`
- `metadata/manual_captions.csv`
- `embeddings/manual/manual_embeddings.npy`
- `embeddings/manual/manual_paths.csv`

## Batch 4 - VLM End-to-End
- `scripts/generate_vlm_embeddings.py`
- `scripts/test_vlm_retrieval.py`
- `metadata/vlm_captions.csv`
- `embeddings/vlm/vlm_embeddings.npy`
- `embeddings/vlm/vlm_paths.csv`

## Batch 5 - Shared Benchmark + Comparison Reports
- `metadata/benchmark_queries.csv`
- `scripts/evaluate_methods.py`
- `results/metrics.csv`
- `results/per_query_results.csv`
- `results/integrity_checks.json`
- `results/executive_summary.md`
- `results/final_report.md`
- `results/final_report.html`
- `results/charts/accuracy_comparison_bar.png`
- `results/charts/category_top1_comparison.png`
- `results/charts/methodology_diagram.png`
- `results/charts/mrr_comparison_chart.png`
- `results/charts/retrieval_performance_summary_chart.png`
- `results/charts/topk_comparison_chart.png`

## Do Not Push
- `scripts/__pycache__/generate_clip_embeddings.cpython-310.pyc`

