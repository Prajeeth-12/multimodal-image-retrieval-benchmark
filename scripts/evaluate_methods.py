import os
import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
from transformers import CLIPModel, CLIPProcessor
from sentence_transformers import SentenceTransformer


# Keep model loading fully offline-safe once cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Restaurant_food_datasets"
METADATA_DIR = ROOT / "metadata"
EMBEDDINGS_DIR = ROOT / "embeddings"
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"

SIMILARITY_METRICS = ["cosine", "euclidean", "dot_product"]
METHOD_ORDER = ["CLIP", "Manual", "VLM"]


STOPWORDS = {
    "a", "an", "the", "and", "or", "with", "of", "to", "in", "on", "for", "at", "by",
    "style", "authentic", "served", "restaurant", "dining", "area", "setting", "large",
    "modern", "traditional", "cozy", "fine", "well", "lit", "brown", "golden", "refreshing",
}

FOOD_TERMS = {
    "biryani", "idli", "sambar", "dosa", "vada", "paneer", "parotta", "gulab", "jamun",
    "chicken", "fish", "curry", "kesari", "chutney", "grill", "grilled", "roast", "meal",
    "thali", "poori", "sweet", "buffet", "mocktail", "rice", "kebab", "kothu",
}
AMBIENCE_TERMS = {
    "ambiance", "ambience", "interior", "decor", "dining", "hall", "lights", "lighting",
    "seating", "spacious", "cozy", "modern", "traditional", "exterior", "facade",
}
EXPERIENCE_TERMS = {
    "family", "group", "dinner", "tableside", "live", "grilling", "banana", "leaf",
    "brunch", "service", "counter",
}


def normalize_path(path: str) -> str:
    p = str(path).replace("\\", "/").strip()
    while p.startswith("../") or p.startswith("..\\"):
        p = p[3:]
    if p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


def classify_image_category(text: str) -> str:
    tokens = set(tokenize(text))
    food_score = len(tokens & FOOD_TERMS)
    ambience_score = len(tokens & AMBIENCE_TERMS)
    exp_score = len(tokens & EXPERIENCE_TERMS)
    if ambience_score > food_score and ambience_score >= exp_score:
        return "Ambience"
    if exp_score > food_score and exp_score > ambience_score:
        return "Dining Experience"
    return "Food"


def validate_integrity() -> Dict[str, object]:
    image_files = sorted(
        p for p in DATASET_DIR.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    image_set = {normalize_path(str(p.relative_to(ROOT))) for p in image_files}

    clip_emb = np.load(EMBEDDINGS_DIR / "clip" / "image_embeddings.npy")
    manual_emb = np.load(EMBEDDINGS_DIR / "manual" / "manual_embeddings.npy")
    vlm_emb = np.load(EMBEDDINGS_DIR / "vlm" / "vlm_embeddings.npy")

    clip_paths = pd.read_csv(EMBEDDINGS_DIR / "clip" / "image_paths.csv")
    manual_paths = pd.read_csv(EMBEDDINGS_DIR / "manual" / "manual_paths.csv")
    vlm_paths = pd.read_csv(EMBEDDINGS_DIR / "vlm" / "vlm_paths.csv")

    clip_set = {normalize_path(x) for x in clip_paths["image_path"].tolist()}
    manual_set = {normalize_path(x) for x in manual_paths["image_path"].tolist()}
    vlm_set = {normalize_path(x) for x in vlm_paths["image_path"].tolist()}

    checks = {
        "dataset_image_count": len(image_set),
        "clip_embeddings_shape": list(clip_emb.shape),
        "manual_embeddings_shape": list(manual_emb.shape),
        "vlm_embeddings_shape": list(vlm_emb.shape),
        "clip_path_count": len(clip_set),
        "manual_path_count": len(manual_set),
        "vlm_path_count": len(vlm_set),
        "clip_dim": int(clip_emb.shape[1]),
        "manual_dim": int(manual_emb.shape[1]),
        "vlm_dim": int(vlm_emb.shape[1]),
        "clip_vs_manual_paths_match": clip_set == manual_set,
        "manual_vs_vlm_paths_match": manual_set == vlm_set,
        "missing_from_dataset_clip": sorted(list(clip_set - image_set)),
        "missing_from_dataset_manual": sorted(list(manual_set - image_set)),
        "missing_from_dataset_vlm": sorted(list(vlm_set - image_set)),
        "missing_from_clip_embeddings": sorted(list(image_set - clip_set)),
        "missing_from_manual_embeddings": sorted(list(image_set - manual_set)),
        "missing_from_vlm_embeddings": sorted(list(image_set - vlm_set)),
    }
    checks["all_checks_passed"] = (
        checks["dataset_image_count"] > 0
        and checks["clip_path_count"] == checks["manual_path_count"] == checks["vlm_path_count"] == checks["dataset_image_count"]
        and checks["clip_vs_manual_paths_match"]
        and checks["manual_vs_vlm_paths_match"]
        and checks["clip_dim"] == 512
        and checks["manual_dim"] == 384
        and checks["vlm_dim"] == 384
        and not checks["missing_from_dataset_clip"]
        and not checks["missing_from_dataset_manual"]
        and not checks["missing_from_dataset_vlm"]
        and not checks["missing_from_clip_embeddings"]
        and not checks["missing_from_manual_embeddings"]
        and not checks["missing_from_vlm_embeddings"]
    )
    return checks


def load_retrieval_assets():
    clip_embeddings = np.load(EMBEDDINGS_DIR / "clip" / "image_embeddings.npy")
    manual_embeddings = np.load(EMBEDDINGS_DIR / "manual" / "manual_embeddings.npy")
    vlm_embeddings = np.load(EMBEDDINGS_DIR / "vlm" / "vlm_embeddings.npy")

    clip_paths = [normalize_path(x) for x in pd.read_csv(EMBEDDINGS_DIR / "clip" / "image_paths.csv")["image_path"].tolist()]
    manual_paths = [normalize_path(x) for x in pd.read_csv(EMBEDDINGS_DIR / "manual" / "manual_paths.csv")["image_path"].tolist()]
    vlm_paths = [normalize_path(x) for x in pd.read_csv(EMBEDDINGS_DIR / "vlm" / "vlm_paths.csv")["image_path"].tolist()]

    return {
        "clip": (clip_embeddings, clip_paths),
        "manual": (manual_embeddings, manual_paths),
        "vlm": (vlm_embeddings, vlm_paths),
    }


def build_image_text_table() -> pd.DataFrame:
    manual = pd.read_csv(METADATA_DIR / "manual_captions.csv").copy()
    vlm = pd.read_csv(METADATA_DIR / "vlm_captions.csv").copy()

    manual["image_path"] = manual["image_path"].map(normalize_path)
    vlm["image_path"] = vlm["image_path"].map(normalize_path)

    merged = manual.merge(vlm, on="image_path", how="outer", suffixes=("_manual", "_vlm"))
    merged["caption_manual"] = merged["caption_manual"].fillna("")
    merged["caption_vlm"] = merged["caption_vlm"].fillna("")
    merged["combined_text"] = (
        merged["caption_manual"] + " " + merged["caption_vlm"] + " " + merged["image_path"]
    ).str.strip()
    merged["predicted_category"] = merged["combined_text"].map(classify_image_category)
    return merged


def build_relevance_set(query: str, query_category: str, image_text_df: pd.DataFrame) -> Tuple[Set[str], List[Tuple[str, float]]]:
    q_tokens = set(tokenize(query))
    scored: List[Tuple[str, float, str]] = []

    for row in image_text_df.itertuples(index=False):
        text_tokens = set(tokenize(row.combined_text))
        overlap = len(q_tokens & text_tokens)
        denom = max(1, len(q_tokens))
        overlap_ratio = overlap / denom
        phrase_bonus = 0.2 if str(query).lower() in str(row.combined_text).lower() else 0.0
        filename_bonus = 0.1 if len(q_tokens & set(tokenize(row.image_path))) > 0 else 0.0
        score = overlap_ratio + phrase_bonus + filename_bonus
        scored.append((row.image_path, score, row.predicted_category))

    scored.sort(key=lambda x: x[1], reverse=True)

    relevant = {
        p for p, s, c in scored
        if (c == query_category and s >= 0.2) or (s >= 0.45)
    }

    if not relevant:
        same_cat = [(p, s) for p, s, c in scored if c == query_category]
        seed = same_cat if same_cat else [(p, s) for p, s, _ in scored]
        fallback_n = min(3, len(seed))
        relevant = {p for p, _ in seed[:fallback_n]}

    return relevant, [(p, s) for p, s, _ in scored[:10]]


def rank_metrics(sorted_paths: List[str], relevant_set: Set[str]) -> Dict[str, float]:
    first_rel_rank = None
    for rank, path in enumerate(sorted_paths, start=1):
        if path in relevant_set:
            first_rel_rank = rank
            break
    top1 = 1.0 if first_rel_rank == 1 else 0.0
    top3 = 1.0 if first_rel_rank is not None and first_rel_rank <= 3 else 0.0
    top5 = 1.0 if first_rel_rank is not None and first_rel_rank <= 5 else 0.0
    mrr = 1.0 / first_rel_rank if first_rel_rank is not None else 0.0
    return {
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "mrr": mrr,
        "first_relevant_rank": float(first_rel_rank) if first_rel_rank is not None else np.nan,
    }


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return arr / denom


def compute_scores(embeddings: np.ndarray, query_embedding: np.ndarray, similarity_metric: str) -> np.ndarray:
    if similarity_metric == "dot_product":
        return (embeddings @ query_embedding.T).reshape(-1)

    if similarity_metric == "cosine":
        emb_n = l2_normalize(embeddings)
        q_n = l2_normalize(query_embedding)
        return (emb_n @ q_n.T).reshape(-1)

    if similarity_metric == "euclidean":
        diff = embeddings - query_embedding
        dists = np.linalg.norm(diff, axis=1)
        return -dists

    raise ValueError(f"Unsupported similarity metric: {similarity_metric}")


def build_similarity_matrix(metrics_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    matrix = metrics_df.pivot(index="method", columns="similarity_metric", values=value_col)
    matrix = matrix.reindex(index=METHOD_ORDER, columns=SIMILARITY_METRICS)
    matrix = matrix.reset_index()
    return matrix


def evaluate_all() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], pd.DataFrame, pd.DataFrame]:
    queries = pd.read_csv(METADATA_DIR / "benchmark_queries.csv")
    assets = load_retrieval_assets()
    image_text_df = build_image_text_table()

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        local_files_only=True,
        use_safetensors=False,
    )
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        local_files_only=True,
    )
    text_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

    per_query_rows = []
    integrity = validate_integrity()

    for row in queries.itertuples(index=False):
        query = row.query
        category = row.category

        relevant_set, silver_top = build_relevance_set(query, category, image_text_df)

        # CLIP query encoding
        clip_inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
        with torch.no_grad():
            clip_features = clip_model.get_text_features(**clip_inputs)
            if not isinstance(clip_features, torch.Tensor):
                if hasattr(clip_features, "pooler_output"):
                    clip_features = clip_features.pooler_output
                else:
                    raise ValueError(f"Unexpected CLIP output type: {type(clip_features)}")
        clip_query = clip_features.detach().cpu().numpy()

        # Text-embedding query encoding for manual and VLM
        txt_query = text_model.encode([query], normalize_embeddings=False)

        method_payload = {
            "CLIP": (assets["clip"][0], assets["clip"][1], clip_query),
            "Manual": (assets["manual"][0], assets["manual"][1], txt_query),
            "VLM": (assets["vlm"][0], assets["vlm"][1], txt_query),
        }

        for method_name, (emb, paths, q_emb) in method_payload.items():
            for similarity_metric in SIMILARITY_METRICS:
                scores = compute_scores(emb, q_emb, similarity_metric)
                order = np.argsort(scores)[::-1]
                sorted_paths = [paths[idx] for idx in order]
                top5_paths = sorted_paths[:5]
                top5_scores = [float(scores[idx]) for idx in order[:5]]
                m = rank_metrics(sorted_paths, relevant_set)

                per_query_rows.append(
                    {
                        "query": query,
                        "category": category,
                        "method": method_name,
                        "similarity_metric": similarity_metric,
                        "relevant_count": len(relevant_set),
                        "first_relevant_rank": m["first_relevant_rank"],
                        "top1": m["top1"],
                        "top3": m["top3"],
                        "top5": m["top5"],
                        "mrr": m["mrr"],
                        "top1_path": top5_paths[0] if top5_paths else "",
                        "top1_score": top5_scores[0] if top5_scores else np.nan,
                        "top5_paths": " | ".join(top5_paths),
                        "silver_relevance_top_examples": " | ".join([f"{p}:{s:.2f}" for p, s in silver_top[:3]]),
                    }
                )

    per_query_df = pd.DataFrame(per_query_rows)

    metrics_df = (
        per_query_df
        .groupby(["method", "similarity_metric"], as_index=False)[["top1", "top3", "top5", "mrr"]]
        .mean()
        .rename(columns={"top1": "top1_accuracy", "top3": "top3_accuracy", "top5": "top5_accuracy"})
    )
    metrics_df["top1_accuracy_pct"] = (metrics_df["top1_accuracy"] * 100).round(2)
    metrics_df["top3_accuracy_pct"] = (metrics_df["top3_accuracy"] * 100).round(2)
    metrics_df["top5_accuracy_pct"] = (metrics_df["top5_accuracy"] * 100).round(2)
    metrics_df["mrr"] = metrics_df["mrr"].round(4)

    metrics_df["method"] = pd.Categorical(metrics_df["method"], METHOD_ORDER, ordered=True)
    metrics_df["similarity_metric"] = pd.Categorical(metrics_df["similarity_metric"], SIMILARITY_METRICS, ordered=True)
    metrics_df = metrics_df.sort_values(["method", "similarity_metric"]).reset_index(drop=True)

    matrix_top1_df = build_similarity_matrix(metrics_df, "top1_accuracy")
    matrix_mrr_df = build_similarity_matrix(metrics_df, "mrr")

    return metrics_df, per_query_df, integrity, matrix_top1_df, matrix_mrr_df


def make_charts(metrics_df: pd.DataFrame, per_query_df: pd.DataFrame) -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Keep existing charts comparable by using dot-product slice.
    chart_df = metrics_df[metrics_df["similarity_metric"] == "dot_product"].copy()

    methods = chart_df["method"].tolist()
    top1 = chart_df["top1_accuracy"].tolist()
    top3 = chart_df["top3_accuracy"].tolist()
    top5 = chart_df["top5_accuracy"].tolist()
    mrr = chart_df["mrr"].tolist()

    # 1) Accuracy comparison bar chart (Top-1)
    plt.figure(figsize=(8, 5))
    plt.bar(methods, top1, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    plt.title("Top-1 Accuracy by Retrieval Method (Dot Product)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for i, v in enumerate(top1):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "accuracy_comparison_bar.png", dpi=220)
    plt.close()

    # 2) Top-K comparison chart
    plt.figure(figsize=(8, 5))
    ks = [1, 3, 5]
    for method in methods:
        row = chart_df[chart_df["method"] == method].iloc[0]
        vals = [row["top1_accuracy"], row["top3_accuracy"], row["top5_accuracy"]]
        plt.plot(ks, vals, marker="o", linewidth=2, label=method)
    plt.title("Top-K Accuracy Comparison (Dot Product)")
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.xticks([1, 3, 5])
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "topk_comparison_chart.png", dpi=220)
    plt.close()

    # 3) MRR comparison chart
    plt.figure(figsize=(8, 5))
    plt.bar(methods, mrr, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    plt.title("MRR by Retrieval Method (Dot Product)")
    plt.ylabel("MRR")
    plt.ylim(0, 1)
    for i, v in enumerate(mrr):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "mrr_comparison_chart.png", dpi=220)
    plt.close()

    # 4) Retrieval performance summary chart (grouped)
    x = np.arange(len(methods))
    w = 0.2
    plt.figure(figsize=(10, 5))
    plt.bar(x - 1.5 * w, top1, width=w, label="Top-1")
    plt.bar(x - 0.5 * w, top3, width=w, label="Top-3")
    plt.bar(x + 0.5 * w, top5, width=w, label="Top-5")
    plt.bar(x + 1.5 * w, mrr, width=w, label="MRR")
    plt.xticks(x, methods)
    plt.ylim(0, 1)
    plt.title("Retrieval Performance Summary (Dot Product)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "retrieval_performance_summary_chart.png", dpi=220)
    plt.close()

    # Methodology diagram
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    labels = [
        "Benchmark Queries\n(metadata/benchmark_queries.csv)",
        "Query Embedding\n(CLIP / MiniLM)",
        "Similarity Search\n(cosine / euclidean / dot)",
        "Metrics\n(Top-1/3/5, MRR)",
        "Analysis & Reports\n(csv + charts + docs)",
    ]
    xs = [0.02, 0.22, 0.42, 0.62, 0.82]
    for x0, lbl in zip(xs, labels):
        box = FancyBboxPatch((x0, 0.3), 0.15, 0.4, boxstyle="round,pad=0.02", linewidth=1.2)
        ax.add_patch(box)
        ax.text(x0 + 0.075, 0.5, lbl, ha="center", va="center", fontsize=9)
    for x0 in [0.17, 0.37, 0.57, 0.77]:
        ax.annotate("", xy=(x0 + 0.04, 0.5), xytext=(x0, 0.5), arrowprops=dict(arrowstyle="->", lw=1.3))
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "methodology_diagram.png", dpi=220)
    plt.close()

    # Extra: category-wise Top-1 by method for analysis visuals (dot-product slice)
    cat_source = per_query_df[per_query_df["similarity_metric"] == "dot_product"]
    cat_table = cat_source.groupby(["category", "method"], as_index=False)["top1"].mean()
    pivot = cat_table.pivot(index="category", columns="method", values="top1").fillna(0)
    pivot.plot(kind="bar", figsize=(9, 5))
    plt.title("Top-1 Accuracy by Query Category (Dot Product)")
    plt.ylabel("Top-1 Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "category_top1_comparison.png", dpi=220)
    plt.close()


def build_analysis(metrics_df: pd.DataFrame, per_query_df: pd.DataFrame) -> Dict[str, object]:
    analysis_metric = "dot_product"
    metric_slice = metrics_df[metrics_df["similarity_metric"] == analysis_metric].copy()

    best_method = metric_slice.sort_values("mrr", ascending=False).iloc[0]["method"]
    weakest_method = metric_slice.sort_values("mrr", ascending=True).iloc[0]["method"]

    per_query_metric = per_query_df[per_query_df["similarity_metric"] == analysis_metric].copy()

    failures = (
        per_query_metric[per_query_metric["top1"] == 0]
        .sort_values(["method", "first_relevant_rank"], ascending=[True, True])
        .head(12)
    )

    qual_success = (
        per_query_metric[per_query_metric["top1"] == 1]
        .sort_values(["method", "top1_score"], ascending=[True, False])
        .groupby("method")
        .head(2)
    )

    by_category = (
        per_query_metric
        .groupby(["category", "method"], as_index=False)[["top1", "top3", "top5", "mrr"]]
        .mean()
    )

    manual = metric_slice[metric_slice["method"] == "Manual"].iloc[0]
    vlm = metric_slice[metric_slice["method"] == "VLM"].iloc[0]
    clip = metric_slice[metric_slice["method"] == "CLIP"].iloc[0]

    best_combo = metrics_df.sort_values("mrr", ascending=False).iloc[0]

    return {
        "analysis_metric": analysis_metric,
        "best_method": best_method,
        "weakest_method": weakest_method,
        "failures": failures,
        "qual_success": qual_success,
        "by_category": by_category,
        "manual_vs_vlm": {
            "top1_delta": float(vlm["top1_accuracy"] - manual["top1_accuracy"]),
            "top3_delta": float(vlm["top3_accuracy"] - manual["top3_accuracy"]),
            "top5_delta": float(vlm["top5_accuracy"] - manual["top5_accuracy"]),
            "mrr_delta": float(vlm["mrr"] - manual["mrr"]),
        },
        "clip_vs_best_mrr_gap": float(metric_slice["mrr"].max() - clip["mrr"]),
        "best_method_metric_combo": {
            "method": str(best_combo["method"]),
            "similarity_metric": str(best_combo["similarity_metric"]),
            "mrr": float(best_combo["mrr"]),
        },
    }


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for row in df.itertuples(index=False):
        vals = [str(v) for v in row]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def write_report_files(
    metrics_df: pd.DataFrame,
    per_query_df: pd.DataFrame,
    integrity: Dict[str, object],
    analysis: Dict[str, object],
    matrix_top1_df: pd.DataFrame,
    matrix_mrr_df: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_table_md = df_to_markdown(metrics_df.round(4))
    matrix_top1_md = df_to_markdown(matrix_top1_df.round(4))
    matrix_mrr_md = df_to_markdown(matrix_mrr_df.round(4))
    category_table_md = df_to_markdown(analysis["by_category"].round(4))
    failure_table_md = df_to_markdown(
        analysis["failures"][["query", "category", "method", "first_relevant_rank", "top1_path"]]
    )
    success_table_md = df_to_markdown(
        analysis["qual_success"][["query", "category", "method", "top1_path", "top1_score"]]
    )

    report_md = f"""# Restaurant Image Retrieval Benchmark Report

## Abstract
This report benchmarks three restaurant image retrieval methods over the same 103-image corpus and 25 benchmark queries: CLIP image embeddings, manual caption embeddings, and VLM-generated caption embeddings. We evaluate with Top-1/Top-3/Top-5 Accuracy and Mean Reciprocal Rank (MRR), then analyze category behavior, failure modes, and practical limitations.

## Introduction
The project objective is to compare multimodal and text-based retrieval quality on a restaurant-focused dataset spanning food, ambience, and dining experience intents.

## Dataset Description
- Total images: **{integrity["dataset_image_count"]}**
- Benchmark queries: **{per_query_df['query'].nunique()}**
- Query categories: **Food, Ambience, Dining Experience**
- Embedding dimensions:
  - CLIP image vectors: **{integrity["clip_dim"]}**
  - Manual caption vectors: **{integrity["manual_dim"]}**
  - VLM caption vectors: **{integrity["vlm_dim"]}**

## Methodology
1. Load benchmark queries from `metadata/benchmark_queries.csv`.
2. Retrieve against each embedding index independently.
3. Evaluate all three similarity metrics: cosine, euclidean, and dot product.
4. Build transparent silver relevance sets from existing captions and query-category constraints.
5. Compute Top-1, Top-3, Top-5, and MRR per query and aggregate per method-metric pair.

![Methodology Diagram](charts/methodology_diagram.png)
*Figure: End-to-end evaluation pipeline.*

## Experimental Setup
- Runtime mode: offline-safe model loading (`local_files_only=True`)
- Similarities: cosine, euclidean, dot product
- Top-K reported at K = 1, 3, 5
- Query count: {per_query_df['query'].nunique()}

## Results: Full Method x Similarity Table
{metrics_table_md}

## 3x3 Matrix Report (Method x Similarity)
Top-1 Accuracy Matrix:
{matrix_top1_md}

MRR Matrix:
{matrix_mrr_md}

## Discussion
- Best method (dot-product slice): **{analysis["best_method"]}**
- Weakest method (dot-product slice): **{analysis["weakest_method"]}**
- Best overall method-metric combo (by MRR): **{analysis["best_method_metric_combo"]["method"]} + {analysis["best_method_metric_combo"]["similarity_metric"]}** with **MRR={analysis["best_method_metric_combo"]["mrr"]:.4f}**
- CLIP MRR gap vs best (dot-product slice): **{analysis["clip_vs_best_mrr_gap"]:.4f}**
- Manual vs VLM deltas (VLM - Manual) for dot-product:
  - Top-1: **{analysis["manual_vs_vlm"]["top1_delta"]:+.4f}**
  - Top-3: **{analysis["manual_vs_vlm"]["top3_delta"]:+.4f}**
  - Top-5: **{analysis["manual_vs_vlm"]["top5_delta"]:+.4f}**
  - MRR: **{analysis["manual_vs_vlm"]["mrr_delta"]:+.4f}**

Category-level behavior (dot-product):
{category_table_md}

## Failure Analysis (dot-product)
Representative failure cases (top-1 miss):
{failure_table_md}

Qualitative strong examples (top-1 hit):
{success_table_md}

## Conclusion
The benchmark now reports all 3 methods across all 3 similarity metrics (3x3), with reproducible per-query outputs and summary matrices.
"""

    (RESULTS_DIR / "final_report.md").write_text(report_md, encoding="utf-8")

    metrics_html = metrics_df.round(4).to_html(index=False, classes="table")
    matrix_top1_html = matrix_top1_df.round(4).to_html(index=False, classes="table")
    matrix_mrr_html = matrix_mrr_df.round(4).to_html(index=False, classes="table")

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Restaurant Retrieval Benchmark Report</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 32px; line-height: 1.5; color: #111; }}
    h1, h2 {{ color: #0f172a; }}
    .table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px 0; }}
    .table th, .table td {{ border: 1px solid #ddd; padding: 8px; font-size: 14px; }}
    .table th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
  <h1>Restaurant Image Retrieval Benchmark Report</h1>
  <p><strong>Images:</strong> {integrity["dataset_image_count"]} | <strong>Queries:</strong> {per_query_df['query'].nunique()} | <strong>Similarities:</strong> cosine, euclidean, dot product</p>

  <h2>Full Method x Similarity Metrics</h2>
  {metrics_html}

  <h2>3x3 Matrix (Top-1 Accuracy)</h2>
  {matrix_top1_html}

  <h2>3x3 Matrix (MRR)</h2>
  {matrix_mrr_html}
</body>
</html>
"""
    (RESULTS_DIR / "final_report.html").write_text(report_html, encoding="utf-8")

    readme = f"""# Restaurant Image Retrieval Benchmark

## Project Overview
This project benchmarks three retrieval approaches on a restaurant image corpus:
1. CLIP Image Embeddings
2. Manual Human-Written Caption Embeddings
3. VLM (GPT-generated) Caption Embeddings

Each method is evaluated with three similarity metrics:
- cosine
- euclidean
- dot product

## Execution Instructions
```bash
python scripts/evaluate_methods.py
```

Generated outputs:
- `results/metrics.csv` (all 9 method-metric combinations)
- `results/metrics_matrix_top1.csv` (3x3 matrix)
- `results/metrics_matrix_mrr.csv` (3x3 matrix)
- `results/per_query_results.csv`
- `results/final_report.md`
- `results/final_report.html`

## 3x3 MRR Matrix
{matrix_mrr_md}
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_df, per_query_df, integrity, matrix_top1_df, matrix_mrr_df = evaluate_all()
    analysis = build_analysis(metrics_df, per_query_df)

    metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    matrix_top1_df.to_csv(RESULTS_DIR / "metrics_matrix_top1.csv", index=False)
    matrix_mrr_df.to_csv(RESULTS_DIR / "metrics_matrix_mrr.csv", index=False)
    per_query_df.to_csv(RESULTS_DIR / "per_query_results.csv", index=False)
    (RESULTS_DIR / "integrity_checks.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    make_charts(metrics_df, per_query_df)
    write_report_files(metrics_df, per_query_df, integrity, analysis, matrix_top1_df, matrix_mrr_df)

    print("Evaluation completed successfully.")
    print(f"Saved metrics: {RESULTS_DIR / 'metrics.csv'}")
    print(f"Saved 3x3 Top-1 matrix: {RESULTS_DIR / 'metrics_matrix_top1.csv'}")
    print(f"Saved 3x3 MRR matrix: {RESULTS_DIR / 'metrics_matrix_mrr.csv'}")
    print(f"Saved per-query results: {RESULTS_DIR / 'per_query_results.csv'}")
    print(f"Saved charts: {CHARTS_DIR}")
    print(f"Saved report: {RESULTS_DIR / 'final_report.md'}")
    print(f"Saved html report: {RESULTS_DIR / 'final_report.html'}")


if __name__ == "__main__":
    main()
