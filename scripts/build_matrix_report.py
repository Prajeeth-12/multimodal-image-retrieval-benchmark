from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
REPORT_DIR = RESULTS_DIR / "3x3_matrix_report"
MATRICES_DIR = REPORT_DIR / "matrices"
DIAGRAMS_DIR = REPORT_DIR / "diagrams"

METHOD_ORDER = ["CLIP", "Manual", "VLM"]
SIMILARITY_ORDER = ["cosine", "euclidean", "dot_product"]
DISPLAY_SIM = {"cosine": "Cosine", "euclidean": "Euclidean", "dot_product": "Dot Product"}


def build_matrix(metrics: pd.DataFrame, value_col: str) -> pd.DataFrame:
    matrix = metrics.pivot(index="method", columns="similarity_metric", values=value_col)
    matrix = matrix.reindex(index=METHOD_ORDER, columns=SIMILARITY_ORDER)
    return matrix.reset_index()


def format_matrix_for_markdown(matrix_df: pd.DataFrame) -> pd.DataFrame:
    out = matrix_df.copy()
    for col in ["cosine", "euclidean", "dot_product"]:
        out[col] = out[col].map(lambda x: f"{float(x):.4f}")
    return out


def write_markdown_table(matrix_df: pd.DataFrame, title: str, path: Path) -> None:
    df = format_matrix_for_markdown(matrix_df)
    cols = ["method", "cosine", "euclidean", "dot_product"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df[cols].iterrows():
        rows.append("| " + " | ".join([str(r[c]) for c in cols]) + " |")

    content = [f"# {title}", "", header, sep] + rows + [""]
    path.write_text("\n".join(content), encoding="utf-8")


def save_heatmap(matrix_df: pd.DataFrame, title: str, output_name: str) -> None:
    values = matrix_df[["cosine", "euclidean", "dot_product"]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([DISPLAY_SIM[c] for c in ["cosine", "euclidean", "dot_product"]], fontsize=10)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(matrix_df["method"].tolist(), fontsize=10)
    ax.set_xlabel("Similarity Metric", fontsize=10)
    ax.set_ylabel("Embedding Method", fontsize=10)
    ax.set_title(title, fontsize=12, weight="bold")

    max_val = np.max(values)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            is_best = abs(val - max_val) < 1e-12
            text = f"{val:.3f}" + ("  *" if is_best else "")
            color = "white" if val > 0.65 else "black"
            bbox = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.45) if is_best else None
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color=color,
                fontweight="bold" if is_best else "normal",
                bbox=bbox,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score (higher is better)", fontsize=9)

    fig.text(0.5, 0.01, "* marks best score in this matrix", ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(DIAGRAMS_DIR / output_name, dpi=220)
    plt.close(fig)


def get_interpretation(top1: pd.DataFrame, top3: pd.DataFrame, top5: pd.DataFrame, mrr: pd.DataFrame) -> str:
    def best_combo(df: pd.DataFrame, metric_name: str) -> str:
        long_df = df.melt(
            id_vars=["method"],
            value_vars=["cosine", "euclidean", "dot_product"],
            var_name="sim",
            value_name="score",
        )
        best = long_df.sort_values("score", ascending=False).iloc[0]
        return f"- Best {metric_name}: {best['method']} + {best['sim']} = {best['score']:.4f}"

    top1_best_method = (
        top1.set_index("method")[["cosine", "euclidean", "dot_product"]]
        .mean(axis=1)
        .sort_values(ascending=False)
        .index[0]
    )
    mrr_best_method = (
        mrr.set_index("method")[["cosine", "euclidean", "dot_product"]]
        .mean(axis=1)
        .sort_values(ascending=False)
        .index[0]
    )

    lines = [
        "# Interpretation",
        "",
        "Easy reading guide:",
        "- Rows compare methods: CLIP vs Manual vs VLM.",
        "- Columns compare similarity metrics: cosine vs euclidean vs dot_product.",
        "- Higher value means better retrieval.",
        "",
        "Key findings:",
        best_combo(top1, "Top-1"),
        best_combo(top3, "Top-3"),
        best_combo(top5, "Top-5"),
        best_combo(mrr, "MRR"),
        f"- Best overall method trend: {top1_best_method} (Top-1), {mrr_best_method} (MRR).",
        "",
        "Note: In this run, each method has the same score across cosine/euclidean/dot_product.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    metrics_path = RESULTS_DIR / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MATRICES_DIR.mkdir(parents=True, exist_ok=True)
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(metrics_path)

    top1 = build_matrix(metrics, "top1_accuracy")
    top3 = build_matrix(metrics, "top3_accuracy")
    top5 = build_matrix(metrics, "top5_accuracy")
    mrr = build_matrix(metrics, "mrr")

    write_markdown_table(top1, "3x3 Matrix - Top-1 Accuracy", MATRICES_DIR / "matrix_top1_accuracy.md")
    write_markdown_table(top3, "3x3 Matrix - Top-3 Accuracy", MATRICES_DIR / "matrix_top3_accuracy.md")
    write_markdown_table(top5, "3x3 Matrix - Top-5 Accuracy", MATRICES_DIR / "matrix_top5_accuracy.md")
    write_markdown_table(mrr, "3x3 Matrix - MRR", MATRICES_DIR / "matrix_mrr.md")

    save_heatmap(top1, "3x3 Matrix: Top-1 Accuracy", "matrix_top1_accuracy_heatmap.png")
    save_heatmap(top3, "3x3 Matrix: Top-3 Accuracy", "matrix_top3_accuracy_heatmap.png")
    save_heatmap(top5, "3x3 Matrix: Top-5 Accuracy", "matrix_top5_accuracy_heatmap.png")
    save_heatmap(mrr, "3x3 Matrix: MRR", "matrix_mrr_heatmap.png")

    interpretation = get_interpretation(top1, top3, top5, mrr)
    (REPORT_DIR / "interpretation.md").write_text(interpretation, encoding="utf-8")

    for old in [
        MATRICES_DIR / "matrix_top1_accuracy.csv",
        MATRICES_DIR / "matrix_top3_accuracy.csv",
        MATRICES_DIR / "matrix_top5_accuracy.csv",
        MATRICES_DIR / "matrix_mrr.csv",
        MATRICES_DIR / "metrics_matrix_top1.csv",
        MATRICES_DIR / "metrics_matrix_mrr.csv",
    ]:
        if old.exists():
            try:
                old.unlink()
            except PermissionError:
                pass

    readme = """# 3x3 Matrix Comparison Report

This folder is the primary comparison package for 3 methods x 3 similarity metrics.

## Matrices (Markdown)
- matrices/matrix_top1_accuracy.md
- matrices/matrix_top3_accuracy.md
- matrices/matrix_top5_accuracy.md
- matrices/matrix_mrr.md

## Diagrams (Easy-to-read Heatmaps)
- diagrams/matrix_top1_accuracy_heatmap.png
- diagrams/matrix_top3_accuracy_heatmap.png
- diagrams/matrix_top5_accuracy_heatmap.png
- diagrams/matrix_mrr_heatmap.png

## Interpretation
- interpretation.md
"""
    (REPORT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"Saved matrix report to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
