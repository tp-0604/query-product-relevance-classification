"""
evaluate.py
-----------
Metrics computation, confusion matrix, per-class F1,
and comparative analysis plots. All figures saved to outputs/figures/.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Exact", "Substitute", "Complement", "Irrelevant"]
FIGURES_DIR = "outputs/figures"
REPORTS_DIR = "outputs/reports"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision_weighted": round(precision, 4),
        "recall_weighted": round(recall, 4),
        "f1_weighted": round(f1, 4),
        "f1_macro": round(f1_macro, 4),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Accuracy:            {acc:.4f}")
    logger.info(f"Precision (weighted):{precision:.4f}")
    logger.info(f"Recall (weighted):   {recall:.4f}")
    logger.info(f"F1 (weighted):       {f1:.4f}")
    logger.info(f"F1 (macro):          {f1_macro:.4f}")
    logger.info(f"{'='*50}")

    return metrics


def save_classification_report(y_true, y_pred, model_name: str):
    report = classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0
    )
    df = pd.DataFrame(report).transpose()
    path = os.path.join(REPORTS_DIR, f"classification_report_{model_name}.csv")
    df.to_csv(path)
    logger.info(f"Classification report saved: {path}")

    print(f"\nClassification Report — {model_name}")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))

    return df


def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=axes[0]
    )
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
        ax=axes[1]
    )
    axes[1].set_title("Normalized")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {path}")


def plot_per_class_f1(y_true, y_pred, model_name: str):
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(LABEL_NAMES, f1s, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"])
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-Class F1 Score — {model_name}", fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_xlabel("Class")

    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"f1_per_class_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Per-class F1 plot saved: {path}")


def plot_comparative_analysis(all_metrics: list):
    """
    Generate side-by-side bar plots comparing all models across metrics.
    all_metrics: list of dicts returned by compute_metrics()
    """
    df = pd.DataFrame(all_metrics)
    df = df.set_index("model")

    metrics_to_plot = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro"]
    metric_labels = ["Accuracy", "Precision\n(weighted)", "Recall\n(weighted)", "F1\n(weighted)", "F1\n(macro)"]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metrics_to_plot))
    n_models = len(df)
    width = 0.15
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    for i, (model_name, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics_to_plot]
        offset = (i - n_models / 2) * width + width / 2
        bars = ax.bar(x + offset, vals, width, label=model_name, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Comparative Analysis — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "comparative_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparative analysis plot saved: {path}")


def plot_f1_heatmap(all_metrics_per_class: dict):
    """
    Heatmap of per-class F1 across all models.
    all_metrics_per_class: {model_name: [f1_E, f1_S, f1_C, f1_I]}
    """
    df = pd.DataFrame(all_metrics_per_class, index=LABEL_NAMES).T

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        df, annot=True, fmt=".3f", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        vmin=0, vmax=1
    )
    ax.set_title("Per-Class F1 Heatmap — All Models", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Model")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "f1_heatmap_all_models.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"F1 heatmap saved: {path}")


def save_comparative_summary(all_metrics: list):
    df = pd.DataFrame(all_metrics)
    path = os.path.join(REPORTS_DIR, "comparative_summary.csv")
    df.to_csv(path, index=False)
    logger.info(f"Comparative summary saved: {path}")
    print("\n" + "="*60)
    print("COMPARATIVE SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)


def full_evaluation(y_true, y_pred, model_name: str):
    """Run all evaluation steps for a single model."""
    metrics = compute_metrics(y_true, y_pred, model_name)
    save_classification_report(y_true, y_pred, model_name)
    plot_confusion_matrix(y_true, y_pred, model_name)
    plot_per_class_f1(y_true, y_pred, model_name)
    return metrics
