"""
evaluation.py
=============
Evaluation utilities: metrics display, chart generation, and report writing.

Functions
---------
    print_metrics_summary(metrics)                  — tabular console output
    plot_confusion_matrix(metrics, split, out_path) — heatmap PNG
    plot_feature_importance(fi_df, top_n, out_path) — horizontal bar chart PNG
    plot_roi_distribution(y_true, y_pred, split)    — grouped bar chart PNG
    plot_model_comparison(comparison, out_path)     — bar chart comparing models
    generate_report(metrics, out_path)              — plain-text report .txt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import OUTPUT_DIR, ROI_LABELS
from logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_metrics_summary(metrics: Dict) -> None:
    SEP = "─" * 68
    print(f"\n{SEP}")
    print("  MOVIE ROI CLASSIFICATION — EVALUATION SUMMARY")
    print(SEP)

    for split in ("val", "test"):
        acc      = metrics.get(f"{split}_roi_accuracy")
        r2       = metrics.get(f"{split}_gross_r2_log")
        rmse_log = metrics.get(f"{split}_gross_rmse_log")
        mape     = metrics.get(f"{split}_gross_mape_pct")

        print(f"\n  [{split.upper()} SET]")
        if acc      is not None: print(f"  {'ROI Classification Accuracy':40s}: {acc:.4f}")
        if r2       is not None: print(f"  {'Gross R² (log scale)':40s}: {r2:.4f}")
        if rmse_log is not None: print(f"  {'Gross RMSE (log scale)':40s}: {rmse_log:.4f}")
        if mape     is not None: print(f"  {'Gross Median APE (raw $)':40s}: {mape:.1f}%")

        report = metrics.get(f"{split}_roi_classification_report", {})
        if report:
            print(f"\n  {'Class':12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
            print(f"  {'─' * 52}")
            for cls in ROI_LABELS:
                row = report.get(cls, {})
                print(
                    f"  {cls:12s} {row.get('precision', 0):10.3f} "
                    f"{row.get('recall', 0):10.3f} {row.get('f1-score', 0):10.3f} "
                    f"{int(row.get('support', 0)):10d}"
                )

    bi_r2 = metrics.get("test_budget_imputer_r2_log")
    if bi_r2:
        print(f"\n  [BUDGET IMPUTER — test set]")
        print(f"  {'R² on log(budget)':40s}: {bi_r2:.4f}")
        print(f"  {'MAPE on raw budget':40s}: {metrics.get('test_budget_imputer_mape_pct', 0):.1f}%")

    # Model comparison summary
    comp = metrics.get("model_comparison")
    if comp:
        print(f"\n  [MODEL COMPARISON]  Winner → {comp.get('winner', 'N/A')}")
        for name in comp.get("ranking", []):
            m = comp["models"][name]
            marker = "  ← selected" if name == comp["winner"] else ""
            print(f"    {name:20s} RMSE={m['val_rmse_log']:.4f}  "
                  f"R²={m['val_r2_log']:.4f}  MAPE={m['val_mape_pct']:.1f}%{marker}")

    print(f"\n{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    metrics: Dict,
    split: str = "test",
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    cm     = metrics.get(f"{split}_roi_confusion_matrix")
    labels = metrics.get(f"{split}_roi_confusion_matrix_labels", ROI_LABELS)
    if cm is None:
        log.warning("No confusion matrix for split=%s", split)
        return None

    cm_arr = np.array(cm)
    cm_pct = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_pct, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="grey", ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(f"ROI Class Confusion Matrix — {split.upper()} set", fontsize=13, pad=12)
    plt.tight_layout()

    out_path = out_path or OUTPUT_DIR / f"confusion_matrix_{split}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    fi_df: pd.DataFrame,
    top_n: int = 20,
    out_path: Optional[Path] = None,
) -> Path:
    fi_top = fi_df.head(top_n).copy().sort_values("gain")

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    bars = ax.barh(fi_top["feature"], fi_top["gain_pct"], color="#2563EB", edgecolor="white")
    ax.set_xlabel("Gain contribution (%)", fontsize=11)
    ax.set_title(f"Top {top_n} Features — XGBoost Gain Importance", fontsize=13)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()

    out_path = out_path or OUTPUT_DIR / "feature_importance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Feature importance saved → %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# ROI distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_roi_distribution(
    y_true: pd.Series,
    y_pred: np.ndarray,
    split: str = "test",
    out_path: Optional[Path] = None,
) -> Path:
    df_true = pd.DataFrame({"roi_class": pd.Categorical(y_true, categories=ROI_LABELS), "source": "Actual"})
    df_pred = pd.DataFrame({"roi_class": pd.Categorical(y_pred, categories=ROI_LABELS), "source": "Predicted"})
    combined = pd.concat([df_true, df_pred])

    counts   = combined.groupby(["source", "roi_class"], observed=True).size().reset_index(name="count")
    pivot    = counts.pivot(index="roi_class", columns="source", values="count").fillna(0)
    pivot_pct = pivot.div(pivot.sum()) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ROI_LABELS))
    w = 0.35
    ax.bar(x - w/2, pivot_pct.get("Actual",    [0]*4), w, label="Actual",    color="#1D4ED8")
    ax.bar(x + w/2, pivot_pct.get("Predicted", [0]*4), w, label="Predicted", color="#10B981")
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_LABELS, fontsize=11)
    ax.set_ylabel("% of films", fontsize=11)
    ax.set_title(f"ROI Class Distribution — {split.upper()} set", fontsize=13)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()

    out_path = out_path or OUTPUT_DIR / f"roi_distribution_{split}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("ROI distribution saved → %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison chart 
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    comparison: Dict,
    out_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Grouped bar chart comparing Val RMSE, R², and MAPE across candidate models.
    The winner model bar is highlighted.
    """
    models_data = comparison.get("models", {})
    winner      = comparison.get("winner", "")
    if not models_data:
        return None

    names   = list(models_data.keys())
    rmse    = [models_data[n]["val_rmse_log"]  for n in names]
    r2      = [models_data[n]["val_r2_log"]    for n in names]
    mape    = [models_data[n]["val_mape_pct"]  for n in names]

    colors  = ["#2563EB" if n != winner else "#10B981" for n in names]
    x       = np.arange(len(names))
    w       = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Model Comparison — Gross Predictor (Val Set)", fontsize=14, fontweight="bold")

    for ax, values, title, fmt in zip(
        axes,
        [rmse, r2, mape],
        ["RMSE (log scale) ↓", "R² (log scale) ↑", "Median APE % ↓"],
        [".4f", ".4f", ".1f"],
    ):
        bars = ax.bar(x, values, color=colors, edgecolor="white", width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9, rotation=10)
        ax.set_title(title, fontsize=11)
        ax.bar_label(bars, fmt=f"%{fmt}", padding=3, fontsize=9)

    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#10B981", label="Winner"),
            plt.Rectangle((0, 0), 1, 1, color="#2563EB", label="Other"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = out_path or OUTPUT_DIR / "model_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Model comparison chart saved → %s", out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Text evaluation report
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(metrics: Dict, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or OUTPUT_DIR / "evaluation_report.txt"
    lines = [
        "=" * 70,
        "  MOVIE BOX-OFFICE PREDICTION — EVALUATION REPORT",
        "=" * 70,
        "",
        "Pipeline Architecture",
        "─────────────────────",
        "  Pipeline A (budget known)   : GrossPredictor → ROI bins → label",
        "  Pipeline B (budget missing) : BudgetImputer → GrossPredictor → ROI bins → label",
        "",
        "ROI Bins (Cocuzzo & Wu 2013)",
        "────────────────────────────",
        "  [0, 0.5)   → flop",
        "  [0.5, 1)   → below_avg",
        "  [1, 2)     → hit",
        "  [2, ∞)     → blockbuster",
        "",
    ]

    # Model comparison block
    comp = metrics.get("model_comparison")
    if comp:
        lines += [
            "─" * 70,
            "  MODEL COMPARISON (Val Set — Gross Predictor)",
            "─" * 70,
            f"  {'Model':20s} {'RMSE (log)':>12s} {'R² (log)':>10s} {'MAPE %':>8s}  {'Selected':>10s}",
            f"  {'─' * 64}",
        ]
        for name in comp.get("ranking", []):
            m = comp["models"][name]
            sel = "YES ←" if name == comp["winner"] else ""
            lines.append(
                f"  {name:20s} {m['val_rmse_log']:>12.4f} {m['val_r2_log']:>10.4f} "
                f"{m['val_mape_pct']:>7.1f}%  {sel:>10s}"
            )
        lines.append("")

    for split in ("val", "test"):
        acc      = metrics.get(f"{split}_roi_accuracy")
        r2       = metrics.get(f"{split}_gross_r2_log")
        rmse_log = metrics.get(f"{split}_gross_rmse_log")
        mape     = metrics.get(f"{split}_gross_mape_pct")

        lines += [f"{'─' * 70}", f"  {split.upper()} SET RESULTS", f"{'─' * 70}"]
        if acc      is not None: lines.append(f"  ROI Classification Accuracy : {acc:.4f}")
        if r2       is not None: lines.append(f"  Gross R² (log scale)        : {r2:.4f}")
        if rmse_log is not None: lines.append(f"  Gross RMSE (log scale)      : {rmse_log:.4f}")
        if mape     is not None: lines.append(f"  Gross Median APE (raw $)    : {mape:.1f}%")

        report = metrics.get(f"{split}_roi_classification_report", {})
        if report:
            lines += [
                "",
                f"  {'Class':12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}",
                f"  {'─' * 52}",
            ]
            for cls in ROI_LABELS:
                row = report.get(cls, {})
                lines.append(
                    f"  {cls:12s} {row.get('precision', 0):10.3f} "
                    f"{row.get('recall', 0):10.3f} {row.get('f1-score', 0):10.3f} "
                    f"{int(row.get('support', 0)):10d}"
                )
        lines.append("")

    bi_r2   = metrics.get("test_budget_imputer_r2_log")
    bi_mape = metrics.get("test_budget_imputer_mape_pct")
    if bi_r2 is not None:
        lines += [
            "─" * 70,
            "  BUDGET IMPUTER (test set — rows with known budget)",
            "─" * 70,
            f"  R² on log(budget)  : {bi_r2:.4f}",
            f"  MAPE on raw budget : {bi_mape:.1f}%",
            "",
        ]

    lines.append("=" * 70)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Evaluation report saved → %s", out_path)
    return out_path
