"""
train.py
========
End-to-end training pipeline for Movie Box-Office ROI Prediction.

Architecture
------------
  Pipeline A  (budget known)   : GrossPredictor → ROI bins → label
  Pipeline B  (budget missing) : BudgetImputer → GrossPredictor → ROI bins → label

Model Comparison
----------------
  Three candidate regressors are trained and benchmarked on the validation set:
    1. XGBoost   — primary candidate (native NaN handling, best on tabular data)
    2. LightGBM  — secondary candidate (faster, similar accuracy)
    3. Random Forest — interpretability baseline
  The winner (best val RMSE on log-gross) is saved as the production model.

Usage
-----
    python train.py --data data/movies_cleaned.xlsx
    python train.py --data data/movies_cleaned.xlsx --trials 10 --timeout 60
    python train.py --data data/movies_cleaned.xlsx --no-compare   # skip comparison, use XGB directly

Outputs
-------
    models/roi_classifier.pkl           — best production model
    outputs/metrics.json                — all evaluation metrics
    outputs/model_comparison.json       — head-to-head comparison table
    outputs/feature_importance.csv      — gain importances from best model
    outputs/evaluation_report.txt       — human-readable report
    outputs/confusion_matrix_*.png      — heatmaps
    outputs/feature_importance.png      — bar chart
    outputs/roi_distribution_test.png   — class distribution chart
    logs/pipeline.log                   — full execution log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OUTPUT_DIR, MODEL_DIR,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT_SEC,
    ROI_LABELS,
)
from features import (
    load_raw, build_roi_bins, engineer_features,
    temporal_split, target_encode, build_studio_tier,
)
from models import (
    RoiClassifier,
    BudgetImputer,
    GrossPredictor,
    train_and_evaluate,
    compare_gross_predictors,
)
from evaluation import (
    print_metrics_summary,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roi_distribution,
    generate_report,
    plot_model_comparison,
)
from logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation 
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(
    data_path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load → ROI bins → Feature engineering → Temporal split →
    Target encoding → Studio tiers.

    Returns (train, val, test) DataFrames ready for model training.
    All encodings are computed on training fold only (no leakage).
    """
    log.info("━" * 60)
    log.info("STEP 1/3  Loading & engineering features")
    log.info("━" * 60)

    df_raw = load_raw(data_path)
    df     = build_roi_bins(df_raw)
    df     = engineer_features(df)

    train, val, test = temporal_split(df)

    log.info("STEP 2/3  Leak-free target encoding (training fold only)")
    for col in ["director", "star", "writer", "company"]:
        train, val, test = target_encode(train, val, test,
                                         col=col, target="log_gross")
        train, val, test = target_encode(train, val, test,
                                         col=col, target="score")

    log.info("STEP 3/3  Studio tier bucketing (training fold only)")
    train, val, test = build_studio_tier(train, val, test)

    log.info(
        "Data ready — train: %d | val: %d | test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison  (optional — skipped with --no-compare)
# ─────────────────────────────────────────────────────────────────────────────

def run_model_comparison(
    train: pd.DataFrame,
    val: pd.DataFrame,
    n_trials: int,
    timeout: int,
) -> Dict:
    """
    Train three candidate gross-predictors and compare on val RMSE (log scale).

    Returns a dict with per-model metrics and the name of the winner.
    """
    log.info("━" * 60)
    log.info("MODEL COMPARISON  (XGBoost vs LightGBM vs Random Forest)")
    log.info("━" * 60)

    # First impute budgets so all three models see the same feature set
    imputer = BudgetImputer(n_trials=max(n_trials // 2, 5), timeout=timeout)
    imputer.fit(train, val)
    train_imp = imputer.impute(train)
    val_imp   = imputer.impute(val)

    comparison = compare_gross_predictors(
        train_imp, val_imp, n_trials=n_trials, timeout=timeout
    )

    # Persist comparison table
    out_path = OUTPUT_DIR / "model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    log.info("Model comparison saved → %s", out_path)

    # Pretty-print to console
    SEP = "─" * 68
    print(f"\n{SEP}")
    print("  MODEL COMPARISON — GROSS PREDICTOR (val set, log-RMSE ↓ wins)")
    print(SEP)
    print(f"  {'Model':20s} {'Val RMSE (log)':>16s} {'Val R² (log)':>14s} {'Val MAPE':>10s}")
    print(f"  {'─'*60}")
    for name, m in comparison["models"].items():
        marker = "  ← WINNER" if name == comparison["winner"] else ""
        print(
            f"  {name:20s} {m['val_rmse_log']:>16.4f} "
            f"{m['val_r2_log']:>14.4f} {m['val_mape_pct']:>9.1f}%"
            f"{marker}"
        )
    print(f"{SEP}\n")

    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline  
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    data_path: str | Path,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int = OPTUNA_TIMEOUT_SEC,
    save_models: bool = True,
    run_comparison: bool = True,
) -> Tuple[RoiClassifier, Dict, Dict[str, Path]]:
    """
    Full training + evaluation pipeline.

    Parameters
    ----------
    data_path      : path to movies_cleaned.xlsx or .csv
    n_trials       : Optuna trials per sub-model
    timeout        : wall-clock limit per model (seconds)
    save_models    : persist .pkl files to models/
    run_comparison : benchmark XGBoost vs LightGBM vs RandomForest first

    Returns
    -------
    clf     : fitted RoiClassifier
    metrics : dict of all evaluation metrics
    charts  : dict mapping chart-name → output Path
    """
    log.info("=" * 60)
    log.info("  MOVIE ROI PREDICTION PIPELINE  —  TRAINING START")
    log.info("=" * 60)

    train, val, test = prepare_data(data_path)

    # ── Optional model comparison ─────────────────────────────────────────
    comparison: Dict = {}
    if run_comparison:
        comparison = run_model_comparison(train, val, n_trials=n_trials, timeout=timeout)

    # ── Train full pipeline with best model type ──────────────────────────
    best_model_type = comparison.get("winner", "XGBoost")
    log.info("━" * 60)
    log.info("FULL PIPELINE TRAINING  (model_type=%s)", best_model_type)
    log.info("━" * 60)

    clf, metrics = train_and_evaluate(
        train, val, test,
        n_trials=n_trials,
        timeout=timeout,
        save_models=save_models,
        model_type=best_model_type,
    )

    if comparison:
        metrics["model_comparison"] = comparison

    # ── Console summary ───────────────────────────────────────────────────
    print_metrics_summary(metrics)

    # ── Charts ────────────────────────────────────────────────────────────
    log.info("Generating evaluation charts …")
    charts: Dict[str, Path] = {}

    for split in ("val", "test"):
        p = plot_confusion_matrix(metrics, split=split)
        if p:
            charts[f"confusion_matrix_{split}"] = p

    fi_df = clf.gross_predictor_.feature_importance_df()
    charts["feature_importance"] = plot_feature_importance(fi_df)

    test_imp = clf.budget_imputer_.impute(test)
    roi_pred = clf.gross_predictor_.predict_roi_class(test_imp)
    mask_cls = test["roi_class"].notna()
    if mask_cls.sum() > 0:
        charts["roi_distribution_test"] = plot_roi_distribution(
            test.loc[mask_cls, "roi_class"],
            roi_pred[mask_cls],
            split="test",
        )

    if comparison:
        p = plot_model_comparison(comparison)
        if p:
            charts["model_comparison"] = p

    charts["evaluation_report"] = generate_report(metrics)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE  —  Outputs saved to %s", OUTPUT_DIR)
    log.info("=" * 60)
    return clf, metrics, charts


# ─────────────────────────────────────────────────────────────────────────────
# Single-movie inference helper  (used by Streamlit + API)
# ─────────────────────────────────────────────────────────────────────────────

def predict_single_movie(clf: RoiClassifier, movie: dict) -> dict:
    """
    Predict ROI class + estimated gross for a single unseen movie.

    Parameters
    ----------
    clf   : fitted RoiClassifier (loaded from disk or freshly trained)
    movie : dict with movie attributes (budget may be None/missing)

    Returns
    -------
    dict with keys:
        predicted_roi_class   : str  — "flop" | "below_avg" | "hit" | "blockbuster"
        predicted_gross_usd   : float
        roi_probability       : dict — per-class confidence (if available)
        budget_imputed        : bool
        pipeline_used         : str
        imputed_budget_usd    : float | None
    """
    from features import engineer_features, build_roi_bins
    from config import STUDIO_TIER_MAJOR, STUDIO_TIER_MID

    row = pd.DataFrame([movie])
    row = build_roi_bins(row)
    row = engineer_features(row)

    # Apply stored target-encoding maps (inference safe — no leakage)
    for col in ["director", "star", "writer", "company"]:
        te_col_lg = f"{col}_te_log_g"
        te_col_sc = f"{col}_te_score"
        global_lg = clf._te_global_log_g.get(col, 0.0)
        global_sc = clf._te_global_score.get(col, 0.0)
        row[te_col_lg] = row[col].map(clf._te_maps_log_g.get(col, {})).fillna(global_lg)
        row[te_col_sc] = row[col].map(clf._te_maps_score.get(col, {})).fillna(global_sc)

    def _tier(company):
        if pd.isna(company):
            return 0
        m = clf._studio_median.get(company, 0)
        if m >= STUDIO_TIER_MAJOR: return 2
        if m >= STUDIO_TIER_MID:   return 1
        return 0

    row["studio_tier_enc"] = row["company"].apply(_tier)

    budget_was_missing = pd.isna(row["budget"].iloc[0])
    row_imp = clf.budget_imputer_.impute(row)

    imputed_budget: float | None = None
    if budget_was_missing:
        imputed_budget = float(np.expm1(
            clf.budget_imputer_.model_.predict(
                row_imp[[c for c in clf.budget_imputer_.feature_cols_
                          if c in row_imp.columns]]
            )[0]
        ))

    log_gross_pred = clf.gross_predictor_.predict_log_gross(row_imp)
    gross_pred     = float(np.expm1(log_gross_pred)[0])
    roi_class_pred = clf.gross_predictor_.predict_roi_class(row_imp)

    # Soft probabilities if available
    proba: dict = {}
    try:
        proba_arr = clf.gross_predictor_.predict_proba(row_imp)
        if proba_arr is not None:
            proba = {label: round(float(p), 4)
                     for label, p in zip(ROI_LABELS, proba_arr[0])}
    except Exception:
        pass

    return {
        "predicted_roi_class" : str(roi_class_pred[0]),
        "predicted_gross_usd" : round(gross_pred, 2),
        "roi_probability"     : proba,
        "budget_imputed"      : bool(budget_was_missing),
        "imputed_budget_usd"  : round(imputed_budget, 2) if imputed_budget else None,
        "pipeline_used"       : "B (budget imputed)" if budget_was_missing else "A (budget known)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Movie ROI Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data",       required=True,
                   help="Path to movies_cleaned.xlsx or .csv")
    p.add_argument("--trials",     type=int, default=OPTUNA_N_TRIALS,
                   help=f"Optuna trials per model (default: {OPTUNA_N_TRIALS})")
    p.add_argument("--timeout",    type=int, default=OPTUNA_TIMEOUT_SEC,
                   help=f"Tuning timeout per model in seconds (default: {OPTUNA_TIMEOUT_SEC})")
    p.add_argument("--no-save",    action="store_true",
                   help="Skip persisting trained models to disk")
    p.add_argument("--no-compare", action="store_true",
                   help="Skip model comparison and go straight to XGBoost")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    clf, metrics, charts = run_pipeline(
        data_path      = args.data,
        n_trials       = args.trials,
        timeout        = args.timeout,
        save_models    = not args.no_save,
        run_comparison = not args.no_compare,
    )
    print(f"\nDone. Outputs saved to: {OUTPUT_DIR}")
    print(f"Production model : {MODEL_DIR / 'roi_classifier.pkl'}")
