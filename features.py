"""
features.py
===========
Data loading, cleaning, and feature engineering for the movie pipeline.

Public API
----------
    load_raw(path)              → pd.DataFrame  (raw, no changes)
    engineer_features(df)      → pd.DataFrame  (feature-rich, ready for models)
    temporal_split(df)         → (train, val, test)
    build_roi_bins(df)         → pd.DataFrame  (with roi_class column added/re-created)
    target_encode(train, val, test, col, target, agg)  → (train, val, test)

Notes
-----
- All target-encoding is computed on *training fold only* to prevent leakage.
- Log transforms are applied as new columns (originals kept for interpretability).
- The `budget_available` binary flag is created before any imputation so that the
  model can learn from the MNAR pattern (Missing Not At Random).
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from config import (
    IDENTIFIER_COLS, HIGH_CARD_CATS, LOW_CARD_CATS, NUMERIC_FEATS,
    ROI_BINS, ROI_LABELS,
    TRAIN_END_YEAR, VAL_END_YEAR,
    STUDIO_TIER_MAJOR, STUDIO_TIER_MID,
    TARGET_GROSS, TARGET_BUDGET,
)
from logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Raw loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the movies spreadsheet and do minimal dtype coercion."""
    path = Path(path)
    log.info("Loading dataset from %s", path)

    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Drop unnamed index column if present
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df.drop(columns=drop_cols, inplace=True)

    # Force numeric columns
    for col in ["budget", "gross", "votes", "runtime", "score", "year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info("Loaded %d rows × %d columns", *df.shape)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ROI binning  (Cocuzzo & Wu 2013 / user-specified bins)
# ─────────────────────────────────────────────────────────────────────────────

def build_roi_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute roi = gross / budget and assign roi_class label.
    Rows where budget or gross is missing / zero get NaN roi_class.
    """
    df = df.copy()
    mask = df["budget"].notna() & df["gross"].notna() & (df["budget"] > 0)
    df["roi"] = np.where(mask, df["gross"] / df["budget"], np.nan)
    df["roi_class"] = pd.cut(
        df["roi"],
        bins=ROI_BINS,
        labels=ROI_LABELS,
        right=False,
    ).astype(object)  # keep as str/object for XGBoost label encoder
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _extract_season(month: pd.Series) -> pd.Series:
    """Map release month to season bucket aligned with box-office patterns."""
    def _map(m):
        if pd.isna(m):
            return "unknown"
        m = int(m)
        if m in (5, 6, 7, 8):
            return "summer"          # blockbuster season
        if m in (11, 12):
            return "holiday"         # award + holiday push
        if m in (3, 4):
            return "spring"
        return "other"
    return month.map(_map)


def _decade(year: pd.Series) -> pd.Series:
    return ((year // 10) * 10).astype("Int64").astype(str)


def _sequel_flag(name: pd.Series) -> pd.Series:
    """Heuristic: detect sequels / franchises from title keywords."""
    pattern = (
        r"\b(?:II|III|IV|V|VI|VII|VIII|IX|X|2|3|4|5|6|7|8|9|"
        r"part|chapter|episode|returns|rises|reloaded|revolution|"
        r"origins|legacy|begins|forever|beyond|strikes back|"
        r"reborn|rebirth|resurrection|continued|continues)\b"
    )
    return name.str.contains(pattern, na=False, regex=True, case=False).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature-engineering transformations described in the
    ML Model Selection Report (§3.4, §5.4).

    Returns a *new* DataFrame — original is unchanged.
    """
    df = df.copy()

    # ── Coerce to numeric (Python None from inference dicts → NaN) ──────────
    for _col in ["votes", "budget", "gross", "score", "runtime"]:
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")

    # ── Missing-data flag (MNAR indicator — must be created first) ──────────
    df["budget_available"] = df["budget"].notna().astype(int)

    # ── Log transforms ───────────────────────────────────────────────────────
    df["log_votes"] = np.log1p(df["votes"].clip(lower=0))
    df["log_budget"] = np.log1p(df["budget"].clip(lower=0))  # NaN stays NaN
    df["log_gross"] = np.log1p(df["gross"].clip(lower=0))  # used as target

    # ── Temporal features ────────────────────────────────────────────────────
    df["release_season"] = _extract_season(df.get("release_month"))
    df["decade"]         = _decade(df["year"])

    # ── Sequel heuristic ─────────────────────────────────────────────────────
    df["sequel_flag"] = _sequel_flag(df["name"])

    # ── High-cardinality label encoding (integer codes for XGBoost) ─────────
    for col in HIGH_CARD_CATS:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].astype("category").cat.codes

    # ── Low-cardinality label encoding ───────────────────────────────────────
    for col in LOW_CARD_CATS:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].astype("category").cat.codes

    # ── Season & decade encoding ─────────────────────────────────────────────
    df["season_enc"] = df["release_season"].astype("category").cat.codes
    df["decade_enc"] = df["decade"].astype("category").cat.codes

    log.info("Feature engineering complete.  Columns now: %d", df.shape[1])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Target encoding  (computed inside CV folds to prevent leakage)
# ─────────────────────────────────────────────────────────────────────────────

def target_encode(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    target: str,
    agg: str = "mean",
    smoothing: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bayesian / smoothed target encoding for a categorical column.

    smoothing controls the strength of the global prior relative to the
    per-group estimate.  Higher = more regularisation toward the global mean.
    """
    global_mean = train[target].mean()
    stats = (
        train.groupby(col)[target]
        .agg(["mean", "count"])
        .rename(columns={"mean": "group_mean", "count": "n"})
    )
    stats["smoothed"] = (
        (stats["n"] * stats["group_mean"] + smoothing * global_mean)
        / (stats["n"] + smoothing)
    )

    new_col = f"{col}_te_{target[:5]}"
    for split in (train, val, test):
        split[new_col] = split[col].map(stats["smoothed"]).fillna(global_mean)

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Studio tier  (from training fold only)
# ─────────────────────────────────────────────────────────────────────────────

def build_studio_tier(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Bucket 'company' into major / mid / indie based on training-set median gross.
    """
    median_by_studio = train.groupby("company")["gross"].median()

    def _tier(company):
        if pd.isna(company):
            return "indie"
        m = median_by_studio.get(company, 0)
        if m >= STUDIO_TIER_MAJOR:
            return "major"
        if m >= STUDIO_TIER_MID:
            return "mid"
        return "indie"

    tier_enc = {"major": 2, "mid": 1, "indie": 0}
    for split in (train, val, test):
        split["studio_tier"] = split["company"].apply(_tier)
        split["studio_tier_enc"] = split["studio_tier"].map(tier_enc)

    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Temporal split
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by year to simulate real-world deployment (no future leakage).
    Train: ≤ TRAIN_END_YEAR | Val: TRAIN_END_YEAR+1 – VAL_END_YEAR | Test: rest
    """
    train = df[df["year"] <= TRAIN_END_YEAR].copy()
    val   = df[(df["year"] > TRAIN_END_YEAR) & (df["year"] <= VAL_END_YEAR)].copy()
    test  = df[df["year"] > VAL_END_YEAR].copy()

    log.info(
        "Temporal split — train: %d | val: %d | test: %d",
        len(train), len(val), len(test),
    )
    return train, val, test
