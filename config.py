"""
config.py
=========
Central configuration for the Movie Revenue Prediction Pipeline.
All constants, paths, and hyperparameter search spaces live here.
Import this module in every other module — never hardcode values elsewhere.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR    = BASE_DIR / "logs"

for _d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Dataset schema
# ─────────────────────────────────────────────
TARGET_GROSS      = "gross"
TARGET_BUDGET     = "budget"        # used by the budget imputation model
TARGET_ROI_CLASS  = "roi_class"     # final classification label

IDENTIFIER_COLS = ["Unnamed: 0", "name", "released",
                   "roi", "roi_class"]  # never used as features

HIGH_CARD_CATS  = ["director", "writer", "star", "company"]
LOW_CARD_CATS   = ["rating", "genre", "country"]
NUMERIC_FEATS   = ["year", "votes", "runtime", "release_month", "release_day", "budget"]

# ─────────────────────────────────────────────
# ROI bins — from the Cocuzzo & Wu (2013) paper
# ─────────────────────────────────────────────
ROI_BINS   = [0, 0.5, 1, 2, float("inf")]
ROI_LABELS = ["flop", "below_avg", "hit", "blockbuster"]   # 4 ordinal classes

# ─────────────────────────────────────────────
# Temporal split boundaries  (per model-selection report §3.5 / §5.5)
# ─────────────────────────────────────────────
TRAIN_END_YEAR = 2015
VAL_END_YEAR   = 2018
# test = 2019–2020

# ─────────────────────────────────────────────
# XGBoost Optuna search space
# ─────────────────────────────────────────────
OPTUNA_N_TRIALS     = 40          # increase to 100+ for production tuning
OPTUNA_TIMEOUT_SEC  = 300         # 5-minute hard cap per model

XGB_FIXED_PARAMS = dict(
    tree_method   = "hist",
    eval_metric   = "rmse",
    verbosity     = 0,
    random_state  = 42,
    n_jobs        = -1,
)

XGB_SEARCH_SPACE = dict(
    n_estimators       = (200, 1000),
    max_depth          = (3, 8),
    learning_rate      = (1e-3, 0.3),     # log-uniform
    subsample          = (0.5, 1.0),
    colsample_bytree   = (0.5, 1.0),
    reg_alpha          = (1e-8, 10.0),    # log-uniform
    reg_lambda         = (1e-8, 10.0),    # log-uniform
    min_child_weight   = (1, 10),
)

# ─────────────────────────────────────────────
# Studio-tier thresholds  (gross in $M)
# ─────────────────────────────────────────────
STUDIO_TIER_MAJOR  = 100_000_000   # ≥ $100M historical median → "major"
STUDIO_TIER_MID    = 20_000_000    # ≥  $20M                   → "mid"
# below → "indie"

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = LOG_DIR / "pipeline.log"
