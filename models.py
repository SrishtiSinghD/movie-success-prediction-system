"""
models.py
=========
Two-stage prediction pipeline + multi-model comparison framework.

Classes
-------
    BudgetImputer      — XGBoost regressor for missing budget imputation
    GrossPredictor     — Pluggable regressor (XGBoost / LightGBM / RandomForest)
    RoiClassifier      — Unified facade (Pipeline A + B)

Functions
---------
    compare_gross_predictors   — benchmark three regressors, return comparison dict
    train_and_evaluate         — full experiment runner, returns (clf, metrics)
    _regression_metrics        — shared regression metric computation
    _classification_metrics    — shared classification metric computation
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import optuna
import shap
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

from config import (
    MODEL_DIR, OUTPUT_DIR,
    ROI_BINS, ROI_LABELS,
    OPTUNA_N_TRIALS, OPTUNA_TIMEOUT_SEC,
    XGB_FIXED_PARAMS, XGB_SEARCH_SPACE,
    TARGET_GROSS, TARGET_BUDGET,
)
from logger import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

log = get_logger(__name__)

ModelType = Literal["XGBoost", "LightGBM", "RandomForest"]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _median_ape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _regression_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    label: str = "",
) -> Dict[str, float]:
    rmse_log = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae_log  = float(mean_absolute_error(y_true_log, y_pred_log))
    r2_log   = float(r2_score(y_true_log, y_pred_log))
    mape     = _median_ape(np.expm1(y_true_log), np.expm1(y_pred_log))
    return {
        f"{label}rmse_log"  : round(rmse_log, 4),
        f"{label}mae_log"   : round(mae_log,  4),
        f"{label}r2_log"    : round(r2_log,   4),
        f"{label}mape_pct"  : round(mape,     2),
    }


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> Dict:
    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=ROI_LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=ROI_LABELS).tolist()
    return {
        f"{label}accuracy"                : round(acc, 4),
        f"{label}classification_report"   : report,
        f"{label}confusion_matrix"        : cm,
        f"{label}confusion_matrix_labels" : ROI_LABELS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature column lists
# ─────────────────────────────────────────────────────────────────────────────

_BASE_FEATS: List[str] = [
    "year", "log_votes", "runtime", "release_month", "release_day",
    "decade_enc", "season_enc", "sequel_flag",
    "budget_available",
    "genre_enc", "rating_enc", "country_enc",
    "director_enc", "writer_enc", "star_enc", "company_enc",
    "studio_tier_enc",
    "director_te_log_g", "director_te_score",
    "star_te_log_g",     "star_te_score",
    "writer_te_log_g",   "writer_te_score",
    "company_te_log_g",  "company_te_score",
]

_BUDGET_FEATS: List[str] = ["log_budget", "budget"]


def _feature_cols(include_budget: bool = True) -> List[str]:
    cols = list(_BASE_FEATS)
    if include_budget:
        cols += _BUDGET_FEATS
    return cols


def _safe_cols(df: pd.DataFrame, wanted: List[str]) -> List[str]:
    return [c for c in wanted if c in df.columns]


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective factory  (XGBoost only — other models use defaults)
# ─────────────────────────────────────────────────────────────────────────────

def _make_xgb_objective(
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
) -> callable:
    space = XGB_SEARCH_SPACE

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators",     *space["n_estimators"]),
            max_depth        = trial.suggest_int("max_depth",        *space["max_depth"]),
            learning_rate    = trial.suggest_float("learning_rate",  *space["learning_rate"], log=True),
            subsample        = trial.suggest_float("subsample",      *space["subsample"]),
            colsample_bytree = trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
            reg_alpha        = trial.suggest_float("reg_alpha",      *space["reg_alpha"], log=True),
            reg_lambda       = trial.suggest_float("reg_lambda",     *space["reg_lambda"], log=True),
            min_child_weight = trial.suggest_int("min_child_weight", *space["min_child_weight"]),
            early_stopping_rounds = 20,
        )
        params.update(XGB_FIXED_PARAMS)
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return float(np.sqrt(mean_squared_error(y_val, model.predict(X_val))))

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Budget Imputer
# ─────────────────────────────────────────────────────────────────────────────

class BudgetImputer:
    """
    XGBoost regressor that predicts log1p(budget) for films where budget is
    missing.  Trained only on rows with known budget.

    Attributes
    ----------
    model_         : fitted XGBRegressor
    feature_cols_  : column list used at fit time
    best_params_   : Optuna-selected hyperparameters
    val_rmse_log   : validation RMSE on log scale (set after fit)
    """

    def __init__(self, n_trials: int = OPTUNA_N_TRIALS, timeout: int = OPTUNA_TIMEOUT_SEC):
        self.n_trials     = n_trials
        self.timeout      = timeout
        self.model_: Optional[XGBRegressor] = None
        self.feature_cols_: List[str] = []
        self.best_params_: Dict       = {}
        self.val_rmse_log: float      = float("inf")

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "BudgetImputer":
        # Only rows with known budget
        tr = train[train["budget"].notna()].copy()
        vl = val[val["budget"].notna()].copy()

        feat_cols = _safe_cols(tr, _feature_cols(include_budget=False))
        self.feature_cols_ = feat_cols

        y_tr  = np.log1p(tr["budget"].values)
        y_val = np.log1p(vl["budget"].values)
        X_tr  = tr[feat_cols].fillna(0)
        X_val = vl[feat_cols].fillna(0)

        log.info("BudgetImputer — Optuna tuning (%d trials, %ds timeout)", self.n_trials, self.timeout)
        study = optuna.create_study(direction="minimize")
        study.optimize(
            _make_xgb_objective(X_tr, y_tr, X_val, y_val),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        self.best_params_ = study.best_params
        params = {**XGB_FIXED_PARAMS, **self.best_params_, "early_stopping_rounds": 20}
        self.model_ = XGBRegressor(**params)
        self.model_.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        pred = self.model_.predict(X_val)
        self.val_rmse_log = float(np.sqrt(mean_squared_error(y_val, pred)))
        log.info("BudgetImputer — val RMSE (log) = %.4f", self.val_rmse_log)
        return self

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing budget/log_budget using the trained model. Returns a copy."""
        df = df.copy()
        missing = df["budget"].isna()
        if missing.sum() > 0:
            X_miss = df.loc[missing, _safe_cols(df, self.feature_cols_)].fillna(0)
            log_bud_pred = self.model_.predict(X_miss)
            df.loc[missing, "log_budget"] = log_bud_pred
            df.loc[missing, "budget"]     = np.expm1(log_bud_pred)
        else:
            df["log_budget"] = np.log1p(df["budget"].clip(lower=0))
        return df

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "budget_imputer.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("BudgetImputer saved → %s", path)
        return path

    @staticmethod
    def load(path: Path) -> "BudgetImputer":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Gross Predictor  (pluggable model type)
# ─────────────────────────────────────────────────────────────────────────────

class GrossPredictor:
    """
    Regressor that predicts log1p(gross), then converts to ROI class via bins.

    Supports three backends: "XGBoost" (default), "LightGBM", "RandomForest".
    XGBoost uses Optuna tuning; LightGBM and RandomForest use sensible defaults
    (full Optuna integration can be added without changing the interface).

    Attributes
    ----------
    model_        : fitted regressor
    model_type    : "XGBoost" | "LightGBM" | "RandomForest"
    feature_cols_ : column list used at fit time
    best_params_  : hyperparameters (Optuna for XGB, defaults otherwise)
    val_rmse_log  : validation RMSE on log scale
    val_r2_log    : validation R² on log scale
    val_mape_pct  : validation Median APE on raw $
    """

    def __init__(
        self,
        model_type: ModelType = "XGBoost",
        include_budget: bool  = True,
        n_trials: int         = OPTUNA_N_TRIALS,
        timeout: int          = OPTUNA_TIMEOUT_SEC,
        name: str             = "GrossPredictor",
    ):
        self.model_type     = model_type
        self.include_budget = include_budget
        self.n_trials       = n_trials
        self.timeout        = timeout
        self.name           = name

        self.model_: object             = None
        self.feature_cols_: List[str]   = []
        self.best_params_: Dict         = {}
        self.val_rmse_log: float        = float("inf")
        self.val_r2_log: float          = float("-inf")
        self.val_mape_pct: float        = float("inf")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_model(self, params: Dict):
        if self.model_type == "XGBoost":
            return XGBRegressor(**{**XGB_FIXED_PARAMS, **params})
        if self.model_type == "LightGBM":
            if not _LGBM_AVAILABLE:
                raise ImportError("lightgbm not installed. Run: pip install lightgbm")
            return LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
        if self.model_type == "RandomForest":
            return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def _default_params(self) -> Dict:
        if self.model_type == "XGBoost":
            return {}  # tuned by Optuna
        if self.model_type == "LightGBM":
            return dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0)
        if self.model_type == "RandomForest":
            return dict(n_estimators=300, max_depth=12, min_samples_leaf=5,
                        max_features="sqrt")
        return {}

    # ── Fit ───────────────────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "GrossPredictor":
        feat_cols = _safe_cols(train, _feature_cols(self.include_budget))
        self.feature_cols_ = feat_cols

        tr_mask = train["gross"].notna()
        vl_mask = val["gross"].notna()
        X_tr  = train.loc[tr_mask, feat_cols].fillna(0)
        y_tr  = train.loc[tr_mask, "log_gross"].values
        X_val = val.loc[vl_mask, feat_cols].fillna(0)
        y_val = val.loc[vl_mask, "log_gross"].values

        log.info("%s — model_type=%s  features=%d",
                 self.name, self.model_type, len(feat_cols))

        if self.model_type == "XGBoost":
            log.info("%s — Optuna tuning (%d trials, %ds)", self.name, self.n_trials, self.timeout)
            study = optuna.create_study(direction="minimize")
            study.optimize(
                _make_xgb_objective(X_tr, y_tr, X_val, y_val),
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=False,
            )
            self.best_params_ = study.best_params
            model = XGBRegressor(**{**XGB_FIXED_PARAMS, **self.best_params_,
                                    "early_stopping_rounds": 20})
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.best_params_ = self._default_params()
            model = self._build_model(self.best_params_)
            if self.model_type == "LightGBM":
                model.fit(X_tr, y_tr,
                          eval_set=[(X_val, y_val)],
                          callbacks=[])
            else:
                model.fit(X_tr, y_tr)

        self.model_ = model

        # Val metrics
        pred = model.predict(X_val)
        m = _regression_metrics(y_val, pred)
        self.val_rmse_log = m["rmse_log"]
        self.val_r2_log   = m["r2_log"]
        self.val_mape_pct = m["mape_pct"]
        log.info("%s — val RMSE=%.4f  R²=%.4f  MAPE=%.1f%%",
                 self.name, self.val_rmse_log, self.val_r2_log, self.val_mape_pct)
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_log_gross(self, df: pd.DataFrame) -> np.ndarray:
        X = df[_safe_cols(df, self.feature_cols_)].fillna(0)
        return self.model_.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Return per-class soft probabilities if available, else None."""
        if not hasattr(self.model_, "predict_proba"):
            return None
        X = df[_safe_cols(df, self.feature_cols_)].fillna(0)
        return self.model_.predict_proba(X)

    def predict_roi_class(self, df: pd.DataFrame) -> np.ndarray:
        log_gross = self.predict_log_gross(df)
        gross     = np.expm1(log_gross)

        # We need an imputed budget in df to compute ROI
        budget = df["budget"].values.copy()
        budget = np.where(np.isnan(budget) | (budget <= 0), np.nan, budget)

        roi = np.where(
            ~np.isnan(budget),
            gross / budget,
            np.nan,
        )

        labels = np.full(len(roi), "blockbuster", dtype=object)
        labels[roi < 2]   = "hit"
        labels[roi < 1]   = "below_avg"
        labels[roi < 0.5] = "flop"
        labels[np.isnan(roi)] = "blockbuster"   # fallback when budget still unknown
        return labels

    # ── Feature importance ────────────────────────────────────────────────

    def feature_importance_df(self) -> pd.DataFrame:
        if self.model_type == "XGBoost":
            scores = self.model_.get_booster().get_score(importance_type="gain")
        elif self.model_type == "LightGBM":
            scores = dict(zip(self.feature_cols_, self.model_.feature_importances_))
        elif self.model_type == "RandomForest":
            scores = dict(zip(self.feature_cols_, self.model_.feature_importances_))
        else:
            return pd.DataFrame(columns=["feature", "gain", "gain_pct"])

        total = sum(scores.values()) or 1
        rows  = [{"feature": k, "gain": v, "gain_pct": v / total * 100}
                 for k, v in scores.items()]
        return pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / f"{self.name.lower().replace(' ', '_')}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("%s saved → %s", self.name, path)
        return path

    @staticmethod
    def load(path: Path) -> "GrossPredictor":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Model Comparison  — benchmark three regressors
# ─────────────────────────────────────────────────────────────────────────────

def compare_gross_predictors(
    train_imp: pd.DataFrame,
    val_imp: pd.DataFrame,
    n_trials: int = OPTUNA_N_TRIALS,
    timeout: int  = OPTUNA_TIMEOUT_SEC,
) -> Dict:
    """
    Train XGBoost, LightGBM, and RandomForest on the same imputed data
    and compare val-set RMSE.

    Returns
    -------
    dict with keys:
        "models"  : {model_name: {val_rmse_log, val_r2_log, val_mape_pct, params}}
        "winner"  : model_name with lowest val RMSE
        "ranking" : list of model names ordered by val RMSE (ascending)
    """
    candidates: List[ModelType] = ["XGBoost"]
    if _LGBM_AVAILABLE:
        candidates.append("LightGBM")
    candidates.append("RandomForest")

    results: Dict[str, Dict] = {}

    for mtype in candidates:
        log.info("  ── Comparing: %s", mtype)
        gp = GrossPredictor(
            model_type=mtype,
            include_budget=True,
            n_trials=n_trials,
            timeout=timeout,
            name=f"GrossPredictor_{mtype}",
        )
        gp.fit(train_imp, val_imp)
        results[mtype] = {
            "val_rmse_log" : gp.val_rmse_log,
            "val_r2_log"   : gp.val_r2_log,
            "val_mape_pct" : gp.val_mape_pct,
            "best_params"  : gp.best_params_,
        }

    ranking = sorted(results, key=lambda k: results[k]["val_rmse_log"])
    return {
        "models"  : results,
        "winner"  : ranking[0],
        "ranking" : ranking,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RoiClassifier  — unified facade
# ─────────────────────────────────────────────────────────────────────────────

class RoiClassifier:
    """
    Wraps BudgetImputer + GrossPredictor into one inference interface.

    Pipeline A (budget known)   : GrossPredictor → ROI bins → label
    Pipeline B (budget missing) : BudgetImputer → GrossPredictor → ROI bins → label
    """

    def __init__(
        self,
        n_trials: int         = OPTUNA_N_TRIALS,
        timeout: int          = OPTUNA_TIMEOUT_SEC,
        model_type: ModelType = "XGBoost",
    ):
        self.n_trials   = n_trials
        self.timeout    = timeout
        self.model_type = model_type

        self.budget_imputer_  = BudgetImputer(n_trials=n_trials, timeout=timeout)
        self.gross_predictor_ = GrossPredictor(
            model_type=model_type, include_budget=True,
            n_trials=n_trials, timeout=timeout,
            name="GrossPredictor_WithBudget",
        )

        # Populated during fit — needed for inference-time encoding
        self._te_maps_log_g: Dict  = {}
        self._te_maps_score: Dict  = {}
        self._te_global_log_g: Dict = {}
        self._te_global_score: Dict = {}
        self._studio_median: Dict  = {}

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "RoiClassifier":
        # Store encoding maps for inference
        for col in ["director", "star", "writer", "company"]:
            for target, store in [
                ("log_gross", self._te_maps_log_g),
                ("score",     self._te_maps_score),
            ]:
                gm = train[target].mean()
                if target == "log_gross":
                    self._te_global_log_g[col] = gm
                else:
                    self._te_global_score[col] = gm
                stats = train.groupby(col)[target].agg(["mean", "count"])
                sm = 10.0
                store[col] = (
                    (stats["count"] * stats["mean"] + sm * gm) / (stats["count"] + sm)
                ).to_dict()

        self._studio_median = train.groupby("company")["gross"].median().to_dict()

        log.info("Training BudgetImputer …")
        self.budget_imputer_.fit(train, val)

        train_imp = self.budget_imputer_.impute(train)
        val_imp   = self.budget_imputer_.impute(val)

        log.info("Training GrossPredictor (%s) …", self.model_type)
        self.gross_predictor_.fit(train_imp, val_imp)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        df_imp = self.budget_imputer_.impute(df)
        return self.gross_predictor_.predict_roi_class(df_imp)

    def evaluate(self, df: pd.DataFrame, split_label: str = "test") -> Dict:
        df_imp     = self.budget_imputer_.impute(df)
        log_pred   = self.gross_predictor_.predict_log_gross(df_imp)
        metrics: Dict = {}

        # Regression metrics
        gross_mask = df["gross"].notna()
        if gross_mask.sum() > 0:
            log_true = df.loc[gross_mask, "log_gross"].values
            metrics.update(_regression_metrics(log_true, log_pred[gross_mask],
                                               label=f"{split_label}_gross_"))

        # Classification metrics
        roi_pred = self.gross_predictor_.predict_roi_class(df_imp)
        clf_mask = df["roi_class"].notna()
        if clf_mask.sum() > 0:
            metrics.update(_classification_metrics(
                df.loc[clf_mask, "roi_class"].values,
                roi_pred[clf_mask],
                label=f"{split_label}_roi_",
            ))

        # Budget imputation quality (on rows with known budget)
        bud_mask = df["budget"].notna()
        if bud_mask.sum() > 0:
            fc = _safe_cols(df_imp, self.budget_imputer_.feature_cols_)
            bud_pred = self.budget_imputer_.model_.predict(df_imp.loc[bud_mask, fc])
            bud_true = np.log1p(df.loc[bud_mask, "budget"].values)
            metrics.update(_regression_metrics(bud_true, bud_pred,
                                               label=f"{split_label}_budget_imputer_"))

        log.info("[%s] accuracy=%.4f  r2_log_gross=%.4f  mape=%.1f%%",
                 split_label,
                 metrics.get(f"{split_label}_roi_accuracy", float("nan")),
                 metrics.get(f"{split_label}_gross_r2_log",  float("nan")),
                 metrics.get(f"{split_label}_gross_mape_pct", float("nan")))
        return metrics

    def save(self, dir_: Optional[Path] = None) -> Path:
        dir_  = dir_ or MODEL_DIR
        path  = dir_ / "roi_classifier.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("RoiClassifier saved → %s", path)
        return path

    @staticmethod
    def load(path: Path) -> "RoiClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner  (called by train.py)
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    n_trials: int         = OPTUNA_N_TRIALS,
    timeout: int          = OPTUNA_TIMEOUT_SEC,
    save_models: bool     = True,
    model_type: ModelType = "XGBoost",
) -> Tuple[RoiClassifier, Dict]:
    clf = RoiClassifier(n_trials=n_trials, timeout=timeout, model_type=model_type)
    clf.fit(train, val)

    metrics: Dict = {}
    metrics.update(clf.evaluate(val,  split_label="val"))
    metrics.update(clf.evaluate(test, split_label="test"))

    # Feature importance CSV
    fi = clf.gross_predictor_.feature_importance_df()
    fi.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # Metrics JSON (serialisation-safe subset)
    safe_metrics = {
        k: v for k, v in metrics.items()
        if not isinstance(v, dict) or k.endswith("confusion_matrix") or k.endswith("_labels")
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(safe_metrics, f, indent=2)

    if save_models:
        clf.save()
        clf.budget_imputer_.save()
        clf.gross_predictor_.save()

    return clf, metrics
