"""
api.py
======
Production-grade FastAPI REST API for the Movie ROI Prediction Pipeline.

Endpoints
---------
  GET  /                        — health check + version info
  GET  /health                  — readiness probe (model loaded?)
  POST /predict                 — single-movie prediction
  POST /predict/batch           — batch prediction (up to 50 movies)
  GET  /model/info              — model metadata and feature list
  GET  /model/metrics           — last-run evaluation metrics
  POST /movie/summarize         — AI-powered movie summary (Gemini — placeholder)
  POST /train                   — trigger a fresh training run (admin)

Run locally
-----------
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Docker
------
    docker build -t movie-roi-api .
    docker run -p 8000:8000 movie-roi-api

Authentication
--------------
  Set the environment variable API_KEY to enable Bearer token auth.
  If unset, auth is disabled (suitable for local development only).

Environment Variables
---------------------
  MODEL_PATH   : path to roi_classifier.pkl   (default: models/roi_classifier.pkl)
  API_KEY      : bearer token for auth         (default: disabled)
  LOG_LEVEL    : logging verbosity             (default: INFO)
"""

from __future__ import annotations

import json
import os
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator

# Pipeline imports (assumes api.py lives alongside the other modules)
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_DIR, OUTPUT_DIR, ROI_LABELS
from logger import get_logger
from models import RoiClassifier
from train import predict_single_movie, run_pipeline

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

API_VERSION  = "1.0.0"
API_TITLE    = "Movie Box-Office ROI Prediction API"
MODEL_PATH   = Path(os.getenv("MODEL_PATH", str(MODEL_DIR / "roi_classifier.pkl")))
_API_KEY     = os.getenv("API_KEY", "")          # empty string = auth disabled
MAX_BATCH    = 50

# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown  (model loading)
# ─────────────────────────────────────────────────────────────────────────────

_state: Dict[str, Any] = {"clf": None, "loaded_at": None, "metrics": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup; release on shutdown."""
    _load_model()
    yield
    log.info("API shutting down.")


def _load_model(path: Optional[Path] = None) -> None:
    p = path or MODEL_PATH
    if p.exists():
        try:
            _state["clf"]       = RoiClassifier.load(p)
            _state["loaded_at"] = datetime.now(timezone.utc).isoformat()
            log.info("Model loaded from %s", p)
        except Exception as exc:
            log.error("Failed to load model: %s", exc)
            _state["clf"] = None
    else:
        log.warning("Model file not found: %s — /predict will return 503 until trained.", p)

    # Load last metrics if available
    metrics_path = OUTPUT_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            _state["metrics"] = json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = API_TITLE,
    version     = API_VERSION,
    description = __doc__,
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Auth (optional Bearer token)
# ─────────────────────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> None:
    if not _API_KEY:
        return   # auth disabled
    if credentials is None or credentials.credentials != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class MovieInput(BaseModel):
    """Input schema for a single movie.  budget may be omitted if unknown."""

    name          : str   = Field(...,  example="Inception",     description="Movie title")
    genre         : str   = Field(...,  example="Sci-Fi",        description="Primary genre")
    rating        : str   = Field(...,  example="PG-13",         description="MPAA rating")
    year          : int   = Field(...,  ge=1920, le=2035,        description="Release year")
    runtime       : float = Field(...,  ge=30, le=600,           description="Runtime in minutes")
    release_month : int   = Field(...,  ge=1, le=12,             description="Release month (1–12)")
    director      : str   = Field(...,  example="Christopher Nolan")
    writer        : str   = Field(...,  example="Christopher Nolan")
    star          : str   = Field(...,  example="Leonardo DiCaprio")
    company       : str   = Field(...,  example="Warner Bros.")
    country       : str   = Field("United States", example="United States")
    votes         : float = Field(0.0, ge=0,                    description="Estimated IMDb votes")
    score         : Optional[float] = Field(None, ge=0, le=10,  description="IMDb score (optional)")
    budget        : Optional[float] = Field(None, ge=0,         description="Production budget $; omit if unknown")
    release_day   : int   = Field(15, ge=1, le=31,              description="Release day of month")

    @validator("rating")
    def rating_must_be_valid(cls, v):
        valid = {"G", "PG", "PG-13", "R", "NC-17", "NR", "TV-MA", "TV-14", "Not Rated"}
        if v not in valid:
            v = "R"   # fallback — don't hard-fail on non-standard ratings
        return v

    class Config:
        schema_extra = {
            "example": {
                "name": "Inception", "genre": "Sci-Fi", "rating": "PG-13",
                "year": 2010, "runtime": 148, "release_month": 7,
                "director": "Christopher Nolan", "writer": "Christopher Nolan",
                "star": "Leonardo DiCaprio", "company": "Warner Bros.",
                "country": "United States", "votes": 2200000, "score": 8.8,
                "budget": 160000000,
            }
        }


class PredictionResponse(BaseModel):
    predicted_roi_class  : str
    predicted_gross_usd  : float
    roi_probability      : Dict[str, float]
    budget_imputed       : bool
    imputed_budget_usd   : Optional[float]
    pipeline_used        : str
    latency_ms           : float


class BatchPredictionRequest(BaseModel):
    movies: List[MovieInput] = Field(..., max_items=MAX_BATCH)


class BatchPredictionResponse(BaseModel):
    results   : List[Dict]
    total     : int
    errors    : List[Dict]
    latency_ms: float


class HealthResponse(BaseModel):
    status       : str
    model_loaded : bool
    loaded_at    : Optional[str]
    version      : str


class ModelInfoResponse(BaseModel):
    model_type    : str
    feature_count : int
    roi_labels    : List[str]
    loaded_at     : Optional[str]
    val_metrics   : Dict
    test_metrics  : Dict


class TrainRequest(BaseModel):
    data_path      : str  = Field(..., description="Server-side path to dataset")
    n_trials       : int  = Field(40,  ge=5,  le=200)
    timeout        : int  = Field(300, ge=30, le=1800)
    run_comparison : bool = Field(True)


class TrainResponse(BaseModel):
    status       : str
    model_type   : str
    val_accuracy : Optional[float]
    test_accuracy: Optional[float]
    message      : str


class SummarizeRequest(BaseModel):
    """Input for the Gemini-powered movie summary endpoint."""
    movie_name : str  = Field(..., example="The Dark Knight")
    context    : Optional[str] = Field(None, description="Extra context (genre, cast, etc.)")


class SummarizeResponse(BaseModel):
    movie_name : str
    summary    : str
    source     : str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _require_model() -> RoiClassifier:
    clf = _state.get("clf")
    if clf is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. POST /train first or ensure models/roi_classifier.pkl exists.",
        )
    return clf


def _movie_input_to_dict(movie: MovieInput) -> dict:
    d = movie.dict()
    d["gross"] = None   # never known at inference time
    return d


def _extract_split_metrics(prefix: str) -> Dict:
    m = _state.get("metrics", {})
    return {
        k.replace(f"{prefix}_", ""): v
        for k, v in m.items()
        if k.startswith(f"{prefix}_") and not isinstance(v, (dict, list))
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes — General
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"], summary="Root / version info")
def root():
    return {
        "service"  : API_TITLE,
        "version"  : API_VERSION,
        "docs"     : "/docs",
        "health"   : "/health",
        "model_loaded": _state["clf"] is not None,
    }


@app.get("/health", response_model=HealthResponse, tags=["General"], summary="Readiness probe")
def health():
    return HealthResponse(
        status       = "ok" if _state["clf"] else "degraded",
        model_loaded = _state["clf"] is not None,
        loaded_at    = _state.get("loaded_at"),
        version      = API_VERSION,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Prediction
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict ROI class and estimated gross for a single movie",
    dependencies=[Depends(require_auth)],
)
def predict(movie: MovieInput):
    clf = _require_model()
    t0  = time.perf_counter()

    try:
        result = predict_single_movie(clf, _movie_input_to_dict(movie))
    except Exception as exc:
        log.error("Prediction error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    latency = (time.perf_counter() - t0) * 1000
    return PredictionResponse(**result, latency_ms=round(latency, 2))


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary=f"Batch predict up to {MAX_BATCH} movies",
    dependencies=[Depends(require_auth)],
)
def predict_batch(request: BatchPredictionRequest):
    clf = _require_model()
    t0  = time.perf_counter()

    results, errors = [], []
    for i, movie in enumerate(request.movies):
        try:
            r = predict_single_movie(clf, _movie_input_to_dict(movie))
            r["input_name"] = movie.name
            results.append(r)
        except Exception as exc:
            errors.append({"index": i, "name": movie.name, "error": str(exc)})

    latency = (time.perf_counter() - t0) * 1000
    return BatchPredictionResponse(
        results    = results,
        total      = len(results),
        errors     = errors,
        latency_ms = round(latency, 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Model Info
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Model metadata and feature list",
    dependencies=[Depends(require_auth)],
)
def model_info():
    clf = _require_model()
    return ModelInfoResponse(
        model_type    = clf.model_type,
        feature_count = len(clf.gross_predictor_.feature_cols_),
        roi_labels    = ROI_LABELS,
        loaded_at     = _state.get("loaded_at"),
        val_metrics   = _extract_split_metrics("val"),
        test_metrics  = _extract_split_metrics("test"),
    )


@app.get(
    "/model/metrics",
    tags=["Model"],
    summary="Full evaluation metrics from last training run",
    dependencies=[Depends(require_auth)],
)
def model_metrics():
    m = _state.get("metrics", {})
    if not m:
        raise HTTPException(status_code=404, detail="No metrics available — train the model first.")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Training  (admin)
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/train",
    response_model=TrainResponse,
    tags=["Admin"],
    summary="Trigger a fresh training run (long-running — use background tasks in prod)",
    dependencies=[Depends(require_auth)],
)
def train(request: TrainRequest, background_tasks: BackgroundTasks):
    data_path = Path(request.data_path)
    if not data_path.exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found: {data_path}")

    def _train_and_reload():
        try:
            clf, metrics, _ = run_pipeline(
                data_path      = data_path,
                n_trials       = request.n_trials,
                timeout        = request.timeout,
                save_models    = True,
                run_comparison = request.run_comparison,
            )
            _state["clf"]     = clf
            _state["metrics"] = {
                k: v for k, v in metrics.items()
                if not isinstance(v, (dict, list)) or k.endswith("confusion_matrix")
            }
            _state["loaded_at"] = datetime.now(timezone.utc).isoformat()
            log.info("Background training complete.")
        except Exception as exc:
            log.error("Background training failed: %s", exc)

    background_tasks.add_task(_train_and_reload)
    return TrainResponse(
        status        = "training_started",
        model_type    = "TBD",
        val_accuracy  = None,
        test_accuracy = None,
        message       = "Training started in the background. Poll /health for status.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes — Gemini Movie Summariser  
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/movie/summarize",
    response_model=SummarizeResponse,
    tags=["AI Features"],
    summary="Get an AI-generated summary for a movie (Gemini API — placeholder)",
    dependencies=[Depends(require_auth)],
)
def summarize_movie(request: SummarizeRequest):
    """
    ──────────────────────────────────────────────────────────────────────────
    PLACEHOLDER — Gemini integration not yet implemented.

    TO IMPLEMENT:
        1. Install: pip install google-generativeai
        2. Set env var: GEMINI_API_KEY=your_key
        3. Replace the stub below with the logic from gemini_service.py
    ──────────────────────────────────────────────────────────────────────────
    """
  
    try:
        from gemini_service import get_movie_summary  # noqa: F401
        summary = get_movie_summary(request.movie_name, request.context)
    except ImportError:
        summary = (
            f"[PLACEHOLDER] Gemini summary for '{request.movie_name}' "
            "will appear here once gemini_service.py is implemented."
        )

    return SummarizeResponse(
        movie_name = request.movie_name,
        summary    = summary,
        source     = "gemini-pro (placeholder)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host    = "0.0.0.0",
        port    = int(os.getenv("PORT", 8000)),
        reload  = os.getenv("ENV", "production") == "development",
        workers = int(os.getenv("WORKERS", 1)),
        log_level = os.getenv("LOG_LEVEL", "info").lower(),
    )
