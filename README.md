# 🎬 Movie Box-Office ROI Prediction Pipeline

> Two-stage XGBoost pipeline implementing the Cocuzzo \& Wu (2013) \*"Hit or Flop"\* methodology, extended with multi-model comparison, a production FastAPI, and a Streamlit UI.

\---

## Architecture

```
Input Movie
     │
     ├─ budget known? ──YES──► Pipeline A
     │                           GrossPredictor → log(gross) → ROI bin → label
     │
     └─ budget missing? ─NO──► Pipeline B
                                 BudgetImputer → imputed budget
                                 GrossPredictor → log(gross) → ROI bin → label
```

Both pipelines share the same `GrossPredictor`. The imputed budget becomes just another input feature.

\---

## ROI Bins

|Label|ROI Range|Meaning|
|-|-|-|
|`flop`|\[0, 0.5)|Lost money|
|`below\_avg`|\[0.5, 1)|Broke even or modest returns|
|`hit`|\[1, 2)|Profitable|
|`blockbuster`|\[2, ∞)|Major commercial success|

\---

## File Structure

```
movie\_pipeline/
├── config.py           # All constants, paths, hyperparameter spaces
├── logger.py           # Centralised logging (file + console)
├── features.py         # Loading, ROI binning, feature engineering, splits
├── models.py           # BudgetImputer, GrossPredictor, RoiClassifier, compare\_gross\_predictors
├── evaluation.py       # Metrics, all charts, text report
├── train.py            # CLI + run\_pipeline() + predict\_single\_movie()
├── api.py              # Production FastAPI REST API
├── streamlit\_app.py    # Full Streamlit UI (3 tabs)
├── gemini\_service.py   # AI movie summariser boilerplate (Gemini placeholder)
├── requirements.txt
├── data/               # Place movies\_cleaned.xlsx here
├── models/             # Saved .pkl files
├── outputs/            # Charts, metrics.json, evaluation\_report.txt
└── logs/               # pipeline.log
```

\---

## Quick Start

### 1\. Install

```bash
pip install -r requirements.txt
```

### 2\. Train (CLI)

```bash
# Full run with model comparison (XGBoost vs LightGBM vs RandomForest)
python train.py --data data/movies\_cleaned.xlsx

# Fast dev run
python train.py --data data/movies\_cleaned.xlsx --trials 10 --timeout 60 --no-compare

# Skip model saving
python train.py --data data/movies\_cleaned.xlsx --no-save
```

### 3\. Streamlit UI

```bash
streamlit run streamlit\_app.py
```

### 4\. FastAPI

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

### 5\. Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD \["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

\---

## API Endpoints

|Method|Endpoint|Description|
|-|-|-|
|GET|`/`|Root / version info|
|GET|`/health`|Readiness probe|
|POST|`/predict`|Single movie prediction|
|POST|`/predict/batch`|Batch prediction (≤50 movies)|
|GET|`/model/info`|Model metadata + feature list|
|GET|`/model/metrics`|Last training metrics|
|POST|`/movie/summarize`|Gemini AI summary (placeholder)|
|POST|`/train`|Trigger training (background task)|

### Example: Single Predict

```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "name": "Inception", "genre": "Sci-Fi", "rating": "PG-13",
    "year": 2010, "runtime": 148, "release\_month": 7,
    "director": "Christopher Nolan", "writer": "Christopher Nolan",
    "star": "Leonardo DiCaprio", "company": "Warner Bros.",
    "votes": 2200000, "budget": 160000000
  }'
```

### Response

```json
{
  "predicted\_roi\_class": "blockbuster",
  "predicted\_gross\_usd": 836836967.00,
  "roi\_probability": {"flop": 0.01, "below\_avg": 0.02, "hit": 0.07, "blockbuster": 0.90},
  "budget\_imputed": false,
  "imputed\_budget\_usd": null,
  "pipeline\_used": "A (budget known)",
  "latency\_ms": 12.4
}
```

\---

## Model Comparison

Training automatically benchmarks three regressors on the validation set and selects the winner:

|Model|Val RMSE (log)|Notes|
|-|-|-|
|**XGBoost**|Optuna-tuned|Native NaN handling; usually wins|
|LightGBM|Default params|Faster; comparable accuracy|
|Random Forest|Default params|Interpretability baseline|

Results saved to `outputs/model\_comparison.json` and visualised in `outputs/model\_comparison.png`.

\---

## Key Design Decisions

|Decision|Rationale|
|-|-|
|`log1p(gross)` as target|Extreme right skew (skewness \~5.3)|
|Temporal split|No future leakage — train ≤2015, val 2016–2018, test 2019–2020|
|Target encoding inside fold|Prevents leakage from high-cardinality director/star|
|`budget\_available` flag|Missingness is MNAR — carries signal|
|Optuna Bayesian tuning|More efficient than grid search|
|XGBoost selected by default|Best benchmark on this dataset size|

\---

## Gemini Integration (Placeholder)

The **Movie Summarizer** feature in the Streamlit UI and the `/movie/summarize` API endpoint are scaffolded in `gemini\_service.py`.

To activate:

1. `pip install google-generativeai`
2. `export GEMINI\_API\_KEY=your\_key`
3. Implement `get\_movie\_summary()` in `gemini\_service.py` (guide is inline)

\---

## Environment Variables

|Variable|Default|Description|
|-|-|-|
|`MODEL\_PATH`|`models/roi\_classifier.pkl`|Path to saved model|
|`API\_KEY`|*(empty = disabled)*|Bearer token for API auth|
|`GEMINI\_API\_KEY`|*(empty)*|Google Gemini API key|
|`GEMINI\_MODEL`|`gemini-pro`|Gemini model name|
|`TMDB\_API\_KEY`|*(empty)*|TMDB API key for movie metadata|
|`PORT`|`8000`|API server port|
|`LOG\_LEVEL`|`INFO`|Logging verbosity|

\---

## Extending the Pipeline

* **Add a score predictor**: Create a `GrossPredictor` targeting `score`; feature engineering already includes `director\_te\_score`, `star\_te\_score`.
* **Add more features**: Edit `features.py` — `\_feature\_cols()` in `models.py` picks them up automatically.
* **Increase tuning budget**: `OPTUNA\_N\_TRIALS=200` in `config.py` or `--trials 200` on CLI.
* **Switch to LightGBM globally**: Pass `model\_type="LightGBM"` to `RoiClassifier`.
* **Add Gemini**: Implement `gemini\_service.get\_movie\_summary()` — no other file needs changing.

