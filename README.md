# HumanGuard

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?style=flat&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-EC6B2D?style=flat)
![AWS Lambda](https://img.shields.io/badge/AWS_Lambda-deployed-FF9900?style=flat&logo=awslambda&logoColor=white)
![pytest](https://img.shields.io/badge/tests-61_passed-2ea44f?style=flat&logo=pytest&logoColor=white)

HumanGuard is a machine learning bot detection system that classifies web sessions as human or bot by analyzing behavioral signals — mouse trajectories, keystroke timing, click patterns, and scroll dynamics — collected passively from the browser with no user friction. A lightweight JavaScript tracker batches raw events and sends them to a Flask API, which extracts 33 behavioral features and scores each batch in real time using a trained XGBoost classifier. Predictions include a continuous `prob_bot` score, a binary label, and a SHAP explanation identifying the top features that drove the decision, making every classification auditable and interpretable.

---

## Live

| | URL |
|---|---|
| **Dashboard** | http://humanguard-dashboard.s3-website-us-east-1.amazonaws.com |
| **API** | https://9ixzk5e9u4.execute-api.us-east-1.amazonaws.com |
| **Health check** | https://9ixzk5e9u4.execute-api.us-east-1.amazonaws.com/health |

---

## Architecture

```
Browser
  │  tracker.js — mouse/key/click events, batched every 3s
  │
  ▼
API Gateway (HTTP API v2)
  │
  ▼
AWS Lambda  ─────────────────────────────────────────────────────────
  │                                                                   │
  ├─ POST /api/signals ──► SignalCollector                            │
  │                            └─ append → /tmp/signals.jsonl         │
  │                            └─ dual-write → PostgreSQL (if set)    │
  │                                                                   │
  ├─ POST /api/score ────► FeatureExtractor (33 features)             │
  │   POST /api/session-score   └─ StandardScaler                     │
  │                             └─ XGBoost.predict_proba()            │
  │                             └─ SHAP TreeExplainer (top 5)         │
  │                             └─ log → /tmp/predictions_log.jsonl   │
  │                                                                   │
  └─ GET  /api/dashboard-stats ◄─ predictions_log.jsonl / PostgreSQL  │
                                                                      │
AWS S3 (static)                                                       │
  └─ dashboard.html ──────────────────────────────────── fetches ─────┘
```

**Feature categories (33 total):**
- **Mouse** — velocity, acceleration, path efficiency, angular velocity, hover ratio, pause count
- **Click** — rate, inter-click timing variance, clustering ratio, left/right ratio
- **Keystroke** — inter-key delay mean/std, Shannon entropy, rapid-press count, typing rate
- **Session** — batch duration, event rate, signal diversity entropy, cross-signal ratios

---

## API Reference

All endpoints are at `https://9ixzk5e9u4.execute-api.us-east-1.amazonaws.com`.

### `GET /health`
Liveness check. Does not load model artifacts.

```json
{
  "status": "ok",
  "model": "XGBoost",
  "version": "1.0.0",
  "uptime_seconds": 4.2,
  "timestamp": "2026-03-29T02:42:14.202846+00:00"
}
```

---

### `POST /api/signals`
Save a raw signal batch. Called automatically by `tracker.js` every 3 seconds.

**Request**
```json
{
  "sessionID": "abc-123",
  "signals": {
    "mouseMoves": [{ "x": 100, "y": 200, "ts": 1000 }],
    "clicks":     [{ "x": 140, "y": 250, "button": 0, "ts": 1800 }],
    "keys":       [{ "key": "a", "code": "KeyA", "ts": 2100 }]
  },
  "metadata": { "userAgent": "...", "viewportWidth": 1440, "viewportHeight": 900 }
}
```

**Response**
```json
{ "success": true, "message": "Saved batch for session abc-123", "Total Batches": 42 }
```

---

### `POST /api/score`
Score a single signal batch. Returns bot probability, label, and SHAP explanation.
Add `?explain=false` to skip SHAP for lower latency.

**Request** — same shape as `/api/signals`

**Response**
```json
{
  "success": true,
  "sessionID": "abc-123",
  "prob_bot": 0.034,
  "label": "human",
  "threshold": 0.5,
  "explanation": {
    "top_features": [
      { "feature": "batchDurationMs",   "contribution": -2.157 },
      { "feature": "mouseStdVelocity",  "contribution": -0.673 },
      { "feature": "mouseHoverTimeRatio","contribution": -0.354 },
      { "feature": "clickToMoveRatio",  "contribution": -0.348 },
      { "feature": "clickRatePerSec",   "contribution": -0.319 }
    ],
    "interpretation": "Session classified as human; top signal: abnormal batch duration."
  }
}
```

---

### `POST /api/session-score` · `GET /api/session-score/<sessionID>`
Session-level scoring — aggregates all batches for a session with linear weighting (later batches weighted higher), drift analysis, and SHAP for the peak batch. Requires ≥ 2 batches.

**Response**
```json
{
  "success": true,
  "sessionID": "abc-123",
  "session_prob_bot": 0.061,
  "label": "human",
  "threshold": 0.5,
  "batch_count": 5,
  "batch_scores": [0.04, 0.07, 0.05, 0.08, 0.06],
  "drift": {
    "trend": "stable",
    "drift_score": 0.014,
    "max_prob_bot": 0.08,
    "mean_prob_bot": 0.06
  },
  "explanation": { "..." : "..." }
}
```

---

### `GET /api/stats`
Signal collection statistics (batch count, session count, file size).

### `GET /api/dashboard-stats`
Aggregated monitoring stats for the live dashboard: totals, bot rate, avg response time, last 10 predictions, top 5 SHAP-flagged features. Reads from PostgreSQL if `DATABASE_URL` is set, otherwise falls back to the local predictions log.

---

## Local Development

**Prerequisites:** Python 3.11+, pip

```bash
# 1. Clone and install
git clone https://github.com/RBdishoo/HumanGuard.git
cd HumanGuard
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run the server (http://localhost:5050)
python -m backend.app

# 3. Open the tracker demo
open http://localhost:5050

# 4. Open the dashboard
open http://localhost:5050/dashboard
```

**Train the model** (required before `/api/score` works):
```bash
# Generate synthetic training data
python scripts/seed_bot_session.py
python scripts/seed_bot_session.py --stealthy
python scripts/seed_human_session.py

# Train and save artifacts to models/trained/
python -m models.run_training

# Validate results
python scripts/report_training_summary.py
```

**Run tests:**
```bash
pytest tests/ -v          # all 61 tests
pytest tests/test_api.py  # single file
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5050` | Flask server port |
| `DATABASE_URL` | — | PostgreSQL connection string (optional) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Signal capture | Vanilla JS (`tracker.js`), throttled to 100ms mouse events |
| API server | Python 3.11 · Flask 3.1 · Werkzeug |
| Feature engineering | NumPy 2.4 · Pandas 2.3 · 33 behavioral features |
| ML model | XGBoost 3.2 · scikit-learn 1.8 · StandardScaler |
| Explainability | SHAP 0.51 · TreeExplainer · top-5 feature attribution |
| Storage | JSONL (local/Lambda) · PostgreSQL via `DATABASE_URL` |
| Deployment | Docker (linux/amd64) · AWS Lambda · API Gateway HTTP v2 |
| Static hosting | AWS S3 static website |
| Tests | pytest · 61 tests across 8 files |

---

## Project Status

See [PROGRESS.md](PROGRESS.md) for the full phase-by-phase build log including architecture decisions, model comparison metrics, and deployment notes.

| Phase | Status |
|---|---|
| 1 — Signal Collection | ✅ Complete |
| 2 — Feature Engineering | ✅ Complete |
| 3 — Data Collection & Labeling | ✅ Complete |
| 4 — ML Model Training | ✅ Complete (XGBoost, ROC-AUC 0.9135) |
| 5 — Deployment & Monitoring | ✅ Complete (Lambda + S3 dashboard live) |
| 6 — Advanced Features | 🔄 In progress |
