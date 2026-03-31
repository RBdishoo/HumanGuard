# HumanGuard

Real-time bot detection API using behavioral biometrics and machine learning.

---

## Live Demo

| Page | URL |
|---|---|
| **Demo** (take the challenge) | https://d1hi33wespusty.cloudfront.net/demo.html |
| **Leaderboard** | https://d1hi33wespusty.cloudfront.net/leaderboard.html |
| **Dashboard** (monitoring) | https://d1hi33wespusty.cloudfront.net/dashboard.html |
| **Register** (get an API key) | https://d1hi33wespusty.cloudfront.net/register.html |
| **Client Dashboard** | https://d1hi33wespusty.cloudfront.net/client_dashboard.html |
| **SDK** | https://d1hi33wespusty.cloudfront.net/sdk/humanGuard.min.js |
| **Lambda API** | https://9ixzk5e9u4.execute-api.us-east-1.amazonaws.com |

---

## Overview

HumanGuard analyzes raw browser behavioral signals — mouse trajectories, keystroke dynamics, click patterns, and session timing — to classify web sessions as human or bot in real time. A 30-feature extraction pipeline feeds a RandomForest classifier that achieves 99.6% F1 on cross-validation, with every prediction accompanied by a SHAP feature attribution breakdown. A session-level temporal blender catches adaptive bots that mimic humans early in a session and revert to scripted behavior later, achieving **100% detection across 5 hard adversarial bot patterns**. The system is deployed as a containerized Flask API on AWS Lambda behind API Gateway, with RDS PostgreSQL for persistence, CloudWatch metrics and alarms for production observability, and an S3-hosted live dashboard for monitoring.

---

## Architecture

```
Browser (tracker.js)
        │  POST /api/signals  (batches every 3s)
        ▼
  API Gateway (HTTP API)
        │
        ▼
  Lambda — Flask + XGBoost
        │
        ├──► RDS PostgreSQL ──► sessions / signal_batches / predictions
        │
        ├──► CloudWatch Metrics ──► HumanGuard namespace
        │              │
        │              └──► 4 Alarms ──► SNS ──► Email Alerts
        │
        └──► S3 Static Site (dashboard.html)
```

Signal collection, model inference, and persistence are fully decoupled. A JSONL flat-file path remains active as a fallback whenever PostgreSQL is unavailable.

---

## Features

- **Behavioral signal collection** — JavaScript tracker captures mouse movements (100 ms throttle), keystroke timings, click coordinates, and scroll events; auto-batches every 3 seconds
- **30-feature extraction pipeline** — mouse velocity/acceleration/path efficiency, click clustering/rate, keystroke entropy/inter-key delay statistics, session-consistency features (timing regularity, rhythm autocorrelation, acceleration variance)
- **RandomForest classifier** — 99.6% F1, 1.0000 ROC-AUC on 5-fold cross-validation; selected over LogisticRegression and XGBoost baselines
- **Temporal drift scoring** — session blender detects adaptive bots by measuring behavioral drift between first-half and second-half of a session; `temporal_drift_score`, `early_late_timing_delta`, and `behavior_consistency_score` expose pattern shifts invisible to batch-level scoring
- **Adversarial robustness** — 100% session-level detection across 5 hard bot patterns: human_speed_typer, bezier_mouse, jitter_bot, hybrid_bot, and adaptive_bot; hard test F1: 1.0000
- **SHAP explainability** — every `/api/score` response includes top-5 feature contributions with human-readable interpretation text
- **Session-layer scoring** — `/api/session-score` aggregates all batches for a session, applies linear recency weighting, blends with temporal drift via session blender for sessions ≥10 batches
- **RDS PostgreSQL persistence** — connection-pooled writes via psycopg2; `DatabaseManager` auto-selects SQLite for local dev when `DATABASE_URL` is unset
- **CloudWatch monitoring** — 5 custom metrics (`score_requests`, `bot_detections`, `human_detections`, `prediction_latency_ms`, `validation_errors`); 4 production alarms
- **SNS alerting** — alarms publish to `HumanGuard-Alerts` topic; email subscription via `SNS_ALERT_EMAIL`
- **Live dashboard** — S3-hosted dark-theme monitoring UI with real-time chart, prediction feed, and SHAP feature importance bars
- **112 automated tests** — pytest suite covering API endpoints, feature extraction, classifier pipeline, SHAP output, DB layer, session scoring, signal validation, demo/export endpoints, model registry, and leaderboard

---

## API Reference

### `GET /health`

Liveness check. Does not load model artifacts.

```http
GET /health
```

```json
{
  "status": "ok",
  "model": "XGBoost",
  "version": "1.0.0",
  "uptime_seconds": 142.3,
  "timestamp": "2026-03-29T22:19:35.741528+00:00"
}
```

---

### `POST /api/score`

Score a single signal batch. Returns bot probability, label, classification threshold, and SHAP explanation.

Append `?explain=false` to skip SHAP computation for lower latency.

**Request**

```json
{
  "sessionID": "session_1743200000_abc123xyz",
  "signals": {
    "mouseMoves": [
      { "x": 412, "y": 308, "ts": 1000 },
      { "x": 428, "y": 315, "ts": 1087 },
      { "x": 441, "y": 323, "ts": 1201 }
    ],
    "clicks": [
      { "x": 441, "y": 323, "ts": 1450, "button": 0 }
    ],
    "keys": [
      { "key": "h", "ts": 1520 },
      { "key": "e", "ts": 1634 },
      { "key": "l", "ts": 1741 }
    ]
  }
}
```

**Response**

```json
{
  "success": true,
  "sessionID": "session_1743200000_abc123xyz",
  "prob_bot": 0.032,
  "label": "human",
  "threshold": 0.5,
  "explanation": {
    "interpretation": "Session classified as human; top signal: abnormal batch duration.",
    "top_features": [
      { "feature": "batchDurationMs",     "contribution": -2.174 },
      { "feature": "mouseStdVelocity",    "contribution":  0.768 },
      { "feature": "clickToMoveRatio",    "contribution": -0.348 },
      { "feature": "clickRatePerSec",     "contribution": -0.319 },
      { "feature": "mouseHoverFrequency", "contribution": -0.268 }
    ]
  }
}
```

---

### `POST /api/signals`

Ingest a raw signal batch for storage. Used by the browser tracker.

**Request** — same schema as `/api/score`.

**Response**

```json
{
  "success": true,
  "message": "Saved batch for session session_1743200000_abc123xyz",
  "Total Batches": 14,
  "Session ID": "session_1743200000_abc123xyz"
}
```

---

### `GET /api/stats`

Collection statistics plus live prediction counts from RDS.

```json
{
  "Total Batches": 214,
  "Unique Sessions": 38,
  "Signals File Size (in Kb)": 512.4,
  "total_predictions": 189,
  "bot_count": 42,
  "human_count": 147,
  "bot_rate": 0.2222,
  "Server Timestamp": "2026-03-29T22:22:55.075875Z"
}
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **ML** | XGBoost, scikit-learn, SHAP | Gradient boosting for tabular behavioral features; SHAP for production explainability |
| **Backend** | Flask, Python 3.11 | Lightweight WSGI; minimal cold-start footprint on Lambda |
| **Runtime** | AWS Lambda (container image) + awslambdaric | Serverless; scales to zero; container image supports the full ML dependency stack |
| **API Gateway** | AWS API Gateway HTTP API | Low-latency HTTP proxy; routes all methods via `ANY /{proxy+}` |
| **Database** | PostgreSQL (AWS RDS), SQLite (local) | `DatabaseManager` auto-selects backend from `DATABASE_URL`; psycopg2 connection pooling |
| **Container Registry** | AWS ECR | Private registry for Lambda container images |
| **Monitoring** | AWS CloudWatch | Custom metrics namespace, metric-math ratio alarms, p95 latency alarm |
| **Alerting** | AWS SNS | Email notifications on alarm state transitions |
| **Secrets** | AWS Secrets Manager | RDS credentials stored as `humanGuard/rds`; fetched at deploy time |
| **Dashboard** | S3 static website | Zero-server frontend; canvas chart polling `/api/dashboard-stats` |
| **Testing** | pytest | 112 tests across 13 test files |

---

## Local Development

```bash
# 1. Clone and create virtual environment
git clone https://github.com/RBdishoo/HumanGuard.git
cd HumanGuard
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Generate synthetic training data and retrain the model
python scripts/seed_bot_session.py
python scripts/seed_human_session.py
python -m models.run_training

# 4. Start the server  (frontend + API on http://localhost:5050)
python -m backend.app

# 5. Run the test suite
pytest tests/ -v
```

The server auto-detects the absence of `DATABASE_URL` and falls back to a local SQLite file at `backend/data/humanguard.db`. CloudWatch metrics are no-ops unless `CLOUDWATCH_ENABLED=true` is set.

---

## Deployment

Deployment targets AWS Lambda using a container image. The full pipeline is automated in `scripts/aws_deploy.sh`:

1. **ECR** — creates a private repository and pushes the `linux/amd64` Docker image
2. **IAM** — creates the Lambda execution role with CloudWatch and basic execution policies
3. **Lambda** — deploys the container image with 1 GB memory and a 60 s timeout
4. **API Gateway** — creates an HTTP API with a catch-all `ANY /{proxy+}` → Lambda integration

```bash
# One-time infrastructure setup
DB_PASSWORD=<strong-password> python infrastructure/rds_setup.py
python infrastructure/cloudwatch_alarms.py

# Full deploy (or initial deploy)
bash scripts/aws_deploy.sh
```

For redeployments when Lambda already exists:

```bash
docker build --platform linux/amd64 -t humanguard:latest .
docker tag humanguard:latest <account>.dkr.ecr.us-east-1.amazonaws.com/humanguard:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/humanguard:latest
aws lambda update-function-code --function-name humanguard \
  --image-uri <account>.dkr.ecr.us-east-1.amazonaws.com/humanguard:latest
```

---

## Project Structure

```
HumanGuard/
├── backend/
│   ├── app.py                      # Flask application, all API routes
│   ├── monitoring.py               # CloudWatch metrics singleton
│   ├── collectors/
│   │   └── signal_collector.py     # JSONL signal batch writer
│   ├── db/
│   │   ├── __init__.py             # DatabaseManager (SQLite / PostgreSQL)
│   │   ├── db_client.py            # PostgreSQL connection pool
│   │   ├── migrate.py              # JSONL → PostgreSQL migration script
│   │   └── schema.sql              # DDL: sessions, signal_batches, predictions
│   ├── features/
│   │   ├── feature_extractor.py    # 33-feature extraction from raw signals
│   │   ├── feature_utils.py        # Mouse trajectory and keystroke math utilities
│   │   ├── dataset_builder.py      # Batch/session-level CSV dataset builder
│   │   └── data_loader.py          # JSONL signal loader and validator
│   └── utils/
│       └── helpers.py              # Validation, normalization, timestamp helpers
├── models/
│   ├── dataset.py                  # ModelDataset: feature loading, train/test split, scaling
│   ├── train.py                    # ModelTrainer: RandomForest, LogisticRegression, XGBoost
│   ├── evaluate.py                 # Metrics reports and confusion matrix plots
│   ├── run_training.py             # Training entry point
│   └── trained/                    # Serialized artifacts (XGBoost.pkl, scaler.pkl, …)
├── frontend/
│   ├── tracker.js                  # Browser signal collector (auto-batches every 3s)
│   ├── dashboard.html              # Live monitoring dashboard
│   ├── index.html                  # Demo frontend
│   └── style.css
├── infrastructure/
│   ├── rds_setup.py                # Idempotent RDS + Secrets Manager provisioning
│   └── cloudwatch_alarms.py        # Creates/updates 4 CloudWatch alarms and SNS topic
├── scripts/
│   ├── aws_deploy.sh               # Full ECR → Lambda → API Gateway deploy
│   ├── seed_bot_session.py         # Synthetic bot session generator
│   ├── seed_human_session.py       # Synthetic human session generator
│   └── report_training_summary.py  # Prints model comparison metrics
├── tests/                          # 70 pytest tests across 11 files
├── Dockerfile                      # python:3.11-slim, linux/amd64, awslambdaric entrypoint
├── requirements.txt                # Development dependencies
└── requirements-prod.txt           # Production dependencies (boto3, psycopg2-binary, shap)
```

---

## License

MIT
