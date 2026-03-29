# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HumanGuard is a machine learning bot detection system that classifies web sessions as human or bot
by analyzing behavioral signals (mouse movements, keystrokes, clicks, scrolling). The pipeline flows:
**Frontend signal collection → Flask API → JSONL storage → Feature extraction → ML training → Real-time scoring API**.

---

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run Flask server (serves frontend + API on localhost:5050, configurable via PORT env var)
python -m backend.app

# Run tests (53 tests across 8 test files)
pytest tests/ -v
pytest tests/test_signals.py          # single test file

# Train model (extracts features from signals.jsonl, trains, saves artifacts)
python -m models.run_training

# Generate synthetic data
python scripts/seed_bot_session.py           # bot sessions (--stealthy flag for evasive bots)
python scripts/seed_human_session.py         # human sessions

# Validate trained model
python scripts/report_training_summary.py
```

No linter or formatter is currently configured.

## Architecture

### Data Flow
Frontend (frontend/tracker.js): Vanilla JS captures mouse/click/key events (mouse throttled to 100ms), auto-sends batches every 3 seconds to /api/signals

Signal Collection (backend/collectors/signal_collector.py): Appends signal batches as JSONL to backend/data/raw/signals.jsonl, dual-writes to PostgreSQL if available

Feature Engineering (backend/features/): Extracts 33 behavioral features (mouse velocity/acceleration/path efficiency, click rate/clustering, keystroke entropy/timing, temporal composites) into CSV

Training (models/): ModelDataset loads features + labels, splits/scales; ModelTrainer trains RandomForest/LogisticRegression/XGBoost; selects best by ROC-AUC

Inference (backend/app.py): Lazy-loads trained model artifacts from models/trained/, extracts features from submitted signals, returns prob(bot) + label + SHAP explanation

### Key Modules
backend/features/feature_extractor.py — FeatureExtractor class: raw signals → 33 features

backend/features/feature_utils.py — Static math utilities (MouseTrajectoryUtils, KeystrokeUtils)

backend/features/dataset_builder.py — Builds batch/session-level CSV datasets

backend/features/data_loader.py — Loads and validates signals from JSONL

backend/utils/helpers.py — isValidSignalBatch(), formatTimestamp()

models/dataset.py — ModelDataset: merges features with labels, train/test split, scaling

models/train.py — ModelTrainer: trains classifiers, saves .pkl artifacts

models/evaluate.py — Evaluator: metrics reports, confusion matrix plots

### API Endpoints
POST /api/signals — Submit a signal batch

POST /api/score — Score a single signal batch. Returns bot probability, label, and SHAP explanation (top 5 feature contributions with human-readable interpretation). Caps input at 5000 mouse moves, 2000 clicks, 5000 keystrokes. Pass `?explain=false` to skip SHAP computation for lower latency

POST /api/session-score — Session-level scoring: aggregates all batches for a sessionID, returns linearly-weighted session prob_bot (later batches weighted higher), per-batch scores, drift analysis (trend, drift_score, max/mean prob_bot), and SHAP explanation for the peak batch

GET /api/session-score/<sessionID> — Convenience GET alias for session-level scoring

GET /api/stats — Signal collection statistics

GET /api/dashboard-stats — Aggregated stats for the live dashboard: total_predictions, bot_count, human_count, bot_rate, avg_response_time_ms, recent_predictions (last 10), top_flagged_features (top 5 SHAP feature names from bot detections), model, threshold. Reads from PostgreSQL if available, falls back to backend/data/predictions_log.jsonl

GET /dashboard — Serves frontend/dashboard.html (live monitoring dashboard)

GET /health — Liveness check (status, model name, version, uptime, timestamp). Does not load model artifacts

GET / — Serves frontend/index.html

### Trained Model Artifacts (models/trained/)
XGBoost.pkl — Active classifier (selected by ROC-AUC)

scaler.pkl — StandardScaler for feature normalization

feature_names.json — Ordered 33-feature list (must match extraction order)

threshold.json — Classification threshold (default 0.5)

model_comparison.json — Full metrics for all three models

⚠️ Never hand-edit these files. Always regenerate via `python -m models.run_training`.
When adding new features to FeatureExtractor, always re-run training to keep /api/score in sync.

### Data Storage
Signal events (mouseMoves, clicks, keys) use `ts` as the timestamp key (milliseconds since epoch).
The batch-level `timestamp` field is separate ISO-format metadata added by SignalCollector.

Raw signals: backend/data/raw/signals.jsonl — append-only, treat as read-only

Labels: backend/data/raw/labels.csv

Extracted features: backend/data/processed/training_data_batches.csv

.gitignore excludes data/raw/*.jsonl, models/*.pkl, .env

### Database
- Local dev: JSONL + CSV (no DATABASE_URL needed)
- Production: PostgreSQL via DATABASE_URL env var (AWS RDS)
- Dual-write: JSONL always written first; PostgreSQL written if available
- Schema: `backend/db/schema.sql` (sessions, signal_batches, labels, predictions)
- Predictions table includes `scoring_type` column ('batch' or 'session')
- Client: `backend/db/db_client.py` — connection pool, is_available() guard
- To run migration: `python -m backend.db.migrate`

### Phase 6 Progress
Completed:
- SHAP explainability — per-prediction top-5 feature attribution with 33-feature interpretation mapping
- Session-level scoring — drift detection, linearly-weighted batch aggregation, trend analysis
- scoring_type column in predictions table to distinguish batch vs session scores
- Live dashboard at GET /dashboard — dark-theme canvas chart, live feed, feature importance bars
- Local predictions log at backend/data/predictions_log.jsonl — always written; used as dashboard fallback when PostgreSQL is unavailable

### Monitoring
- CloudWatch namespace: `HumanGuard`
- Metrics emitted by `backend/monitoring.py` (module-level `metrics` singleton):
  - `score_requests` — count of scored batches
  - `bot_detections` / `human_detections` — per-label counts
  - `prediction_latency_ms` — end-to-end scoring time in milliseconds
  - `validation_errors` — count of rejected payloads (isValidSignalBatch failures)
  - `lambda_errors` — count of unhandled exceptions in /api/score
- Metrics are emitted only when `CLOUDWATCH_ENABLED=true` env var is set; no-op in local dev
- Alarms (created by `infrastructure/cloudwatch_alarms.py`, run once at deploy time):
  - `HumanGuard-BotRateSpike` — bot_detections/score_requests > 80% for 5 min
  - `HumanGuard-HighLatency` — p95 latency > 2000ms for 3 consecutive minutes
  - `HumanGuard-ValidationErrorRate` — validation_errors > 10 in 5 min
  - `HumanGuard-ErrorRate` — lambda_errors > 5 in 5 min
- All alarms publish to SNS topic `HumanGuard-Alerts`; subscribe via `SNS_ALERT_EMAIL` env var

### Deployment
- Dockerfile: python:3.11-slim, PORT=8080, `python -m backend.app`
- AWS deploy script: `scripts/aws_deploy.sh` (ECR + Lambda + API Gateway)
- Production deps: requirements-prod.txt (includes shap, psycopg2-binary)
