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

# Run tests
pytest tests/ -v
pytest tests/test_signals.py          # single test file

# Train model (extracts features from signals.jsonl, trains, saves artifacts)
python -m models.run_training

# Generate synthetic bot data
python scripts/seed_bot_session.py

# Validate trained model
python scripts/report_training_summary.py
```

No linter or formatter is currently configured.

Architecture
Data Flow
Frontend (frontend/tracker.js): Vanilla JS captures mouse/click/key events (mouse throttled to 100ms), auto-sends batches every 3 seconds to /api/signals

Signal Collection (backend/collectors/signal_collector.py): Appends signal batches as JSONL to backend/data/raw/signals.jsonl

Feature Engineering (backend/features/): Extracts 34 behavioral features (mouse velocity/acceleration/path efficiency, click rate/clustering, keystroke entropy/timing, temporal composites) into CSV

Training (models/): ModelDataset loads features + labels, splits/scales; ModelTrainer trains RandomForest/XGBoost; Evaluator generates reports

Inference (/api/score in backend/app.py): Lazy-loads trained model artifacts from models/trained/, extracts features from submitted signals, returns prob(bot) + label

Key Modules
backend/features/feature_extractor.py — FeatureExtractor class: raw signals → 34 features

backend/features/feature_utils.py — Static math utilities (MouseTrajectoryUtils, KeystrokeUtils)

backend/features/dataset_builder.py — Builds batch/session-level CSV datasets

backend/features/data_loader.py — Loads and validates signals from JSONL

backend/utils/helpers.py — isValidSignalBatch(), formatTimestamp()

models/dataset.py — ModelDataset: merges features with labels, train/test split, scaling

models/train.py — ModelTrainer: trains classifiers, saves .pkl artifacts

models/evaluate.py — Evaluator: metrics reports, confusion matrix plots

API Endpoints
POST /api/signals — Submit a signal batch

POST /api/score — Score a signal batch (returns bot probability). Caps input at 5000 mouse moves, 2000 clicks, 5000 keystrokes

GET /api/stats — Signal collection statistics

GET /health — Liveness check (status, model name, version, uptime, timestamp). Does not load model artifacts

GET / — Serves frontend/index.html

Trained Model Artifacts (models/trained/)
RandomForest.pkl — Trained classifier

scaler.pkl — StandardScaler for feature normalization

feature_names.json — Ordered feature list (must match extraction order)

threshold.json — Classification threshold (default 0.5)

metrics.json — Training metrics

⚠️ Never hand-edit these files. Always regenerate via python models/run_training.py.
When adding new features to FeatureExtractor, always re-run training to keep /api/score in sync.

Data Storage
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
- Client: `backend/db/db_client.py` — connection pool, is_available() guard
- To run migration: `python -m backend.db.migrate`
