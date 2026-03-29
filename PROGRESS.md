# HumanGuard — Technical Progress Log

---

## System Status

| Component | Status | Notes |
|---|---|---|
| Signal Collection | ✅ Complete | JSONL + PostgreSQL dual-write, Lambda `/tmp` redirect |
| Feature Engineering | ✅ Complete | 33 features across 6 behavioral categories |
| ML Model | ✅ Complete | XGBoost deployed; LogisticRegression holds highest ROC-AUC on current dataset |
| API | ✅ Complete | 8 endpoints live on AWS Lambda behind API Gateway |
| Database | ✅ Complete | RDS PostgreSQL in production; SQLite auto-fallback for local dev |
| Monitoring | ✅ Complete | 5 CloudWatch metrics, 4 alarms, SNS email alerts active |
| Dashboard | ✅ Complete | S3-hosted, polls `/api/dashboard-stats`, live chart + SHAP bars |
| Tests | ✅ Complete | 70 tests across 9 files, all passing |

---

## Build Log

### Phase 1 — Signal Collection

**Built:**
- `frontend/tracker.js` — vanilla JS event listener that captures `mousemove` (throttled to 100 ms), `click`, `keydown`, and `scroll` events into in-memory arrays, then auto-POSTs batches to `/api/signals` every 3 seconds
- `backend/collectors/signal_collector.py` — `SignalCollector` class appends each batch as a newline-delimited JSON record to `signals.jsonl`; dual-writes to PostgreSQL when `DATABASE_URL` is set

**Key decisions:**
- JSONL over a database for raw signal storage: append-only, human-readable, zero-schema overhead during early iteration; PostgreSQL layer added later without disrupting the JSONL path
- 100 ms mouse throttle: balances signal fidelity against payload size; sub-100 ms events provide diminishing bot-detection value while significantly increasing batch sizes
- 3-second auto-batch interval: keeps browser network overhead negligible while providing enough density per batch for statistically meaningful feature extraction
- `sessionID` generated client-side with `session_{timestamp}_{9-char-random}` — avoids a server round-trip for session init

---

### Phase 2 — Feature Engineering

**Built:** `backend/features/feature_extractor.py` — `FeatureExtractor.extractBatchFeatures()` takes a raw signals dict and returns a 33-element ordered feature vector. Supporting math in `backend/features/feature_utils.py`.

**33 features across 6 categories:**

| Category | Features | Bot-detection signal |
|---|---|---|
| **Batch-level** | `batch_event_count`, `has_mouse_moves`, `has_clicks`, `has_keys` | Bots often send empty or single-event batches |
| **Mouse trajectory** | `mouseMoveCount`, `mouseAvgVelocity`, `mouseStdVelocity`, `mouseMaxVelocity`, `mousePauseCount`, `mouseAvgPauseDurationMs`, `mousePathEfficiency`, `mouseAngularVelocityStd`, `mouseHoverTimeRatio`, `mouseHoverFrequency` | Bots produce unnaturally linear paths (high `mousePathEfficiency`), zero variance velocity (`mouseStdVelocity ≈ 0`), and no hover events |
| **Click dynamics** | `clickCount`, `clickIntervalMeanMs`, `clickIntervalStdMs`, `clickIntervalMinMs`, `clickIntervalMaxMs`, `clickClusteringRatio`, `clickRatePerSec`, `clickLeftRatio` | Scripted clicks have near-zero `clickIntervalStdMs` and inhuman `clickRatePerSec` |
| **Keystroke dynamics** | `keyCount`, `keyInterKeyDelayMeanMs`, `keyInterKeyDelayStdMs`, `keyRapidPresses`, `keyEntropy`, `keyRatePerSec` | Bots type at fixed inter-key delays (`keyInterKeyDelayStdMs ≈ 0`) and use a narrow key distribution (low `keyEntropy`) |
| **Temporal composites** | `batchDurationMs`, `eventRatePerSec`, `signalDiversityEntropy` | Session timing and cross-signal diversity detect bots that produce one signal type only |
| **Ratios** | `clickToMoveRatio`, `keyToMoveRatio` | Humans exhibit predictable cross-signal ratios; bots that simulate only clicks or only keys create extreme ratio values |

**Key decisions:**
- Path efficiency (`mousePathEfficiency = straight-line distance / actual path length`) was the single most discriminating mouse feature — human paths are never fully straight
- `signalDiversityEntropy` (Shannon entropy over signal type distribution) catches bots that emit only mouse events with zero keystrokes, or vice versa
- All 33 features extracted from a single batch pass — no cross-batch state required — keeping `/api/score` stateless and horizontally scalable

---

### Phase 3 — Data Collection & Labeling

**Dataset:**
- 41 labeled sessions total: **11 bot sessions**, **30 human sessions**
- Bot sessions generated via `scripts/seed_bot_session.py` — two modes: standard (fixed-velocity linear mouse, regular click cadence) and `--stealthy` (adds Gaussian jitter to mimic human noise)
- Human sessions generated via `scripts/seed_human_session.py` — randomized velocities, natural pause distributions, variable keystroke timing drawn from a log-normal distribution
- Labels stored in `backend/data/raw/labels.csv` as `sessionID,label`

**Key decisions:**
- Synthetic data ensures ground-truth labels without manual annotation; the feature pipeline is identical for real browser data
- Stealthy bot mode was included to harden the model against evasion — bots that add noise remain distinguishable via path efficiency and inter-key delay distribution shape, since positional jitter does not restore the entropy of a naturally curved human path

---

### Phase 4 — ML Model Training

**Pipeline:** `models/dataset.py` loads `backend/data/processed/training_data_batches.csv`, merges with `labels.csv`, applies `StandardScaler`, and performs an 80/20 stratified train/test split. `models/train.py` trains three classifiers and selects the best by ROC-AUC.

**Model comparison (held-out test set):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| RandomForest | 90.91% | 91.67% | 91.67% | 91.67% | 90.31% |
| **LogisticRegression** | 86.36% | 82.14% | 95.83% | 88.46% | **90.73%** ← selected |
| XGBoost | 88.64% | 88.00% | 91.67% | **89.80%** | 88.65% |

**Deployed model:** XGBoost (`MODEL_NAME = "XGBoost"` in `app.py`), despite LogisticRegression holding the highest ROC-AUC.

**Why XGBoost is deployed over the AUC winner:**
- `shap.TreeExplainer` requires a tree-based model — SHAP explainability is incompatible with LogisticRegression at this scale
- XGBoost has the highest F1 (89.80%) of the three, which matters more than AUC for per-prediction labeling decisions
- XGBoost's 88.65% AUC is within 1% of LogisticRegression — not a meaningful gap on a 41-session dataset

**Key decisions:**
- ROC-AUC chosen as the automated selection metric over accuracy because the dataset is class-imbalanced (11 bot : 30 human)
- `StandardScaler` fit on training set only; scaler state serialized to `models/trained/scaler.pkl` to prevent data leakage at inference time
- `feature_names.json` serialized alongside model artifacts to enforce feature order consistency between training and serving — mismatched order would silently corrupt predictions

---

### Phase 5 — Deployment & Infrastructure

**AWS stack:**

| Resource | Configuration |
|---|---|
| ECR | Private registry; image built `--platform linux/amd64` for Lambda x86_64 |
| Lambda | Container image, 1024 MB, 60 s timeout, `awslambdaric` entrypoint; custom WSGI bridge handles API Gateway v2 HTTP event format |
| API Gateway | HTTP API; `ANY /{proxy+}` → Lambda; 30 s integration timeout |
| RDS | PostgreSQL 15, `db.t3.micro`, `humanguard-db`; `BackupRetentionPeriod=0` (free tier) |
| Secrets Manager | `humanGuard/rds` — `{host, port, dbname, username, password}`; fetched by `aws_deploy.sh` to build `DATABASE_URL` |
| IAM | `humanguard-lambda-role`: `AWSLambdaBasicExecutionRole` + `CloudWatchFullAccess` |

**Deploy issues encountered and resolved:**

| # | Issue | Root Cause | Fix |
|---|---|---|---|
| 1 | `boto3` missing in container | `requirements-prod.txt` omitted `boto3`; CloudWatch metrics silently no-oped in production | Added `boto3` to `requirements-prod.txt` |
| 2 | FK constraint on `predictions` insert | `db_client.save_prediction()` inserted into `predictions` without first ensuring a `sessions` row existed | Added `INSERT INTO sessions … ON CONFLICT DO NOTHING` before each prediction insert |
| 3 | `shap.TreeExplainer` init blowing Lambda's 10 s module init limit | `TreeExplainer` takes ~20 s to initialize; was called inside `_load_scoring_bundle()` which runs at module load time | Made SHAP lazy via `_get_shap_explainer(bundle)` with a `_SHAP_PENDING` sentinel; explainer created on first explain call during the invoke phase, not init |
| 4 | RDS `CreateDBInstance` failing | `BackupRetentionPeriod=7` exceeds the free-tier maximum of 0 | Set `BackupRetentionPeriod=0` in `infrastructure/rds_setup.py` |
| 5 | CloudWatch `PutMetricData` AccessDenied | Lambda execution role only had `AWSLambdaBasicExecutionRole` — no CloudWatch write permissions | Attached `CloudWatchFullAccess` to `humanguard-lambda-role` |

**Earlier Lambda constraints resolved:**

| Issue | Fix |
|---|---|
| `OSError: Read-only file system` on `signals.jsonl` | `IS_LAMBDA` detection → redirect file writes to `/tmp` |
| `flask debug=True` crashes on missing `/dev/shm` | `debug=not IS_LAMBDA` |
| `mangum` incompatible with Flask WSGI | Dropped `mangum` (ASGI-only); wrote minimal custom WSGI bridge |
| joblib hanging on Lambda (no `/dev/shm` for multiprocessing) | `parallel_backend('sequential')` wrapper around `joblib.load()` |
| numba JIT stalling at first SHAP call | `NUMBA_DISABLE_JIT=1` env var on Lambda |
| API Gateway `ANY /` not matching `/api/score` | Added explicit `ANY /{proxy+}` route to cover all sub-paths |

---

### Phase 6 — Advanced Features

**SHAP Explainability**

Every `/api/score` and `/api/session-score` response includes:
- `top_features` — top-5 feature attributions sorted by `|contribution|`
- `interpretation` — human-readable sentence mapping the top feature to a behavioral description from a 33-entry `FEATURE_INTERPRETATIONS` dict in `app.py`

Implementation: `shap.TreeExplainer(model).shap_values(x_scaled)` on the scaled feature vector. SHAP values represent the additive contribution of each feature to the model output (log-odds); a large negative `batchDurationMs` contribution means that feature is strongly pushing the prediction toward "human".

**Session-Layer Scoring (`/api/session-score`)**

Aggregates all stored batches for a `sessionID`, scores each independently, then applies linear recency weighting (`weight_i = i / sum(1..n)` — later batches weighted higher). Returns:
- `session_prob_bot` — weighted aggregate probability
- `per_batch_scores` — individual batch probabilities in chronological order
- `drift` — `{ trend, drift_score, max_prob_bot, mean_prob_bot }`

The drift score (`max_prob_bot − mean_prob_bot`) detects sessions where bot probability increases over time — a pattern consistent with bots that open with human-like behavior and degrade as scripted actions take over.

**Adversarial Robustness**

`--stealthy` bot mode adds Gaussian noise to mouse positions, randomizes click intervals, and varies keystroke timing. The model continues to classify these correctly because positional noise does not restore the entropy of a human's naturally curved mouse path — `mousePathEfficiency` and inter-key delay distribution shape remain discriminating even with added noise.

---

## Key Technical Decisions

| Decision | Choice Made | Rejected Alternatives | Reasoning |
|---|---|---|---|
| **Raw signal storage** | JSONL flat file + PostgreSQL dual-write | PostgreSQL-only from start | JSONL is zero-dependency, append-only, and trivially portable for training export; PostgreSQL added non-destructively without a migration of existing storage logic |
| **ML framework** | XGBoost (deployed) + SHAP | Neural network, SVM, LightGBM | Tree-based gradient boosting handles tabular behavioral features well on small datasets; SHAP `TreeExplainer` requires a tree model — rules out SVM or neural networks for production explainability |
| **Deployment runtime** | AWS Lambda (container image) | EC2, ECS Fargate | Zero idle cost; container image removes Lambda layer size constraints for numpy/scikit-learn/XGBoost/SHAP; `awslambdaric` enables a standard WSGI handler |
| **API framework** | Flask | FastAPI, Django | Minimal WSGI surface; no async complexity; compatible with the custom WSGI bridge needed for Lambda's API Gateway v2 event format |
| **SHAP init timing** | Lazy (`_SHAP_PENDING` sentinel) | Eager at model load | `TreeExplainer` takes ~20 s to initialize; Lambda's module init phase has a 10 s limit; lazy init moves the cost to the invoke phase where the 60 s function timeout applies |
| **Schema design** | FK: `predictions → sessions` | Denormalized predictions table | Enforces session integrity; enables efficient join-free queries for dashboard; the FK constraint required an explicit session upsert guard before each prediction insert |
| **Test database** | SQLite in-memory | Mocked DB calls, real PostgreSQL in CI | In-memory SQLite exercises real SQL paths (insert, query, ordering) without requiring a running PostgreSQL server; `DatabaseManager` backend selection is transparent to test code |

---

## Infrastructure

| Resource | Value |
|---|---|
| **API (Lambda + API Gateway)** | `https://9ixzk5e9u4.execute-api.us-east-1.amazonaws.com` |
| **RDS endpoint** | `humanguard-db.c8p60woyyitr.us-east-1.rds.amazonaws.com:5432` |
| **S3 dashboard** | `http://humanguard-dashboard.s3-website-us-east-1.amazonaws.com` |
| **SNS topic ARN** | `arn:aws:sns:us-east-1:796793347388:HumanGuard-Alerts` |
| **Secrets Manager** | `humanGuard/rds` (us-east-1) |
| **ECR repository** | `796793347388.dkr.ecr.us-east-1.amazonaws.com/humanguard` |

**CloudWatch Alarms (namespace: `HumanGuard`):**

| Alarm | Condition | Window |
|---|---|---|
| `HumanGuard-BotRateSpike` | `bot_detections / score_requests > 0.80` | 1 × 5 min |
| `HumanGuard-HighLatency` | p95 `prediction_latency_ms > 2000 ms` | 3 consecutive × 1 min |
| `HumanGuard-ValidationErrorRate` | `validation_errors > 10` | 1 × 5 min |
| `HumanGuard-ErrorRate` | `lambda_errors > 5` | 1 × 5 min |

All alarms: `TreatMissingData=notBreaching`; `AlarmActions` and `OKActions` publish to `HumanGuard-Alerts`.

---

## Test Coverage

**70 tests across 9 test files — all passing.**

| File | Tests | What it covers |
|---|---|---|
| `test_db.py` | 15 | `DatabaseManager` SQLite round-trips; `save_prediction`/`get_stats`/`get_recent_predictions` correctness; result ordering and limit enforcement; graceful handling of unreachable postgres URL; `db_client` availability checks; `SignalCollector` dual-write paths (DB available, unavailable, raises) |
| `test_session_score.py` | 8 | Session-level scoring aggregation; linear recency weighting; drift score computation; single-batch edge case; missing `sessionID` rejection; `/api/session-score` GET alias |
| `test_features.py` | 8 | `FeatureExtractor` output length (must be 33); zero-event edge cases for all signal types; mouse velocity and path-efficiency math; keystroke entropy; click clustering ratio |
| `test_dashboard.py` | 8 | `/api/dashboard-stats` PostgreSQL and JSONL fallback paths; required field presence; recent prediction ordering; empty-state zeros; model and threshold fields |
| `test_helpers.py` | 7 | `isValidSignalBatch` acceptance/rejection matrix; `normalizeSignalBatch` field aliasing (`sessionId`→`sessionID`, `mouseEvents`→`mouseMoves`, `keyEvents`→`keys`); `genSeshID` format validation |
| `test_api.py` | 7 | `/api/signals` valid and invalid payloads; `/api/score` with mocked bundle (prob_bot in [0,1], correct label, threshold field); oversized payload 413 response; CORS header presence |
| `test_shap.py` | 6 | SHAP response structure; top-5 feature list length; `contribution` numeric type; `_SHAP_PENDING` sentinel (no `KeyError` when explainer key absent); graceful `None` return when explainer unavailable; `?explain=false` suppresses explanation |
| `test_health.py` | 6 | `/health` 200 status; required fields (`status`, `model`, `version`, `uptime_seconds`, `timestamp`); uptime is float; no model artifact dependency |
| `test_signal_collector.py` | 5 | `saveSignalBatch` JSONL write; batch count increment; session count deduplication; Lambda `/tmp` path redirect; file-not-found graceful handling |
