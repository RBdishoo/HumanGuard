# HumanGuard — Technical Progress Log

---

## System Status

| Component | Status | Notes |
|---|---|---|
| Signal Collection | ✅ Complete | JSONL + PostgreSQL dual-write, Lambda `/tmp` redirect |
| Feature Engineering | ✅ Complete | 30 features across 6 behavioral categories (pruned from 33) |
| ML Model | ✅ Complete | RandomForest champion (CV AUC 1.0000); session blender catches adaptive bots |
| API | ✅ Complete | 8 endpoints live on AWS Lambda behind API Gateway |
| Database | ✅ Complete | RDS PostgreSQL in production; SQLite auto-fallback for local dev |
| Monitoring | ✅ Complete | 5 CloudWatch metrics, 4 alarms, SNS email alerts active |
| Dashboard | ✅ Complete | S3-hosted, polls `/api/dashboard-stats`, live chart + SHAP bars |
| CloudFront | ✅ Complete | HTTPS distribution `E3F5RTWRNWWQB0`; HTTP→HTTPS redirect; tuned cache TTLs |
| Adversarial Robustness | ✅ Complete | 100% session-level detection across 5 hard bot patterns; temporal drift scoring |
| Tests | ✅ Complete | 87 tests across 11 files, all passing |

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

**Built:** `backend/features/feature_extractor.py` — `FeatureExtractor.extractBatchFeatures()` takes a raw signals dict and returns a 30-element ordered feature vector. Supporting math in `backend/features/feature_utils.py`.

**30 features across 6 categories (pruned from 33 after adversarial hardening):**

| Category | Features | Bot-detection signal |
|---|---|---|
| **Batch-level** | `batch_event_count`, `has_mouse_moves`, `has_clicks`, `has_keys` | Bots often send empty or single-event batches |
| **Mouse trajectory** | `mouseMoveCount`, `mouseAvgVelocity`, `mouseStdVelocity`, `mouseMaxVelocity`, `mouseAvgPauseDurationMs`, `mousePathEfficiency`, `mouseAngularVelocityStd`, `mouseHoverTimeRatio`, `mouseHoverFrequency` | Bots produce unnaturally linear paths (high `mousePathEfficiency`), zero variance velocity (`mouseStdVelocity ≈ 0`), and no hover events |
| **Click dynamics** | `clickIntervalMeanMs`, `clickIntervalStdMs`, `clickIntervalMinMs`, `clickIntervalMaxMs`, `clickRatePerSec` | Scripted clicks have near-zero `clickIntervalStdMs` and inhuman `clickRatePerSec` |
| **Keystroke dynamics** | `keyCount`, `keyInterKeyDelayMeanMs`, `keyInterKeyDelayStdMs`, `keyRapidPresses`, `keyEntropy`, `keyRatePerSec` | Bots type at fixed inter-key delays (`keyInterKeyDelayStdMs ≈ 0`) and use a narrow key distribution (low `keyEntropy`) |
| **Temporal composites** | `batchDurationMs`, `eventRatePerSec`, `clickToMoveRatio`, `keyToMoveRatio` | Session timing and cross-signal diversity detect bots that produce one signal type only |
| **Session consistency** | `keystroke_timing_regularity`, `typing_rhythm_autocorrelation`, `mouse_acceleration_variance`, `mouse_keystroke_correlation`, `session_phase_consistency` | Catch human-mimicking bots: uniform-speed bots score low CoV; Gaussian bots score near-zero autocorrelation; linear-path bots have near-zero acceleration variance |

**Key decisions:**
- Path efficiency (`mousePathEfficiency = straight-line distance / actual path length`) was the single most discriminating mouse feature — human paths are never fully straight
- 5 session-consistency features added to catch human_speed_typer and hybrid_bot patterns that evaded the original 25-feature model
- All 30 features extracted from a single batch pass — no cross-batch state required — keeping `/api/score` stateless and horizontally scalable

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

**Model comparison (held-out test set, post-adversarial hardening with 30 features):**

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **RandomForest** | 99.44% | **99.59%** | **1.0000** ← selected |
| LogisticRegression | 99.41% | 99.57% | 0.9977 |
| XGBoost | 99.94% | 99.96% | 1.0000 |

**Hard test results (adversarial bot patterns):**

| Evaluation | F1 | Detection Rate |
|---|---|---|
| Batch-level (per batch vs session label) | 0.9474 | 90.0% |
| Session-level (temporal blender) | **1.0000** | **100.0%** |

**Per-pattern session-level detection:**

| Pattern | Detection Rate |
|---|---|
| human_speed_typer | 100% |
| bezier_mouse | 100% |
| jitter_bot | 100% |
| hybrid_bot | 100% |
| adaptive_bot | 100% |

**Deployed model:** RandomForest champion selected by 5-fold CV AUC (1.0000). A LogisticRegression session blender (`session_blender.pkl`) runs on top for sessions with ≥10 batches, blending batch probability with temporal drift features to catch adaptive bots.

**Why temporal session blending:**
- Adaptive bots mimic human behavior for the first half of a session then revert to scripted patterns — batch-level scoring alone detects only 50% of their batches
- Session blender computes `temporal_drift_score`, `early_late_timing_delta`, `behavior_consistency_score` across the full session and blends with `avg_batch_prob` via LogisticRegression (C=0.5)
- Gated on ≥10 batches so all existing unit tests (3–6 batch sessions) are unaffected

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

### Phase 7 — Adversarial Hardening

**Problem:** A hard test suite of 5 adversarial bot patterns (500 batches across 50 sessions) revealed two critical weaknesses:
- `human_speed_typer` (0% detection) — Gaussian keystroke timing + linear mouse looked human to the original feature set
- `hybrid_bot` (32% detection) — Bézier mouse + uniform 80ms keystrokes partially evaded detection
- `adaptive_bot` (50% detection) — bot behaves human-like in batches 0–4, bot-like in batches 5–9; batch-level scoring sees 50% false negatives

**Fix 1 — Session-consistency features (5 new batch-level features):**

| Feature | What it measures | Signal |
|---|---|---|
| `keystroke_timing_regularity` | CoV (std/mean) of inter-key delays | Uniform-speed bots ≈ 0.0 |
| `typing_rhythm_autocorrelation` | Lag-1 autocorrelation of key delay sequence | Gaussian bots ≈ 0.0; humans > 0 |
| `mouse_acceleration_variance` | Variance of per-step mouse acceleration | Linear-path bots ≈ 0.0 |
| `mouse_keystroke_correlation` | Mouse velocity ratio during vs outside keystrokes | Bots ≈ 1.0 (no coordination) |
| `session_phase_consistency` | Velocity std between first/second batch half | Consistent bots ≈ 0.0 |

**Fix 2 — Temporal session blender (session-level meta-model):**

For sessions with ≥10 batches, a LogisticRegression blender operates on 4 session-level features:
- `avg_batch_prob` — linearly-weighted average of per-batch bot probabilities (later batches weighted higher)
- `temporal_drift` — normalised L2 distance between first-half and second-half feature vectors
- `early_late_delta_ms` — change in mean inter-key delay between first 30% and last 30% of batches
- `behavior_consistency` — cosine similarity of first-half vs second-half average feature vectors

Adaptive bot signature: moderate `avg_batch_prob` (≈0.45) + high `temporal_drift` + large `early_late_delta_ms` (≈95ms) + low `behavior_consistency`. The blender catches this pattern even when per-batch scores are ambiguous.

**Results before → after adversarial hardening:**

| Pattern | Batch Before | Batch After | Session After |
|---|---|---|---|
| human_speed_typer | 0% | 100% | 100% |
| hybrid_bot | 32% | 100% | 100% |
| adaptive_bot | 50% | 50% | **100%** |
| bezier_mouse | 100% | 100% | 100% |
| jitter_bot | 100% | 100% | 100% |
| **Overall** | **56.4%** | **90.0%** | **100.0%** |

Hard test session F1: **1.0000** — all 50 adversarial sessions correctly classified.

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
| **CloudFront distribution** | `https://d1hi33wespusty.cloudfront.net` (ID: `E3F5RTWRNWWQB0`) |
| **Demo** | `https://d1hi33wespusty.cloudfront.net/demo.html` |
| **Leaderboard** | `https://d1hi33wespusty.cloudfront.net/leaderboard.html` |
| **Dashboard** | `https://d1hi33wespusty.cloudfront.net/dashboard.html` |
| **S3 origin (HTTP)** | `http://humanguard-frontend-796793347388.s3-website-us-east-1.amazonaws.com` |
| **RDS endpoint** | `humanguard-db.c8p60woyyitr.us-east-1.rds.amazonaws.com:5432` |
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

**87 tests across 11 test files — all passing.**

| File | Tests | What it covers |
|---|---|---|
| `test_db.py` | 15 | `DatabaseManager` SQLite round-trips; `save_prediction`/`get_stats`/`get_recent_predictions` correctness; result ordering and limit enforcement; graceful handling of unreachable postgres URL; `db_client` availability checks; `SignalCollector` dual-write paths (DB available, unavailable, raises) |
| `test_session_score.py` | 8 | Session-level scoring aggregation; linear recency weighting; drift score computation; single-batch edge case; missing `sessionID` rejection; `/api/session-score` GET alias |
| `test_features.py` | 8 | `FeatureExtractor` output length (must be 30); zero-event edge cases for all signal types; mouse velocity and path-efficiency math; keystroke entropy; click clustering ratio |
| `test_dashboard.py` | 8 | `/api/dashboard-stats` PostgreSQL and JSONL fallback paths; required field presence; recent prediction ordering; empty-state zeros; model and threshold fields |
| `test_helpers.py` | 7 | `isValidSignalBatch` acceptance/rejection matrix; `normalizeSignalBatch` field aliasing (`sessionId`→`sessionID`, `mouseEvents`→`mouseMoves`, `keyEvents`→`keys`); `genSeshID` format validation |
| `test_api.py` | 7 | `/api/signals` valid and invalid payloads; `/api/score` with mocked bundle (prob_bot in [0,1], correct label, threshold field); oversized payload 413 response; CORS header presence |
| `test_shap.py` | 6 | SHAP response structure; top-5 feature list length; `contribution` numeric type; `_SHAP_PENDING` sentinel (no `KeyError` when explainer key absent); graceful `None` return when explainer unavailable; `?explain=false` suppresses explanation |
| `test_health.py` | 6 | `/health` 200 status; required fields (`status`, `model`, `version`, `uptime_seconds`, `timestamp`); uptime is float; no model artifact dependency |
| `test_signal_collector.py` | 5 | `saveSignalBatch` JSONL write; batch count increment; session count deduplication; Lambda `/tmp` path redirect; file-not-found graceful handling |
| `test_demo.py` | 6 | `source`/`label` fields accepted by `/api/signals` and `/api/score`; `/api/export` access control (missing key → 401, wrong key → 401, valid key → 200 CSV with correct columns) |
| `test_leaderboard.py` | 6 | `POST /api/leaderboard` nickname/session_id validation; 404 for unknown session; rank and percentile message on success; `GET /api/leaderboard` entries with rank and stats; nickname sanitization (special chars stripped, max 20 chars) |
| `test_model_registry.py` | 5 | `ModelRegistry.push()` returns semantic version; `load('latest')` resolves champion; `list_versions()` sorted ascending; `promote()` updates champion flag on old and new versions; `rollback()` reinstates previous champion |
