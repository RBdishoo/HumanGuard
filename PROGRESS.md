# HumanGuard — Technical Progress Log

---

## System Status

| Component | Status | Notes |
|---|---|---|
| Signal Collection | ✅ Complete | JSONL + PostgreSQL dual-write, Lambda `/tmp` redirect |
| Feature Engineering | ✅ Complete | 37 features across 7 categories (30 behavioral + 7 network/device) |
| ML Model | ✅ Complete | RandomForest champion (CV AUC 1.0000); session blender catches adaptive bots |
| API | ✅ Complete | 12 endpoints live on AWS Lambda behind API Gateway |
| Database | ✅ Complete | RDS PostgreSQL in production; SQLite auto-fallback for local dev |
| Monitoring | ✅ Complete | 5 CloudWatch metrics, 4 alarms, SNS email alerts active |
| Dashboard | ✅ Complete | S3-hosted, polls `/api/dashboard-stats`, live chart + SHAP bars |
| CloudFront | ✅ Complete | HTTPS distribution; custom domain `humanguard.net`; HTTP→HTTPS redirect; tuned cache TTLs |
| Adversarial Robustness | ✅ Complete | 100% session-level detection across 5 hard bot patterns; temporal drift scoring |
| Security Hardening | ✅ Complete | Hashed key storage, CORS allowlist, EXPORT key in Secrets Manager, IP rate limiting, atomic quota enforcement |
| Auto-Retrain Pipeline | ✅ Complete | EventBridge 6h trigger; retrain threshold 50 real sessions; S3 model registry with versioning + champion promotion |
| Network/Device Enrichment | ✅ Complete | 7 IP/UA features via ipinfo.io + UA parsing; 1h in-memory cache |
| Webhooks | ✅ Complete | HMAC-SHA256 signed delivery; 3 event types; auto-disable after 5 failures |
| Email Verification | ✅ Complete | AWS SES; 24h token; 10-request trial before verification required |
| Leaderboard | ✅ Complete | Public demo leaderboard with rank, percentile, social sharing |
| Real-Data Tracking | ✅ Complete | `/api/data-stats` counts real human sessions from DB; retrain readiness at 22/50 |
| Tests | ✅ Complete | 158 tests across 15 files, all passing |

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

**30 behavioral features across 6 categories (pruned from 33 after adversarial hardening):**

| Category | Features | Bot-detection signal |
|---|---|---|
| **Batch-level** | `batch_event_count`, `has_mouse_moves`, `has_clicks`, `has_keys` | Bots often send empty or single-event batches |
| **Mouse trajectory** | `mouseMoveCount`, `mouseAvgVelocity`, `mouseStdVelocity`, `mouseMaxVelocity`, `mouseAvgPauseDurationMs`, `mousePathEfficiency`, `mouseAngularVelocityStd`, `mouseHoverTimeRatio`, `mouseHoverFrequency` | Bots produce unnaturally linear paths, zero variance velocity, and no hover events |
| **Click dynamics** | `clickIntervalMeanMs`, `clickIntervalStdMs`, `clickIntervalMinMs`, `clickIntervalMaxMs`, `clickRatePerSec` | Scripted clicks have near-zero `clickIntervalStdMs` and inhuman `clickRatePerSec` |
| **Keystroke dynamics** | `keyCount`, `keyInterKeyDelayMeanMs`, `keyInterKeyDelayStdMs`, `keyRapidPresses`, `keyEntropy`, `keyRatePerSec` | Bots type at fixed inter-key delays and use a narrow key distribution |
| **Temporal composites** | `batchDurationMs`, `eventRatePerSec`, `clickToMoveRatio`, `keyToMoveRatio` | Cross-signal diversity detects bots that produce one signal type only |
| **Session consistency** | `keystroke_timing_regularity`, `typing_rhythm_autocorrelation`, `mouse_acceleration_variance`, `mouse_keystroke_correlation`, `session_phase_consistency` | Catch human-mimicking bots: uniform-speed bots score low CoV; Gaussian bots score near-zero autocorrelation |

**Key decisions:**
- Path efficiency (`mousePathEfficiency = straight-line distance / actual path length`) was the single most discriminating mouse feature — human paths are never fully straight
- 5 session-consistency features added to catch `human_speed_typer` and `hybrid_bot` patterns that evaded the original 25-feature model
- All 30 features extracted from a single batch pass — no cross-batch state required — keeping `/api/score` stateless and horizontally scalable

---

### Phase 3 — Data Collection & Labeling

**Dataset:**
- 330 labeled sessions: **230 bot sessions**, **100 human sessions** (3,300 batches)
- Bot sessions generated via `scripts/seed_bot_session.py` — two modes: standard (fixed-velocity linear mouse, regular click cadence) and `--stealthy` (adds Gaussian jitter to mimic human noise)
- Human sessions generated via `scripts/seed_human_session.py` — randomized velocities, natural pause distributions, variable keystroke timing drawn from a log-normal distribution
- Labels stored in `backend/data/raw/labels.csv` as `sessionID,label`
- Dataset regenerated under Docker + sklearn 1.8.0 after a version mismatch caused Lambda `ValueError` on predict

**Key decisions:**
- Synthetic data ensures ground-truth labels without manual annotation; the feature pipeline is identical for real browser data
- Stealthy bot mode included to harden the model against evasion — positional jitter does not restore the entropy of a naturally curved human path

---

### Phase 4 — ML Model Training

**Pipeline:** `models/dataset.py` loads `backend/data/processed/training_data_batches.csv`, merges with `labels.csv`, applies `StandardScaler`, and performs an 80/20 stratified train/test split. `models/train.py` trains three classifiers and selects the best by ROC-AUC.

**Model comparison (held-out test set, post-adversarial hardening with 37 features):**

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

**Key decisions:**
- ROC-AUC chosen as the automated selection metric over accuracy because the dataset is class-imbalanced (230 bot : 100 human)
- `StandardScaler` fit on training set only; scaler state serialized to `models/trained/scaler.pkl` to prevent data leakage at inference time
- `feature_names.json` serialized alongside model artifacts to enforce feature order consistency between training and serving

---

### Phase 5 — Deployment & Infrastructure

**AWS stack:**

| Resource | Configuration |
|---|---|
| ECR | Private registry; image built `--platform linux/amd64` for Lambda x86_64 |
| Lambda | Container image, 1024 MB, 60 s timeout, `awslambdaric` entrypoint; custom WSGI bridge handles API Gateway v2 HTTP event format |
| API Gateway | HTTP API; `ANY /{proxy+}` → Lambda; 30 s integration timeout |
| RDS | PostgreSQL 15, `db.t3.micro`, `humanguard-db`; `BackupRetentionPeriod=0` (free tier) |
| Secrets Manager | `humanGuard/rds` — `{host, port, dbname, username, password}`; Lambda receives only `RDS_SECRET_NAME=humanGuard/rds`; `db_client.py` fetches and caches credentials at cold-start |
| IAM | `humanguard-lambda-role`: `AWSLambdaBasicExecutionRole` + `CloudWatchFullAccess` + `ses:SendEmail` |

**Deploy issues encountered and resolved:**

| # | Issue | Root Cause | Fix |
|---|---|---|---|
| 1 | `boto3` missing in container | `requirements-prod.txt` omitted `boto3` | Added `boto3` to `requirements-prod.txt` |
| 2 | FK constraint on `predictions` insert | `save_prediction()` inserted without ensuring a `sessions` row existed | Added `INSERT INTO sessions … ON CONFLICT DO NOTHING` before each prediction insert |
| 3 | `shap.TreeExplainer` init blowing Lambda's 10 s module init limit | `TreeExplainer` takes ~20 s to initialize; called inside `_load_scoring_bundle()` at module load time | Made SHAP lazy via `_get_shap_explainer(bundle)` with a `_SHAP_PENDING` sentinel |
| 4 | RDS `CreateDBInstance` failing | `BackupRetentionPeriod=7` exceeds the free-tier maximum of 0 | Set `BackupRetentionPeriod=0` in `infrastructure/rds_setup.py` |
| 5 | CloudWatch `PutMetricData` AccessDenied | Lambda execution role only had `AWSLambdaBasicExecutionRole` | Attached `CloudWatchFullAccess` to `humanguard-lambda-role` |
| 6 | sklearn `ValueError` on Lambda predict | `.pkl` files generated under sklearn 1.7.0; Lambda runs 1.8.0 (`missing_go_to_left` field added) | Regenerated all training data and retrained inside Docker with `requirements-prod.txt` |
| 7 | Leaderboard always 404 on Lambda | `/api/score` was conditionally calling `db_client.save_prediction()` instead of `db_manager.save_prediction()`; JSONL is per-container on Lambda | Fixed to unconditional `db_manager.save_prediction()` in both batch and session paths |
| 8 | `source` null for all predictions | `_session_score_logic` called `save_prediction()` without passing the `source` kwarg | Added `source=_pred_source` to the session-score call site |

**Earlier Lambda constraints resolved:**

| Issue | Fix |
|---|---|
| `OSError: Read-only file system` on `signals.jsonl` | `IS_LAMBDA` detection → redirect file writes to `/tmp` |
| `flask debug=True` crashes on missing `/dev/shm` | `debug=not IS_LAMBDA` |
| `mangum` incompatible with Flask WSGI | Dropped `mangum`; wrote minimal custom WSGI bridge |
| joblib hanging on Lambda | `parallel_backend('sequential')` wrapper around `joblib.load()` |
| numba JIT stalling at first SHAP call | `NUMBA_DISABLE_JIT=1` env var on Lambda |
| API Gateway `ANY /` not matching `/api/score` | Added explicit `ANY /{proxy+}` route |

---

### Phase 6 — Advanced Features

**SHAP Explainability**

Every `/api/score` and `/api/session-score` response includes:
- `top_features` — top-5 feature attributions sorted by `|contribution|`
- `interpretation` — human-readable sentence mapping the top feature to a behavioral description from a 37-entry `FEATURE_INTERPRETATIONS` dict in `app.py`

**Session-Layer Scoring (`/api/session-score`)**

Aggregates all stored batches for a `sessionID`, scores each independently, then applies linear recency weighting. Returns `session_prob_bot`, `per_batch_scores`, and `drift` including trend, drift score, max, and mean.

**Confidence Intervals**

Every `/api/score` response includes 95% CI computed from per-tree prediction variance on the RandomForest. Falls back to point estimate for non-ensemble models. Fields: `confidence` (high/medium/low), `confidence_interval.lower`, `confidence_interval.upper`, `std`.

**Adversarial Robustness**

`--stealthy` bot mode adds Gaussian noise to mouse positions, randomizes click intervals, and varies keystroke timing. The model continues to classify these correctly because positional noise does not restore the entropy of a naturally curved human path.

---

### Phase 7 — Adversarial Hardening

**Problem:** A hard test suite of 5 adversarial bot patterns (500 batches across 50 sessions) revealed two critical weaknesses:
- `human_speed_typer` (0% detection) — Gaussian keystroke timing + linear mouse looked human to the original feature set
- `hybrid_bot` (32% detection) — Bézier mouse + uniform 80ms keystrokes partially evaded detection
- `adaptive_bot` (50% detection) — bot behaves human-like in batches 0–4, bot-like in batches 5–9

**Fix 1 — Session-consistency features (5 new batch-level features):**

| Feature | What it measures | Signal |
|---|---|---|
| `keystroke_timing_regularity` | CoV (std/mean) of inter-key delays | Uniform-speed bots ≈ 0.0 |
| `typing_rhythm_autocorrelation` | Lag-1 autocorrelation of key delay sequence | Gaussian bots ≈ 0.0; humans > 0 |
| `mouse_acceleration_variance` | Variance of per-step mouse acceleration | Linear-path bots ≈ 0.0 |
| `mouse_keystroke_correlation` | Mouse velocity ratio during vs outside keystrokes | Bots ≈ 1.0 (no coordination) |
| `session_phase_consistency` | Velocity std between first/second batch half | Consistent bots ≈ 0.0 |

**Fix 2 — Temporal session blender (session-level meta-model):**

For sessions with ≥10 batches, a LogisticRegression blender operates on 4 session-level features: `avg_batch_prob`, `temporal_drift`, `early_late_delta_ms`, `behavior_consistency`.

**Results before → after adversarial hardening:**

| Pattern | Batch Before | Batch After | Session After |
|---|---|---|---|
| human_speed_typer | 0% | 100% | 100% |
| hybrid_bot | 32% | 100% | 100% |
| adaptive_bot | 50% | 50% | **100%** |
| bezier_mouse | 100% | 100% | 100% |
| jitter_bot | 100% | 100% | 100% |
| **Overall** | **56.4%** | **90.0%** | **100.0%** |

---

### Phase 8 — Security Hardening

**Built:**
- **Hashed API key storage** — new key format `hg_live_<8-char-id>.<32-char-secret>`; id stored in `key_id` column, secret hashed with SHA-256 and stored in `key_hash`; plaintext never persisted; `validate_api_key()` uses `hmac.compare_digest` for timing-safe comparison
- **CORS allowlist** — replaced `origins: "*"` with env-var-driven `ALLOWED_ORIGINS`; defaults to `humanguard.net`, CloudFront, S3 static, and localhost; production Lambda excludes localhost
- **IP rate limiting on `/api/register`** — max 3 registrations per IP per hour; in-memory, thread-safe via `threading.Lock()`
- **Atomic quota enforcement** — `atomic_increment_usage()` uses `UPDATE … WHERE count < limit RETURNING` to eliminate TOCTOU race condition on monthly quota check
- **Sanitized error responses** — all `except` blocks return generic `"internal server error"`; full exception logged server-side only
- **EXPORT_API_KEY from Secrets Manager** — no `devkey` default; returns 503 when unset

---

### Phase 9 — Network/Device Enrichment + Auto-Retrain Pipeline

**Network/Device Features (7 new features → 37 total):**

`backend/enrichment.py` — `enrich_request()` computes:

| Feature | Source | Bot signal |
|---|---|---|
| `is_headless_browser` | UA string parsing | Headless Chrome/Playwright signatures |
| `is_known_bot_ua` | UA string matching | Googlebot, curl, Python-requests, etc. |
| `is_datacenter_ip` | ipinfo.io `org` field | AWS/GCP/Azure ASN prefixes |
| `ua_entropy` | Shannon entropy of UA string | Template-generated UAs have low entropy |
| `has_accept_language` | Request headers | Bots often omit Accept-Language |
| `accept_language_count` | Accept-Language parsing | Abnormal language count |
| `suspicious_header_count` | Presence of 5 standard browser headers | Missing headers suggest automation |

ipinfo.io responses cached in-memory with 1-hour TTL per IP (Lambda-instance-local; resets on cold start).

**Auto-Retrain Pipeline:**
- `scripts/retrain.py --auto` checks `retrain_readiness` from `/api/data-stats`; triggers when real human session count ≥ threshold (default 50)
- EventBridge rule fires `retrain.py --auto --push` every 6 hours
- `ModelRegistry` (S3) stores versioned artifacts with semantic versioning and champion promotion; `rollback()` available
- `retrain.py` wraps S3 registry push in `try/except` so `retrain_log.jsonl` always saves even if S3 is unreachable
- `/api/retrain-status` and `/api/model-info` endpoints expose current model version and last retrain timestamp
- End-to-end pipeline verified: 55 sessions triggered threshold, model v1.0.0 trained (AUC 1.0000), pushed to S3, promoted champion

---

### Phase 10 — Webhooks, Email Verification & Production Fixes

**Webhook System:**
- `POST /api/webhooks` — register a webhook URL with event subscriptions (`bot_detected`, `score_completed`, `high_confidence_bot`)
- Async daemon thread delivery after every `/api/score` response; never blocks scoring latency
- HMAC-SHA256 payload signing via `X-HumanGuard-Signature: sha256=<hex>` header
- Auto-disables webhook after 5 consecutive delivery failures; `db_manager.update_webhook_status()` tracks failure count
- `POST /api/webhooks/<id>/test` — sends a test payload to verify endpoint reachability

**Email Verification:**
- `POST /api/register` sends verification email via AWS SES; 24-hour token
- Unverified keys allowed 10 scored requests (trial period) before `require_api_key` blocks with a `403` directing to `verify.html`
- `GET /api/verify?token=` validates token, marks key `verified=true` in DB
- Falls back to logging the verify link when `SENDER_EMAIL` env var is unset (local dev)

**Production Fixes:**
- **`/api/data-stats` reads from DB not CSV** — `labels.csv` doesn't exist on Lambda; `DatabaseManager.get_data_stats()` queries predictions table grouped by `source`; CSV/JSONL fallback retained for local dev
- **`source` always null** — `_session_score_logic` was calling `save_prediction()` without the `source` kwarg; fixed at call site; `source` now correctly written for all prediction rows
- **Leaderboard 404 on Lambda** — leaderboard POST was looking up predictions from per-container JSONL; fixed to query DB first (cross-container persistent)
- **`@require_api_key` on public leaderboard endpoints** — removed from `POST/GET /api/leaderboard`; demo frontend has no API key
- **UTM source tagging** — `tracker.js` reads `utm_source` query param and passes it through signal batches; `real_human_by_source` breakdown in `/api/data-stats`

---

## Key Technical Decisions

| Decision | Choice Made | Rejected Alternatives | Reasoning |
|---|---|---|---|
| **Raw signal storage** | JSONL flat file + PostgreSQL dual-write | PostgreSQL-only from start | JSONL is zero-dependency, append-only, and trivially portable for training export |
| **ML framework** | RandomForest (deployed) + SHAP | Neural network, SVM, LightGBM | Tree-based models handle tabular behavioral features well on small datasets; SHAP `TreeExplainer` requires a tree model |
| **Deployment runtime** | AWS Lambda (container image) | EC2, ECS Fargate | Zero idle cost; container image removes Lambda layer size constraints for numpy/scikit-learn/SHAP |
| **API framework** | Flask | FastAPI, Django | Minimal WSGI surface; compatible with the custom WSGI bridge needed for Lambda's API Gateway v2 event format |
| **SHAP init timing** | Lazy (`_SHAP_PENDING` sentinel) | Eager at model load | `TreeExplainer` takes ~20 s to initialize; Lambda's module init phase has a 10 s limit |
| **Schema design** | FK: `predictions → sessions` | Denormalized predictions table | Enforces session integrity; enables efficient join-free queries for dashboard |
| **Test database** | SQLite in-memory | Mocked DB calls, real PostgreSQL in CI | In-memory SQLite exercises real SQL paths without requiring a running PostgreSQL server |
| **API key storage** | id+hash pattern (SHA-256) | Plaintext storage, bcrypt | Timing-safe comparison via `hmac.compare_digest`; SHA-256 is sufficient for high-entropy random secrets |
| **Retrain trigger** | DB session count ≥ threshold | Time-based, manual-only | Ensures model only retrains when meaningful new behavioral data exists |

---

## Infrastructure

| Resource | Value |
|---|---|
| **API (Lambda + API Gateway)** | `https://humanguard.net` (custom domain; raw API Gateway URL omitted) |
| **CloudFront distribution** | `https://humanguard.net` (distribution ID omitted) |
| **Demo** | `https://humanguard.net/demo` |
| **Leaderboard** | `https://humanguard.net/leaderboard` |
| **Dashboard** | `https://humanguard.net/dashboard` |
| **S3 origin (HTTP)** | private; fronted by CloudFront only |
| **RDS endpoint** | private subnet; endpoint omitted |
| **SNS topic ARN** | `arn:aws:sns:us-east-1:<account-id>:HumanGuard-Alerts` |
| **Secrets Manager** | `humanGuard/rds`, `humanGuard/exportKey` (us-east-1) |
| **ECR repository** | `<account-id>.dkr.ecr.us-east-1.amazonaws.com/humanguard` |
| **S3 model registry** | `humanguard-models` |

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

**158 tests across 15 test files — all passing.**

| File | Tests | What it covers |
|---|---|---|
| `test_db.py` | 15 | `DatabaseManager` SQLite round-trips; `save_prediction`/`get_stats`/`get_recent_predictions`; source column correctness; graceful handling of unreachable postgres; `SignalCollector` dual-write paths |
| `test_session_score.py` | 8 | Session-level scoring aggregation; linear recency weighting; drift score computation; single-batch edge case; missing `sessionID` rejection; GET alias |
| `test_features.py` | 8 | `FeatureExtractor` output length (must be 37); zero-event edge cases; mouse velocity and path-efficiency math; keystroke entropy; click clustering ratio |
| `test_dashboard.py` | 8 | `/api/dashboard-stats` PostgreSQL and JSONL fallback paths; required field presence; recent prediction ordering; empty-state zeros; model and threshold fields |
| `test_helpers.py` | 7 | `isValidSignalBatch` acceptance/rejection; `normalizeSignalBatch` field aliasing; `genSeshID` format validation |
| `test_api.py` | 7 | `/api/signals` valid and invalid payloads; `/api/score` with mocked bundle; oversized payload 413; CORS header presence |
| `test_shap.py` | 6 | SHAP response structure; top-5 feature list length; `_SHAP_PENDING` sentinel; graceful `None` when explainer unavailable; `?explain=false` suppresses explanation |
| `test_health.py` | 6 | `/health` 200 status; required fields; uptime is float; no model artifact dependency |
| `test_signal_collector.py` | 5 | `saveSignalBatch` JSONL write; batch count increment; session count deduplication; Lambda `/tmp` path redirect |
| `test_demo.py` | 6 | `source`/`label` fields accepted; `/api/export` access control (missing key → 401, wrong key → 401, valid key → 200 CSV) |
| `test_leaderboard.py` | 6 | `POST /api/leaderboard` validation; 404 for unknown session; rank/percentile on success; `GET /api/leaderboard` entries; nickname sanitization; full end-to-end flow (score → submit) |
| `test_model_registry.py` | 5 | `push()` returns semantic version; `load('latest')` resolves champion; `list_versions()` sorted ascending; `promote()` updates champion flag; `rollback()` reinstates previous champion |
| `test_webhooks.py` | 8 | Registration; scoping; delivery; HMAC correctness; wrong-secret rejection; auto-disable after 5 failures; test endpoint; `score_completed` on human scores |
| `test_email_verification.py` | 6 | Verification token generation; 24h expiry; `GET /api/verify` success and expired-token paths; trial limit enforcement; verified key bypass |
| `test_enrichment.py` | 8 | `parse_user_agent()` headless/bot detection; `get_ip_info()` datacenter classification; `enrich_request()` feature vector shape; cache hit behaviour; missing header handling |
| `test_scoring.py` | 5 | Confidence interval fields present; `std` numeric; CI bounds in [0,1]; `confidence` in {high, medium, low}; non-ensemble fallback to point estimate |
