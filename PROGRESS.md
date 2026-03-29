# HumanGuard — Development Log

A phased build log tracking architectural decisions, implementation results, and upcoming work.

---

## Phase 1: Signal Collection ✅
**Completed**: Feb 3, 2026

Built the data collection foundation — Flask server, frontend signal capture, and JSONL storage pipeline.

**Delivered:**
- Flask API serving `frontend/index.html` with live signal capture (`tracker.js`)
- `POST /api/signals` — receives and validates signal batches from the browser
- `SignalCollector` appends batches as JSONL to `backend/data/raw/signals.jsonl`
- `GET /api/stats` — returns batch count, session count, file size, and server timestamp
- Input validation via `isValidSignalBatch()` with structured error responses

**Technical notes:**
- JSONL chosen over JSON for append-only writes without full file reloads
- Server invoked as `python -m backend.app` for correct absolute imports
- Mouse events throttled to 100ms in `tracker.js`; batches auto-sent every 3 seconds

---

## Phase 2: Feature Engineering ✅
**Completed**: Feb 17, 2026

Extracted 33 behavioral features from raw signal streams.

**Feature categories:**
- **Mouse** — path efficiency (start-to-end vs actual distance), velocity, acceleration, direction changes, pause frequency
- **Keystroke** — inter-key delay mean/variance, Shannon entropy, typing rhythm consistency
- **Click** — rate, spatial clustering ratio, button distribution, inter-click timing variance
- **Session** — tab focus ratio, idle periods, temporal composite features

**Key modules:**
- `backend/features/feature_extractor.py` — `FeatureExtractor.extractBatchFeatures()` → 33-feature dict
- `backend/features/feature_utils.py` — `MouseTrajectoryUtils`, `KeystrokeUtils` static math helpers
- `backend/features/dataset_builder.py` — builds batch-level and session-level CSV datasets
- `backend/features/data_loader.py` — loads and validates JSONL signal files

---

## Phase 3: Data Collection & Labeling ✅
**Completed**: Mar 10, 2026

Built synthetic data generation pipeline and labeled training dataset.

**Dataset (final):** 233 total batches — 120 bot, 113 human

**Bot sessions (8 total):**
- 4 classic bot sessions — linear mouse paths, uniform keystroke timing, instant clicks
- 4 stealthy bot sessions (`--stealthy` flag) — randomized timing, positional jitter, varied key phrases

**Human sessions (11 total):**
- 1 real browser session
- 10 simulated sessions via `scripts/seed_human_session.py`:
  - Quadratic Bézier curved mouse paths with random control points
  - Per-point Gaussian jitter (σ=4px) and variable velocity (acceleration curve)
  - 15% probability pause simulation (400–1500ms) for reading/hover behavior
  - Gaussian inter-keystroke delays (μ=180ms, σ=60ms) with diverse phrase corpus
  - Sessions 3–4 intentionally ambiguous (bot-like behavior) to improve decision boundary

**Label storage:** `backend/data/raw/labels.csv` — `sessionID`, `label` (human/bot)

---

## Phase 4: ML Model Training ✅
**Completed**: Mar 27, 2026

Trained and evaluated three classifiers on the same stratified train/test split (80/20, fixed seed).

**Model comparison:**

| Metric       | Logistic Regression | Random Forest | XGBoost  |
|--------------|--------------------:|-------------:|---------:|
| Accuracy     | 0.8409              | 0.9091       | **0.9091** |
| Precision    | 0.8148              | 0.9167       | **0.9167** |
| Recall       | 0.9167              | 0.9167       | **0.9167** |
| F1           | 0.8627              | 0.9167       | **0.9167** |
| ROC-AUC      | 0.8802              | 0.9031       | **0.9135** |

**Selected: XGBoost** (`max_depth=3`, `n_estimators=100`, `learning_rate=0.1`)

XGBoost and RandomForest tied on all threshold-dependent metrics. XGBoost's higher ROC-AUC (0.9135 vs 0.9031) indicates better probability calibration across all thresholds — relevant because `/api/score` returns a continuous `prob_bot` value, not just a binary label.

**Regularization approach:** `max_depth=3`, `min_samples_leaf=5` (RF), 10% symmetric label noise injected during training to prevent overfitting on the 233-batch dataset and simulate real-world annotation ambiguity.

**Artifacts saved to `models/trained/`:**
- `XGBoost.pkl` — active classifier
- `scaler.pkl` — StandardScaler fitted on training data
- `feature_names.json` — ordered 33-feature list
- `threshold.json` — classification threshold (default 0.5)
- `model_comparison.json` — full metrics for all three models

---

## Phase 5: Deployment & Monitoring ✅
**Completed**: Mar 29, 2026

**Live API:** https://9ixzk5e9u4.execute-api.us-east-1.amazonaws.com

- [x] Dockerfile — containerize Flask app + model artifacts
- [x] AWS Lambda deployment — serverless inference endpoint
- [x] CloudWatch monitoring — latency, error rate, prediction distribution
- [ ] Alerting — anomaly detection on bot rate spikes
- [ ] PostgreSQL migration — replace JSONL storage for session/label persistence

### Deployment Notes

Getting Flask running in a Lambda container required fixing several environment incompatibilities in sequence:

**Filesystem (read-only everywhere except `/tmp`):**
- Added `IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None` to detect Lambda at runtime
- `SignalCollector` redirects `signals.jsonl` → `/tmp/signals.jsonl` when `IS_LAMBDA` is true
- `PREDICTIONS_LOG` redirects `predictions_log.jsonl` → `/tmp/predictions_log.jsonl` on Lambda

**Flask dev server incompatibilities:**
- `debug=True` activates Werkzeug's `DebuggedApplication` which requires `/dev/shm` (shared memory) — Lambda doesn't have it; fixed with `debug=not IS_LAMBDA`
- Flask's default `host='127.0.0.1'` is unreachable by Lambda's runtime interface client; fixed with `host='0.0.0.0'`

**Lambda invocation model (no HTTP server):**
- Lambda calls a handler function, not a persistent HTTP server; added a minimal WSGI bridge (`handler(event, context)`) that translates API Gateway v2 HTTP events into WSGI `environ` dicts and calls `app()` directly
- Added `awslambdaric` to `requirements-prod.txt`; Dockerfile CMD changed to `python -m awslambdaric backend.app.handler`

**joblib / scikit-learn parallelism:**
- `joblib.load()` can attempt to spawn worker processes that require `/dev/shm`; wrapped both `load()` calls in `with parallel_backend('sequential'):`
- Set `JOBLIB_MULTIPROCESSING=0` and `LOKY_MAX_CPU_COUNT=1` as Lambda env vars for belt-and-suspenders coverage

**XGBoost OpenMP:**
- Set `OMP_NUM_THREADS=1` to keep XGBoost single-threaded and avoid inter-process OpenMP locking

**Root cause of 60s timeout — numba JIT compilation (SHAP):**
- SHAP's `TreeExplainer` uses numba to JIT-compile tree traversal routines on first call; numba's compilation stalled for the full timeout duration in Lambda's restricted environment
- Fixed with `NUMBA_DISABLE_JIT=1` (SHAP falls back to pure Python); `NUMBA_CACHE_DIR=/tmp/.numba` set as a safety net if JIT is re-enabled later

---

## Phase 6: Advanced Features
**Target**: Apr 10+ (Ongoing)

- [ ] SHAP explainability — per-prediction feature attribution
- [ ] Adversarial testing — evaluate against crafted evasion attempts
- [ ] Ensemble models — stack XGBoost + RandomForest for edge cases
- [ ] Continuous learning — retrain pipeline on incoming labeled production data

---

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage (Phase 1–4) | JSONL | Append-only, no lock contention, human-readable |
| Storage (Phase 5+) | PostgreSQL | Indexed queries, relational labels, production scale |
| Frontend | Vanilla JS | No build step, direct DOM access for low-latency signal capture |
| Feature count | 33 | Covers all behavioral signal types without high-dimensional noise |
| Active classifier | XGBoost | Best ROC-AUC; superior probability calibration for continuous scoring |
| Model depth | max_depth=3 | Prevents overfitting on sub-500 sample dataset |
| Label noise | 10% symmetric | Regularization against clean-label overfitting |
| Server port | 5050 (via `PORT` env var) | Avoids macOS AirPlay conflict on 5000 |

---

## Resources

- Python 3.11 · Flask 3.1.2 · XGBoost · scikit-learn 1.8.0 · NumPy 2.4 · Pandas 2.3
- Deployment target: AWS Lambda · PostgreSQL (Phase 5+) · CloudWatch
