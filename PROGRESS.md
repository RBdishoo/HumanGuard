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

## Phase 5: Deployment & Monitoring 🔄
**Target**: Apr 10, 2026

- [ ] Dockerfile — containerize Flask app + model artifacts
- [ ] AWS Lambda deployment — serverless inference endpoint
- [ ] CloudWatch monitoring — latency, error rate, prediction distribution
- [ ] Alerting — anomaly detection on bot rate spikes
- [ ] PostgreSQL migration — replace JSONL storage for session/label persistence

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
