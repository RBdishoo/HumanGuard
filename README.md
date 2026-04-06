<div align="center">

# 🛡️ HumanGuard

**Real-time behavioral bot detection API**

Classifies web sessions as human or bot by analyzing mouse trajectories, keystroke dynamics, click patterns, and network signals — no CAPTCHAs required.

[![Tests](https://img.shields.io/badge/tests-158%20passing-brightgreen?style=flat-square)](#testing)
[![Python](https://img.shields.io/badge/python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![AWS Lambda](https://img.shields.io/badge/AWS-Lambda-orange?style=flat-square&logo=amazonaws)](https://aws.amazon.com/lambda/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)](LICENSE)
[![Live Demo](https://img.shields.io/badge/demo-live-success?style=flat-square)](https://humanguard.net/demo.html)

[**Live Demo**](https://humanguard.net/demo.html) · [**Leaderboard**](https://humanguard.net/leaderboard.html) · [**Dashboard**](https://humanguard.net/dashboard.html) · [**Register**](https://humanguard.net/register.html)

</div>

---

## What It Does

HumanGuard embeds a lightweight JavaScript tracker on any webpage. The tracker collects raw behavioral signals every 3 seconds and sends them to a scoring API. A **37-feature extraction pipeline** feeds a RandomForest classifier that returns a bot probability score, a 95% confidence interval, and a SHAP-powered explanation of which signals drove the decision — all in a single JSON response.

A second-layer **temporal session blender** catches adaptive bots that behave human-like in early batches then revert to scripted patterns. It achieves **100% session-level detection across 5 hard adversarial bot patterns** where batch-level scoring alone reached only 56%.

> 🏆 **99.6% F1 · 1.0000 ROC-AUC · 100% adversarial detection**

---

## Quick Start

**Option 1 — Drop the tracker into any page:**

```html
<script src="https://humanguard.net/sdk/humanGuard.min.js"
        data-session-id="your-session-id"
        data-api-key="hg_live_xxxxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx">
</script>
```

**Option 2 — Score a batch directly:**

```bash
curl -X POST https://humanguard.net/api/score \
  -H "Content-Type: application/json" \
  -H "X-API-Key: hg_live_xxxxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  -d '{
    "sessionID": "session_1743200000_abc123",
    "signals": {
      "mouseMoves": [{"x": 412, "y": 308, "ts": 1000}, {"x": 428, "y": 315, "ts": 1087}],
      "clicks":    [{"x": 441, "y": 323, "ts": 1450, "button": 0}],
      "keys":      [{"code": "KeyH", "ts": 1520}, {"code": "KeyE", "ts": 1634}]
    }
  }'
```

```json
{
  "success": true,
  "sessionID": "session_1743200000_abc123",
  "prob_bot": 0.032,
  "label": "human",
  "threshold": 0.5,
  "confidence": "high",
  "confidence_interval": { "lower": 0.011, "upper": 0.053 },
  "network_signals": {
    "is_headless_browser": false,
    "is_datacenter_ip": false,
    "is_known_bot_ua": false
  },
  "explanation": {
    "interpretation": "Session classified as human; top signal: natural mouse velocity variance.",
    "top_features": [
      { "feature": "mouseStdVelocity",    "contribution":  0.768 },
      { "feature": "batchDurationMs",     "contribution": -2.174 },
      { "feature": "keystroke_timing_regularity", "contribution": -0.412 }
    ]
  }
}
```

[**→ Get an API key**](https://humanguard.net/register.html)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser                                                        │
│  tracker.js — captures mouse, keys, clicks, scroll             │
│  auto-batches every 3s → POST /api/signals or /api/score        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS
                  ┌────────▼────────┐
                  │  API Gateway    │  HTTP API · ANY /{proxy+}
                  └────────┬────────┘
                           │
                  ┌────────▼────────────────────────────────────┐
                  │  AWS Lambda  (Flask · Python 3.11)           │
                  │                                             │
                  │  ┌─────────────────────────────────────┐   │
                  │  │  enrichment.py                      │   │
                  │  │  parse_user_agent · get_ip_info      │   │
                  │  │  7 network/device features          │   │
                  │  └──────────────┬──────────────────────┘   │
                  │                 │                           │
                  │  ┌──────────────▼──────────────────────┐   │
                  │  │  feature_extractor.py               │   │
                  │  │  37 features (30 behavioral +        │   │
                  │  │               7 network/device)      │   │
                  │  └──────────────┬──────────────────────┘   │
                  │                 │                           │
                  │  ┌──────────────▼──────────────────────┐   │
                  │  │  RandomForest  +  SHAP explainer    │   │
                  │  │  session blender (≥10 batches)       │   │
                  │  └──────────────┬──────────────────────┘   │
                  │                 │                           │
                  └─────────────────┼───────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
   ┌──────▼──────┐        ┌─────────▼──────┐        ┌────────▼──────┐
   │ RDS Postgres│        │  CloudWatch    │        │  S3 Static    │
   │ sessions    │        │  5 metrics     │        │  Dashboard    │
   │ predictions │        │  4 alarms      │        │  Leaderboard  │
   │ webhooks    │        │  SNS alerts    │        │  Demo         │
   └─────────────┘        └────────────────┘        └───────────────┘
```

---

## Features

### Signal Collection
- JavaScript tracker captures **mouse movements** (100 ms throttle), **keystroke timings**, **click coordinates**, and **scroll events**
- Auto-batches every 3 seconds; works headlessly via `POST /api/signals` or inline via `POST /api/score`
- UTM source tagging passes `utm_source` through each batch for real-data attribution

### 37-Feature Extraction Pipeline

| Category | Count | Key Features |
|---|---|---|
| Mouse trajectory | 9 | velocity (avg/std/max), path efficiency, angular velocity std, hover time/frequency, pause duration |
| Click dynamics | 5 | interval mean/std/min/max, click rate |
| Keystroke dynamics | 6 | inter-key delay mean/std, rapid presses, entropy, key rate, count |
| Temporal composites | 4 | batch duration, event rate, click-to-move ratio, key-to-move ratio |
| Batch-level | 4 | event count, has_mouse_moves, has_clicks, has_keys |
| Session consistency | 5 | keystroke timing regularity, typing rhythm autocorrelation, mouse acceleration variance, mouse-keystroke correlation, session phase consistency |
| Network / device | 7 | headless browser, known bot UA, datacenter IP, UA entropy, Accept-Language presence/count, suspicious header count |

### Model & Scoring
- **RandomForest champion** — 99.6% F1, 1.0000 ROC-AUC on 5-fold cross-validation
- **95% confidence intervals** on every score — derived from per-tree variance; exposes `confidence`, `confidence_interval`, and `std` fields
- **SHAP explainability** — top-5 feature attributions with human-readable interpretation text on every response (`?explain=false` to skip)
- **Temporal session blender** — LogisticRegression meta-model aggregates batches for sessions ≥10; measures behavioral drift between first-half and second-half patterns

### Adversarial Robustness

| Bot Pattern | Batch Detection | Session Detection |
|---|---|---|
| human_speed_typer | 100% | **100%** |
| bezier_mouse | 100% | **100%** |
| jitter_bot | 100% | **100%** |
| hybrid_bot | 100% | **100%** |
| adaptive_bot | 50% | **100%** |
| **Overall** | **90.0%** | **100.0%** |

### Infrastructure
- **Webhooks** — HMAC-SHA256 signed delivery; `bot_detected`, `score_completed`, `high_confidence_bot` events; auto-disables after 5 failures
- **Email verification** — AWS SES verification flow on API key registration; 10-request trial period
- **Auto-retrain pipeline** — EventBridge (6h) triggers `retrain.py --auto`; retrains when real human session count ≥ 50; promotes champion to S3 model registry
- **CloudWatch monitoring** — 5 custom metrics, 4 production alarms (bot rate spike, p95 latency, error rate, validation errors)
- **CORS allowlist** — env-var-driven `ALLOWED_ORIGINS`; no wildcard in production
- **Atomic quota enforcement** — `UPDATE … WHERE count < limit RETURNING` prevents TOCTOU race on monthly usage checks

---

## API Reference

### Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | — | Liveness check |
| `POST` | `/api/score` | API key | Score a signal batch; returns prob, label, SHAP, CI |
| `POST` | `/api/signals` | — | Ingest a raw signal batch for storage |
| `GET/POST` | `/api/session-score` | API key | Aggregate all batches for a session with temporal blending |
| `GET` | `/api/stats` | — | Collection statistics and live prediction counts |
| `GET` | `/api/dashboard-stats` | — | Dashboard data: recent predictions, bot rate, top features |
| `GET` | `/api/data-stats` | — | Real session counts and retrain readiness |
| `GET` | `/api/retrain-status` | API key | Last retrain timestamp and trigger status |
| `GET` | `/api/model-info` | API key | Active model version and champion metadata |
| `POST` | `/api/register` | — | Register and obtain an API key (email verification required) |
| `GET/POST` | `/api/leaderboard` | — | Public demo leaderboard |
| `POST/GET/DELETE` | `/api/webhooks` | API key | Manage webhook subscriptions |

### `/api/score` — Full Response Schema

```json
{
  "success": true,
  "sessionID": "session_1743200000_abc123",
  "prob_bot": 0.032,
  "label": "human",
  "threshold": 0.5,
  "confidence": "high",
  "confidence_interval": { "lower": 0.011, "upper": 0.053 },
  "std": 0.021,
  "network_signals": {
    "is_headless_browser": false,
    "is_known_bot_ua": false,
    "is_datacenter_ip": false,
    "ua_entropy": 4.87,
    "has_accept_language": true,
    "accept_language_count": 2,
    "suspicious_header_count": 0
  },
  "explanation": {
    "interpretation": "Session classified as human; top signal: natural mouse velocity variance.",
    "top_features": [
      { "feature": "mouseStdVelocity",             "contribution":  0.768 },
      { "feature": "batchDurationMs",              "contribution": -2.174 },
      { "feature": "keystroke_timing_regularity",  "contribution": -0.412 },
      { "feature": "clickToMoveRatio",             "contribution": -0.348 },
      { "feature": "mouseHoverFrequency",          "contribution": -0.268 }
    ]
  }
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **ML** | scikit-learn RandomForest, SHAP TreeExplainer |
| **Backend** | Flask 3, Python 3.11 |
| **Runtime** | AWS Lambda (container image) + `awslambdaric` |
| **API** | AWS API Gateway HTTP API |
| **Database** | PostgreSQL 15 on AWS RDS · SQLite auto-fallback for local dev |
| **Object Storage** | AWS S3 (model registry + static frontend) |
| **Monitoring** | AWS CloudWatch (custom namespace) + SNS email alerts |
| **Email** | AWS SES (API key verification) |
| **Secrets** | AWS Secrets Manager (`humanGuard/rds`, `humanGuard/exportKey`) |
| **CDN** | AWS CloudFront (`humanguard.net`) |
| **Container Registry** | AWS ECR |
| **Testing** | pytest (158 tests · 15 files) |

---

## Local Development

```bash
# 1. Clone and set up environment
git clone https://github.com/RBdishoo/HumanGuard.git
cd HumanGuard
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Generate synthetic training data and retrain
python scripts/seed_bot_session.py
python scripts/seed_human_session.py
python -m models.run_training

# 3. Start the server (http://localhost:5050)
python -m backend.app

# 4. Run the test suite
pytest tests/ -v
```

The server auto-selects its database backend: set `DATABASE_URL=postgres://...` to use PostgreSQL locally, or set `RDS_SECRET_NAME=humanGuard/rds` to resolve credentials from AWS Secrets Manager (production path). With neither set, it falls back to a local SQLite file at `backend/data/humanguard.db`. CloudWatch metrics are no-ops unless `CLOUDWATCH_ENABLED=true` is set. The email service logs verification links to stdout when `SENDER_EMAIL` is unset.

---

## Deployment

Full deploy is automated in `scripts/aws_deploy.sh`.

```bash
# One-time infrastructure setup
DB_PASSWORD=<strong-password> python infrastructure/rds_setup.py
python infrastructure/cloudwatch_alarms.py
python infrastructure/ses_setup.py       # SES sender verification

# Full deploy
bash scripts/aws_deploy.sh
```

For redeployments after Lambda already exists:

```bash
docker build --platform linux/amd64 -t humanguard:latest .
docker tag humanguard:latest <account>.dkr.ecr.us-east-1.amazonaws.com/humanguard:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/humanguard:latest
aws lambda update-function-code --function-name humanguard \
  --image-uri <account>.dkr.ecr.us-east-1.amazonaws.com/humanguard:latest
```

The deploy script provisions: ECR repository → Docker image push → IAM execution role → Lambda function → API Gateway HTTP API → CloudFront distribution.

---

## Project Structure

```
HumanGuard/
├── backend/
│   ├── app.py                    # Flask application — all 12 API routes
│   ├── enrichment.py             # IP/UA enrichment — ipinfo.io + UA parsing
│   ├── email_service.py          # AWS SES sender with HTML template
│   ├── monitoring.py             # CloudWatch metrics singleton
│   ├── collectors/
│   │   └── signal_collector.py   # JSONL + PostgreSQL dual-write
│   ├── db/
│   │   ├── __init__.py           # DatabaseManager (SQLite / PostgreSQL)
│   │   ├── db_client.py          # PostgreSQL connection pool
│   │   └── schema.sql            # DDL: sessions, signal_batches, predictions, webhooks
│   └── features/
│       ├── feature_extractor.py  # 37-feature extraction (30 behavioral + 7 network)
│       ├── feature_utils.py      # Mouse trajectory and keystroke math utilities
│       ├── dataset_builder.py    # Batch/session-level CSV dataset builder
│       └── data_loader.py        # JSONL signal loader and validator
├── models/
│   ├── dataset.py                # ModelDataset: feature loading, scaling, split
│   ├── train.py                  # ModelTrainer: RandomForest, LR, XGBoost comparison
│   ├── run_training.py           # Training entry point
│   └── trained/                  # Serialized artifacts (.pkl)
├── frontend/
│   ├── tracker.js                # Browser signal collector
│   ├── demo.html                 # Interactive challenge demo
│   ├── dashboard.html            # Live monitoring dashboard
│   ├── leaderboard.html          # Public leaderboard
│   ├── register.html             # API key registration
│   └── verify.html               # Email verification landing page
├── scripts/
│   ├── aws_deploy.sh             # Full ECR → Lambda → API Gateway deploy
│   ├── retrain.py                # Auto-retrain with S3 model registry
│   ├── seed_bot_session.py       # Synthetic bot session generator
│   └── seed_human_session.py     # Synthetic human session generator
├── infrastructure/
│   ├── rds_setup.py              # Idempotent RDS + Secrets Manager provisioning
│   ├── cloudwatch_alarms.py      # 4 CloudWatch alarms + SNS topic
│   └── ses_setup.py              # SES sender verification
├── tests/                        # 158 pytest tests across 15 files
├── Dockerfile                    # python:3.11-slim · linux/amd64 · awslambdaric
├── requirements.txt              # Development dependencies
├── requirements-prod.txt         # Production dependencies
└── PROGRESS.md                   # Full technical build log
```

---

## Testing

158 tests across 15 files — all passing.

```bash
pytest tests/ -v
```

| File | Tests | Coverage |
|---|---|---|
| `test_features.py` | 8 | 37-feature vector shape, math correctness, edge cases |
| `test_api.py` | 7 | `/api/signals`, `/api/score`, oversized payload, CORS headers |
| `test_db.py` | 15 | `DatabaseManager` round-trips, source column, dual-write, fallbacks |
| `test_session_score.py` | 8 | Aggregation, recency weighting, drift scoring, edge cases |
| `test_shap.py` | 6 | Top-5 features, `_SHAP_PENDING` sentinel, `?explain=false` |
| `test_scoring.py` | 5 | Confidence interval fields, bounds in [0,1], non-ensemble fallback |
| `test_webhooks.py` | 8 | Registration, scoping, HMAC correctness, auto-disable |
| `test_email_verification.py` | 6 | 24h token, expiry, trial limit, verified bypass |
| `test_enrichment.py` | 8 | Headless/bot UA detection, datacenter IP, cache hit, missing headers |
| `test_model_registry.py` | 5 | Push, load latest, promote, rollback |
| `test_leaderboard.py` | 6 | Validation, rank/percentile, full score→submit flow |
| `test_dashboard.py` | 8 | Stats fields, PostgreSQL and JSONL fallback paths |
| `test_demo.py` | 6 | source/label fields, export access control |
| `test_health.py` | 6 | Status, required fields, uptime |
| `test_signal_collector.py` | 5 | JSONL write, session deduplication, Lambda `/tmp` redirect |
| `test_helpers.py` | 7 | Signal validation, normalization, session ID format |

---

## Live URLs

| Resource | URL |
|---|---|
| Demo | https://humanguard.net/demo.html |
| Leaderboard | https://humanguard.net/leaderboard.html |
| Dashboard | https://humanguard.net/dashboard.html |
| Register | https://humanguard.net/register.html |
| SDK | https://humanguard.net/sdk/humanGuard.min.js |
| API | https://humanguard.net |

---

## License

MIT
