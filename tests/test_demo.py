"""
Tests for demo page features:
  - source and label fields propagated through /api/signals and /api/score
  - /api/export endpoint access control and CSV format
"""

import sys
import os
import json
import csv
import io
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module

app_module._server_start_time = 1.0
app_module.app.config["TESTING"] = True


def _client():
    return app_module.app.test_client()


def _demo_batch():
    return {
        "sessionID": "demo-test-session-001",
        "source": "demo",
        "label": "human",
        "metadata": {"userAgent": "test-browser", "viewportWidth": 1280, "viewportHeight": 800},
        "signals": {
            "mouseMoves": [
                {"x": 100, "y": 150, "ts": 1000},
                {"x": 200, "y": 160, "ts": 1100},
                {"x": 310, "y": 170, "ts": 1200},
            ],
            "clicks": [{"x": 310, "y": 170, "ts": 1500, "button": 0}],
            "keys": [
                {"key": "T", "ts": 2000},
                {"key": "h", "ts": 2120},
                {"key": "e", "ts": 2250},
            ],
        },
    }


def _fake_bundle():
    import numpy as np

    fake_model = mock.MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.8, 0.2]])

    fake_scaler = mock.MagicMock()
    fake_scaler.transform.side_effect = lambda x: x

    fake_extractor = mock.MagicMock()
    fake_extractor.extractBatchFeatures.return_value = {}

    return {
        "model": fake_model,
        "scaler": fake_scaler,
        "feature_names": ["feat1", "feat2"],
        "threshold": 0.5,
        "extractor": fake_extractor,
        "explainer": app_module._SHAP_PENDING,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. /api/signals accepts source and label without error
# ──────────────────────────────────────────────────────────────────────────────

def test_signals_accepts_source_and_label(tmp_path):
    """POST /api/signals with source='demo' and label='human' returns 200."""
    collector = app_module.collector
    original_file = collector.signalsFile
    collector.signalsFile = str(tmp_path / "signals.jsonl")
    (tmp_path / "signals.jsonl").touch()

    try:
        with mock.patch("db.db_client.is_available", return_value=False):
            resp = _client().post(
                "/api/signals",
                data=json.dumps(_demo_batch()),
                content_type="application/json",
            )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["success"] is True
    finally:
        collector.signalsFile = original_file


# ──────────────────────────────────────────────────────────────────────────────
# 2. /api/score accepts source and label and logs them
# ──────────────────────────────────────────────────────────────────────────────

def test_score_accepts_source_and_label(tmp_path):
    """POST /api/score with source/label returns 200 and logs source in JSONL."""
    log_path = tmp_path / "pred.jsonl"

    with mock.patch("backend.app._load_scoring_bundle", return_value=_fake_bundle()), \
         mock.patch("db.db_client.is_available", return_value=False), \
         mock.patch.object(app_module, "PREDICTIONS_LOG", log_path):
        resp = _client().post(
            "/api/score?explain=false",
            data=json.dumps(_demo_batch()),
            content_type="application/json",
        )

    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body["success"] is True

    # Verify source and ground_truth_label written to log
    assert log_path.exists()
    entry = json.loads(log_path.read_text().strip().splitlines()[-1])
    assert entry.get("source") == "demo"
    assert entry.get("ground_truth_label") == "human"


# ──────────────────────────────────────────────────────────────────────────────
# 3. /api/export returns 401 without a valid API key
# ──────────────────────────────────────────────────────────────────────────────

def test_export_requires_api_key():
    """GET /api/export without X-Export-Key header returns 401."""
    resp = _client().get("/api/export")
    assert resp.status_code == 401
    body = json.loads(resp.data)
    assert "error" in body


def test_export_wrong_key_returns_401():
    """GET /api/export with a wrong key returns 401."""
    with mock.patch.dict(os.environ, {"EXPORT_API_KEY": "secretkey"}):
        resp = _client().get("/api/export", headers={"X-Export-Key": "wrongkey"})
    assert resp.status_code == 401


# ──────────────────────────────────────────────────────────────────────────────
# 4. /api/export returns CSV with correct content-type and columns
# ──────────────────────────────────────────────────────────────────────────────

def test_export_returns_csv_with_valid_key(tmp_path):
    """GET /api/export with correct key returns text/csv response."""
    log_path = tmp_path / "pred.jsonl"
    log_path.write_text(json.dumps({
        "sessionID": "s1",
        "source": "demo",
        "ground_truth_label": "human",
        "prob_bot": 0.1,
        "label": "human",
        "threshold": 0.5,
        "scoring_type": "batch",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }) + "\n")

    with mock.patch.object(app_module, "PREDICTIONS_LOG", log_path), \
         mock.patch.dict(os.environ, {"EXPORT_API_KEY": "testkey"}):
        resp = _client().get("/api/export", headers={"X-Export-Key": "testkey"})

    assert resp.status_code == 200
    assert "text/csv" in resp.content_type


def test_export_csv_has_required_columns(tmp_path):
    """CSV export contains all expected column headers."""
    log_path = tmp_path / "pred.jsonl"
    log_path.write_text(json.dumps({
        "sessionID": "s1",
        "source": "demo",
        "ground_truth_label": "human",
        "prob_bot": 0.1,
        "label": "human",
        "threshold": 0.5,
        "scoring_type": "batch",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }) + "\n")

    with mock.patch.object(app_module, "PREDICTIONS_LOG", log_path), \
         mock.patch.dict(os.environ, {"EXPORT_API_KEY": "devkey"}):
        resp = _client().get("/api/export", headers={"X-Export-Key": "devkey"})

    assert resp.status_code == 200
    reader = csv.DictReader(io.StringIO(resp.data.decode("utf-8")))
    required_cols = {
        "session_id", "source", "ground_truth_label",
        "prob_bot", "predicted_label", "threshold", "scoring_type", "timestamp",
    }
    assert required_cols.issubset(set(reader.fieldnames))

    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "s1"
    assert rows[0]["source"] == "demo"
    assert rows[0]["ground_truth_label"] == "human"
