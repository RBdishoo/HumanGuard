"""
Tests for GET /dashboard and GET /api/dashboard-stats.
"""

import json
import sys
import os
from unittest import mock
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module

app_module._server_start_time = 1.0
app_module.app.config["TESTING"] = True


def _client():
    return app_module.app.test_client()


def _write_predictions(path, records):
    """Write a list of prediction dicts as JSONL to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── /dashboard ────────────────────────────────────────────────────────────────

def test_dashboard_returns_200():
    resp = _client().get("/dashboard")
    assert resp.status_code == 200


# ── /api/dashboard-stats ─────────────────────────────────────────────────────

def test_dashboard_stats_returns_200(tmp_path):
    app_module.PREDICTIONS_LOG = tmp_path / "predictions_log.jsonl"
    app_module.PREDICTIONS_LOG.touch()
    with mock.patch("db.db_client.is_available", return_value=False):
        resp = _client().get("/api/dashboard-stats")
    assert resp.status_code == 200


def test_dashboard_stats_has_required_keys(tmp_path):
    app_module.PREDICTIONS_LOG = tmp_path / "predictions_log.jsonl"
    app_module.PREDICTIONS_LOG.touch()
    with mock.patch("db.db_client.is_available", return_value=False):
        resp = _client().get("/api/dashboard-stats")
    data = json.loads(resp.data)
    required = {
        "total_predictions", "bot_count", "human_count", "bot_rate",
        "recent_predictions", "top_flagged_features", "model", "threshold",
    }
    assert required <= set(data.keys())


def test_dashboard_stats_bot_rate_in_range(tmp_path):
    records = [
        {"sessionID": "s1", "prob_bot": 0.9, "label": "bot",
         "threshold": 0.5, "scoring_type": "batch", "timestamp": "2026-03-28T00:00:00+00:00"},
        {"sessionID": "s2", "prob_bot": 0.1, "label": "human",
         "threshold": 0.5, "scoring_type": "batch", "timestamp": "2026-03-28T00:00:01+00:00"},
    ]
    log = tmp_path / "predictions_log.jsonl"
    _write_predictions(log, records)
    app_module.PREDICTIONS_LOG = log
    with mock.patch("db.db_client.is_available", return_value=False):
        resp = _client().get("/api/dashboard-stats")
    data = json.loads(resp.data)
    assert 0.0 <= data["bot_rate"] <= 1.0


def test_dashboard_stats_recent_predictions_max_10(tmp_path):
    records = [
        {"sessionID": f"s{i}", "prob_bot": 0.5, "label": "human",
         "threshold": 0.5, "scoring_type": "batch", "timestamp": "2026-03-28T00:00:00+00:00"}
        for i in range(15)
    ]
    log = tmp_path / "predictions_log.jsonl"
    _write_predictions(log, records)
    app_module.PREDICTIONS_LOG = log
    with mock.patch("db.db_client.is_available", return_value=False):
        resp = _client().get("/api/dashboard-stats")
    data = json.loads(resp.data)
    assert isinstance(data["recent_predictions"], list)
    assert len(data["recent_predictions"]) <= 10


def test_dashboard_stats_top_flagged_features_max_5(tmp_path):
    explanation = {
        "top_features": [
            {"feature": f"feat_{j}", "contribution": 0.1}
            for j in range(8)  # more than 5 distinct features
        ]
    }
    records = [
        {"sessionID": f"s{i}", "prob_bot": 0.9, "label": "bot",
         "threshold": 0.5, "scoring_type": "batch",
         "timestamp": "2026-03-28T00:00:00+00:00", "explanation": explanation}
        for i in range(5)
    ]
    log = tmp_path / "predictions_log.jsonl"
    _write_predictions(log, records)
    app_module.PREDICTIONS_LOG = log
    with mock.patch("db.db_client.is_available", return_value=False):
        resp = _client().get("/api/dashboard-stats")
    data = json.loads(resp.data)
    assert isinstance(data["top_flagged_features"], list)
    assert len(data["top_flagged_features"]) <= 5


def test_dashboard_stats_empty_log(tmp_path):
    log = tmp_path / "predictions_log.jsonl"
    log.touch()
    app_module.PREDICTIONS_LOG = log
    with mock.patch("db.db_client.is_available", return_value=False):
        resp = _client().get("/api/dashboard-stats")
    data = json.loads(resp.data)
    assert data["total_predictions"] == 0
    assert data["bot_rate"] == 0.0
    assert data["recent_predictions"] == []
    assert data["top_flagged_features"] == []


def test_dashboard_stats_mixed_results(tmp_path):
    records = [
        {"sessionID": "bot-1", "prob_bot": 0.95, "label": "bot",
         "threshold": 0.5, "scoring_type": "batch", "timestamp": "2026-03-28T10:00:00+00:00",
         "response_time_ms": 30.0,
         "explanation": {"top_features": [{"feature": "mouseStdVelocity", "contribution": 0.4}]}},
        {"sessionID": "human-1", "prob_bot": 0.15, "label": "human",
         "threshold": 0.5, "scoring_type": "batch", "timestamp": "2026-03-28T10:00:01+00:00",
         "response_time_ms": 25.0},
        {"sessionID": "bot-2", "prob_bot": 0.80, "label": "bot",
         "threshold": 0.5, "scoring_type": "session", "timestamp": "2026-03-28T10:00:02+00:00",
         "response_time_ms": 50.0,
         "explanation": {"top_features": [{"feature": "mouseStdVelocity", "contribution": 0.3},
                                          {"feature": "clickIntervalStdMs", "contribution": 0.2}]}},
    ]
    log = tmp_path / "predictions_log.jsonl"
    _write_predictions(log, records)
    app_module.PREDICTIONS_LOG = log
    with mock.patch("db.db_client.is_available", return_value=False):
        resp = _client().get("/api/dashboard-stats")
    data = json.loads(resp.data)
    assert data["total_predictions"] == 3
    assert data["bot_count"] == 2
    assert data["human_count"] == 1
    assert 0.0 <= data["bot_rate"] <= 1.0
    assert "mouseStdVelocity" in data["top_flagged_features"]
    assert data["avg_response_time_ms"] > 0
