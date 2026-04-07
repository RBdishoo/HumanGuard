import sys
import os
import json
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module

app_module._server_start_time = 1.0
app_module.app.config["TESTING"] = True


def _client():
    return app_module.app.test_client()


def _valid_batch():
    return {
        "sessionID": "api-test-session",
        "metadata": {"userAgent": "test", "viewportWidth": 1920, "viewportHeight": 1080},
        "signals": {
            "mouseMoves": [{"x": 10, "y": 20, "ts": 1000}],
            "clicks": [],
            "keys": [],
        },
    }


# -------------------------------------------------------------------
# POST /api/signals
# -------------------------------------------------------------------

def test_signals_valid_payload(tmp_path):
    collector = app_module.collector
    original_file = collector.signalsFile
    collector.signalsFile = str(tmp_path / "signals.jsonl")
    (tmp_path / "signals.jsonl").touch()

    try:
        with mock.patch("db.db_client.is_available", return_value=False):
            resp = _client().post(
                "/api/signals",
                data=json.dumps(_valid_batch()),
                content_type="application/json",
            )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["success"] is True
    finally:
        collector.signalsFile = original_file


def test_signals_missing_fields():
    resp = _client().post(
        "/api/signals",
        data=json.dumps({"sessionID": "x"}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_signals_empty_body():
    resp = _client().post(
        "/api/signals",
        data="{}",
        content_type="application/json",
    )
    assert resp.status_code == 400


# -------------------------------------------------------------------
# POST /api/signals — size cap (413)
# -------------------------------------------------------------------

def test_signals_oversized_mouseMoves_returns_413(tmp_path):
    """saveSignals returns 413 when mouseMoves exceeds MAX_MOUSE_MOVES."""
    collector = app_module.collector
    original_file = collector.signalsFile
    collector.signalsFile = str(tmp_path / "signals.jsonl")
    (tmp_path / "signals.jsonl").touch()
    try:
        batch = _valid_batch()
        batch["signals"]["mouseMoves"] = [{"x": i, "y": i, "ts": i}
                                          for i in range(app_module.MAX_MOUSE_MOVES + 1)]
        with mock.patch("db.db_client.is_available", return_value=False):
            resp = _client().post(
                "/api/signals",
                data=json.dumps(batch),
                content_type="application/json",
            )
        assert resp.status_code == 413
        body = json.loads(resp.data)
        assert body["max"]["mouseMoves"] == app_module.MAX_MOUSE_MOVES
    finally:
        collector.signalsFile = original_file


def test_signals_oversized_clicks_returns_413(tmp_path):
    """saveSignals returns 413 when clicks exceeds MAX_CLICKS."""
    collector = app_module.collector
    original_file = collector.signalsFile
    collector.signalsFile = str(tmp_path / "signals.jsonl")
    (tmp_path / "signals.jsonl").touch()
    try:
        batch = _valid_batch()
        batch["signals"]["clicks"] = [{"ts": i, "button": 0}
                                       for i in range(app_module.MAX_CLICKS + 1)]
        with mock.patch("db.db_client.is_available", return_value=False):
            resp = _client().post(
                "/api/signals",
                data=json.dumps(batch),
                content_type="application/json",
            )
        assert resp.status_code == 413
        body = json.loads(resp.data)
        assert body["max"]["clicks"] == app_module.MAX_CLICKS
    finally:
        collector.signalsFile = original_file


def test_signals_oversized_keys_returns_413(tmp_path):
    """saveSignals returns 413 when keys exceeds MAX_KEYS."""
    collector = app_module.collector
    original_file = collector.signalsFile
    collector.signalsFile = str(tmp_path / "signals.jsonl")
    (tmp_path / "signals.jsonl").touch()
    try:
        batch = _valid_batch()
        batch["signals"]["keys"] = [{"code": "KeyA", "ts": i}
                                     for i in range(app_module.MAX_KEYS + 1)]
        with mock.patch("db.db_client.is_available", return_value=False):
            resp = _client().post(
                "/api/signals",
                data=json.dumps(batch),
                content_type="application/json",
            )
        assert resp.status_code == 413
        body = json.loads(resp.data)
        assert body["max"]["keys"] == app_module.MAX_KEYS
    finally:
        collector.signalsFile = original_file


# -------------------------------------------------------------------
# POST /api/score
# -------------------------------------------------------------------

def test_score_oversized_batch():
    batch = _valid_batch()
    batch["signals"]["mouseMoves"] = [{"x": i, "y": i, "ts": i} for i in range(5001)]

    with mock.patch("backend.app._load_scoring_bundle"):
        resp = _client().post(
            "/api/score",
            data=json.dumps(batch),
            content_type="application/json",
        )
    assert resp.status_code == 413


def test_score_missing_model_artifacts():
    # Point MODEL_DIR at a nonexistent path so _load_scoring_bundle raises
    original = app_module.MODEL_DIR
    app_module.MODEL_DIR = app_module.Path("/tmp/nonexistent_model_dir")
    app_module._scoring_bundle = None  # force reload

    try:
        resp = _client().post(
            "/api/score",
            data=json.dumps(_valid_batch()),
            content_type="application/json",
        )
        assert resp.status_code == 503
    finally:
        app_module.MODEL_DIR = original
        app_module._scoring_bundle = None


# -------------------------------------------------------------------
# GET /api/stats
# -------------------------------------------------------------------

def test_stats_returns_200():
    resp = _client().get("/api/stats")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "Total Batches" in data
    assert "Unique Sessions" in data
    assert "Server Timestamp" in data


# -------------------------------------------------------------------
# GET /
# -------------------------------------------------------------------

def test_frontend_returns_200():
    resp = _client().get("/")
    assert resp.status_code == 200
