import time
import json
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import backend.app as app_module


def _client():
    """Create a Flask test client with server start time set."""
    app_module._server_start_time = time.time()
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_health_returns_200():
    client = _client()
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_has_all_keys():
    client = _client()
    data = json.loads(client.get("/health").data)
    expected = {"status", "model", "version", "uptime_seconds", "timestamp"}
    assert expected == set(data.keys())


def test_health_status_is_ok():
    client = _client()
    data = json.loads(client.get("/health").data)
    assert data["status"] == "ok"


def test_health_uptime_is_positive_float():
    app_module._server_start_time = time.time() - 5.0
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()
    data = json.loads(client.get("/health").data)
    assert isinstance(data["uptime_seconds"], float)
    assert data["uptime_seconds"] > 0


def test_health_timestamp_is_iso8601():
    client = _client()
    data = json.loads(client.get("/health").data)
    # Will raise ValueError if not valid ISO format
    parsed = datetime.fromisoformat(data["timestamp"])
    assert parsed is not None


def test_health_works_without_model_artifacts(tmp_path):
    """Endpoint must respond even when models/trained/ is empty."""
    original = app_module.MODEL_DIR
    app_module.MODEL_DIR = tmp_path / "nonexistent"
    try:
        client = _client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "ok"
    finally:
        app_module.MODEL_DIR = original
