"""
Tests for the client dashboard endpoints:
  GET /api/client/stats
  GET /api/client/predictions
  GET /client
  GET /register
  get_client_stats / get_client_predictions on DatabaseManager
"""
import sys
import os
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module
from backend.db import DatabaseManager

app_module._server_start_time = 1.0


def _fresh_db():
    with mock.patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}):
        return DatabaseManager()


def _client_with_key(key, db):
    """Return a Flask test client with auth bypassed and db patched."""
    app_module.app.config["TESTING"] = False
    with mock.patch.object(app_module, "db_manager", db):
        yield app_module.app.test_client(), key


# ---------------------------------------------------------------------------
# 1. GET /api/client/stats — returns empty stats for a fresh key
# ---------------------------------------------------------------------------

def test_client_stats_empty_for_new_key():
    db = _fresh_db()
    key = db.generate_api_key("client@example.com")
    app_module.app.config["TESTING"] = False
    try:
        with mock.patch.object(app_module, "db_manager", db):
            c = app_module.app.test_client()
            resp = c.get("/api/client/stats", headers={"X-Api-Key": key})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_sessions"] == 0
        assert data["bot_count"] == 0
        assert data["bot_rate"] == 0.0
        assert "usage_count" in data
        assert "usage_limit" in data
        assert isinstance(data["daily_stats"], list)
    finally:
        app_module.app.config["TESTING"] = True


# ---------------------------------------------------------------------------
# 2. GET /api/client/predictions — returns empty list for a fresh key
# ---------------------------------------------------------------------------

def test_client_predictions_empty_for_new_key():
    db = _fresh_db()
    key = db.generate_api_key("pred@example.com")
    app_module.app.config["TESTING"] = False
    try:
        with mock.patch.object(app_module, "db_manager", db):
            c = app_module.app.test_client()
            resp = c.get("/api/client/predictions", headers={"X-Api-Key": key})
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) == 0
    finally:
        app_module.app.config["TESTING"] = True


# ---------------------------------------------------------------------------
# 3. get_client_predictions returns predictions scoped to the api_key
# ---------------------------------------------------------------------------

def test_get_client_predictions_scoped():
    db = _fresh_db()
    key_a = db.generate_api_key("a@example.com")
    key_b = db.generate_api_key("b@example.com")

    # Save a prediction for key_a
    db.save_prediction("sess-001", 0.9, True, api_key=key_a)
    # Save a prediction for key_b
    db.save_prediction("sess-002", 0.1, False, api_key=key_b)

    preds_a = db.get_client_predictions(key_a)
    preds_b = db.get_client_predictions(key_b)

    assert len(preds_a) == 1
    assert preds_a[0]["session_id"] == "sess-001"
    assert preds_a[0]["label"] == "bot"

    assert len(preds_b) == 1
    assert preds_b[0]["session_id"] == "sess-002"
    assert preds_b[0]["label"] == "human"


# ---------------------------------------------------------------------------
# 4. get_client_stats reflects saved predictions
# ---------------------------------------------------------------------------

def test_get_client_stats_counts():
    db = _fresh_db()
    key = db.generate_api_key("stats@example.com")

    db.save_prediction("s1", 0.95, True, api_key=key)
    db.save_prediction("s2", 0.85, True, api_key=key)
    db.save_prediction("s3", 0.1, False, api_key=key)

    stats = db.get_client_stats(key)
    assert stats["total_sessions"] == 3
    assert stats["bot_count"] == 2
    assert stats["human_count"] == 1
    assert abs(stats["bot_rate"] - 2 / 3) < 0.01


# ---------------------------------------------------------------------------
# 5. GET /client and GET /register serve HTML pages
# ---------------------------------------------------------------------------

def test_client_and_register_pages_served():
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()

    resp_client = c.get("/client")
    assert resp_client.status_code == 200
    assert b"HumanGuard" in resp_client.data

    resp_register = c.get("/register")
    assert resp_register.status_code == 200
    assert b"API Key" in resp_register.data
