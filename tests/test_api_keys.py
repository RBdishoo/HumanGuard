"""
Tests for the API key system: generation, validation, rate limiting,
master key bypass, usage endpoint, register endpoint, and monthly reset.
"""
import sys
import os
import json
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module
from backend.db import DatabaseManager

# Use an isolated in-memory DatabaseManager for each test so we don't
# pollute the shared singleton.
app_module._server_start_time = 1.0


def _fresh_db():
    """Return an in-memory DatabaseManager (api_keys table auto-created)."""
    with mock.patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}):
        return DatabaseManager()


def _client():
    # Disable TESTING bypass so auth logic runs
    app_module.app.config["TESTING"] = False
    return app_module.app.test_client()


def _restore_testing():
    app_module.app.config["TESTING"] = True


# ---------------------------------------------------------------------------
# 1. generate_api_key creates a valid, retrievable key
# ---------------------------------------------------------------------------

def test_generate_and_validate_key():
    db = _fresh_db()
    key = db.generate_api_key("alice@example.com")
    assert key.startswith("hg_live_")
    record = db.validate_api_key(key)
    assert record is not None
    assert record["owner_email"] == "alice@example.com"
    assert record["plan"] == "free"
    assert record["monthly_limit"] == 1000
    assert record["active"] is True


# ---------------------------------------------------------------------------
# 2. Invalid key returns 401
# ---------------------------------------------------------------------------

def test_invalid_key_returns_401():
    try:
        # Patch db_manager used inside app with an isolated in-memory DB
        db = _fresh_db()
        with mock.patch.object(app_module, "db_manager", db):
            with mock.patch.dict(os.environ, {"HUMANGUARD_MASTER_KEY": ""}):
                resp = _client().get(
                    "/api/stats",
                    headers={"X-Api-Key": "hg_live_notarealkey"},
                )
        assert resp.status_code == 401
        body = json.loads(resp.data)
        assert "Invalid" in body["error"] or "inactive" in body["error"].lower()
    finally:
        _restore_testing()


# ---------------------------------------------------------------------------
# 3. Inactive key returns 401
# ---------------------------------------------------------------------------

def test_inactive_key_returns_401():
    try:
        db = _fresh_db()
        key = db.generate_api_key("bob@example.com")
        # Deactivate the key directly in SQLite
        with db._sqlite_cursor() as cur:
            cur.execute("UPDATE api_keys SET active = 0 WHERE key = ?", (key,))

        with mock.patch.object(app_module, "db_manager", db):
            with mock.patch.dict(os.environ, {"HUMANGUARD_MASTER_KEY": ""}):
                resp = _client().get(
                    "/api/stats",
                    headers={"X-Api-Key": key},
                )
        assert resp.status_code == 401
    finally:
        _restore_testing()


# ---------------------------------------------------------------------------
# 4. Rate limit returns 429 after free-tier limit is reached
# ---------------------------------------------------------------------------

def test_rate_limit_returns_429():
    try:
        db = _fresh_db()
        key = db.generate_api_key("carol@example.com", monthly_limit=5)
        # Exhaust the quota
        with db._sqlite_cursor() as cur:
            cur.execute("UPDATE api_keys SET current_month_count = 5 WHERE key = ?", (key,))

        with mock.patch.object(app_module, "db_manager", db):
            with mock.patch.dict(os.environ, {"HUMANGUARD_MASTER_KEY": ""}):
                resp = _client().get(
                    "/api/stats",
                    headers={"X-Api-Key": key},
                )
        assert resp.status_code == 429
        body = json.loads(resp.data)
        assert body["error"] == "monthly limit reached"
        assert body["limit"] == 5
        assert body["plan"] == "free"
    finally:
        _restore_testing()


# ---------------------------------------------------------------------------
# 5. Master key bypasses rate limit
# ---------------------------------------------------------------------------

def test_master_key_bypasses_rate_limit():
    try:
        db = _fresh_db()
        master = "hg_master_testkey"
        with mock.patch.object(app_module, "db_manager", db):
            with mock.patch.dict(os.environ, {"HUMANGUARD_MASTER_KEY": master}):
                resp = _client().get(
                    "/api/stats",
                    headers={"X-Api-Key": master},
                )
        # /api/stats returns 200 when master key is used (no rate limit applied)
        assert resp.status_code == 200
    finally:
        _restore_testing()


# ---------------------------------------------------------------------------
# 6. Usage endpoint returns correct counts
# ---------------------------------------------------------------------------

def test_usage_endpoint_returns_correct_counts():
    try:
        db = _fresh_db()
        key = db.generate_api_key("dave@example.com")
        # Simulate 42 prior requests
        with db._sqlite_cursor() as cur:
            cur.execute("UPDATE api_keys SET current_month_count = 42 WHERE key = ?", (key,))

        with mock.patch.object(app_module, "db_manager", db):
            with mock.patch.dict(os.environ, {"HUMANGUARD_MASTER_KEY": ""}):
                resp = _client().get(
                    "/api/usage",
                    headers={"X-Api-Key": key},
                )
        assert resp.status_code == 200
        body = json.loads(resp.data)
        assert body["count"] == 42
        assert body["limit"] == 1000
        assert body["plan"] == "free"
        assert body["percentage_used"] == 4.2
    finally:
        _restore_testing()


# ---------------------------------------------------------------------------
# 7. Register endpoint creates a valid key
# ---------------------------------------------------------------------------

def test_register_endpoint_creates_valid_key():
    db = _fresh_db()
    # /api/register is not protected — TESTING flag doesn't matter here,
    # but we keep TESTING=False for consistency.
    try:
        with mock.patch.object(app_module, "db_manager", db):
            resp = _client().post(
                "/api/register",
                data=json.dumps({"email": "eve@example.com"}),
                content_type="application/json",
            )
        assert resp.status_code == 201
        body = json.loads(resp.data)
        assert "api_key" in body
        assert body["api_key"].startswith("hg_live_")
        assert body["plan"] == "free"
        assert body["monthly_limit"] == 1000
        assert "docs_url" in body

        # Key should be retrievable from the DB
        record = db.validate_api_key(body["api_key"])
        assert record is not None
        assert record["owner_email"] == "eve@example.com"
    finally:
        _restore_testing()


# ---------------------------------------------------------------------------
# 8. Monthly reset sets count to 0
# ---------------------------------------------------------------------------

def test_monthly_reset_sets_count_to_zero():
    db = _fresh_db()
    key1 = db.generate_api_key("frank@example.com")
    key2 = db.generate_api_key("grace@example.com")
    # Set usage counts
    with db._sqlite_cursor() as cur:
        cur.execute("UPDATE api_keys SET current_month_count = 500 WHERE key = ?", (key1,))
        cur.execute("UPDATE api_keys SET current_month_count = 999 WHERE key = ?", (key2,))

    db.reset_monthly_counts()

    assert db.get_usage(key1)["count"] == 0
    assert db.get_usage(key2)["count"] == 0
