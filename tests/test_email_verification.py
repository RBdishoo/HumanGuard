"""
Tests for email verification of API keys.

Covers:
- Unverified key works for first 10 requests
- Unverified key blocked after 10 requests
- Valid token verifies successfully
- Expired token rejected
- Already-used (consumed) token rejected
- Verified key has no extra request limit
"""
import sys
import os
import json
from datetime import datetime, timedelta, timezone
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module
from backend.db import DatabaseManager

app_module._server_start_time = 1.0

VERIFY_URL = "https://d1hi33wespusty.cloudfront.net/verify.html"


def _fresh_db():
    with mock.patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}):
        return DatabaseManager()


def _client(db):
    """Return a test client with auth enabled and db_manager patched."""
    app_module.app.config["TESTING"] = False
    c = app_module.app.test_client()
    return c


def _restore():
    app_module.app.config["TESTING"] = True


def _score_request(client, api_key):
    """Make a minimal /api/score POST using the given API key."""
    payload = {
        "sessionID": "verify-test-session",
        "signals": {
            "mouseMoves": [{"x": 10, "y": 20, "ts": 1000}],
            "clicks": [],
            "keys": [],
        },
    }
    return client.post(
        "/api/score?explain=false",
        data=json.dumps(payload),
        content_type="application/json",
        headers={"X-Api-Key": api_key},
    )


# ── Test 1: Unverified key works for first 10 requests ─────────────────────

def test_unverified_key_allowed_for_first_10_requests():
    db = _fresh_db()
    key = db.generate_api_key("trial@example.com")

    record = db.validate_api_key(key)
    assert record is not None
    assert record["verified"] is False, "Newly generated key should be unverified"

    # Simulate 9 prior uses (one short of the limit)
    for _ in range(9):
        db.increment_usage(key)

    client = _client(db)
    try:
        with mock.patch.object(app_module, "db_manager", db), \
             mock.patch.object(app_module, "_load_scoring_bundle") as mock_bundle, \
             mock.patch.object(app_module, "_log_prediction_local"), \
             mock.patch.object(app_module, "metrics"):
            mock_bundle.return_value = _make_bundle()
            resp = _score_request(client, key)
            # count is 9 before the request; decorator increments it atomically,
            # so the 10th request (count becomes 10 after) should still be allowed
            # because the check is > UNVERIFIED_TRIAL_LIMIT (>=10 blocks).
            assert resp.status_code in (200, 429), f"Unexpected status: {resp.status_code}"
            # With 9 prior uses, the request count is 9 — below the 10 limit — so 200.
            assert resp.status_code == 200, (
                "Unverified key with 9 prior uses should still be allowed (below limit)"
            )
    finally:
        _restore()


# ── Test 2: Unverified key blocked after 10 requests ───────────────────────

def test_unverified_key_blocked_after_10_requests():
    db = _fresh_db()
    key = db.generate_api_key("blocked@example.com")

    # Use up all 10 trial requests
    for _ in range(10):
        db.increment_usage(key)

    client = _client(db)
    try:
        with mock.patch.object(app_module, "db_manager", db), \
             mock.patch.object(app_module, "_load_scoring_bundle") as mock_bundle, \
             mock.patch.object(app_module, "_log_prediction_local"), \
             mock.patch.object(app_module, "metrics"):
            mock_bundle.return_value = _make_bundle()
            resp = _score_request(client, key)
            assert resp.status_code == 403, (
                f"Expected 403 after 10 unverified uses, got {resp.status_code}"
            )
            data = resp.get_json()
            assert "email verification required" in data.get("error", "")
            assert "verify_url" in data
    finally:
        _restore()


# ── Test 3: Valid token verifies successfully ───────────────────────────────

def test_valid_token_verifies_key():
    db = _fresh_db()
    key = db.generate_api_key("verify@example.com")

    token = db.get_verification_token(key)
    assert token is not None, "Should have a pending token"

    ok = db.verify_api_key_email(token)
    assert ok is True, "Valid token should verify successfully"

    record = db.validate_api_key(key)
    assert record is not None
    assert record["verified"] is True, "Key should be marked verified after token use"

    # Token should now be consumed
    assert db.get_verification_token(key) is None, "Token should be gone after use"


# ── Test 4: Expired token is rejected ──────────────────────────────────────

def test_expired_token_rejected():
    db = _fresh_db()
    key = db.generate_api_key("expired@example.com")

    # Directly expire the token in the DB by back-dating token_expires_at
    key_id = key.split(".")[0]  # e.g. hg_live_abcd1234
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with db._sqlite_cursor() as cur:
        cur.execute(
            "UPDATE api_keys SET token_expires_at = ? WHERE key_id = ?",
            (past, key_id),
        )

    token = db.get_verification_token(key)
    assert token is None, "Expired token should not be retrievable"

    # Try to verify directly with the raw token (simulate an attacker reusing a known token)
    with db._sqlite_cursor() as cur:
        cur.execute("SELECT verification_token FROM api_keys WHERE key_id = ?", (key_id,))
        row = cur.fetchone()
    raw_token = row["verification_token"] if row else None

    if raw_token:
        ok = db.verify_api_key_email(raw_token)
        assert ok is False, "Expired token should be rejected by verify_api_key_email"


# ── Test 5: Already-used token is rejected ─────────────────────────────────

def test_already_used_token_rejected():
    db = _fresh_db()
    key = db.generate_api_key("reuse@example.com")
    token = db.get_verification_token(key)
    assert token is not None

    # Use the token once — should succeed
    assert db.verify_api_key_email(token) is True

    # Use the same token again — should fail (already consumed)
    assert db.verify_api_key_email(token) is False, (
        "Token should be rejected after it has already been used"
    )


# ── Test 6: Verified key has no extra unverified request limit ──────────────

def test_verified_key_has_no_unverified_limit():
    db = _fresh_db()
    key = db.generate_api_key("power@example.com")

    # Verify the key
    token = db.get_verification_token(key)
    assert db.verify_api_key_email(token) is True

    # Use 15 requests (well above UNVERIFIED_TRIAL_LIMIT=10)
    for _ in range(15):
        db.increment_usage(key)

    record = db.validate_api_key(key)
    assert record is not None
    assert record["verified"] is True
    assert record["current_month_count"] == 15

    client = _client(db)
    try:
        with mock.patch.object(app_module, "db_manager", db), \
             mock.patch.object(app_module, "_load_scoring_bundle") as mock_bundle, \
             mock.patch.object(app_module, "_log_prediction_local"), \
             mock.patch.object(app_module, "metrics"):
            mock_bundle.return_value = _make_bundle()
            resp = _score_request(client, key)
            # Verified key must not hit the 403 path — expect 200 or 429 (monthly limit)
            assert resp.status_code != 403, (
                "Verified key should not be blocked by the unverified trial limit"
            )
    finally:
        _restore()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_bundle():
    """Minimal scoring bundle backed by a real RandomForest."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    FEAT = ["f1", "f2", "f3", "f4", "f5"]
    rng = np.random.RandomState(7)
    X = rng.rand(40, len(FEAT))
    y = (X[:, 0] > 0.5).astype(int)
    model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=7)
    model.fit(X, y)

    scaler = mock.MagicMock()
    scaler.transform.side_effect = lambda x: x

    extractor = mock.MagicMock()
    extractor.extractBatchFeatures.return_value = {f: rng.rand() for f in FEAT}

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": FEAT,
        "threshold": 0.5,
        "explainer": None,
        "extractor": extractor,
        "session_blender": None,
        "session_blender_features": None,
        "_registry_version": None,
        "_registry_metadata": {},
    }
