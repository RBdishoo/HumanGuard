"""
Tests for the webhook system: registration, listing, delivery, HMAC signing,
failure tracking, auto-disable, and the test endpoint.
"""
import hashlib
import hmac
import json
import os
import sys
import threading
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module
from backend.db import DatabaseManager

app_module._server_start_time = 1.0


def _fresh_db():
    with mock.patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}):
        return DatabaseManager()


def _client(db):
    """Return a test client wired to an isolated in-memory DB."""
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()
    return client


# ---------------------------------------------------------------------------
# 1. Register webhook returns id
# ---------------------------------------------------------------------------

def test_register_webhook_returns_id():
    db = _fresh_db()
    with mock.patch.object(app_module, "db_manager", db):
        client = _client(db)
        resp = client.post(
            "/api/webhooks",
            data=json.dumps({
                "url": "https://example.com/hook",
                "secret": "mysecret",
                "events": ["bot_detected"],
            }),
            content_type="application/json",
        )
    assert resp.status_code == 201, resp.data
    body = json.loads(resp.data)
    assert "id" in body
    assert isinstance(body["id"], int)
    assert body["id"] > 0
    assert body["url"] == "https://example.com/hook"
    assert body["active"] is True


# ---------------------------------------------------------------------------
# 2. List webhooks scoped to api key
# ---------------------------------------------------------------------------

def test_list_webhooks_scoped_to_api_key():
    db = _fresh_db()
    # Register two webhooks under different (mocked) key IDs
    db.register_webhook("key-a", "https://a.com/hook", "sec", "bot_detected")
    db.register_webhook("key-b", "https://b.com/hook", "sec", "bot_detected")

    with mock.patch.object(app_module, "db_manager", db):
        # g.api_key == "test" in TESTING mode; _key_id_from_full("test") == "test"
        # Register one more via the endpoint so it belongs to key_id "test"
        client = _client(db)
        client.post(
            "/api/webhooks",
            data=json.dumps({"url": "https://test.com/hook", "secret": "s", "events": ["bot_detected"]}),
            content_type="application/json",
        )
        resp = client.get("/api/webhooks")

    assert resp.status_code == 200
    body = json.loads(resp.data)
    # Only the webhook registered under "test" should be visible
    urls = [w["url"] for w in body]
    assert "https://test.com/hook" in urls
    assert "https://a.com/hook" not in urls
    assert "https://b.com/hook" not in urls


# ---------------------------------------------------------------------------
# 3. Webhook fires on bot detection
# ---------------------------------------------------------------------------

def test_webhook_fires_on_bot_detection():
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from unittest.mock import MagicMock

    db = _fresh_db()
    db.register_webhook("test", "https://recv.example.com/hook", "secret123", "bot_detected")

    calls = []
    done = threading.Event()

    def fake_deliver(webhook, payload_bytes):
        calls.append(json.loads(payload_bytes))
        done.set()

    # Train on two classes so predict_proba always returns 2 columns
    rng = np.random.RandomState(42)
    X = rng.rand(20, 5)
    y = [0] * 10 + [1] * 10
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    scaler = MagicMock()
    scaler.transform.return_value = X[15:16]  # bot-like row

    bundle = {
        "model": clf,
        "scaler": scaler,
        "feature_names": ["f1", "f2", "f3", "f4", "f5"],
        "threshold": 0.0,  # force bot label
        "explainer": None,
        "extractor": MagicMock(extractBatchFeatures=MagicMock(return_value={})),
        "_registry_version": None,
        "_registry_metadata": {},
        "session_blender": None,
        "session_blender_features": None,
    }

    with mock.patch.object(app_module, "db_manager", db), \
         mock.patch.object(app_module, "_deliver_webhook", side_effect=fake_deliver), \
         mock.patch.object(app_module, "_load_scoring_bundle", return_value=bundle):
        client = _client(db)
        resp = client.post(
            "/api/score",
            data=json.dumps({"sessionID": "sess-fire",
                             "signals": {"mouseMoves": [], "clicks": [], "keys": []}}),
            content_type="application/json",
        )

    done.wait(timeout=2.0)

    assert resp.status_code == 200
    assert json.loads(resp.data)["label"] == "bot"
    assert len(calls) >= 1
    assert calls[0]["event"] == "bot_detected"
    assert calls[0]["session_id"] == "sess-fire"


# ---------------------------------------------------------------------------
# 4. HMAC signature covers the final (delivery_id-injected) payload
# ---------------------------------------------------------------------------

def test_hmac_signature_is_correct():
    secret = "supersecret"
    payload = {
        "event": "bot_detected",
        "session_id": "s1",
        "prob_bot": 0.9,
        "confidence": "high",
        "confidence_interval": {"lower": 0.85, "upper": 0.99},
        "top_features": [],
    }
    payload_bytes = json.dumps(payload).encode("utf-8")

    received = {}

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_open(req, timeout):
        received["sig"] = req.get_header("X-humanguard-signature")
        received["body"] = req.data
        received["delivery"] = req.get_header("X-humanguard-delivery")
        return FakeResp()

    webhook = {"id": 1, "url": "https://example.com/hook", "secret": secret}

    db = _fresh_db()
    with mock.patch.object(app_module, "db_manager", db), \
         mock.patch.object(app_module, "_is_safe_resolved_host", return_value=True), \
         mock.patch.object(app_module._WEBHOOK_OPENER, "open", side_effect=fake_open):
        app_module._deliver_webhook(webhook, payload_bytes)

    # Signature must cover the *actual* bytes sent (which include delivery_id + timestamp)
    actual_bytes = received["body"]
    expected_sig = hmac.new(secret.encode(), actual_bytes, hashlib.sha256).hexdigest()
    assert received["sig"] == f"sha256={expected_sig}"

    # delivery_id in header must match the field injected into the payload
    actual_payload = json.loads(actual_bytes)
    assert received["delivery"] == actual_payload["delivery_id"]


# ---------------------------------------------------------------------------
# 5. Wrong secret fails verification
# ---------------------------------------------------------------------------

def test_wrong_secret_fails_verification():
    payload_bytes = b'{"event":"bot_detected"}'
    correct_secret = "correct"
    wrong_secret = "wrong"

    correct_sig = hmac.new(correct_secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
    wrong_sig = hmac.new(wrong_secret.encode(), payload_bytes, hashlib.sha256).hexdigest()

    assert correct_sig != wrong_sig

    # A receiver verifying with the wrong secret would compute a different digest
    submitted = f"sha256={correct_sig}"
    recomputed = f"sha256={wrong_sig}"
    assert not hmac.compare_digest(submitted, recomputed)


# ---------------------------------------------------------------------------
# 6. Webhook disabled after 5 consecutive failures
# ---------------------------------------------------------------------------

def test_webhook_disabled_after_5_failures():
    db = _fresh_db()
    wh_id = db.register_webhook("key-x", "https://fail.example.com/hook", "s", "bot_detected")
    assert wh_id > 0

    for _ in range(5):
        db.update_webhook_status(wh_id, success=False)

    active = db.get_webhooks_for_key("key-x", active_only=True)
    assert len(active) == 0, "Webhook should be disabled after 5 failures"

    all_hooks = db.get_webhooks_for_key("key-x", active_only=False)
    assert len(all_hooks) == 1
    assert all_hooks[0]["failure_count"] == 5
    assert not bool(all_hooks[0]["active"])


# ---------------------------------------------------------------------------
# 7. Test endpoint sends payload (uses _deliver_webhook signing path)
# ---------------------------------------------------------------------------

def test_test_endpoint_sends_payload():
    db = _fresh_db()
    wh_id = db.register_webhook("test", "https://test-recv.example.com/hook", "testsecret", "bot_detected")

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass

    captured = {}

    def fake_open(req, timeout):
        captured["url"] = req.full_url
        captured["body"] = req.data
        captured["sig"] = req.get_header("X-humanguard-signature")
        captured["delivery"] = req.get_header("X-humanguard-delivery")
        return FakeResp()

    with mock.patch.object(app_module, "db_manager", db), \
         mock.patch.object(app_module, "_is_safe_resolved_host", return_value=True), \
         mock.patch.object(app_module._WEBHOOK_OPENER, "open", side_effect=fake_open):
        client = _client(db)
        resp = client.post(f"/api/webhooks/{wh_id}/test")

    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body["success"] is True
    assert "url" in captured
    assert captured["url"] == "https://test-recv.example.com/hook"
    payload = json.loads(captured["body"])
    assert payload["event"] == "test"
    assert payload["session_id"] == "test-session"
    assert captured["sig"].startswith("sha256=")
    # delivery_id must appear in both the header and the payload
    assert "delivery_id" in payload
    assert captured["delivery"] == payload["delivery_id"]


# ---------------------------------------------------------------------------
# 8. score_completed fires on every score (including human)
# ---------------------------------------------------------------------------

def test_score_completed_fires_on_every_score():
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from unittest.mock import MagicMock

    db = _fresh_db()
    db.register_webhook("test", "https://all.example.com/hook", "sec", "score_completed")

    calls = []
    done = threading.Event()

    def fake_deliver(webhook, payload_bytes):
        calls.append(json.loads(payload_bytes))
        done.set()

    rng = np.random.RandomState(42)
    X = rng.rand(20, 5)
    y = [0] * 10 + [1] * 10
    clf = RandomForestClassifier(n_estimators=10, random_state=0)
    clf.fit(X, y)
    scaler = MagicMock()
    scaler.transform.return_value = X[2:3]  # human-like row

    bundle = {
        "model": clf,
        "scaler": scaler,
        "feature_names": ["f1", "f2", "f3", "f4", "f5"],
        "threshold": 0.99,  # always "human"
        "explainer": None,
        "extractor": MagicMock(extractBatchFeatures=MagicMock(return_value={})),
        "_registry_version": None,
        "_registry_metadata": {},
        "session_blender": None,
        "session_blender_features": None,
    }

    with mock.patch.object(app_module, "db_manager", db), \
         mock.patch.object(app_module, "_deliver_webhook", side_effect=fake_deliver), \
         mock.patch.object(app_module, "_load_scoring_bundle", return_value=bundle):
        client = _client(db)
        resp = client.post(
            "/api/score",
            data=json.dumps({"sessionID": "sess-human",
                             "signals": {"mouseMoves": [], "clicks": [], "keys": []}}),
            content_type="application/json",
        )

    done.wait(timeout=2.0)

    assert resp.status_code == 200
    assert json.loads(resp.data)["label"] == "human"
    assert len(calls) >= 1
    assert calls[0]["event"] == "score_completed"


# ---------------------------------------------------------------------------
# 9. _is_safe_resolved_host rejects private / reserved IP ranges
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ip,label", [
    ("10.0.0.1",       "RFC1918 class-A"),
    ("10.255.255.255",  "RFC1918 class-A boundary"),
    ("172.16.0.1",     "RFC1918 class-B"),
    ("172.31.255.255",  "RFC1918 class-B boundary"),
    ("192.168.0.1",    "RFC1918 class-C"),
    ("192.168.255.254", "RFC1918 class-C boundary"),
    ("127.0.0.1",      "loopback"),
    ("127.255.255.255", "loopback boundary"),
    ("169.254.0.1",    "link-local / AWS metadata"),
    ("169.254.169.254", "AWS IMDS endpoint"),
])
def test_is_safe_resolved_host_rejects_private(ip, label):
    """_is_safe_resolved_host returns False for all private/reserved ranges."""
    with mock.patch("socket.getaddrinfo", return_value=[(None, None, None, None, (ip, 0))]):
        assert app_module._is_safe_resolved_host("internal.example.com") is False, (
            f"Expected False for {label} ({ip})"
        )


# ---------------------------------------------------------------------------
# 10. _is_safe_resolved_host accepts a real public IP
# ---------------------------------------------------------------------------

def test_is_safe_resolved_host_accepts_public_ip():
    """_is_safe_resolved_host returns True for a globally routable address."""
    public_ip = "93.184.216.34"  # example.com
    with mock.patch("socket.getaddrinfo", return_value=[(None, None, None, None, (public_ip, 0))]):
        assert app_module._is_safe_resolved_host("example.com") is True


# ---------------------------------------------------------------------------
# 11. _deliver_webhook skips delivery when resolved host is private
# ---------------------------------------------------------------------------

def test_deliver_webhook_blocks_private_resolved_host():
    """_deliver_webhook returns (False, …) and does not open a connection when
    the hostname resolves to a private address."""
    db = _fresh_db()
    webhook = {"id": 42, "url": "https://rebind.example.com/hook", "secret": "s"}

    opener_called = []

    with mock.patch.object(app_module, "db_manager", db), \
         mock.patch.object(app_module, "_is_safe_resolved_host", return_value=False), \
         mock.patch.object(app_module._WEBHOOK_OPENER, "open",
                           side_effect=lambda *a, **kw: opener_called.append(True)):
        success, delivery_id, sig = app_module._deliver_webhook(webhook, b'{"event":"test"}')

    assert success is False
    assert opener_called == [], "HTTP request must not be made for a private resolved host"


# ---------------------------------------------------------------------------
# 12. _deliver_webhook skips delivery for http:// URLs
# ---------------------------------------------------------------------------

def test_deliver_webhook_blocks_http_url():
    """_deliver_webhook returns (False, …) and does not open a connection for non-HTTPS URLs."""
    db = _fresh_db()
    webhook = {"id": 99, "url": "http://example.com/hook", "secret": "s"}

    opener_called = []

    with mock.patch.object(app_module, "db_manager", db), \
         mock.patch.object(app_module._WEBHOOK_OPENER, "open",
                           side_effect=lambda *a, **kw: opener_called.append(True)):
        success, delivery_id, sig = app_module._deliver_webhook(webhook, b'{"event":"test"}')

    assert success is False
    assert opener_called == [], "HTTP request must not be made for a plain-HTTP webhook URL"


# ---------------------------------------------------------------------------
# 13. Webhook payloads include delivery_id and timestamp
# ---------------------------------------------------------------------------

def test_webhook_payload_includes_delivery_id_and_timestamp():
    """_deliver_webhook injects delivery_id and timestamp into every payload."""
    db = _fresh_db()
    webhook = {"id": 7, "url": "https://recv.example.com/hook", "secret": "sec"}
    original = {"event": "bot_detected", "session_id": "s1", "prob_bot": 0.9}

    captured_body = {}

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_open(req, timeout):
        captured_body["data"] = req.data
        return FakeResp()

    with mock.patch.object(app_module, "db_manager", db), \
         mock.patch.object(app_module, "_is_safe_resolved_host", return_value=True), \
         mock.patch.object(app_module._WEBHOOK_OPENER, "open", side_effect=fake_open):
        app_module._deliver_webhook(webhook, json.dumps(original).encode())

    sent = json.loads(captured_body["data"])
    assert "delivery_id" in sent, "delivery_id must be present in sent payload"
    assert "timestamp" in sent, "timestamp must be present in sent payload"
    assert len(sent["delivery_id"]) == 32, "delivery_id should be 32 hex chars (token_hex(16))"
    # Original fields must still be present
    assert sent["event"] == "bot_detected"
    assert sent["session_id"] == "s1"


# ---------------------------------------------------------------------------
# 14. X-HumanGuard-Delivery header matches delivery_id in payload
# ---------------------------------------------------------------------------

def test_webhook_delivery_header_present():
    """_deliver_webhook sets X-HumanGuard-Delivery to the same value as payload delivery_id."""
    db = _fresh_db()
    webhook = {"id": 8, "url": "https://recv.example.com/hook", "secret": "sec"}

    captured = {}

    class FakeResp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_open(req, timeout):
        captured["header"] = req.get_header("X-humanguard-delivery")
        captured["body"] = req.data
        return FakeResp()

    with mock.patch.object(app_module, "db_manager", db), \
         mock.patch.object(app_module, "_is_safe_resolved_host", return_value=True), \
         mock.patch.object(app_module._WEBHOOK_OPENER, "open", side_effect=fake_open):
        app_module._deliver_webhook(webhook, b'{"event":"score_completed"}')

    sent = json.loads(captured["body"])
    assert captured["header"] is not None, "X-HumanGuard-Delivery header must be set"
    assert captured["header"] == sent["delivery_id"], (
        "X-HumanGuard-Delivery header must equal the delivery_id field in the payload"
    )
