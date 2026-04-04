"""
Tests for leaderboard endpoints:
  - POST /api/leaderboard — validation, session lookup, rank computation
  - GET  /api/leaderboard — returns entries with ranks and stats
"""

import sys
import os
import json
import tempfile
from unittest import mock
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module

app_module._server_start_time = 1.0
app_module.app.config["TESTING"] = True


def _client():
    return app_module.app.test_client()


def _score_batch(session_id="lb-test-session-001"):
    """Minimal signal batch for /api/score."""
    return {
        "sessionID": session_id,
        "source": "demo",
        "label": "human",
        "metadata": {"userAgent": "test", "viewportWidth": 1280, "viewportHeight": 800},
        "signals": {
            "mouseMoves": [
                {"x": 100, "y": 150, "ts": 1000},
                {"x": 200, "y": 160, "ts": 1100},
                {"x": 310, "y": 170, "ts": 1200},
            ],
            "clicks": [{"x": 310, "y": 170, "ts": 1500, "button": 0}],
            "keys": [
                {"key": "T", "ts": 2000},
                {"key": "h", "ts": 2080},
                {"key": "e", "ts": 2160},
            ],
        },
    }


def test_leaderboard_post_requires_nickname():
    """POST /api/leaderboard with missing nickname returns 400."""
    client = _client()
    resp = client.post(
        "/api/leaderboard",
        data=json.dumps({"session_id": "some-session"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    body = json.loads(resp.data)
    assert "nickname" in body.get("error", "").lower()


def test_leaderboard_post_requires_session_id():
    """POST /api/leaderboard with missing session_id returns 400."""
    client = _client()
    resp = client.post(
        "/api/leaderboard",
        data=json.dumps({"nickname": "Alice"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    body = json.loads(resp.data)
    assert "session_id" in body.get("error", "").lower()


def test_leaderboard_post_session_not_found():
    """POST /api/leaderboard with unknown session_id returns 404."""
    client = _client()
    with mock.patch.object(app_module, "PREDICTIONS_LOG", Path("/nonexistent/path.jsonl")):
        with mock.patch.object(app_module.db_manager, "get_recent_predictions", return_value=[]):
            resp = client.post(
                "/api/leaderboard",
                data=json.dumps({"nickname": "Bob", "session_id": "unknown-session-xyz"}),
                content_type="application/json",
            )
    assert resp.status_code == 404
    body = json.loads(resp.data)
    assert "session" in body.get("error", "").lower()


def test_leaderboard_post_success():
    """POST /api/leaderboard with valid session returns rank and percentile message."""
    client = _client()
    session_id = "lb-test-post-success-001"

    # Write a fake predictions log entry for this session
    log_entry = json.dumps({
        "sessionID": session_id,
        "prob_bot": 0.12,
        "label": "human",
        "threshold": 0.5,
        "scoring_type": "batch",
        "source": "demo",
        "timestamp": "2026-03-29T00:00:00+00:00",
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(log_entry + "\n")
        tmp_path = Path(f.name)

    try:
        with mock.patch.object(app_module, "PREDICTIONS_LOG", tmp_path):
            # Use in-memory DB so the leaderboard table is clean
            with mock.patch.object(
                app_module.db_manager, "save_leaderboard_entry", return_value=1
            ):
                with mock.patch.object(
                    app_module.db_manager,
                    "get_leaderboard",
                    return_value=[{
                        "nickname": "TestUser",
                        "prob_bot": 0.12,
                        "verdict": "human",
                        "session_id": session_id,
                        "created_at": "2026-03-29T00:00:00",
                    }],
                ):
                    resp = client.post(
                        "/api/leaderboard",
                        data=json.dumps({"nickname": "TestUser", "session_id": session_id}),
                        content_type="application/json",
                    )
    finally:
        tmp_path.unlink(missing_ok=True)

    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body["rank"] == 1
    assert body["total"] == 1
    assert body["nickname"] == "TestUser"
    assert isinstance(body["prob_bot"], float)
    assert "message" in body
    assert "%" in body["message"]


def test_leaderboard_get_returns_entries():
    """GET /api/leaderboard returns entries list and stats dict."""
    client = _client()
    fake_entries = [
        {
            "nickname": "Alice",
            "prob_bot": 0.08,
            "verdict": "human",
            "session_id": "sess-alice",
            "created_at": "2026-03-29T10:00:00",
        },
        {
            "nickname": "Bob",
            "prob_bot": 0.45,
            "verdict": "human",
            "session_id": "sess-bob",
            "created_at": "2026-03-29T10:05:00",
        },
    ]
    fake_stats = {"total": 2, "avg_prob_bot": 0.265, "pct_human": 100.0}

    with mock.patch.object(app_module.db_manager, "get_leaderboard", return_value=fake_entries):
        with mock.patch.object(
            app_module.db_manager, "get_leaderboard_stats", return_value=fake_stats
        ):
            resp = client.get("/api/leaderboard")

    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert "entries" in body
    assert "stats" in body
    assert body["stats"]["total"] == 2

    entries = body["entries"]
    assert len(entries) == 2
    assert entries[0]["rank"] == 1
    assert entries[0]["nickname"] == "Alice"
    assert entries[0]["human_confidence"] == 92  # round((1-0.08)*100)
    assert entries[1]["rank"] == 2


def test_leaderboard_sanitizes_nickname():
    """POST /api/leaderboard strips special characters and enforces 20-char limit."""
    client = _client()
    session_id = "lb-sanitize-test-001"
    log_entry = json.dumps({
        "sessionID": session_id,
        "prob_bot": 0.2,
        "label": "human",
        "threshold": 0.5,
        "scoring_type": "batch",
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(log_entry + "\n")
        tmp_path = Path(f.name)

    try:
        with mock.patch.object(app_module, "PREDICTIONS_LOG", tmp_path):
            captured = {}

            def _capture(nickname, prob_bot, verdict, session_id):
                captured["nickname"] = nickname
                return 1

            with mock.patch.object(app_module.db_manager, "save_leaderboard_entry", side_effect=_capture):
                with mock.patch.object(
                    app_module.db_manager,
                    "get_leaderboard",
                    return_value=[{
                        "nickname": "hacker",
                        "prob_bot": 0.2,
                        "verdict": "human",
                        "session_id": session_id,
                        "created_at": "2026-03-29T00:00:00",
                    }],
                ):
                    resp = client.post(
                        "/api/leaderboard",
                        data=json.dumps({
                            "nickname": "<script>alert(1)</script>VERYLONGNAMETHATEXCEEDS20CHARS",
                            "session_id": session_id,
                        }),
                        content_type="application/json",
                    )
    finally:
        tmp_path.unlink(missing_ok=True)

    assert resp.status_code == 200
    # Special chars stripped, max 20 chars
    assert "<" not in captured.get("nickname", "")
    assert len(captured.get("nickname", "")) <= 20


def test_leaderboard_full_flow():
    """End-to-end: score via /api/score then submit to /api/leaderboard.

    Verifies the prediction written by /api/score is found by /api/leaderboard
    and that the response contains rank and total fields.
    """
    client = _client()
    session_id = "lb-e2e-flow-test-001"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        with mock.patch.object(app_module, "PREDICTIONS_LOG", tmp_path):
            # Step 1: score the session
            score_resp = client.post(
                "/api/score",
                data=json.dumps(_score_batch(session_id)),
                content_type="application/json",
            )
            assert score_resp.status_code == 200, (
                f"Expected 200 from /api/score, got {score_resp.status_code}: {score_resp.data}"
            )

            # Step 2: submit to leaderboard using the same session_id
            with mock.patch.object(app_module.db_manager, "save_leaderboard_entry", return_value=1):
                with mock.patch.object(
                    app_module.db_manager,
                    "get_leaderboard",
                    return_value=[{
                        "nickname": "E2ETester",
                        "prob_bot": json.loads(score_resp.data).get("prob_bot", 0.1),
                        "verdict": json.loads(score_resp.data).get("label", "human"),
                        "session_id": session_id,
                        "created_at": "2026-04-04T00:00:00",
                    }],
                ):
                    lb_resp = client.post(
                        "/api/leaderboard",
                        data=json.dumps({"nickname": "E2ETester", "session_id": session_id}),
                        content_type="application/json",
                    )
    finally:
        tmp_path.unlink(missing_ok=True)

    assert lb_resp.status_code == 200, (
        f"Expected 200 from /api/leaderboard, got {lb_resp.status_code}: {lb_resp.data}"
    )
    body = json.loads(lb_resp.data)
    assert "rank" in body
    assert "total" in body
