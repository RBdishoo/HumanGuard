"""
Tests for the automated retraining pipeline:
  - get_unlabeled_session_count()
  - mark_sessions_as_trained()
  - RETRAIN_THRESHOLD constant
  - --check-threshold exit behaviour (via subprocess)
  - GET /api/retrain-status endpoint
"""

import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

# ── Repo root on path ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

import backend.db as db_module
from backend.db import DatabaseManager, RETRAIN_THRESHOLD


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fresh_db():
    """Create an in-memory DatabaseManager for isolation."""
    return DatabaseManager.__new__(DatabaseManager)


def _init_mem_db():
    """Fully initialise an in-memory DatabaseManager and return it."""
    import os
    orig = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    try:
        db = DatabaseManager()
    finally:
        if orig is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = orig
    return db


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGetUnlabeledSessionCount:
    def test_zero_when_no_sessions(self):
        db = _init_mem_db()
        assert db.get_unlabeled_session_count() == 0

    def test_counts_only_untrained_demo_simulator_sessions(self):
        db = _init_mem_db()
        # Insert sessions: 2 demo, 1 simulator, 1 live (should be excluded)
        with db._sqlite_cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (session_id, source) VALUES "
                "('s1','demo'), ('s2','demo'), ('s3','simulator'), ('s4','live')"
            )
        assert db.get_unlabeled_session_count() == 3  # demo×2 + simulator×1

    def test_already_trained_sessions_excluded(self):
        db = _init_mem_db()
        with db._sqlite_cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (session_id, source) VALUES "
                "('s1','demo'), ('s2','demo'), ('s3','simulator')"
            )
            # Mark s1 as trained
            cur.execute(
                "UPDATE sessions SET trained_at = CURRENT_TIMESTAMP WHERE session_id='s1'"
            )
        assert db.get_unlabeled_session_count() == 2  # s2 + s3 only


class TestMarkSessionsAsTrained:
    def test_mark_reduces_untrained_count(self):
        db = _init_mem_db()
        with db._sqlite_cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (session_id, source) VALUES "
                "('a1','demo'), ('a2','demo'), ('a3','simulator')"
            )
        assert db.get_unlabeled_session_count() == 3
        db.mark_sessions_as_trained(["a1", "a2"])
        assert db.get_unlabeled_session_count() == 1

    def test_mark_empty_list_is_noop(self):
        db = _init_mem_db()
        with db._sqlite_cursor() as cur:
            cur.execute("INSERT INTO sessions (session_id, source) VALUES ('b1','demo')")
        db.mark_sessions_as_trained([])  # should not raise
        assert db.get_unlabeled_session_count() == 1


class TestRetrainThreshold:
    def test_threshold_value(self):
        assert RETRAIN_THRESHOLD == 50


class TestCheckThresholdFlag:
    """Test --check-threshold exit code via subprocess using a temp SQLite file."""

    def _run_check_with_db(self, session_count: int) -> int:
        """Create a temp SQLite file, seed it, run --check-threshold, return exit code."""
        import os
        import tempfile
        import sqlite3

        script = str(REPO_ROOT / "scripts" / "retrain.py")
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Seed the temp DB
            conn = sqlite3.connect(db_path)
            conn.execute(
                "CREATE TABLE sessions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_id TEXT UNIQUE NOT NULL, "
                "source TEXT, "
                "trained_at TIMESTAMP DEFAULT NULL, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            for i in range(session_count):
                conn.execute(
                    "INSERT INTO sessions (session_id, source) VALUES (?, 'demo')",
                    (f"thresh_session_{i}",),
                )
            conn.commit()
            conn.close()

            env = os.environ.copy()
            env["DATABASE_URL"] = f"sqlite:///{db_path}"
            result = subprocess.run(
                [sys.executable, script, "--check-threshold"],
                capture_output=True,
                cwd=str(REPO_ROOT),
                env=env,
            )
            return result.returncode
        finally:
            try:
                os.unlink(db_path)
            except Exception:
                pass

    def test_exits_0_when_below_threshold(self):
        assert self._run_check_with_db(session_count=5) == 0

    def test_exits_1_when_at_or_above_threshold(self):
        assert self._run_check_with_db(session_count=50) == 1


class TestRetrainStatusEndpoint:
    def setup_method(self):
        import backend.app as app_module
        from backend.app import app
        self.app_module = app_module
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_retrain_status_returns_200(self):
        resp = self.client.get("/api/retrain-status")
        assert resp.status_code == 200

    def test_retrain_status_shape(self):
        resp = self.client.get("/api/retrain-status")
        data = resp.get_json()
        assert "last_retrain" in data
        assert "sessions_since_retrain" in data
        assert "threshold" in data
        assert "next_check" in data
        assert "champion_version" in data

    def test_retrain_status_threshold_value(self):
        resp = self.client.get("/api/retrain-status")
        data = resp.get_json()
        assert data["threshold"] == 50

    def test_retrain_status_sessions_is_integer(self):
        resp = self.client.get("/api/retrain-status")
        data = resp.get_json()
        assert isinstance(data["sessions_since_retrain"], int)
