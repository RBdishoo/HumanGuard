import sys
import os
import json
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def _reset_db_client():
    """Reset db_client cached state between tests."""
    from backend.db import db_client
    db_client.reset()
    return db_client


# -------------------------------------------------------------------
# is_available() tests
# -------------------------------------------------------------------

def test_is_available_false_when_no_database_url():
    """is_available() returns False when DATABASE_URL is not set."""
    db_client = _reset_db_client()
    with mock.patch.dict(os.environ, {}, clear=True):
        # Remove DATABASE_URL if it happens to be set
        os.environ.pop("DATABASE_URL", None)
        db_client.reset()
        assert db_client.is_available() is False


def test_is_available_false_when_connection_fails():
    """is_available() returns False when psycopg2 connection fails."""
    db_client = _reset_db_client()
    with mock.patch.dict(os.environ, {"DATABASE_URL": "postgres://fake:5432/fake"}):
        db_client.reset()
        # Mock _get_pool to raise, simulating a connection failure
        with mock.patch.object(db_client, "_get_pool", side_effect=Exception("conn refused")):
            assert db_client.is_available() is False


# -------------------------------------------------------------------
# signal_collector dual-write tests
# -------------------------------------------------------------------

def _make_batch():
    return {
        "sessionID": "test-session-1",
        "metadata": {"userAgent": "test", "viewportWidth": 1920, "viewportHeight": 1080},
        "signals": {"mouseMoves": [], "clicks": [], "keys": []},
    }


def test_signal_collector_calls_db_when_available(tmp_path):
    """signal_collector dual-write calls DB when available."""
    db_client = _reset_db_client()

    from backend.collectors.signal_collector import SignalCollector

    collector = SignalCollector()
    collector.signalsFile = str(tmp_path / "signals.jsonl")
    (tmp_path / "signals.jsonl").touch()

    batch = _make_batch()

    with mock.patch("db.db_client.is_available", return_value=True) as mock_avail, \
         mock.patch("db.db_client.save_signal_batch") as mock_save:
        result = collector.saveSignalBatch(batch)

    assert result is True
    mock_avail.assert_called_once()
    mock_save.assert_called_once()


def test_signal_collector_continues_when_db_unavailable(tmp_path):
    """signal_collector continues normally when DB is unavailable."""
    db_client = _reset_db_client()

    from backend.collectors.signal_collector import SignalCollector

    collector = SignalCollector()
    collector.signalsFile = str(tmp_path / "signals.jsonl")
    (tmp_path / "signals.jsonl").touch()

    batch = _make_batch()

    with mock.patch("db.db_client.is_available", return_value=False) as mock_avail, \
         mock.patch("db.db_client.save_signal_batch") as mock_save:
        result = collector.saveSignalBatch(batch)

    assert result is True
    mock_save.assert_not_called()

    # JSONL file should still have the batch
    lines = (tmp_path / "signals.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_signal_collector_continues_when_db_write_raises(tmp_path):
    """signal_collector returns True even when DB write throws."""
    db_client = _reset_db_client()

    from backend.collectors.signal_collector import SignalCollector

    collector = SignalCollector()
    collector.signalsFile = str(tmp_path / "signals.jsonl")
    (tmp_path / "signals.jsonl").touch()

    batch = _make_batch()

    with mock.patch("db.db_client.is_available", return_value=True), \
         mock.patch("db.db_client.save_signal_batch", side_effect=Exception("DB down")):
        result = collector.saveSignalBatch(batch)

    assert result is True


# -------------------------------------------------------------------
# /api/score prediction saving test
# -------------------------------------------------------------------

def test_prediction_saved_after_score_when_db_available():
    """predictions are saved after /api/score when DB is available."""
    import backend.app as app_module

    app_module._server_start_time = 1.0
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    # Build a fake scoring bundle so /api/score doesn't need real model files
    fake_model = mock.MagicMock()
    fake_model.predict_proba.return_value = __import__("numpy").array([[0.3, 0.7]])

    fake_scaler = mock.MagicMock()
    fake_scaler.transform.side_effect = lambda x: x

    fake_extractor = mock.MagicMock()
    fake_extractor.extractBatchFeatures.return_value = {}

    fake_bundle = {
        "model": fake_model,
        "scaler": fake_scaler,
        "feature_names": ["feat1", "feat2"],
        "threshold": 0.5,
        "extractor": fake_extractor,
    }

    payload = _make_batch()

    with mock.patch("backend.app._load_scoring_bundle", return_value=fake_bundle), \
         mock.patch.object(app_module.db_manager, "save_prediction") as mock_save_pred:
        resp = client.post(
            "/api/score",
            data=json.dumps(payload),
            content_type="application/json",
        )

    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["label"] == "bot"
    mock_save_pred.assert_called_once_with(
        "test-session-1", mock.ANY, True,
        threshold=0.5, scoring_type="batch", source=mock.ANY, api_key=mock.ANY,
    )


# -------------------------------------------------------------------
# DatabaseManager tests (SQLite in-memory backend)
# -------------------------------------------------------------------

def _sqlite_mgr():
    """Instantiate a fresh in-memory DatabaseManager."""
    with mock.patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}):
        from backend.db import DatabaseManager
        return DatabaseManager()


def test_db_manager_empty_stats():
    """get_stats() returns zeros on an empty database."""
    mgr = _sqlite_mgr()
    stats = mgr.get_stats()
    assert stats["total_predictions"] == 0
    assert stats["bot_count"] == 0
    assert stats["human_count"] == 0
    assert stats["bot_rate"] == 0.0


def test_db_manager_save_prediction_bot():
    """save_prediction stores a bot prediction and get_stats reflects it."""
    mgr = _sqlite_mgr()
    mgr.save_prediction("session-abc", 0.9, is_bot=True)
    stats = mgr.get_stats()
    assert stats["total_predictions"] == 1
    assert stats["bot_count"] == 1
    assert stats["human_count"] == 0
    assert stats["bot_rate"] == 1.0


def test_db_manager_save_prediction_human():
    """save_prediction stores a human prediction correctly."""
    mgr = _sqlite_mgr()
    mgr.save_prediction("session-xyz", 0.1, is_bot=False)
    stats = mgr.get_stats()
    assert stats["total_predictions"] == 1
    assert stats["bot_count"] == 0
    assert stats["human_count"] == 1
    assert stats["bot_rate"] == 0.0


def test_db_manager_multiple_predictions_stats():
    """Stats are correct after multiple mixed predictions."""
    mgr = _sqlite_mgr()
    mgr.save_prediction("s1", 0.9, is_bot=True)
    mgr.save_prediction("s2", 0.8, is_bot=True)
    mgr.save_prediction("s3", 0.2, is_bot=False)
    stats = mgr.get_stats()
    assert stats["total_predictions"] == 3
    assert stats["bot_count"] == 2
    assert stats["human_count"] == 1
    assert abs(stats["bot_rate"] - round(2 / 3, 4)) < 1e-6


def test_db_manager_get_recent_predictions_order():
    """get_recent_predictions returns newest first."""
    mgr = _sqlite_mgr()
    mgr.save_prediction("s1", 0.9, is_bot=True)
    mgr.save_prediction("s2", 0.1, is_bot=False)
    preds = mgr.get_recent_predictions(limit=10)
    assert len(preds) == 2
    # Most recent should be "s2" (inserted last)
    assert preds[0]["session_id"] == "s2"
    assert preds[1]["session_id"] == "s1"


def test_db_manager_get_recent_predictions_limit():
    """get_recent_predictions respects the limit parameter."""
    mgr = _sqlite_mgr()
    for i in range(5):
        mgr.save_prediction(f"s{i}", 0.5, is_bot=True)
    preds = mgr.get_recent_predictions(limit=3)
    assert len(preds) == 3


def test_db_manager_save_session_round_trip():
    """save_session persists a signal batch without errors."""
    mgr = _sqlite_mgr()
    batch = {
        "sessionID": "sess-round-trip",
        "metadata": {"userAgent": "test-ua", "viewportWidth": 1920, "viewportHeight": 1080},
        "signals": {"mouseMoves": [{"x": 1, "y": 2, "ts": 100}], "clicks": [], "keys": []},
        "timestamp": "2024-01-01T00:00:00Z",
    }
    # Should not raise
    mgr.save_session(batch)
    # Calling again for same session should not raise (ON CONFLICT DO NOTHING)
    mgr.save_session(batch)


def test_db_manager_get_recent_predictions_fields():
    """get_recent_predictions rows contain expected fields."""
    mgr = _sqlite_mgr()
    mgr.save_prediction("sess-fields", 0.75, is_bot=True, threshold=0.5, scoring_type="batch")
    preds = mgr.get_recent_predictions(limit=1)
    assert len(preds) == 1
    row = preds[0]
    assert row["session_id"] == "sess-fields"
    assert abs(row["prob_bot"] - 0.75) < 1e-6
    assert row["label"] == "bot"
    assert abs(row["threshold"] - 0.5) < 1e-6
    assert row["scoring_type"] == "batch"


def test_db_manager_bad_postgres_url_no_crash():
    """DatabaseManager with an unreachable postgres URL does not crash on save_prediction."""
    with mock.patch.dict(os.environ, {"DATABASE_URL": "postgresql://bad:5432/nope"}):
        from backend.db import DatabaseManager
        mgr = DatabaseManager()
        # The pool is lazy; pg operations fail gracefully with a warning log
        # This should not raise regardless of whether psycopg2 is installed
        try:
            mgr.save_prediction("s", 0.5, is_bot=True)
        except Exception as exc:
            raise AssertionError(f"save_prediction raised unexpectedly: {exc}") from exc


# -------------------------------------------------------------------
# _resolve_db_kwargs() tests
# -------------------------------------------------------------------

def test_resolve_db_kwargs_secrets_manager_has_sslmode_require():
    """_resolve_db_kwargs() returns sslmode='require' when built from Secrets Manager."""
    db_client = _reset_db_client()
    secret_payload = json.dumps({
        "host": "mydb.rds.amazonaws.com",
        "port": 5432,
        "dbname": "humanguard",
        "username": "admin",
        "password": "s3cr3t",
    })
    fake_sm = mock.MagicMock()
    fake_sm.get_secret_value.return_value = {"SecretString": secret_payload}

    env = {"RDS_SECRET_NAME": "humanGuard/rds", "AWS_REGION": "us-east-1"}
    with mock.patch.dict(os.environ, env, clear=True):
        # Remove DATABASE_URL so the Secrets Manager path is taken
        os.environ.pop("DATABASE_URL", None)
        db_client.reset()
        import boto3
        with mock.patch.object(boto3, "client", return_value=fake_sm):
            kwargs = db_client._resolve_db_kwargs()

    assert kwargs["sslmode"] == "require"
    assert kwargs["host"] == "mydb.rds.amazonaws.com"
    assert kwargs["dbname"] == "humanguard"
    assert kwargs["user"] == "admin"
    assert kwargs["password"] == "s3cr3t"


def test_resolve_db_kwargs_sqlite_url_has_sslmode_disable():
    """_resolve_db_kwargs() returns sslmode='disable' for a sqlite:/// DATABASE_URL."""
    db_client = _reset_db_client()
    with mock.patch.dict(os.environ, {"DATABASE_URL": "sqlite:///./test.db"}):
        db_client.reset()
        kwargs = db_client._resolve_db_kwargs()

    assert kwargs["sslmode"] == "disable"


def test_resolve_db_kwargs_postgres_url_has_sslmode_require():
    """_resolve_db_kwargs() returns sslmode='require' for a postgres:// DATABASE_URL."""
    db_client = _reset_db_client()
    with mock.patch.dict(os.environ, {"DATABASE_URL": "postgres://user:pass@host:5432/db"}):
        db_client.reset()
        kwargs = db_client._resolve_db_kwargs()

    assert kwargs["sslmode"] == "require"
    assert kwargs["host"] == "host"
    assert kwargs["dbname"] == "db"
    assert kwargs["user"] == "user"
    assert kwargs["password"] == "pass"
