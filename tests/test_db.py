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
         mock.patch("db.db_client.is_available", return_value=True) as mock_avail, \
         mock.patch("db.db_client.save_prediction") as mock_save_pred:
        resp = client.post(
            "/api/score",
            data=json.dumps(payload),
            content_type="application/json",
        )

    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["label"] == "bot"
    mock_save_pred.assert_called_once_with("test-session-1", mock.ANY, "bot", 0.5)
