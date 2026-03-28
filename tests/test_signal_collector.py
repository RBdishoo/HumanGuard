import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from unittest import mock
from backend.collectors.signal_collector import SignalCollector


def _make_collector(tmp_path):
    collector = SignalCollector()
    jsonl = tmp_path / "signals.jsonl"
    jsonl.touch()
    collector.signalsFile = str(jsonl)
    return collector, jsonl


def _batch(session_id="sess-1"):
    return {
        "sessionID": session_id,
        "metadata": {"userAgent": "test"},
        "signals": {"mouseMoves": [], "clicks": [], "keys": []},
    }


# -------------------------------------------------------------------
# saveSignalBatch tests
# -------------------------------------------------------------------

def test_save_writes_to_jsonl(tmp_path):
    collector, jsonl = _make_collector(tmp_path)
    with mock.patch("db.db_client.is_available", return_value=False):
        collector.saveSignalBatch(_batch())
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["sessionID"] == "sess-1"


def test_save_returns_true(tmp_path):
    collector, _ = _make_collector(tmp_path)
    with mock.patch("db.db_client.is_available", return_value=False):
        assert collector.saveSignalBatch(_batch()) is True


# -------------------------------------------------------------------
# getBatchCount tests
# -------------------------------------------------------------------

def test_batch_count_after_saves(tmp_path):
    collector, _ = _make_collector(tmp_path)
    with mock.patch("db.db_client.is_available", return_value=False):
        collector.saveSignalBatch(_batch("s1"))
        collector.saveSignalBatch(_batch("s2"))
        collector.saveSignalBatch(_batch("s3"))
    assert collector.getBatchCount() == 3


# -------------------------------------------------------------------
# getSessionCount tests
# -------------------------------------------------------------------

def test_session_count_unique(tmp_path):
    collector, _ = _make_collector(tmp_path)
    with mock.patch("db.db_client.is_available", return_value=False):
        collector.saveSignalBatch(_batch("sess-a"))
        collector.saveSignalBatch(_batch("sess-b"))
        collector.saveSignalBatch(_batch("sess-c"))
    assert collector.getSessionCount() == 3


def test_same_session_counted_once(tmp_path):
    collector, _ = _make_collector(tmp_path)
    with mock.patch("db.db_client.is_available", return_value=False):
        collector.saveSignalBatch(_batch("sess-dup"))
        collector.saveSignalBatch(_batch("sess-dup"))
        collector.saveSignalBatch(_batch("sess-dup"))
    assert collector.getSessionCount() == 1
