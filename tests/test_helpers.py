import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from utils.helpers import isValidSignalBatch, formatTimestamp


def _valid_batch():
    return {
        "sessionID": "test-session",
        "metadata": {"userAgent": "test"},
        "signals": {"mouseMoves": [], "clicks": [], "keys": []},
    }


# -------------------------------------------------------------------
# isValidSignalBatch tests
# -------------------------------------------------------------------

def test_valid_batch_returns_true():
    assert isValidSignalBatch(_valid_batch()) is True


def test_missing_session_id_returns_false():
    batch = _valid_batch()
    del batch["sessionID"]
    assert isValidSignalBatch(batch) is False


def test_missing_signals_key_returns_false():
    batch = _valid_batch()
    del batch["signals"]
    assert isValidSignalBatch(batch) is False


def test_wrong_signals_type_returns_false():
    batch = _valid_batch()
    batch["signals"]["mouseMoves"] = "not a list"
    assert isValidSignalBatch(batch) is False


def test_empty_dict_returns_false():
    assert isValidSignalBatch({}) is False


# -------------------------------------------------------------------
# formatTimestamp tests
# -------------------------------------------------------------------

def test_format_timestamp_is_iso8601():
    ts = formatTimestamp()
    # Strip trailing Z for fromisoformat compatibility
    parsed = datetime.fromisoformat(ts.rstrip("Z"))
    assert parsed is not None


def test_format_timestamp_contains_utc_indicator():
    ts = formatTimestamp()
    assert ts.endswith("Z")
