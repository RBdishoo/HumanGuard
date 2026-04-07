import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from utils.helpers import isValidSignalBatch, normalizeSignalBatch, formatTimestamp


def _valid_batch():
    return {
        "session_id": "test-session",
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
    del batch["session_id"]
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


# -------------------------------------------------------------------
# normalizeSignalBatch tests
# -------------------------------------------------------------------

def test_normalize_converts_sessionID_to_session_id():
    """camelCase sessionID from the frontend is renamed to snake_case session_id."""
    batch = {
        "sessionID": "abc-123",
        "signals": {"mouseMoves": [], "clicks": [], "keys": []},
    }
    result = normalizeSignalBatch(batch)
    assert "session_id" in result
    assert result["session_id"] == "abc-123"
    assert "sessionID" not in result


def test_normalize_leaves_session_id_unchanged():
    """A payload already containing session_id is not modified."""
    batch = {
        "session_id": "abc-123",
        "signals": {"mouseMoves": [], "clicks": [], "keys": []},
    }
    result = normalizeSignalBatch(batch)
    assert result["session_id"] == "abc-123"
    assert "sessionID" not in result


# -------------------------------------------------------------------
# isValidSignalBatch — element-type checks
# -------------------------------------------------------------------

def test_mouseMoves_non_dict_element_returns_false():
    """isValidSignalBatch rejects a batch where mouseMoves contains a non-dict element."""
    batch = _valid_batch()
    batch["signals"]["mouseMoves"] = [{"x": 1, "y": 2, "ts": 100}, "not-a-dict"]
    assert isValidSignalBatch(batch) is False


def test_clicks_non_dict_element_returns_false():
    """isValidSignalBatch rejects a batch where clicks contains a non-dict element."""
    batch = _valid_batch()
    batch["signals"]["clicks"] = [42]
    assert isValidSignalBatch(batch) is False


def test_keys_non_dict_element_returns_false():
    """isValidSignalBatch rejects a batch where keys contains a non-dict element."""
    batch = _valid_batch()
    batch["signals"]["keys"] = [None]
    assert isValidSignalBatch(batch) is False


def test_all_dict_elements_accepted():
    """isValidSignalBatch accepts a batch where every event list contains only dicts."""
    batch = _valid_batch()
    batch["signals"]["mouseMoves"] = [{"x": 1, "y": 2, "ts": 100}]
    batch["signals"]["clicks"] = [{"ts": 200, "button": 0}]
    batch["signals"]["keys"] = [{"code": "KeyA", "ts": 300}]
    assert isValidSignalBatch(batch) is True
