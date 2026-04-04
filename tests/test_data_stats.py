"""Tests for GET /api/data-stats endpoint."""
import csv
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module

app_module._server_start_time = 1.0
app_module.app.config["TESTING"] = True


def _client():
    return app_module.app.test_client()


def _make_labels_csv(path: Path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sessionID", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _make_signals_jsonl(path: Path, batches):
    with open(path, "w") as f:
        for b in batches:
            f.write(json.dumps(b) + "\n")


@contextmanager
def _data_files(label_rows, signal_batches):
    """Create temp CSV/JSONL and redirect the endpoint's file paths to them."""
    with tempfile.TemporaryDirectory() as tmpdir:
        labels_csv = Path(tmpdir) / "labels.csv"
        signals_jsonl = Path(tmpdir) / "signals.jsonl"
        _make_labels_csv(labels_csv, label_rows)
        _make_signals_jsonl(signals_jsonl, signal_batches)

        real_open = open

        def patched_open(file, *args, **kwargs):
            s = str(file)
            if s.endswith("labels.csv") and "data" in s:
                return real_open(labels_csv, *args, **kwargs)
            if s.endswith("signals.jsonl") and "data" in s:
                return real_open(signals_jsonl, *args, **kwargs)
            return real_open(file, *args, **kwargs)

        # Patch Path.exists to avoid recursion — use os.path.exists directly
        _labels_str = str(labels_csv)
        _signals_str = str(signals_jsonl)

        orig_exists = Path.exists

        def patched_exists(self):
            s = str(self)
            if s.endswith("labels.csv") and "data" in s:
                return os.path.exists(_labels_str)
            if s.endswith("signals.jsonl") and "data" in s:
                return os.path.exists(_signals_str)
            return orig_exists(self)

        with mock.patch("builtins.open", side_effect=patched_open):
            with mock.patch.object(Path, "exists", patched_exists):
                yield _client()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_data_stats_200_status():
    """Endpoint returns HTTP 200."""
    with _data_files(
        [{"sessionID": "s1", "label": "human", "source": "real_human"}],
        [{"sessionID": "s1", "source": "real_human", "utm_source": "twitter"}],
    ) as client:
        resp = client.get("/api/data-stats")
    assert resp.status_code == 200


def test_data_stats_required_keys():
    """Response JSON contains all required top-level keys."""
    with _data_files([], []) as client:
        resp = client.get("/api/data-stats")
    data = resp.get_json()
    required = {
        "total_labeled_sessions",
        "real_human_sessions",
        "synthetic_human_sessions",
        "synthetic_bot_sessions",
        "real_human_by_source",
        "retrain_readiness",
    }
    assert required.issubset(data.keys()), f"Missing keys: {required - data.keys()}"


def test_data_stats_real_human_sessions_is_int():
    """real_human_sessions is an integer and counts correctly."""
    rows = [
        {"sessionID": "rh1", "label": "human", "source": "real_human"},
        {"sessionID": "rh2", "label": "human", "source": "real_human"},
    ]
    with _data_files(rows, []) as client:
        resp = client.get("/api/data-stats")
    data = resp.get_json()
    assert isinstance(data["real_human_sessions"], int)
    assert data["real_human_sessions"] == 2


def test_data_stats_retrain_readiness_percent_range():
    """retrain_readiness.percent is between 0 and 100 inclusive."""
    rows = [{"sessionID": f"rh{i}", "label": "human", "source": "real_human"} for i in range(30)]
    with _data_files(rows, []) as client:
        resp = client.get("/api/data-stats")
    data = resp.get_json()
    pct = data["retrain_readiness"]["percent"]
    assert isinstance(pct, (int, float))
    assert 0 <= pct <= 100, f"percent={pct} out of [0, 100]"
