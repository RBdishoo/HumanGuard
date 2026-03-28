import sys
import os
import json
from unittest import mock

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module

FEATURE_NAMES = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e", "feat_f"]


def _make_bundle():
    """Build a scoring bundle with a real XGBoost model on dummy data."""
    rng = np.random.RandomState(42)
    X_train = rng.rand(80, len(FEATURE_NAMES))
    y_train = (X_train[:, 0] > 0.5).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=10, max_depth=2, random_state=42, eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    scaler = mock.MagicMock()
    scaler.transform.side_effect = lambda x: x

    import shap
    explainer = shap.TreeExplainer(model)

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "threshold": 0.5,
        "explainer": explainer,
        "extractor": mock.MagicMock(),
    }


def _write_jsonl(tmp_path, session_id, n_batches, feature_values_fn=None):
    """Write n_batches to a temp signals.jsonl for the given session_id."""
    jsonl = tmp_path / "signals.jsonl"
    lines = []
    for i in range(n_batches):
        batch = {
            "sessionID": session_id,
            "timestamp": f"2026-01-01T00:00:{i:02d}Z",
            "metadata": {"userAgent": "test"},
            "signals": {
                "mouseMoves": [{"x": i * 10, "y": i * 10, "ts": 1000 + i * 100}],
                "clicks": [{"ts": 1050 + i * 100, "button": 0}],
                "keys": [{"code": "KeyA", "ts": 1020 + i * 100}],
            },
        }
        lines.append(json.dumps(batch))
    jsonl.write_text("\n".join(lines) + "\n")
    return str(jsonl)


def _client():
    app_module._server_start_time = 1.0
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _post_session_score(client, session_id, bundle, signals_file):
    """Helper to POST /api/session-score with mocked bundle and signals file."""
    with mock.patch("backend.app._load_scoring_bundle", return_value=bundle), \
         mock.patch("db.db_client.is_available", return_value=False), \
         mock.patch.object(app_module.collector, "signalsFile", signals_file):
        # Configure extractor to return different features per call
        call_count = [0]
        rng = np.random.RandomState(99)

        def _fake_extract(signals_dict):
            call_count[0] += 1
            return {name: float(rng.rand()) for name in FEATURE_NAMES}

        bundle["extractor"].extractBatchFeatures.side_effect = _fake_extract

        return client.post(
            "/api/session-score",
            data=json.dumps({"sessionID": session_id}),
            content_type="application/json",
        )


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_valid_session_returns_all_keys(tmp_path):
    session_id = "test-sess-3batch"
    signals_file = _write_jsonl(tmp_path, session_id, 3)
    bundle = _make_bundle()
    resp = _post_session_score(_client(), session_id, bundle, signals_file)

    assert resp.status_code == 200
    data = json.loads(resp.data)
    expected_keys = {"success", "sessionID", "session_prob_bot", "label",
                     "threshold", "batch_count", "batch_scores", "drift"}
    assert expected_keys.issubset(set(data.keys()))
    assert data["batch_count"] == 3


def test_single_batch_returns_400(tmp_path):
    session_id = "test-sess-1batch"
    signals_file = _write_jsonl(tmp_path, session_id, 1)
    bundle = _make_bundle()
    resp = _post_session_score(_client(), session_id, bundle, signals_file)

    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert data["error"] == "Insufficient data"
    assert data["batch_count"] == 1


def test_unknown_session_returns_400(tmp_path):
    # Write batches for a different session
    signals_file = _write_jsonl(tmp_path, "other-session", 5)
    bundle = _make_bundle()
    resp = _post_session_score(_client(), "nonexistent-session", bundle, signals_file)

    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert data["batch_count"] == 0


def test_batch_scores_matches_count(tmp_path):
    session_id = "test-sess-5batch"
    signals_file = _write_jsonl(tmp_path, session_id, 5)
    bundle = _make_bundle()
    resp = _post_session_score(_client(), session_id, bundle, signals_file)

    data = json.loads(resp.data)
    assert isinstance(data["batch_scores"], list)
    assert len(data["batch_scores"]) == data["batch_count"]
    for score in data["batch_scores"]:
        assert isinstance(score, float)


def test_drift_trend_valid(tmp_path):
    session_id = "test-sess-drift"
    signals_file = _write_jsonl(tmp_path, session_id, 4)
    bundle = _make_bundle()
    resp = _post_session_score(_client(), session_id, bundle, signals_file)

    data = json.loads(resp.data)
    assert data["drift"]["trend"] in ("increasing", "decreasing", "stable")


def test_session_prob_bot_between_0_and_1(tmp_path):
    session_id = "test-sess-range"
    signals_file = _write_jsonl(tmp_path, session_id, 4)
    bundle = _make_bundle()
    resp = _post_session_score(_client(), session_id, bundle, signals_file)

    data = json.loads(resp.data)
    assert 0.0 <= data["session_prob_bot"] <= 1.0


def test_weighted_scoring_favors_later_batches(tmp_path):
    """
    When early batches are human-like (low prob) and later batches are bot-like
    (high prob), the weighted session score should exceed the simple average.
    """
    session_id = "test-sess-weighted"
    signals_file = _write_jsonl(tmp_path, session_id, 6)
    bundle = _make_bundle()

    # Make extractor return features that produce increasing bot probability:
    # early batches get feat_a < 0.5 (human), later batches get feat_a > 0.5 (bot)
    call_idx = [0]

    def _escalating_features(signals_dict):
        i = call_idx[0]
        call_idx[0] += 1
        # feat_a is the primary split feature in our dummy model
        feat_a = 0.1 if i < 3 else 0.9
        return {name: (feat_a if name == "feat_a" else 0.5) for name in FEATURE_NAMES}

    bundle["extractor"].extractBatchFeatures.side_effect = _escalating_features

    with mock.patch("backend.app._load_scoring_bundle", return_value=bundle), \
         mock.patch("db.db_client.is_available", return_value=False), \
         mock.patch.object(app_module.collector, "signalsFile", signals_file):
        resp = _client().post(
            "/api/session-score",
            data=json.dumps({"sessionID": session_id}),
            content_type="application/json",
        )

    data = json.loads(resp.data)
    scores = data["batch_scores"]
    simple_avg = sum(scores) / len(scores)
    # Weighted average should be higher because later (bot-like) batches
    # have higher weight
    assert data["session_prob_bot"] > simple_avg


def test_get_alias_works(tmp_path):
    session_id = "test-sess-get"
    signals_file = _write_jsonl(tmp_path, session_id, 3)
    bundle = _make_bundle()

    rng = np.random.RandomState(77)
    bundle["extractor"].extractBatchFeatures.side_effect = lambda s: {
        name: float(rng.rand()) for name in FEATURE_NAMES
    }

    with mock.patch("backend.app._load_scoring_bundle", return_value=bundle), \
         mock.patch("db.db_client.is_available", return_value=False), \
         mock.patch.object(app_module.collector, "signalsFile", signals_file):
        resp = _client().get(f"/api/session-score/{session_id}")

    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["sessionID"] == session_id
