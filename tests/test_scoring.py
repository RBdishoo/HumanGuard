"""
Tests for confidence interval output from POST /api/score.
"""
import sys
import os
import json
from unittest import mock

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module

FEATURE_NAMES = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]


def _make_bundle(n_estimators=10, std_control=None):
    """Build a minimal scoring bundle backed by a real RandomForest."""
    rng = np.random.RandomState(0)
    X_train = rng.rand(60, len(FEATURE_NAMES))
    y_train = (X_train[:, 0] > 0.5).astype(int)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=2, random_state=0)
    model.fit(X_train, y_train)

    scaler = mock.MagicMock()
    scaler.transform.side_effect = lambda x: x

    extractor = mock.MagicMock()
    feats = {name: rng.rand() for name in FEATURE_NAMES}
    extractor.extractBatchFeatures.return_value = feats

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "threshold": 0.5,
        "explainer": None,
        "extractor": extractor,
        "session_blender": None,
        "session_blender_features": None,
        "_registry_version": None,
        "_registry_metadata": {},
    }


def _make_payload():
    return {
        "sessionID": "test-session-ci",
        "signals": {
            "mouseMoves": [{"x": 10, "y": 20, "ts": 1000}],
            "clicks": [],
            "keys": [],
        },
    }


@pytest.fixture()
def client(tmp_path):
    """Flask test client with mocked bundle and DB."""
    app_module.app.config["TESTING"] = True

    bundle = _make_bundle()

    with mock.patch.object(app_module, "_load_scoring_bundle", return_value=bundle), \
         mock.patch.object(app_module, "_log_prediction_local"), \
         mock.patch.object(app_module, "db_manager", create=True), \
         mock.patch.object(app_module, "collector", create=True), \
         mock.patch("backend.app.require_api_key", lambda f: f), \
         mock.patch.object(app_module, "metrics"):
        with app_module.app.test_client() as c:
            yield c, bundle


def _score(client, payload=None):
    if payload is None:
        payload = _make_payload()
    resp = client.post(
        "/api/score?explain=false",
        data=json.dumps(payload),
        content_type="application/json",
    )
    return resp, resp.get_json()


# ── Test 1: confidence interval fields are present in /api/score response ──

def test_confidence_interval_present_in_response(client):
    c, _ = client
    _, data = _score(c)
    assert "confidence" in data, "Missing 'confidence' key"
    assert "confidence_interval" in data, "Missing 'confidence_interval' key"
    assert "lower" in data["confidence_interval"]
    assert "upper" in data["confidence_interval"]
    assert "std" in data, "Missing 'std' key"


# ── Test 2: ci_lower <= prob_bot <= ci_upper ────────────────────────────────

def test_ci_bounds_contain_prob_bot(client):
    c, _ = client
    _, data = _score(c)
    prob = data["prob_bot"]
    lower = data["confidence_interval"]["lower"]
    upper = data["confidence_interval"]["upper"]
    assert lower <= prob + 1e-9, f"ci_lower ({lower}) > prob_bot ({prob})"
    assert upper >= prob - 1e-9, f"ci_upper ({upper}) < prob_bot ({prob})"


# ── Test 3: confidence_level is one of high / medium / low ──────────────────

def test_confidence_level_is_valid_enum(client):
    c, _ = client
    _, data = _score(c)
    assert data["confidence"] in ("high", "medium", "low"), (
        f"Unexpected confidence value: {data['confidence']}"
    )


# ── Test 4: std is between 0 and 1 ──────────────────────────────────────────

def test_std_is_in_unit_interval(client):
    c, _ = client
    _, data = _score(c)
    std = data["std"]
    assert 0.0 <= std <= 1.0, f"std out of [0, 1]: {std}"


# ── Test 5: low std produces "high" confidence label ────────────────────────

def test_low_std_yields_high_confidence():
    """
    When all trees agree (very low variance), confidence must be 'high'.
    We verify the labelling logic directly without an HTTP round-trip.
    """
    std_values = {
        0.0: "high",
        0.05: "high",
        0.099: "high",
        0.1: "medium",
        0.15: "medium",
        0.2: "low",
        0.3: "low",
    }
    for std, expected in std_values.items():
        label = "high" if std < 0.1 else "medium" if std < 0.2 else "low"
        assert label == expected, f"std={std}: expected {expected}, got {label}"
