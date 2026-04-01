import sys
import os
import json
from unittest import mock

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import backend.app as app_module

FEATURE_NAMES = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e", "feat_f"]


def _make_bundle(with_explainer=True):
    """Build a scoring bundle with a real RandomForest model trained on dummy data."""
    rng = np.random.RandomState(42)
    X_train = rng.rand(50, len(FEATURE_NAMES))
    y_train = (X_train[:, 0] > 0.5).astype(int)

    model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X_train, y_train)

    scaler = mock.MagicMock()
    scaler.transform.side_effect = lambda x: x

    extractor = mock.MagicMock()
    extractor.extractBatchFeatures.return_value = {
        name: float(rng.rand()) for name in FEATURE_NAMES
    }

    explainer = None
    if with_explainer:
        import shap
        explainer = shap.TreeExplainer(model)

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "threshold": 0.5,
        "extractor": extractor,
        "explainer": explainer,
    }


def _payload():
    return {
        "sessionID": "shap-test-session",
        "metadata": {"userAgent": "test", "viewportWidth": 1920, "viewportHeight": 1080},
        "signals": {"mouseMoves": [{"x": 10, "y": 20, "ts": 1000}], "clicks": [], "keys": []},
    }


def _client():
    app_module._server_start_time = 1.0
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _score(client, bundle, query_string=""):
    with mock.patch("backend.app._load_scoring_bundle", return_value=bundle), \
         mock.patch("db.db_client.is_available", return_value=False):
        return client.post(
            f"/api/score{query_string}",
            data=json.dumps(_payload()),
            content_type="application/json",
        )


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

def test_score_contains_explanation_by_default():
    bundle = _make_bundle(with_explainer=True)
    resp = _score(_client(), bundle)
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "explanation" in data


def test_explanation_has_top_features_max_5():
    bundle = _make_bundle(with_explainer=True)
    resp = _score(_client(), bundle)
    data = json.loads(resp.data)
    top = data["explanation"]["top_features"]
    assert isinstance(top, list)
    assert 1 <= len(top) <= 5


def test_top_feature_has_required_keys():
    bundle = _make_bundle(with_explainer=True)
    resp = _score(_client(), bundle)
    data = json.loads(resp.data)
    for entry in data["explanation"]["top_features"]:
        assert "feature" in entry
        assert "contribution" in entry
        assert isinstance(entry["contribution"], float)


def test_explanation_has_interpretation():
    bundle = _make_bundle(with_explainer=True)
    resp = _score(_client(), bundle)
    data = json.loads(resp.data)
    assert "interpretation" in data["explanation"]
    assert isinstance(data["explanation"]["interpretation"], str)
    assert len(data["explanation"]["interpretation"]) > 10


def test_explain_false_skips_explanation():
    bundle = _make_bundle(with_explainer=True)
    resp = _score(_client(), bundle, query_string="?explain=false")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "explanation" not in data
    assert "prob_bot" in data


def test_missing_explainer_returns_no_explanation():
    bundle = _make_bundle(with_explainer=False)
    assert bundle["explainer"] is None
    resp = _score(_client(), bundle)
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "explanation" not in data
    assert data["success"] is True
    assert "prob_bot" in data
