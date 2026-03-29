"""
Unit tests for ModelRegistry.

All tests use an in-memory mock S3 client — no real AWS calls.
"""

import io
import json
import sys
import os
from unittest import mock

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from backend.model_registry import ModelRegistry, RegistryError


# ── Mock S3 client ─────────────────────────────────────────────────────────────

class _FakePageIterator:
    """Minimal paginator that returns a single page from the store."""
    def __init__(self, store: dict, bucket: str, prefix: str):
        self._store = store
        self._bucket = bucket
        self._prefix = prefix

    def __iter__(self):
        common_prefixes = set()
        for (b, k) in self._store:
            if b == self._bucket and k.startswith(self._prefix):
                rest = k[len(self._prefix):]
                parts = rest.split("/")
                if len(parts) >= 2:
                    common_prefixes.add(self._prefix + parts[0] + "/")
        yield {"CommonPrefixes": [{"Prefix": p} for p in sorted(common_prefixes)]}


class _FakePaginator:
    def __init__(self, store, operation_name):
        self._store = store
        self._op = operation_name

    def paginate(self, Bucket, Prefix, Delimiter="/"):
        return _FakePageIterator(self._store, Bucket, Prefix)


class _MockS3:
    """Simple in-memory S3 store."""
    def __init__(self):
        self._store: dict = {}  # (bucket, key) → bytes

    def put_object(self, Bucket, Key, Body, **kwargs):
        if isinstance(Body, (bytes, bytearray)):
            self._store[(Bucket, Key)] = bytes(Body)
        else:
            self._store[(Bucket, Key)] = Body.encode() if isinstance(Body, str) else bytes(Body)

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self._store:
            # Use a plain KeyError so tests don't require botocore to be installed
            raise KeyError(f"NoSuchKey: {Bucket}/{Key}")
        return {"Body": io.BytesIO(self._store[(Bucket, Key)])}

    def get_paginator(self, operation_name):
        return _FakePaginator(self._store, operation_name)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def s3():
    return _MockS3()


@pytest.fixture
def registry(s3):
    return ModelRegistry(bucket="test-bucket", _s3_client=s3)


def _tiny_model():
    """Return a fitted LogisticRegression (tiny, fast to serialize)."""
    X = np.array([[0.1, 0.2], [0.9, 0.8], [0.2, 0.1], [0.8, 0.9]])
    y = np.array([0, 1, 0, 1])
    clf = LogisticRegression(random_state=42, max_iter=200)
    clf.fit(X, y)
    return clf


def _tiny_scaler():
    X = np.array([[0.1, 0.2], [0.9, 0.8]])
    sc = StandardScaler()
    sc.fit(X)
    return sc


def _sample_metadata(**overrides):
    base = {
        "accuracy": 0.92,
        "precision": 0.91,
        "recall": 0.90,
        "f1": 0.905,
        "roc_auc": 0.95,
        "training_date": "2026-03-29T00:00:00+00:00",
        "training_samples": 500,
        "model_type": "XGBoost",
    }
    base.update(overrides)
    return base


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_registry_push_returns_version(registry):
    """push() stores artifacts in S3 and returns a semantic version string."""
    model   = _tiny_model()
    scaler  = _tiny_scaler()
    fn      = ["feat_a", "feat_b"]
    version = registry.push(model, scaler, fn, threshold=0.5,
                            metadata=_sample_metadata())
    assert version == "v1.0.0"
    # Re-push → minor version incremented
    version2 = registry.push(model, scaler, fn, threshold=0.5,
                              metadata=_sample_metadata(accuracy=0.94))
    assert version2 == "v1.1.0"


def test_registry_load_latest_resolves_champion(registry):
    """load('latest') returns the promoted version's bundle after promote()."""
    model  = _tiny_model()
    scaler = _tiny_scaler()
    fn     = ["feat_a", "feat_b"]

    version = registry.push(model, scaler, fn, threshold=0.5,
                             metadata=_sample_metadata())
    registry.promote(version)

    bundle = registry.load("latest")
    assert bundle["version"] == version
    assert bundle["threshold"] == 0.5
    assert bundle["feature_names"] == fn
    assert bundle["metadata"]["accuracy"] == pytest.approx(0.92)
    # Model round-trips correctly
    X_test = np.array([[0.5, 0.5]])
    proba = bundle["model"].predict_proba(X_test)
    assert proba.shape == (1, 2)


def test_registry_list_versions_returns_sorted(registry):
    """list_versions() returns all pushed versions in ascending semver order."""
    model  = _tiny_model()
    scaler = _tiny_scaler()
    fn     = ["feat_a", "feat_b"]

    assert registry.list_versions() == []

    v1 = registry.push(model, scaler, fn, 0.5, _sample_metadata())
    v2 = registry.push(model, scaler, fn, 0.5, _sample_metadata())
    v3 = registry.push(model, scaler, fn, 0.5, _sample_metadata())

    versions = registry.list_versions()
    assert versions == ["v1.0.0", "v1.1.0", "v1.2.0"]
    assert versions == sorted(versions, key=ModelRegistry._parse_version)


def test_registry_promote_updates_champion(registry):
    """promote() writes champion.json and updates metadata champion flag."""
    model  = _tiny_model()
    scaler = _tiny_scaler()
    fn     = ["f1", "f2"]

    v1 = registry.push(model, scaler, fn, 0.5, _sample_metadata(accuracy=0.88))
    v2 = registry.push(model, scaler, fn, 0.5, _sample_metadata(accuracy=0.95))

    # Before any promotion
    assert registry.get_champion() is None

    registry.promote(v1)
    assert registry.get_champion() == v1

    registry.promote(v2)
    assert registry.get_champion() == v2

    # v1 should now have champion=False in its metadata
    meta_v1 = registry.get_metadata(v1)
    assert meta_v1["champion"] is False

    # v2 should have champion=True
    meta_v2 = registry.get_metadata(v2)
    assert meta_v2["champion"] is True


def test_registry_rollback_promotes_previous(registry):
    """rollback() reinstates the previous champion; raises when no predecessor exists."""
    model  = _tiny_model()
    scaler = _tiny_scaler()
    fn     = ["f1", "f2"]

    v1 = registry.push(model, scaler, fn, 0.5, _sample_metadata(accuracy=0.88))
    v2 = registry.push(model, scaler, fn, 0.5, _sample_metadata(accuracy=0.94))

    # Promote v1 as the very first champion (previous_version=None in champion.json)
    registry.promote(v1)
    assert registry.get_champion() == v1

    # Rollback with no prior champion recorded should raise RegistryError
    with pytest.raises(RegistryError, match="No previous champion"):
        registry.rollback()

    # Now promote v2 (predecessor is v1) and verify rollback works
    registry.promote(v2)
    assert registry.get_champion() == v2

    rolled = registry.rollback()
    assert rolled == v1
    assert registry.get_champion() == v1
