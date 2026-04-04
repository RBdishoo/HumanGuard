"""
Tests for backend/enrichment.py — network/device feature extraction.

8 tests covering:
  1. Headless UA detected correctly
  2. Known bot UA detected
  3. Clean browser UA passes
  4. Datacenter IP flagged (mock ipinfo.io response)
  5. Missing Accept-Language flagged
  6. Network features present in /api/score response
  7. ipinfo.io failure returns safe defaults
  8. Feature vector has 37 features (30 behavioural + 7 network)
"""

import sys
import os
import json
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from enrichment import parse_user_agent, get_ip_info, parse_request_headers
from features.feature_extractor import FeatureExtractor


# ── helpers ───────────────────────────────────────────────────────────────

def _minimal_batch():
    return {
        "mouseMoves": [
            {"x": 100, "y": 200, "ts": 1000},
            {"x": 120, "y": 210, "ts": 1100},
            {"x": 140, "y": 220, "ts": 1300},
            {"x": 180, "y": 250, "ts": 1500},
        ],
        "clicks": [{"ts": 1050, "button": 0}, {"ts": 1600, "button": 0}],
        "keys": [
            {"code": "KeyH", "ts": 1020},
            {"code": "KeyE", "ts": 1080},
            {"code": "KeyL", "ts": 1160},
            {"code": "KeyL", "ts": 1230},
            {"code": "KeyO", "ts": 1350},
        ],
    }


# ── Test 1: headless UA detected correctly ────────────────────────────────

@pytest.mark.parametrize("ua", [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 HeadlessChrome/114",
    "Mozilla/5.0 (compatible; Puppeteer/19.0)",
    "Mozilla/5.0 Playwright/1.30",
    "PhantomJS/2.1.1",
    "Selenium/4.0",
])
def test_headless_ua_detected(ua):
    result = parse_user_agent(ua)
    assert result["is_headless_browser"] is True, f"Expected headless for UA: {ua}"


# ── Test 2: known bot UA detected ─────────────────────────────────────────

@pytest.mark.parametrize("ua", [
    "python-requests/2.28.0",
    "curl/7.88.1",
    "wget/1.21",
    "Scrapy/2.8",
])
def test_known_bot_ua_detected(ua):
    result = parse_user_agent(ua)
    assert result["is_known_bot_ua"] is True, f"Expected bot for UA: {ua}"


# ── Test 3: clean browser UA passes ──────────────────────────────────────

def test_clean_browser_ua_passes():
    ua = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
    result = parse_user_agent(ua)
    assert result["is_headless_browser"] is False
    assert result["is_known_bot_ua"] is False
    assert result["browser_type"] == "chrome"
    assert result["os_type"] == "mac"
    assert result["ua_entropy"] > 0


# ── Test 4: datacenter IP flagged (mocked ipinfo.io) ─────────────────────

def test_datacenter_ip_flagged():
    fake_response = json.dumps({
        "ip": "3.80.0.1",
        "org": "AS16509 Amazon.com, Inc.",
        "country": "US",
    }).encode("utf-8")

    class _FakeResp:
        def read(self): return fake_response
        def __enter__(self): return self
        def __exit__(self, *a): pass

    import enrichment as enr
    # Clear cache to force fresh lookup
    enr._ip_cache.clear()

    with mock.patch("urllib.request.urlopen", return_value=_FakeResp()):
        result = get_ip_info("3.80.0.1")

    assert result["is_datacenter"] is True
    assert result["country"] == "US"


# ── Test 5: missing Accept-Language flagged ──────────────────────────────

def test_missing_accept_language_flagged():
    # No headers at all
    result = parse_request_headers({})
    assert result["has_accept_language"] is False
    assert result["accept_language_count"] == 0
    # Missing expected headers → suspicious_header_count should be > 0
    assert result["suspicious_header_count"] > 0


def test_present_accept_language_parsed():
    headers = {
        "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
        "Accept": "text/html",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "Mozilla/5.0",
    }
    result = parse_request_headers(headers)
    assert result["has_accept_language"] is True
    assert result["accept_language_count"] == 3


# ── Test 6: network_signals present in /api/score response ───────────────

def test_network_signals_in_score_response():
    import backend.app as app_module

    app_module.app.config["TESTING"] = True

    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    rng = np.random.RandomState(1)
    feat_names = ["feat_a", "feat_b", "feat_c"]
    X = rng.rand(40, 3)
    y = (X[:, 0] > 0.5).astype(int)
    rf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=1)
    rf.fit(X, y)

    scaler_mock = mock.MagicMock()
    scaler_mock.transform.side_effect = lambda x: x

    extractor_mock = mock.MagicMock()
    extractor_mock.extractBatchFeatures.return_value = {n: rng.rand() for n in feat_names}
    extractor_mock.extract_network_features.return_value = {}

    bundle = {
        "model": rf,
        "scaler": scaler_mock,
        "feature_names": feat_names,
        "threshold": 0.5,
        "explainer": None,
        "extractor": extractor_mock,
        "session_blender": None,
        "session_blender_features": None,
        "_registry_version": None,
        "_registry_metadata": {},
    }

    fake_network_info = {
        "is_headless_browser": False,
        "is_known_bot_ua": False,
        "browser_type": "chrome",
        "os_type": "mac",
        "ua_entropy": 85.0,
        "is_datacenter_ip": False,
        "is_vpn": False,
        "country": "US",
        "org": "Comcast",
        "has_accept_language": True,
        "has_referer": False,
        "accept_language_count": 3,
        "suspicious_header_count": 0,
    }

    payload = {
        "sessionID": "enrichment-test",
        "signals": _minimal_batch(),
    }

    with mock.patch.object(app_module, "_load_scoring_bundle", return_value=bundle), \
         mock.patch.object(app_module, "_log_prediction_local"), \
         mock.patch.object(app_module, "db_manager", create=True), \
         mock.patch.object(app_module, "collector", create=True), \
         mock.patch.object(app_module, "metrics"), \
         mock.patch("backend.app.enrich_request", return_value=fake_network_info):
        with app_module.app.test_client() as c:
            resp = c.post(
                "/api/score?explain=false",
                data=json.dumps(payload),
                content_type="application/json",
            )

    assert resp.status_code == 200
    data = resp.get_json()
    assert "network_signals" in data, "network_signals missing from response"
    ns = data["network_signals"]
    assert "is_headless" in ns
    assert "is_datacenter" in ns
    assert "is_vpn" in ns
    assert "browser" in ns
    assert "country" in ns
    assert ns["browser"] == "chrome"
    assert ns["country"] == "US"


# ── Test 7: ipinfo.io failure returns safe defaults ───────────────────────

def test_ipinfo_failure_returns_defaults():
    import urllib.error
    import enrichment as enr

    enr._ip_cache.clear()

    with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
        result = get_ip_info("8.8.8.8")

    assert result["is_datacenter"] is False
    assert result["is_vpn"] is False
    assert result["country"] == "unknown"
    assert result["org"] == ""


# ── Test 8: feature vector has 37 features ────────────────────────────────

def test_feature_vector_has_37_features():
    ext = FeatureExtractor()
    behavioural = ext.extractBatchFeatures(_minimal_batch())
    network = ext.extract_network_features({})
    combined = {**behavioural, **network}
    assert len(combined) == 37, (
        f"Expected 37 features (30 behavioural + 7 network), got {len(combined)}. "
        f"Behavioural: {len(behavioural)}, Network: {len(network)}"
    )
