import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from features.feature_extractor import FeatureExtractor
from features.feature_utils import MouseTrajectoryUtils, KeystrokeUtils

EXPECTED_KEYS = {
    "batch_event_count",
    "mouseMoveCount", "mouseAvgVelocity", "mouseStdVelocity", "mouseMaxVelocity",
    "mouseAvgPauseDurationMs", "mousePathEfficiency",
    "mouseAngularVelocityStd", "mouseHoverTimeRatio", "mouseHoverFrequency",
    "clickRatePerSec", "clickIntervalMeanMs", "clickIntervalStdMs",
    "clickIntervalMinMs", "clickIntervalMaxMs",
    "keyCount", "keyRatePerSec", "keyInterKeyDelayMeanMs", "keyInterKeyDelayStdMs",
    "keyEntropy", "keyRapidPresses",
    "batchDurationMs", "eventRatePerSec",
    "clickToMoveRatio", "keyToMoveRatio",
    # Session-consistency features
    "keystroke_timing_regularity", "typing_rhythm_autocorrelation",
    "mouse_acceleration_variance", "mouse_keystroke_correlation",
    "session_phase_consistency",
}


def _realistic_batch():
    return {
        "mouseMoves": [
            {"x": 100, "y": 200, "ts": 1000},
            {"x": 120, "y": 210, "ts": 1100},
            {"x": 140, "y": 220, "ts": 1300},
            {"x": 180, "y": 250, "ts": 1500},
        ],
        "clicks": [
            {"ts": 1050, "button": 0},
            {"ts": 1600, "button": 0},
        ],
        "keys": [
            {"code": "KeyH", "ts": 1020},
            {"code": "KeyE", "ts": 1080},
            {"code": "KeyL", "ts": 1160},
            {"code": "KeyL", "ts": 1230},
            {"code": "KeyO", "ts": 1350},
        ],
    }


# -------------------------------------------------------------------
# FeatureExtractor tests
# -------------------------------------------------------------------

def test_extract_full_batch_returns_all_keys():
    ext = FeatureExtractor()
    feats = ext.extractBatchFeatures(_realistic_batch())
    assert EXPECTED_KEYS == set(feats.keys())


def test_extract_full_batch_values_are_floats():
    ext = FeatureExtractor()
    feats = ext.extractBatchFeatures(_realistic_batch())
    for key, val in feats.items():
        assert isinstance(val, float), f"{key} is {type(val)}, expected float"


def test_extract_empty_signals_no_crash():
    ext = FeatureExtractor()
    feats = ext.extractBatchFeatures({"mouseMoves": [], "clicks": [], "keys": []})
    assert EXPECTED_KEYS == set(feats.keys())
    for key, val in feats.items():
        assert isinstance(val, float), f"{key} is {type(val)}"
        assert val == 0.0, f"{key} should be 0.0 for empty signals, got {val}"


def test_extract_single_mouse_move_no_crash():
    ext = FeatureExtractor()
    batch = {
        "mouseMoves": [{"x": 50, "y": 50, "ts": 1000}],
        "clicks": [],
        "keys": [],
    }
    feats = ext.extractBatchFeatures(batch)
    assert "mouseMoveCount" in feats
    assert feats["mouseMoveCount"] == 1.0


# -------------------------------------------------------------------
# MouseTrajectoryUtils tests
# -------------------------------------------------------------------

def test_path_efficiency_straight_line():
    """Path efficiency of a perfectly straight line is 1.0."""
    utils = MouseTrajectoryUtils()
    p1 = (0.0, 0.0)
    p2 = (5.0, 0.0)
    p3 = (10.0, 0.0)
    # Total distance = 5 + 5 = 10, straight-line distance = 10 → ratio = 1.0
    d1 = utils.distance(p1, p2)
    d2 = utils.distance(p2, p3)
    straight = utils.distance(p1, p3)
    efficiency = (d1 + d2) / straight
    assert efficiency == 1.0


def test_path_efficiency_curved_path_gt_one():
    """Path efficiency of a curved path should be > 1.0 (longer than straight line)."""
    utils = MouseTrajectoryUtils()
    p1 = (0.0, 0.0)
    p2 = (5.0, 10.0)  # detour upward
    p3 = (10.0, 0.0)
    d1 = utils.distance(p1, p2)
    d2 = utils.distance(p2, p3)
    straight = utils.distance(p1, p3)
    efficiency = (d1 + d2) / straight
    assert efficiency > 1.0


# -------------------------------------------------------------------
# KeystrokeUtils tests
# -------------------------------------------------------------------

def test_entropy_uniform_keystrokes_is_low():
    """Entropy of repeated identical keys should be 0."""
    utils = KeystrokeUtils()
    items = ["KeyA"] * 20
    assert utils.calculateEntropy(items) == 0.0


def test_entropy_diverse_keystrokes_is_higher():
    """Entropy of diverse keys should be greater than uniform."""
    utils = KeystrokeUtils()
    uniform = ["KeyA"] * 10
    diverse = ["KeyA", "KeyB", "KeyC", "KeyD", "KeyE",
               "KeyF", "KeyG", "KeyH", "KeyI", "KeyJ"]
    assert utils.calculateEntropy(diverse) > utils.calculateEntropy(uniform)
