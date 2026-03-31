"""
Bot Detection System - Flask Backend 

Main application server which does the following:
    1. Serves the Frontend (index.html)
    2. Receives Signal Batches from JavaScript
    3. Stores signals using SignalCollector
    4. Provides statistics endpoints


Architecture:
    - POST /api/signals -> Save Signal Batch
    - GET /api/stats -> Get collection statistics
    - GET / -> Serve Frontend HTML
"""

from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
import functools
import logging
import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent Directory to path so we can import backend modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collectors.signal_collector import SignalCollector
from utils.helpers import isValidSignalBatch, normalizeSignalBatch, formatTimestamp
from monitoring import metrics
from db import db as db_manager

logger = logging.getLogger(__name__)

#Initialize Flask application
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app, resources={r"/api/*": {
    "origins": "*",
    "allow_headers": ["Content-Type", "X-Api-Key", "X-Export-Key"],
    "methods": ["GET", "POST", "OPTIONS"],
}})

#Initialize Signal Collector
collector = SignalCollector()

#Store active session Count for simple tracking
activeSessions = set()

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "trained"
MODEL_NAME = "RandomForest"
IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
PREDICTIONS_LOG = (
    Path("/tmp/predictions_log.jsonl") if IS_LAMBDA
    else Path(__file__).resolve().parent / "data" / "predictions_log.jsonl"
)

_scoring_bundle = None
_server_start_time = None
_SHAP_PENDING = object()  # sentinel: explainer not yet attempted

# Human-readable explanations for each feature when it contributes to a bot prediction
FEATURE_INTERPRETATIONS = {
    "batch_event_count": "unusual total number of events in the batch",
    "has_mouse_moves": "presence or absence of mouse movement",
    "has_clicks": "presence or absence of click events",
    "has_keys": "presence or absence of keystroke events",
    "mouseMoveCount": "abnormal number of mouse move events",
    "mouseAvgVelocity": "unusual average mouse movement speed",
    "mouseStdVelocity": "unusually consistent mouse speed (low variance suggests automation)",
    "mouseMaxVelocity": "abnormal peak mouse speed",
    "mousePauseCount": "unusual number of pauses in mouse movement",
    "mouseAvgPauseDurationMs": "abnormal average pause duration",
    "mousePathEfficiency": "abnormally linear mouse movement (bot-like path efficiency)",
    "mouseAngularVelocityStd": "unusually uniform turning angles in mouse path",
    "mouseHoverTimeRatio": "abnormal time spent hovering",
    "mouseHoverFrequency": "unusual hover frequency pattern",
    "clickCount": "abnormal number of clicks",
    "clickIntervalMeanMs": "unusual average time between clicks",
    "clickIntervalStdMs": "unusually regular click timing (low variance suggests automation)",
    "clickIntervalMinMs": "abnormally fast minimum click interval",
    "clickIntervalMaxMs": "unusual maximum click interval",
    "clickClusteringRatio": "abnormal click burst pattern",
    "clickRatePerSec": "unusual click rate per second",
    "clickLeftRatio": "unusual left-click vs right-click ratio",
    "keyCount": "abnormal number of keystrokes",
    "keyInterKeyDelayMeanMs": "unusual average delay between keystrokes",
    "keyInterKeyDelayStdMs": "unusually regular keystroke timing (low variance suggests automation)",
    "keyRapidPresses": "high number of rapid key presses (bot-like speed)",
    "keyEntropy": "low keystroke diversity (repetitive key patterns)",
    "keyRatePerSec": "unusual typing speed",
    "batchDurationMs": "abnormal batch duration",
    "eventRatePerSec": "unusual overall event rate",
    "signalDiversityEntropy": "low signal type diversity (missing expected event types)",
    "clickToMoveRatio": "unusual ratio of clicks to mouse movements",
    "keyToMoveRatio": "unusual ratio of keystrokes to mouse movements",
}


FREE_TIER_LIMIT = 1000


def require_api_key(f=None, *, count_usage=True):
    """Decorator: validate X-Api-Key header; enforce per-plan rate limits.

    Bypasses auth when app.config["TESTING"] is True so the existing test
    suite works without modification.  test_api_keys.py sets TESTING=False
    for tests that exercise the auth logic directly.

    Pass count_usage=False (e.g. on /api/usage) to skip incrementing the
    request counter for that endpoint.
    """
    def decorator(func):
        @functools.wraps(func)
        def decorated(*args, **kwargs):
            if request.method == "OPTIONS":
                return func(*args, **kwargs)

            # Bypass in test mode so existing tests are unaffected
            if app.config.get("TESTING"):
                g.api_key = "test"
                g.is_master = True
                return func(*args, **kwargs)

            api_key = request.headers.get("X-Api-Key", "")

            # Master key bypasses all limits
            master_key = os.environ.get("HUMANGUARD_MASTER_KEY", "")
            if master_key and api_key == master_key:
                g.api_key = api_key
                g.is_master = True
                return func(*args, **kwargs)

            # Validate key
            record = db_manager.validate_api_key(api_key)
            if record is None:
                return jsonify({"error": "Invalid or inactive API key"}), 401

            # Rate limit check
            if record["current_month_count"] >= record["monthly_limit"]:
                return jsonify({
                    "error": "monthly limit reached",
                    "limit": record["monthly_limit"],
                    "plan": record["plan"],
                }), 429

            g.api_key = api_key
            g.is_master = False
            if count_usage:
                db_manager.increment_usage(api_key)
            return func(*args, **kwargs)

        return decorated

    # Support both @require_api_key and @require_api_key(count_usage=False)
    if f is not None:
        return decorator(f)
    return decorator


def _load_scoring_bundle():
    """
    Lazily load model/scaler/feature names for /api/score.

    Load order:
      1. ModelRegistry (S3) if MODEL_BUCKET env var is set.
      2. Local disk artifacts in models/trained/ (fallback).
    """
    global _scoring_bundle
    if _scoring_bundle is not None:
        return _scoring_bundle

    import joblib
    from features.feature_extractor import FeatureExtractor

    # ── Attempt registry load ──────────────────────────────────────────────────
    model_bucket = os.environ.get("MODEL_BUCKET")
    if model_bucket:
        try:
            from model_registry import ModelRegistry
            registry = ModelRegistry(bucket=model_bucket)
            reg_bundle = registry.load("latest")
            _scoring_bundle = {
                "model": reg_bundle["model"],
                "scaler": reg_bundle["scaler"],
                "feature_names": reg_bundle["feature_names"],
                "threshold": reg_bundle["threshold"],
                "explainer": _SHAP_PENDING,
                "extractor": FeatureExtractor(),
                "_registry_version": reg_bundle.get("version"),
                "_registry_metadata": reg_bundle.get("metadata", {}),
            }
            logger.info("Loaded model from registry: %s", reg_bundle.get("version"))
            return _scoring_bundle
        except Exception as exc:
            logger.warning("Registry load failed, falling back to local disk: %s", exc)

    # ── Local disk fallback ────────────────────────────────────────────────────
    model_path = MODEL_DIR / f"{MODEL_NAME}.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    feature_names_path = MODEL_DIR / "feature_names.json"
    threshold_path = MODEL_DIR / "threshold.json"

    if not model_path.exists() or not scaler_path.exists() or not feature_names_path.exists():
        missing = []
        if not model_path.exists():
            missing.append(str(model_path))
        if not scaler_path.exists():
            missing.append(str(scaler_path))
        if not feature_names_path.exists():
            missing.append(str(feature_names_path))
        raise FileNotFoundError(f"Missing scoring artifacts: {missing}")

    with open(feature_names_path, "r") as f:
        feature_names = json.load(f)

    threshold = 0.5
    if threshold_path.exists():
        try:
            with open(threshold_path, "r") as f:
                threshold = float(json.load(f).get("threshold", 0.5))
        except Exception:
            threshold = 0.5

    from joblib import parallel_backend
    with parallel_backend('sequential'):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

    # Load optional session-level blender (catches adaptive bots via temporal drift)
    session_blender = None
    session_blender_features = None
    blender_path   = MODEL_DIR / "session_blender.pkl"
    bl_feats_path  = MODEL_DIR / "session_blender_features.json"
    if blender_path.exists() and bl_feats_path.exists():
        try:
            session_blender = joblib.load(blender_path)
            with open(bl_feats_path) as _f:
                session_blender_features = json.load(_f)
            logger.info("Loaded session blender")
        except Exception as _exc:
            logger.warning("Failed to load session blender: %s", _exc)

    _scoring_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "threshold": threshold,
        "explainer": _SHAP_PENDING,
        "extractor": FeatureExtractor(),
        "_registry_version": None,
        "_registry_metadata": {},
        "session_blender": session_blender,
        "session_blender_features": session_blender_features,
    }
    return _scoring_bundle


def _get_shap_explainer(bundle: dict):
    """Return the SHAP TreeExplainer, creating it on first call (lazy init).

    Sentinel logic:
      _SHAP_PENDING — not yet attempted → try to create now
      None          — attempted and unavailable → return None immediately
      explainer obj — ready to use → return as-is
    """
    current = bundle.get("explainer", _SHAP_PENDING)
    if current is not _SHAP_PENDING:
        return current  # None (disabled) or a live explainer
    try:
        import shap
        bundle["explainer"] = shap.TreeExplainer(bundle["model"])
        logger.info("SHAP TreeExplainer created (lazy init)")
    except ImportError:
        logger.warning("shap not installed — SHAP explanations unavailable")
        bundle["explainer"] = None
    except Exception as exc:
        logger.warning("Failed to create SHAP explainer: %s", exc)
        bundle["explainer"] = None
    return bundle["explainer"]


def _log_prediction_local(session_id, prob_bot, label, threshold,
                          scoring_type="batch", response_time_ms=None, explanation=None,
                          source=None, ground_truth_label=None):
    """Append a prediction entry to the local JSONL log for dashboard fallback."""
    entry = {
        "sessionID": session_id,
        "prob_bot": prob_bot,
        "label": label,
        "threshold": threshold,
        "scoring_type": scoring_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if response_time_ms is not None:
        entry["response_time_ms"] = response_time_ms
    if explanation is not None:
        entry["explanation"] = explanation
    if source is not None:
        entry["source"] = source
    if ground_truth_label is not None:
        entry["ground_truth_label"] = ground_truth_label
    try:
        PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PREDICTIONS_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Failed to write prediction log: %s", exc)


@app.route('/api/score', methods=['POST'])
@require_api_key
def scoreSignals():
    """
    POST /api/score
    Batch-level bot probability scoring (MVP).

    Input: same payload shape as /api/signals
    Output: prob_bot + label (human|bot)
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"Error": "No JSON data received"}), 400

        data = normalizeSignalBatch(data)

        if not isValidSignalBatch(data):
            metrics.record_validation_error()
            return jsonify({
                "Error": "Invalid signal batch format",
                "received__keys": list(data.keys()),
                "signals_type": str(type(data.get("signals"))),
            }), 400

        # Extract source/ground-truth label metadata from payload (pass-through fields)
        payload_source = data.get("source")
        payload_ground_truth = data.get("label")  # ground truth, not predicted label

        # Basic abuse prevention: cap number of events to keep feature extraction bounded.
        signals = data.get("signals") or {}
        mouseMoves = signals.get("mouseMoves") or []
        clicks = signals.get("clicks") or []
        keys = signals.get("keys") or []

        MAX_MOUSE_MOVES = 5000
        MAX_CLICKS = 2000
        MAX_KEYS = 5000

        if len(mouseMoves) > MAX_MOUSE_MOVES or len(clicks) > MAX_CLICKS or len(keys) > MAX_KEYS:
            return jsonify({
                "Error": "Signal batch too large",
                "counts": {
                    "mouseMoves": len(mouseMoves),
                    "clicks": len(clicks),
                    "keys": len(keys),
                },
                "max": {
                    "mouseMoves": MAX_MOUSE_MOVES,
                    "clicks": MAX_CLICKS,
                    "keys": MAX_KEYS,
                }
            }), 413

        _score_start = time.time()
        bundle = _load_scoring_bundle()
        feature_names = bundle["feature_names"]

        feats = bundle["extractor"].extractBatchFeatures(signals)

        # Create feature vector in the exact trained feature order.
        x_row = [float(feats.get(name, 0.0)) for name in feature_names]

        import numpy as np

        x = np.array([x_row], dtype=float)
        x_scaled = bundle["scaler"].transform(x)

        # sklearn classifiers expose predict_proba; if not, fail loudly.
        prob_bot = float(bundle["model"].predict_proba(x_scaled)[0, 1])
        label = "bot" if prob_bot >= float(bundle["threshold"]) else "human"

        # Save prediction to PostgreSQL if available; tag with api_key as source
        try:
            from db.db_client import is_available as db_available, save_prediction
            if db_available():
                _pred_source = payload_source or getattr(g, "api_key", None)
                _caller_key = getattr(g, "api_key", None)
                save_prediction(data.get("sessionID"), prob_bot, label, float(bundle["threshold"]),
                                source=_pred_source, api_key=_caller_key)
        except Exception as exc:
            logger.warning("Failed to save prediction to PostgreSQL: %s", exc)

        # Persist session metadata (source/label) when provided via demo or simulator.
        # Also save raw signals to JSONL so demo/simulator batches are available for
        # retraining via scripts/retrain.py.
        if payload_source or payload_ground_truth:
            try:
                db_manager.save_session(data)
            except Exception as exc:
                logger.warning("Failed to save session metadata from score endpoint: %s", exc)
            try:
                collector.saveSignalBatch(data)
            except Exception as exc:
                logger.warning("Failed to save signals from score endpoint: %s", exc)

        response = {
            "success": True,
            "sessionID": data.get("sessionID"),
            "prob_bot": prob_bot,
            "label": label,
            "threshold": float(bundle["threshold"]),
        }

        # SHAP explanation — opt-out with ?explain=false
        explain = request.args.get("explain", "true").lower() != "false"
        if explain:
            explanation = _build_shap_explanation(bundle, x_scaled, label)
            if explanation is not None:
                response["explanation"] = explanation

        _response_time_ms = round((time.time() - _score_start) * 1000, 1)
        _log_prediction_local(
            data.get("sessionID"), prob_bot, label, float(bundle["threshold"]),
            scoring_type="batch",
            response_time_ms=_response_time_ms,
            explanation=response.get("explanation"),
            source=payload_source,
            ground_truth_label=payload_ground_truth,
        )
        metrics.record_prediction(is_bot=(label == "bot"), latency_ms=_response_time_ms)

        return jsonify(response), 200

    except FileNotFoundError as e:
        metrics.record_lambda_error()
        return jsonify({"Error": str(e)}), 503
    except Exception as e:
        metrics.record_lambda_error()
        logger.exception("Error scoring signals")
        return jsonify({"Error": f"Server error: {str(e)}"}), 500

def _load_session_batches(session_id):
    """
    Load all signal batches for a given sessionID from signals.jsonl.
    Returns a list of signal dicts (the 'signals' sub-object from each batch).
    """
    batches = []
    signals_path = collector.getSignalsFile()
    if not os.path.exists(signals_path):
        return batches
    with open(signals_path, "r") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if record.get("sessionID") == session_id:
                    batches.append(record.get("signals") or {})
            except (json.JSONDecodeError, Exception):
                continue
    return batches


def _score_single_batch(bundle, signals_dict):
    """Score a single signals dict, return prob_bot and the scaled feature vector."""
    import numpy as np
    feature_names = bundle["feature_names"]
    feats = bundle["extractor"].extractBatchFeatures(signals_dict)
    x_row = [float(feats.get(name, 0.0)) for name in feature_names]
    x = np.array([x_row], dtype=float)
    x_scaled = bundle["scaler"].transform(x)
    prob_bot = float(bundle["model"].predict_proba(x_scaled)[0, 1])
    return prob_bot, x_scaled


def _compute_trend(scores):
    """Determine if bot probability is increasing, decreasing, or stable."""
    if len(scores) < 2:
        return "stable"
    import numpy as np
    x = np.arange(len(scores), dtype=float)
    slope = np.polyfit(x, scores, 1)[0]
    if slope > 0.02:
        return "increasing"
    elif slope < -0.02:
        return "decreasing"
    return "stable"


def _build_shap_explanation(bundle, x_scaled, label):
    """Build SHAP explanation block for a single scaled feature vector."""
    explainer = _get_shap_explainer(bundle)
    if explainer is None:
        return None
    try:
        feature_names = bundle["feature_names"]
        shap_values = explainer.shap_values(x_scaled)
        sv = shap_values[0] if hasattr(shap_values, '__len__') and len(shap_values) > 0 else shap_values
        if hasattr(sv, 'ndim') and sv.ndim == 2:
            sv = sv[0]
        paired = list(zip(feature_names, sv))
        paired.sort(key=lambda p: abs(p[1]), reverse=True)
        top_features = [
            {"feature": name, "contribution": round(float(val), 4)}
            for name, val in paired[:5]
        ]
        top_name = paired[0][0]
        interpretation_text = FEATURE_INTERPRETATIONS.get(
            top_name, f"unusual value for {top_name}"
        )
        if label == "bot":
            interpretation = f"Session flagged as bot due to {interpretation_text}."
        else:
            interpretation = f"Session classified as human; top signal: {interpretation_text}."
        return {"top_features": top_features, "interpretation": interpretation}
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        return None


def _session_score_logic(session_id):
    """
    Core session-level scoring logic shared by POST and GET routes.
    Returns (response_dict, status_code).
    """
    import numpy as np

    try:
        batches = _load_session_batches(session_id)
        batch_count = len(batches)

        if batch_count < 2:
            return {
                "error": "Insufficient data",
                "message": "Session requires at least 2 batches for session-level scoring",
                "batch_count": batch_count,
            }, 400

        bundle = _load_scoring_bundle()
        threshold = float(bundle["threshold"])

        batch_scores = []
        batch_x_scaled = []
        for signals_dict in batches:
            prob, x_sc = _score_single_batch(bundle, signals_dict)
            batch_scores.append(prob)
            batch_x_scaled.append(x_sc)

        # Weighted average — linear weights giving later batches higher weight
        weights = np.arange(1, batch_count + 1, dtype=float)
        session_prob_bot = float(np.average(batch_scores, weights=weights))

        # Drift analysis
        scores_arr = np.array(batch_scores)
        drift = {
            "trend": _compute_trend(batch_scores),
            "drift_score": round(float(np.std(scores_arr)), 4),
            "max_prob_bot": round(float(np.max(scores_arr)), 4),
            "mean_prob_bot": round(float(np.mean(scores_arr)), 4),
        }

        # ── Temporal blending for long sessions (≥ 10 batches) ───────────
        # Detects adaptive bots that behave human-like early, bot-like later.
        if batch_count >= 10:
            extractor        = bundle["extractor"]
            temporal_drift   = extractor.temporal_drift_score(batches)
            delta_ms         = extractor.early_late_timing_delta(batches)
            consistency      = extractor.behavior_consistency_score(batches)

            blender     = bundle.get("session_blender")
            bl_features = bundle.get("session_blender_features") or []

            if blender is not None and bl_features:
                feat_map = {
                    "avg_batch_prob":       session_prob_bot,
                    "temporal_drift":       temporal_drift,
                    "early_late_delta_ms":  delta_ms,
                    "behavior_consistency": consistency,
                }
                x_meta = np.array([[feat_map.get(f, 0.0) for f in bl_features]])
                session_prob_bot = float(
                    np.clip(blender.predict_proba(x_meta)[0, 1], 0.0, 1.0)
                )
            else:
                # Fixed-rule fallback when no blender is available
                norm_delta         = float(min(delta_ms / 100.0, 1.0))
                temporal_suspicion = float(np.clip(
                    2.0 * temporal_drift + 0.4 * norm_delta + 0.3 * (1.0 - consistency),
                    0.0, 1.0,
                ))
                session_prob_bot = float(np.clip(
                    0.65 * session_prob_bot + 0.35 * temporal_suspicion, 0.0, 1.0
                ))

            drift["temporal_drift"]       = round(temporal_drift, 4)
            drift["early_late_delta_ms"]  = round(delta_ms, 1)
            drift["behavior_consistency"] = round(consistency, 4)

        label = "bot" if session_prob_bot >= threshold else "human"

        # Save to PostgreSQL if available
        try:
            from db.db_client import is_available as db_available, save_prediction
            if db_available():
                _caller_key = getattr(g, "api_key", None)
                save_prediction(session_id, session_prob_bot, label, threshold,
                                scoring_type="session", api_key=_caller_key)
        except Exception as exc:
            logger.warning("Failed to save session prediction to PostgreSQL: %s", exc)

        response = {
            "success": True,
            "sessionID": session_id,
            "session_prob_bot": round(session_prob_bot, 4),
            "label": label,
            "threshold": threshold,
            "batch_count": batch_count,
            "batch_scores": [round(s, 4) for s in batch_scores],
            "drift": drift,
        }

        # SHAP explanation for the highest-scoring batch
        peak_idx = int(np.argmax(scores_arr))
        explanation = _build_shap_explanation(bundle, batch_x_scaled[peak_idx], label)
        if explanation is not None:
            explanation["source_batch"] = peak_idx
            response["explanation"] = explanation

        _log_prediction_local(
            session_id, session_prob_bot, label, threshold,
            scoring_type="session",
            explanation=response.get("explanation"),
        )

        return response, 200

    except FileNotFoundError as e:
        return {"Error": str(e)}, 503
    except Exception as e:
        logger.exception("Error in session scoring")
        return {"Error": f"Server error: {str(e)}"}, 500


@app.route('/api/session-score', methods=['POST'])
@require_api_key
def sessionScore():
    """
    POST /api/session-score
    Session-level bot scoring — aggregates all batches for a sessionID.
    """
    data = request.get_json(silent=True)
    if not data or "sessionID" not in data:
        return jsonify({"error": "Missing sessionID"}), 400
    response, status = _session_score_logic(data["sessionID"])
    return jsonify(response), status


@app.route('/api/session-score/<session_id>', methods=['GET'])
@require_api_key
def sessionScoreGet(session_id):
    """
    GET /api/session-score/<sessionID>
    Convenience alias for session-level scoring.
    """
    response, status = _session_score_logic(session_id)
    return jsonify(response), status


@app.route('/api/stats', methods=['GET'])
@require_api_key
def getStats():
    """
    GET /api/stats - Returns collection statistics (total batches, num of unique sessions, signals file size, and server timestamp)
    """

    try:
        totalBatches = collector.getBatchCount()
        uniqueSessions = collector.getSessionCount()
        signalsFile = collector.getSignalsFile()
        fileSizeKb = 0
        if os.path.exists(signalsFile):
            fileSizeKb = os.path.getsize(signalsFile) / 1024

        result = {
            "Total Batches": totalBatches,
            "Unique Sessions": uniqueSessions,
            "Signals File Size (in Kb)": round(fileSizeKb, 1),
            "Signals File": signalsFile,
            "Server Timestamp": formatTimestamp(),
        }
        result.update(db_manager.get_stats())
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/signals', methods=['POST'])
@require_api_key
def saveSignals():
    """
    POST /api/signals - Receives signal batches from frontend JavaScripts
    """

    try:
        #Get JSON data from request
        data = request.get_json()

        if not data:
             return jsonify({"Error": "No JSON data received"}), 400

        data = normalizeSignalBatch(data)

        #Validate structure
        if not isValidSignalBatch(data):
            metrics.record_validation_error()
            return jsonify({
                "Error": "Invalid signal batch format",
                "received__keys": list(data.keys()),
                "signals_type": str(type(data.get("signals"))),
                }), 400

        sig = data.get("signals") or {}
        logger.info(
            "Received signal batch session=%s moves=%s clicks=%s keys=%s",
            data.get("sessionID"),
            len(sig.get("mouseMoves") or []),
            len(sig.get("clicks") or []),
            len(sig.get("keys") or []),
        )

        #Save to file
        success = collector.saveSignalBatch(data)

        if success:
            try:
                db_manager.save_session(data)
            except Exception as exc:
                logger.warning("db_manager.save_session failed: %s", exc)

            return jsonify({
                "success": True,
                "message": f"Saved batch for session {data.get('sessionID', 'unknown')}",
                "Total Batches": collector.getBatchCount(),
                "Session ID": data.get('sessionID')
            }), 200
        else:
            return jsonify({"Error": "Failed to save batch"}), 500
        
    except Exception as e:
        return jsonify({"Error": f"Server error: {str(e)}"}), 500
    
@app.route('/', methods=['GET'])

def serveFrontend():
    """
    GET / -> Serves the main frontend page. Returns index.html from frontend/ folder
    """
    return send_from_directory('../frontend', 'index.html')

@app.route('/health', methods=['GET'])
def health():
    """
    GET /health - Lightweight liveness check.
    Does NOT load model artifacts so it always responds, even if
    models/trained/ is empty.
    """
    now = time.time()
    start = _server_start_time if _server_start_time is not None else now
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "version": "1.0.0",
        "uptime_seconds": round(now - start, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200


def _dashboard_stats_from_log(threshold):
    """Build dashboard stats by reading predictions_log.jsonl."""
    records = []
    if PREDICTIONS_LOG.exists():
        with open(PREDICTIONS_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

    total = len(records)
    bot_count = sum(1 for r in records if r.get("label") == "bot")
    human_count = total - bot_count
    bot_rate = round(bot_count / total, 4) if total > 0 else 0.0

    times = [r["response_time_ms"] for r in records if "response_time_ms" in r]
    avg_response_time_ms = round(sum(times) / len(times), 1) if times else 0.0

    # Last 10, most-recent-first
    recent_raw = records[-10:][::-1]
    recent_predictions = [
        {
            "sessionID": r.get("sessionID"),
            "prob_bot": r.get("prob_bot"),
            "label": r.get("label"),
            "timestamp": r.get("timestamp"),
            "scoring_type": r.get("scoring_type", "batch"),
        }
        for r in recent_raw
    ]

    # Top 5 flagged features from SHAP data of bot detections
    feature_counts: dict = {}
    for r in records:
        if r.get("label") == "bot":
            for feat in ((r.get("explanation") or {}).get("top_features") or []):
                name = feat.get("feature")
                if name:
                    feature_counts[name] = feature_counts.get(name, 0) + 1
    top_flagged_features = sorted(feature_counts, key=lambda k: feature_counts[k], reverse=True)[:5]

    return {
        "total_predictions": total,
        "bot_count": bot_count,
        "human_count": human_count,
        "bot_rate": bot_rate,
        "avg_response_time_ms": avg_response_time_ms,
        "recent_predictions": recent_predictions,
        "top_flagged_features": top_flagged_features,
        "model": MODEL_NAME,
        "threshold": threshold,
    }


def _dashboard_stats_from_pg(threshold):
    """Build dashboard stats from PostgreSQL predictions table."""
    from db.db_client import execute_query

    cur = execute_query(
        "SELECT COUNT(*), SUM(CASE WHEN label='bot' THEN 1 ELSE 0 END) FROM predictions",
        commit=False,
    )
    row = cur.fetchone()
    total = int(row[0] or 0)
    bot_count = int(row[1] or 0)
    human_count = total - bot_count
    bot_rate = round(bot_count / total, 4) if total > 0 else 0.0

    cur = execute_query(
        "SELECT session_id, prob_bot, label, created_at, scoring_type "
        "FROM predictions ORDER BY created_at DESC LIMIT 10",
        commit=False,
    )
    rows = cur.fetchall()
    recent_predictions = [
        {
            "sessionID": r[0],
            "prob_bot": float(r[1]),
            "label": r[2],
            "timestamp": r[3].isoformat() if r[3] else None,
            "scoring_type": r[4],
        }
        for r in rows
    ]

    return {
        "total_predictions": total,
        "bot_count": bot_count,
        "human_count": human_count,
        "bot_rate": bot_rate,
        "avg_response_time_ms": 0.0,
        "recent_predictions": recent_predictions,
        "top_flagged_features": [],
        "model": MODEL_NAME,
        "threshold": threshold,
    }


@app.route('/api/dashboard-stats', methods=['GET'])
@require_api_key
def dashboardStats():
    """
    GET /api/dashboard-stats
    Returns aggregated stats for the live dashboard: totals, bot rate,
    recent predictions, top flagged SHAP features, model info.
    Reads from PostgreSQL if available, falls back to predictions_log.jsonl.
    """
    try:
        threshold = 0.5
        try:
            bundle = _load_scoring_bundle()
            threshold = float(bundle["threshold"])
        except Exception:
            pass

        try:
            from db.db_client import is_available as db_available
            if db_available():
                return jsonify(_dashboard_stats_from_pg(threshold)), 200
        except Exception as exc:
            logger.warning("PostgreSQL dashboard stats failed, using log: %s", exc)

        return jsonify(_dashboard_stats_from_log(threshold)), 200

    except Exception as e:
        logger.exception("Error getting dashboard stats")
        return jsonify({"error": str(e)}), 500


@app.route('/dashboard', methods=['GET'])
def serveDashboard():
    """GET /dashboard — Serves the live monitoring dashboard."""
    return send_from_directory('../frontend', 'dashboard.html')


@app.route('/api/model-info', methods=['GET'])
@require_api_key
def modelInfo():
    """
    GET /api/model-info
    Returns metadata for the currently active model (champion version from registry,
    or local model info when no registry is configured).
    """
    try:
        # Try to get live info from the loaded scoring bundle first
        bundle = None
        try:
            bundle = _load_scoring_bundle()
        except Exception:
            pass

        if bundle and bundle.get("_registry_version"):
            meta = bundle.get("_registry_metadata", {})
            return jsonify({
                "version": bundle["_registry_version"],
                "source": "registry",
                "accuracy": meta.get("accuracy"),
                "precision": meta.get("precision"),
                "recall": meta.get("recall"),
                "f1": meta.get("f1"),
                "roc_auc": meta.get("roc_auc"),
                "training_date": meta.get("training_date"),
                "training_samples": meta.get("training_samples"),
                "model_type": meta.get("model_type", MODEL_NAME),
                "champion": meta.get("champion", True),
            }), 200

        # Fallback: read local model_comparison.json
        comparison_path = MODEL_DIR / "model_comparison.json"
        if comparison_path.exists():
            with open(comparison_path) as f:
                comparison = json.load(f)
            winner = comparison.get("winner", MODEL_NAME)
            models = comparison.get("models", [])
            winner_metrics = next((m for m in models if m.get("model") == winner), {})
            return jsonify({
                "version": "local",
                "source": "local",
                "accuracy": winner_metrics.get("accuracy"),
                "precision": winner_metrics.get("precision"),
                "recall": winner_metrics.get("recall"),
                "f1": winner_metrics.get("f1"),
                "roc_auc": winner_metrics.get("roc_auc"),
                "training_date": None,
                "training_samples": None,
                "model_type": winner,
                "champion": True,
            }), 200

        return jsonify({"version": "unknown", "source": "none"}), 200

    except Exception as exc:
        logger.exception("Error in /api/model-info")
        return jsonify({"error": str(exc)}), 500


@app.route('/demo', methods=['GET'])
def serveDemo():
    """GET /demo — Serves the public-facing human verification demo."""
    return send_from_directory('../frontend', 'demo.html')


@app.route('/simulate', methods=['GET'])
def serveSimulator():
    """GET /simulate — Serves the internal bot behavior simulator."""
    return send_from_directory('../frontend', 'bot_simulator.html')


@app.route('/leaderboard', methods=['GET'])
def serveLeaderboard():
    """GET /leaderboard — Serves the public leaderboard page."""
    return send_from_directory('../frontend', 'leaderboard.html')


@app.route('/api/leaderboard', methods=['POST'])
@require_api_key
def leaderboardPost():
    """
    POST /api/leaderboard
    Submit a nickname + session_id to add to the leaderboard.
    Looks up the prediction from the predictions log and stores the entry.
    Returns rank, total, and a percentile message.
    """
    import re as _re
    data = request.get_json(silent=True) or {}

    raw_nickname = str(data.get("nickname", "")).strip()
    session_id = str(data.get("session_id", "")).strip()

    if not raw_nickname:
        return jsonify({"error": "nickname is required"}), 400

    # Sanitize: alphanumeric + spaces, max 20 chars
    nickname = _re.sub(r"[^a-zA-Z0-9 ]", "", raw_nickname)[:20].strip()
    if not nickname:
        return jsonify({"error": "nickname must contain alphanumeric characters"}), 400

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    # Look up prediction from JSONL log
    prob_bot = None
    verdict = None
    if PREDICTIONS_LOG.exists():
        with open(PREDICTIONS_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("sessionID") == session_id:
                        prob_bot = rec.get("prob_bot")
                        verdict = rec.get("label")
                except Exception:
                    continue

    # Fall back to DB if not found in JSONL
    if prob_bot is None:
        rows = db_manager.get_recent_predictions(limit=1000)
        for row in rows:
            if row.get("session_id") == session_id:
                prob_bot = row.get("prob_bot")
                verdict = row.get("label")
                break

    if prob_bot is None:
        return jsonify({"error": "session not found — complete the challenge first"}), 404

    # Save to leaderboard
    db_manager.save_leaderboard_entry(nickname, prob_bot, verdict, session_id)

    # Compute rank from all entries (sorted by prob_bot ASC)
    all_entries = db_manager.get_leaderboard(limit=100000)
    rank = len(all_entries)  # default to last
    for i, entry in enumerate(all_entries):
        if entry.get("session_id") == session_id:
            rank = i + 1
            break

    total = len(all_entries)
    percentile = round((1 - (rank - 1) / max(total - 1, 1)) * 100) if total > 1 else 100

    return jsonify({
        "rank": rank,
        "total": total,
        "nickname": nickname,
        "prob_bot": prob_bot,
        "verdict": verdict,
        "percentile": percentile,
        "message": f"You scored more human than {percentile}% of participants",
    }), 200


@app.route('/api/leaderboard', methods=['GET'])
@require_api_key
def leaderboardGet():
    """
    GET /api/leaderboard
    Returns top 20 leaderboard entries (lowest prob_bot first) with rank,
    human_confidence percentage, verdict, and time_ago.
    """
    try:
        limit = min(int(request.args.get("limit", 20)), 50)
        entries = db_manager.get_leaderboard(limit=limit)
        stats = db_manager.get_leaderboard_stats()

        now = datetime.utcnow()
        result = []
        for i, e in enumerate(entries):
            created_at = e.get("created_at", "")
            try:
                if isinstance(created_at, str) and created_at:
                    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    diff = now - ts.replace(tzinfo=None)
                    mins = int(diff.total_seconds() / 60)
                    if mins < 1:
                        time_ago = "just now"
                    elif mins < 60:
                        time_ago = f"{mins}m ago"
                    else:
                        time_ago = f"{mins // 60}h ago"
                else:
                    time_ago = ""
            except Exception:
                time_ago = str(created_at) if created_at else ""

            result.append({
                "rank": i + 1,
                "nickname": e.get("nickname"),
                "prob_bot": round(float(e.get("prob_bot", 0)), 4),
                "human_confidence": round((1 - float(e.get("prob_bot", 0))) * 100),
                "verdict": e.get("verdict"),
                "session_id": e.get("session_id"),
                "time_ago": time_ago,
            })

        return jsonify({"entries": result, "stats": stats}), 200

    except Exception as exc:
        logger.exception("Error in GET /api/leaderboard")
        return jsonify({"error": str(exc)}), 500


@app.route('/api/export', methods=['GET'])
@require_api_key
def exportSessions():
    """
    GET /api/export
    Returns all labeled sessions as CSV for retraining.
    Protected by X-Export-Key header (set EXPORT_API_KEY env var; defaults to 'devkey').
    """
    import csv
    import io
    from flask import Response

    api_key = request.headers.get("X-Export-Key", "")
    export_key = os.environ.get("EXPORT_API_KEY", "devkey")
    if not api_key or api_key != export_key:
        return jsonify({"error": "Unauthorized"}), 401

    records = []
    if PREDICTIONS_LOG.exists():
        with open(PREDICTIONS_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "session_id", "source", "ground_truth_label",
        "prob_bot", "predicted_label", "threshold", "scoring_type", "timestamp",
    ])
    for r in records:
        writer.writerow([
            r.get("sessionID", ""),
            r.get("source", ""),
            r.get("ground_truth_label", ""),
            r.get("prob_bot", ""),
            r.get("label", ""),
            r.get("threshold", ""),
            r.get("scoring_type", "batch"),
            r.get("timestamp", ""),
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=humanguard_sessions.csv"},
    )


@app.route('/api/register', methods=['POST'])
def registerApiKey():
    """
    POST /api/register
    Accepts {email}, generates a hg_live_XXXX API key, returns key + plan info.
    No API key required to call this endpoint.
    """
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip()
    if not email or "@" not in email:
        return jsonify({"error": "A valid email address is required"}), 400

    api_key = db_manager.generate_api_key(email)
    return jsonify({
        "api_key": api_key,
        "plan": "free",
        "monthly_limit": FREE_TIER_LIMIT,
        "docs_url": "https://github.com/rubenbetabdishoo/HumanGuard#api",
    }), 201


@app.route('/api/usage', methods=['GET'])
@require_api_key(count_usage=False)
def getUsage():
    """
    GET /api/usage
    Returns current month usage for the X-Api-Key provided.
    """
    api_key = getattr(g, "api_key", "")
    usage = db_manager.get_usage(api_key)
    return jsonify(usage), 200


@app.route('/api/client/stats', methods=['GET'])
@require_api_key(count_usage=False)
def clientStats():
    """
    GET /api/client/stats
    Returns prediction stats scoped to the calling API key.
    """
    api_key = getattr(g, "api_key", "")
    stats = db_manager.get_client_stats(api_key)
    return jsonify(stats), 200


@app.route('/api/client/predictions', methods=['GET'])
@require_api_key(count_usage=False)
def clientPredictions():
    """
    GET /api/client/predictions
    Returns recent predictions scoped to the calling API key.
    """
    api_key = getattr(g, "api_key", "")
    limit = min(int(request.args.get("limit", 50)), 200)
    predictions = db_manager.get_client_predictions(api_key, limit=limit)
    # Serialize datetime objects
    for p in predictions:
        if hasattr(p.get("created_at"), "isoformat"):
            p["created_at"] = p["created_at"].isoformat()
    return jsonify(predictions), 200


@app.route('/client', methods=['GET'])
def clientDashboard():
    """GET /client — Serves the client-facing bot monitoring dashboard."""
    return send_from_directory('../frontend', 'client_dashboard.html')


@app.route('/register', methods=['GET'])
def registerPage():
    """GET /register — Serves the API key registration page."""
    return send_from_directory('../frontend', 'register.html')


if IS_LAMBDA:
    import base64
    import io

    def handler(event, context):
        """Minimal API Gateway v2 → Flask WSGI bridge for Lambda."""
        rc = event.get("requestContext", {}).get("http", {})
        method = rc.get("method", "GET")
        path = event.get("rawPath", "/")
        query_string = event.get("rawQueryString", "") or ""
        headers = event.get("headers", {}) or {}
        body = event.get("body") or ""
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body)
        elif isinstance(body, str):
            body = body.encode("utf-8")
        else:
            body = b""

        environ = {
            "REQUEST_METHOD": method,
            "SCRIPT_NAME": "",
            "PATH_INFO": path,
            "QUERY_STRING": query_string,
            "SERVER_NAME": "lambda",
            "SERVER_PORT": "443",
            "SERVER_PROTOCOL": "HTTP/1.1",
            "CONTENT_LENGTH": str(len(body)),
            "wsgi.version": (1, 0),
            "wsgi.url_scheme": "https",
            "wsgi.input": io.BytesIO(body),
            "wsgi.errors": sys.stderr,
            "wsgi.multithread": False,
            "wsgi.multiprocess": False,
            "wsgi.run_once": False,
        }
        for k, v in headers.items():
            key = k.upper().replace("-", "_")
            if key == "CONTENT_TYPE":
                environ["CONTENT_TYPE"] = v
            elif key == "CONTENT_LENGTH":
                environ["CONTENT_LENGTH"] = v
            else:
                environ[f"HTTP_{key}"] = v

        resp_status = [None]
        resp_headers = [{}]
        resp_body = []

        def start_response(status, response_headers, exc_info=None):
            resp_status[0] = status
            resp_headers[0] = dict(response_headers)

        for chunk in app(environ, start_response):
            resp_body.append(chunk)

        body_bytes = b"".join(resp_body)
        try:
            body_str = body_bytes.decode("utf-8")
            is_b64 = False
        except UnicodeDecodeError:
            body_str = base64.b64encode(body_bytes).decode("utf-8")
            is_b64 = True

        return {
            "statusCode": int(resp_status[0].split(" ", 1)[0]),
            "headers": resp_headers[0],
            "body": body_str,
            "isBase64Encoded": is_b64,
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _server_start_time = time.time()
    app.run(host="0.0.0.0", debug=not IS_LAMBDA, port=int(os.environ.get("PORT", 5050)))