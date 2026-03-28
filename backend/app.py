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

from flask import Flask, request, jsonify, send_from_directory
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
from utils.helpers import isValidSignalBatch, formatTimestamp

logger = logging.getLogger(__name__)

#Initialize Flask application
app = Flask(__name__, static_folder='../frontend', static_url_path='')

#Initialize Signal Collector
collector = SignalCollector()

#Store active session Count for simple tracking
activeSessions = set()

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "trained"
MODEL_NAME = "XGBoost"

_scoring_bundle = None
_server_start_time = None

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


def _load_scoring_bundle():
    """
    Lazily load model/scaler/feature names for /api/score.

    Expected artifacts created by models/run_training.py:
      - models/trained/RandomForest.pkl
      - models/trained/scaler.pkl
      - models/trained/feature_names.json
    """
    global _scoring_bundle
    if _scoring_bundle is not None:
        return _scoring_bundle

    import joblib
    from features.feature_extractor import FeatureExtractor  # ensure feature code is importable

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

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load SHAP explainer if shap is installed
    explainer = None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer loaded successfully")
    except ImportError:
        logger.warning("shap not installed — SHAP explanations will be unavailable")
    except Exception as exc:
        logger.warning("Failed to create SHAP explainer: %s", exc)

    _scoring_bundle = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "threshold": threshold,
        "explainer": explainer,
        # Keep a single extractor instance to reduce per-request overhead.
        "extractor": FeatureExtractor(),
    }
    return _scoring_bundle


@app.route('/api/score', methods=['POST'])
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

        if not isValidSignalBatch(data):
            return jsonify({
                "Error": "Invalid signal batch format",
                "received__keys": list(data.keys()),
                "signals_type": str(type(data.get("signals"))),
            }), 400

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

        # Save prediction to PostgreSQL if available
        try:
            from db.db_client import is_available as db_available, save_prediction
            if db_available():
                save_prediction(data.get("sessionID"), prob_bot, label, float(bundle["threshold"]))
        except Exception as exc:
            logger.warning("Failed to save prediction to PostgreSQL: %s", exc)

        response = {
            "success": True,
            "sessionID": data.get("sessionID"),
            "prob_bot": prob_bot,
            "label": label,
            "threshold": float(bundle["threshold"]),
        }

        # SHAP explanation — opt-out with ?explain=false
        explain = request.args.get("explain", "true").lower() != "false"
        explainer = bundle.get("explainer")
        if explain and explainer is not None:
            try:
                shap_values = explainer.shap_values(x_scaled)
                sv = shap_values[0] if hasattr(shap_values, '__len__') and len(shap_values) > 0 else shap_values
                # For binary classifiers shap_values may be 2D [n_samples, n_features]
                if hasattr(sv, 'ndim') and sv.ndim == 2:
                    sv = sv[0]

                paired = list(zip(feature_names, sv))
                paired.sort(key=lambda p: abs(p[1]), reverse=True)
                top_features = [
                    {"feature": name, "contribution": round(float(val), 4)}
                    for name, val in paired[:5]
                ]

                top_name = paired[0][0]
                top_direction = "high" if paired[0][1] > 0 else "low"
                interpretation_text = FEATURE_INTERPRETATIONS.get(
                    top_name, f"unusual value for {top_name}"
                )
                if label == "bot":
                    interpretation = f"Session flagged as bot due to {interpretation_text}."
                else:
                    interpretation = f"Session classified as human; top signal: {interpretation_text}."

                response["explanation"] = {
                    "top_features": top_features,
                    "interpretation": interpretation,
                }
            except Exception as exc:
                logger.warning("SHAP explanation failed: %s", exc)

        return jsonify(response), 200

    except FileNotFoundError as e:
        return jsonify({"Error": str(e)}), 503
    except Exception as e:
        logger.exception("Error scoring signals")
        return jsonify({"Error": f"Server error: {str(e)}"}), 500

@app.route('/api/stats', methods=['GET'])
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

        return jsonify({
            "Total Batches": totalBatches,
            "Unique Sessions": uniqueSessions,
            "Signals File Size (in Kb)": round(fileSizeKb, 1),
            "Signals File": signalsFile,
            "Server Timestamp": formatTimestamp()
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/signals', methods=['POST'])
def saveSignals():
    """
    POST /api/signals - Receives signal batches from frontend JavaScripts
    """

    try:
        #Get JSON data from request
        data = request.get_json()

        if not data:
             return jsonify({"Error": "No JSON data received"}), 400

        #Validate structure
        if not isValidSignalBatch(data):
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _server_start_time = time.time()
    app.run(debug=True, port=int(os.environ.get("PORT", 5050)))