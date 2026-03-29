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
from flask_cors import CORS
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
CORS(app, resources={r"/api/*": {"origins": "*"}})

#Initialize Signal Collector
collector = SignalCollector()

#Store active session Count for simple tracking
activeSessions = set()

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "trained"
MODEL_NAME = "XGBoost"
IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
PREDICTIONS_LOG = (
    Path("/tmp/predictions_log.jsonl") if IS_LAMBDA
    else Path(__file__).resolve().parent / "data" / "predictions_log.jsonl"
)

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

    from joblib import parallel_backend
    with parallel_backend('sequential'):
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


def _log_prediction_local(session_id, prob_bot, label, threshold,
                          scoring_type="batch", response_time_ms=None, explanation=None):
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
    try:
        PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PREDICTIONS_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.warning("Failed to write prediction log: %s", exc)


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
        )

        return jsonify(response), 200

    except FileNotFoundError as e:
        return jsonify({"Error": str(e)}), 503
    except Exception as e:
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
    explainer = bundle.get("explainer")
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
        label = "bot" if session_prob_bot >= threshold else "human"

        # Drift analysis
        scores_arr = np.array(batch_scores)
        drift = {
            "trend": _compute_trend(batch_scores),
            "drift_score": round(float(np.std(scores_arr)), 4),
            "max_prob_bot": round(float(np.max(scores_arr)), 4),
            "mean_prob_bot": round(float(np.mean(scores_arr)), 4),
        }

        # Save to PostgreSQL if available
        try:
            from db.db_client import is_available as db_available, save_prediction
            if db_available():
                save_prediction(session_id, session_prob_bot, label, threshold, scoring_type="session")
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
def sessionScoreGet(session_id):
    """
    GET /api/session-score/<sessionID>
    Convenience alias for session-level scoring.
    """
    response, status = _session_score_logic(session_id)
    return jsonify(response), status


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