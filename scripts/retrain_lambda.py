#!/usr/bin/env python3
"""
HumanGuard Retrain Lambda Handler

Triggered by EventBridge on a 6-hour schedule. Checks whether enough new
labeled sessions have accumulated and, if so, runs the retrain pipeline.

Environment variables:
    API_URL          — HumanGuard API base URL (used for health check only)
    EXPORT_API_KEY   — X-Export-Key for /api/export
    SNS_ALERT_TOPIC  — ARN of SNS topic to publish retrain results
    MODEL_BUCKET     — S3 bucket for ModelRegistry (optional)
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrain_lambda")


def _publish_sns(topic_arn: str, subject: str, message: str):
    """Post a message to an SNS topic (best-effort, never raises)."""
    if not topic_arn:
        return
    try:
        import boto3
        sns = boto3.client("sns", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        sns.publish(TopicArn=topic_arn, Subject=subject, Message=message)
        logger.info("SNS notification sent to %s", topic_arn)
    except Exception as exc:
        logger.warning("SNS publish failed: %s", exc)


def handler(event, context):
    """
    Lambda entry point — called by EventBridge every 6 hours.

    Returns a dict with:
        status      — "skipped" | "retrained" | "error"
        session_count  — number of unlabeled sessions found
        threshold   — RETRAIN_THRESHOLD value
        message     — human-readable summary
    """
    from backend.db import db as db_manager, RETRAIN_THRESHOLD

    sns_topic = os.environ.get("SNS_ALERT_TOPIC", "")
    export_key = os.environ.get("EXPORT_API_KEY", "devkey")
    api_url = os.environ.get("API_URL", "http://localhost:5050")

    # 1. Check how many new labeled sessions are waiting
    try:
        session_count = db_manager.get_unlabeled_session_count()
    except Exception as exc:
        msg = f"Failed to query unlabeled session count: {exc}"
        logger.error(msg)
        _publish_sns(sns_topic, "HumanGuard retrain ERROR", msg)
        return {"status": "error", "message": msg}

    logger.info(
        "Retrain check: %d unlabeled sessions (threshold=%d)",
        session_count, RETRAIN_THRESHOLD,
    )

    if session_count < RETRAIN_THRESHOLD:
        msg = (
            f"Retrain skipped: {session_count}/{RETRAIN_THRESHOLD} "
            "unlabeled sessions accumulated."
        )
        logger.info(msg)
        return {
            "status": "skipped",
            "session_count": session_count,
            "threshold": RETRAIN_THRESHOLD,
            "message": msg,
        }

    # 2. Threshold met — run retrain pipeline as a subprocess
    logger.info("Threshold met — starting retrain pipeline…")
    retrain_script = str(REPO_ROOT / "scripts" / "retrain.py")
    cmd = [
        sys.executable, retrain_script,
        "--auto",
        "--api-url", api_url,
        "--export-key", export_key,
    ]
    if os.environ.get("MODEL_BUCKET"):
        cmd += ["--push"]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
            cwd=str(REPO_ROOT),
        )
        success = proc.returncode == 0
        output = (proc.stdout or "") + (proc.stderr or "")
        logger.info("Retrain subprocess exit code: %d", proc.returncode)
        if proc.stdout:
            logger.info("stdout: %s", proc.stdout[-2000:])
        if proc.stderr:
            logger.warning("stderr: %s", proc.stderr[-2000:])
    except subprocess.TimeoutExpired:
        msg = "Retrain subprocess timed out after 10 minutes"
        logger.error(msg)
        _publish_sns(sns_topic, "HumanGuard retrain TIMEOUT", msg)
        return {"status": "error", "session_count": session_count, "message": msg}
    except Exception as exc:
        msg = f"Retrain subprocess failed to launch: {exc}"
        logger.error(msg)
        _publish_sns(sns_topic, "HumanGuard retrain ERROR", msg)
        return {"status": "error", "session_count": session_count, "message": msg}

    status = "retrained" if success else "error"
    subject = (
        f"HumanGuard retrain {'complete' if success else 'FAILED'} "
        f"({session_count} sessions)"
    )
    _publish_sns(sns_topic, subject, output[-4000:])

    return {
        "status": status,
        "session_count": session_count,
        "threshold": RETRAIN_THRESHOLD,
        "message": subject,
    }


if __name__ == "__main__":
    # Local test invocation
    result = handler({}, None)
    print(json.dumps(result, indent=2))
