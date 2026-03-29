"""
CloudWatch Metrics — HumanGuard

Emits custom metrics to the "HumanGuard" CloudWatch namespace.
Gracefully no-ops when CLOUDWATCH_ENABLED is not "true" or when
boto3 / network is unavailable (local dev, CI).

Usage:
    from monitoring import metrics          # module-level singleton
    metrics.record_prediction(is_bot=True, latency_ms=120.4)
    metrics.record_validation_error()
"""

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class CloudWatchMetrics:
    NAMESPACE = "HumanGuard"

    def __init__(self):
        self._enabled = os.environ.get("CLOUDWATCH_ENABLED", "").lower() == "true"
        self._client = None
        self._region = os.environ.get("AWS_REGION", "us-east-1")

        if self._enabled:
            try:
                import boto3
                self._client = boto3.client("cloudwatch", region_name=self._region)
                logger.info("CloudWatch metrics enabled (namespace=%s, region=%s)",
                            self.NAMESPACE, self._region)
            except Exception as exc:
                logger.warning("CloudWatch unavailable — metrics disabled: %s", exc)
                self._enabled = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _put(self, metric_name: str, value: float, unit: str = "Count"):
        """Emit a single data point. Silently drops on any error."""
        if not self._enabled or self._client is None:
            return
        try:
            self._client.put_metric_data(
                Namespace=self.NAMESPACE,
                MetricData=[{
                    "MetricName": metric_name,
                    "Value": value,
                    "Unit": unit,
                    "Timestamp": datetime.now(timezone.utc),
                }],
            )
        except Exception as exc:
            logger.warning("CloudWatch put_metric_data(%s) failed: %s", metric_name, exc)

    # ── Public helpers ────────────────────────────────────────────────────────

    def record_prediction(self, is_bot: bool, latency_ms: float):
        """Emit score_requests + bot/human_detections + prediction_latency_ms."""
        self._put("score_requests", 1)
        self._put("bot_detections" if is_bot else "human_detections", 1)
        self._put("prediction_latency_ms", latency_ms, unit="Milliseconds")

    def record_validation_error(self):
        """Emit a validation_errors data point."""
        self._put("validation_errors", 1)

    def record_lambda_error(self):
        """Emit a lambda_errors data point."""
        self._put("lambda_errors", 1)


# Module-level singleton — import this everywhere
metrics = CloudWatchMetrics()
