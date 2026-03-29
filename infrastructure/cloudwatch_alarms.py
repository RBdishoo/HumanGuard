"""
HumanGuard CloudWatch Alarms — run once at deploy time.

Creates (or updates) four alarms in the "HumanGuard" namespace and an
SNS topic to receive alert emails.

Usage:
    SNS_ALERT_EMAIL=you@example.com python infrastructure/cloudwatch_alarms.py

Environment variables:
    AWS_REGION         — defaults to us-east-1
    AWS_ACCOUNT_ID     — required for alarm ARN construction
    SNS_ALERT_EMAIL    — if set, subscribes this email to the alert topic
"""

import os

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "796793347388")
NAMESPACE = "HumanGuard"
TOPIC_NAME = "HumanGuard-Alerts"
SNS_ALERT_EMAIL = os.environ.get("SNS_ALERT_EMAIL", "")

cw = boto3.client("cloudwatch", region_name=REGION)
sns = boto3.client("sns", region_name=REGION)


# ── SNS topic ─────────────────────────────────────────────────────────────────

def get_or_create_topic():
    resp = sns.create_topic(Name=TOPIC_NAME)   # idempotent
    topic_arn = resp["TopicArn"]
    print(f"SNS topic: {topic_arn}")

    if SNS_ALERT_EMAIL:
        sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=SNS_ALERT_EMAIL)
        print(f"Subscribed {SNS_ALERT_EMAIL} — check inbox to confirm.")

    return topic_arn


# ── Alarms ────────────────────────────────────────────────────────────────────

def create_alarms(topic_arn):
    alarms = []

    # 1. BotRateSpike — bot_detections / score_requests > 80% over one 5-min period
    #    Uses metric math: IF(total > 0, bot/total, 0)
    cw.put_metric_alarm(
        AlarmName="HumanGuard-BotRateSpike",
        AlarmDescription="Bot detection rate exceeded 80% of scored requests over 5 minutes.",
        Metrics=[
            {
                "Id": "bot",
                "MetricStat": {
                    "Metric": {"Namespace": NAMESPACE, "MetricName": "bot_detections"},
                    "Period": 300,
                    "Stat": "Sum",
                },
                "ReturnData": False,
            },
            {
                "Id": "total",
                "MetricStat": {
                    "Metric": {"Namespace": NAMESPACE, "MetricName": "score_requests"},
                    "Period": 300,
                    "Stat": "Sum",
                },
                "ReturnData": False,
            },
            {
                "Id": "rate",
                "Expression": "IF(total > 0, bot/total, 0)",
                "Label": "BotRate",
                "ReturnData": True,
            },
        ],
        ComparisonOperator="GreaterThanThreshold",
        Threshold=0.8,
        EvaluationPeriods=1,
        TreatMissingData="notBreaching",
        AlarmActions=[topic_arn],
        OKActions=[topic_arn],
    )
    alarms.append("HumanGuard-BotRateSpike")

    # 2. HighLatency — p95 prediction_latency_ms > 2000ms for 3 consecutive minutes
    cw.put_metric_alarm(
        AlarmName="HumanGuard-HighLatency",
        AlarmDescription="p95 prediction latency exceeded 2000ms for 3 consecutive minutes.",
        Namespace=NAMESPACE,
        MetricName="prediction_latency_ms",
        ExtendedStatistic="p95",
        Period=60,
        EvaluationPeriods=3,
        Threshold=2000,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[topic_arn],
        OKActions=[topic_arn],
    )
    alarms.append("HumanGuard-HighLatency")

    # 3. ValidationErrorRate — more than 10 validation errors in a 5-min window
    cw.put_metric_alarm(
        AlarmName="HumanGuard-ValidationErrorRate",
        AlarmDescription="More than 10 validation errors in a 5-minute window.",
        Namespace=NAMESPACE,
        MetricName="validation_errors",
        Statistic="Sum",
        Period=300,
        EvaluationPeriods=1,
        Threshold=10,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[topic_arn],
        OKActions=[topic_arn],
    )
    alarms.append("HumanGuard-ValidationErrorRate")

    # 4. ErrorRate — more than 5 Lambda errors in a 5-min window
    cw.put_metric_alarm(
        AlarmName="HumanGuard-ErrorRate",
        AlarmDescription="More than 5 Lambda errors in a 5-minute window.",
        Namespace=NAMESPACE,
        MetricName="lambda_errors",
        Statistic="Sum",
        Period=300,
        EvaluationPeriods=1,
        Threshold=5,
        ComparisonOperator="GreaterThanThreshold",
        TreatMissingData="notBreaching",
        AlarmActions=[topic_arn],
        OKActions=[topic_arn],
    )
    alarms.append("HumanGuard-ErrorRate")

    return alarms


if __name__ == "__main__":
    print(f"Setting up CloudWatch alarms in {REGION} (namespace={NAMESPACE})")
    topic_arn = get_or_create_topic()
    alarms = create_alarms(topic_arn)
    print(f"\nCreated/updated {len(alarms)} alarms:")
    for name in alarms:
        print(f"  {name}")
    print("\nDone.")
