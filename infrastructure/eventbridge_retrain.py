"""
HumanGuard EventBridge Retraining Schedule

Sets up a Lambda function (humanguard-retrain) and an EventBridge rule that
invokes it every 6 hours to check whether automated retraining is needed.

Idempotent — safe to run multiple times.

Usage:
    python infrastructure/eventbridge_retrain.py

Environment variables:
    AWS_REGION      — defaults to us-east-1
    AWS_ACCOUNT_ID  — optional override; resolved via STS if not set
"""

import json
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

REGION     = os.environ.get("AWS_REGION", "us-east-1")
# Resolve account ID at runtime; allow an explicit env override for cross-account deploys.
ACCOUNT_ID = os.environ.get(
    "AWS_ACCOUNT_ID",
    boto3.client("sts").get_caller_identity()["Account"],
)

ECR_REPO_NAME   = "humanguard"
FUNCTION_NAME   = "humanguard-retrain"
ROLE_NAME       = "humanguard-retrain-role"
RULE_NAME       = "humanguard-retrain-schedule"
# EventBridge cron: every 6 hours at :00 (UTC)
SCHEDULE_EXPR   = "rate(6 hours)"
IMAGE_URI       = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPO_NAME}:latest"
# Lambda CMD override — invokes the retrain_lambda handler instead of the Flask app
LAMBDA_CMD      = ["python", "scripts/retrain_lambda.py"]

iam      = boto3.client("iam",           region_name=REGION)
lam      = boto3.client("lambda",        region_name=REGION)
events   = boto3.client("events",        region_name=REGION)
sm       = boto3.client("secretsmanager", region_name=REGION)


# ── IAM Role ──────────────────────────────────────────────────────────────────

def get_or_create_role() -> str:
    """Return the ARN of the retrain Lambda execution role."""
    try:
        resp = iam.get_role(RoleName=ROLE_NAME)
        arn = resp["Role"]["Arn"]
        print(f"IAM role already exists: {arn}")
        return arn
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    trust = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }],
    })
    resp = iam.create_role(
        RoleName=ROLE_NAME,
        AssumeRolePolicyDocument=trust,
        Description="HumanGuard automated retraining Lambda execution role",
    )
    arn = resp["Role"]["Arn"]
    print(f"IAM role created: {arn}")

    for policy_arn in [
        "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
        "arn:aws:iam::aws:policy/AmazonSNSFullAccess",
        "arn:aws:iam::aws:policy/SecretsManagerReadWrite",
    ]:
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=policy_arn)
        print(f"  Attached {policy_arn.split('/')[-1]}")

    print("Waiting 15 seconds for IAM role propagation…")
    time.sleep(15)
    return arn


# ── Lambda Function ───────────────────────────────────────────────────────────

def _env_vars() -> dict:
    """Build environment variable dict for the retrain Lambda."""
    env = {
        "PYTHONPATH": "/var/task",
        "SNS_ALERT_TOPIC": f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:HumanGuard-Alerts",
    }

    # Inherit DATABASE_URL and other secrets from the main function if set
    for key in ("DATABASE_URL", "HUMANGUARD_MASTER_KEY", "EXPORT_API_KEY", "MODEL_BUCKET", "API_URL"):
        val = os.environ.get(key, "")
        if val:
            env[key] = val

    # Try to fetch DATABASE_URL from Secrets Manager
    if "DATABASE_URL" not in env:
        try:
            secret_json = sm.get_secret_value(
                SecretId="humanGuard/rds", Region=REGION
            )["SecretString"]
            d = json.loads(secret_json)
            env["DATABASE_URL"] = (
                f"postgresql://{d['username']}:{d['password']}"
                f"@{d['host']}:{d['port']}/{d['dbname']}"
            )
            print("DATABASE_URL loaded from Secrets Manager.")
        except Exception:
            pass

    return env


def get_or_create_lambda(role_arn: str) -> str:
    """Create or update the retrain Lambda function; return its ARN."""
    env_vars = {"Variables": _env_vars()}

    try:
        resp = lam.get_function(FunctionName=FUNCTION_NAME)
        fn_arn = resp["Configuration"]["FunctionArn"]
        print(f"Lambda function already exists: {fn_arn}")

        # Update code to match the latest ECR image
        lam.update_function_code(
            FunctionName=FUNCTION_NAME,
            ImageUri=IMAGE_URI,
        )
        waiter = lam.get_waiter("function_updated_v2")
        waiter.wait(FunctionName=FUNCTION_NAME)

        lam.update_function_configuration(
            FunctionName=FUNCTION_NAME,
            Timeout=600,
            MemorySize=1024,
            Environment=env_vars,
            ImageConfig={"Command": LAMBDA_CMD},
        )
        waiter.wait(FunctionName=FUNCTION_NAME)
        print(f"Lambda function '{FUNCTION_NAME}' updated.")
        return fn_arn

    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    resp = lam.create_function(
        FunctionName=FUNCTION_NAME,
        PackageType="Image",
        Code={"ImageUri": IMAGE_URI},
        Role=role_arn,
        Timeout=600,
        MemorySize=1024,
        Environment=env_vars,
        ImageConfig={"Command": LAMBDA_CMD},
        Description="HumanGuard automated retraining — triggered by EventBridge every 6 h",
    )
    fn_arn = resp["FunctionArn"]
    print(f"Waiting for Lambda function to become active…")
    waiter = lam.get_waiter("function_active_v2")
    waiter.wait(FunctionName=FUNCTION_NAME)
    print(f"Lambda function '{FUNCTION_NAME}' created: {fn_arn}")
    return fn_arn


# ── EventBridge Rule ──────────────────────────────────────────────────────────

def get_or_create_rule(fn_arn: str) -> str:
    """Create (or update) the 6-hour EventBridge rule; return its ARN."""
    resp = events.put_rule(
        Name=RULE_NAME,
        ScheduleExpression=SCHEDULE_EXPR,
        State="ENABLED",
        Description="Triggers HumanGuard automated retraining every 6 hours",
    )
    rule_arn = resp["RuleArn"]
    print(f"EventBridge rule '{RULE_NAME}' set: {rule_arn}")

    events.put_targets(
        Rule=RULE_NAME,
        Targets=[{
            "Id": "humanguard-retrain-target",
            "Arn": fn_arn,
            "Input": json.dumps({"source": "eventbridge", "schedule": SCHEDULE_EXPR}),
        }],
    )
    print("EventBridge target set.")
    return rule_arn


def grant_eventbridge_permission(fn_arn: str):
    """Allow EventBridge to invoke the Lambda function."""
    try:
        lam.add_permission(
            FunctionName=FUNCTION_NAME,
            StatementId="eventbridge-retrain-invoke",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=f"arn:aws:events:{REGION}:{ACCOUNT_ID}:rule/{RULE_NAME}",
        )
        print("EventBridge → Lambda invoke permission granted.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceConflictException":
            print("Permission already exists, skipping.")
        else:
            raise


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== HumanGuard EventBridge Retrain Setup (region={REGION}) ===\n")

    role_arn = get_or_create_role()
    fn_arn   = get_or_create_lambda(role_arn)
    rule_arn = get_or_create_rule(fn_arn)
    grant_eventbridge_permission(fn_arn)

    print("\n=== Setup complete ===")
    print(f"  Lambda function : {fn_arn}")
    print(f"  EventBridge rule: {rule_arn}")
    print(f"  Schedule        : {SCHEDULE_EXPR}")
    print(f"  Threshold       : 50 new labeled sessions per check")
    print()
    print("To check the next scheduled invocation:")
    print(f"  aws events describe-rule --name {RULE_NAME} --region {REGION}")
    print()
    print("To manually trigger a retrain check:")
    print(f"  aws lambda invoke --function-name {FUNCTION_NAME} "
          f"--payload '{{\"source\":\"manual\"}}' /tmp/retrain_result.json")
    print(f"  cat /tmp/retrain_result.json")
