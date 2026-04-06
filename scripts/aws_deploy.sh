#!/usr/bin/env bash
#
# AWS Deployment Script for HumanGuard
# Deploys the Flask bot-detection API as a Lambda container behind API Gateway.
#
# Prerequisites:
#   - AWS CLI v2 configured with credentials (aws sts get-caller-identity)
#   - Docker running locally
#   - Dockerfile and requirements-prod.txt in repo root
#
# Usage:
#   chmod +x scripts/aws_deploy.sh
#   ./scripts/aws_deploy.sh

set -e

# --- CONFIG ---
# Resolve account ID at runtime so no literal ID is stored in source control.
# Set AWS_ACCOUNT_ID in the environment to override (e.g. for cross-account deploys).
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
AWS_REGION=us-east-1
REPO_NAME=humanguard
FUNCTION_NAME=humanguard
ROLE_NAME=humanguard-lambda-role
IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$REPO_NAME:latest
API_NAME=humanguard-api
CLOUDWATCH_ENABLED=true
SNS_ALERT_EMAIL=${SNS_ALERT_EMAIL:-"rbdishoo@gmail.com"}
SENDER_EMAIL=${SENDER_EMAIL:-"noreply@humanguard.net"}
DB_MAX_CONNECTIONS=5
ALLOWED_ORIGINS_PROD="https://humanguard.net,https://www.humanguard.net,https://d1hi33wespusty.cloudfront.net"

# Fetch master key from Secrets Manager; generate and store one on first run.
HUMANGUARD_MASTER_KEY=${HUMANGUARD_MASTER_KEY:-""}
if [ -z "$HUMANGUARD_MASTER_KEY" ]; then
  if MASTER_KEY_JSON=$(aws secretsmanager get-secret-value \
      --secret-id "humanGuard/masterKey" \
      --region "$AWS_REGION" \
      --query "SecretString" \
      --output text 2>/dev/null); then
    HUMANGUARD_MASTER_KEY=$(echo "$MASTER_KEY_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['key'])")
    echo "Master key loaded from Secrets Manager."
  else
    HUMANGUARD_MASTER_KEY="hg_master_$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
    aws secretsmanager create-secret \
        --name "humanGuard/masterKey" \
        --secret-string "{\"key\":\"$HUMANGUARD_MASTER_KEY\"}" \
        --region "$AWS_REGION" > /dev/null
    echo "New master key generated and stored in Secrets Manager as humanGuard/masterKey."
    echo "  HUMANGUARD_MASTER_KEY: $HUMANGUARD_MASTER_KEY"
  fi
fi

# Fetch EXPORT_API_KEY from Secrets Manager; generate and store one on first run.
EXPORT_API_KEY=${EXPORT_API_KEY:-""}
if [ -z "$EXPORT_API_KEY" ]; then
  if EXPORT_KEY_JSON=$(aws secretsmanager get-secret-value \
      --secret-id "humanGuard/exportKey" \
      --region "$AWS_REGION" \
      --query "SecretString" \
      --output text 2>/dev/null); then
    EXPORT_API_KEY=$(echo "$EXPORT_KEY_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['key'])")
    echo "Export API key loaded from Secrets Manager."
  else
    EXPORT_API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    aws secretsmanager create-secret \
        --name "humanGuard/exportKey" \
        --secret-string "{\"key\":\"$EXPORT_API_KEY\"}" \
        --region "$AWS_REGION" > /dev/null
    echo "New export API key generated and stored in Secrets Manager as humanGuard/exportKey."
  fi
fi

# Optional: set FRONTEND_S3_BUCKET to upload all frontend/ files to S3
# e.g. FRONTEND_S3_BUCKET=humanguard-frontend ./scripts/aws_deploy.sh
FRONTEND_S3_BUCKET=${FRONTEND_S3_BUCKET:-""}
# If a frontend bucket is configured, add its S3 website origin to CORS allowed list.
if [ -n "$FRONTEND_S3_BUCKET" ]; then
    ALLOWED_ORIGINS_PROD="$ALLOWED_ORIGINS_PROD,http://$FRONTEND_S3_BUCKET.s3-website-$AWS_REGION.amazonaws.com"
fi

# Fetch DATABASE_URL from Secrets Manager (set to empty if secret not yet created)
DATABASE_URL=""
if SECRET_JSON=$(aws secretsmanager get-secret-value \
        --secret-id "humanGuard/rds" \
        --region "$AWS_REGION" \
        --query "SecretString" \
        --output text 2>/dev/null); then
    DB_HOST=$(echo "$SECRET_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['host'])")
    DB_PORT=$(echo "$SECRET_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['port'])")
    DB_NAME=$(echo "$SECRET_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['dbname'])")
    DB_USER=$(echo "$SECRET_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['username'])")
    DB_PASS=$(echo "$SECRET_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['password'])")
    DATABASE_URL="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
    echo "DATABASE_URL loaded from Secrets Manager."
else
    echo "Warning: humanGuard/rds secret not found — Lambda will run without PostgreSQL."
fi

echo "=== HumanGuard AWS Deployment ==="
echo "Account:  $AWS_ACCOUNT_ID"
echo "Region:   $AWS_REGION"
echo "Function: $FUNCTION_NAME"
echo ""

# ---------------------------------------------------------------
# STEP 1: Create ECR repository
# Creates a private container registry to store the Docker image.
# Skips gracefully if the repository already exists.
# ---------------------------------------------------------------
echo "--- Step 1: Create ECR repository ---"
aws ecr create-repository \
    --repository-name "$REPO_NAME" \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null \
    && echo "ECR repository '$REPO_NAME' created." \
    || echo "ECR repository '$REPO_NAME' already exists, skipping."

# ---------------------------------------------------------------
# STEP 2: Authenticate Docker to ECR
# Retrieves a temporary auth token and pipes it to docker login
# so subsequent push commands are authorized.
# ---------------------------------------------------------------
echo ""
echo "--- Step 2: Authenticate Docker to ECR ---"
aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin \
    "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
echo "Docker authenticated to ECR."

# ---------------------------------------------------------------
# STEP 3: Build, tag, and push image
# Builds from the repo-root Dockerfile, tags as :latest, and
# pushes to the ECR repository created in Step 1.
# ---------------------------------------------------------------
echo ""
echo "--- Step 3: Build, tag, and push image ---"

# Build the Docker image from the repository root
docker build --platform linux/amd64 -t "$REPO_NAME:latest" .

# Tag the image for ECR
docker tag "$REPO_NAME:latest" "$IMAGE_URI"

# Push the image to ECR
docker push "$IMAGE_URI"
echo "Image pushed to $IMAGE_URI"

# ---------------------------------------------------------------
# STEP 4: Create IAM role for Lambda
# Creates an execution role that Lambda assumes to run the
# container. The trust policy allows lambda.amazonaws.com to
# call sts:AssumeRole. Skips if the role already exists.
# ---------------------------------------------------------------
echo ""
echo "--- Step 4: Create IAM role for Lambda ---"

TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "lambda.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}'

aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document "$TRUST_POLICY" \
    2>/dev/null \
    && echo "IAM role '$ROLE_NAME' created." \
    || echo "IAM role '$ROLE_NAME' already exists, skipping."

ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${ROLE_NAME}"

# ---------------------------------------------------------------
# STEP 5: Attach execution policy to role
# Grants the Lambda function permission to write CloudWatch logs.
# attach-role-policy is idempotent — safe to run repeatedly.
# ---------------------------------------------------------------
echo ""
echo "--- Step 5: Attach execution policy to role ---"

# Attach the AWS-managed basic execution policy for CloudWatch Logs access
aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
echo "AWSLambdaBasicExecutionRole attached to '$ROLE_NAME'."

# Grant Lambda permission to send emails via SES
SES_POLICY_NAME="humanguard-ses-send"
SES_POLICY_DOC='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ses:SendEmail",
        "ses:SendRawEmail"
      ],
      "Resource": "*"
    }
  ]
}'
aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name "$SES_POLICY_NAME" \
    --policy-document "$SES_POLICY_DOC" \
    2>/dev/null \
    && echo "SES send policy '$SES_POLICY_NAME' attached to '$ROLE_NAME'." \
    || echo "SES send policy already exists or attachment failed — continuing."

# Allow IAM role to propagate before Lambda creation
echo "Waiting 10 seconds for IAM role propagation..."
sleep 10

# ---------------------------------------------------------------
# STEP 6: Create or update Lambda function (image-based)
# Creates the function on first deploy; updates code + config on
# subsequent deploys. Waits for the function to reach Active/Updated.
# ---------------------------------------------------------------
echo ""
echo "--- Step 6: Create or update Lambda function ---"

# Build env JSON via python3 so commas inside ALLOWED_ORIGINS don't break parsing.
ENV_VARS=$(env \
    _PORT="8080" \
    _CW="$CLOUDWATCH_ENABLED" \
    _SNS="$SNS_ALERT_EMAIL" \
    _DB="$DATABASE_URL" \
    _DBMAX="$DB_MAX_CONNECTIONS" \
    _MK="$HUMANGUARD_MASTER_KEY" \
    _EK="$EXPORT_API_KEY" \
    _AO="$ALLOWED_ORIGINS_PROD" \
    _SE="$SENDER_EMAIL" \
    python3 -c "
import json, os
print(json.dumps({'Variables': {
    'PORT':                 os.environ['_PORT'],
    'CLOUDWATCH_ENABLED':   os.environ['_CW'],
    'SNS_ALERT_EMAIL':      os.environ['_SNS'],
    'DATABASE_URL':         os.environ['_DB'],
    'DB_MAX_CONNECTIONS':   os.environ['_DBMAX'],
    'HUMANGUARD_MASTER_KEY':os.environ['_MK'],
    'EXPORT_API_KEY':       os.environ['_EK'],
    'ALLOWED_ORIGINS':      os.environ['_AO'],
    'SENDER_EMAIL':         os.environ['_SE'],
}}))")

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$AWS_REGION" > /dev/null 2>&1; then
    echo "Lambda function '$FUNCTION_NAME' exists — updating code and configuration."
    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --image-uri "$IMAGE_URI" \
        --region "$AWS_REGION" > /dev/null
    echo "Waiting for code update to complete..."
    aws lambda wait function-updated-v2 \
        --function-name "$FUNCTION_NAME" \
        --region "$AWS_REGION"
    aws lambda update-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --memory-size 1024 \
        --timeout 30 \
        --environment "$ENV_VARS" \
        --region "$AWS_REGION" > /dev/null
    echo "Waiting for configuration update to complete..."
    aws lambda wait function-updated-v2 \
        --function-name "$FUNCTION_NAME" \
        --region "$AWS_REGION"
    echo "Lambda function '$FUNCTION_NAME' updated."
else
    aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --package-type Image \
        --code "ImageUri=$IMAGE_URI" \
        --role "$ROLE_ARN" \
        --memory-size 1024 \
        --timeout 30 \
        --environment "$ENV_VARS" \
        --region "$AWS_REGION"
    echo "Waiting for Lambda function to become Active..."
    aws lambda wait function-active-v2 \
        --function-name "$FUNCTION_NAME" \
        --region "$AWS_REGION"
    echo "Lambda function '$FUNCTION_NAME' created."
fi

# ---------------------------------------------------------------
# STEP 7: Create or reuse API Gateway HTTP API
# On first deploy, creates the API; on subsequent deploys, reuses
# the existing one by name lookup.
# ---------------------------------------------------------------
echo ""
echo "--- Step 7: Create or reuse API Gateway HTTP API ---"

# Check if API already exists
API_ID=$(aws apigatewayv2 get-apis \
    --region "$AWS_REGION" \
    --query "Items[?Name=='$API_NAME'].ApiId | [0]" \
    --output text 2>/dev/null)

if [ -n "$API_ID" ] && [ "$API_ID" != "None" ]; then
    echo "API Gateway '$API_NAME' already exists: API_ID=$API_ID"
else
    API_ID=$(aws apigatewayv2 create-api \
        --name "$API_NAME" \
        --protocol-type HTTP \
        --target "arn:aws:lambda:${AWS_REGION}:${AWS_ACCOUNT_ID}:function:${FUNCTION_NAME}" \
        --region "$AWS_REGION" \
        --query "ApiId" \
        --output text)
    echo "API Gateway created: API_ID=$API_ID"

    INTEGRATION_ID=$(aws apigatewayv2 get-integrations \
        --api-id "$API_ID" \
        --region "$AWS_REGION" \
        --query "Items[0].IntegrationId" \
        --output text)

    aws apigatewayv2 create-route \
        --api-id "$API_ID" \
        --route-key 'ANY /{proxy+}' \
        --target "integrations/$INTEGRATION_ID" \
        --region "$AWS_REGION" > /dev/null
    echo "Route ANY /{proxy+} created."
fi

# ---------------------------------------------------------------
# STEP 8: Grant API Gateway permission to invoke Lambda
# Skipped gracefully if the permission already exists.
# ---------------------------------------------------------------
echo ""
echo "--- Step 8: Grant API Gateway → Lambda permission ---"

aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "apigateway-invoke" \
    --action "lambda:InvokeFunction" \
    --principal "apigateway.amazonaws.com" \
    --source-arn "arn:aws:execute-api:${AWS_REGION}:${AWS_ACCOUNT_ID}:${API_ID}/*" \
    --region "$AWS_REGION" > /dev/null 2>&1 \
    && echo "Permission granted." \
    || echo "Permission already exists, skipping."

# ---------------------------------------------------------------
# STEP 9: Print the live API Gateway URL
# The default stage for HTTP APIs is $default, served at the root.
# ---------------------------------------------------------------
echo ""
echo "--- Step 9: Deployment complete ---"

API_URL="https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com"
echo ""
echo "============================================"
echo "  HumanGuard is live at:"
echo "  $API_URL"
echo "============================================"
echo ""
echo "Endpoints:"
echo "  GET  $API_URL/health"
echo "  GET  $API_URL/api/stats"
echo "  POST $API_URL/api/signals"
echo "  POST $API_URL/api/score"
echo "  GET  $API_URL/demo          (public demo page)"
echo "  GET  $API_URL/leaderboard   (public leaderboard)"
echo "  GET  $API_URL/simulate      (internal bot simulator)"
echo "  GET  $API_URL/api/leaderboard   (leaderboard JSON — GET top 20)"
echo "  POST $API_URL/api/leaderboard   (submit nickname + session_id)"
echo "  GET  $API_URL/api/export    (CSV export — requires X-Export-Key header)"
echo ""

# ---------------------------------------------------------------
# STEP 10: Smoke test — curl /health and /api/stats
# Waits 5 seconds for Lambda cold start, then verifies the
# deployment is responding.
# ---------------------------------------------------------------
echo "--- Step 10: Smoke test ---"
echo "Waiting 5 seconds for Lambda initialization..."
sleep 5

echo ""
echo "GET /health:"
curl -s "$API_URL/health" | python3 -m json.tool
echo ""

echo "GET /api/stats:"
curl -s "$API_URL/api/stats" | python3 -m json.tool
echo ""

echo "=== Deployment complete ==="

# ---------------------------------------------------------------
# STEP 11 (optional): Upload frontend/ files to S3
# Set FRONTEND_S3_BUCKET env var to enable static hosting of
# demo.html, bot_simulator.html, dashboard.html, etc. on S3.
# ---------------------------------------------------------------
if [ -n "$FRONTEND_S3_BUCKET" ]; then
    echo ""
    echo "--- Step 11: Upload frontend/ files to S3 ---"

    # Create bucket if it doesn't exist
    aws s3api create-bucket \
        --bucket "$FRONTEND_S3_BUCKET" \
        --region "$AWS_REGION" \
        --create-bucket-configuration LocationConstraint="$AWS_REGION" \
        2>/dev/null \
        || echo "Bucket '$FRONTEND_S3_BUCKET' already exists."

    # Enable static website hosting
    aws s3 website "s3://$FRONTEND_S3_BUCKET" \
        --index-document index.html \
        --error-document index.html

    # Sync all frontend/ files (bucket policy handles public read — ACL not needed)
    aws s3 sync frontend/ "s3://$FRONTEND_S3_BUCKET/" \
        --delete \
        --region "$AWS_REGION"

    echo "Frontend files uploaded to s3://$FRONTEND_S3_BUCKET"
    echo "Static site URL: http://$FRONTEND_S3_BUCKET.s3-website-$AWS_REGION.amazonaws.com"
else
    echo ""
    echo "(Skipping S3 frontend upload — set FRONTEND_S3_BUCKET to enable)"
fi


# ---------------------------------------------------------------
# TEARDOWN — uncomment to delete all AWS resources created by this script
#
# Run these commands in order to fully remove the deployment.
# API_ID must match the value printed in Step 9 above.
# ---------------------------------------------------------------

# # Delete API Gateway HTTP API
# aws apigatewayv2 delete-api \
#     --api-id "$API_ID" \
#     --region "$AWS_REGION"
# echo "API Gateway '$API_NAME' deleted."

# # Delete Lambda function
# aws lambda delete-function \
#     --function-name "$FUNCTION_NAME" \
#     --region "$AWS_REGION"
# echo "Lambda function '$FUNCTION_NAME' deleted."

# # Delete all images in ECR repo, then delete the repo
# aws ecr batch-delete-image \
#     --repository-name "$REPO_NAME" \
#     --image-ids imageTag=latest \
#     --region "$AWS_REGION"
# aws ecr delete-repository \
#     --repository-name "$REPO_NAME" \
#     --region "$AWS_REGION"
# echo "ECR repository '$REPO_NAME' deleted."

# # Detach policy and delete IAM role
# aws iam detach-role-policy \
#     --role-name "$ROLE_NAME" \
#     --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
# aws iam delete-role \
#     --role-name "$ROLE_NAME"
# echo "IAM role '$ROLE_NAME' deleted."

# echo "=== Teardown complete ==="
