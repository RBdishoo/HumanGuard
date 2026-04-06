"""
HumanGuard CloudFront Distribution Setup — run once at deploy time.

Creates (or reuses) a CloudFront distribution in front of the S3 static
website bucket, enabling HTTPS with cache behaviours tuned for a frontend
that calls a separate Lambda API.

Usage:
    python infrastructure/cloudfront_setup.py

Environment variables:
    AWS_REGION  — informational only; CloudFront is a global service
"""

import os
import sys

import boto3
from botocore.exceptions import ClientError

# Set FRONTEND_S3_BUCKET to the name of the S3 static-website bucket.
# AWS_REGION is used to construct the correct regional S3 website endpoint.
_bucket = os.environ.get("FRONTEND_S3_BUCKET", "")
_region = os.environ.get("AWS_REGION", "us-east-1")
if not _bucket:
    sys.exit("ERROR: FRONTEND_S3_BUCKET environment variable is required.")
S3_WEBSITE_DOMAIN = f"{_bucket}.s3-website-{_region}.amazonaws.com"
DISTRIBUTION_COMMENT = "HumanGuard frontend HTTPS distribution"
CALLER_REFERENCE = "humanguard-frontend-v1"

cf = boto3.client("cloudfront", region_name="us-east-1")


# ── Idempotency check ─────────────────────────────────────────────────────────

def find_existing_distribution():
    """Return the first distribution whose Comment matches, or None."""
    paginator = cf.get_paginator("list_distributions")
    for page in paginator.paginate():
        items = page.get("DistributionList", {}).get("Items", [])
        for dist in items:
            if dist.get("Comment") == DISTRIBUTION_COMMENT:
                return dist
    return None


# ── Distribution config ───────────────────────────────────────────────────────

def _get_methods(full=False):
    if full:
        return {
            "Quantity": 7,
            "Items": ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"],
            "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]},
        }
    return {
        "Quantity": 2,
        "Items": ["GET", "HEAD"],
        "CachedMethods": {"Quantity": 2, "Items": ["GET", "HEAD"]},
    }


def _forwarded_all():
    return {
        "QueryString": True,
        "Cookies": {"Forward": "all"},
        "Headers": {"Quantity": 0, "Items": []},
    }


def _forwarded_none():
    return {
        "QueryString": False,
        "Cookies": {"Forward": "none"},
        "Headers": {"Quantity": 0, "Items": []},
    }


def build_distribution_config():
    return {
        "Comment": DISTRIBUTION_COMMENT,
        "DefaultRootObject": "demo.html",
        "Enabled": True,
        "HttpVersion": "http2",
        # ── Origin ────────────────────────────────────────────────────────────
        "Origins": {
            "Quantity": 1,
            "Items": [
                {
                    "Id": "S3WebsiteOrigin",
                    "DomainName": S3_WEBSITE_DOMAIN,
                    # S3 static-website endpoints are HTTP-only; CloudFront
                    # handles TLS termination for viewers.
                    "CustomOriginConfig": {
                        "HTTPPort": 80,
                        "HTTPSPort": 443,
                        "OriginProtocolPolicy": "http-only",
                    },
                }
            ],
        },
        # ── Ordered cache behaviors ───────────────────────────────────────────
        "CacheBehaviors": {
            "Quantity": 4,
            "Items": [
                # 1. /api/* — no-cache pass-through (API calls go directly to
                #    Lambda via config.js; this path exists as a safety net).
                {
                    "PathPattern": "/api/*",
                    "TargetOriginId": "S3WebsiteOrigin",
                    "ViewerProtocolPolicy": "redirect-to-https",
                    "AllowedMethods": _get_methods(full=True),
                    "ForwardedValues": _forwarded_all(),
                    "MinTTL": 0,
                    "DefaultTTL": 0,
                    "MaxTTL": 0,
                    "Compress": False,
                },
                # 2. *.html — short TTL so frontend updates propagate quickly
                {
                    "PathPattern": "*.html",
                    "TargetOriginId": "S3WebsiteOrigin",
                    "ViewerProtocolPolicy": "redirect-to-https",
                    "AllowedMethods": _get_methods(),
                    "ForwardedValues": _forwarded_none(),
                    "MinTTL": 0,
                    "DefaultTTL": 300,
                    "MaxTTL": 300,
                    "Compress": True,
                },
                # 3. *.js — longer TTL (1 day)
                {
                    "PathPattern": "*.js",
                    "TargetOriginId": "S3WebsiteOrigin",
                    "ViewerProtocolPolicy": "redirect-to-https",
                    "AllowedMethods": _get_methods(),
                    "ForwardedValues": _forwarded_none(),
                    "MinTTL": 0,
                    "DefaultTTL": 86400,
                    "MaxTTL": 86400,
                    "Compress": True,
                },
                # 4. *.css — longer TTL (1 day)
                {
                    "PathPattern": "*.css",
                    "TargetOriginId": "S3WebsiteOrigin",
                    "ViewerProtocolPolicy": "redirect-to-https",
                    "AllowedMethods": _get_methods(),
                    "ForwardedValues": _forwarded_none(),
                    "MinTTL": 0,
                    "DefaultTTL": 86400,
                    "MaxTTL": 86400,
                    "Compress": True,
                },
            ],
        },
        # ── Default cache behavior ────────────────────────────────────────────
        "DefaultCacheBehavior": {
            "TargetOriginId": "S3WebsiteOrigin",
            "ViewerProtocolPolicy": "redirect-to-https",
            "AllowedMethods": _get_methods(),
            "ForwardedValues": _forwarded_none(),
            "MinTTL": 0,
            "DefaultTTL": 86400,
            "MaxTTL": 86400,
            "Compress": True,
        },
        # ── Custom error pages ────────────────────────────────────────────────
        # All unknown paths land on demo.html so deep-links work.
        "CustomErrorResponses": {
            "Quantity": 1,
            "Items": [
                {
                    "ErrorCode": 404,
                    "ResponsePagePath": "/demo.html",
                    "ResponseCode": "200",
                    "ErrorCachingMinTTL": 0,
                }
            ],
        },
        # ── Viewer certificate (CloudFront default domain) ────────────────────
        "ViewerCertificate": {
            "CloudFrontDefaultCertificate": True,
            "MinimumProtocolVersion": "TLSv1.2_2021",
        },
        # US, Canada, Europe — cheapest price class that covers most users
        "PriceClass": "PriceClass_100",
        "Restrictions": {
            "GeoRestriction": {"RestrictionType": "none", "Quantity": 0}
        },
    }


def create_distribution():
    config = build_distribution_config()
    resp = cf.create_distribution(
        DistributionConfig={**config, "CallerReference": CALLER_REFERENCE}
    )
    return resp["Distribution"]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("HumanGuard — CloudFront distribution setup")
    print(f"  S3 origin : {S3_WEBSITE_DOMAIN}\n")

    existing = find_existing_distribution()

    if existing:
        domain = existing["DomainName"]
        dist_id = existing["Id"]
        status = existing["Status"]
        print(f"Existing distribution found — no changes made.")
        print(f"  ID      : {dist_id}")
        print(f"  Status  : {status}")
    else:
        print("No existing distribution — creating …")
        dist = create_distribution()
        domain = dist["DomainName"]
        dist_id = dist["Id"]
        status = dist["Status"]
        print(f"Distribution created!")
        print(f"  ID      : {dist_id}")
        print(f"  Status  : {status}")

    print(f"\nCloudFront URL : https://{domain}")
    print()
    print("Cache behaviors:")
    print("  /api/*  → TTL 0        (no cache)")
    print("  *.html  → TTL 300 s    (5 minutes)")
    print("  *.js    → TTL 86400 s  (1 day)")
    print("  *.css   → TTL 86400 s  (1 day)")
    print("  default → TTL 86400 s  (1 day)")
    print()
    print("Custom error : 404 → /demo.html (HTTP 200)")
    print("Viewer protocol : HTTP redirects to HTTPS")
    print()
    print("⚠  CloudFront distributions take 10–15 minutes to propagate globally.")
    print(
        f"   Poll: aws cloudfront get-distribution --id {dist_id}"
        " --query 'Distribution.Status'"
    )
