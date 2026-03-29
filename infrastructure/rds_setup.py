"""
HumanGuard RDS PostgreSQL Setup — run once at deploy time.

Creates (or reuses) an RDS PostgreSQL instance and stores the connection
credentials in AWS Secrets Manager.

Usage:
    python infrastructure/rds_setup.py

Environment variables:
    AWS_REGION         — defaults to us-east-1
    AWS_ACCOUNT_ID     — defaults to 796793347388
    DB_PASSWORD        — master password (required; never hard-coded)
"""

import json
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

REGION = os.environ.get("AWS_REGION", "us-east-1")
ACCOUNT_ID = os.environ.get("AWS_ACCOUNT_ID", "796793347388")
DB_INSTANCE_ID = "humanguard-db"
DB_NAME = "humanguard"
DB_USERNAME = "humanguard_admin"
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
SECRET_NAME = "humanGuard/rds"
SG_NAME = "humanguard-rds-sg"

ec2 = boto3.client("ec2", region_name=REGION)
rds = boto3.client("rds", region_name=REGION)
sm = boto3.client("secretsmanager", region_name=REGION)


# ── Security Group ────────────────────────────────────────────────────────────

def get_or_create_security_group() -> str:
    """Return the security group ID that allows inbound PostgreSQL (5432)."""
    # Check if it already exists
    resp = ec2.describe_security_groups(
        Filters=[{"Name": "group-name", "Values": [SG_NAME]}]
    )
    if resp["SecurityGroups"]:
        sg_id = resp["SecurityGroups"][0]["GroupId"]
        print(f"Security group already exists: {sg_id}")
        return sg_id

    # Create a new security group in the default VPC
    vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])
    vpc_id = vpcs["Vpcs"][0]["VpcId"]

    sg = ec2.create_security_group(
        GroupName=SG_NAME,
        Description="HumanGuard RDS PostgreSQL access",
        VpcId=vpc_id,
    )
    sg_id = sg["GroupId"]
    print(f"Security group created: {sg_id}")

    # Allow inbound PostgreSQL from anywhere (restrict in production)
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[{
            "IpProtocol": "tcp",
            "FromPort": 5432,
            "ToPort": 5432,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "PostgreSQL"}],
        }],
    )
    print("Inbound rule for port 5432 added.")
    return sg_id


# ── RDS Instance ──────────────────────────────────────────────────────────────

def get_or_create_rds(sg_id: str) -> dict:
    """Return RDS instance info, creating it if it doesn't exist."""
    try:
        resp = rds.describe_db_instances(DBInstanceIdentifier=DB_INSTANCE_ID)
        instance = resp["DBInstances"][0]
        print(f"RDS instance already exists: {DB_INSTANCE_ID}")
        return instance
    except ClientError as e:
        if e.response["Error"]["Code"] != "DBInstanceNotFound":
            raise

    if not DB_PASSWORD:
        print("ERROR: DB_PASSWORD environment variable is required to create the RDS instance.")
        sys.exit(1)

    print(f"Creating RDS instance '{DB_INSTANCE_ID}' (this takes ~5 minutes)...")
    rds.create_db_instance(
        DBInstanceIdentifier=DB_INSTANCE_ID,
        DBInstanceClass="db.t3.micro",
        Engine="postgres",
        EngineVersion="15",
        MasterUsername=DB_USERNAME,
        MasterUserPassword=DB_PASSWORD,
        DBName=DB_NAME,
        AllocatedStorage=20,
        StorageType="gp2",
        PubliclyAccessible=True,
        VpcSecurityGroupIds=[sg_id],
        BackupRetentionPeriod=0,
        MultiAZ=False,
        Tags=[{"Key": "Project", "Value": "HumanGuard"}],
    )

    # Wait for the instance to become available
    print("Waiting for RDS instance to become available (up to 10 minutes)...")
    waiter = rds.get_waiter("db_instance_available")
    waiter.wait(
        DBInstanceIdentifier=DB_INSTANCE_ID,
        WaiterConfig={"Delay": 30, "MaxAttempts": 20},
    )

    resp = rds.describe_db_instances(DBInstanceIdentifier=DB_INSTANCE_ID)
    instance = resp["DBInstances"][0]
    print(f"RDS instance is available: {instance['Endpoint']['Address']}")
    return instance


# ── Secrets Manager ───────────────────────────────────────────────────────────

def store_secret(instance: dict):
    """Store the RDS connection details in Secrets Manager."""
    host = instance["Endpoint"]["Address"]
    port = instance["Endpoint"]["Port"]
    secret_value = json.dumps({
        "host": host,
        "port": port,
        "dbname": DB_NAME,
        "username": DB_USERNAME,
        "password": DB_PASSWORD,
    })

    try:
        sm.describe_secret(SecretId=SECRET_NAME)
        # Secret exists — update it
        sm.put_secret_value(SecretId=SECRET_NAME, SecretString=secret_value)
        print(f"Secret '{SECRET_NAME}' updated.")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        # Create new secret
        sm.create_secret(
            Name=SECRET_NAME,
            Description="HumanGuard RDS PostgreSQL credentials",
            SecretString=secret_value,
            Tags=[{"Key": "Project", "Value": "HumanGuard"}],
        )
        print(f"Secret '{SECRET_NAME}' created.")

    database_url = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{host}:{port}/{DB_NAME}"
    print(f"\nDATABASE_URL (set this on Lambda):")
    print(f"  {database_url}")
    return database_url


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"=== HumanGuard RDS Setup (region={REGION}) ===\n")

    sg_id = get_or_create_security_group()
    instance = get_or_create_rds(sg_id)
    database_url = store_secret(instance)

    print("\n=== Setup complete ===")
    print("Next steps:")
    print("  1. Set DATABASE_URL on Lambda:")
    print(f"     aws lambda update-function-configuration \\")
    print(f"       --function-name humanguard \\")
    print(f"       --environment Variables={{DATABASE_URL={database_url},CLOUDWATCH_ENABLED=true}}")
    print("  2. Run migrations: DATABASE_URL=... python -m backend.db.migrate")
