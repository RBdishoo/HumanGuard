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


# ── Schema Migration ──────────────────────────────────────────────────────────

def add_source_label_columns(database_url: str):
    """
    Add source, label, and trained_at columns to sessions and predictions tables if not present.
    Safe to run multiple times — uses IF NOT EXISTS guard.
    """
    import psycopg2

    migrations = [
        "ALTER TABLE sessions    ADD COLUMN IF NOT EXISTS source     VARCHAR(100)",
        "ALTER TABLE sessions    ADD COLUMN IF NOT EXISTS label      VARCHAR(10)",
        "ALTER TABLE sessions    ADD COLUMN IF NOT EXISTS trained_at TIMESTAMPTZ DEFAULT NULL",
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS source VARCHAR(100)",
    ]

    conn = psycopg2.connect(database_url)
    try:
        cur = conn.cursor()
        for sql in migrations:
            cur.execute(sql)
            print(f"  OK: {sql}")
        conn.commit()
        print("Migration complete.")
    except Exception as exc:
        conn.rollback()
        print(f"Migration failed: {exc}")
        raise
    finally:
        conn.close()


def add_leaderboard_table(database_url: str):
    """
    Create leaderboard table in PostgreSQL if it does not exist.
    Safe to run multiple times.
    """
    import psycopg2

    create_sql = """
    CREATE TABLE IF NOT EXISTS leaderboard (
        id          SERIAL PRIMARY KEY,
        nickname    VARCHAR(20) NOT NULL,
        prob_bot    REAL NOT NULL,
        verdict     VARCHAR(10) NOT NULL,
        session_id  TEXT NOT NULL,
        created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """

    conn = psycopg2.connect(database_url)
    try:
        cur = conn.cursor()
        cur.execute(create_sql)
        conn.commit()
        print("  OK: leaderboard table created (or already exists)")
    except Exception as exc:
        conn.rollback()
        print(f"Leaderboard migration failed: {exc}")
        raise
    finally:
        conn.close()


def add_api_keys_table(database_url: str):
    """
    Create api_keys table in PostgreSQL if it does not exist.
    Safe to run multiple times.
    """
    import psycopg2

    create_sql = """
    CREATE TABLE IF NOT EXISTS api_keys (
        id                  SERIAL PRIMARY KEY,
        key                 VARCHAR(64) UNIQUE NOT NULL,
        key_id              VARCHAR(16) UNIQUE,
        key_hash            VARCHAR(64),
        owner_email         VARCHAR(255) NOT NULL,
        plan                VARCHAR(20) NOT NULL DEFAULT 'free',
        monthly_limit       INTEGER NOT NULL DEFAULT 1000,
        current_month_count INTEGER NOT NULL DEFAULT 0,
        created_at          TIMESTAMPTZ DEFAULT NOW(),
        active              BOOLEAN NOT NULL DEFAULT TRUE
    );
    CREATE INDEX IF NOT EXISTS idx_api_keys_key    ON api_keys (key);
    CREATE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys (key_id);
    """

    conn = psycopg2.connect(database_url)
    try:
        cur = conn.cursor()
        cur.execute(create_sql)
        conn.commit()
        print("  OK: api_keys table created (or already exists)")
    except Exception as exc:
        conn.rollback()
        print(f"api_keys migration failed: {exc}")
        raise
    finally:
        conn.close()


def add_key_hash_columns(database_url: str):
    """
    Add key_id and key_hash columns to api_keys for hashed key storage.
    Safe to run multiple times — uses IF NOT EXISTS guard.
    """
    import psycopg2

    migrations = [
        "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS key_id   VARCHAR(16) UNIQUE",
        "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS key_hash VARCHAR(64)",
        "CREATE INDEX IF NOT EXISTS idx_api_keys_key_id ON api_keys (key_id)",
    ]

    conn = psycopg2.connect(database_url)
    try:
        cur = conn.cursor()
        for sql in migrations:
            cur.execute(sql)
            print(f"  OK: {sql}")
        conn.commit()
        print("key_hash migration complete.")
    except Exception as exc:
        conn.rollback()
        print(f"key_hash migration failed: {exc}")
        raise
    finally:
        conn.close()


def add_email_verification_columns(database_url: str):
    """
    Add email-verification columns to api_keys and create the webhooks table.
    Safe to run multiple times — uses IF NOT EXISTS / ADD COLUMN IF NOT EXISTS.
    """
    import psycopg2

    migrations = [
        # Email verification columns on api_keys
        "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS verified            BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS verification_token  VARCHAR(64)",
        "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS token_expires_at    TIMESTAMPTZ",
        "CREATE INDEX IF NOT EXISTS idx_api_keys_token ON api_keys (verification_token)",
        # Webhooks table (api_key_id is TEXT to accommodate master keys and any key format)
        """
        CREATE TABLE IF NOT EXISTS webhooks (
            id                  SERIAL PRIMARY KEY,
            api_key_id          TEXT NOT NULL,
            url                 TEXT NOT NULL,
            secret              TEXT NOT NULL,
            events              TEXT NOT NULL DEFAULT 'bot_detected',
            active              BOOLEAN NOT NULL DEFAULT TRUE,
            created_at          TIMESTAMPTZ DEFAULT NOW(),
            last_triggered_at   TIMESTAMPTZ,
            failure_count       INTEGER NOT NULL DEFAULT 0
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_webhooks_api_key_id ON webhooks (api_key_id)",
        # Widen api_key_id if table was previously created with VARCHAR(16)
        "ALTER TABLE webhooks ALTER COLUMN api_key_id TYPE TEXT",
    ]

    conn = psycopg2.connect(database_url)
    try:
        cur = conn.cursor()
        for sql in migrations:
            cur.execute(sql)
            print(f"  OK: {sql.strip().splitlines()[0][:80]}")
        conn.commit()
        print("email-verification + webhooks migration complete.")
    except Exception as exc:
        conn.rollback()
        print(f"Migration failed: {exc}")
        raise
    finally:
        conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HumanGuard RDS setup and migration")
    parser.add_argument("--migrate-only", action="store_true",
                        help="Run column migration against DATABASE_URL (no RDS creation)")
    args = parser.parse_args()

    if args.migrate_only:
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            print("ERROR: DATABASE_URL must be set for --migrate-only")
            sys.exit(1)
        print("=== HumanGuard Column Migration ===\n")
        add_source_label_columns(db_url)
        add_leaderboard_table(db_url)
        add_api_keys_table(db_url)
        add_key_hash_columns(db_url)
        add_email_verification_columns(db_url)
        sys.exit(0)

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
    print("  3. Run column migration: DATABASE_URL=... python infrastructure/rds_setup.py --migrate-only")
