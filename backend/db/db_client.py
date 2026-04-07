"""
PostgreSQL client for HumanGuard.

Connection resolution order:
  1. DATABASE_URL env var — parsed into libpq keyword arguments (local dev /
     explicit override).  sqlite:/// URLs get sslmode="disable"; postgres://
     URLs get sslmode="require".
  2. RDS_SECRET_NAME env var — fetched from AWS Secrets Manager at cold-start
     and cached for the container lifetime so Secrets Manager is never called
     more than once per Lambda instance.  Always enforces sslmode="require".

When neither is set, or the connection fails, every public function degrades
gracefully so the JSONL path keeps working.
"""

import os
import json
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_pool = None
_available = None  # tri-state: None = not checked, True/False = result
_db_kwargs = None  # resolved once per cold start; never stored in env


def _resolve_db_kwargs() -> dict:
    """
    Return libpq keyword arguments for psycopg2.connect(), resolving
    credentials exactly once per Lambda container lifetime.

    Fast path  — DATABASE_URL is set in the environment (local dev or an
                 explicit override).  Parsed with urlparse() so passwords
                 containing reserved URL characters (@ / : ?) are handled
                 correctly.  sqlite:/// URLs get sslmode="disable";
                 postgres:// URLs get sslmode="require".

    Cold-start — RDS_SECRET_NAME names a Secrets Manager secret whose JSON
                 payload contains {host, port, dbname, username, password}.
                 Always sets sslmode="require".  The result is cached in
                 _db_kwargs so subsequent calls within the same container
                 are instant.
    """
    global _db_kwargs
    if _db_kwargs is not None:
        return _db_kwargs

    # Fast path: explicit DATABASE_URL (local dev)
    url = os.environ.get("DATABASE_URL")
    if url:
        parsed = urlparse(url)
        if parsed.scheme.startswith("sqlite"):
            _db_kwargs = {"database": parsed.path, "sslmode": "disable"}
        else:
            _db_kwargs = {
                "host": parsed.hostname,
                "port": parsed.port or 5432,
                "dbname": parsed.path.lstrip("/"),
                "user": parsed.username,
                "password": parsed.password,
                "sslmode": "require",
            }
        return _db_kwargs

    # Cold-start path: fetch from Secrets Manager
    secret_name = os.environ.get("RDS_SECRET_NAME")
    if not secret_name:
        raise RuntimeError(
            "Neither DATABASE_URL nor RDS_SECRET_NAME is set — "
            "cannot connect to PostgreSQL"
        )

    import boto3
    region = os.environ.get("AWS_REGION", "us-east-1")
    sm = boto3.client("secretsmanager", region_name=region)
    try:
        resp = sm.get_secret_value(SecretId=secret_name)
        secret = json.loads(resp["SecretString"])
        _db_kwargs = {
            "host": secret["host"],
            "port": int(secret.get("port", 5432)),
            "dbname": secret["dbname"],
            "user": secret["username"],
            "password": secret["password"],
            "sslmode": "require",
        }
        logger.info("DB credentials fetched from Secrets Manager: %s", secret_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch DB credentials from Secrets Manager "
            f"(secret='{secret_name}'): {exc}"
        ) from exc

    return _db_kwargs


def _get_pool():
    """Lazily create the connection pool on first use."""
    global _pool
    if _pool is not None:
        return _pool

    import psycopg2
    from psycopg2.pool import SimpleConnectionPool

    _pool = SimpleConnectionPool(minconn=1, maxconn=5, **_resolve_db_kwargs())
    return _pool


def is_available():
    """Return True if DB kwargs can be resolved and a connection succeeds."""
    global _available
    if _available is not None:
        return _available

    try:
        _resolve_db_kwargs()
    except RuntimeError:
        _available = False
        return False

    try:
        pool = _get_pool()
        conn = pool.getconn()
        pool.putconn(conn)
        _available = True
    except Exception as exc:
        logger.warning("PostgreSQL unavailable: %s", exc)
        _available = False

    return _available


def get_connection():
    """Borrow a connection from the pool."""
    return _get_pool().getconn()


def release_connection(conn):
    """Return a connection to the pool."""
    try:
        _get_pool().putconn(conn)
    except Exception:
        pass


def execute_query(query, params=None, commit=True):
    """
    Execute a single query. Returns the cursor (useful for SELECT).
    Caller does NOT need to manage the connection.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        if commit:
            conn.commit()
        return cur
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def save_signal_batch(batch_data):
    """
    Insert a signal batch into PostgreSQL.
    Creates the session row on first sight (ON CONFLICT DO NOTHING).
    """
    session_id = batch_data.get("sessionID")
    meta = batch_data.get("metadata") or {}
    raw_signals = batch_data.get("signals") or {}
    batch_timestamp = batch_data.get("timestamp")

    source = batch_data.get("source")
    ground_truth_label = batch_data.get("label")

    conn = get_connection()
    try:
        cur = conn.cursor()

        cur.execute(
            """INSERT INTO sessions (session_id, user_agent, viewport_width, viewport_height,
                                     source, label)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (session_id) DO UPDATE SET
                   source = COALESCE(EXCLUDED.source, sessions.source),
                   label  = COALESCE(EXCLUDED.label,  sessions.label)""",
            (
                session_id,
                meta.get("userAgent"),
                meta.get("viewportWidth"),
                meta.get("viewportHeight"),
                source,
                ground_truth_label,
            ),
        )

        cur.execute(
            """INSERT INTO signal_batches (session_id, raw_signals, batch_timestamp)
               VALUES (%s, %s, %s)""",
            (session_id, json.dumps(raw_signals), batch_timestamp),
        )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def save_prediction(session_id, prob_bot, label, threshold, scoring_type="batch", source=None, api_key=None):
    """Insert a prediction row, ensuring the session row exists first."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        # Ensure session exists (predictions has a FK to sessions)
        cur.execute(
            "INSERT INTO sessions (session_id) VALUES (%s) ON CONFLICT (session_id) DO NOTHING",
            (session_id,),
        )
        cur.execute(
            """INSERT INTO predictions (session_id, prob_bot, label, threshold, scoring_type,
                                        source, api_key)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (session_id, prob_bot, label, threshold, scoring_type, source, api_key),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def reset():
    """Reset cached state (for testing)."""
    global _pool, _available, _db_kwargs
    _pool = None
    _available = None
    _db_kwargs = None
