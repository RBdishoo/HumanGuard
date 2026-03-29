"""
PostgreSQL client for HumanGuard.

Reads DATABASE_URL from env. When the var is missing or the connection
fails, every public function degrades gracefully so the JSONL path
keeps working.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

_pool = None
_available = None  # tri-state: None = not checked, True/False = result


def _get_pool():
    """Lazily create the connection pool on first use."""
    global _pool
    if _pool is not None:
        return _pool

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    import psycopg2
    from psycopg2.pool import SimpleConnectionPool

    _pool = SimpleConnectionPool(minconn=1, maxconn=5, dsn=database_url)
    return _pool


def is_available():
    """Return True if DATABASE_URL is set and a connection succeeds."""
    global _available
    if _available is not None:
        return _available

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
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

    conn = get_connection()
    try:
        cur = conn.cursor()

        cur.execute(
            """INSERT INTO sessions (session_id, user_agent, viewport_width, viewport_height)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (session_id) DO NOTHING""",
            (
                session_id,
                meta.get("userAgent"),
                meta.get("viewportWidth"),
                meta.get("viewportHeight"),
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


def save_prediction(session_id, prob_bot, label, threshold, scoring_type="batch"):
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
            """INSERT INTO predictions (session_id, prob_bot, label, threshold, scoring_type)
               VALUES (%s, %s, %s, %s, %s)""",
            (session_id, prob_bot, label, threshold, scoring_type),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def reset():
    """Reset cached state (for testing)."""
    global _pool, _available
    _pool = None
    _available = None
