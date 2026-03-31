"""
DatabaseManager for HumanGuard.

Provides a unified interface over SQLite (dev/test) and PostgreSQL (production).
Backend is chosen at instantiation based on DATABASE_URL:

  - Starts with "postgres" → PostgreSQL via db_client connection pool
  - Starts with "sqlite:///" → SQLite at the specified path (":memory:" supported)
  - Empty / anything else  → SQLite at backend/data/humanguard.db

Gracefully degrades: every public method silently returns empty/zero values on
failure so the JSONL path continues working uninterrupted.
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

RETRAIN_THRESHOLD = 50


# ── Key format helpers ────────────────────────────────────────────────────────
# New format: hg_live_<8-char-id>.<32-char-secret>
# Old format: hg_live_<24-char-hex>  (no dot — legacy, kept for backward compat)

def _hash_secret(secret: str) -> str:
    """Return SHA-256 hex digest of the raw secret."""
    return hashlib.sha256(secret.encode()).hexdigest()


def _split_key(key: str):
    """Split 'hg_live_<id>.<secret>' → ('hg_live_<id>', '<secret>').

    Returns (None, None) for old-format keys that contain no dot.
    """
    if not key or "." not in key:
        return None, None
    idx = key.index(".")
    return key[:idx], key[idx + 1:]


def _key_id_from_full(key: str) -> str:
    """Return the non-secret identifier portion of a key.

    New format → 'hg_live_<id>'   (strips the secret after the dot)
    Old format → key unchanged     (used as-is for backward compat)
    """
    key_id, _ = _split_key(key)
    return key_id if key_id else key

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT UNIQUE NOT NULL,
    user_agent      TEXT,
    viewport_width  INTEGER,
    viewport_height INTEGER,
    source          TEXT,
    label           TEXT,
    trained_at      TIMESTAMP DEFAULT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS signal_batches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    raw_signals     TEXT NOT NULL,
    batch_timestamp TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    prob_bot        REAL NOT NULL,
    label           TEXT NOT NULL,
    threshold       REAL NOT NULL DEFAULT 0.5,
    scoring_type    TEXT NOT NULL DEFAULT 'batch',
    source          TEXT,
    api_key         TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS leaderboard (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    nickname    TEXT NOT NULL,
    prob_bot    REAL NOT NULL,
    verdict     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS api_keys (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    key                 TEXT UNIQUE NOT NULL,
    key_id              TEXT UNIQUE,
    key_hash            TEXT,
    owner_email         TEXT NOT NULL,
    plan                TEXT NOT NULL DEFAULT 'free',
    monthly_limit       INTEGER NOT NULL DEFAULT 1000,
    current_month_count INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    active              INTEGER NOT NULL DEFAULT 1
);
"""


class DatabaseManager:
    """Unified DB manager with automatic SQLite/PostgreSQL selection."""

    def __init__(self):
        self._db_url = os.environ.get("DATABASE_URL", "")
        self._use_postgres = self._db_url.startswith("postgres")
        self._sqlite_path = None
        self._sqlite_mem_conn = None  # persistent connection for ":memory:"
        self._pg = None

        if self._use_postgres:
            try:
                from . import db_client
                self._pg = db_client
                logger.info("DatabaseManager: PostgreSQL mode")
            except Exception as exc:
                logger.warning("DatabaseManager: PostgreSQL init failed, falling back to SQLite: %s", exc)
                self._use_postgres = False

        if not self._use_postgres:
            if self._db_url.startswith("sqlite:///"):
                raw_path = self._db_url[len("sqlite:///"):]
                self._sqlite_path = ":memory:" if raw_path == ":memory:" else raw_path
            else:
                data_dir = Path(__file__).resolve().parent.parent / "data"
                data_dir.mkdir(parents=True, exist_ok=True)
                self._sqlite_path = str(data_dir / "humanguard.db")

            self._init_sqlite()
            logger.info("DatabaseManager: SQLite mode (path=%s)", self._sqlite_path)

    # ── SQLite internals ──────────────────────────────────────────────────────

    def _get_sqlite_conn(self):
        if self._sqlite_path == ":memory:":
            if self._sqlite_mem_conn is None:
                self._sqlite_mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._sqlite_mem_conn.row_factory = sqlite3.Row
            return self._sqlite_mem_conn
        conn = sqlite3.connect(self._sqlite_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self):
        conn = self._get_sqlite_conn()
        conn.executescript(_SQLITE_DDL)
        conn.commit()
        # Migration: add new columns to existing databases
        for migration_sql in [
            "ALTER TABLE sessions ADD COLUMN source TEXT",
            "ALTER TABLE sessions ADD COLUMN label TEXT",
            "ALTER TABLE sessions ADD COLUMN trained_at TIMESTAMP DEFAULT NULL",
            "ALTER TABLE predictions ADD COLUMN source TEXT",
            "ALTER TABLE predictions ADD COLUMN api_key TEXT",
            (
                "CREATE TABLE IF NOT EXISTS api_keys ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "key TEXT UNIQUE NOT NULL, "
                "key_id TEXT UNIQUE, "
                "key_hash TEXT, "
                "owner_email TEXT NOT NULL, "
                "plan TEXT NOT NULL DEFAULT 'free', "
                "monthly_limit INTEGER NOT NULL DEFAULT 1000, "
                "current_month_count INTEGER NOT NULL DEFAULT 0, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "active INTEGER NOT NULL DEFAULT 1)"
            ),
            "ALTER TABLE api_keys ADD COLUMN key_id TEXT UNIQUE",
            "ALTER TABLE api_keys ADD COLUMN key_hash TEXT",
        ]:
            try:
                conn.execute(migration_sql)
                conn.commit()
            except Exception:
                pass  # Column already exists — safe to ignore
        if self._sqlite_path != ":memory:":
            conn.close()

    @contextmanager
    def _sqlite_cursor(self):
        conn = self._get_sqlite_conn()
        try:
            cur = conn.cursor()
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if self._sqlite_path != ":memory:":
                conn.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def save_session(self, session_data: dict):
        """Upsert session and append a signal batch."""
        if not isinstance(session_data, dict):
            return
        session_id = session_data.get("sessionID")
        if not session_id:
            return

        if self._use_postgres:
            try:
                self._pg.save_signal_batch(session_data)
            except Exception as exc:
                logger.warning("DB save_session failed: %s", exc)
            return

        meta = session_data.get("metadata") or {}
        raw_signals = session_data.get("signals") or {}
        batch_ts = session_data.get("timestamp")
        source = session_data.get("source")
        ground_truth_label = session_data.get("label")
        try:
            with self._sqlite_cursor() as cur:
                cur.execute(
                    "INSERT OR IGNORE INTO sessions "
                    "(session_id, user_agent, viewport_width, viewport_height, source, label) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, meta.get("userAgent"),
                     meta.get("viewportWidth"), meta.get("viewportHeight"),
                     source, ground_truth_label),
                )
                cur.execute(
                    "INSERT INTO signal_batches (session_id, raw_signals, batch_timestamp) "
                    "VALUES (?, ?, ?)",
                    (session_id, json.dumps(raw_signals), batch_ts),
                )
        except Exception as exc:
            logger.warning("SQLite save_session failed: %s", exc)

    def save_prediction(self, session_id: str, score: float, is_bot: bool,
                        features=None, threshold: float = 0.5,
                        scoring_type: str = "batch", source: str = None,
                        api_key: str = None):
        """Persist a model prediction. Stores key_id (not the full secret key)."""
        label = "bot" if is_bot else "human"
        # Store only the non-secret identifier so the predictions table never holds raw secrets
        stored_key_id = _key_id_from_full(api_key) if api_key else None

        if self._use_postgres:
            try:
                self._pg.save_prediction(session_id, score, label, threshold, scoring_type,
                                         source=source, api_key=stored_key_id)
            except Exception as exc:
                logger.warning("DB save_prediction failed: %s", exc)
            return

        try:
            with self._sqlite_cursor() as cur:
                cur.execute(
                    "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                    (session_id,),
                )
                cur.execute(
                    "INSERT INTO predictions "
                    "(session_id, prob_bot, label, threshold, scoring_type, source, api_key) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, score, label, threshold, scoring_type, source, stored_key_id),
                )
        except Exception as exc:
            logger.warning("SQLite save_prediction failed: %s", exc)

    def get_recent_predictions(self, limit: int = 100) -> list:
        """Return the *limit* most recent predictions, newest first."""
        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT session_id, prob_bot, label, threshold, scoring_type, created_at "
                        "FROM predictions ORDER BY id DESC LIMIT %s",
                        (limit,),
                    )
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, row)) for row in cur.fetchall()]
                finally:
                    self._pg.release_connection(conn)
            except Exception as exc:
                logger.warning("DB get_recent_predictions failed: %s", exc)
                return []

        try:
            with self._sqlite_cursor() as cur:
                cur.execute(
                    "SELECT session_id, prob_bot, label, threshold, scoring_type, created_at "
                    "FROM predictions ORDER BY id DESC LIMIT ?",
                    (limit,),
                )
                return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.warning("SQLite get_recent_predictions failed: %s", exc)
            return []

    def get_stats(self) -> dict:
        """Return aggregated prediction counts and bot rate."""
        empty = {"total_predictions": 0, "bot_count": 0, "human_count": 0, "bot_rate": 0.0}

        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT COUNT(*), "
                        "SUM(CASE WHEN label='bot' THEN 1 ELSE 0 END), "
                        "SUM(CASE WHEN label='human' THEN 1 ELSE 0 END) "
                        "FROM predictions"
                    )
                    row = cur.fetchone()
                finally:
                    self._pg.release_connection(conn)
                total = row[0] or 0
                bots = int(row[1] or 0)
                humans = int(row[2] or 0)
            except Exception as exc:
                logger.warning("DB get_stats failed: %s", exc)
                return empty
            return {
                "total_predictions": total,
                "bot_count": bots,
                "human_count": humans,
                "bot_rate": round(bots / total, 4) if total > 0 else 0.0,
            }

        try:
            with self._sqlite_cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*), "
                    "SUM(CASE WHEN label='bot' THEN 1 ELSE 0 END), "
                    "SUM(CASE WHEN label='human' THEN 1 ELSE 0 END) "
                    "FROM predictions"
                )
                row = cur.fetchone()
            total = row[0] or 0
            bots = int(row[1] or 0)
            humans = int(row[2] or 0)
        except Exception as exc:
            logger.warning("SQLite get_stats failed: %s", exc)
            return empty
        return {
            "total_predictions": total,
            "bot_count": bots,
            "human_count": humans,
            "bot_rate": round(bots / total, 4) if total > 0 else 0.0,
        }


    def save_leaderboard_entry(self, nickname: str, prob_bot: float,
                               verdict: str, session_id: str) -> int:
        """Insert a leaderboard entry and return the new row id."""
        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "INSERT INTO leaderboard (nickname, prob_bot, verdict, session_id) "
                        "VALUES (%s, %s, %s, %s) RETURNING id",
                        (nickname, prob_bot, verdict, session_id),
                    )
                    row_id = cur.fetchone()[0]
                    conn.commit()
                    return row_id
                finally:
                    self._pg.release_connection(conn)
            except Exception as exc:
                logger.warning("DB save_leaderboard_entry failed: %s", exc)
                return -1

        try:
            with self._sqlite_cursor() as cur:
                cur.execute(
                    "INSERT INTO leaderboard (nickname, prob_bot, verdict, session_id) "
                    "VALUES (?, ?, ?, ?)",
                    (nickname, prob_bot, verdict, session_id),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.warning("SQLite save_leaderboard_entry failed: %s", exc)
            return -1

    def get_leaderboard(self, limit: int = 20) -> list:
        """Return top *limit* entries ordered by prob_bot ASC (most-human first)."""
        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT nickname, prob_bot, verdict, session_id, created_at "
                        "FROM leaderboard ORDER BY prob_bot ASC LIMIT %s",
                        (limit,),
                    )
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, row)) for row in cur.fetchall()]
                finally:
                    self._pg.release_connection(conn)
            except Exception as exc:
                logger.warning("DB get_leaderboard failed: %s", exc)
                return []

        try:
            with self._sqlite_cursor() as cur:
                cur.execute(
                    "SELECT nickname, prob_bot, verdict, session_id, created_at "
                    "FROM leaderboard ORDER BY prob_bot ASC LIMIT ?",
                    (limit,),
                )
                return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.warning("SQLite get_leaderboard failed: %s", exc)
            return []

    def get_leaderboard_stats(self) -> dict:
        """Return total participants, avg prob_bot, and % identified as human."""
        empty = {"total": 0, "avg_prob_bot": 0.0, "pct_human": 0.0}
        _sql = (
            "SELECT COUNT(*), AVG(prob_bot), "
            "SUM(CASE WHEN verdict='human' THEN 1 ELSE 0 END) "
            "FROM leaderboard"
        )

        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(_sql)
                    row = cur.fetchone()
                finally:
                    self._pg.release_connection(conn)
            except Exception as exc:
                logger.warning("DB get_leaderboard_stats failed: %s", exc)
                return empty
        else:
            try:
                with self._sqlite_cursor() as cur:
                    cur.execute(_sql)
                    row = cur.fetchone()
            except Exception as exc:
                logger.warning("SQLite get_leaderboard_stats failed: %s", exc)
                return empty

        total = int(row[0] or 0)
        avg = float(row[1] or 0.0)
        human_count = int(row[2] or 0)
        return {
            "total": total,
            "avg_prob_bot": round(avg, 4),
            "pct_human": round(human_count / total * 100, 1) if total else 0.0,
        }


    # ── API Key Management ────────────────────────────────────────────────────

    def generate_api_key(self, email: str, plan: str = "free", monthly_limit: int = 1000) -> str:
        """Create a new API key, store only the id + hash, return the full key once.

        Key format: hg_live_<8-char-id>.<32-char-secret>
        - id part  stored in key_id and key columns (non-secret, used for lookup)
        - secret part hashed with SHA-256, stored in key_hash (raw secret never stored)
        The full key is returned to the caller exactly once and is not retrievable later.
        """
        key_id_part = secrets.token_hex(4)   # 8 hex chars
        secret_part = secrets.token_hex(16)  # 32 hex chars
        full_key = f"hg_live_{key_id_part}.{secret_part}"
        key_id = f"hg_live_{key_id_part}"
        key_hash = _hash_secret(secret_part)

        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "INSERT INTO api_keys (key, key_id, key_hash, owner_email, plan, monthly_limit) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (key_id, key_id, key_hash, email, plan, monthly_limit),
                    )
                    conn.commit()
                finally:
                    self._pg.release_connection(conn)
            except Exception as exc:
                logger.warning("DB generate_api_key failed: %s", exc)
            return full_key

        try:
            with self._sqlite_cursor() as cur:
                cur.execute(
                    "INSERT INTO api_keys (key, key_id, key_hash, owner_email, plan, monthly_limit) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (key_id, key_id, key_hash, email, plan, monthly_limit),
                )
        except Exception as exc:
            logger.warning("SQLite generate_api_key failed: %s", exc)
        return full_key

    def validate_api_key(self, key: str) -> dict | None:
        """Return the key record if valid and active, else None.

        New format (hg_live_<id>.<secret>):
          - look up by key_id column, compare sha256(secret) with stored key_hash
          - uses hmac.compare_digest() for constant-time comparison
        Old format (no dot, legacy):
          - look up by key column directly (backward compat)
          - only succeeds for rows that have no key_hash (genuine old keys)
        """
        if not key:
            return None

        key_id, secret = _split_key(key)

        if key_id and secret:
            # ── New format ──────────────────────────────────────────────────
            submitted_hash = _hash_secret(secret)

            if self._use_postgres:
                try:
                    conn = self._pg.get_connection()
                    try:
                        cur = conn.cursor()
                        cur.execute(
                            "SELECT key_id, key_hash, owner_email, plan, monthly_limit, "
                            "current_month_count, active "
                            "FROM api_keys WHERE key_id = %s",
                            (key_id,),
                        )
                        row = cur.fetchone()
                    finally:
                        self._pg.release_connection(conn)
                    if row is None or not row[6]:
                        return None
                    if not hmac.compare_digest(row[1] or "", submitted_hash):
                        return None
                    return {
                        "key": key_id, "owner_email": row[2], "plan": row[3],
                        "monthly_limit": row[4], "current_month_count": row[5],
                        "active": bool(row[6]),
                    }
                except Exception as exc:
                    logger.warning("DB validate_api_key (new format) failed: %s", exc)
                    return None

            try:
                with self._sqlite_cursor() as cur:
                    cur.execute(
                        "SELECT key_id, key_hash, owner_email, plan, monthly_limit, "
                        "current_month_count, active "
                        "FROM api_keys WHERE key_id = ?",
                        (key_id,),
                    )
                    row = cur.fetchone()
                if row is None or not row["active"]:
                    return None
                if not hmac.compare_digest(row["key_hash"] or "", submitted_hash):
                    return None
                return {
                    "key": row["key_id"], "owner_email": row["owner_email"], "plan": row["plan"],
                    "monthly_limit": row["monthly_limit"], "current_month_count": row["current_month_count"],
                    "active": bool(row["active"]),
                }
            except Exception as exc:
                logger.warning("SQLite validate_api_key (new format) failed: %s", exc)
                return None

        # ── Old format (no dot) — backward compat only ──────────────────────
        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT key, owner_email, plan, monthly_limit, current_month_count, "
                        "active, key_hash "
                        "FROM api_keys WHERE key = %s",
                        (key,),
                    )
                    row = cur.fetchone()
                finally:
                    self._pg.release_connection(conn)
                if row is None or not row[5]:
                    return None
                # Block old-style auth for new-format rows (key_hash present)
                if row[6]:
                    return None
                return {
                    "key": row[0], "owner_email": row[1], "plan": row[2],
                    "monthly_limit": row[3], "current_month_count": row[4], "active": bool(row[5]),
                }
            except Exception as exc:
                logger.warning("DB validate_api_key (old format) failed: %s", exc)
                return None

        try:
            with self._sqlite_cursor() as cur:
                cur.execute(
                    "SELECT key, owner_email, plan, monthly_limit, current_month_count, "
                    "active, key_hash "
                    "FROM api_keys WHERE key = ?",
                    (key,),
                )
                row = cur.fetchone()
            if row is None or not row["active"]:
                return None
            # Block old-style auth for new-format rows (key_hash present)
            if row["key_hash"]:
                return None
            return {
                "key": row["key"], "owner_email": row["owner_email"], "plan": row["plan"],
                "monthly_limit": row["monthly_limit"], "current_month_count": row["current_month_count"],
                "active": bool(row["active"]),
            }
        except Exception as exc:
            logger.warning("SQLite validate_api_key (old format) failed: %s", exc)
            return None

    def increment_usage(self, key: str):
        """Bump current_month_count for the given key."""
        key_id, _ = _split_key(key)
        if key_id:
            pg_col, sq_col, lookup = "key_id", "key_id", key_id
        else:
            pg_col, sq_col, lookup = "key", "key", key

        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        f"UPDATE api_keys SET current_month_count = current_month_count + 1 "
                        f"WHERE {pg_col} = %s",
                        (lookup,),
                    )
                    conn.commit()
                finally:
                    self._pg.release_connection(conn)
            except Exception as exc:
                logger.warning("DB increment_usage failed: %s", exc)
            return

        try:
            with self._sqlite_cursor() as cur:
                cur.execute(
                    f"UPDATE api_keys SET current_month_count = current_month_count + 1 "
                    f"WHERE {sq_col} = ?",
                    (lookup,),
                )
        except Exception as exc:
            logger.warning("SQLite increment_usage failed: %s", exc)

    def atomic_increment_usage(self, key: str) -> dict | None:
        """Atomically increment usage only if the key is active and under its monthly limit.

        Returns a dict with current_month_count/monthly_limit/plan on success,
        or None if the quota is exhausted (or the key doesn't exist/is inactive).
        """
        key_id, _ = _split_key(key)
        if key_id:
            pg_col, sq_col, lookup = "key_id", "key_id", key_id
        else:
            pg_col, sq_col, lookup = "key", "key", key

        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        f"UPDATE api_keys SET current_month_count = current_month_count + 1 "
                        f"WHERE {pg_col} = %s AND active = true AND current_month_count < monthly_limit "
                        f"RETURNING current_month_count, monthly_limit, plan",
                        (lookup,),
                    )
                    row = cur.fetchone()
                    conn.commit()
                finally:
                    self._pg.release_connection(conn)
                if row is None:
                    return None
                return {"current_month_count": row[0], "monthly_limit": row[1], "plan": row[2]}
            except Exception as exc:
                logger.warning("DB atomic_increment_usage failed: %s", exc)
                return None

        try:
            with self._sqlite_cursor() as cur:
                cur.execute(
                    f"UPDATE api_keys SET current_month_count = current_month_count + 1 "
                    f"WHERE {sq_col} = ? AND active = 1 AND current_month_count < monthly_limit "
                    f"RETURNING current_month_count, monthly_limit, plan",
                    (lookup,),
                )
                row = cur.fetchone()
            if row is None:
                return None
            return {
                "current_month_count": row["current_month_count"],
                "monthly_limit": row["monthly_limit"],
                "plan": row["plan"],
            }
        except Exception as exc:
            logger.warning("SQLite atomic_increment_usage failed: %s", exc)
            return None

    def get_usage(self, key: str) -> dict:
        """Return usage stats for the given key."""
        empty = {"count": 0, "limit": 1000, "percentage_used": 0.0, "plan": "free"}
        key_id_part, _ = _split_key(key)
        if key_id_part:
            pg_col, sq_col, lookup = "key_id", "key_id", key_id_part
        else:
            pg_col, sq_col, lookup = "key", "key", key

        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        f"SELECT current_month_count, monthly_limit, plan "
                        f"FROM api_keys WHERE {pg_col} = %s",
                        (lookup,),
                    )
                    row = cur.fetchone()
                finally:
                    self._pg.release_connection(conn)
                if row is None:
                    return empty
                count, limit, plan = int(row[0]), int(row[1]), row[2]
            except Exception as exc:
                logger.warning("DB get_usage failed: %s", exc)
                return empty
        else:
            try:
                with self._sqlite_cursor() as cur:
                    cur.execute(
                        f"SELECT current_month_count, monthly_limit, plan "
                        f"FROM api_keys WHERE {sq_col} = ?",
                        (lookup,),
                    )
                    row = cur.fetchone()
                if row is None:
                    return empty
                count, limit, plan = int(row["current_month_count"]), int(row["monthly_limit"]), row["plan"]
            except Exception as exc:
                logger.warning("SQLite get_usage failed: %s", exc)
                return empty

        pct = round(count / limit * 100, 1) if limit > 0 else 0.0
        return {"count": count, "limit": limit, "percentage_used": pct, "plan": plan}

    def get_client_stats(self, api_key: str) -> dict:
        """Return prediction stats scoped to the given API key."""
        empty = {
            "total_sessions": 0, "bot_count": 0, "human_count": 0,
            "bot_rate": 0.0, "sessions_today": 0, "sessions_this_month": 0,
            "usage_count": 0, "usage_limit": 1000, "daily_stats": [],
        }
        if not api_key:
            return empty

        # Predictions are stored with key_id, not the full secret key
        api_key = _key_id_from_full(api_key)
        usage = self.get_usage(api_key)

        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT COUNT(DISTINCT session_id), "
                        "SUM(CASE WHEN label='bot' THEN 1 ELSE 0 END), "
                        "SUM(CASE WHEN label='human' THEN 1 ELSE 0 END), "
                        "COUNT(DISTINCT CASE WHEN DATE(created_at)=CURRENT_DATE THEN session_id END), "
                        "COUNT(DISTINCT CASE WHEN DATE_TRUNC('month',created_at)=DATE_TRUNC('month',NOW()) THEN session_id END) "
                        "FROM predictions WHERE api_key=%s",
                        (api_key,),
                    )
                    row = cur.fetchone()
                    cur.execute(
                        "SELECT DATE(created_at) as day, COUNT(*) as total, "
                        "SUM(CASE WHEN label='bot' THEN 1 ELSE 0 END) as bots "
                        "FROM predictions WHERE api_key=%s AND created_at >= CURRENT_DATE - INTERVAL '6 days' "
                        "GROUP BY DATE(created_at) ORDER BY day ASC",
                        (api_key,),
                    )
                    daily_rows = cur.fetchall()
                finally:
                    self._pg.release_connection(conn)
                total = int(row[0] or 0)
                bots = int(row[1] or 0)
                humans = int(row[2] or 0)
                today = int(row[3] or 0)
                month = int(row[4] or 0)
                daily = [{"date": str(r[0]), "total": int(r[1] or 0),
                           "bot_rate": round(int(r[2] or 0) / int(r[1]) if int(r[1]) > 0 else 0, 4)}
                          for r in daily_rows]
            except Exception as exc:
                logger.warning("DB get_client_stats failed: %s", exc)
                return empty
        else:
            try:
                with self._sqlite_cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(DISTINCT session_id), "
                        "SUM(CASE WHEN label='bot' THEN 1 ELSE 0 END), "
                        "SUM(CASE WHEN label='human' THEN 1 ELSE 0 END), "
                        "COUNT(DISTINCT CASE WHEN DATE(created_at)=DATE('now') THEN session_id END), "
                        "COUNT(DISTINCT CASE WHEN strftime('%Y-%m',created_at)=strftime('%Y-%m','now') THEN session_id END) "
                        "FROM predictions WHERE api_key=?",
                        (api_key,),
                    )
                    row = cur.fetchone()
                    cur.execute(
                        "SELECT DATE(created_at) as day, COUNT(*) as total, "
                        "SUM(CASE WHEN label='bot' THEN 1 ELSE 0 END) as bots "
                        "FROM predictions WHERE api_key=? AND created_at >= DATE('now','-6 days') "
                        "GROUP BY DATE(created_at) ORDER BY day ASC",
                        (api_key,),
                    )
                    daily_rows = cur.fetchall()
                total = int(row[0] or 0)
                bots = int(row[1] or 0)
                humans = int(row[2] or 0)
                today = int(row[3] or 0)
                month = int(row[4] or 0)
                daily = [{"date": str(r["day"]), "total": int(r["total"] or 0),
                           "bot_rate": round(int(r["bots"] or 0) / int(r["total"]) if int(r["total"] or 0) > 0 else 0, 4)}
                          for r in daily_rows]
            except Exception as exc:
                logger.warning("SQLite get_client_stats failed: %s", exc)
                return empty

        return {
            "total_sessions": total,
            "bot_count": bots,
            "human_count": humans,
            "bot_rate": round(bots / total, 4) if total > 0 else 0.0,
            "sessions_today": today,
            "sessions_this_month": month,
            "usage_count": usage.get("count", 0),
            "usage_limit": usage.get("limit", 1000),
            "daily_stats": daily,
        }

    def get_client_predictions(self, api_key: str, limit: int = 50) -> list:
        """Return recent predictions scoped to the given API key, newest first."""
        if not api_key:
            return []
        # Predictions are stored with key_id, not the full secret key
        api_key = _key_id_from_full(api_key)

        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "SELECT session_id, prob_bot, label, scoring_type, source, created_at "
                        "FROM predictions WHERE api_key=%s ORDER BY id DESC LIMIT %s",
                        (api_key, limit),
                    )
                    cols = [d[0] for d in cur.description]
                    return [dict(zip(cols, row)) for row in cur.fetchall()]
                finally:
                    self._pg.release_connection(conn)
            except Exception as exc:
                logger.warning("DB get_client_predictions failed: %s", exc)
                return []

        try:
            with self._sqlite_cursor() as cur:
                cur.execute(
                    "SELECT session_id, prob_bot, label, scoring_type, source, created_at "
                    "FROM predictions WHERE api_key=? ORDER BY id DESC LIMIT ?",
                    (api_key, limit),
                )
                return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.warning("SQLite get_client_predictions failed: %s", exc)
            return []

    def reset_monthly_counts(self):
        """Reset current_month_count to 0 for all keys (call on 1st of month)."""
        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute("UPDATE api_keys SET current_month_count = 0")
                    conn.commit()
                finally:
                    self._pg.release_connection(conn)
            except Exception as exc:
                logger.warning("DB reset_monthly_counts failed: %s", exc)
            return

        try:
            with self._sqlite_cursor() as cur:
                cur.execute("UPDATE api_keys SET current_month_count = 0")
        except Exception as exc:
            logger.warning("SQLite reset_monthly_counts failed: %s", exc)

    # ── Retraining Support ────────────────────────────────────────────────────

    def get_unlabeled_session_count(self) -> int:
        """Count sessions from demo/simulator sources that have not been used in training."""
        _sql_pg = (
            "SELECT COUNT(*) FROM sessions "
            "WHERE source IN ('demo', 'simulator') AND trained_at IS NULL"
        )
        _sql_sq = _sql_pg  # same syntax

        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(_sql_pg)
                    row = cur.fetchone()
                    return int(row[0] or 0)
                finally:
                    self._pg.release_connection(conn)
            except Exception as exc:
                logger.warning("DB get_unlabeled_session_count failed: %s", exc)
                return 0

        try:
            with self._sqlite_cursor() as cur:
                cur.execute(_sql_sq)
                row = cur.fetchone()
            return int(row[0] or 0)
        except Exception as exc:
            logger.warning("SQLite get_unlabeled_session_count failed: %s", exc)
            return 0

    def mark_sessions_as_trained(self, session_ids: list):
        """Set trained_at = now() for the given session IDs so they won't retrigger training."""
        if not session_ids:
            return

        if self._use_postgres:
            try:
                conn = self._pg.get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        "UPDATE sessions SET trained_at = NOW() "
                        "WHERE session_id = ANY(%s)",
                        (list(session_ids),),
                    )
                    conn.commit()
                finally:
                    self._pg.release_connection(conn)
            except Exception as exc:
                logger.warning("DB mark_sessions_as_trained failed: %s", exc)
            return

        placeholders = ",".join("?" * len(session_ids))
        try:
            with self._sqlite_cursor() as cur:
                cur.execute(
                    f"UPDATE sessions SET trained_at = CURRENT_TIMESTAMP "
                    f"WHERE session_id IN ({placeholders})",
                    list(session_ids),
                )
        except Exception as exc:
            logger.warning("SQLite mark_sessions_as_trained failed: %s", exc)


# Module-level singleton — import this everywhere
db = DatabaseManager()
