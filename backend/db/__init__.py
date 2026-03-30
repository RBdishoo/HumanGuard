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

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT UNIQUE NOT NULL,
    user_agent      TEXT,
    viewport_width  INTEGER,
    viewport_height INTEGER,
    source          TEXT,
    label           TEXT,
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
            "ALTER TABLE predictions ADD COLUMN source TEXT",
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
                        scoring_type: str = "batch", source: str = None):
        """Persist a model prediction."""
        label = "bot" if is_bot else "human"

        if self._use_postgres:
            try:
                self._pg.save_prediction(session_id, score, label, threshold, scoring_type,
                                         source=source)
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
                    "(session_id, prob_bot, label, threshold, scoring_type, source) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, score, label, threshold, scoring_type, source),
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


# Module-level singleton — import this everywhere
db = DatabaseManager()
