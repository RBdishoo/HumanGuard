"""
One-time migration: JSONL + labels.csv → PostgreSQL.

Usage (local dev — pass URL directly):
    DATABASE_URL=postgres://user:pass@host/db python -m backend.db.migrate

Usage (resolve from Secrets Manager — same as production):
    RDS_SECRET_NAME=humanGuard/rds python -m backend.db.migrate

Safe to re-run — uses ON CONFLICT DO NOTHING for all inserts.
"""

import csv
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import db_client

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SIGNALS_FILE = DATA_DIR / "signals.jsonl"
LABELS_FILE = DATA_DIR / "labels.csv"


def migrate():
    if not db_client.is_available():
        print("ERROR: could not connect to PostgreSQL.")
        print("  Set DATABASE_URL=postgres://... or RDS_SECRET_NAME=humanGuard/rds")
        sys.exit(1)

    sessions_migrated = set()
    batches_migrated = 0
    labels_migrated = 0

    # --- Migrate signals.jsonl ---
    if SIGNALS_FILE.exists():
        with open(SIGNALS_FILE, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at line {line_num}")
                    continue

                session_id = record.get("sessionID")
                if not session_id:
                    continue

                meta = record.get("metadata") or {}
                raw_signals = record.get("signals") or {}
                batch_timestamp = record.get("timestamp")

                conn = db_client.get_connection()
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
                    sessions_migrated.add(session_id)
                    batches_migrated += 1
                except Exception as exc:
                    conn.rollback()
                    print(f"Error migrating line {line_num}: {exc}")
                finally:
                    db_client.release_connection(conn)
    else:
        print(f"No signals file at {SIGNALS_FILE}, skipping batches.")

    # --- Migrate labels.csv ---
    if LABELS_FILE.exists():
        with open(LABELS_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                session_id = row.get("sessionID")
                label = row.get("label")
                if not session_id or not label:
                    continue

                conn = db_client.get_connection()
                try:
                    cur = conn.cursor()
                    # Ensure session exists (may not be in signals.jsonl)
                    cur.execute(
                        """INSERT INTO sessions (session_id)
                           VALUES (%s)
                           ON CONFLICT (session_id) DO NOTHING""",
                        (session_id,),
                    )
                    cur.execute(
                        """INSERT INTO labels (session_id, label)
                           VALUES (%s, %s)
                           ON CONFLICT (session_id) DO NOTHING""",
                        (session_id, label),
                    )
                    conn.commit()
                    labels_migrated += 1
                except Exception as exc:
                    conn.rollback()
                    print(f"Error migrating label for {session_id}: {exc}")
                finally:
                    db_client.release_connection(conn)
    else:
        print(f"No labels file at {LABELS_FILE}, skipping labels.")

    print(f"\nMigration complete:")
    print(f"  {len(sessions_migrated)} sessions migrated")
    print(f"  {batches_migrated} batches migrated")
    print(f"  {labels_migrated} labels migrated")


if __name__ == "__main__":
    migrate()
