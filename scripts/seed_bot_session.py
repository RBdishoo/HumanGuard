import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random


SIGNALS_PATH = Path("backend/data/raw/signals.jsonl")
LABELS_PATH = Path("backend/data/raw/labels.csv")


def _iso(ts: datetime) -> str:
    return ts.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _make_bot_batch(session_id: str, batch_index: int, base_time: datetime) -> dict:
    """
    Create a deterministic, bot-like batch:
      - mostly linear mouse movement at near-constant timing
      - very regular key delays
      - clustered clicks
    """
    batch_start_ms = 1000 + (batch_index * 3000)

    # Linear mouse path with small jitter and fixed-ish cadence (100 ms)
    mouse_moves = []
    x0 = 100 + (batch_index % 5) * 10
    y0 = 200 + (batch_index % 3) * 8
    for i in range(18):
        mouse_moves.append(
            {
                "x": x0 + (i * 22) + (i % 2),
                "y": y0 + (i * 9),
                "ts": batch_start_ms + (i * 100),
            }
        )

    # Click bursts every ~250 ms
    clicks = [
        {"x": x0 + 360, "y": y0 + 160, "button": 0, "ts": batch_start_ms + 1700},
        {"x": x0 + 362, "y": y0 + 162, "button": 0, "ts": batch_start_ms + 1950},
        {"x": x0 + 365, "y": y0 + 165, "button": 0, "ts": batch_start_ms + 2200},
    ]

    # Uniform typing cadence (80 ms)
    key_codes = ["KeyL", "KeyO", "KeyG", "KeyI", "KeyN", "Enter"]
    keys = []
    key_start = batch_start_ms + 900
    for i, code in enumerate(key_codes):
        keys.append({"key": code.replace("Key", ""), "code": code, "ts": key_start + (i * 80)})

    return {
        "sessionID": session_id,
        "timestamp": _iso(base_time + timedelta(seconds=batch_index * 3)),
        "signals": {
            "mouseMoves": mouse_moves,
            "clicks": clicks,
            "keys": keys,
        },
        "metadata": {
            "userAgent": "BotSimulator/1.0",
            "viewportWidth": 1366,
            "viewportHeight": 768,
        },
    }


def _upsert_label(session_id: str, label: str) -> None:
    rows = []
    if LABELS_PATH.exists():
        with LABELS_PATH.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("sessionID"):
                    rows.append({"sessionID": row["sessionID"], "label": row.get("label", "")})

    updated = False
    for row in rows:
        if row["sessionID"] == session_id:
            row["label"] = label
            updated = True
            break

    if not updated:
        rows.append({"sessionID": session_id, "label": label})

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sessionID", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(42)
    session_id = f"bot_sim_{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}"
    base_time = datetime.now(tz=timezone.utc) - timedelta(minutes=3)
    num_batches = 15

    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SIGNALS_PATH.open("a") as f:
        for i in range(num_batches):
            rec = _make_bot_batch(session_id, i, base_time)
            f.write(json.dumps(rec) + "\n")

    _upsert_label(session_id, "bot")

    print(f"Appended {num_batches} bot batches to {SIGNALS_PATH}")
    print(f"Upserted label row in {LABELS_PATH}: {session_id},bot")


if __name__ == "__main__":
    main()
