import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random


SIGNALS_PATH = Path("backend/data/raw/signals.jsonl")
LABELS_PATH = Path("backend/data/raw/labels.csv")


def _iso(ts: datetime) -> str:
    return ts.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _make_bot_batch(session_id: str, batch_index: int, base_time: datetime,
                    stealthy: bool = False) -> dict:
    """
    Create a bot-like batch.

    stealthy=False: classic bot — perfectly linear, constant timing.
    stealthy=True:  evasive bot — adds slight randomness to timing and paths
                    to try to pass as human, but still fundamentally automated.
    """
    batch_start_ms = 1000 + (batch_index * 3000)

    mouse_moves = []
    x0 = 100 + (batch_index % 5) * 10
    y0 = 200 + (batch_index % 3) * 8
    num_moves = random.randint(14, 22) if stealthy else 18
    for i in range(num_moves):
        if stealthy:
            # Add some jitter and slight curve to look more human
            jx = random.randint(-3, 3)
            jy = random.randint(-2, 2)
            dt = 100 + random.randint(-25, 25)
        else:
            jx = i % 2
            jy = 0
            dt = 100
        mouse_moves.append(
            {
                "x": x0 + (i * 22) + jx,
                "y": y0 + (i * 9) + jy,
                "ts": batch_start_ms + (i * dt),
            }
        )

    # Clicks
    if stealthy:
        click_base = batch_start_ms + 1700
        clicks = [
            {"x": x0 + 360 + random.randint(-10, 10), "y": y0 + 160 + random.randint(-5, 5),
             "button": 0, "ts": click_base + random.randint(0, 50)},
            {"x": x0 + 362 + random.randint(-10, 10), "y": y0 + 162 + random.randint(-5, 5),
             "button": 0, "ts": click_base + 250 + random.randint(-40, 80)},
            {"x": x0 + 365 + random.randint(-10, 10), "y": y0 + 165 + random.randint(-5, 5),
             "button": 0, "ts": click_base + 500 + random.randint(-40, 120)},
        ]
    else:
        clicks = [
            {"x": x0 + 360, "y": y0 + 160, "button": 0, "ts": batch_start_ms + 1700},
            {"x": x0 + 362, "y": y0 + 162, "button": 0, "ts": batch_start_ms + 1950},
            {"x": x0 + 365, "y": y0 + 165, "button": 0, "ts": batch_start_ms + 2200},
        ]

    # Keys — stealthy bots vary their input strings
    if stealthy:
        phrases = [
            ["KeyS", "KeyE", "KeyA", "KeyR", "KeyC", "KeyH"],
            ["KeyA", "KeyD", "KeyM", "KeyI", "KeyN"],
            ["KeyH", "KeyE", "KeyL", "KeyL", "KeyO", "Space", "KeyW", "KeyO", "KeyR", "KeyL", "KeyD"],
            ["KeyU", "KeyS", "KeyE", "KeyR", "KeyN", "KeyA", "KeyM", "KeyE"],
            ["KeyP", "KeyA", "KeyS", "KeyS", "KeyW", "KeyO", "KeyR", "KeyD", "Enter"],
            ["KeyT", "KeyE", "KeyS", "KeyT", "KeyI", "KeyN", "KeyG"],
        ]
        key_codes = random.choice(phrases)
    else:
        key_codes = ["KeyL", "KeyO", "KeyG", "KeyI", "KeyN", "Enter"]
    keys = []
    key_start = batch_start_ms + 900
    for i, code in enumerate(key_codes):
        if stealthy:
            delay = 80 + random.randint(-15, 25)
        else:
            delay = 80
        keys.append({"key": code.replace("Key", "").lower(), "code": code, "ts": key_start + (i * delay)})

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
    # Use --stealthy flag to generate evasive bots
    import sys
    stealthy = "--stealthy" in sys.argv

    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SIGNALS_PATH.open("a") as f:
        for i in range(num_batches):
            rec = _make_bot_batch(session_id, i, base_time, stealthy=stealthy)
            f.write(json.dumps(rec) + "\n")

    _upsert_label(session_id, "bot")

    kind = "stealthy" if stealthy else "classic"
    print(f"Appended {num_batches} {kind} bot batches to {SIGNALS_PATH}")
    print(f"Upserted label row in {LABELS_PATH}: {session_id},bot")


if __name__ == "__main__":
    main()
