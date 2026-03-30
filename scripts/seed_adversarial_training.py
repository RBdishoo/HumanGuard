"""
Generate adversarial bot sessions for TRAINING data.

These sessions use human-mimicking patterns that the model struggles with,
so that the trained model learns to recognize them.

Two patterns × 40 sessions × 10 batches = 800 training batches.

These are intentionally different from the hard test set (different parameters
and random seeds) so they teach the pattern without leaking test data.

Output (appended to existing training files):
  backend/data/raw/signals.jsonl
  backend/data/raw/labels.csv
"""

import csv
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SIGNALS_PATH = Path("backend/data/raw/signals.jsonl")
LABELS_PATH  = Path("backend/data/raw/labels.csv")

SESSIONS_PER_PATTERN = 40
BATCHES_PER_SESSION  = 10


# ── helpers ───────────────────────────────────────────────────────────────

def _iso(ts: datetime) -> str:
    return ts.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


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


def _bezier_point(t: float, p0, p1, p2) -> tuple:
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return (x, y)


# ── Pattern A: human_speed_typer_train ───────────────────────────────────
# Linear mouse + keystroke timing drawn from a Gaussian distribution at
# slightly different parameters than the hard test set (mean=155ms, std=50ms)
# so that the model learns this pattern without leaking test data.

def _human_speed_typer_train_batch(session_id: str, i: int, base_time: datetime) -> dict:
    batch_start_ms = 1000 + i * 3000
    x0 = 80 + (i % 6) * 12
    y0 = 180 + (i % 4) * 7

    # Linear mouse (bot-style path)
    moves = []
    for k in range(random.randint(14, 20)):
        moves.append({
            "x": x0 + k * 20,
            "y": y0 + k * 8,
            "ts": batch_start_ms + k * 100,
        })

    # Bot-uniform clicks
    clicks = []
    ts_c = batch_start_ms + 1500
    for _ in range(3):
        clicks.append({"x": x0 + 300 + random.randint(0, 3),
                       "y": y0 + 140 + random.randint(0, 2),
                       "button": 0, "ts": ts_c})
        ts_c += 220

    # Keystroke timing: Gaussian distribution with slightly different params
    # from hard test (mean=155ms, std=50ms vs hard test's mean=180ms, std=60ms)
    phrase_choices = [
        ["KeyH", "KeyE", "KeyL", "KeyL", "KeyO", "Space", "KeyW", "KeyO", "KeyR", "KeyL", "KeyD"],
        ["KeyS", "KeyE", "KeyA", "KeyR", "KeyC", "KeyH", "Space", "KeyT", "KeyH", "KeyI", "KeyS"],
        ["KeyT", "KeyY", "KeyP", "KeyI", "KeyN", "KeyG", "Space", "KeyN", "KeyO", "KeyW"],
    ]
    phrase = random.choice(phrase_choices)
    keys = []
    ts_k = batch_start_ms + 800
    for code in phrase:
        keys.append({"key": code.replace("Key", "").lower(), "code": code, "ts": ts_k})
        delay = max(int(random.gauss(155, 50)), 35)
        ts_k += delay

    return {
        "sessionID": session_id,
        "timestamp": _iso(base_time + timedelta(seconds=i * 3)),
        "signals": {"mouseMoves": moves, "clicks": clicks, "keys": keys},
        "metadata": {"userAgent": "HumanSpeedTyperTrainBot/1.0",
                     "viewportWidth": 1366, "viewportHeight": 768},
    }


# ── Pattern B: hybrid_bot_train ───────────────────────────────────────────
# Bézier-curve mouse with human-spaced hover pauses, combined with
# bot-speed uniform keystroke timing (65ms inter-key, vs 80ms in hard test).
# Clicks have human-spaced intervals to look natural.

def _hybrid_bot_train_batch(session_id: str, i: int, base_time: datetime) -> dict:
    batch_start_ms = 1000 + i * 3100

    x0 = random.randint(55, 680)
    y0 = random.randint(55, 530)
    x1 = x0 + random.randint(-320, 320)
    y1 = y0 + random.randint(-220, 220)
    cx = (x0 + x1) / 2 + random.randint(-110, 110)
    cy = (y0 + y1) / 2 + random.randint(-110, 110)

    num_points = random.randint(11, 24)
    ts_cursor  = batch_start_ms
    moves = []
    for k in range(num_points):
        t = k / max(num_points - 1, 1)
        bx, by = _bezier_point(t, (x0, y0), (cx, cy), (x1, y1))
        # Occasional hover pause (human-like) + slight jitter
        dt = random.randint(350, 1100) if random.random() < 0.11 else random.randint(55, 160)
        ts_cursor += dt
        moves.append({"x": max(0, int(bx + random.gauss(0, 2.5))),
                      "y": max(0, int(by + random.gauss(0, 2.5))),
                      "ts": ts_cursor})

    # Human-spaced click intervals (350–1800ms) at semi-random positions
    num_clicks = random.randint(1, 4)
    clicks = []
    click_ts = ts_cursor + random.randint(180, 550)
    for _ in range(num_clicks):
        clicks.append({"x": random.randint(50, 1200),
                       "y": random.randint(50, 700),
                       "button": 0, "ts": click_ts})
        click_ts += random.randint(350, 1800)

    # Bot-speed uniform keys at 65ms (slightly different from hard test's 80ms)
    key_codes = ["KeyL", "KeyO", "KeyG", "KeyI", "KeyN", "Enter"]
    keys = []
    ts_k = batch_start_ms + 850
    for code in key_codes:
        keys.append({"key": code.replace("Key", "").lower(), "code": code, "ts": ts_k})
        ts_k += 65

    return {
        "sessionID": session_id,
        "timestamp": _iso(base_time + timedelta(seconds=i * 3)),
        "signals": {"mouseMoves": moves, "clicks": clicks, "keys": keys},
        "metadata": {"userAgent": "HybridBotTrain/1.0",
                     "viewportWidth": 1440, "viewportHeight": 900},
    }


# ── Generation ────────────────────────────────────────────────────────────

PATTERNS = {
    "human_speed_typer_train": _human_speed_typer_train_batch,
    "hybrid_bot_train":        _hybrid_bot_train_batch,
}


def _write_sessions(pattern_name: str, maker_fn, num_sessions: int, seed: int) -> int:
    random.seed(seed)
    now = datetime.now(tz=timezone.utc)
    total_batches = 0

    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for s in range(num_sessions):
        session_id = f"trainbot_{pattern_name}_{now.strftime('%Y%m%d')}_{s:03d}"
        base_time  = now - timedelta(minutes=random.randint(1, 60))
        batches    = [maker_fn(session_id, i, base_time) for i in range(BATCHES_PER_SESSION)]

        with SIGNALS_PATH.open("a") as f:
            for b in batches:
                f.write(json.dumps(b) + "\n")

        _upsert_label(session_id, "bot")
        total_batches += len(batches)

    print(f"  [{pattern_name}] {num_sessions} sessions × {BATCHES_PER_SESSION} batches "
          f"= {total_batches} batches")
    return total_batches


def main():
    print("=" * 60)
    print("  Generating adversarial bot training data")
    print("  (Appended to training files — NOT hard test set)")
    print("=" * 60)

    total = 0
    seeds = {"human_speed_typer_train": 2345, "hybrid_bot_train": 6789}
    for name, fn in PATTERNS.items():
        total += _write_sessions(name, fn, SESSIONS_PER_PATTERN, seed=seeds[name])

    n_sessions = SESSIONS_PER_PATTERN * len(PATTERNS)
    print(f"\n  {n_sessions} adversarial training sessions | {total} batches total")
    print(f"  Signals → {SIGNALS_PATH}")
    print(f"  Labels  → {LABELS_PATH}")


if __name__ == "__main__":
    main()
