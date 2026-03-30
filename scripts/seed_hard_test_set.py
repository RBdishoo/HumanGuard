"""
Generate 50 adversarial "hard" bot sessions for out-of-sample evaluation.

These sessions DELIBERATELY mimic human behavioral patterns to stress-test
the trained model. They go into a SEPARATE hard test set and must NOT be
added to signals.jsonl / labels.csv (training data).

5 patterns × 10 sessions × 10 batches = 500 hard test batches.

Output:
  backend/data/raw/hard_test_signals.jsonl
  backend/data/raw/hard_test_labels.csv
"""

import csv
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

HARD_SIGNALS_PATH = Path("backend/data/raw/hard_test_signals.jsonl")
HARD_LABELS_PATH  = Path("backend/data/raw/hard_test_labels.csv")

SESSIONS_PER_PATTERN = 10
BATCHES_PER_SESSION  = 10


# ── shared helpers ────────────────────────────────────────────────────────

def _iso(ts: datetime) -> str:
    return ts.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _bezier_point(t: float, p0, p1, p2) -> tuple:
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return (x, y)


def _bot_keys_uniform(batch_start_ms: int, delay_ms: int = 80) -> list:
    """Uniform-timing bot keystroke sequence."""
    key_codes = ["KeyL", "KeyO", "KeyG", "KeyI", "KeyN", "Enter"]
    keys = []
    ts = batch_start_ms + 900
    for code in key_codes:
        keys.append({"key": code.replace("Key", "").lower(), "code": code, "ts": ts})
        ts += delay_ms
    return keys


def _bot_clicks_uniform(batch_start_ms: int, x0: int, y0: int, interval_ms: int = 250) -> list:
    """Uniform-timing bot click sequence."""
    clicks = []
    ts = batch_start_ms + 1700
    for i in range(3):
        clicks.append({"x": x0 + 360 + i * 2, "y": y0 + 160 + i, "button": 0, "ts": ts})
        ts += interval_ms
    return clicks


def _linear_mouse(batch_start_ms: int, x0: int, y0: int, num_moves: int = 18,
                  dt_ms: int = 100) -> list:
    """Classic perfectly-linear mouse path at fixed cadence."""
    return [
        {"x": x0 + i * 22, "y": y0 + i * 9, "ts": batch_start_ms + i * dt_ms}
        for i in range(num_moves)
    ]


def _wrap_batch(session_id: str, batch_index: int, base_time: datetime,
                mouse_moves: list, clicks: list, keys: list,
                agent: str = "AdversarialBot/1.0") -> dict:
    return {
        "sessionID": session_id,
        "timestamp": _iso(base_time + timedelta(seconds=batch_index * 3)),
        "signals": {"mouseMoves": mouse_moves, "clicks": clicks, "keys": keys},
        "metadata": {"userAgent": agent, "viewportWidth": 1366, "viewportHeight": 768},
    }


# ── Pattern 1: human_speed_typer ─────────────────────────────────────────
# Linear bot mouse + bot clicks, but keystroke timing sampled from a human
# Gaussian distribution (mean=180ms, std=60ms inter-key delay).

def _human_speed_typer_batch(session_id: str, i: int, base_time: datetime) -> dict:
    batch_start_ms = 1000 + i * 3000
    x0 = 100 + (i % 5) * 10
    y0 = 200 + (i % 3) * 8
    moves  = _linear_mouse(batch_start_ms, x0, y0)
    clicks = _bot_clicks_uniform(batch_start_ms, x0, y0)

    # Keystroke timing drawn from human distribution (mean=180ms, std=60ms)
    key_codes = ["KeyH", "KeyE", "KeyL", "KeyL", "KeyO", "Space",
                 "KeyW", "KeyO", "KeyR", "KeyL", "KeyD"]
    keys = []
    ts = batch_start_ms + 900
    for code in key_codes:
        keys.append({"key": code.replace("Key", "").lower(), "code": code, "ts": ts})
        delay = max(int(random.gauss(180, 60)), 40)
        ts += delay

    return _wrap_batch(session_id, i, base_time, moves, clicks, keys,
                       agent="HumanSpeedTyperBot/1.0")


# ── Pattern 2: bezier_mouse ───────────────────────────────────────────────
# Mouse follows a Bézier curve (same as human generator), but clicks and
# keystrokes are uniform-timing bot behaviour.

def _bezier_mouse_batch(session_id: str, i: int, base_time: datetime) -> dict:
    batch_start_ms = 1000 + i * 3000
    x0 = random.randint(50, 700)
    y0 = random.randint(50, 550)
    x1 = x0 + random.randint(-300, 300)
    y1 = y0 + random.randint(-200, 200)
    cx = (x0 + x1) / 2 + random.randint(-150, 150)
    cy = (y0 + y1) / 2 + random.randint(-150, 150)

    num_points = random.randint(10, 22)
    ts_cursor  = batch_start_ms
    moves = []
    for k in range(num_points):
        t = k / max(num_points - 1, 1)
        bx, by = _bezier_point(t, (x0, y0), (cx, cy), (x1, y1))
        dt = random.randint(70, 180)          # variable speed
        ts_cursor += dt
        moves.append({"x": max(0, int(bx + random.gauss(0, 2))),
                      "y": max(0, int(by + random.gauss(0, 2))),
                      "ts": ts_cursor})

    clicks = _bot_clicks_uniform(batch_start_ms, int(x0), int(y0))
    keys   = _bot_keys_uniform(batch_start_ms, delay_ms=80)
    return _wrap_batch(session_id, i, base_time, moves, clicks, keys,
                       agent="BezierMouseBot/1.0")


# ── Pattern 3: jitter_bot ─────────────────────────────────────────────────
# Classic linear bot path + Gaussian noise σ=5px on every coordinate.
# Also adds small timing jitter. Clicks and keys stay uniform.

def _jitter_bot_batch(session_id: str, i: int, base_time: datetime) -> dict:
    batch_start_ms = 1000 + i * 3000
    x0 = 100 + (i % 5) * 10
    y0 = 200 + (i % 3) * 8
    moves = []
    for k in range(18):
        jx = int(random.gauss(0, 5))
        jy = int(random.gauss(0, 5))
        jt = int(random.gauss(0, 8))
        moves.append({
            "x": max(0, x0 + k * 22 + jx),
            "y": max(0, y0 + k * 9  + jy),
            "ts": batch_start_ms + k * 100 + jt,
        })
    # Sort by ts to keep ordering valid
    moves.sort(key=lambda m: m["ts"])

    clicks = _bot_clicks_uniform(batch_start_ms, x0, y0)
    keys   = _bot_keys_uniform(batch_start_ms, delay_ms=80)
    return _wrap_batch(session_id, i, base_time, moves, clicks, keys,
                       agent="JitterBot/1.0")


# ── Pattern 4: hybrid_bot ─────────────────────────────────────────────────
# Human Bézier mouse + human-distributed hover pauses,
# combined with bot-speed uniform keystroke timing (80ms).
# The click intervals are also varied to look human.

def _hybrid_bot_batch(session_id: str, i: int, base_time: datetime) -> dict:
    batch_start_ms = 1000 + i * 3200
    x0 = random.randint(60, 700)
    y0 = random.randint(60, 550)
    x1 = x0 + random.randint(-350, 350)
    y1 = y0 + random.randint(-250, 250)
    cx = (x0 + x1) / 2 + random.randint(-120, 120)
    cy = (y0 + y1) / 2 + random.randint(-120, 120)

    num_points = random.randint(12, 26)
    ts_cursor  = batch_start_ms
    moves = []
    for k in range(num_points):
        t   = k / max(num_points - 1, 1)
        bx, by = _bezier_point(t, (x0, y0), (cx, cy), (x1, y1))
        # Occasional hover pause (human-like)
        dt = random.randint(400, 1200) if random.random() < 0.12 else random.randint(60, 170)
        ts_cursor += dt
        moves.append({"x": max(0, int(bx + random.gauss(0, 3))),
                      "y": max(0, int(by + random.gauss(0, 3))),
                      "ts": ts_cursor})

    # Clicks: human-spaced intervals (400–2000ms) but bot-exact positions
    num_clicks = random.randint(1, 4)
    clicks = []
    click_ts = ts_cursor + random.randint(200, 600)
    for _ in range(num_clicks):
        clicks.append({"x": random.randint(50, 1200),
                       "y": random.randint(50, 700),
                       "button": 0, "ts": click_ts})
        click_ts += random.randint(400, 2000)

    # Keys: bot-speed uniform 80ms — the tell-tale bot signal
    keys = _bot_keys_uniform(batch_start_ms, delay_ms=80)
    return _wrap_batch(session_id, i, base_time, moves, clicks, keys,
                       agent="HybridBot/1.0")


# ── Pattern 5: adaptive_bot ───────────────────────────────────────────────
# Batches 0–4: behaves like a real human (Bézier, natural timing).
# Batches 5–9: switches to classic bot behaviour (linear, uniform).
# Mid-session drift is the key detection challenge.

def _adaptive_bot_batch(session_id: str, i: int, base_time: datetime) -> dict:
    batch_start_ms = 1000 + i * 3000
    human_phase = i < BATCHES_PER_SESSION // 2

    if human_phase:
        # Identical to bezier_mouse logic
        x0 = random.randint(50, 700)
        y0 = random.randint(50, 550)
        x1 = x0 + random.randint(-300, 300)
        y1 = y0 + random.randint(-200, 200)
        cx = (x0 + x1) / 2 + random.randint(-150, 150)
        cy = (y0 + y1) / 2 + random.randint(-150, 150)
        num_points = random.randint(10, 24)
        ts_cursor = batch_start_ms
        moves = []
        for k in range(num_points):
            t = k / max(num_points - 1, 1)
            bx, by = _bezier_point(t, (x0, y0), (cx, cy), (x1, y1))
            dt = random.randint(400, 1000) if random.random() < 0.13 else random.randint(60, 190)
            ts_cursor += dt
            moves.append({"x": max(0, int(bx + random.gauss(0, 3))),
                          "y": max(0, int(by + random.gauss(0, 3))),
                          "ts": ts_cursor})

        # Natural key timing
        phrase = random.choice(["hello world", "search this", "open new tab", "go back"])
        keys = []
        ts_k = batch_start_ms + 900
        for ch in phrase:
            code = f"Key{ch.upper()}" if ch.isalpha() else "Space"
            keys.append({"key": ch, "code": code, "ts": ts_k})
            ts_k += max(int(random.gauss(175, 55)), 40)

        clicks = []
        if random.random() < 0.7:
            click_ts = batch_start_ms + random.randint(1500, 4000)
            for _ in range(random.randint(1, 3)):
                clicks.append({"x": random.randint(50, 1200), "y": random.randint(50, 700),
                               "button": 0, "ts": click_ts})
                click_ts += random.randint(500, 2500)

        agent = "AdaptiveBot/1.0-human-phase"
        x0_int = int(x0)
        y0_int = int(y0)
    else:
        # Classic bot
        x0_int = 100 + (i % 5) * 10
        y0_int = 200 + (i % 3) * 8
        moves  = _linear_mouse(batch_start_ms, x0_int, y0_int)
        clicks = _bot_clicks_uniform(batch_start_ms, x0_int, y0_int)
        keys   = _bot_keys_uniform(batch_start_ms, delay_ms=80)
        agent  = "AdaptiveBot/1.0-bot-phase"

    return _wrap_batch(session_id, i, base_time, moves, clicks, keys, agent=agent)


# ── Generation ────────────────────────────────────────────────────────────

PATTERNS = {
    "human_speed_typer": _human_speed_typer_batch,
    "bezier_mouse":      _bezier_mouse_batch,
    "jitter_bot":        _jitter_bot_batch,
    "hybrid_bot":        _hybrid_bot_batch,
    "adaptive_bot":      _adaptive_bot_batch,
}


def _write_sessions(pattern_name: str, maker_fn, num_sessions: int, seed: int) -> int:
    random.seed(seed)
    now = datetime.now(tz=timezone.utc)
    total_batches = 0
    label_rows = []

    for s in range(num_sessions):
        session_id = f"hardbot_{pattern_name}_{now.strftime('%Y%m%d')}_{s:03d}"
        base_time  = now - timedelta(minutes=random.randint(1, 30))
        batches = [maker_fn(session_id, i, base_time) for i in range(BATCHES_PER_SESSION)]
        with HARD_SIGNALS_PATH.open("a") as f:
            for b in batches:
                f.write(json.dumps(b) + "\n")
        label_rows.append({"sessionID": session_id, "label": "bot"})
        total_batches += len(batches)

    # Append to labels CSV
    existing = []
    if HARD_LABELS_PATH.exists():
        with HARD_LABELS_PATH.open("r", newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
    existing.extend(label_rows)
    with HARD_LABELS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sessionID", "label"])
        writer.writeheader()
        writer.writerows(existing)

    print(f"  [{pattern_name}] {num_sessions} sessions × {BATCHES_PER_SESSION} batches = {total_batches} batches")
    return total_batches


def main():
    # Clear previous hard test data so this script is idempotent
    HARD_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HARD_SIGNALS_PATH.unlink(missing_ok=True)
    HARD_LABELS_PATH.unlink(missing_ok=True)

    print("=" * 60)
    print("  Generating adversarial hard test set")
    print("  (NOT added to training data)")
    print("=" * 60)

    total = 0
    for offset, (name, fn) in enumerate(PATTERNS.items()):
        total += _write_sessions(name, fn, SESSIONS_PER_PATTERN, seed=7000 + offset * 13)

    n_sessions = SESSIONS_PER_PATTERN * len(PATTERNS)
    print(f"\n  {n_sessions} hard bot sessions | {total} batches total")
    print(f"  Signals → {HARD_SIGNALS_PATH}")
    print(f"  Labels  → {HARD_LABELS_PATH}")


if __name__ == "__main__":
    main()
