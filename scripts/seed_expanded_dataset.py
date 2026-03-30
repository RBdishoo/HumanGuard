"""
Generate expanded training dataset with 5 new bot behavior patterns and
additional human sessions.

Bot patterns (30 sessions each = 150 new bot sessions):
  1. slow_deliberate  — human-speed but perfectly linear mouse
  2. mobile_bot       — touch-like: no hover, click-heavy, mobile viewport
  3. headless         — no mouse, no clicks, instant keystrokes
  4. semi_human       — classic bot + Gaussian jitter on coords/timing
  5. replay           — every batch in the session is identical

Human additions: 100 new sessions with natural variance.
"""

import csv
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SIGNALS_PATH = Path("backend/data/raw/signals.jsonl")
LABELS_PATH  = Path("backend/data/raw/labels.csv")

BATCHES_PER_BOT_SESSION   = 10
BATCHES_PER_HUMAN_SESSION = 10


# ── helpers ────────────────────────────────────────────────────────────────

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


def _write_batches(session_id: str, batches: list) -> None:
    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SIGNALS_PATH.open("a") as f:
        for b in batches:
            f.write(json.dumps(b) + "\n")


# ── Bot pattern 1: slow_deliberate ────────────────────────────────────────
# Human-speed (~200 ms between moves) but perfectly linear trajectory.
# Uniform click timing, uniform keystroke timing.

def _slow_deliberate_batch(session_id: str, batch_index: int, base_time: datetime) -> dict:
    batch_start_ms = 1000 + batch_index * 3500
    x0 = 80 + (batch_index % 6) * 15
    y0 = 150 + (batch_index % 4) * 12
    num_moves = 15
    mouse_moves = []
    for i in range(num_moves):
        mouse_moves.append({
            "x": x0 + i * 18,
            "y": y0 + i * 7,
            "ts": batch_start_ms + i * 200,   # human-speed, no jitter
        })

    # 3 clicks at perfectly uniform ~1 500 ms intervals
    clicks = []
    for i in range(3):
        clicks.append({
            "x": x0 + 270 + i * 2,
            "y": y0 + 100 + i,
            "button": 0,
            "ts": batch_start_ms + 1500 + i * 1500,
        })

    # Keys: uniform 120 ms delay
    key_codes = ["KeyU", "KeyS", "KeyE", "KeyR", "KeyN", "KeyA", "KeyM", "KeyE"]
    keys = []
    key_ts = batch_start_ms + 800
    for code in key_codes:
        keys.append({"key": code.replace("Key", "").lower(), "code": code, "ts": key_ts})
        key_ts += 120

    return {
        "sessionID": session_id,
        "timestamp": _iso(base_time + timedelta(seconds=batch_index * 3)),
        "signals": {"mouseMoves": mouse_moves, "clicks": clicks, "keys": keys},
        "metadata": {"userAgent": "SlowBotSim/1.0", "viewportWidth": 1366, "viewportHeight": 768},
    }


# ── Bot pattern 2: mobile_bot ─────────────────────────────────────────────
# Simulates touch-based automation: no mouse hover events (or very sparse),
# click-heavy rapid taps, small mobile viewport.

def _mobile_bot_batch(session_id: str, batch_index: int, base_time: datetime) -> dict:
    batch_start_ms = 1000 + batch_index * 2500

    # Sparse mouse moves (touch drag, 3-4 points only)
    num_moves = random.randint(3, 5)
    x0 = random.randint(50, 300)
    y0 = random.randint(100, 500)
    mouse_moves = []
    for i in range(num_moves):
        mouse_moves.append({
            "x": x0 + i * 10,
            "y": y0 + i * 5,
            "ts": batch_start_ms + i * 80,
        })

    # Rapid taps — 5 clicks clustered together
    clicks = []
    tap_x = random.randint(50, 350)
    tap_y = random.randint(100, 600)
    for i in range(5):
        clicks.append({
            "x": tap_x + random.randint(-2, 2),
            "y": tap_y + random.randint(-2, 2),
            "button": 0,
            "ts": batch_start_ms + 500 + i * 250,  # 250 ms uniform tap rate
        })

    # Short uniform key sequence
    key_codes = ["KeyO", "KeyK", "Enter"]
    keys = []
    key_ts = batch_start_ms + 1800
    for code in key_codes:
        keys.append({"key": code.replace("Key", "").lower(), "code": code, "ts": key_ts})
        key_ts += 90

    return {
        "sessionID": session_id,
        "timestamp": _iso(base_time + timedelta(seconds=batch_index * 2)),
        "signals": {"mouseMoves": mouse_moves, "clicks": clicks, "keys": keys},
        "metadata": {
            "userAgent": "MobileBotSim/1.0 (Mobile; Android)",
            "viewportWidth": 390,
            "viewportHeight": 844,
        },
    }


# ── Bot pattern 3: headless ───────────────────────────────────────────────
# No mouse, no clicks. Only keystrokes with near-instant timing (< 20 ms).

def _headless_batch(session_id: str, batch_index: int, base_time: datetime) -> dict:
    batch_start_ms = 1000 + batch_index * 2000

    phrases = [
        ["KeyA", "KeyD", "KeyM", "KeyI", "KeyN"],
        ["KeyP", "KeyA", "KeyS", "KeyS", "Enter"],
        ["KeyS", "KeyE", "KeyA", "KeyR", "KeyC", "KeyH", "Enter"],
        ["KeyL", "KeyO", "KeyG", "KeyI", "KeyN"],
        ["KeyS", "KeyU", "KeyB", "KeyM", "KeyI", "KeyT"],
    ]
    key_codes = phrases[batch_index % len(phrases)]
    keys = []
    key_ts = batch_start_ms + 50
    for code in key_codes:
        keys.append({"key": code.replace("Key", "").lower(), "code": code, "ts": key_ts})
        key_ts += random.randint(8, 18)   # headless: < 20 ms between keys

    return {
        "sessionID": session_id,
        "timestamp": _iso(base_time + timedelta(seconds=batch_index * 2)),
        "signals": {"mouseMoves": [], "clicks": [], "keys": keys},
        "metadata": {
            "userAgent": "HeadlessChrome/120.0",
            "viewportWidth": 1280,
            "viewportHeight": 720,
        },
    }


# ── Bot pattern 4: semi_human ─────────────────────────────────────────────
# Classic bot path + Gaussian jitter on coordinates and timing.
# Still clearly automated: no curvature, deterministic structure.

def _semi_human_batch(session_id: str, batch_index: int, base_time: datetime) -> dict:
    batch_start_ms = 1000 + batch_index * 3000
    x0 = 100 + (batch_index % 5) * 10
    y0 = 200 + (batch_index % 3) * 8
    num_moves = 18
    mouse_moves = []
    for i in range(num_moves):
        jx = int(random.gauss(0, 5))   # small Gaussian jitter
        jy = int(random.gauss(0, 3))
        dt = int(100 + random.gauss(0, 20))
        dt = max(dt, 30)
        mouse_moves.append({
            "x": x0 + i * 22 + jx,
            "y": y0 + i * 9 + jy,
            "ts": batch_start_ms + i * dt,
        })

    click_base = batch_start_ms + 1700
    clicks = [
        {"x": x0 + 360 + int(random.gauss(0, 4)), "y": y0 + 160 + int(random.gauss(0, 3)),
         "button": 0, "ts": click_base},
        {"x": x0 + 362 + int(random.gauss(0, 4)), "y": y0 + 162 + int(random.gauss(0, 3)),
         "button": 0, "ts": click_base + int(250 + random.gauss(0, 15))},
        {"x": x0 + 365 + int(random.gauss(0, 4)), "y": y0 + 165 + int(random.gauss(0, 3)),
         "button": 0, "ts": click_base + int(500 + random.gauss(0, 20))},
    ]

    key_codes = ["KeyL", "KeyO", "KeyG", "KeyI", "KeyN", "Enter"]
    keys = []
    key_ts = batch_start_ms + 900
    for code in key_codes:
        keys.append({"key": code.replace("Key", "").lower(), "code": code, "ts": key_ts})
        key_ts += int(80 + random.gauss(0, 10))

    return {
        "sessionID": session_id,
        "timestamp": _iso(base_time + timedelta(seconds=batch_index * 3)),
        "signals": {"mouseMoves": mouse_moves, "clicks": clicks, "keys": keys},
        "metadata": {"userAgent": "SemiHumanBot/1.0", "viewportWidth": 1366, "viewportHeight": 768},
    }


# ── Bot pattern 5: replay ─────────────────────────────────────────────────
# Every batch in the session repeats the exact same fixed pattern.

def _replay_template(session_id: str, base_time: datetime) -> dict:
    """Build the single fixed template (no randomness)."""
    mouse_moves = [
        {"x": 200 + i * 20, "y": 300 + i * 8, "ts": 1000 + i * 100}
        for i in range(16)
    ]
    clicks = [
        {"x": 520, "y": 428, "button": 0, "ts": 2700},
        {"x": 522, "y": 429, "button": 0, "ts": 2950},
        {"x": 524, "y": 430, "button": 0, "ts": 3200},
    ]
    key_codes = ["KeyT", "KeyE", "KeyS", "KeyT", "Enter"]
    keys = [{"key": c.replace("Key", "").lower(), "code": c, "ts": 1800 + i * 80}
            for i, c in enumerate(key_codes)]
    return {"mouseMoves": mouse_moves, "clicks": clicks, "keys": keys}


def _replay_batch(session_id: str, batch_index: int, base_time: datetime,
                  template: dict) -> dict:
    return {
        "sessionID": session_id,
        "timestamp": _iso(base_time + timedelta(seconds=batch_index * 3)),
        "signals": template,
        "metadata": {"userAgent": "ReplayBot/1.0", "viewportWidth": 1366, "viewportHeight": 768},
    }


# ── Human session generator ───────────────────────────────────────────────

def _bezier_point(t: float, p0, p1, p2) -> tuple:
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return (x, y)


def _human_batch(session_id: str, batch_index: int, base_time: datetime,
                 style: str = "normal") -> dict:
    """
    Styles:
      normal    — Bézier curved paths, irregular timing, diverse clicks
      fast      — quick cursor movements, shorter pauses
      explorer  — lots of mouse movement, fewer keystrokes
      reader    — minimal movement, slow scrolling, long pauses
    """
    batch_start_ms = 1000 + batch_index * random.randint(2000, 5000)

    x_start = random.randint(50, 900)
    y_start = random.randint(50, 650)
    x_end   = x_start + random.randint(-400, 400)
    y_end   = y_start + random.randint(-300, 300)
    cx = (x_start + x_end) / 2 + random.randint(-200, 200)
    cy = (y_start + y_end) / 2 + random.randint(-200, 200)

    if style == "fast":
        num_points = random.randint(6, 14)
        pause_prob = 0.05
        base_dt = (40, 120)
    elif style == "explorer":
        num_points = random.randint(20, 35)
        pause_prob = 0.10
        base_dt = (50, 180)
    elif style == "reader":
        num_points = random.randint(4, 10)
        pause_prob = 0.25
        base_dt = (80, 300)
    else:  # normal
        num_points = random.randint(8, 25)
        pause_prob = 0.15
        base_dt = (60, 200)

    ts_cursor = batch_start_ms
    mouse_moves = []
    for i in range(num_points):
        t = i / max(num_points - 1, 1)
        bx, by = _bezier_point(t, (x_start, y_start), (cx, cy), (x_end, y_end))
        jx = bx + random.gauss(0, 4)
        jy = by + random.gauss(0, 4)
        dt = random.randint(400, 1500) if random.random() < pause_prob else random.randint(*base_dt)
        ts_cursor += dt
        mouse_moves.append({"x": max(0, int(jx)), "y": max(0, int(jy)), "ts": ts_cursor})

    # Clicks
    num_clicks = random.randint(0, 5) if style == "explorer" else random.randint(0, 3)
    clicks = []
    click_ts = ts_cursor + random.randint(200, 800)
    for _ in range(num_clicks):
        clicks.append({
            "x": random.randint(50, 1200),
            "y": random.randint(50, 700),
            "button": 0 if random.random() < 0.9 else random.choice([1, 2]),
            "ts": click_ts,
        })
        click_ts += random.randint(400, 3500)

    # Keys
    phrases = [
        "hello world", "search query", "username here", "password",
        "the quick brown fox", "testing one two three", "open new tab",
        "how does this work", "submit the form", "next page please",
        "login now", "yes no maybe", "admin panel", "test input",
        "select all", "copy paste", "undo redo", "back home",
    ]
    phrase = random.choice(phrases)
    keys = []
    key_ts = (click_ts if clicks else ts_cursor) + random.randint(100, 700)
    mean_delay = 160 if style in ("normal", "reader") else 120
    std_delay  = 55  if style in ("normal", "reader") else 40
    for ch in phrase:
        code = (f"Key{ch.upper()}" if ch.isalpha()
                else "Space" if ch == " "
                else f"Digit{ch}" if ch.isdigit()
                else "Minus")
        keys.append({"key": ch, "code": code, "ts": key_ts})
        key_ts += max(int(random.gauss(mean_delay, std_delay)), 30)

    return {
        "sessionID": session_id,
        "timestamp": _iso(base_time + timedelta(seconds=batch_index * random.randint(3, 7))),
        "signals": {"mouseMoves": mouse_moves, "clicks": clicks, "keys": keys},
        "metadata": {
            "userAgent": random.choice([
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
            ]),
            "viewportWidth":  random.choice([375, 768, 1280, 1366, 1440, 1920]),
            "viewportHeight": random.choice([667, 1024, 720, 768, 900, 1080]),
        },
    }


# ── Generation runners ────────────────────────────────────────────────────

BOT_PATTERNS = {
    "slow_deliberate": _slow_deliberate_batch,
    "mobile_bot":      _mobile_bot_batch,
    "headless":        _headless_batch,
    "semi_human":      _semi_human_batch,
}

HUMAN_STYLES = ["normal", "normal", "normal", "fast", "explorer", "reader"]


def generate_bot_sessions(pattern_name: str, maker_fn, num_sessions: int, seed: int):
    random.seed(seed)
    total_batches = 0
    now = datetime.now(tz=timezone.utc)
    for s in range(num_sessions):
        session_id = f"bot_{pattern_name}_{now.strftime('%Y%m%d')}_{s:03d}"
        base_time  = now - timedelta(minutes=random.randint(1, 30))
        batches = [
            maker_fn(session_id, i, base_time)
            for i in range(BATCHES_PER_BOT_SESSION)
        ]
        _write_batches(session_id, batches)
        _upsert_label(session_id, "bot")
        total_batches += len(batches)
    print(f"  [{pattern_name}] {num_sessions} sessions × {BATCHES_PER_BOT_SESSION} batches = {total_batches} batches")
    return total_batches


def generate_replay_sessions(num_sessions: int, seed: int):
    random.seed(seed)
    total_batches = 0
    now = datetime.now(tz=timezone.utc)
    for s in range(num_sessions):
        session_id = f"bot_replay_{now.strftime('%Y%m%d')}_{s:03d}"
        base_time  = now - timedelta(minutes=random.randint(1, 30))
        template   = _replay_template(session_id, base_time)
        batches = [
            _replay_batch(session_id, i, base_time, template)
            for i in range(BATCHES_PER_BOT_SESSION)
        ]
        _write_batches(session_id, batches)
        _upsert_label(session_id, "bot")
        total_batches += len(batches)
    print(f"  [replay] {num_sessions} sessions × {BATCHES_PER_BOT_SESSION} batches = {total_batches} batches")
    return total_batches


def generate_human_sessions(num_sessions: int, seed: int):
    random.seed(seed)
    total_batches = 0
    now = datetime.now(tz=timezone.utc)
    styles = HUMAN_STYLES
    for s in range(num_sessions):
        style      = styles[s % len(styles)]
        session_id = f"human_exp_{now.strftime('%Y%m%d')}_{s:03d}"
        base_time  = now - timedelta(minutes=random.randint(1, 60))
        batches = [
            _human_batch(session_id, i, base_time, style=style)
            for i in range(BATCHES_PER_HUMAN_SESSION)
        ]
        _write_batches(session_id, batches)
        _upsert_label(session_id, "human")
        total_batches += len(batches)
    print(f"  [human/{style}] {num_sessions} sessions × {BATCHES_PER_HUMAN_SESSION} batches = {total_batches} batches")
    return total_batches


# ── main ──────────────────────────────────────────────────────────────────

def main():
    NUM_SESSIONS_PER_PATTERN = 30
    NUM_HUMAN_SESSIONS       = 100

    print("=" * 60)
    print("  Generating expanded HumanGuard training dataset")
    print("=" * 60)
    print(f"\nBot sessions ({NUM_SESSIONS_PER_PATTERN} × 5 patterns = {NUM_SESSIONS_PER_PATTERN*5}):")

    total_bot_batches = 0
    for seed_offset, (name, fn) in enumerate(BOT_PATTERNS.items()):
        total_bot_batches += generate_bot_sessions(name, fn, NUM_SESSIONS_PER_PATTERN,
                                                   seed=100 + seed_offset * 31)
    total_bot_batches += generate_replay_sessions(NUM_SESSIONS_PER_PATTERN, seed=999)

    print(f"\nHuman sessions ({NUM_HUMAN_SESSIONS}):")
    total_human_batches = generate_human_sessions(NUM_HUMAN_SESSIONS, seed=42)

    print(f"\nDone.")
    print(f"  New bot batches   : {total_bot_batches}")
    print(f"  New human batches : {total_human_batches}")
    print(f"  Total new batches : {total_bot_batches + total_human_batches}")
    print(f"\nSignals  → {SIGNALS_PATH}")
    print(f"Labels   → {LABELS_PATH}")


if __name__ == "__main__":
    main()
