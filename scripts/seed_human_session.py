"""
Generate realistic human-like signal sessions for training data.

Produces batches with natural behavioral patterns:
  - Curved mouse paths with variable speed, jitter, and pauses
  - Irregular keystroke timing with diverse characters
  - Naturally spaced clicks with varied positions
"""

import csv
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

SIGNALS_PATH = Path("backend/data/raw/signals.jsonl")
LABELS_PATH = Path("backend/data/raw/labels.csv")


def _iso(ts: datetime) -> str:
    return ts.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _bezier_point(t: float, p0, p1, p2) -> tuple:
    """Quadratic Bézier curve point at parameter t."""
    x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    return (x, y)


def _make_human_batch(session_id: str, batch_index: int, base_time: datetime,
                      ambiguous: bool = False) -> dict:
    batch_start_ms = 1000 + (batch_index * random.randint(2500, 4500))

    # --- Mouse path ---
    mouse_moves = []
    x_start = random.randint(50, 800)
    y_start = random.randint(50, 600)

    if ambiguous:
        # Near-linear path mimicking a human using a drawing tablet / trackpad.
        # Matches bot structure: ~18 points, ~100ms cadence, linear increments.
        num_points = random.randint(16, 20)
        ts_cursor = batch_start_ms
        for i in range(num_points):
            ts_cursor += 100 + random.randint(-3, 3)
            mouse_moves.append({
                "x": x_start + (i * 22) + (i % 2) + random.randint(-1, 1),
                "y": y_start + (i * 9) + random.randint(-1, 1),
                "ts": ts_cursor,
            })
    else:
        x_end = x_start + random.randint(-300, 300)
        y_end = y_start + random.randint(-200, 200)
        cx = (x_start + x_end) / 2 + random.randint(-150, 150)
        cy = (y_start + y_end) / 2 + random.randint(-150, 150)
        num_points = random.randint(8, 25)
        ts_cursor = batch_start_ms
        for i in range(num_points):
            t = i / max(num_points - 1, 1)
            bx, by = _bezier_point(t, (x_start, y_start), (cx, cy), (x_end, y_end))
            jx = bx + random.gauss(0, 4)
            jy = by + random.gauss(0, 4)
            if random.random() < 0.15:
                dt = random.randint(400, 1500)
            else:
                dt = random.randint(60, 200)
            ts_cursor += dt
            mouse_moves.append({
                "x": max(0, int(jx)),
                "y": max(0, int(jy)),
                "ts": ts_cursor,
            })

    # --- Clicks ---
    if ambiguous:
        # 3 clustered clicks like bots, positioned near end of mouse path
        click_ts = batch_start_ms + random.randint(1650, 1750)
        num_clicks = 3
        clicks = []
        for _ in range(num_clicks):
            clicks.append({
                "x": x_start + 360 + random.randint(-8, 8),
                "y": y_start + 160 + random.randint(-5, 5),
                "button": 0,
                "ts": click_ts,
            })
            click_ts += random.randint(220, 280)
    else:
        num_clicks = random.randint(0, 4)
        clicks = []
        click_ts = ts_cursor + random.randint(200, 800)
        for _ in range(num_clicks):
            clicks.append({
                "x": random.randint(50, 1200),
                "y": random.randint(50, 700),
                "button": 0 if random.random() < 0.9 else random.choice([1, 2]),
                "ts": click_ts,
            })
            click_ts += random.randint(400, 3000)

    # --- Keys: natural typing with diverse characters and variable delays ---
    phrases = [
        "hello world", "search query", "username", "password123",
        "click here", "the quick brown", "testing", "open new tab",
        "www dot example", "let me check", "back to home",
        "how does this work", "submit form", "next page",
        "login", "ok", "yes", "no", "admin", "test",  # short phrases like bot patterns
    ]
    if ambiguous:
        # Varied short strings (5-8 chars) with near-uniform timing — fast typist
        phrase = random.choice(["login", "admins", "test123", "pass12", "search", "signup", "logout1"])
    else:
        phrase = random.choice(phrases)
    keys = []
    if ambiguous:
        # Start keys at similar offset as bot (batch_start + ~900ms)
        key_ts = batch_start_ms + random.randint(850, 950)
    else:
        key_ts = (click_ts if clicks else ts_cursor) + random.randint(100, 600)
    for ch in phrase:
        code = f"Key{ch.upper()}" if ch.isalpha() else "Space" if ch == " " else f"Digit{ch}" if ch.isdigit() else ch
        keys.append({"key": ch, "code": code, "ts": key_ts})
        if ambiguous:
            # Near-constant delay (~80ms like bots, occasional rapid press)
            key_ts += int(random.gauss(80, 15))
        else:
            key_ts += int(random.gauss(180, 60))
        key_ts = max(key_ts, keys[-1]["ts"] + 30)  # floor at 30ms

    return {
        "sessionID": session_id,
        "timestamp": _iso(base_time + timedelta(seconds=batch_index * random.randint(3, 6))),
        "signals": {
            "mouseMoves": mouse_moves,
            "clicks": clicks,
            "keys": keys,
        },
        "metadata": {
            "userAgent": random.choice([
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            ]),
            "viewportWidth": random.choice([1280, 1366, 1440, 1920]),
            "viewportHeight": random.choice([720, 768, 900, 1080]),
            "network_features": {
                "is_headless_browser": False,
                "is_known_bot_ua": False,
                "is_datacenter_ip": False,
                "ua_entropy": round(random.uniform(72.0, 98.0), 2),
                "has_accept_language": True,
                "accept_language_count": random.randint(2, 5),
                "suspicious_header_count": 0,
            },
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
    num_sessions = 5
    batches_per_session = 10
    # 2 of the 5 sessions are "ambiguous" — humans with more linear/regular patterns
    ambiguous_sessions = {3, 4}

    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    for s in range(num_sessions):
        session_id = f"human_sim_{datetime.now(tz=timezone.utc).strftime('%Y%m%d%H%M%S')}_{s}"
        base_time = datetime.now(tz=timezone.utc) - timedelta(minutes=random.randint(1, 10))
        is_ambiguous = s in ambiguous_sessions

        with SIGNALS_PATH.open("a") as f:
            for i in range(batches_per_session):
                rec = _make_human_batch(session_id, i, base_time, ambiguous=is_ambiguous)
                f.write(json.dumps(rec) + "\n")
                total += 1

        _upsert_label(session_id, "human")
        kind = "ambiguous" if is_ambiguous else "normal"
        print(f"  Session {session_id}: {batches_per_session} batches ({kind})")

    print(f"\nAppended {total} human batches to {SIGNALS_PATH}")
    print(f"Updated labels in {LABELS_PATH}")


if __name__ == "__main__":
    main()
