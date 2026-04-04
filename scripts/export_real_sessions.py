"""
Export real_human signal batches to a labeled CSV for retraining.

Usage:
    python scripts/export_real_sessions.py
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# Allow imports from project root (backend.features.*)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.features.feature_extractor import FeatureExtractor

SIGNALS_FILE = ROOT / "backend" / "data" / "raw" / "signals.jsonl"
OUTPUT_FILE  = ROOT / "backend" / "data" / "raw" / "real_human_sessions.csv"


def main():
    if not SIGNALS_FILE.exists():
        print("No signals file found")
        sys.exit(0)

    extractor = FeatureExtractor()
    rows = []
    utm_counts = Counter()

    with open(SIGNALS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("source") != "real_human":
                continue

            signals = record.get("signals", {})
            try:
                features = extractor.extractBatchFeatures(signals)
            except Exception:
                continue

            utm_source = record.get("utm_source", "direct")
            utm_counts[utm_source] += 1

            row = {
                "sessionID":  record.get("sessionID", ""),
                "source":     "real_human",
                "utm_source": utm_source,
                "label":      "human",
            }
            row.update(features)
            rows.append(row)

    if not rows:
        print("Found 0 real human sessions")
        print(f"Exported to {OUTPUT_FILE}")
        print("Next step: run scripts/retrain.py when you have 50+ sessions")
        sys.exit(0)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found {len(rows)} real human sessions")
    for utm_src, count in sorted(utm_counts.items()):
        print(f"  {utm_src}: {count}")
    print(f"Exported to {OUTPUT_FILE}")
    print("Next step: run scripts/retrain.py when you have 50+ sessions")


if __name__ == "__main__":
    main()
