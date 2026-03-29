#!/usr/bin/env python3
"""
HumanGuard Retrain Pipeline

Pulls ground-truth labels from /api/export, merges with existing training data,
retrains XGBoost, and optionally promotes to the model registry.

Usage:
    # Dry-run (evaluate only, no push):
    python scripts/retrain.py --api-url http://localhost:5050 --export-key devkey

    # Train and push to registry if accuracy >= current champion:
    MODEL_BUCKET=humanguard-models python scripts/retrain.py \\
        --api-url http://localhost:5050 --export-key devkey --push

    # Use a local labels CSV instead of the API:
    python scripts/retrain.py --local-labels path/to/extra_labels.csv
"""

import argparse
import csv
import io
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "models"))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retrain")

# Default paths
SIGNALS_JSONL  = REPO_ROOT / "backend" / "data" / "raw" / "signals.jsonl"
LABELS_CSV     = REPO_ROOT / "backend" / "data" / "raw" / "labels.csv"
FEATURES_CSV   = REPO_ROOT / "backend" / "data" / "processed" / "training_data_batches.csv"
TRAINED_DIR    = REPO_ROOT / "models" / "trained"


# ── Helpers ────────────────────────────────────────────────────────────────────

def fetch_export_labels(api_url: str, export_key: str,
                        sources: list = ("demo", "simulator")) -> pd.DataFrame:
    """Fetch labeled sessions from GET /api/export and return a filtered DataFrame."""
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests library is required: pip install requests")

    url = f"{api_url.rstrip('/')}/api/export"
    logger.info("Fetching labels from %s", url)
    resp = requests.get(url, headers={"X-Export-Key": export_key}, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    logger.info("Export returned %d rows", len(df))

    # Keep only rows with a ground-truth label from the requested sources
    if "source" in df.columns:
        df = df[df["source"].isin(sources)]
    if "ground_truth_label" in df.columns:
        df = df[df["ground_truth_label"].isin(("human", "bot"))]
        df = df.rename(columns={"session_id": "sessionID", "ground_truth_label": "label"})
    else:
        logger.warning("Export CSV missing ground_truth_label column — no new labels added")
        df = pd.DataFrame(columns=["sessionID", "label"])

    logger.info("  %d labeled demo/simulator sessions", len(df))
    return df[["sessionID", "label"]]


def load_local_extra_labels(path: str) -> pd.DataFrame:
    """Load extra labels from a local CSV with columns sessionID,label."""
    df = pd.read_csv(path)
    required = {"sessionID", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Extra labels CSV is missing columns: {missing}")
    df = df[df["label"].isin(("human", "bot"))]
    return df[["sessionID", "label"]]


def merge_labels(existing_path: Path, new_labels: pd.DataFrame) -> pd.DataFrame:
    """Merge new labels into existing labels, with new labels winning on conflict."""
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame(columns=["sessionID", "label"])
    combined = pd.concat([existing, new_labels], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sessionID"], keep="last")
    logger.info("Labels: %d existing + %d new = %d total (after dedup)",
                len(existing), len(new_labels), len(combined))
    return combined


def build_features(signals_path: Path, features_out: Path):
    """Run DatasetBuilder to extract batch-level features from signals.jsonl."""
    from backend.features.dataset_builder import DatasetBuilder
    builder = DatasetBuilder(str(signals_path))
    logger.info("Extracting features from %s…", signals_path)
    builder.buildBatchLevelDataset(str(features_out))
    logger.info("Feature CSV written to %s", features_out)


def train_xgboost(features_csv: Path, labels_csv: Path):
    """Train XGBoost and return (model, scaler, metrics, feature_names)."""
    from models.dataset import ModelDataset
    from models.train import ModelTrainer

    dataset = ModelDataset(str(features_csv), str(labels_csv))
    xTrain, xTest, yTrain, yTest, feature_names, scaler = dataset.prepare()

    logger.info("Train: %d rows  Test: %d rows  Features: %d",
                xTrain.shape[0], xTest.shape[0], xTrain.shape[1])

    trainer = ModelTrainer(outputDir=str(TRAINED_DIR))
    xgb_model, metrics, _ = trainer.trainXGBoost(xTrain, xTest, yTrain, yTest, feature_names)
    return xgb_model, scaler, metrics, feature_names


def load_champion_metrics() -> dict:
    """Load metrics for the current champion (registry or local model_comparison.json)."""
    model_bucket = os.environ.get("MODEL_BUCKET")
    if model_bucket:
        try:
            sys.path.insert(0, str(REPO_ROOT / "backend"))
            from model_registry import ModelRegistry
            registry = ModelRegistry(bucket=model_bucket)
            meta = registry.get_metadata("latest")
            if meta:
                return meta
        except Exception as exc:
            logger.warning("Could not read registry champion: %s", exc)

    # Fall back to local comparison file
    comp_path = TRAINED_DIR / "model_comparison.json"
    if comp_path.exists():
        with open(comp_path) as f:
            comp = json.load(f)
        winner = comp.get("winner", "XGBoost")
        for m in comp.get("models", []):
            if m.get("model") == winner:
                return m
    return {}


def print_comparison(new_metrics: dict, champion_metrics: dict):
    """Print a side-by-side comparison table."""
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    champion_name = champion_metrics.get("model") or champion_metrics.get("model_type") or "Champion"
    print("\n" + "=" * 64)
    print(f"{'Metric':<14} {'New Model':>14} {'Current Champion':>18}   Delta")
    print("-" * 64)
    for k in keys:
        new_val = new_metrics.get(k)
        old_val = champion_metrics.get(k)
        if new_val is None:
            continue
        new_str  = f"{new_val:.4f}"
        old_str  = f"{old_val:.4f}" if old_val is not None else "    —   "
        delta    = f"{new_val - old_val:+.4f}" if old_val is not None else "      —"
        better   = "▲" if old_val is not None and new_val > old_val else ("▼" if old_val is not None and new_val < old_val else " ")
        print(f"{k:<14} {new_str:>14} {old_str:>18}   {delta} {better}")
    print("=" * 64)
    print(f"  New model type: {new_metrics.get('model', 'XGBoost')}")
    print(f"  Current champion: {champion_name}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HumanGuard Retrain Pipeline")
    parser.add_argument("--api-url", default=os.environ.get("API_URL", "http://localhost:5050"),
                        help="Base URL of the HumanGuard API (default: http://localhost:5050)")
    parser.add_argument("--export-key", default=os.environ.get("EXPORT_API_KEY", "devkey"),
                        help="X-Export-Key header value for /api/export")
    parser.add_argument("--local-labels", default=None,
                        help="Path to a local extra labels CSV (sessionID,label) — skips API fetch")
    parser.add_argument("--push", action="store_true",
                        help="Push to ModelRegistry and promote if accuracy >= current champion")
    parser.add_argument("--min-accuracy", type=float, default=0.0,
                        help="Minimum accuracy improvement threshold for auto-promote (default: 0.0)")
    args = parser.parse_args()

    logger.info("=== HumanGuard Retrain Pipeline ===")

    # 1. Collect new labels
    if args.local_labels:
        logger.info("Loading extra labels from %s", args.local_labels)
        new_labels = load_local_extra_labels(args.local_labels)
    else:
        new_labels = fetch_export_labels(args.api_url, args.export_key)

    if new_labels.empty:
        logger.warning("No new labeled sessions found — retraining on existing data only")

    # 2. Merge with existing labels
    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_labels.csv", dir=str(LABELS_CSV.parent), delete=False
    ) as tmp_labels_f:
        tmp_labels_path = Path(tmp_labels_f.name)

    try:
        combined_labels = merge_labels(LABELS_CSV, new_labels)
        if len(combined_labels) < 4:
            logger.error("Insufficient labeled data (%d rows) — need at least 4 for train/test split",
                         len(combined_labels))
            sys.exit(1)
        combined_labels.to_csv(tmp_labels_path, index=False)
        logger.info("Wrote %d combined labels to %s", len(combined_labels), tmp_labels_path)

        # 3. Build features (re-uses existing signals.jsonl)
        with tempfile.NamedTemporaryFile(
            suffix="_features.csv", dir=str(FEATURES_CSV.parent), delete=False
        ) as tmp_feat_f:
            tmp_features_path = Path(tmp_feat_f.name)

        try:
            build_features(SIGNALS_JSONL, tmp_features_path)

            # 4. Train
            logger.info("Training XGBoost…")
            xgb_model, scaler, new_metrics, feature_names = train_xgboost(
                tmp_features_path, tmp_labels_path
            )

        finally:
            tmp_features_path.unlink(missing_ok=True)

    finally:
        tmp_labels_path.unlink(missing_ok=True)

    # 5. Load current champion metrics and compare
    champion_metrics = load_champion_metrics()
    print_comparison(new_metrics, champion_metrics)

    champ_acc = champion_metrics.get("accuracy", 0.0)
    new_acc   = new_metrics.get("accuracy", 0.0)
    should_promote = new_acc >= (champ_acc + args.min_accuracy)

    logger.info("New accuracy: %.4f  Champion accuracy: %.4f  Should promote: %s",
                new_acc, champ_acc, should_promote)

    # 6. Push to registry
    if args.push:
        model_bucket = os.environ.get("MODEL_BUCKET")
        if not model_bucket:
            logger.error("MODEL_BUCKET env var is required to push to registry")
            sys.exit(1)

        from model_registry import ModelRegistry
        registry = ModelRegistry(bucket=model_bucket)

        training_date = datetime.now(timezone.utc).isoformat()
        metadata = {
            "accuracy":         new_metrics.get("accuracy"),
            "precision":        new_metrics.get("precision"),
            "recall":           new_metrics.get("recall"),
            "f1":               new_metrics.get("f1"),
            "roc_auc":          new_metrics.get("roc_auc"),
            "training_date":    training_date,
            "training_samples": len(combined_labels) if "combined_labels" in dir() else None,
            "model_type":       "XGBoost",
        }

        threshold = 0.5
        threshold_path = TRAINED_DIR / "threshold.json"
        if threshold_path.exists():
            with open(threshold_path) as f:
                threshold = float(json.load(f).get("threshold", 0.5))

        logger.info("Pushing new model to registry (bucket=%s)…", model_bucket)
        version = registry.push(xgb_model, scaler, feature_names, threshold, metadata)
        logger.info("Pushed as %s", version)

        if should_promote:
            registry.promote(version)
            logger.info("Promoted %s to champion", version)
            print(f"\n✓ New model {version} promoted to champion")
            print(f"  Accuracy: {new_acc:.4f} vs previous champion: {champ_acc:.4f}")
        else:
            logger.info("New model NOT promoted (accuracy %.4f < threshold %.4f)",
                        new_acc, champ_acc + args.min_accuracy)
            print(f"\n— Model pushed as {version} but NOT promoted")
            print(f"  Accuracy {new_acc:.4f} did not exceed champion {champ_acc:.4f}")
    else:
        print("(Dry run — use --push to push to registry)")


if __name__ == "__main__":
    main()
