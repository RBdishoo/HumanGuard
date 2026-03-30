"""
Train a session-level meta-model (LogisticRegression) that blends
batch-level bot probability with temporal drift features.

The blender catches adaptive bots that behave human-like early in a session
and bot-like later — patterns the batch model cannot detect on its own.

Features per session:
  avg_batch_prob         — linearly-weighted average of per-batch bot probabilities
  temporal_drift         — normalised drift between first/second-half feature vectors
  early_late_delta_ms    — change in mean inter-key delay (first 30% vs last 30%)
  behavior_consistency   — cosine similarity of first/second-half feature vectors

Synthetic adaptive-bot rows are added at training time so the blender learns
the pattern (high drift + delta at moderate batch probability → bot) even
though no real adaptive-bot sessions are in the training signals.

Output:
  models/trained/session_blender.pkl
  models/trained/session_blender_features.json
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).parent.parent
SIGNALS_PATH = ROOT / "backend/data/raw/signals.jsonl"
LABELS_PATH  = ROOT / "backend/data/raw/labels.csv"
TRAINED_DIR  = ROOT / "models/trained"
MIN_BATCHES  = 10

sys.path.insert(0, str(ROOT))
from backend.features.feature_extractor import FeatureExtractor
from backend.features.data_loader import SignalDataLoader

FEATURE_COLS = ["avg_batch_prob", "temporal_drift", "early_late_delta_ms", "behavior_consistency"]


# ── Load champion artifacts ───────────────────────────────────────────────

def _load_champion():
    with open(TRAINED_DIR / "model_comparison.json") as f:
        comp = json.load(f)
    winner = comp["winner"]
    model  = joblib.load(TRAINED_DIR / f"{winner}.pkl")
    scaler = joblib.load(TRAINED_DIR / "scaler.pkl")
    with open(TRAINED_DIR / "feature_names.json") as f:
        feat_names = json.load(f)
    return model, scaler, feat_names, winner


def _score_batch(model, scaler, feat_names, extractor, signals_dict):
    feats = extractor.extractBatchFeatures(signals_dict)
    x_row = np.array([[float(feats.get(n, 0.0)) for n in feat_names]])
    x_sc  = scaler.transform(x_row)
    return float(model.predict_proba(x_sc)[0, 1])


# ── Build session-level dataset from training signals ────────────────────

def _build_session_dataset(model, scaler, feat_names, extractor):
    if not SIGNALS_PATH.exists():
        return pd.DataFrame()

    loader     = SignalDataLoader(str(SIGNALS_PATH))
    signals_df = loader.loadSignals()
    if signals_df.empty:
        return pd.DataFrame()

    labels_df = pd.read_csv(LABELS_PATH)
    label_map = dict(zip(labels_df["sessionID"], labels_df["label"]))

    rows = []
    for sid in signals_df["sessionID"].unique():
        session_df = signals_df[signals_df["sessionID"] == sid]
        if len(session_df) < MIN_BATCHES:
            continue
        label_str = label_map.get(sid)
        if label_str is None:
            continue

        sig_list = [
            {"mouseMoves": r["mouseMoves"], "clicks": r["clicks"], "keys": r["keys"]}
            for _, r in session_df.iterrows()
        ]

        # Linearly-weighted average batch probability
        n           = len(sig_list)
        weights     = np.arange(1, n + 1, dtype=float)
        batch_probs = [_score_batch(model, scaler, feat_names, extractor, s)
                       for s in sig_list]
        avg_prob    = float(np.average(batch_probs, weights=weights))

        # Temporal features
        drift       = extractor.temporal_drift_score(sig_list)
        delta_ms    = extractor.early_late_timing_delta(sig_list)
        consistency = extractor.behavior_consistency_score(sig_list)

        rows.append({
            "sessionID":            sid,
            "avg_batch_prob":       avg_prob,
            "temporal_drift":       drift,
            "early_late_delta_ms":  delta_ms,
            "behavior_consistency": consistency,
            "label":                1 if label_str == "bot" else 0,
        })

    return pd.DataFrame(rows)


# ── Synthetic adaptive-bot rows ───────────────────────────────────────────
# These teach the blender that moderate avg_prob + high drift → bot,
# a combination absent from the training signals (adaptive bots are hard-test only).

def _synthetic_rows() -> pd.DataFrame:
    """
    Synthetic feature rows covering adaptive-bot and human-like distributions.

    Adaptive bots (label=1): moderate avg_prob, high drift, high delta, low consistency.
    Humans       (label=0): low avg_prob, low drift, low delta, high consistency.

    These ensure the blender can always be cross-validated and learns the
    adaptive-bot pattern even when the training signals file is unavailable.
    """
    rng   = np.random.RandomState(999)
    n_bot = 25
    n_hum = 15

    bots = pd.DataFrame({
        "sessionID":            [f"synthetic_adaptive_{i}" for i in range(n_bot)],
        "avg_batch_prob":       rng.uniform(0.38, 0.65, n_bot),
        "temporal_drift":       rng.uniform(0.20, 0.45, n_bot),
        "early_late_delta_ms":  rng.uniform(55.0, 110.0, n_bot),
        "behavior_consistency": rng.uniform(0.45, 0.72, n_bot),
        "label":                1,
    })
    humans = pd.DataFrame({
        "sessionID":            [f"synthetic_human_{i}" for i in range(n_hum)],
        "avg_batch_prob":       rng.uniform(0.02, 0.20, n_hum),
        "temporal_drift":       rng.uniform(0.01, 0.09, n_hum),
        "early_late_delta_ms":  rng.uniform(5.0,  30.0, n_hum),
        "behavior_consistency": rng.uniform(0.88, 0.99, n_hum),
        "label":                0,
    })
    return pd.concat([bots, humans], ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Training session-level meta-model (session blender)")
    print("=" * 60)

    model, scaler, feat_names, winner = _load_champion()
    extractor = FeatureExtractor()
    print(f"\n  Champion model : {winner}")

    print(f"  Loading sessions with ≥{MIN_BATCHES} batches from {SIGNALS_PATH.name}...")
    session_df = _build_session_dataset(model, scaler, feat_names, extractor)

    if session_df.empty:
        print("  WARNING: No qualifying sessions found. Fitting on synthetic data only.")
        session_df = pd.DataFrame(columns=["sessionID"] + FEATURE_COLS + ["label"])

    print(f"  Real sessions  : {len(session_df)}"
          f"  (bots={int(session_df['label'].sum())}, "
          f"humans={int((session_df['label']==0).sum())})")

    synthetic_df = _synthetic_rows()
    combined_df  = pd.concat([session_df, synthetic_df], ignore_index=True)
    print(f"  After synthetic augmentation: {len(combined_df)} rows "
          f"(+{len(synthetic_df)} synthetic examples)")

    X = combined_df[FEATURE_COLS].values
    y = combined_df["label"].values

    # 5-fold CV to estimate generalisation
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_validate(pipe, X, y, cv=skf, scoring=["f1", "roc_auc"])
    print(f"\n  5-fold CV (blender):")
    print(f"    F1    : {np.mean(cv['test_f1']):.4f} ± {np.std(cv['test_f1']):.4f}")
    print(f"    AUC   : {np.mean(cv['test_roc_auc']):.4f} ± {np.std(cv['test_roc_auc']):.4f}")

    # Fit final blender on all data
    pipe.fit(X, y)

    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, TRAINED_DIR / "session_blender.pkl")
    with open(TRAINED_DIR / "session_blender_features.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    print(f"\n  Blender saved  → {TRAINED_DIR}/session_blender.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
