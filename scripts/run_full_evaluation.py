"""
Full evaluation pipeline for HumanGuard.

Produces a three-tier performance summary:

  1. 5-fold stratified CV score (mean ± std)    → honest, bias-corrected estimate
  2. Easy test set score (80/20 clean split)     → upper-bound / sanity check
  3. Hard test set score (adversarial bots only) → realistic production lower-bound

Output: backend/data/hard_test_results.txt
"""

import json
import sys
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).parent.parent
TRAINED_DIR = ROOT / "models/trained"
OUTPUT_PATH = ROOT / "backend/data/hard_test_results.txt"

BATCH_CSV     = ROOT / "backend/data/processed/training_data_batches.csv"
LABELS_CSV    = ROOT / "backend/data/raw/labels.csv"
HARD_SIG_JSONL = ROOT / "backend/data/raw/hard_test_signals.jsonl"
HARD_LAB_CSV  = ROOT / "backend/data/raw/hard_test_labels.csv"

sys.path.insert(0, str(ROOT))
from backend.features.dataset_builder import DatasetBuilder
from backend.features.feature_extractor import FeatureExtractor
from models.dataset import ModelDataset


# ── Load champion artifacts ───────────────────────────────────────────────

def load_champion():
    with open(TRAINED_DIR / "model_comparison.json") as f:
        comp = json.load(f)
    name   = comp["winner"]
    model  = joblib.load(TRAINED_DIR / f"{name}.pkl")
    scaler = joblib.load(TRAINED_DIR / "scaler.pkl")
    with open(TRAINED_DIR / "feature_names.json") as f:
        feat_names = json.load(f)

    # Load optional session blender
    session_blender  = None
    bl_features      = None
    blender_path     = TRAINED_DIR / "session_blender.pkl"
    bl_feats_path    = TRAINED_DIR / "session_blender_features.json"
    if blender_path.exists() and bl_feats_path.exists():
        try:
            session_blender = joblib.load(blender_path)
            with open(bl_feats_path) as f:
                bl_features = json.load(f)
        except Exception:
            pass

    return model, scaler, feat_names, name, comp, session_blender, bl_features


# ── Easy test set ─────────────────────────────────────────────────────────

def easy_test_metrics(model, scaler, feat_names) -> dict:
    """
    Reproduce the same 80/20 stratified split used during training,
    then evaluate the champion model on the clean test subset.
    """
    dataset = ModelDataset(str(BATCH_CSV), str(LABELS_CSV))
    _, xTest, _, yTest, featureNames, _ = dataset.prepare(add_noise=False)

    # Re-apply the SAVED scaler (not the freshly refitted one)
    # Because prepare() refits internally, we need to re-scale using the saved scaler.
    # Approach: load raw data and re-split with the same seed, then apply saved scaler.
    ds2 = ModelDataset(str(BATCH_CSV), str(LABELS_CSV))
    X_all, y_all, cols = ds2.get_raw_dataset()
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_te_scaled = scaler.transform(X_te[feat_names])

    y_pred = model.predict(X_te_scaled)
    y_prob = model.predict_proba(X_te_scaled)[:, 1]
    cm = confusion_matrix(y_te, y_pred)
    return {
        "n_samples":  len(y_te),
        "accuracy":   float(accuracy_score(y_te, y_pred)),
        "precision":  float(precision_score(y_te, y_pred, zero_division=0)),
        "recall":     float(recall_score(y_te, y_pred, zero_division=0)),
        "f1":         float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc":    float(roc_auc_score(y_te, y_prob)),
        "cm":         cm.tolist(),
    }


# ── Hard test set ─────────────────────────────────────────────────────────

def build_hard_test_features(feat_names: list) -> tuple:
    """
    Extract features from hard_test_signals.jsonl, align to feature_names,
    return (X_scaled, y_all_bot).
    """
    if not HARD_SIG_JSONL.exists():
        raise FileNotFoundError(f"Hard test signals not found: {HARD_SIG_JSONL}")
    if not HARD_LAB_CSV.exists():
        raise FileNotFoundError(f"Hard test labels not found: {HARD_LAB_CSV}")

    # Extract features to a temp CSV using the DatasetBuilder
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name

    builder = DatasetBuilder(str(HARD_SIG_JSONL))
    feat_df = builder.buildBatchLevelDataset(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)

    hard_labels = pd.read_csv(HARD_LAB_CSV)
    # Drop the placeholder label column that DatasetBuilder injects
    if "label" in feat_df.columns:
        feat_df = feat_df.drop(columns=["label"])
    # Merge on sessionID
    feat_df = feat_df.merge(hard_labels, on="sessionID", how="inner")

    # Align to trained feature set
    missing = [f for f in feat_names if f not in feat_df.columns]
    if missing:
        raise ValueError(f"Missing features in hard test: {missing}")

    X = feat_df[feat_names].fillna(0).values
    y = feat_df["label"].map({"human": 0, "bot": 1}).fillna(1).values
    return X, y, feat_df


def hard_test_metrics(model, scaler, feat_names: list) -> dict:
    X_raw, y, feat_df = build_hard_test_features(feat_names)
    X_scaled = scaler.transform(X_raw)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    # AUC is undefined when only one class is present (all bots)
    n_classes = len(np.unique(y))
    overall_auc = (float(roc_auc_score(y, y_prob))
                   if n_classes > 1 else float("nan"))

    # Per-pattern breakdown — extract pattern name without date suffix
    feat_df = feat_df.copy()
    feat_df["pred"] = y_pred
    feat_df["prob_bot"] = y_prob
    feat_df["pattern"] = feat_df["sessionID"].str.extract(
        r"hardbot_([a-z_]+)_\d{8}_")
    per_pattern = {}
    for pat in feat_df["pattern"].dropna().unique():
        mask     = feat_df["pattern"] == pat
        sub_y    = y[mask.values]
        sub_pred = y_pred[mask.values]
        per_pattern[pat] = {
            "n":             int(mask.sum()),
            "detection_rate": float(recall_score(sub_y, sub_pred, zero_division=0)),
            "f1":            float(f1_score(sub_y, sub_pred, zero_division=0)),
        }

    return {
        "n_samples":    len(y),
        "accuracy":     float(accuracy_score(y, y_pred)),
        "precision":    float(precision_score(y, y_pred, zero_division=0)),
        "recall":       float(recall_score(y, y_pred, zero_division=0)),
        "f1":           float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc":      overall_auc,
        "cm":           cm.tolist(),
        "per_pattern":  per_pattern,
    }


# ── Session-level hard test (adaptive bot detection) ─────────────────────

def hard_test_session_metrics(model, scaler, feat_names: list,
                               session_blender, bl_features) -> dict:
    """
    Evaluate the hard test set at SESSION level.

    Each session is scored as a unit: per-batch probabilities are linearly
    weighted (later batches carry more weight), then the session blender
    (if available) blends in temporal drift features.

    This properly measures adaptive_bot detection, where batch-level scoring
    counts 50 % of batches as false negatives (the human-phase batches),
    but session-level scoring correctly classifies the whole session.
    """
    import re
    from collections import defaultdict

    if not HARD_SIG_JSONL.exists():
        return {}

    # Load raw signals grouped by session
    sessions: dict = defaultdict(list)
    with open(HARD_SIG_JSONL) as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                sid = rec.get("sessionID")
                sig = rec.get("signals") or {}
                if sid:
                    sessions[sid].append(sig)
            except Exception:
                continue

    hard_labels = pd.read_csv(HARD_LAB_CSV)
    label_map   = dict(zip(hard_labels["sessionID"], hard_labels["label"]))
    extractor   = FeatureExtractor()

    session_rows = []
    for sid, sig_list in sessions.items():
        y_true = 1 if label_map.get(sid) == "bot" else 0

        # Per-batch scores
        batch_probs = []
        for sig in sig_list:
            feats = extractor.extractBatchFeatures(sig)
            x_row = np.array([[float(feats.get(n, 0.0)) for n in feat_names]])
            x_sc  = scaler.transform(x_row)
            batch_probs.append(float(model.predict_proba(x_sc)[0, 1]))

        n       = len(batch_probs)
        weights = np.arange(1, n + 1, dtype=float)
        avg_prob = float(np.average(batch_probs, weights=weights))

        # Temporal blending for sessions with ≥ 10 batches
        if n >= 10 and session_blender is not None and bl_features:
            drift       = extractor.temporal_drift_score(sig_list)
            delta_ms    = extractor.early_late_timing_delta(sig_list)
            consistency = extractor.behavior_consistency_score(sig_list)
            feat_map    = {
                "avg_batch_prob":       avg_prob,
                "temporal_drift":       drift,
                "early_late_delta_ms":  delta_ms,
                "behavior_consistency": consistency,
            }
            x_meta   = np.array([[feat_map.get(f, 0.0) for f in bl_features]])
            final_prob = float(np.clip(session_blender.predict_proba(x_meta)[0, 1], 0.0, 1.0))
        elif n >= 10:
            # Fixed-rule fallback
            drift       = extractor.temporal_drift_score(sig_list)
            delta_ms    = extractor.early_late_timing_delta(sig_list)
            consistency = extractor.behavior_consistency_score(sig_list)
            norm_delta  = float(min(delta_ms / 100.0, 1.0))
            temp_susp   = float(np.clip(
                2.0 * drift + 0.4 * norm_delta + 0.3 * (1.0 - consistency), 0.0, 1.0
            ))
            final_prob = float(np.clip(0.65 * avg_prob + 0.35 * temp_susp, 0.0, 1.0))
        else:
            final_prob = avg_prob

        m       = re.search(r"hardbot_([a-z_]+)_\d{8}_", sid)
        pattern = m.group(1) if m else "unknown"
        session_rows.append({
            "sessionID": sid,
            "pattern":   pattern,
            "y_true":    y_true,
            "final_prob": final_prob,
            "y_pred":    1 if final_prob >= 0.5 else 0,
        })

    df = pd.DataFrame(session_rows)
    if df.empty:
        return {}

    y_t = df["y_true"].values
    y_p = df["y_pred"].values

    per_pattern = {}
    for pat in df["pattern"].unique():
        mask = df["pattern"] == pat
        sub  = df[mask]
        per_pattern[pat] = {
            "n":             int(mask.sum()),
            "detection_rate": float(recall_score(sub["y_true"], sub["y_pred"], zero_division=0)),
            "f1":            float(f1_score(sub["y_true"], sub["y_pred"], zero_division=0)),
        }

    return {
        "n_sessions":  len(df),
        "accuracy":    float(accuracy_score(y_t, y_p)),
        "precision":   float(precision_score(y_t, y_p, zero_division=0)),
        "recall":      float(recall_score(y_t, y_p, zero_division=0)),
        "f1":          float(f1_score(y_t, y_p, zero_division=0)),
        "per_pattern": per_pattern,
    }


# ── Cross-validation ──────────────────────────────────────────────────────

def cv_metrics(feat_names: list) -> dict:
    dataset = ModelDataset(str(BATCH_CSV), str(LABELS_CSV))
    X_all, y_all, _ = dataset.get_raw_dataset()
    X_feat = X_all[feat_names]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1", "roc_auc"]

    estimators = {
        "RandomForest": RandomForestClassifier(
            n_estimators=50, max_depth=4, min_samples_leaf=5,
            random_state=42, n_jobs=-1),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="logloss"),
    }

    results = {}
    for name, clf in estimators.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv = cross_validate(pipe, X_feat, y_all, cv=skf, scoring=scoring, n_jobs=-1)
        results[name] = {
            "accuracy_mean": float(np.mean(cv["test_accuracy"])),
            "accuracy_std":  float(np.std(cv["test_accuracy"])),
            "f1_mean":       float(np.mean(cv["test_f1"])),
            "f1_std":        float(np.std(cv["test_f1"])),
            "roc_auc_mean":  float(np.mean(cv["test_roc_auc"])),
            "roc_auc_std":   float(np.std(cv["test_roc_auc"])),
        }
    return results


# ── Reporting ─────────────────────────────────────────────────────────────

def _cm_str(cm) -> str:
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    return (f"  TN={tn}  FP={fp}\n"
            f"  FN={fn}  TP={tp}")


def main():
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("  HUMANGUARD — FULL EVALUATION REPORT")
    log("  Generated: 2026-03-30")
    log("=" * 70)

    # Load champion
    model, scaler, feat_names, champion_name, comp, session_blender, bl_features = load_champion()
    blender_status = "loaded" if session_blender is not None else "not found"
    log(f"\n  Champion model  : {champion_name}")
    log(f"  Feature count   : {len(feat_names)}")
    log(f"  Session blender : {blender_status}")

    # ── CV ──
    log("\n── 1. 5-FOLD STRATIFIED CROSS-VALIDATION ──────────────────────────")
    log("  (Pipeline-internal scaling; no data leakage; most honest estimate)")
    log()
    log(f"  {'Model':<22} {'Accuracy':>18} {'F1':>18} {'AUC-ROC':>18}")
    log("  " + "-" * 76)
    cv = cv_metrics(feat_names)
    for name, v in cv.items():
        log(f"  {name:<22} "
            f"{v['accuracy_mean']:.4f} ± {v['accuracy_std']:.4f}   "
            f"{v['f1_mean']:.4f} ± {v['f1_std']:.4f}   "
            f"{v['roc_auc_mean']:.4f} ± {v['roc_auc_std']:.4f}")
    champ_cv = cv.get(champion_name, {})
    log()

    # ── Easy test ──
    log("── 2. EASY TEST SET (80/20 stratified split, clean labels) ────────")
    log("  (Upper bound — same distribution as training data)")
    log()
    easy = easy_test_metrics(model, scaler, feat_names)
    log(f"  Samples   : {easy['n_samples']}")
    log(f"  Accuracy  : {easy['accuracy']:.4f}")
    log(f"  Precision : {easy['precision']:.4f}")
    log(f"  Recall    : {easy['recall']:.4f}")
    log(f"  F1        : {easy['f1']:.4f}")
    log(f"  AUC-ROC   : {easy['roc_auc']:.4f}")
    log(f"  Confusion matrix:")
    log(_cm_str(easy['cm']))
    log()

    # ── Hard test — batch level ──
    log("── 3a. HARD TEST SET — BATCH-LEVEL SCORING ────────────────────────")
    log("  (Each batch scored independently — adaptive_bot under-detected here)")
    log()
    hard = hard_test_metrics(model, scaler, feat_names)
    log(f"  Samples   : {hard['n_samples']} batches")
    log(f"  Accuracy  : {hard['accuracy']:.4f}")
    log(f"  Precision : {hard['precision']:.4f}")
    log(f"  Recall    : {hard['recall']:.4f}  ← fraction of adversarial bots caught")
    log(f"  F1        : {hard['f1']:.4f}")
    log(f"  Confusion matrix:")
    log(_cm_str(hard['cm']))
    log()
    log(f"  {'Pattern':<22} {'N':>6} {'DetRate':>9} {'F1':>8}")
    log("  " + "-" * 47)
    batch_per_pattern = hard["per_pattern"]
    for pat, v in batch_per_pattern.items():
        log(f"  {pat:<22} {v['n']:>6} {v['detection_rate']:>9.4f} {v['f1']:>8.4f}")
    log()

    # ── Hard test — session level ──
    log("── 3b. HARD TEST SET — SESSION-LEVEL SCORING (batch + temporal) ───")
    log("  (Sessions scored as a unit; adaptive_bot properly evaluated here)")
    log()
    sess = hard_test_session_metrics(model, scaler, feat_names,
                                     session_blender, bl_features)
    if sess:
        log(f"  Sessions  : {sess['n_sessions']}")
        log(f"  Accuracy  : {sess['accuracy']:.4f}")
        log(f"  Precision : {sess['precision']:.4f}")
        log(f"  Recall    : {sess['recall']:.4f}  ← fraction of adversarial sessions caught")
        log(f"  F1        : {sess['f1']:.4f}")
        log()
        log(f"  {'Pattern':<22} {'N':>6} {'DetRate':>9} {'F1':>8}")
        log("  " + "-" * 47)
        sess_per_pattern = sess["per_pattern"]
        for pat, v in sess_per_pattern.items():
            log(f"  {pat:<22} {v['n']:>6} {v['detection_rate']:>9.4f} {v['f1']:>8.4f}")
    else:
        log("  (No session-level data available)")
        sess_per_pattern = {}
    log()

    # ── Before / after comparison ──
    log("── 3c. BEFORE / AFTER COMPARISON (batch → session scoring) ─────────")
    log()
    log(f"  {'Pattern':<22} {'Batch DetRate':>14} {'Session DetRate':>16} {'Delta':>8}")
    log("  " + "-" * 62)
    all_patterns = sorted(set(list(batch_per_pattern.keys()) + list(sess_per_pattern.keys())))
    for pat in all_patterns:
        b_rate = batch_per_pattern.get(pat, {}).get("detection_rate", float("nan"))
        s_rate = sess_per_pattern.get(pat, {}).get("detection_rate", float("nan"))
        delta  = s_rate - b_rate if not (np.isnan(b_rate) or np.isnan(s_rate)) else float("nan")
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "   N/A"
        log(f"  {pat:<22} {b_rate:>14.4f} {s_rate:>16.4f} {delta_str:>8}")
    log()

    # ── Summary table ──
    log("── 4. SUMMARY TABLE ────────────────────────────────────────────────")
    log()
    cv_auc  = champ_cv.get("roc_auc_mean", float("nan"))
    cv_std  = champ_cv.get("roc_auc_std",  float("nan"))
    cv_f1   = champ_cv.get("f1_mean",      float("nan"))
    cv_f1s  = champ_cv.get("f1_std",       float("nan"))
    hard_auc_str = (f"{hard['roc_auc']:.4f}"
                   if not np.isnan(hard['roc_auc']) else "N/A(1-class)")
    sess_det = sess.get("recall", float("nan")) if sess else float("nan")
    sess_f1  = sess.get("f1",     float("nan")) if sess else float("nan")

    log(f"  {'Tier':<35} {'AUC-ROC':>14} {'F1':>10} {'DetRate':>9}")
    log("  " + "-" * 70)
    log(f"  {'CV (5-fold, honest)':<35} "
        f"{cv_auc:.4f}±{cv_std:.4f}  {cv_f1:.4f}±{cv_f1s:.4f}        —")
    log(f"  {'Easy test (upper bound)':<35} "
        f"{'':>5}{easy['roc_auc']:.4f}{'':>9}  {easy['f1']:.4f}        —")
    log(f"  {'Hard test batch (lower bound)':<35} "
        f"{'':>5}{hard_auc_str:<12}  {hard['f1']:.4f}   {hard['recall']:.4f}")
    if sess:
        log(f"  {'Hard test session (w/ temporal)':<35} "
            f"{'':>5}{'N/A(1-class)':<12}  {sess_f1:.4f}   {sess_det:.4f}")
    log()
    log("  Interpretation:")
    log("  ─ CV AUC is the headline metric (no data leakage).")
    log("  ─ Batch hard-test DetRate: each batch scored independently vs session label.")
    log("    adaptive_bot reads low here because 50% of its batches look human.")
    log("  ─ Session hard-test DetRate: the whole session is scored as a unit,")
    log("    correctly measuring adaptive_bot detection via temporal drift blending.")
    det_rate = sess_det if not np.isnan(sess_det) else hard["recall"]
    f1_gap   = easy["f1"] - (sess_f1 if not np.isnan(sess_f1) else hard["f1"])
    log(f"\n  Easy → Hard F1 gap (session)  : {f1_gap:+.4f}")
    log(f"  Session detection rate        : {det_rate:.4f}  ({det_rate*100:.1f}% of adversarial bots caught)")
    if det_rate < 0.75:
        log("  △  Below 75% target. Improve temporal features or add more adversarial training data.")
    elif det_rate < 0.90:
        log("  ✓  Good detection. Some adaptive bots still evade — monitor in production.")
    else:
        log("  ✓  Strong detection even on adaptive adversarial patterns.")

    log()
    log("=" * 70)
    log("  END OF REPORT")
    log("=" * 70)

    # Write file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\n✓ Report saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
