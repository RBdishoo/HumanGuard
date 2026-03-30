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
    return model, scaler, feat_names, name, comp


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
    model, scaler, feat_names, champion_name, comp = load_champion()
    log(f"\n  Champion model : {champion_name}")
    log(f"  Feature count  : {len(feat_names)}")

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

    # ── Hard test ──
    log("── 3. HARD TEST SET (adversarial bots, NOT in training) ───────────")
    log("  (Realistic production lower-bound)")
    log()
    hard = hard_test_metrics(model, scaler, feat_names)
    log(f"  Samples   : {hard['n_samples']}")
    log(f"  Accuracy  : {hard['accuracy']:.4f}")
    log(f"  Precision : {hard['precision']:.4f}")
    log(f"  Recall    : {hard['recall']:.4f}  ← fraction of adversarial bots caught")
    log(f"  F1        : {hard['f1']:.4f}")
    log(f"  AUC-ROC   : {hard['roc_auc']:.4f}")
    log(f"  Confusion matrix:")
    log(_cm_str(hard['cm']))
    log()
    log("  Per-pattern breakdown (detection_rate = recall for bots):")
    log(f"  {'Pattern':<22} {'N':>6} {'DetRate':>9} {'F1':>8}")
    log("  " + "-" * 47)
    for pat, v in hard["per_pattern"].items():
        log(f"  {pat:<22} {v['n']:>6} {v['detection_rate']:>9.4f} {v['f1']:>8.4f}")
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
    log(f"  {'Tier':<30} {'AUC-ROC':>14} {'F1':>10} {'DetRate':>9}")
    log("  " + "-" * 65)
    log(f"  {'CV (5-fold, honest)':<30} "
        f"{cv_auc:.4f}±{cv_std:.4f}  {cv_f1:.4f}±{cv_f1s:.4f}        —")
    log(f"  {'Easy test (upper bound)':<30} "
        f"{'':>5}{easy['roc_auc']:.4f}{'':>9}  {easy['f1']:.4f}        —")
    log(f"  {'Hard test (prod. lower bound)':<30} "
        f"{'':>5}{hard_auc_str:<12}  {hard['f1']:.4f}   {hard['recall']:.4f}")
    log()
    log("  Interpretation:")
    log("  ─ CV AUC is the headline metric — computed on clean folds without")
    log("    label leakage. Put this number on your resume.")
    log("  ─ Easy test AUC is the upper bound. Simulated data is separable;")
    log("    real-world signals will be harder.")
    log("  ─ Hard test AUC is the realistic production estimate. These bots")
    log("    deliberately mimic human behaviour. A gap from Easy → Hard")
    log("    shows the model's brittleness to adversarial evasion.")
    # Use F1 gap since hard-test AUC is undefined (all bots, single class)
    f1_gap = easy["f1"] - hard["f1"]
    det_rate = hard["recall"]
    log(f"\n  Easy → Hard F1 gap         : {f1_gap:+.4f}")
    log(f"  Hard test detection rate   : {det_rate:.4f}  ({det_rate*100:.1f}% of adversarial bots caught)")
    if det_rate < 0.50:
        log("  ⚠  < 50% detection on adversarial set — significant blind spot.")
        log("     Add hard-test patterns to training data.")
    elif det_rate < 0.80:
        log("  △  Moderate detection. Adversarial bots evade ~{:.0f}% of the time.".format(
            (1 - det_rate) * 100))
        log("     Consider adversarial training or session-level aggregation.")
    else:
        log("  ✓  Strong detection even on adversarial patterns.")

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
