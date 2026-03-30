"""
Model audit script: training data composition, model performance, feature importance, weaknesses.
Outputs a summary to backend/data/model_audit.txt
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FEATURES_CSV = ROOT / "backend/data/processed/training_data_batches.csv"
LABELS_CSV   = ROOT / "backend/data/raw/labels.csv"
TRAINED_DIR  = ROOT / "models/trained"
OUTPUT_PATH  = ROOT / "backend/data/model_audit.txt"

NOISE_RATE   = 0.10
TEST_SIZE    = 0.20
RANDOM_STATE = 42

NON_FEATURE_COLS = {"sessionID", "timestamp", "timestampRelativeMs", "batchCount", "label"}

lines = []
def log(s=""):
    print(s)
    lines.append(s)

# ── 1. Load data ─────────────────────────────────────────────────────────────
featuresDF = pd.read_csv(FEATURES_CSV)
labelsDF   = pd.read_csv(LABELS_CSV)

if "label" in featuresDF.columns:
    featuresDF = featuresDF.drop(columns=["label"])

merged = (
    featuresDF.set_index("sessionID")
    .join(labelsDF.set_index("sessionID")[["label"]], how="inner")
    .reset_index()
)

feature_cols = [c for c in merged.columns if c not in NON_FEATURE_COLS]
X = merged[feature_cols].fillna(0)
y = merged["label"].map({"human": 0, "bot": 1})

# ── 2. Training data composition ─────────────────────────────────────────────
log("=" * 65)
log("  HUMANGUARD MODEL AUDIT")
log("  Generated: 2026-03-30")
log("=" * 65)
log()

log("── 1. TRAINING DATA COMPOSITION ────────────────────────────────")
total_samples = len(merged)
bot_count   = int((y == 1).sum())
human_count = int((y == 0).sum())
bot_pct     = bot_count / total_samples * 100
human_pct   = human_count / total_samples * 100

log(f"  Total samples (batch rows) : {total_samples}")
log(f"  Human                      : {human_count}  ({human_pct:.1f}%)")
log(f"  Bot                        : {bot_count}  ({bot_pct:.1f}%)")
log()

# Session-level breakdown
total_sessions = labelsDF.shape[0]
bot_sessions   = (labelsDF["label"] == "bot").sum()
human_sessions = (labelsDF["label"] == "human").sum()
log(f"  Unique labelled sessions   : {total_sessions}")
log(f"    bot_sim_*  (simulated)   : {bot_sessions}")
log(f"    human_sim_* (simulated)  : {human_sessions}")
log(f"  Real / demo sessions       : 0  (all data is simulated)")
log()

# Data sources inferred from session IDs
bot_ids   = labelsDF[labelsDF["label"] == "bot"]["sessionID"].tolist()
human_ids = labelsDF[labelsDF["label"] == "human"]["sessionID"].tolist()
log(f"  Bot session IDs   : {bot_ids}")
log(f"  Human session IDs : {human_ids}")
log()

# Feature stats
log(f"  Features ({len(feature_cols)} total) — descriptive statistics:")
stats = X.describe().T[["mean", "std", "min", "max"]]
stats.columns = ["mean", "std", "min", "max"]
header = f"    {'feature':<35} {'mean':>10} {'std':>10} {'min':>10} {'max':>10}"
log(header)
log("    " + "-" * 75)
for feat, row in stats.iterrows():
    log(f"    {feat:<35} {row['mean']:>10.4f} {row['std']:>10.4f} {row['min']:>10.4f} {row['max']:>10.4f}")
log()

# ── 3. Train / test split (same procedure as ModelDataset) ───────────────────
xTrain, xTest, yTrain, yTest = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Apply label noise (mirrors ModelDataset.prepare)
rng = np.random.RandomState(RANDOM_STATE)
train_flip = rng.random(len(yTrain)) < NOISE_RATE
yTrain = yTrain.copy(); yTrain.iloc[train_flip] = 1 - yTrain.iloc[train_flip]
test_flip = rng.random(len(yTest)) < NOISE_RATE
yTest = yTest.copy(); yTest.iloc[test_flip] = 1 - yTest.iloc[test_flip]

scaler = StandardScaler()
xTrainScaled = scaler.fit_transform(xTrain)
xTestScaled  = scaler.transform(xTest)

# ── 4. Champion model performance ────────────────────────────────────────────
log("── 2. MODEL PERFORMANCE (held-out test set) ─────────────────────")

comparison_path = TRAINED_DIR / "model_comparison.json"
with open(comparison_path) as f:
    comparison = json.load(f)

champion_name = comparison["winner"]
log(f"  Champion model             : {champion_name}")
log(f"  Selection metric           : {comparison['selection_metric']}")
log()

# Try champion first, fall back to any loadable model
def try_load_pkl(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

champion_model = try_load_pkl(TRAINED_DIR / f"{champion_name}.pkl")
loaded_name = champion_name
if champion_model is None:
    log(f"  ⚠  {champion_name}.pkl is corrupted — trying other models...")
    for alt in ["XGBoost", "RandomForest", "LogisticRegression"]:
        m = try_load_pkl(TRAINED_DIR / f"{alt}.pkl")
        if m is not None:
            champion_model = m
            loaded_name = alt
            log(f"  Using {alt} for live evaluation.")
            break
if champion_model is None:
    raise RuntimeError("No loadable model found in models/trained/")

log(f"  Evaluated model            : {loaded_name}")
yPred      = champion_model.predict(xTestScaled)
yPredProba = champion_model.predict_proba(xTestScaled)[:, 1]

acc  = accuracy_score(yTest, yPred)
prec = precision_score(yTest, yPred, zero_division=0)
rec  = recall_score(yTest, yPred, zero_division=0)
f1   = f1_score(yTest, yPred, zero_division=0)
auc  = roc_auc_score(yTest, yPredProba)
cm   = confusion_matrix(yTest, yPred)

log(f"  Accuracy                   : {acc:.4f}")
log(f"  Precision                  : {prec:.4f}")
log(f"  Recall                     : {rec:.4f}")
log(f"  F1 Score                   : {f1:.4f}")
log(f"  AUC-ROC                    : {auc:.4f}")
log()
log("  Confusion Matrix (rows=Actual, cols=Predicted):")
log("                  Pred Human   Pred Bot")
log(f"  Actual Human :     {cm[0][0]:>5}       {cm[0][1]:>5}")
log(f"  Actual Bot   :     {cm[1][0]:>5}       {cm[1][1]:>5}")
log()
log("  All-model comparison (from registry):")
log(f"    {'Model':<20} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1':>9} {'AUC':>9}")
log("    " + "-" * 65)
for m in comparison["models"]:
    log(f"    {m['model']:<20} {m['accuracy']:>9.4f} {m['precision']:>9.4f} {m['recall']:>9.4f} {m['f1']:>9.4f} {m['roc_auc']:>9.4f}")
log()

# ── 5. Feature importance — XGBoost gain ─────────────────────────────────────
log("── 3. FEATURE IMPORTANCE (XGBoost gain, top 10) ─────────────────")

xgb_fi_path = TRAINED_DIR / "xgboost_feature_importance.csv"
fi_df = pd.read_csv(xgb_fi_path).sort_values("importance", ascending=False)
top10 = fi_df.head(10)

log(f"  {'Rank':<5} {'Feature':<35} {'Gain':>10}")
log("  " + "-" * 52)
for rank, (_, row) in enumerate(top10.iterrows(), 1):
    log(f"  {rank:<5} {row['feature']:<35} {row['importance']:>10.6f}")
log()

# ── 6. Weaknesses ────────────────────────────────────────────────────────────
log("── 4. WEAKNESSES & RECOMMENDATIONS ─────────────────────────────")

# 6a. Class imbalance
imbalance_ratio = max(bot_count, human_count) / min(bot_count, human_count)
log(f"  Class imbalance (batch-level):")
log(f"    Bot : Human ratio = {bot_count} : {human_count}  (ratio {imbalance_ratio:.2f}x)")
if imbalance_ratio > 1.5:
    log("    ⚠  Imbalanced — consider oversampling bots or class-weight tuning.")
else:
    log("    ✓  Roughly balanced.")
log()

# 6b. Near-zero importance features (candidates for removal)
zero_importance = fi_df[fi_df["importance"] == 0.0]["feature"].tolist()
near_zero       = fi_df[fi_df["importance"] < 0.005]["feature"].tolist()
log(f"  Zero-importance features ({len(zero_importance)}) — candidates for removal:")
for f in zero_importance:
    log(f"    - {f}")
log()
log(f"  Near-zero-importance features (gain < 0.005, {len(near_zero)}):")
for f in near_zero:
    log(f"    - {f}")
log()

# 6c. Highest-variance features
std_series = X.std().sort_values(ascending=False)
log("  Highest-variance features (top 10 by std dev):")
for feat, s in std_series.head(10).items():
    log(f"    {feat:<35}  std={s:.4f}")
log()

# 6d. Data size concern
log(f"  Dataset size:")
log(f"    Only {total_samples} batch rows across {total_sessions} sessions.")
log("    ⚠  Very small — metrics have high variance. Add more labelled sessions.")
log()

# 6e. Data source concern
log("  Data source:")
log("    ⚠  100% simulated data. No real user / real bot sessions.")
log("    Generalisation to production traffic is unproven.")
log()

# 6f. Label noise
log("  Label noise:")
log(f"    A {int(NOISE_RATE*100)}% noise rate is applied to both train and test sets during")
log("    training (ModelDataset.prepare). This reduces reported metrics")
log("    below real performance; and it pollutes the test set — making")
log("    true test-set evaluation impossible without a clean holdout.")
log()

# 6g. Corrupted pickles
corrupted = []
for name in ["RandomForest", "LogisticRegression", "XGBoost"]:
    m = try_load_pkl(TRAINED_DIR / f"{name}.pkl")
    if m is None:
        corrupted.append(name)
if corrupted:
    log(f"  Corrupted model artifacts:")
    for c in corrupted:
        log(f"    ⚠  models/trained/{c}.pkl — failed to unpickle (invalid load key).")
    log("    Run `python -m models.run_training` to regenerate all artifacts.")
    log()

# 6h. Champion model note
log(f"  Champion model note:")
log(f"    Stored champion is '{champion_name}', which has lower accuracy ({comparison['models'][1]['accuracy']:.4f})")
log(f"    and F1 ({comparison['models'][1]['f1']:.4f}) than RandomForest but higher AUC")
log(f"    ({comparison['models'][1]['roc_auc']:.4f} vs {comparison['models'][0]['roc_auc']:.4f}).")
log("    With a test set of only 22 samples, AUC differences of <0.02 are not significant.")
log()

log("=" * 65)
log("  END OF AUDIT")
log("=" * 65)

# ── 7. Write to file ──────────────────────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text("\n".join(lines) + "\n")
print(f"\n✓ Audit written to {OUTPUT_PATH}")
