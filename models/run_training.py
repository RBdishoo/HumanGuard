from models.dataset import ModelDataset
from models.train import ModelTrainer
import logging
import json
import joblib
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from backend.features.dataset_builder import DatasetBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAINED_DIR = Path("models/trained")


def _load_previous_champion_auc() -> float | None:
    """Return previous champion's ROC-AUC from registry, or None if absent."""
    comparison_path = TRAINED_DIR / "model_comparison.json"
    if not comparison_path.exists():
        return None
    try:
        with open(comparison_path) as f:
            data = json.load(f)
        winner = data.get("winner")
        for m in data.get("models", []):
            if m.get("model") == winner:
                return m.get("roc_auc")
    except Exception:
        return None
    return None


def _run_cross_validation(X_raw, y, random_state: int = 42) -> dict:
    """
    5-fold stratified cross-validation with pipeline-internal scaling.
    Returns {model_name: {metric: (mean, std), ...}} for all three classifiers.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'f1', 'roc_auc']

    estimators = {
        'RandomForest': RandomForestClassifier(
            n_estimators=50, max_depth=4, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1),
        'LogisticRegression': LogisticRegression(
            max_iter=1000, C=1.0, random_state=random_state),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=random_state, n_jobs=-1, eval_metric='logloss'),
    }

    results = {}
    for name, clf in estimators.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        cv   = cross_validate(pipe, X_raw, y, cv=skf, scoring=scoring, n_jobs=-1)
        results[name] = {
            'accuracy': (float(np.mean(cv['test_accuracy'])),
                         float(np.std(cv['test_accuracy']))),
            'f1':       (float(np.mean(cv['test_f1'])),
                         float(np.std(cv['test_f1']))),
            'roc_auc':  (float(np.mean(cv['test_roc_auc'])),
                         float(np.std(cv['test_roc_auc']))),
        }
    return results


def _print_cv_table(cv_results: dict) -> None:
    print("\n── 5-Fold Stratified Cross-Validation ───────────────────────────")
    print(f"  {'Model':<22} {'Accuracy (mean±std)':>22} {'F1 (mean±std)':>20} "
          f"{'AUC (mean±std)':>20}")
    print("  " + "-" * 86)
    for name, metrics in cv_results.items():
        acc_m, acc_s = metrics['accuracy']
        f1_m,  f1_s  = metrics['f1']
        auc_m, auc_s = metrics['roc_auc']
        print(f"  {name:<22}  {acc_m:.4f} ± {acc_s:.4f}      "
              f"{f1_m:.4f} ± {f1_s:.4f}      {auc_m:.4f} ± {auc_s:.4f}")
    print()


def main():
    signalsJsonlPath = 'backend/data/raw/signals.jsonl'
    batchCSV  = 'backend/data/processed/training_data_batches.csv'
    labelsCSV = 'backend/data/raw/labels.csv'

    # 1) Build features from raw signals
    builder = DatasetBuilder(signalsJsonlPath)
    logger.info("Building batch-level dataset...")
    builder.buildBatchLevelDataset(batchCSV)

    # 2) Cross-validation on full dataset (clean, unscaled — Pipeline handles scaling)
    logger.info("Running 5-fold cross-validation...")
    dataset_cv = ModelDataset(batchCSV, labelsCSV)
    X_raw, y_raw, _ = dataset_cv.get_raw_dataset()
    cv_results = _run_cross_validation(X_raw, y_raw)
    _print_cv_table(cv_results)

    # 3) Final training split with Gaussian noise augmentation on train set
    dataset = ModelDataset(batchCSV, labelsCSV)
    xTrain, xTest, yTrain, yTest, featureNames, scaler = dataset.prepare(add_noise=True)

    print(f"Train shape: {xTrain.shape}, Test shape: {xTest.shape}")
    print(f"Feature count: {len(featureNames)} (noise augmented on train set)")

    # 4) Train all three models on the noise-augmented split
    trainer = ModelTrainer()

    rfModel,  rfMetrics,  rfImportance  = trainer.trainRandomForest(
        xTrain, xTest, yTrain, yTest, featureNames
    )
    lrModel,  lrMetrics,  lrImportance  = trainer.trainLogisticRegression(
        xTrain, xTest, yTrain, yTest, featureNames
    )
    xgbModel, xgbMetrics, xgbImportance = trainer.trainXGBoost(
        xTrain, xTest, yTrain, yTest, featureNames
    )

    # 5) Print easy-test comparison table
    allMetrics = [rfMetrics, lrMetrics, xgbMetrics]
    allModels  = [
        ('RandomForest',       rfModel,  rfImportance),
        ('LogisticRegression', lrModel,  lrImportance),
        ('XGBoost',            xgbModel, xgbImportance),
    ]

    print("\n── Easy Test Set (80/20 stratified split) ───────────────────────")
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} "
          f"{'F1':>9} {'ROC-AUC':>9}")
    print("  " + "-" * 66)
    for m in allMetrics:
        print(f"  {m['model']:<22} {m['accuracy']:>9.4f} {m['precision']:>9.4f} "
              f"{m['recall']:>9.4f} {m['f1']:>9.4f} {m['roc_auc']:>9.4f}")
    print()

    # 6) Select best by CV AUC (not just easy-test AUC)
    # Use CV AUC as the primary selection metric for a more honest comparison
    bestName = max(cv_results, key=lambda n: cv_results[n]['roc_auc'][0])
    bestIdx   = next(i for i, (n, _, _) in enumerate(allModels) if n == bestName)
    _, bestModel, bestImportance = allModels[bestIdx]
    bestMetrics = allMetrics[bestIdx]
    cv_auc_mean, cv_auc_std = cv_results[bestName]['roc_auc']

    print(f"Best model by CV AUC: {bestName}  "
          f"(CV AUC {cv_auc_mean:.4f} ± {cv_auc_std:.4f}  |  "
          f"easy-test AUC {bestMetrics['roc_auc']:.4f})")

    # 7) Promotion gate — based on CV AUC
    prev_auc = _load_previous_champion_auc()
    PROMOTE_THRESHOLD = 0.90

    print("\n── Champion promotion decision ──────────────────────────────────")
    if prev_auc is not None:
        print(f"  Previous champion AUC (easy-test) : {prev_auc:.4f}")
    else:
        print("  Previous champion AUC             : (none — first run)")
    print(f"  New CV AUC (mean)                 : {cv_auc_mean:.4f}")
    print(f"  Promotion threshold               : {PROMOTE_THRESHOLD:.2f}")

    if cv_auc_mean < PROMOTE_THRESHOLD:
        print(f"\n  ✗ NOT promoted — CV AUC {cv_auc_mean:.4f} < {PROMOTE_THRESHOLD:.2f}")
        return

    print(f"\n  ✓ Promoted — CV AUC {cv_auc_mean:.4f} >= {PROMOTE_THRESHOLD:.2f}")

    # 8) Save artifacts
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    trainer.saveModel(bestModel, bestName)
    joblib.dump(scaler, TRAINED_DIR / "scaler.pkl")
    with open(TRAINED_DIR / "feature_names.json", "w") as f:
        json.dump(featureNames, f)
    with open(TRAINED_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": 0.5}, f)
    for name, _, imp_df in allModels:
        imp_df.to_csv(TRAINED_DIR / f"{name.lower()}_feature_importance.csv", index=False)
    trainer.saveMetrics(allMetrics, outputFile="metrics.json")

    # Save CV results alongside the model comparison
    comparison = {
        "winner":           bestName,
        "selection_metric": "cv_roc_auc",
        "models":           allMetrics,
        "cross_validation": {
            name: {
                "accuracy_mean":  v['accuracy'][0],
                "accuracy_std":   v['accuracy'][1],
                "f1_mean":        v['f1'][0],
                "f1_std":         v['f1'][1],
                "roc_auc_mean":   v['roc_auc'][0],
                "roc_auc_std":    v['roc_auc'][1],
            }
            for name, v in cv_results.items()
        },
    }
    with open(TRAINED_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # 9) Old vs new summary
    print("\n── Old vs New comparison ────────────────────────────────────────")
    print(f"  {'Metric':<15} {'Old champion':>14} {'New (CV mean)':>14}")
    print("  " + "-" * 44)
    if prev_auc is not None:
        print(f"  {'AUC-ROC':<15} {prev_auc:>14.4f} {cv_auc_mean:>14.4f}")
    else:
        print(f"  {'AUC-ROC':<15} {'N/A':>14} {cv_auc_mean:>14.4f}")
    print()
    print(f"  Training complete. Active model: {bestName}")


if __name__ == '__main__':
    main()
