from models.dataset import ModelDataset
from models.train import ModelTrainer
import logging
import json
import joblib
from pathlib import Path

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


def main():
    signalsJsonlPath = 'backend/data/raw/signals.jsonl'
    batchCSV  = 'backend/data/processed/training_data_batches.csv'
    labelsCSV = 'backend/data/raw/labels.csv'

    # 1) Build features from raw signals
    builder = DatasetBuilder(signalsJsonlPath)
    logger.info("Building batch-level dataset...")
    builder.buildBatchLevelDataset(batchCSV)

    # 2) Load features + labels and prepare ML arrays (clean, no noise)
    dataset = ModelDataset(batchCSV, labelsCSV)
    xTrain, xTest, yTrain, yTest, featureNames, scaler = dataset.prepare()

    print(f"Train shape: {xTrain.shape}, Test shape: {xTest.shape}")
    print(f"Feature count: {len(featureNames)}")

    # 3) Train all three models on the same stratified split
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

    # 4) Print comparison table
    allMetrics = [rfMetrics, lrMetrics, xgbMetrics]
    allModels  = [
        ('RandomForest',      rfModel,  rfImportance),
        ('LogisticRegression', lrModel, lrImportance),
        ('XGBoost',           xgbModel, xgbImportance),
    ]

    print("\n" + "=" * 78)
    print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 78)
    for m in allMetrics:
        print(f"{m['model']:<22} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['roc_auc']:>10.4f}")
    print("=" * 78)

    # 5) Select best model by ROC-AUC
    bestIdx = max(range(len(allMetrics)), key=lambda i: allMetrics[i]['roc_auc'])
    bestName, bestModel, bestImportance = allModels[bestIdx]
    bestMetrics = allMetrics[bestIdx]
    new_auc = bestMetrics['roc_auc']

    print(f"\nBest model by ROC-AUC: {bestName} ({new_auc:.4f})")

    # 6) Compare against previous champion
    prev_auc = _load_previous_champion_auc()
    PROMOTE_THRESHOLD = 0.90

    print("\n── Champion promotion decision ──────────────────────────────────")
    if prev_auc is not None:
        print(f"  Previous champion AUC : {prev_auc:.4f}")
    else:
        print("  Previous champion AUC : (none — first training run)")
    print(f"  New best model AUC    : {new_auc:.4f}")
    print(f"  Promotion threshold   : {PROMOTE_THRESHOLD:.2f}")

    if new_auc < PROMOTE_THRESHOLD:
        print(f"\n  ✗ NOT promoted — AUC {new_auc:.4f} < threshold {PROMOTE_THRESHOLD:.2f}")
        print("    Artifacts NOT updated. Re-examine training data or hyperparameters.")
        return

    print(f"\n  ✓ Promoted — AUC {new_auc:.4f} >= threshold {PROMOTE_THRESHOLD:.2f}")
    if prev_auc is not None:
        delta = new_auc - prev_auc
        sign  = "+" if delta >= 0 else ""
        print(f"    AUC change vs previous champion: {sign}{delta:.4f}")

    # 7) Save winning model as the active inference artifact
    TRAINED_DIR.mkdir(parents=True, exist_ok=True)

    trainer.saveModel(bestModel, bestName)
    joblib.dump(scaler, TRAINED_DIR / "scaler.pkl")
    with open(TRAINED_DIR / "feature_names.json", "w") as f:
        json.dump(featureNames, f)

    threshold = 0.5
    with open(TRAINED_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f)

    # Save feature importance for all models (useful for analysis)
    for name, _, importance_df in allModels:
        importance_df.to_csv(TRAINED_DIR / f"{name.lower()}_feature_importance.csv", index=False)

    # Save all metrics for offline review
    trainer.saveMetrics(allMetrics, outputFile="metrics.json")

    # 8) Save comparison record
    comparison = {
        "winner":           bestName,
        "selection_metric": "roc_auc",
        "models":           allMetrics,
    }
    with open(TRAINED_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved comparison to {TRAINED_DIR / 'model_comparison.json'}")

    # 9) Print old vs new summary
    print("\n── Old vs New comparison ────────────────────────────────────────")
    print(f"  {'Metric':<12} {'Old champion':>14} {'New champion':>14} {'Delta':>10}")
    print("  " + "-" * 50)
    if prev_auc is not None:
        print(f"  {'AUC-ROC':<12} {prev_auc:>14.4f} {new_auc:>14.4f} {new_auc - prev_auc:>+10.4f}")
    else:
        print(f"  {'AUC-ROC':<12} {'N/A':>14} {new_auc:>14.4f} {'N/A':>10}")
    print()
    print(f"  Training Complete. Active model: {bestName}")


if __name__ == '__main__':
    main()
