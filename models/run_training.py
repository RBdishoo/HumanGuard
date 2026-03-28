from models.dataset import ModelDataset
from models.train import ModelTrainer
import logging
import json
import joblib

from backend.features.dataset_builder import DatasetBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    signalsJsonlPath = 'backend/data/raw/signals.jsonl'
    batchCSV = 'backend/data/processed/training_data_batches.csv'
    labelsCSV = 'backend/data/raw/labels.csv'

    # 1) Build features from raw signals
    builder = DatasetBuilder(signalsJsonlPath)
    logger.info("Building batch-level dataset...")
    builder.buildBatchLevelDataset(batchCSV)

    # 2) Load features + labels and prepare ML arrays
    dataset = ModelDataset(batchCSV, labelsCSV)
    xTrain, xTest, yTrain, yTest, featureNames, scaler = dataset.prepare()

    print(f"Train shape: {xTrain.shape}, Test shape: {xTest.shape}")
    print(f"Feature Names sample: {featureNames[:5]}...")

    # 3) Train all three models on the same split
    trainer = ModelTrainer()

    rfModel, rfMetrics, rfImportance = trainer.trainRandomForest(
        xTrain, xTest, yTrain, yTest, featureNames
    )
    lrModel, lrMetrics, lrImportance = trainer.trainLogisticRegression(
        xTrain, xTest, yTrain, yTest, featureNames
    )
    xgbModel, xgbMetrics, xgbImportance = trainer.trainXGBoost(
        xTrain, xTest, yTrain, yTest, featureNames
    )

    # 4) Print comparison table
    allMetrics = [rfMetrics, lrMetrics, xgbMetrics]
    allModels = [
        ('RandomForest', rfModel, rfImportance),
        ('LogisticRegression', lrModel, lrImportance),
        ('XGBoost', xgbModel, xgbImportance),
    ]

    print("\n" + "=" * 78)
    print(f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-" * 78)
    for m in allMetrics:
        print(f"{m['model']:<22} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['roc_auc']:>10.4f}")
    print("=" * 78)

    # 5) Select best model by ROC-AUC
    bestIdx = max(range(len(allMetrics)), key=lambda i: allMetrics[i]['roc_auc'])
    bestName, bestModel, bestImportance = allModels[bestIdx]
    bestMetrics = allMetrics[bestIdx]

    print(f"\nBest model by ROC-AUC: {bestName} ({bestMetrics['roc_auc']:.4f})")

    # 6) Save winning model as the active inference artifact
    trainedDir = trainer.outputDir

    trainer.saveModel(bestModel, bestName)
    joblib.dump(scaler, trainedDir / "scaler.pkl")
    with open(trainedDir / "feature_names.json", "w") as f:
        json.dump(featureNames, f)

    threshold = 0.5
    with open(trainedDir / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f)

    # Save feature importance for the winner
    bestImportance.to_csv(trainedDir / f"{bestName.lower()}_feature_importance.csv", index=False)

    # Save all metrics for offline review
    trainer.saveMetrics(allMetrics, outputFile="metrics.json")

    # 7) Save comparison record
    comparison = {
        "winner": bestName,
        "selection_metric": "roc_auc",
        "models": allMetrics,
    }
    with open(trainedDir / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved comparison to {trainedDir / 'model_comparison.json'}")

    print(f"\nTraining Complete. Active model: {bestName}")


if __name__ == '__main__':
    main()
