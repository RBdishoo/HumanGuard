from models.dataset import ModelDataset
from models.train import ModelTrainer
import logging
import json

from backend.features.dataset_builder import DatasetBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Batch-level MVP training:
    # - Build a batch-level feature CSV (one row per ingested JSONL batch).
    # - Join labels by sessionID.
    # - Train a classifier that predicts prob(bot) per batch.
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

    # 3) Train model
    trainer = ModelTrainer()
    rfModel, rfMetrics, rfImportance = trainer.trainRandomForest(
        xTrain, xTest, yTrain, yTest, featureNames
    )

    print(f"RandomForest Accuracy: {rfMetrics['accuracy']:.4f}")
    print(f"RandomForest ROC-AUC: {rfMetrics['roc_auc']:.4f}")

    # 4) Save inference artifacts required by /api/score
    import joblib
    trainedDir = trainer.outputDir

    trainer.saveModel(rfModel, 'RandomForest')
    joblib.dump(scaler, trainedDir / "scaler.pkl")
    with open(trainedDir / "feature_names.json", "w") as f:
        json.dump(featureNames, f)

    threshold = 0.5
    with open(trainedDir / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f)

    # Save metrics and feature importance for offline review
    trainer.saveMetrics([rfMetrics], outputFile="metrics.json")
    rfImportance.to_csv(trainedDir / "rf_feature_importance.csv", index=False)

    print("\nTraining Complete. Artifacts saved to models/trained/")


if __name__ == '__main__':
    main()