import pandas as pd
from pathlib import Path
import logging

from .data_loader import SignalDataLoader
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class DatasetBuilder: 
    """
    Builds training dataset by loading signals and extracting features.
    """

    def __init__(self, signalsJsonlPath: str):
        self.loader = SignalDataLoader(signalsJsonlPath)
        self.extractor = FeatureExtractor()

    def buildBatchLevelDataset(self, outputCSVPath: str) -> pd.DataFrame:
        """
        Extract features at batch level (one row = one batch)

        Args: output_csv_path (where to save the training csv)

        returns: Datafram with features
        
        """

        logger.info("Loading signals...")
        signalsDf = self.loader.loadSignals()

        if len(signalsDf) == 0:
            logger.error("No Signals Loaded")
            return pd.DataFrame()
        
        featureRows = []

        for indx, row in signalsDf.iterrows():
            if indx % 10 == 0:
                logger.info(f"Processing batch {indx}/{len(signalsDf)}")

            batchData = {
                "mouseMoves": row["mouseMoves"],
                "clicks": row["clicks"],
                "keys": row["keys"],
            }

            feats = self.extractor.extractBatchFeatures(batchData)

            # Merge in network/device features stored in batch metadata
            nf_raw = row.get("network_features")
            if not isinstance(nf_raw, dict):
                nf_raw = {}
            feats.update(self.extractor.extract_network_features(nf_raw))
            sessionCol = "sessionID"
            if sessionCol not in row.index and "index" in row.index:
                sessionCol = "index"

            feats["sessionID"] = row.get("sessionID", row.name)
            feats["timestamp"] = row["timestamp"]
            feats["timestampRelativeMs"] = row["timestampRelativeMs"]

            featureRows.append(feats)

        featuresDf = pd.DataFrame(featureRows)

        #Add label column placeholder
        featuresDf["label"] = None

        logger.info(f"Extracted features for {len(featuresDf)} batches")
        logger.info(f"features: {list(featuresDf.columns)}")

        #Save CSV
        outputPath = Path(outputCSVPath)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        featuresDf.to_csv(outputCSVPath, index=False)

        logger.info(f"Saved batch-level dataset to {outputCSVPath}")
        return featuresDf
    
    def buildSessionLevelDataset(self, outputCSVPath: str) -> pd.DataFrame:
        """
        Aggregate batch-level features to session level.
        Each row = one session with mean/std/min/max/percentiles.

        Arguments: outputCSVPath: Where to save the session-level CSV

        Returns: DataFrame with aggregated features
        
        """

        #First build batch-level and also keep a batch CSV besides it
        batchCsvPath = outputCSVPath.replace(".csv", "_batches.csv")
        batchDf = self.buildBatchLevelDataset(batchCsvPath)

        if len(batchDf) == 0:
            logger.error("No batch-level data to aggregate")
            return pd.DataFrame()
        
        #Non-feature columns to exclude from aggregation
        exclude = ["sessionID", "timestamp", "timestampRelativeMs", "label"]
        featureCols = [c for c in batchDf.columns if c not in exclude]

        sessionRows = []

        for sessionId in batchDf["sessionID"].unique():
            sessionData = batchDf[batchDf["sessionID"]== sessionId] 

            row = {"sessionID": sessionId, "batchCount": len(sessionData)}

            for col in featureCols:
                values = sessionData[col].dropna()

                if len(values) > 0:
                    row[f"{col}_mean"] = float(values.mean())
                    row[f"{col}_std"] = float(values.std())
                    row[f"{col}_min"] = float(values.min())
                    row[f"{col}_max"] = float(values.max())
                    row[f"{col}_p50"] = float(values.quantile(0.5))
                    row[f"{col}_p95"] = float(values.quantile(0.95))
                
                else:
                    row[f"{col}_mean"] = 0.0
                    row[f"{col}_std"] = 0.0
                    row[f"{col}_min"] = 0.0
                    row[f"{col}_max"] = 0.0
                    row[f"{col}_p50"] = 0.0
                    row[f"{col}_p95"] = 0.0

            row["label"] = None
            sessionRows.append(row)

        sessionDf = pd.DataFrame(sessionRows)
        sessionPath = Path(outputCSVPath)
        sessionPath.parent.mkdir(parents=True, exist_ok=True)
        sessionDf.to_csv(outputCSVPath, index=False)

        logger.info(f"Saved session-level dataset to {outputCSVPath}")
        return sessionDf 
    

if __name__ == "__main__":

    signalsPath = "backend/data/raw/signals.jsonl"
    outputBatch = "backend/data/processed/training_data_batches.csv"
    outputSession = "backend/data/processed/training_data_sessions.csv"

    builder = DatasetBuilder(signalsPath)

    print("\n=== Building Batch-Level Dataset ===")
    batchDf = builder.buildBatchLevelDataset(outputBatch)
    print(f"\nBatch Dataset Shape: {batchDf.shape}")
    if len(batchDf) > 0:
        print(f"Sample row:\n{batchDf.iloc[0]}\n")

    print("=== Building Session-Level Dataset ===")
    sessionDf = builder.buildSessionLevelDataset(outputSession)
    print(f"\nSession Dataset Shape: {sessionDf.shape}")
    if len(sessionDf) > 0:
        print(f"Sample row:\n{sessionDf.iloc[0]}\n")

