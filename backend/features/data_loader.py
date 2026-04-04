import json
import pandas as pd
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalDataLoader:
    """
    Loads and validates signals from JSONL file.
    Handles preprocessing and initial validation
    """
    

    def __init__(self, jsonlPath: str):
        """Initialize loader with path to signals.jsonl"""
        self.jsonlPath = Path(jsonlPath)
        if not self.jsonlPath.exists():
            raise FileNotFoundError(f"Signals file not found: {self.jsonlPath}")
    
    def loadSignals(self) -> pd.DataFrame:
        """
        Load all signals from JSONL into a denormalized DataFrame.

        Returns:
            DataFrame with columns - sessionID, timestamp, mouseMoves (list), clicks (list), keys (list), userAgent, viewportWidth, viewportHeight
        """

        data: List[dict] = []
        invalidCount = 0

        with open(self.jsonlPath, "r") as f:
            for lineNum, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    if not self._validateRecord(record):
                        logger.warning(f"Invalid record at line {lineNum}")
                        invalidCount += 1
                        continue

                    meta = record.get("metadata", {}) or {}
                    # Accept legacy typo from older tracker.js (viewerport*)
                    vw = meta.get("viewportWidth") or meta.get("viewerportWidth")
                    vh = meta.get("viewportHeight") or meta.get("viewerportHeight")
                    flat = {
                        "sessionID": record.get("sessionID"),
                        "timestamp": record.get("timestamp"),
                        "mouseMoves": record.get("signals", {}).get("mouseMoves", []),
                        "clicks": record.get("signals", {}).get("clicks", []),
                        "keys": record.get("signals", {}).get("keys", []),
                        "userAgent": meta.get("userAgent"),
                        "viewportWidth": vw,
                        "viewportHeight": vh,
                        "network_features": meta.get("network_features") or {},
                    }
                    data.append(flat)
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON decode error at line {lineNum}: {e}")
                    invalidCount += 1
                except Exception as e:
                    logger.error(f"Unexpected error at line {lineNum}: {e}")
                    invalidCount += 1

        logger.info(f"Loaded {len(data)} valid batches, {invalidCount} invalid")
        df = pd.DataFrame(data)

        if len(df) == 0:
            logger.warning("No valid signals Loaded")
            return df
        
        #Debug: inspect columns and a few rows
        print("Columns:", list(df.columns))
        print(df.head())
        
        #Normalize timestamps to relative (session start = 0)
        df = self.normalizeTimestamps(df)
        print("After normalize Columns:", list(df.columns))
        logger.info(f"sessionID col after normalize: { 'sessionID' in df.columns }")

        return df
    
    @staticmethod
    def _validateRecord(record:dict) -> bool:
        """Validate that the record has required structure"""
        required = ["sessionID", "signals", "metadata"]
        if not all(k in record for k in required):
            return False
        
        signals = record.get("signals", {})
        for key in ["mouseMoves", "clicks", "keys"]:
            if key in signals and not isinstance(signals[key], list):
                return False
            
        return True
    
    @staticmethod
    def normalizeTimestamps(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert absolute timestamps to relative (offset from session start)

        for each sessionID, sets earliest timestamp to 0.
        """
        #Debug
        print("Normalize in Columns:", list(df.columns))

        #SessionID is in a normal column
        if "sessionID" not in df.columns and "sessionID" in df.index.names:
            df = df.reset_index()

        def normalizeGroup(group: pd.DataFrame) -> pd.DataFrame:
            group["timestamp"] = pd.to_datetime(group["timestamp"])
            sessionStart = group["timestamp"].min()
            group["timestampRelativeMs"] = (
                group["timestamp"] - sessionStart
            ).dt.total_seconds() * 1000.0
            group["timestampRelativeMs"] = group["timestampRelativeMs"].astype(int)
            return group
    
        #groupby without changing the index to sessionID
        df = df.groupby("sessionID", group_keys=False).apply(normalizeGroup)
        

        print("Normalize Out columns:", list(df.columns))
        return df
    
    def getSessionData(self, sessionId: str) -> pd.DataFrame:
        """Get all batches for a specific reason"""
        df = self.loadSignals()
        return df[df["sessionID"] == sessionId]