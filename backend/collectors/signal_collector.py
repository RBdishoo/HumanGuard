"""
Signal Collector Module 

    Handles persistent storage of behavioral signals to JSON lines format.
    Each line is a complete signal batch (valid JSON).

    Why JSON Lines are used?
    Because they can perform quick append-only operations, each batch is independent (making it easier to process), human-readable making it easier to debug, and no need to load entire file into memory.

    
    __init__() sets up the collector and ensure files exists
    saveSignalBatch() appends one batch of signals to the file
    getBatchCount() returns total batches collected
    getSessionCount() returns unique session count
    getLatestSignals() gets the most recent batches for debugging
"""

import json
import logging
import os
from utils.helpers import formatTimestamp
from pathlib import Path

logger = logging.getLogger(__name__)

IS_LAMBDA = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
DATA_DIR = Path("/tmp") if IS_LAMBDA else Path(__file__).parent.parent / "data" / "raw"
signalsFile = DATA_DIR / "signals.jsonl"

class SignalCollector:

    """
    Manages the storage of user behavioral signals. These signals are stored in JSON Lines format (one JSON object per line).

    """

    def getSignalsFile(self):
        return self.signalsFile
    
    
    def __init__(self):
        """Initializes the signal collector"""
        self.signalsFile = str(signalsFile)
        self.ensureFileExists()

    def ensureFileExists(self):
        """Create signals.jsonl file if it doesn't exist"""
        path = Path(self.signalsFile)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.open('a').close()

    def saveSignalBatch(self, batchData):
        """Saves a batch of signals to the file """
        try:
            #add a timestamp if not present
            if 'timestamp' not in batchData:
                batchData['timestamp'] = formatTimestamp()
            
            #convert to JSON and appent to file (one line per batch)
            with open(self.signalsFile, 'a') as f:
                f.write(json.dumps(batchData) + '\n')

            # Dual-write to PostgreSQL if available
            try:
                from db.db_client import is_available, save_signal_batch
                if is_available():
                    save_signal_batch(batchData)
            except Exception as exc:
                logger.warning("PostgreSQL write failed (JSONL saved): %s", exc)

            return True
        except Exception as e:
            print(f"Error saving signal batch: {e}")
            return False
        
    def getBatchCount(self):

        """Get total number of batches collected """
        try:
            with open(self.signalsFile, 'r') as f:
                count = sum(1 for line in f)
            return count
        except Exception as e:
            print(f"Error counting batches: {e}")
            return 0
        
    def getSessionCount(self):
        """Get Number of unique session IDs"""
        try:
            sessions = set()
            with open(self.signalsFile, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    sessions.add(data.get('sessionID'))
            return len(sessions)
        except Exception as e:
            print(f"Error counting sessions: {e}")
            return 0

    def getLatestSignals(self, limit=10):
        """Get the most recent signal batches, useful for debugging and seeing what's being collected."""
        try:
            signals = []
            with open(self.signalsFile, 'r') as f:
                allLines = f.readlines()
            
            # Get last 'limit' lines
            for line in allLines[-limit:]:
                signals.append(json.loads(line))

            return signals
        except Exception as e:
            print(f"Error retrieving signals: {e}")
            return []
        
