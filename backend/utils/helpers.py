""" 
Helper Functions for bot detection system.

These are the utility functions used across the application:
    1) Session ID generation
    2) File Path Management
    3) JSON formatting
"""

import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def genSeshID():
    import time
    import random
    import string

    """This generates a unique session ID using a timestamp and a random string of characters"""
    timestamp = int(time.time())
    randomSuffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
    seshID = f"session_{timestamp}_{randomSuffix}"
    return seshID

def getDataDirectory():
    
    """Gets or creates data directory. Creates data/raw/ if it doesn't exist """

    dataDirectory = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
    os.makedirs(dataDirectory, exist_ok=True)
    return dataDirectory

def getSignalsFile():

    """Gets the path to signals.jsonl file"""
    dataDirectory = getDataDirectory()
    return os.path.join(dataDirectory, 'signals.jsonl')

def formatTimestamp():

    """Current timestamp in ISO format"""
    return datetime.utcnow().isoformat() + 'Z'

def normalizeSignalBatch(data):
    """
    Normalize alternative field names to the canonical schema before validation.

    Handles:
      - sessionId  (lowercase d)  → sessionID
      - mouseEvents at top level  → signals.mouseMoves
      - keyEvents   at top level  → signals.keys
      - clickEvents at top level  → signals.clicks
    """
    if not isinstance(data, dict):
        return data

    normalized = dict(data)

    # sessionId → sessionID (keep existing alias)
    if 'sessionId' in normalized and 'sessionID' not in normalized:
        normalized['sessionID'] = normalized.pop('sessionId')

    # sessionID → session_id (canonical server-side key)
    if 'sessionID' in normalized and 'session_id' not in normalized:
        normalized['session_id'] = normalized.pop('sessionID')

    # flat event arrays → signals wrapper
    if 'signals' not in normalized:
        flat_keys = {'mouseEvents': 'mouseMoves', 'keyEvents': 'keys', 'clickEvents': 'clicks'}
        has_flat = any(k in normalized for k in flat_keys)
        if has_flat:
            signals = {}
            for src, dst in flat_keys.items():
                if src in normalized:
                    signals[dst] = normalized.pop(src)
            normalized['signals'] = signals

    return normalized


def isValidSignalBatch(data):
    """Validate that incoming signal batch has required fields."""
    logger.debug("isValidSignalBatch — incoming keys: %s", list(data.keys()) if isinstance(data, dict) else type(data))

    if not isinstance(data, dict):
        logger.warning("isValidSignalBatch FAILED — data is not a dict (got %s)", type(data))
        return False

    if 'session_id' not in data:
        logger.warning("isValidSignalBatch FAILED — missing 'session_id' (keys present: %s)", list(data.keys()))
        return False

    if 'signals' not in data:
        logger.warning("isValidSignalBatch FAILED — missing 'signals' (keys present: %s)", list(data.keys()))
        return False

    signals = data.get("signals", {})

    if not isinstance(signals, dict):
        logger.warning("isValidSignalBatch FAILED — 'signals' is not a dict (got %s)", type(signals))
        return False

    for key in ["mouseMoves", "clicks", "keys"]:
        val = signals.get(key)
        if val is None:
            continue
        if not isinstance(val, list):
            logger.warning("isValidSignalBatch FAILED — signals.%s must be a list (got %s)", key, type(val))
            return False
        for elem in val:
            if not isinstance(elem, dict):
                logger.warning(
                    "isValidSignalBatch FAILED — signals.%s contains non-dict element (got %s)",
                    key, type(elem),
                )
                return False

    return True
    