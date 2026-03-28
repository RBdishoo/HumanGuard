""" 
Helper Functions for bot detection system.

These are the utility functions used across the application:
    1) Session ID generation
    2) File Path Management
    3) JSON formatting
"""

import os
import json
from datetime import datetime

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

def isValidSignalBatch(data):

    "Validate that incoming signal batch has required fields"
    requiredFields = ['sessionID', 'signals']
    if not all(field in data for field in requiredFields):
        return False
    
    signals = data.get("signals", {})

    #Each signals sub-field should be a list
    for key in ["mouseMoves", "clicks", "keys"]:
        if key in signals and not isinstance(signals[key], list):
            return False
    
    return True
    