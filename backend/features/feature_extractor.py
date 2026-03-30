import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from .feature_utils import MouseTrajectoryUtils, KeystrokeUtils

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extracts engineered features from raw signal batches.
    Focuses on behavioral signs that differentiate humans from robots.
    
    """

    def __init__(self, pauseDistanceThresholdPx: float = 5.0, pauseDurationThresholdMs: int = 1000):
        """
        Initialize Feature Extractor

        """

        self.pauseDistThresh = pauseDistanceThresholdPx
        self.pauseDurThresh = pauseDurationThresholdMs
        self.utilsMouse = MouseTrajectoryUtils()
        self.utilsKey = KeystrokeUtils()

    def extractBatchFeatures(self, batchData: dict) -> Dict[str, float]:
        """
        Extract all features from a single batch

        Arguments:
            mouseMoves, clicks, keys

        Returns:
            Dictionary: featureName -> numerica value (float)
        """

        features: Dict[str, float] = {}
        
        #Parse signals
        moves = batchData.get('mouseMoves', []) or []
        clicks = batchData.get('clicks', []) or []
        keys = batchData.get('keys', []) or []

        #simple counts / presence flags
        features['batch_event_count'] = float(len(moves) + len(clicks) + len(keys))

        #Mouse features
        mouseFeatures = self.extractMouseFeatures(moves)
        features.update(mouseFeatures)

        #Click features
        clickFeatures = self.extractClickFeatures(clicks)
        features.update(clickFeatures)

        #keystroke features
        keystrokeFeatures = self.extractKeystrokeFeatures(keys)
        features.update(keystrokeFeatures)

        #temporal and Composite features
        temporalFeatures = self.extractTemporalFeatures(moves, clicks, keys)
        features.update(temporalFeatures)

        return features
    

    def extractMouseFeatures(self, moves: List[dict]) -> Dict[str, float]:

        """
        Extract mouse movement features from a list of move events.

        Each move event is expected to have:
            {'x': float, 'y': float, 'ts': float}
        
        """
        features: Dict[str, float] = {}

        #Edge Case - if fewer than 2 moves, we cannot compute velocities or angles
        if len(moves) < 2:
            features['mouseMoveCount'] = float(len(moves))
            features['mouseAvgVelocity'] = 0.0
            features['mouseStdVelocity'] = 0.0
            features['mouseMaxVelocity'] = 0.0
            features['mouseAvgPauseDurationMs'] = 0.0
            features['mousePathEfficiency'] = 0.0
            features['mouseAngularVelocityStd'] = 0.0
            features['mouseHoverTimeRatio'] = 0.0
            features['mouseHoverFrequency'] = 0.0
            return features
             
        
        #Convert moves into coordinates and time deltas
        coords, timeDeltas = self.utilsMouse.extractCoordinatesAndTimes(moves)

        #If filtering removed too many points, bail out with zeros
        if len(coords) < 2 or not timeDeltas:
            features['mouseMoveCount'] = float(len(moves))
            features['mouseAvgVelocity'] = 0.0
            features['mouseStdVelocity'] = 0.0
            features['mouseMaxVelocity'] = 0.0
            features['mouseAvgPauseDurationMs'] = 0.0
            features['mousePathEfficiency'] = 0.0
            features['mouseAngularVelocityStd'] = 0.0
            features['mouseHoverTimeRatio'] = 0.0
            features['mouseHoverFrequency'] = 0.0
            return features

        #Calculate distances and velocities between consecutive points
        distances = [
            self.utilsMouse.distance(coords[i], coords[i+1]) for i in range(len(coords)-1)
        ]

        velocities = [self.utilsMouse.velocityPixelsPerSecond(distances[i], timeDeltas[i]) for i in range(len(distances))]

        features['mouseMoveCount'] = float(len(moves))

        features['mouseAvgVelocity'] = float(np.mean(velocities)) if velocities else 0.0

        features['mouseStdVelocity'] = float(np.std(velocities)) if len(velocities) > 1 else 0.0

        features['mouseMaxVelocity'] = float(np.max(velocities)) if velocities else 0.0

        #Pause Detection using thresholds from __init__

        pauseCount, totalPauseMs = self.detectPauses(coords, timeDeltas)
        features['mouseAvgPauseDurationMs'] = float(totalPauseMs / max(pauseCount, 1))

        #Path efficiency = total path / straight line
        totalDistance = float(sum(distances))
        straightLine = self.utilsMouse.distance(coords[0], coords[-1])
        features['mousePathEfficiency'] = float(totalDistance / max(straightLine, 1.0))

        #Angular curviness: variance of turning angles
        if len(coords) >= 3:
            angles = [self.utilsMouse.angleBetween(coords[i], coords[i+1], coords[i+2]) for i in range(len(coords)-2)]
            features['mouseAngularVelocityStd'] = float(np.std(angles)) if len(angles) > 1 else 0.0
        else:
            features['mouseAngularVelocityStd'] = 0.0

        #Hover Behavior - time spent moving slow than 10 px/sec
        hoverMask = np.array(velocities) < 10.0
        hoverTimeMs = sum(timeDeltas[i] for i in range(len(velocities)) if hoverMask[i])
        totalTimeMs = sum(timeDeltas)
        features['mouseHoverTimeRatio'] = float(hoverTimeMs / max(totalTimeMs, 1.0))
        features['mouseHoverFrequency'] = float(np.sum(hoverMask) / max(len(hoverMask), 1))

        return features 
    
    def detectPauses(self, coords: List[tuple], timeDeltas: List[float]) -> tuple:

        """
        Detect Pauses: segments where movement distance is very small for at least self.pauseDurThresh milliseconds

        Returns - pauseCount, totalPauseDurationMs

        """

        #Edge case: no movement or no timing information
        if len(coords) < 2 or not timeDeltas:
            return 0,0

        distances = [
            self.utilsMouse.distance(coords[i], coords[i+1]) for i in range(len(coords)-1)
        ]

        pauseCount = 0
        totalPauseMs = 0
        isPaused = False
        pauseStartMs = 0.0

        for i, dist in enumerate(distances):
            elapsedUntilSegmentStart = float(sum(timeDeltas[:i]))

            if dist < self.pauseDistThresh:
                if not isPaused:
                    isPaused = True
                    pauseStartMs = elapsedUntilSegmentStart
            else:
                if isPaused:
                    pauseEndMs = elapsedUntilSegmentStart
                    pauseDuration = pauseEndMs - pauseStartMs
                    if pauseDuration >= self.pauseDurThresh:
                        pauseCount += 1
                        totalPauseMs += pauseDuration
                    isPaused = False

        if isPaused:
            totalElapsed = float(sum(timeDeltas))
            pauseDuration = totalElapsed - pauseStartMs
            if pauseDuration >= self.pauseDurThresh:
                pauseCount += 1
                totalPauseMs += pauseDuration

        return pauseCount, totalPauseMs

    def extractClickFeatures(self, clicks: List[dict]) -> Dict[str, float]:
       """
       Extract Click behavior features

       Each click event is expected to have: {'ts': float, 'button': int}

       """

       features: Dict[str, float] = {}

       if len(clicks) == 0:
           features['clickRatePerSec'] = 0.0
           features['clickIntervalMeanMs'] = 0.0
           features['clickIntervalStdMs'] = 0.0
           features['clickIntervalMinMs'] = 0.0
           features['clickIntervalMaxMs'] = 0.0
           return features

       #if more than one click, we can measure timing
       if len(clicks) > 1:
            timestamps = np.array([c['ts'] for c in clicks], dtype=float)
            intervals = np.diff(timestamps) #milliseconds between each click

            features['clickIntervalMeanMs'] = float(np.mean(intervals))
            features['clickIntervalStdMs'] = float(np.std(intervals)) if len(intervals) > 1 else 0.0
            features['clickIntervalMinMs'] = float(np.min(intervals))
            features['clickIntervalMaxMs'] = float(np.max(intervals))

            #Click rate per second over the whole batch
            totalDurationMs = timestamps[-1] - timestamps[0]
            if totalDurationMs > 0:
                features['clickRatePerSec'] = float(len(clicks) / totalDurationMs * 1000)
            else:
                features['clickRatePerSec'] = 0.0

       else:
            features['clickRatePerSec'] = 0.0
            features['clickIntervalMeanMs'] = 0.0
            features['clickIntervalStdMs'] = 0.0
            features['clickIntervalMinMs'] = 0.0
            features['clickIntervalMaxMs'] = 0.0

       return features
       
    
    def extractKeystrokeFeatures(self, keys: List[dict]) -> Dict[str, float]:
        """
        Extract Keystroke dynamics features

        Each key event is expected to have: {'code': str, 'ts': float}
        """
        features: Dict[str, float] = {}
        
        if len(keys) == 0:
            features['keyCount'] = 0.0
            features['keyRatePerSec'] = 0.0
            features['keyInterKeyDelayMeanMs'] = 0.0
            features['keyInterKeyDelayStdMs'] = 0.0
            features['keyEntropy'] = 0.0
            features['keyRapidPresses'] = 0.0
            return features
        
        features['keyCount'] = float(len(keys))

        #Inter key delays (timing)
        if len(keys) > 1:
            delays = np.array(self.utilsKey.interKeyDelays(keys), dtype=float)

            if len(delays) > 0:
                features['keyInterKeyDelayMeanMs'] = float(np.mean(delays))
                features['keyInterKeyDelayStdMs'] = float(np.std(delays)) if len(delays) > 1 else 0.0

                #Rapid presses: delays less than 50 milliseconds (bot-like speed)
                rapid = np.sum(delays < 50.0)
                features['keyRapidPresses'] = float(rapid)
            
            else:
                features['keyInterKeyDelayMeanMs'] = 0.0
                features['keyInterKeyDelayStdMs'] = 0.0
                features['keyRapidPresses'] = 0.0

        else:
            features['keyInterKeyDelayMeanMs'] = 0.0
            features['keyInterKeyDelayStdMs'] = 0.0
            features['keyRapidPresses'] = 0.0

        #Key variety (entropy over key codes)
        keyCodes = [k.get('code', 'Unknown') for k in keys]
        features['keyEntropy'] = float(self.utilsKey.calculateEntropy(keyCodes))

        #Overall key rate per second over the batch
        timestamps = [k['ts'] for k in keys]
        if len(timestamps) >= 2:
            durationMs = max(timestamps) - min(timestamps)
            features['keyRatePerSec'] = float(len(keys) / max(durationMs, 1.0) * 1000.0) 

        else:
            features['keyRatePerSec'] = 0.0

        return features
    
    def extractTemporalFeatures(self, moves: List[dict], clicks: List[dict], keys: List[dict]) -> Dict[str, float]:
       """
       extract temporal and composite features across all signals in the batch (mouse, clicks, keys)
       """
       features: Dict[str, float] = {}
       
       allTimestamps: List[float] = []
       allTimestamps.extend([m['ts'] for m in moves if "ts" in m])
       allTimestamps.extend([c['ts'] for c in clicks if "ts" in c])
       allTimestamps.extend([k['ts'] for k in keys if "ts" in k])

       if len(allTimestamps) >=2:
           batchDurationMs = max(allTimestamps) - min(allTimestamps)
           features['batchDurationMs'] = float(batchDurationMs)

           totalEvents = len(moves) + len(clicks) + len(keys)
           features['eventRatePerSec'] = float(totalEvents / max(batchDurationMs, 1.0) * 1000.0)

       else:
           features['batchDurationMs'] = 0.0
           features['eventRatePerSec'] = 0.0 

       #Simple ratios relative to mouse moves
       if len(moves) > 0:
           features['clickToMoveRatio'] = float(len(clicks) / len(moves))
           features['keyToMoveRatio'] = float(len(keys) / len(moves))

       else:
            features['clickToMoveRatio'] = 0.0
            features['keyToMoveRatio'] = 0.0
        

       return features

if __name__ == "__main__":
    # Quick manual test. From project root:
    #   python -m backend.features.feature_extractor

    def _print_features(title: str, batch: dict, extractor: FeatureExtractor) -> None:
        print(f"\n=== {title} ===")
        feats = extractor.extractBatchFeatures(batch)
        for k in sorted(feats.keys()):
            print(f"{k}: {feats[k]}")

    batch_straight_line = {
        "mouseMoves": [
            {"x": 100, "y": 200, "ts": 1000},
            {"x": 120, "y": 210, "ts": 1100},
            {"x": 140, "y": 220, "ts": 1300},
        ],
        "clicks": [
            {"ts": 1050, "button": 0},
            {"ts": 1600, "button": 0},
            {"ts": 1900, "button": 2},
        ],
        "keys": [
            {"code": "KeyH", "ts": 1020},
            {"code": "KeyE", "ts": 1080},
            {"code": "KeyL", "ts": 1160},
            {"code": "KeyL", "ts": 1230},
            {"code": "KeyO", "ts": 1350},
        ],
    }

    # Curved path + ~1.2s micro-movement stretch (distances < pauseDistThresh) then a long jump.
    # Exercises path efficiency, angular variance, and pause detection vs. the straight demo above.
    batch_curved_with_pause = {
        "mouseMoves": [
            {"x": 50, "y": 50, "ts": 10_000},
            {"x": 51, "y": 50, "ts": 10_400},
            {"x": 52, "y": 51, "ts": 10_800},
            {"x": 51, "y": 50, "ts": 11_200},
            {"x": 200, "y": 200, "ts": 11_300},
            {"x": 200, "y": 360, "ts": 11_550},
            {"x": 70, "y": 360, "ts": 11_800},
            {"x": 70, "y": 100, "ts": 12_050},
        ],
        "clicks": [
            {"ts": 10_200, "button": 0},
            {"ts": 11_400, "button": 0},
            {"ts": 11_950, "button": 1},
        ],
        "keys": [
            {"code": "KeyA", "ts": 10_100},
            {"code": "KeyS", "ts": 10_350},
            {"code": "KeyD", "ts": 10_600},
            {"code": "KeyF", "ts": 11_000},
            {"code": "Digit1", "ts": 11_500},
            {"code": "Digit2", "ts": 11_650},
        ],
    }

    extractor = FeatureExtractor()
    _print_features("Synthetic: short straight trace (baseline)", batch_straight_line, extractor)
    _print_features("Synthetic: curved path + dwell / micro-moves", batch_curved_with_pause, extractor)
    print("\n--- demo finished (2 batches) ---\n")
