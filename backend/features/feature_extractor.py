import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

from .feature_utils import MouseTrajectoryUtils, KeystrokeUtils

logger = logging.getLogger(__name__)

# Features used for cross-session temporal comparison.
# Must all exist in extractBatchFeatures() output.
_TEMPORAL_FEATURES = [
    'keyInterKeyDelayMeanMs', 'keyInterKeyDelayStdMs', 'keystroke_timing_regularity',
    'mousePathEfficiency', 'mouseAngularVelocityStd', 'mouseStdVelocity',
    'mouse_acceleration_variance', 'mouseHoverTimeRatio', 'mouseAvgVelocity',
    'typing_rhythm_autocorrelation',
]

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

        # Session-consistency features (catch human-mimicking bots)
        consistencyFeatures = self.extractConsistencyFeatures(moves, keys)
        features.update(consistencyFeatures)

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

    def extractConsistencyFeatures(self, moves: List[dict], keys: List[dict]) -> Dict[str, float]:
        """
        Extract session-consistency features that catch human-mimicking bots.

        Features:
          keystroke_timing_regularity     — CoV of inter-key delays; uniform bots score low
          typing_rhythm_autocorrelation   — lag-1 autocorr of delays; Gaussian bots score ≈0
          mouse_acceleration_variance     — variance of acceleration; linear bots score ≈0
          mouse_keystroke_correlation     — mouse velocity ratio during keystrokes; bots ≈1.0
          session_phase_consistency       — velocity std change between batch halves
        """
        features: Dict[str, float] = {}

        # ── 1. keystroke_timing_regularity ────────────────────────────────
        # Coefficient of variation (std/mean) of inter-key delays.
        # Uniform bots score near 0 even at human-average speed.
        if len(keys) >= 3:
            delays = np.array(self.utilsKey.interKeyDelays(keys), dtype=float)
            if len(delays) >= 2:
                mean_d = float(np.mean(delays))
                std_d  = float(np.std(delays))
                features['keystroke_timing_regularity'] = float(std_d / max(mean_d, 1.0))
            else:
                features['keystroke_timing_regularity'] = 0.0
        else:
            features['keystroke_timing_regularity'] = 0.0

        # ── 2. typing_rhythm_autocorrelation ──────────────────────────────
        # Lag-1 autocorrelation of inter-key delays.
        # Humans type in rhythmic bursts (positive autocorr);
        # Gaussian bots draw independent samples (autocorr ≈ 0).
        if len(keys) >= 4:
            delays = np.array(self.utilsKey.interKeyDelays(keys), dtype=float)
            if len(delays) >= 3 and np.std(delays) > 0:
                corr = float(np.corrcoef(delays[:-1], delays[1:])[0, 1])
                features['typing_rhythm_autocorrelation'] = corr if not np.isnan(corr) else 0.0
            else:
                features['typing_rhythm_autocorrelation'] = 0.0
        else:
            features['typing_rhythm_autocorrelation'] = 0.0

        # ── 3. mouse_acceleration_variance ────────────────────────────────
        # Variance of instantaneous acceleration along the mouse path.
        # Linear constant-speed bots → ≈0. Humans have natural micro-corrections.
        valid_moves = [m for m in moves if 'ts' in m]
        if len(valid_moves) >= 4:
            ts    = np.array([m['ts'] for m in valid_moves], dtype=float)
            xs    = np.array([m['x']  for m in valid_moves], dtype=float)
            ys    = np.array([m['y']  for m in valid_moves], dtype=float)
            dt    = np.diff(ts)
            dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
            vels  = dists / np.maximum(dt, 1.0) * 1000.0          # px/s
            # acceleration: Δv / Δt using time of second segment
            accels = np.diff(vels) / np.maximum(dt[1:], 1.0) * 1000.0  # px/s²
            features['mouse_acceleration_variance'] = float(np.var(accels))
        else:
            features['mouse_acceleration_variance'] = 0.0

        # ── 4. mouse_keystroke_correlation ────────────────────────────────
        # Ratio: avg mouse velocity near keystroke events / overall avg velocity.
        # Humans slow/stop mouse while typing (ratio < 1.0); bots maintain velocity (≈1.0).
        if len(valid_moves) >= 2 and len(keys) >= 1:
            ts    = np.array([m['ts'] for m in valid_moves], dtype=float)
            xs    = np.array([m['x']  for m in valid_moves], dtype=float)
            ys    = np.array([m['y']  for m in valid_moves], dtype=float)
            dt    = np.diff(ts)
            dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
            vels  = dists / np.maximum(dt, 1.0) * 1000.0
            seg_ts = (ts[:-1] + ts[1:]) / 2.0                    # segment midpoints

            overall_avg = float(np.mean(vels))
            key_ts = np.array([k['ts'] for k in keys], dtype=float)
            key_vels = [vels[int(np.argmin(np.abs(seg_ts - kts)))] for kts in key_ts]
            avg_key_v = float(np.mean(key_vels))
            features['mouse_keystroke_correlation'] = float(avg_key_v / max(overall_avg, 1.0))
        else:
            features['mouse_keystroke_correlation'] = 0.0

        # ── 5. session_phase_consistency ──────────────────────────────────
        # Difference in velocity std between first and second half of mouse moves.
        # Bots that shift behaviour mid-batch (e.g., adaptive bots near phase boundary)
        # show high values; consistently behaving bots show values near 0.
        if len(valid_moves) >= 4:
            ts    = np.array([m['ts'] for m in valid_moves], dtype=float)
            xs    = np.array([m['x']  for m in valid_moves], dtype=float)
            ys    = np.array([m['y']  for m in valid_moves], dtype=float)
            dt    = np.diff(ts)
            dists = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
            vels  = dists / np.maximum(dt, 1.0) * 1000.0
            mid   = len(vels) // 2
            std1  = float(np.std(vels[:mid]))  if mid > 0           else 0.0
            std2  = float(np.std(vels[mid:]))  if len(vels) > mid   else 0.0
            denom = std1 + std2
            features['session_phase_consistency'] = float(abs(std1 - std2) / max(denom, 1.0))
        else:
            features['session_phase_consistency'] = 0.0

        return features

    # ── Session-level temporal methods ────────────────────────────────────────
    # These accept a list of signal dicts (one per batch) for a whole session.
    # They are used by the session blender and /api/session-score.

    def split_session_features(self, batches: List[dict]) -> Tuple[Dict, Dict]:
        """
        Split session batches into first and second half.
        Returns (first_half_avg_features, second_half_avg_features) dicts
        keyed by _TEMPORAL_FEATURES.

        batches: list of signal dicts, each with 'mouseMoves', 'clicks', 'keys'.
        """
        n = len(batches)
        zero = {k: 0.0 for k in _TEMPORAL_FEATURES}
        if n < 2:
            return zero, zero

        mid = n // 2

        def _avg(batch_list: List[dict]) -> Dict:
            if not batch_list:
                return {k: 0.0 for k in _TEMPORAL_FEATURES}
            all_feats = [self.extractBatchFeatures(b) for b in batch_list]
            return {k: float(np.mean([f.get(k, 0.0) for f in all_feats]))
                    for k in _TEMPORAL_FEATURES}

        return _avg(batches[:mid]), _avg(batches[mid:])

    def temporal_drift_score(self, batches: List[dict]) -> float:
        """
        Normalised drift between the first-half and second-half feature vectors.
        Returns [0, 1]; high = large behaviour change (adaptive bot), low = consistent.
        """
        first, second = self.split_session_features(batches)
        f1 = np.array([first[k]  for k in _TEMPORAL_FEATURES], dtype=float)
        f2 = np.array([second[k] for k in _TEMPORAL_FEATURES], dtype=float)
        # Normalise each feature by the larger of |f1|, |f2|, or 1
        scale = np.maximum(np.maximum(np.abs(f1), np.abs(f2)), 1.0)
        f1n = f1 / scale
        f2n = f2 / scale
        dist = float(np.linalg.norm(f1n - f2n) / np.sqrt(len(_TEMPORAL_FEATURES)))
        return float(min(dist, 1.0))

    def early_late_timing_delta(self, batches: List[dict]) -> float:
        """
        Absolute difference in mean inter-key delay between the first 30 % and
        last 30 % of batches. High = bot that changes keystroke speed mid-session.
        """
        n = len(batches)
        if n < 3:
            return 0.0

        cut = max(1, round(n * 0.3))

        def _mean_delay(batch_list: List[dict]) -> float:
            delays: List[float] = []
            for b in batch_list:
                keys = b.get('keys', []) or []
                if len(keys) >= 2:
                    delays.extend(self.utilsKey.interKeyDelays(keys))
            return float(np.mean(delays)) if delays else 0.0

        early = _mean_delay(batches[:cut])
        late  = _mean_delay(batches[n - cut:])
        return float(abs(late - early))

    def extract_network_features(self, network_dict: dict) -> dict:
        """
        Convert a pre-computed network/device enrichment dict to 7 numeric features.

        Arguments:
            network_dict: output of enrichment.enrich_request() **or** a
                          metadata sub-dict stored in signals.jsonl by the
                          seed scripts.  Missing keys default to safe values.

        Returns:
            Dict with exactly 7 keys:
              is_headless_browser, is_known_bot_ua, is_datacenter_ip,
              ua_entropy, has_accept_language, accept_language_count,
              suspicious_header_count
        """
        nd = network_dict or {}
        return {
            "is_headless_browser":     float(bool(nd.get("is_headless_browser", False))),
            "is_known_bot_ua":         float(bool(nd.get("is_known_bot_ua", False))),
            "is_datacenter_ip":        float(bool(nd.get("is_datacenter_ip", False))),
            "ua_entropy":              float(nd.get("ua_entropy", 0.0)),
            "has_accept_language":     float(bool(nd.get("has_accept_language", True))),
            "accept_language_count":   float(nd.get("accept_language_count", 0)),
            "suspicious_header_count": float(nd.get("suspicious_header_count", 0)),
        }

    def behavior_consistency_score(self, batches: List[dict]) -> float:
        """
        Cosine similarity between first-half and second-half feature vectors.
        Returns [0, 1]; high ≈ 1.0 = consistent (human or consistent bot),
        low = behaviour changed mid-session (adaptive bot).
        """
        first, second = self.split_session_features(batches)
        v1 = np.array([first[k]  for k in _TEMPORAL_FEATURES], dtype=float)
        v2 = np.array([second[k] for k in _TEMPORAL_FEATURES], dtype=float)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 1.0
        cosine = float(np.dot(v1, v2) / (n1 * n2))
        return float(max(0.0, min(1.0, cosine)))


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
