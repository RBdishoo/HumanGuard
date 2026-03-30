import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# ── Feature noise configuration ───────────────────────────────────────────
# Applied to xTrain only when add_noise=True.
# Velocity features are in px/s (±3px over a ~100ms step ≈ ±30 px/s).
# Timing features are in ms (±5ms).
# Dimensionless ratios and counts get proportionally small perturbations.

_TIMING_FEATURES = {
    'batchDurationMs', 'clickIntervalMeanMs', 'clickIntervalStdMs',
    'clickIntervalMinMs', 'clickIntervalMaxMs',
    'keyInterKeyDelayMeanMs', 'keyInterKeyDelayStdMs', 'mouseAvgPauseDurationMs',
}
_COORD_FEATURES = {
    'mouseAvgVelocity', 'mouseStdVelocity', 'mouseMaxVelocity',
    'mouseMoveCount', 'batch_event_count',
}
_RATIO_FEATURES = {
    'mousePathEfficiency', 'mouseHoverTimeRatio', 'mouseHoverFrequency',
    'mouseAngularVelocityStd', 'clickToMoveRatio', 'keyToMoveRatio',
}
_RATE_FEATURES = {
    'clickRatePerSec', 'keyRatePerSec', 'eventRatePerSec',
}
_KEY_FEATURES = {
    'keyCount', 'keyEntropy', 'keyRapidPresses',
}

_NOISE_SIGMA: dict = {
    **{f: 30.0  for f in _COORD_FEATURES},   # px/s (≡ ±3px over 100ms)
    **{f: 5.0   for f in _TIMING_FEATURES},   # ms
    **{f: 0.02  for f in _RATIO_FEATURES},    # dimensionless
    **{f: 0.20  for f in _RATE_FEATURES},     # events/s
    **{f: 0.20  for f in _KEY_FEATURES},      # counts / bits
}


def _add_training_noise(xTrain: pd.DataFrame, featureNames: list,
                        randomState: int) -> pd.DataFrame:
    """Return a copy of xTrain with per-feature Gaussian noise added."""
    rng = np.random.RandomState(randomState)
    x = xTrain.copy()
    for i, name in enumerate(featureNames):
        sigma = _NOISE_SIGMA.get(name, 0.0)
        if sigma > 0:
            x.iloc[:, i] += rng.normal(0.0, sigma, size=len(x))
    return x


class ModelDataset:
    """Load, prepare, and split features + labels."""

    def __init__(self, featuresCSV: str, labelsCSV: str):
        self.featuresDF = pd.read_csv(featuresCSV)
        self.labelsDF   = pd.read_csv(labelsCSV)

    def _build_merged(self) -> Tuple[pd.DataFrame, list]:
        """Merge features + labels, return (df, featureCols)."""
        featuresDF = self.featuresDF.copy()
        if 'label' in featuresDF.columns:
            featuresDF = featuresDF.drop(columns=['label'])

        features = featuresDF.set_index('sessionID')
        labels   = self.labelsDF.set_index('sessionID')
        df = features.join(labels[['label']], how='inner').reset_index()

        dropCols   = ['sessionID', 'timestamp', 'timestampRelativeMs', 'batchCount']
        featureCols = [c for c in df.columns if c not in dropCols + ['label']]
        return df, featureCols

    def get_raw_dataset(self) -> Tuple[pd.DataFrame, pd.Series, list]:
        """
        Return (X, y, featureNames) without splitting or scaling.
        Used for cross-validation with pipeline-internal scaling.
        """
        df, featureCols = self._build_merged()
        X = df[featureCols].fillna(0)
        y = df['label'].map({'human': 0, 'bot': 1})
        return X, y, featureCols

    def prepare(self, testSize: float = 0.2, randomState: int = 42,
                add_noise: bool = False) -> Tuple:
        """
        Merge labels, stratified-split, optionally add training noise, scale.

        Returns: xTrain, xTest, yTrain, yTest, featureNames, scaler
        """
        df, featureCols = self._build_merged()
        x = df[featureCols].fillna(0)
        y = df['label'].map({'human': 0, 'bot': 1})

        # Stratified split — preserves class ratios in both sets
        xTrain, xTest, yTrain, yTest = train_test_split(
            x, y, test_size=testSize, random_state=randomState, stratify=y
        )

        # Gaussian noise augmentation on training set only
        if add_noise:
            xTrain = _add_training_noise(xTrain, featureCols, randomState)

        scaler       = StandardScaler()
        xTrainScaled = scaler.fit_transform(xTrain)
        xTestScaled  = scaler.transform(xTest)

        return xTrainScaled, xTestScaled, yTrain, yTest, list(featureCols), scaler
