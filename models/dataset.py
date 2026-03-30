import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class ModelDataset:
    """Load, prepare, and split features + labels"""

    def __init__(self, featuresCSV: str, labelsCSV: str):
        self.featuresDF = pd.read_csv(featuresCSV)
        self.labelsDF = pd.read_csv(labelsCSV)

    def prepare(self, testSize: float=0.2, randomState: int = 42) -> Tuple:
        """
        Load features, merge labels, split, and scale.

        Returns: xTrain, xTest, yTrain, yTest, featureNames, scaler
        """


        print("featuresDF columns:", self.featuresDF.columns.tolist())
        print("labelsDF columns:", self.labelsDF.columns.tolist())
        print("featuresDF sessionIDs:", self.featuresDF['sessionID'].unique())
        print("labelsDF sessionIDs:", self.labelsDF['sessionID'].unique())

        #Remove any existing label column from features
        if 'label' in self.featuresDF.columns:
            self.featuresDF = self.featuresDF.drop(columns=['label'])

        #Merge features and labels by sessionID
        features = self.featuresDF.set_index('sessionID')
        labels = self.labelsDF.set_index('sessionID')

        df = features.join(labels[['label']], how='inner').reset_index()
        

        #Drop non-features
        dropCols = ['sessionID', 'timestamp', 'timestampRelativeMs', 'batchCount']
        featureCols = [c for c in df.columns if c not in dropCols + ['label']]

        x = df[featureCols].fillna(0)
        y = df['label'].map({'human': 0, 'bot': 1})     # 0 = human, 1 = bot

        # Split — stratified so class ratios are preserved in both sets
        xTrain, xTest, yTrain, yTest = train_test_split(
            x, y, test_size=testSize, random_state=randomState, stratify=y
        )

        #Scale
        scaler = StandardScaler()
        xTrainScaled = scaler.fit_transform(xTrain)
        xTestScaled = scaler.transform(xTest)

        return xTrainScaled, xTestScaled, yTrain, yTest, list(featureCols), scaler