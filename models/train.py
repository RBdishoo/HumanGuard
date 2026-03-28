import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Train and evaluate bot detection models.
    """

    def __init__(self, outputDir: str = 'models/trained'):
        self.outputDir = Path(outputDir)
        self.outputDir.mkdir(parents=True, exist_ok=True)

    def trainRandomForest(self, xTrain, xTest, yTrain, yTest, featureNames): 
        """
        
        Train randomforest
        n_estimators = number of trees in the forest. more trees -> more stability / accuracy and increases training/prediction time. 100 is default starting point

        random_state = computer shuffling is "random" however it is fake randomness that is controlled by a seed val (42). Computer makes the same random decisions (picks the same fows from the data, builds trees with the same random choices). Hence If someone runs the code a different time, it'd achieve identical results instead of slightly diff ones (making the experiment easier to compare and debug)

        n_jobs = number of CPU cores used in parallel to train and predict with all those trees. -1 = using all available cores, to speed things up compared to 1 (single-threaded)
        """

        print("\n Training RandomForest")

        rf = RandomForestClassifier(
            n_estimators=50, max_depth=4, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        rf.fit(xTrain, yTrain)

        yPrediction = rf.predict(xTest)
        yPredictionProbability = rf.predict_proba(xTest)[:, 1]

        metrics = {
            'model': 'RandomForest',
            'accuracy': float(accuracy_score(yTest, yPrediction)),
            'precision': float(precision_score(yTest, yPrediction)),
            'recall': float(recall_score(yTest, yPrediction)),
            'f1': float(f1_score(yTest, yPrediction)),
            'roc_auc': float(roc_auc_score(yTest, yPredictionProbability)),
        }

        #Feature importance
        featureImportance = pd.DataFrame({
            'feature': featureNames,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        return rf, metrics, featureImportance
    
    def trainXGBoost(self, xTrain, xTest, yTrain, yTest, featureNames):
        """
        Train XGBoost - ensemble of decision trees built sequentially, where each new tree tries to correct the errors of previous trees using gradient boosting.
            -> after ecah boosting round, it fits a new decision tree to current gradient of loss function.
            -> prediction of new trees are scaled by learnning rate and added to existing ensemble.
            -> there's built-in regularization to keep trees small and avoid overfitting.
                    -> Overfitting - when a model learns the training data so well that it performs great on training data but poorly on new data because it fails to generalize.

        numEstimators = number of trees (boosting rounds) in the ensemble
            -> more trees mean higher capacity and better performance but eventually more training time and risk of overfitting

        maxDepth = maximum depth of each individual tree


        """

        print("\n Training XGBoost")
        xgbModel = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
        xgbModel.fit(xTrain, yTrain)

        yPrediction = xgbModel.predict(xTest)
        yPredictionProbability = xgbModel.predict_proba(xTest)[:, 1]

        metrics = {
            'model': 'XGBoost',
            'accuracy': float(accuracy_score(yTest, yPrediction)),
            'precision': float(precision_score(yTest, yPrediction)),
            'recall': float(recall_score(yTest, yPrediction)),
            'f1': float(f1_score(yTest, yPrediction)),
            'roc_auc': float(roc_auc_score(yTest, yPredictionProbability)),
        }

        featureImportance = pd.DataFrame({
            'feature': featureNames,
            'importance': xgbModel.feature_importances_
        }).sort_values('importance', ascending=False)

        return xgbModel, metrics, featureImportance
    
    def trainLogisticRegression(self, xTrain, xTest, yTrain, yTest, featureNames):
        """
        Train Logistic Regression — a linear model that learns a weighted
        combination of features to output a probability via the sigmoid function.

        max_iter: maximum iterations for the solver to converge.
        C: inverse of regularization strength (smaller = stronger regularization).
        """

        print("\n Training LogisticRegression")

        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr.fit(xTrain, yTrain)

        yPrediction = lr.predict(xTest)
        yPredictionProbability = lr.predict_proba(xTest)[:, 1]

        metrics = {
            'model': 'LogisticRegression',
            'accuracy': float(accuracy_score(yTest, yPrediction)),
            'precision': float(precision_score(yTest, yPrediction)),
            'recall': float(recall_score(yTest, yPrediction)),
            'f1': float(f1_score(yTest, yPrediction)),
            'roc_auc': float(roc_auc_score(yTest, yPredictionProbability)),
        }

        # Coefficient magnitudes as feature importance proxy
        featureImportance = pd.DataFrame({
            'feature': featureNames,
            'importance': np.abs(lr.coef_[0])
        }).sort_values('importance', ascending=False)

        return lr, metrics, featureImportance

    def saveModel(self, model, name: str):
        """Save model with joblib."""
        import joblib
        path = self.outputDir / f"{name}.pkl"
        joblib.dump(model, path)
        print(f"Saved {name} to {path}")
        return path
    
    def saveMetrics(self, metricsList: list, outputFile: str = 'metrics.json'):
        """Save all metrics to JSON"""

        path = self.outputDir / outputFile
        with open(path, 'w') as f: 
            json.dump(metricsList, f, indent=2)
        print(f"Saved metrics to {path}")
        

