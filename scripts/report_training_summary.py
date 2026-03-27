import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def main():
    base = Path("models/trained")
    model_path = base / "RandomForest.pkl"
    scaler_path = base / "scaler.pkl"
    feature_names_path = base / "feature_names.json"

    features_csv = Path("backend/data/processed/training_data_batches.csv")
    labels_csv = Path("backend/data/raw/labels.csv")

    required = [model_path, scaler_path, feature_names_path, features_csv, labels_csv]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    with feature_names_path.open("r") as f:
        feature_names = json.load(f)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    features_df = pd.read_csv(features_csv)
    labels_df = pd.read_csv(labels_csv)

    if "label" in features_df.columns:
        features_df = features_df.drop(columns=["label"])

    df = features_df.set_index("sessionID").join(labels_df.set_index("sessionID")[["label"]], how="inner").reset_index()
    if len(df) == 0:
        print("No rows after joining features and labels.")
        return

    # Class distribution
    print("=== Class distribution ===")
    dist = df["label"].value_counts(dropna=False)
    for k, v in dist.items():
        print(f"{k}: {v}")

    y_true = df["label"].map({"human": 0, "bot": 1}).astype(int)
    x = df[feature_names].fillna(0.0)
    x_scaled = scaler.transform(x)
    y_pred = model.predict(x_scaled)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("\n=== Confusion matrix (rows=true [human, bot], cols=pred [human, bot]) ===")
    print(cm)

    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, target_names=["human", "bot"]))


if __name__ == "__main__":
    main()
