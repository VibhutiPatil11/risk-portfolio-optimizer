import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from ensemble import ENSEMBLE_FEATURE_ORDER, SimpleEnsembleModel


def load_csv_data(path: str, target_column: str = "target", label_column: str = "label"):
    df = pd.read_csv(path)
    required = ENSEMBLE_FEATURE_ORDER + [target_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {', '.join(missing)}")

    X = df[ENSEMBLE_FEATURE_ORDER]
    y = df[target_column]
    labels = df[label_column] if label_column in df.columns else None
    return X, y, labels


def main():
    parser = argparse.ArgumentParser(description="Train the ensemble model (regression + classification)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to training CSV file")
    parser.add_argument("--target", type=str, default="target", help="Regression target column name")
    parser.add_argument("--label", type=str, default="label", help="Classification label column name (BUY/HOLD/SELL)")
    parser.add_argument("--output", type=str, default="ensemble_model.joblib", help="Output model filename")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for train/test split")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    X, y, labels = load_csv_data(str(dataset_path), target_column=args.target, label_column=args.label)

    if labels is not None:
        X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
            X, y, labels, test_size=args.test_size, random_state=args.random_state, stratify=labels
        )
    else:
        print(f"Warning: no '{args.label}' column found -- training regression only, no classifier/SHAP-ready model.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state
        )
        labels_train = labels_test = None

    model = SimpleEnsembleModel(model_dir=str(dataset_path.parent))
    model.fit(X_train, y_train, labels=labels_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Regression test MSE: {mse:.4f}")

    if labels_test is not None:
        proba = model.predict_proba(X_test)
        from ensemble import CLASS_LABELS
        predicted_labels = [CLASS_LABELS[i] for i in np.argmax(proba, axis=1)]
        acc = accuracy_score(labels_test, predicted_labels)
        print(f"Classification accuracy: {acc:.4f}")
        print(classification_report(labels_test, predicted_labels, zero_division=0))

    model_path = model.save(filename=args.output)
    print(f"Saved ensemble model to: {model_path}")


if __name__ == "__main__":
    main()
