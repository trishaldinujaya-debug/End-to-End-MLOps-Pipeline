from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import setup_logger


logger = setup_logger("model_training")


TRAIN_PATH = "data/processed/train.csv"
MODEL_PATH = "models/churn_model.pkl"


def train_model(model_path=MODEL_PATH):
    """Train the customer churn prediction model."""

    logger.info("Starting model training.")

    train_df = pd.read_csv(TRAIN_PATH)

    logger.info(
        f"Training dataset loaded: {train_df.shape}"
    )

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    logger.info(
        f"Training features: {X_train.shape}"
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Training Random Forest model...")

    model.fit(X_train, y_train)

    logger.info("Model training completed.")

    predictions = model.predict(X_train)

    accuracy = accuracy_score(
        y_train,
        predictions
    )

    precision = precision_score(
        y_train,
        predictions
    )

    recall = recall_score(
        y_train,
        predictions
    )

    f1 = f1_score(
        y_train,
        predictions
    )

    logger.info(f"Training Accuracy: {accuracy:.4f}")
    logger.info(f"Training Precision: {precision:.4f}")
    logger.info(f"Training Recall: {recall:.4f}")
    logger.info(f"Training F1 Score: {f1:.4f}")

    Path(model_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        model_path
    )

    logger.info(
        f"Model saved to: {model_path}"
    )

    return model


if __name__ == "__main__":
    train_model()
