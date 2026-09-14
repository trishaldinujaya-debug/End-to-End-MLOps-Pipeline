from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.utils import setup_logger


logger = setup_logger("mlflow_training")


TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/churn_model.pkl"

EXPERIMENT_NAME = "Customer Churn Prediction"


def train_with_mlflow():

    logger.info("Starting MLflow model training.")

    # Load training data
    train_df = pd.read_csv(TRAIN_PATH)
    logger.info(f"Training dataset loaded: {train_df.shape}")

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    # Load testing data
    test_df = pd.read_csv(TEST_PATH)
    logger.info(f"Testing dataset loaded: {test_df.shape}")

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    # Model parameters
    n_estimators = 200
    max_depth = 12
    random_state = 42

    # Set MLflow experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        logger.info("MLflow run started.")

        # -----------------------------
        # Log project information
        # -----------------------------

        mlflow.set_tag("project", "customer-churn-mlops")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("dataset", "customer_churn")

        # -----------------------------
        # Log parameters
        # -----------------------------

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("features", X_train.shape[1])

        # -----------------------------
        # Train model
        # -----------------------------

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

        logger.info("Training Random Forest model...")

        model.fit(X_train, y_train)

        logger.info("Model training completed.")

        # -----------------------------
        # Training evaluation
        # -----------------------------

        train_predictions = model.predict(X_train)
        train_probabilities = model.predict_proba(X_train)[:, 1]

        train_accuracy = accuracy_score(
            y_train,
            train_predictions
        )

        train_precision = precision_score(
            y_train,
            train_predictions
        )

        train_recall = recall_score(
            y_train,
            train_predictions
        )

        train_f1 = f1_score(
            y_train,
            train_predictions
        )

        train_roc_auc = roc_auc_score(
            y_train,
            train_probabilities
        )

        # -----------------------------
        # Test evaluation
        # -----------------------------

        test_predictions = model.predict(X_test)
        test_probabilities = model.predict_proba(X_test)[:, 1]

        test_accuracy = accuracy_score(
            y_test,
            test_predictions
        )

        test_precision = precision_score(
            y_test,
            test_predictions
        )

        test_recall = recall_score(
            y_test,
            test_predictions
        )

        test_f1 = f1_score(
            y_test,
            test_predictions
        )

        test_roc_auc = roc_auc_score(
            y_test,
            test_probabilities
        )

        # -----------------------------
        # Log training metrics
        # -----------------------------

        mlflow.log_metric(
            "training_accuracy",
            train_accuracy
        )

        mlflow.log_metric(
            "training_precision",
            train_precision
        )

        mlflow.log_metric(
            "training_recall",
            train_recall
        )

        mlflow.log_metric(
            "training_f1",
            train_f1
        )

        mlflow.log_metric(
            "training_roc_auc",
            train_roc_auc
        )

        # -----------------------------
        # Log test metrics
        # -----------------------------

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        mlflow.log_metric(
            "test_precision",
            test_precision
        )

        mlflow.log_metric(
            "test_recall",
            test_recall
        )

        mlflow.log_metric(
            "test_f1",
            test_f1
        )

        mlflow.log_metric(
            "test_roc_auc",
            test_roc_auc
        )

        # -----------------------------
        # Save model
        # -----------------------------

        Path(MODEL_PATH).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            model,
            MODEL_PATH
        )

        logger.info(
            f"Model saved to: {MODEL_PATH}"
        )

        # -----------------------------
        # Log model to MLflow
        # -----------------------------

        mlflow.sklearn.log_model(
            model,
            name="churn_model"
        )

        # -----------------------------
        # Console output
        # -----------------------------

        logger.info(
            f"Training Accuracy: {train_accuracy:.4f}"
        )

        logger.info(
            f"Training Precision: {train_precision:.4f}"
        )

        logger.info(
            f"Training Recall: {train_recall:.4f}"
        )

        logger.info(
            f"Training F1 Score: {train_f1:.4f}"
        )

        logger.info(
            f"Training ROC-AUC: {train_roc_auc:.4f}"
        )

        logger.info(
            f"Test Accuracy: {test_accuracy:.4f}"
        )

        logger.info(
            f"Test Precision: {test_precision:.4f}"
        )

        logger.info(
            f"Test Recall: {test_recall:.4f}"
        )

        logger.info(
            f"Test F1 Score: {test_f1:.4f}"
        )

        logger.info(
            f"Test ROC-AUC: {test_roc_auc:.4f}"
        )

        logger.info(
            "MLflow run completed successfully."
        )


if __name__ == "__main__":
    train_with_mlflow()