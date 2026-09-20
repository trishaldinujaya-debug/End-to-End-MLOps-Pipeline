import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from sklearn.model_selection import train_test_split
from src.utils import setup_logger


logger = setup_logger("model_evaluation")

RAW_DATA_PATH = "data/raw/customer_churn.csv"
MODEL_PATH = "models/churn_pipeline.pkl"
TARGET_COLUMN = "Churn"


def evaluate_model(model_path=MODEL_PATH):
    logger.info("Starting model evaluation.")

    df = pd.read_csv(RAW_DATA_PATH)

    logger.info(f"Raw dataset loaded: {df.shape}")

    df = df.drop(columns=["CustomerID"])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logger.info(f"Evaluation dataset: {X_test.shape}")

    model = joblib.load(model_path)

    logger.info(f"Model loaded from: {model_path}")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    logger.info(f"Test ROC-AUC: {roc_auc:.4f}")

    cm = confusion_matrix(y_test, predictions)

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\n===== CONFUSION MATRIX =====")
    print(cm)

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, predictions))

    logger.info("Model evaluation completed.")

    return metrics


if __name__ == "__main__":
    evaluate_model()
