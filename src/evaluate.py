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

from src.utils import setup_logger


logger = setup_logger("model_evaluation")

TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/churn_model.pkl"


def evaluate_model(model_path=MODEL_PATH):

    logger.info("Starting model evaluation.")

    test_df = pd.read_csv(TEST_PATH)
    logger.info(f"Test dataset loaded: {test_df.shape}")

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

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
