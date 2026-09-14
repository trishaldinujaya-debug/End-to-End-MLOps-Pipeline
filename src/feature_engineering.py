from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import setup_logger


logger = setup_logger("feature_engineering")


TARGET_COLUMN = "Churn"

NUMERICAL_FEATURES = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Subscription Type",
    "Contract Length",
]


def create_preprocessor():
    """Create the preprocessing pipeline."""

    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor


def engineer_features(
    input_path: str,
    processed_directory: str,
    preprocessor_path: str,
):
    """Prepare customer churn data for machine learning."""

    logger.info("Starting feature engineering.")

    # Load raw data
    df = pd.read_csv(input_path)

    logger.info(
        f"Loaded raw dataset with shape: {df.shape}"
    )

    # Remove identifier
    df = df.drop(columns=["CustomerID"])

    logger.info("Removed CustomerID.")

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split BEFORE fitting preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logger.info(
        f"Raw training features: {X_train.shape}"
    )

    logger.info(
        f"Raw testing features: {X_test.shape}"
    )

    # Create preprocessor
    preprocessor = create_preprocessor()

    # Fit ONLY on training data
    logger.info(
        "Fitting preprocessor on training data only."
    )

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    # Transform test data using training preprocessor
    X_test_processed = preprocessor.transform(
        X_test
    )

    logger.info(
        f"Processed training data: {X_train_processed.shape}"
    )

    logger.info(
        f"Processed testing data: {X_test_processed.shape}"
    )

    # Create output directory
    processed_path = Path(processed_directory)
    processed_path.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed datasets
    train_data = pd.DataFrame(
        X_train_processed
    )
    train_data["Churn"] = y_train.reset_index(
        drop=True
    )

    test_data = pd.DataFrame(
        X_test_processed
    )
    test_data["Churn"] = y_test.reset_index(
        drop=True
    )

    train_path = processed_path / "train.csv"
    test_path = processed_path / "test.csv"

    train_data.to_csv(
        train_path,
        index=False
    )

    test_data.to_csv(
        test_path,
        index=False
    )

    # Save fitted preprocessor
    preprocessor_directory = Path(
        preprocessor_path
    ).parent

    preprocessor_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    logger.info(
        f"Training data saved: {train_path}"
    )

    logger.info(
        f"Testing data saved: {test_path}"
    )

    logger.info(
        f"Preprocessor saved: {preprocessor_path}"
    )

    logger.info(
        "Feature engineering completed successfully."
    )


if __name__ == "__main__":

    engineer_features(
        input_path="data/raw/customer_churn.csv",
        processed_directory="data/processed",
        preprocessor_path="models/preprocessor.pkl",
    )