from pathlib import Path

import pandas as pd

from src.utils import setup_logger


logger = setup_logger("data_validation")


REQUIRED_COLUMNS = [
    "CustomerID",
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
    "Churn",
]

NUMERICAL_COLUMNS = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Subscription Type",
    "Contract Length",
]


def validate_data(file_path: str) -> bool:
    """Validate the customer churn dataset."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info(f"Validating dataset: {path}")

    df = pd.read_csv(path)

    # 1. Check dataset is not empty
    if df.empty:
        raise ValueError("Dataset is empty.")

    logger.info(f"Dataset contains {len(df)} rows.")

    # 2. Check required columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    logger.info("Required columns check passed.")

    # 3. Check duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        logger.warning(
            f"Found {duplicate_count} duplicate rows."
        )
    else:
        logger.info("No duplicate rows found.")

    # 4. Check missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()

    if total_missing > 0:
        logger.warning(
            f"Found {total_missing} missing values."
        )
        logger.warning(
            f"Missing values by column:\n{missing_values[missing_values > 0]}"
        )
    else:
        logger.info("No missing values found.")

    # 5. Check numerical columns
    for column in NUMERICAL_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(
                f"Column '{column}' must be numeric."
            )

    logger.info("Numerical column validation passed.")

    # 6. Check categorical columns
    for column in CATEGORICAL_COLUMNS:
        if not pd.api.types.is_object_dtype(df[column]):
            logger.warning(
                f"Column '{column}' is not stored as object type."
            )

    logger.info("Categorical column validation completed.")

    # 7. Check target values
    target_values = set(df["Churn"].dropna().unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Invalid Churn values found: {target_values}"
        )

    logger.info("Target column validation passed.")

    # 8. Basic numerical range checks
    if (df["Age"] < 0).any():
        raise ValueError("Age contains negative values.")

    if (df["Tenure"] < 0).any():
        raise ValueError("Tenure contains negative values.")

    if (df["Total Spend"] < 0).any():
        raise ValueError("Total Spend contains negative values.")

    logger.info("Numerical range validation passed.")

    logger.info("DATA VALIDATION PASSED.")

    return True


if __name__ == "__main__":

    data_path = "data/raw/customer_churn.csv"

    result = validate_data(data_path)

    if result:
        print("\nDATA VALIDATION PASSED")