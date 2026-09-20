import argparse
from pathlib import Path

import joblib
import pandas as pd

from monitoring.drift_monitor import (
    REFERENCE_DATA,
    CURRENT_DATA,
    calculate_psi,
)

from src.build_pipeline import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)


DRIFT_THRESHOLD = 0.25

CANDIDATE_PIPELINE = Path(
    "models/churn_pipeline_candidate.pkl"
)


def check_drift(current_data_path):
    """Check whether significant data drift exists."""

    reference_df = pd.read_csv(
        REFERENCE_DATA
    )

    current_df = pd.read_csv(
        current_data_path
    )

    feature_columns = [
        column
        for column in reference_df.columns
        if column != TARGET_COLUMN
    ]

    max_psi = 0.0

    print("===== DRIFT CHECK =====")

    for column in feature_columns:

        psi = calculate_psi(
            reference_df[column],
            current_df[column],
        )

        print(
            f"Feature {column}: PSI={psi:.4f}"
        )

        max_psi = max(
            max_psi,
            psi
        )

    print(
        f"\nMaximum PSI: {max_psi:.4f}"
    )

    return max_psi >= DRIFT_THRESHOLD


def build_candidate_pipeline():
    """Build a candidate preprocessing + model pipeline."""

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        OneHotEncoder,
        StandardScaler,
    )
    from sklearn.model_selection import train_test_split

    print(
        "\n===== CANDIDATE PIPELINE TRAINING ====="
    )

    raw_data = pd.read_csv(
        "data/raw/customer_churn.csv"
    )

    print(
        f"Raw dataset loaded: {raw_data.shape}"
    )

    raw_data = raw_data.drop(
        columns=["CustomerID"]
    )

    X = raw_data.drop(
        columns=[TARGET_COLUMN]
    )

    y = raw_data[TARGET_COLUMN]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(
        f"Candidate training data: {X_train.shape}"
    )

    print(
        f"Untouched holdout data: {X_holdout.shape}"
    )

    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
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

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    print(
        "Training candidate pipeline on training data only..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    CANDIDATE_PIPELINE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        CANDIDATE_PIPELINE,
    )

    if not CANDIDATE_PIPELINE.exists():
        raise FileNotFoundError(
            "Candidate pipeline was not created."
        )

    print(
        "Candidate production pipeline created:"
    )

    print(
        CANDIDATE_PIPELINE
    )

    return X_holdout, y_holdout


def promote_candidate():
    """Evaluate and promote the candidate pipeline."""

    print(
        "\n===== MODEL PROMOTION ====="
    )

    from monitoring.model_promotion import (
        main as promotion_main,
    )

    promotion_main()


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Automated model retraining "
            "based on data drift."
        )
    )

    parser.add_argument(
        "--current-data",
        default=str(CURRENT_DATA),
        help="Path to the current dataset.",
    )

    args = parser.parse_args()

    print(
        "===== AUTOMATED RETRAINING CHECK ====="
    )

    print(
        f"Current data: {args.current_data}"
    )

    drift_detected = check_drift(
        args.current_data
    )

    if drift_detected:

        print(
            "\nSignificant drift detected."
        )

        print(
            "Starting candidate pipeline retraining..."
        )

        build_candidate_pipeline()

        promote_candidate()

        print(
            "\nAutomated retraining and promotion "
            "workflow completed."
        )

    else:

        print(
            "\nNo significant drift detected."
        )

        print(
            "Retraining is not required."
        )


if __name__ == "__main__":
    main()
