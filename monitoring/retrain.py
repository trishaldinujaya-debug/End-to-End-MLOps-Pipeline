import argparse
from pathlib import Path

import pandas as pd

from monitoring.drift_monitor import (
    REFERENCE_DATA,
    CURRENT_DATA,
    calculate_psi,
)
from src.train import train_model


DRIFT_THRESHOLD = 0.25

CANDIDATE_MODEL = Path(
    "models/churn_model_candidate.pkl"
)


def check_drift(current_data_path):
    """Check whether significant drift exists."""

    reference_df = pd.read_csv(REFERENCE_DATA)
    current_df = pd.read_csv(current_data_path)

    feature_columns = [
        column
        for column in reference_df.columns
        if column != "Churn"
    ]

    max_psi = 0.0

    print("===== DRIFT CHECK =====")

    for column in feature_columns:
        psi = calculate_psi(
            reference_df[column],
            current_df[column],
        )

        print(f"Feature {column}: PSI={psi:.4f}")

        max_psi = max(max_psi, psi)

    print(f"\nMaximum PSI: {max_psi:.4f}")

    return max_psi >= DRIFT_THRESHOLD


def retrain_candidate():
    """Train a new candidate model."""

    print("\n===== CANDIDATE MODEL TRAINING =====")

    train_model(
        model_path=str(CANDIDATE_MODEL)
    )

    if not CANDIDATE_MODEL.exists():
        raise FileNotFoundError(
            "Candidate model was not created."
        )

    print(
        f"Candidate model created: {CANDIDATE_MODEL}"
    )


def promote_candidate():
    """Evaluate and promote the candidate model."""

    print("\n===== MODEL PROMOTION =====")

    from monitoring.model_promotion import (
        main as promotion_main
    )

    promotion_main()


def main():

    parser = argparse.ArgumentParser(
        description="Automated model retraining based on data drift."
    )

    parser.add_argument(
        "--current-data",
        default=str(CURRENT_DATA),
        help="Path to the current dataset.",
    )

    args = parser.parse_args()

    print("===== AUTOMATED RETRAINING CHECK =====")
    print(f"Current data: {args.current_data}")

    drift_detected = check_drift(
        args.current_data
    )

    if drift_detected:

        print("\nSignificant drift detected.")
        print("Starting candidate model retraining...")

        retrain_candidate()

        promote_candidate()

        print(
            "\nAutomated retraining and promotion "
            "workflow completed."
        )

    else:

        print("\nNo significant drift detected.")
        print("Retraining is not required.")


if __name__ == "__main__":
    main()
