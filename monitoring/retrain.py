import argparse
import subprocess

import pandas as pd

from monitoring.drift_monitor import (
    REFERENCE_DATA,
    CURRENT_DATA,
    calculate_psi,
)


DRIFT_THRESHOLD = 0.25


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


def retrain_model():
    """Run the existing training pipeline."""

    print("\n===== MODEL RETRAINING =====")

    result = subprocess.run(
        ["python", "-m", "src.train"],
        check=True,
    )

    return result.returncode == 0


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

    drift_detected = check_drift(args.current_data)

    if drift_detected:
        print("\nSignificant drift detected.")
        print("Starting model retraining...")

        retrain_model()

        print("\nModel retraining completed successfully.")

    else:
        print("\nNo significant drift detected.")
        print("Retraining is not required.")


if __name__ == "__main__":
    main()
