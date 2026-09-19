from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_DATA = Path("data/processed/train.csv")
CURRENT_DATA = Path("data/processed/test.csv")

TARGET_COLUMN = "Churn"


def calculate_psi(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1,
    )

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints,
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints,
    )

    reference_percent = reference_counts / len(reference)
    current_percent = current_counts / len(current)

    reference_percent = np.where(
        reference_percent == 0,
        0.0001,
        reference_percent,
    )

    current_percent = np.where(
        current_percent == 0,
        0.0001,
        current_percent,
    )

    psi = np.sum(
        (current_percent - reference_percent)
        * np.log(current_percent / reference_percent)
    )

    return float(psi)


def interpret_psi(psi):
    if psi < 0.10:
        return "No significant drift"

    if psi < 0.25:
        return "Moderate drift"

    return "Significant drift"


def main():
    print("===== DATA DRIFT MONITOR =====")

    if not REFERENCE_DATA.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {REFERENCE_DATA}"
        )

    if not CURRENT_DATA.exists():
        raise FileNotFoundError(
            f"Current dataset not found: {CURRENT_DATA}"
        )

    reference_df = pd.read_csv(REFERENCE_DATA)
    current_df = pd.read_csv(CURRENT_DATA)

    print(f"Reference data: {reference_df.shape}")
    print(f"Current data:   {current_df.shape}")

    feature_columns = [
        column
        for column in reference_df.columns
        if column != TARGET_COLUMN
    ]

    print(f"Features monitored: {len(feature_columns)}")

    print("\n===== DRIFT RESULTS =====")

    drift_detected = False

    for column in feature_columns:
        psi = calculate_psi(
            reference_df[column],
            current_df[column],
        )

        interpretation = interpret_psi(psi)

        if psi >= 0.25:
            drift_detected = True

        print(
            f"Feature {column}: "
            f"PSI={psi:.4f} -> {interpretation}"
        )

    print("\n===== DRIFT SUMMARY =====")

    if drift_detected:
        print("WARNING: Significant data drift detected.")
    else:
        print("No significant data drift detected.")


if __name__ == "__main__":
    main()
