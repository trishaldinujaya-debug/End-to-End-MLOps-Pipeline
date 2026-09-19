import shutil
from pathlib import Path

from src.evaluate import evaluate_model


PRODUCTION_MODEL = Path("models/churn_model.pkl")
CANDIDATE_MODEL = Path("models/churn_model_candidate.pkl")

MIN_F1_SCORE = 0.95


def evaluate_candidate():
    """Evaluate the candidate model."""

    print("\n===== CANDIDATE MODEL EVALUATION =====")

    metrics = evaluate_model(
        model_path=str(CANDIDATE_MODEL)
    )

    return metrics


def promote_candidate():
    """Replace the production model with the candidate model."""

    shutil.copy2(
        CANDIDATE_MODEL,
        PRODUCTION_MODEL,
    )

    print("\nCandidate model promoted to production.")


def main():

    print("===== MODEL PROMOTION CHECK =====")

    if not CANDIDATE_MODEL.exists():
        raise FileNotFoundError(
            f"Candidate model not found: {CANDIDATE_MODEL}"
        )

    if not PRODUCTION_MODEL.exists():
        raise FileNotFoundError(
            f"Production model not found: {PRODUCTION_MODEL}"
        )

    candidate_metrics = evaluate_candidate()

    candidate_f1 = candidate_metrics["f1"]

    print(
        f"\nCandidate F1 Score: {candidate_f1:.4f}"
    )

    print(
        f"Minimum required F1: {MIN_F1_SCORE:.4f}"
    )

    if candidate_f1 >= MIN_F1_SCORE:

        promote_candidate()

    else:

        print(
            "\nCandidate model did not meet "
            "the minimum quality threshold."
        )

        print(
            "Production model remains unchanged."
        )


if __name__ == "__main__":
    main()
