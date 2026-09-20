import shutil
from pathlib import Path

from src.evaluate import evaluate_model


PRODUCTION_PIPELINE = Path(
    "models/churn_pipeline.pkl"
)

CANDIDATE_PIPELINE = Path(
    "models/churn_pipeline_candidate.pkl"
)

MIN_F1_SCORE = 0.95


def evaluate_candidate():
    """Evaluate the candidate production pipeline."""

    print(
        "\n===== CANDIDATE PIPELINE EVALUATION ====="
    )

    metrics = evaluate_model(
        model_path=str(CANDIDATE_PIPELINE)
    )

    return metrics


def promote_candidate():
    """Promote the candidate pipeline to production."""

    shutil.copy2(
        CANDIDATE_PIPELINE,
        PRODUCTION_PIPELINE,
    )

    print(
        "\nCandidate pipeline promoted to production."
    )

    print(
        f"Production pipeline: {PRODUCTION_PIPELINE}"
    )


def main():

    print(
        "===== MODEL PROMOTION CHECK ====="
    )

    if not CANDIDATE_PIPELINE.exists():

        raise FileNotFoundError(
            "Candidate pipeline not found: "
            f"{CANDIDATE_PIPELINE}"
        )

    if not PRODUCTION_PIPELINE.exists():

        raise FileNotFoundError(
            "Production pipeline not found: "
            f"{PRODUCTION_PIPELINE}"
        )

    candidate_metrics = evaluate_candidate()

    candidate_f1 = candidate_metrics["f1"]

    print(
        f"\nCandidate F1 Score: "
        f"{candidate_f1:.4f}"
    )

    print(
        f"Minimum required F1: "
        f"{MIN_F1_SCORE:.4f}"
    )

    if candidate_f1 >= MIN_F1_SCORE:

        promote_candidate()

    else:

        print(
            "\nCandidate pipeline did not meet "
            "the minimum quality threshold."
        )

        print(
            "Production pipeline remains unchanged."
        )


if __name__ == "__main__":
    main()