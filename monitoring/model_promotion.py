import shutil
from pathlib import Path

from src.evaluate import evaluate_model


PRODUCTION_PIPELINE = Path(
    "models/churn_pipeline.pkl"
)

CANDIDATE_PIPELINE = Path(
    "models/churn_pipeline_candidate.pkl"
)

PRODUCTION_BACKUP = Path(
    "models/churn_pipeline_backup.pkl"
)

MIN_F1_SCORE = 0.95


def evaluate_pipeline(model_path, label):
    """Evaluate a pipeline on the shared untouched holdout dataset."""

    print(
        f"\n===== {label.upper()} PIPELINE EVALUATION ====="
    )

    metrics = evaluate_model(
        model_path=str(model_path)
    )

    return metrics


def promote_candidate():
    """Backup production and promote the candidate pipeline."""

    print(
        "\n===== MODEL PROMOTION ====="
    )

    if PRODUCTION_PIPELINE.exists():

        shutil.copy2(
            PRODUCTION_PIPELINE,
            PRODUCTION_BACKUP,
        )

        print(
            "Production pipeline backup created:"
        )

        print(
            PRODUCTION_BACKUP
        )

    shutil.copy2(
        CANDIDATE_PIPELINE,
        PRODUCTION_PIPELINE,
    )

    print(
        "\nCandidate pipeline promoted to production."
    )

    print(
        f"Production pipeline: "
        f"{PRODUCTION_PIPELINE}"
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

    production_metrics = evaluate_pipeline(
        PRODUCTION_PIPELINE,
        "Production",
    )

    candidate_metrics = evaluate_pipeline(
        CANDIDATE_PIPELINE,
        "Candidate",
    )

    production_f1 = production_metrics["f1"]
    candidate_f1 = candidate_metrics["f1"]

    print(
        "\n===== MODEL COMPARISON ====="
    )

    print(
        f"Production F1: {production_f1:.4f}"
    )

    print(
        f"Candidate F1 : {candidate_f1:.4f}"
    )

    print(
        f"Minimum required F1: "
        f"{MIN_F1_SCORE:.4f}"
    )

    quality_passed = (
        candidate_f1 >= MIN_F1_SCORE
    )

    no_regression = (
        candidate_f1 >= production_f1
    )

    if quality_passed and no_regression:

        print(
            "\nCandidate passed validation."
        )

        print(
            "Candidate meets the minimum quality "
            "threshold."
        )

        print(
            "Candidate does not regress against "
            "the production model."
        )

        promote_candidate()

    else:

        print(
            "\nCandidate pipeline rejected."
        )

        if not quality_passed:

            print(
                "Reason: Candidate F1 is below "
                "the minimum quality threshold."
            )

        if not no_regression:

            print(
                "Reason: Candidate F1 is lower "
                "than the production F1."
            )

        print(
            "Production pipeline remains unchanged."
        )


if __name__ == "__main__":
    main()