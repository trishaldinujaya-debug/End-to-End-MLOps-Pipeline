import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import setup_logger


logger = setup_logger("pipeline_builder")


RAW_DATA_PATH = "data/raw/customer_churn.csv"
PIPELINE_PATH = "models/churn_pipeline.pkl"

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


def build_pipeline():

    logger.info("Starting production pipeline creation.")

    df = pd.read_csv(RAW_DATA_PATH)

    logger.info(
        f"Raw dataset loaded: {df.shape}"
    )

    # Remove identifier
    df = df.drop(columns=["CustomerID"])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Preprocessing
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
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES
            ),
        ]
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    # Complete pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    logger.info(
        "Training complete preprocessing + model pipeline."
    )

    pipeline.fit(X, y)

    joblib.dump(
        pipeline,
        PIPELINE_PATH
    )

    logger.info(
        f"Production pipeline saved to: {PIPELINE_PATH}"
    )

    print("\n===== PRODUCTION PIPELINE =====")
    print("Pipeline created successfully.")
    print(f"Saved to: {PIPELINE_PATH}")
    print("\nPipeline steps:")

    for name, step in pipeline.named_steps.items():
        print(f"- {name}: {type(step).__name__}")


if __name__ == "__main__":
    build_pipeline()