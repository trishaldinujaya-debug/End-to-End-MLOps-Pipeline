from pathlib import Path
import pandas as pd

from src.utils import setup_logger


logger = setup_logger("data_ingestion")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the raw customer churn dataset.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info(f"Loading dataset from: {path}")

    df = pd.read_csv(path)

    logger.info(f"Dataset loaded successfully")
    logger.info(f"Rows: {df.shape[0]}")
    logger.info(f"Columns: {df.shape[1]}")

    return df


if __name__ == "__main__":

    data_path = "data/raw/customer_churn.csv"

    df = load_data(data_path)

    print("\nDataset Preview:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())