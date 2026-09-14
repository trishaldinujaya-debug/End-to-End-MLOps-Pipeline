import joblib
import pandas as pd


PIPELINE_PATH = "models/churn_pipeline.pkl"


def test_prediction():

    pipeline = joblib.load(PIPELINE_PATH)

    sample_customer = pd.DataFrame(
        [
            {
                "Age": 35,
                "Gender": "Male",
                "Tenure": 12,
                "Usage Frequency": 20,
                "Support Calls": 3,
                "Payment Delay": 5,
                "Subscription Type": "Standard",
                "Contract Length": "Annual",
                "Total Spend": 1500,
                "Last Interaction": 10,
            }
        ]
    )

    prediction = pipeline.predict(sample_customer)

    probability = pipeline.predict_proba(
        sample_customer
    )[:, 1]

    print("\n===== PIPELINE TEST =====")
    print(f"Prediction: {prediction[0]}")
    print(f"Churn Probability: {probability[0]:.4f}")


if __name__ == "__main__":
    test_prediction()