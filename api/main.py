from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app

from api.schemas import CustomerData


MODEL_PATH = Path("models/churn_pipeline.pkl")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="ML API for predicting customer churn.",
    version="1.0.0",
)


# -----------------------------
# Prometheus Metrics
# -----------------------------

prediction_counter = Counter(
    "churn_predictions_total",
    "Total number of churn predictions",
)

prediction_histogram = Histogram(
    "churn_prediction_duration_seconds",
    "Time spent processing churn predictions",
)


# Expose Prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# -----------------------------
# Load Model
# -----------------------------

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    model = None
    print(f"Model loading failed: {error}")


# -----------------------------
# Root Endpoint
# -----------------------------

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "version": "1.0.0",
    }


# -----------------------------
# Health Endpoint
# -----------------------------

@app.get("/health")
def health():
    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False,
        }

    return {
        "status": "healthy",
        "model_loaded": True,
    }


# -----------------------------
# Prediction Endpoint
# -----------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:
        with prediction_histogram.time():

            data = {
                "Age": customer.Age,
                "Gender": customer.Gender,
                "Tenure": customer.Tenure,
                "Usage Frequency": customer.Usage_Frequency,
                "Support Calls": customer.Support_Calls,
                "Payment Delay": customer.Payment_Delay,
                "Subscription Type": customer.Subscription_Type,
                "Contract Length": customer.Contract_Length,
                "Total Spend": customer.Total_Spend,
                "Last Interaction": customer.Last_Interaction,
            }

            input_data = pd.DataFrame([data])

            prediction = model.predict(input_data)[0]

            probability = model.predict_proba(input_data)[0][1]

            prediction_counter.inc()

            return {
                "prediction": int(prediction),
                "churn_probability": round(
                    float(probability),
                    4,
                ),
            }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )