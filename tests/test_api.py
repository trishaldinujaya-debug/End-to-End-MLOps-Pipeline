from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict():
    payload = {
        "Age": 35,
        "Gender": "Male",
        "Tenure": 12,
        "Usage_Frequency": 20,
        "Support_Calls": 2,
        "Payment_Delay": 5,
        "Subscription_Type": "Standard",
        "Contract_Length": "Annual",
        "Total_Spend": 1500.0,
        "Last_Interaction": 10,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "churn_probability" in result

    assert result["prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1


def test_predict_invalid_data():
    payload = {
        "Age": "invalid",
        "Gender": "Male",
        "Tenure": 12,
        "Usage_Frequency": 20,
        "Support_Calls": 2,
        "Payment_Delay": 5,
        "Subscription_Type": "Standard",
        "Contract_Length": "Annual",
        "Total_Spend": 1500.0,
        "Last_Interaction": 10,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422