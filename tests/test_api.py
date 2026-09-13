from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_prediction():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in ["Yes", "No"]
    assert 0 <= data["churn_probability"] <= 1
    assert 0 <= data["churn_percentage"] <= 100
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200

    data = response.json()

    assert "total_predictions" in data
    assert "high_risk_predictions" in data
    assert "medium_risk_predictions" in data
    assert "low_risk_predictions" in data

    assert data["total_predictions"] >= 0
    assert data["high_risk_predictions"] >= 0
    assert data["medium_risk_predictions"] >= 0
    assert data["low_risk_predictions"] >= 0