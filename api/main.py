import os
import csv
from datetime import datetime

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/best_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
LOG_PATH = "monitoring/predictions.csv"


# ============================================================
# Create required directories
# ============================================================

os.makedirs("monitoring", exist_ok=True)


# ============================================================
# Load trained model and preprocessor
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run src\\train.py first."
    )

if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(
        f"Preprocessor not found at {PREPROCESSOR_PATH}. "
        "Run src\\preprocess.py first."
    )


model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# ============================================================
# Create FastAPI application
# ============================================================

app = FastAPI(
    title="AI Customer Churn Prediction API",
    description="Production-style ML API for predicting customer churn.",
    version="1.0.0"
)


# ============================================================
# Request schema
# ============================================================

class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True,
        "preprocessor_loaded": True,
        "service": "customer-churn-api"
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:

        # Convert request into DataFrame
        input_data = pd.DataFrame([
            customer.model_dump()
        ])

        # Preprocess raw customer data
        processed_input = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(processed_input)[0]

        # Get churn probability
        probability = model.predict_proba(
            processed_input
        )[0][1]

        # Determine risk
        if probability >= 0.70:
            risk = "HIGH"

        elif probability >= 0.40:
            risk = "MEDIUM"

        else:
            risk = "LOW"

        # Convert prediction
        prediction_label = (
            "Yes" if prediction == 1 else "No"
        )

        # Log prediction
        log_prediction(
            customer,
            prediction_label,
            probability,
            risk
        )

        return {
            "prediction": prediction_label,
            "churn_probability": round(
                float(probability),
                4
            ),
            "churn_percentage": round(
                float(probability) * 100,
                2
            ),
            "risk_level": risk
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# Prediction logging
# ============================================================

def log_prediction(
    customer,
    prediction,
    probability,
    risk
):

    file_exists = os.path.exists(LOG_PATH)

    row = customer.model_dump()

    row["prediction"] = prediction
    row["churn_probability"] = probability
    row["risk_level"] = risk
    row["timestamp"] = datetime.now().isoformat()

    with open(
        LOG_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# Monitoring endpoint
# ============================================================

@app.get("/metrics")
def get_metrics():

    if not os.path.exists(LOG_PATH):

        return {
            "total_predictions": 0,
            "high_risk_predictions": 0,
            "medium_risk_predictions": 0,
            "low_risk_predictions": 0
        }

    df = pd.read_csv(LOG_PATH)

    return {
        "total_predictions": len(df),

        "high_risk_predictions": int(
            (df["risk_level"] == "HIGH").sum()
        ),

        "medium_risk_predictions": int(
            (df["risk_level"] == "MEDIUM").sum()
        ),

        "low_risk_predictions": int(
            (df["risk_level"] == "LOW").sum()
        )
    }