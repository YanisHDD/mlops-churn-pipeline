"""
FastAPI Microservice for Telco Customer Churn Real-Time Prediction.
"""

import os
from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLOps microservice for serving the Telco Customer Churn scikit-learn pipeline.",
    version="1.0.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
_model = None


def get_model():
    """Lazy load the trained model pipeline."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            return None
        _model = joblib.load(MODEL_PATH)
    return _model


class CustomerInput(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: str = Field(..., example="0")
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: float = Field(..., example=1.0)
    PhoneService: str = Field(..., example="No")
    MultipleLines: str = Field(..., example="No phone service")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=29.85)
    TotalCharges: Optional[float] = Field(None, example=29.85)


class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_code: int
    churn_probability: float
    risk_level: str


@app.get("/")
def root():
    return {
        "service": "Telco Customer Churn Prediction Microservice",
        "status": "active",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health")
def health():
    model = get_model()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Ensure '{MODEL_PATH}' exists by running 'make train'."
        )

    # Convert Pydantic model to single-row DataFrame
    input_df = pd.DataFrame([customer.model_dump()])

    try:
        pred_code = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0][1])

        churn_label = "Yes" if pred_code == 1 else "No"
        risk = "High" if prob >= 0.5 else "Low"

        return PredictionResponse(
            churn_prediction=churn_label,
            churn_code=pred_code,
            churn_probability=round(prob, 4),
            risk_level=risk
        )
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e!s}") from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
