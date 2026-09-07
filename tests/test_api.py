"""
Unit tests for the FastAPI microservice endpoints.
"""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import numpy as np

from app import app


client = TestClient(app)


def test_root_endpoint():
    """Verify that root endpoint responds with status 200."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "docs_url" in data


def test_health_endpoint():
    """Verify healthcheck endpoint response format."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_endpoint_mocked(monkeypatch):
    """Verify /predict endpoint with a mocked scikit-learn model."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])

    monkeypatch.setattr("app.get_model", lambda: mock_model)

    payload = {
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1.0,
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
    assert data["churn_prediction"] == "Yes"
    assert data["churn_code"] == 1
    assert data["churn_probability"] == 0.85
    assert data["risk_level"] == "High"
