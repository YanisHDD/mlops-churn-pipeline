"""
Unit tests for the scikit-learn Pipeline and preprocessing ColumnTransformer.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import build_pipeline


@pytest.fixture
def sample_data():
    numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical = ["gender", "SeniorCitizen", "Contract"]
    df = pd.DataFrame({
        "tenure": [1.0, 12.0, 24.0, 36.0, 48.0],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70],
        "TotalCharges": [29.85, 1889.50, 108.15, 1840.75, 151.65],
        "gender": ["Female", "Male", "Male", "Female", "Female"],
        "SeniorCitizen": ["0", "0", "0", "1", "0"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year", "Month-to-month"],
    })
    y = pd.Series([0, 0, 1, 0, 1])
    return df, y, numeric, categorical


def test_build_pipeline_structure():
    """Verify that build_pipeline creates the expected named steps."""
    pipe = build_pipeline(["num1"], ["cat1"], model_type="logreg")
    steps = dict(pipe.named_steps)
    assert "pre" in steps
    assert "model" in steps

    preprocessor = steps["pre"]
    transformers = [t[0] for t in preprocessor.transformers]
    assert "num" in transformers
    assert "cat" in transformers


def test_pipeline_fit_predict(sample_data):
    """Verify end-to-end fit and predict on a clean dataset."""
    df, y, numeric, categorical = sample_data
    pipe = build_pipeline(numeric, categorical, model_type="logreg")

    pipe.fit(df, y)
    preds = pipe.predict(df)
    probs = pipe.predict_proba(df)

    assert len(preds) == len(df)
    assert probs.shape == (len(df), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_pipeline_handles_missing_values(sample_data):
    """Verify that imputation handles NaNs in both numeric and categorical features."""
    df, y, numeric, categorical = sample_data

    # Inject NaNs
    df_missing = df.copy()
    df_missing.loc[0, "TotalCharges"] = np.nan
    df_missing.loc[1, "gender"] = None

    pipe = build_pipeline(numeric, categorical, model_type="logreg")
    pipe.fit(df_missing, y)

    preds = pipe.predict(df_missing)
    assert len(preds) == len(df_missing)
    assert not np.isnan(preds).any()


def test_unsupported_model_type():
    """Verify that unsupported model types raise a ValueError."""
    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_pipeline(["a"], ["b"], model_type="svm_unknown")
