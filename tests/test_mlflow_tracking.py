"""
Integration tests for MLflow tracking, artifact logging, and Model Registry.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest

from src.evaluate import get_latest_model_version
from src.pipeline import build_pipeline


@pytest.fixture
def mlflow_tmp_backend(tmp_path):
    """Temporary SQLite MLflow tracking backend."""
    old_uri = mlflow.get_tracking_uri()
    db_path = tmp_path / "mlflow_test.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve().as_posix()}")
    mlflow.set_experiment("test-experiment")
    yield
    if old_uri:
        mlflow.set_tracking_uri(old_uri)



def _tiny_dataset():
    X = pd.DataFrame({
        "num1": [1, 2, 3, 4, 5, 6, 7, 8],
        "cat1": ["a", "b", "a", "b", "a", "b", "a", "b"],
    })
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    return X, y


def test_run_logs_params_and_metrics(mlflow_tmp_backend):
    """Verify run exposes logged parameters and metrics via MLflow client."""
    X, y = _tiny_dataset()
    pipe = build_pipeline(["num1"], ["cat1"], model_type="logreg")

    with mlflow.start_run() as run:
        pipe.fit(X, y)
        mlflow.log_param("model_type", "logreg")
        mlflow.log_metric("dummy_auc", 0.75)

    client = mlflow.tracking.MlflowClient()
    fetched = client.get_run(run.info.run_id)

    assert fetched.data.params["model_type"] == "logreg"
    assert fetched.data.metrics["dummy_auc"] == 0.75
    assert fetched.info.status == "FINISHED"


def test_model_registered_and_version_resolvable(mlflow_tmp_backend):
    """Verify model can be registered, retrieved by latest version, and loaded."""
    X, y = _tiny_dataset()
    pipe = build_pipeline(["num1"], ["cat1"], model_type="logreg")
    pipe.fit(X, y)

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name="TestModel",
            serialization_format="pickle",
        )
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name="TestModel",
            serialization_format="pickle",
        )

    latest_version = get_latest_model_version("TestModel")
    assert latest_version == "2"

    loaded = mlflow.sklearn.load_model(f"models:/TestModel/{latest_version}")
    preds = loaded.predict(X)
    assert len(preds) == len(y)


def test_get_latest_model_version_raises_if_missing(mlflow_tmp_backend):
    with pytest.raises(RuntimeError):
        get_latest_model_version("NonExistentModel")
