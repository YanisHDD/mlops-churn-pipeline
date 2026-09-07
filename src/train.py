"""
Model training script with scikit-learn Pipeline, GridSearchCV, and MLflow tracking.
"""

import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.pipeline import build_pipeline
from src.utils import load_config, load_data, split_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train customer churn classifier with MLflow")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # MLflow configuration
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", cfg.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    )
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", cfg.get("mlflow", {}).get("experiment_name", "churn-exp")
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(log_models=False)

    print(f"MLflow Tracking URI : {tracking_uri}")
    print(f"MLflow Experiment   : {experiment_name}")

    # Load and split data
    df = load_data(cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    numeric = cfg["features"]["numeric"]
    categorical = cfg["features"]["categorical"]
    model_type = cfg["model"]["type"]

    pipe = build_pipeline(numeric, categorical, model_type=model_type)
    param_grid = {f"model__{k}": v for k, v in cfg["model"]["params"].items()}

    cv = StratifiedKFold(
        n_splits=cfg["cv"]["n_splits"],
        shuffle=True,
        random_state=cfg["data"]["random_state"]
    )

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=cfg["cv"]["scoring"],
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    run_name = f"{model_type}-gridsearch"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Running GridSearchCV for {model_type}...")
        grid.fit(X_train, y_train)

        best_score = grid.best_score_
        test_score = grid.score(X_test, y_test)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("cv_best_score", best_score)
        mlflow.log_metric("test_score", test_score)

        registered_name = cfg.get("mlflow", {}).get("registered_model_name", "ChurnClassifier")
        mlflow.sklearn.log_model(
            grid.best_estimator_,
            artifact_path="model",
            registered_model_name=registered_name,
            serialization_format="pickle",
        )

        # Also save locally for FastAPI microservice
        output_dir = cfg.get("artifacts", {}).get("output_dir", "artifacts")
        model_filename = cfg.get("artifacts", {}).get("model_filename", "model.joblib")
        os.makedirs(output_dir, exist_ok=True)
        local_model_path = os.path.join(output_dir, model_filename)
        joblib.dump(grid.best_estimator_, local_model_path)

        print(f"Best params: {grid.best_params_}")
        print(f"CV best score ({cfg['cv']['scoring']}): {best_score:.4f}")
        print(f"Test score: {test_score:.4f}")
        print(f"Model saved locally to {local_model_path}")
        print(f"MLflow Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
