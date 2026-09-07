"""
Model training script with scikit-learn Pipeline, GridSearchCV, and MLflow tracking.
"""

import argparse
import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.pipeline import build_pipeline
from src.utils import load_and_preprocess_data, load_config


def train(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)

    # 1. Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", config["mlflow"]["experiment_name"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"MLflow Tracking URI : {tracking_uri}")
    print(f"MLflow Experiment   : {experiment_name}")

    # 2. Load & Preprocess Data
    data_cfg = config["data"]
    feat_cfg = config["features"]
    print(f"Loading data from {data_cfg['csv_path']}...")
    X, y = load_and_preprocess_data(data_cfg["csv_path"], target_col=data_cfg["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y
    )

    # Save test set for evaluate.py
    os.makedirs("data/processed", exist_ok=True)
    test_df = X_test.copy()
    test_df[data_cfg["target"]] = y_test
    test_df.to_csv("data/processed/test.csv", index=False)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 3. Model & Pipeline configuration
    model_type = config["model"]["type"]
    param_grid = config["model"]["hyperparameters"][model_type]
    pipe = build_pipeline(
        numeric_features=feat_cfg["numeric"],
        categorical_features=feat_cfg["categorical"],
        model_type=model_type
    )

    cv_strategy = StratifiedKFold(
        n_splits=config["cv"]["n_splits"],
        shuffle=True,
        random_state=data_cfg["random_state"]
    )

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=config["cv"]["scoring"],
        n_jobs=-1,
        verbose=1
    )

    # 4. Train with MLflow tracking
    with mlflow.start_run(run_name=f"train-{model_type}") as run:
        print(f"Started MLflow run: {run.info.run_id}")

        # Log parameters
        mlflow.log_params({
            "model_type": model_type,
            "test_size": data_cfg["test_size"],
            "cv_splits": config["cv"]["n_splits"],
            "scoring_metric": config["cv"]["scoring"],
            "num_numeric_features": len(feat_cfg["numeric"]),
            "num_categorical_features": len(feat_cfg["categorical"]),
        })

        # Fit GridSearch
        grid_search.fit(X_train, y_train)

        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

        print(f"Best CV {config['cv']['scoring']}: {best_cv_score:.4f}")
        print(f"Best Params: {best_params}")

        # Log best hyperparams
        for p_name, p_val in best_params.items():
            mlflow.log_param(f"best_{p_name}", str(p_val))
        mlflow.log_metric("best_cv_roc_auc", best_cv_score)

        # Evaluate on Test Set
        y_pred = best_pipeline.predict(X_test)
        y_prob = best_pipeline.predict_proba(X_test)[:, 1]

        test_metrics = {
            "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_f1": float(f1_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred)),
            "test_recall": float(recall_score(y_test, y_pred)),
        }

        print("\n--- Test Set Results ---")
        for k, v in test_metrics.items():
            print(f"{k}: {v:.4f}")
            mlflow.log_metric(k, v)

        # Model signature and logging
        sample_input = X_train.head(5)
        signature = infer_signature(sample_input, best_pipeline.predict(sample_input))

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path=config["mlflow"]["artifact_path"],
            signature=signature,
            input_example=sample_input
        )

        # Save local model artifact for FastAPI microservice
        output_dir = config["artifacts"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, config["artifacts"]["model_filename"])
        joblib.dump(best_pipeline, model_path)
        print(f"\nModel saved locally to {model_path}")

        print(f"MLflow Run finished successfully. Run ID: {run.info.run_id}")
        return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train customer churn classifier with MLflow")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    train(args.config)
