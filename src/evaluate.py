"""
Model evaluation script with MLflow tracking and graphical artifacts.
"""

import argparse
import os
import shutil
import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, roc_auc_score

from src.utils import (
    load_config,
    load_data,
    plot_confusion_matrix,
    plot_pr,
    plot_roc,
    split_data,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model from MLflow Model Registry")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Version in model registry to evaluate. Defaults to latest version.",
    )
    return parser.parse_args()


def get_latest_model_version(model_name: str) -> str:
    """Return latest registered model version number."""
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(
            f"No versions found for model '{model_name}'. Run 'python -m src.train' first."
        )
    latest = max(versions, key=lambda v: int(v.version))
    return str(latest.version)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", cfg.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")
    )
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME", cfg.get("mlflow", {}).get("experiment_name", "churn-exp")
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    model_name = cfg.get("mlflow", {}).get("registered_model_name", "ChurnClassifier")
    version = args.model_version or get_latest_model_version(model_name)
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded model: {model_uri}")

    df = load_data(cfg)
    _, X_test, _, y_test = split_data(df, cfg)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    print(f"ROC-AUC (test): {auc:.4f}")
    print(report)

    output_dir = cfg.get("artifacts", {}).get("output_dir", "artifacts")
    os.makedirs(output_dir, exist_ok=True)

    with mlflow.start_run(run_name=f"evaluate-{model_name}-v{version}"):
        mlflow.log_param("evaluated_model_uri", model_uri)
        mlflow.log_metric("eval_roc_auc", auc)

        roc_path = plot_roc(model, X_test, y_test, out_path="roc_curve.png")
        pr_path = plot_pr(model, X_test, y_test, out_path="pr_curve.png")
        cm_path = plot_confusion_matrix(model, X_test, y_test, out_path="confusion_matrix.png")

        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(pr_path)
        mlflow.log_artifact(cm_path)

        preds_df = X_test.copy()
        preds_df["y_true"] = y_test.values
        preds_df["y_proba"] = y_proba
        preds_path = "predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        mlflow.log_artifact(preds_path)

        # Copy to artifacts directory as well
        shutil.copy(roc_path, os.path.join(output_dir, "roc_curve.png"))
        shutil.copy(pr_path, os.path.join(output_dir, "pr_curve.png"))
        shutil.copy(cm_path, os.path.join(output_dir, "confusion_matrix.png"))
        shutil.copy(preds_path, os.path.join(output_dir, "predictions.csv"))

    for f in [roc_path, pr_path, cm_path, preds_path]:
        if os.path.exists(f):
            os.remove(f)

    print("Evaluation completed and artifacts logged to MLflow.")


if __name__ == "__main__":
    main()
