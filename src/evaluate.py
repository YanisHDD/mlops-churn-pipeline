"""
Evaluation script to assess model performance and log graphical artifacts to MLflow.
"""

import argparse
import json
import os
import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import (
    load_config,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)


def evaluate(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)

    # 1. Load Model
    model_path = os.path.join(config["artifacts"]["output_dir"], config["artifacts"]["model_filename"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'. Run 'make train' first.")

    print(f"Loading trained pipeline from {model_path}...")
    pipeline = joblib.load(model_path)

    # 2. Load Test Data
    test_csv_path = "data/processed/test.csv"
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test data not found at '{test_csv_path}'. Run 'make train' first.")

    target_col = config["data"]["target"]
    test_df = pd.read_csv(test_csv_path)
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    print(f"Loaded {len(X_test)} test examples.")

    # 3. Inferences
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 4. Metrics
    metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_score": float(f1_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
    }

    print("\n================ EVALUATION REPORT ================")
    print(classification_report(y_test, y_pred, target_names=["No Churn (0)", "Churn (1)"]))
    print(f"ROC-AUC Score : {metrics['test_roc_auc']:.4f}")
    print("===================================================\n")

    # 5. Generate Graphical Artifacts
    output_dir = config["artifacts"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    roc_path = os.path.join(output_dir, "roc_curve.png")
    pr_path = os.path.join(output_dir, "precision_recall_curve.png")
    metrics_json_path = os.path.join(output_dir, "metrics.json")

    plot_confusion_matrix(y_test, y_pred, cm_path)
    plot_roc_curve(y_test, y_prob, roc_path)
    plot_precision_recall_curve(y_test, y_prob, pr_path)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Artifacts saved in '{output_dir}/':")
    print(f"  - {cm_path}")
    print(f"  - {roc_path}")
    print(f"  - {pr_path}")
    print(f"  - {metrics_json_path}")

    # 6. Log Artifacts into MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", config["mlflow"]["experiment_name"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="final-evaluation"):
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
        mlflow.log_artifact(roc_path, artifact_path="evaluation_plots")
        mlflow.log_artifact(pr_path, artifact_path="evaluation_plots")
        mlflow.log_artifact(metrics_json_path, artifact_path="metrics")
        print("\nAll artifacts successfully logged into MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate customer churn model and generate artifacts")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    evaluate(args.config)
