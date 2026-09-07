"""
Baseline verification: train/test split with a single model without tuning.
"""

import argparse
from sklearn.metrics import roc_auc_score

from src.pipeline import build_pipeline
from src.utils import load_config, load_data, split_data


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline verification for customer churn model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    df = load_data(cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    numeric = cfg["features"]["numeric"]
    categorical = cfg["features"]["categorical"]
    model_type = cfg["model"]["type"]

    pipe = build_pipeline(numeric, categorical, model_type=model_type)
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f"[baseline] model={model_type} | train={len(X_train)} test={len(X_test)}")
    print(f"[baseline] ROC-AUC (baseline) = {auc:.4f}")
    print("[baseline] Pipeline OK - ready for grid search and MLflow tracking.")


if __name__ == "__main__":
    main()
