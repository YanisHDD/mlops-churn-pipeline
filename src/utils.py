"""
Utility functions for data loading, preprocessing, config handling, and plotting.
"""

import os
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
import yaml


def load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load raw CSV and coerce numeric columns to float, turning blanks into NaN."""
    csv_path = cfg["data"]["csv_path"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}. Run download_data.py first.")
    df = pd.read_csv(csv_path)
    for col in cfg["features"]["numeric"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def split_data(df: pd.DataFrame, cfg: dict[str, Any]):
    """Split dataframe into X_train, X_test, y_train, y_test with stratification."""
    target = cfg["data"]["target"]
    numeric = cfg["features"]["numeric"]
    categorical = cfg["features"]["categorical"]

    X = df[numeric + categorical]
    y = df[target]

    if y.dtype == object:
        y = y.map({"Yes": 1, "No": 0}).fillna(y)
        y = y.astype(int)

    return train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )


def plot_roc(model, X_test, y_test, out_path="roc_curve.png") -> str:
    """Generate and save ROC curve figure."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pr(model, X_test, y_test, out_path="pr_curve.png") -> str:
    """Generate and save Precision-Recall curve figure."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(model, X_test, y_test, out_path="confusion_matrix.png") -> str:
    """Generate and save confusion matrix figure."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="Blues")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

