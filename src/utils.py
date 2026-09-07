"""
Utility functions for data loading, preprocessing, config handling, and plotting.
"""

import os
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError as err:
        raise ImportError("pyyaml is required to load YAML configs. Please install it with: pip install pyyaml") from err


def load_and_preprocess_data(
    csv_path: str,
    target_col: str = "Churn"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load raw Telco Churn CSV and perform basic cleaning:
    - Clean 'TotalCharges' (handle empty strings as NaN)
    - Encode target 'Churn' into binary 0/1
    - Drop customerID identifier column
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found at {csv_path}. Run download_data.py first.")

    df = pd.read_csv(csv_path)

    # Drop customerID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges contains blank strings ' ' which need to be coerced to float
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")

    # SeniorCitizen is categorical (0 or 1), cast to string
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Encode target
    if target_col in df.columns:
        if df[target_col].dtype == object:
            y = df[target_col].map({"Yes": 1, "No": 0})
        else:
            y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        raise KeyError(f"Target column '{target_col}' not found in dataset columns.")

    return X, y


def plot_confusion_matrix(y_true, y_pred, output_path: str):
    """Generate and save confusion matrix figure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Churn (0)", "Churn (1)"],
                yticklabels=["No Churn (0)", "Churn (1)"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_prob, output_path: str):
    """Generate and save ROC curve figure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_precision_recall_curve(y_true, y_prob, output_path: str):
    """Generate and save Precision-Recall curve figure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="green", lw=2, label=f"PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
