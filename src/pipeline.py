"""
Machine Learning Pipeline definition with ColumnTransformer and Classifier.
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    model_type: str = "logistic_regression"
) -> Pipeline:
    """
    Construct a complete, leak-free scikit-learn Pipeline with ColumnTransformer.

    Args:
        numeric_features: List of continuous column names.
        categorical_features: List of categorical column names.
        model_type: 'logistic_regression' or 'random_forest'.

    Returns:
        Configured scikit-learn Pipeline.
    """
    # Numeric pipeline: median imputation + standard scaling
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: frequent imputation + one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Preprocessor combining both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    # Instantiate chosen model
    if model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. Choose 'logistic_regression' or 'random_forest'."
        )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", classifier)
    ])
