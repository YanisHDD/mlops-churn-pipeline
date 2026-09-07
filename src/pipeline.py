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
    numeric=None,
    categorical=None,
    model_type="logreg",
    model_params=None,
    numeric_features=None,
    categorical_features=None,
) -> Pipeline:
    """
    Construct a complete, leak-free scikit-learn Pipeline with ColumnTransformer.

    Args:
        numeric: List of continuous column names.
        categorical: List of categorical column names.
        model_type: 'logreg' (or 'logistic_regression') or 'random_forest'.
        model_params: Dict of parameters to pass to model constructor.
        numeric_features: Alias for numeric.
        categorical_features: Alias for categorical.

    Returns:
        Pipeline with 'pre' and 'model' steps.
    """
    num_cols = numeric if numeric is not None else numeric_features
    cat_cols = categorical if categorical is not None else categorical_features
    model_params = model_params or {}

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
    )

    if model_type in ("logreg", "logistic_regression"):
        params = {"max_iter": 500}
        params.update(model_params)
        model = LogisticRegression(**params)
    elif model_type == "random_forest":
        params = {"random_state": 42}
        params.update(model_params)
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(steps=[("pre", preprocessor), ("model", model)])

