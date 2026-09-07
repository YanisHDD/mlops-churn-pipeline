"""
Unit tests for data utility functions.
"""

import pandas as pd
import pytest
import yaml

from src.utils import load_config, load_data, split_data


@pytest.fixture
def sample_cfg(tmp_path):
    """Minimal config and dummy CSV written to disk."""
    csv_path = tmp_path / "raw.csv"
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4, 5, 6, 7, 8, " ", 10],  # blank string coerced to NaN
        "cat1": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        "target": ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"],
    })
    df.to_csv(csv_path, index=False)

    cfg = {
        "data": {
            "csv_path": str(csv_path),
            "target": "target",
            "test_size": 0.3,
            "random_state": 42,
        },
        "features": {
            "numeric": ["num1"],
            "categorical": ["cat1"],
        },
    }
    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    return str(cfg_path), cfg


def test_load_config_reads_yaml(sample_cfg):
    cfg_path, expected = sample_cfg
    loaded = load_config(cfg_path)
    assert loaded == expected


def test_load_data_coerces_numeric_and_handles_blanks(sample_cfg):
    """Blank strings in numeric features are coerced to NaN."""
    _, cfg = sample_cfg
    df = load_data(cfg)
    assert pd.api.types.is_numeric_dtype(df["num1"])
    assert df["num1"].isna().sum() == 1


def test_split_data_shapes_and_target_encoding(sample_cfg):
    _, cfg = sample_cfg
    df = load_data(cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    assert len(X_train) == 7
    assert len(X_test) == 3
    assert len(X_train) + len(X_test) == len(df)
    assert set(y_train.unique()) <= {0, 1}
    assert set(y_test.unique()) <= {0, 1}
    assert "target" not in X_train.columns
