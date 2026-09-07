import pandas as pd

from src.pipeline import build_pipeline


def test_build_pipeline_steps():
    pipe = build_pipeline(["a"], ["b"], "logreg")
    steps = dict(pipe.named_steps)
    assert "pre" in steps
    assert "model" in steps


def test_pipeline_fit_predict():
    pipe = build_pipeline(["a"], ["b"], "logreg")
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})
    y = [0, 1, 0, 1]
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)


def test_invalid_model_type_raises():
    try:
        build_pipeline(["a"], ["b"], "not_a_model")
        assert False, "Devrait lever une ValueError"
    except ValueError:
        pass
