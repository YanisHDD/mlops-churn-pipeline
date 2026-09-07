"""
Batch inference script using a model from the MLflow Model Registry.
"""

import argparse
import mlflow.sklearn
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch predictions with registered model")
    parser.add_argument("--input-csv", required=True, help="CSV containing feature columns")
    parser.add_argument("--output-csv", default="predictions_out.csv")
    parser.add_argument(
        "--model-uri",
        default="models:/ChurnClassifier/1",
        help="MLflow model URI (e.g. models:/ChurnClassifier/1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = mlflow.sklearn.load_model(args.model_uri)

    df = pd.read_csv(args.input_csv)
    df["prediction"] = model.predict(df)
    df["prediction_proba"] = model.predict_proba(df)[:, 1]
    df.to_csv(args.output_csv, index=False)
    print(f"Predictions successfully written to {args.output_csv}")


if __name__ == "__main__":
    main()
