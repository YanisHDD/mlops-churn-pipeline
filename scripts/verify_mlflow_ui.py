"""
Automated verification of MLflow UI API content.

Starts the MLflow tracking server headless, queries the REST API,
and asserts that experiment, runs, metrics, artifacts, and registered models
are properly logged.
"""

import subprocess
import sys
import time
import requests

HOST = "127.0.0.1"
PORT = "5050"
BASE_URL = f"http://{HOST}:{PORT}"


def wait_for_server(timeout=30):
    for _ in range(timeout):
        try:
            r = requests.get(f"{BASE_URL}/", timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


def main():
    tracking_uri = sys.argv[1] if len(sys.argv) > 1 else "sqlite:///mlflow.db"
    experiment_name = sys.argv[2] if len(sys.argv) > 2 else "churn-exp"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "ChurnClassifier"

    print(f"Starting MLflow server ({tracking_uri}) on port {PORT}...")
    proc = subprocess.Popen(
        ["mlflow", "server", "--backend-store-uri", tracking_uri,
         "--host", HOST, "--port", PORT],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        if not wait_for_server():
            print("[ERROR] MLflow server failed to start within timeout.")
            sys.exit(1)
        print("[OK] MLflow server started.")

        # 1. Experiment check
        exps_resp = requests.post(
            f"{BASE_URL}/api/2.0/mlflow/experiments/search",
            json={"max_results": 20},
        )
        exps = exps_resp.json().get("experiments", [])
        exp = next((e for e in exps if e["name"] == experiment_name), None)
        assert exp is not None, f"[ERROR] Experiment '{experiment_name}' not found"
        print(f"[OK] Experiment '{experiment_name}' found (id={exp['experiment_id']})")

        # 2. Runs & metrics check
        runs_resp = requests.post(
            f"{BASE_URL}/api/2.0/mlflow/runs/search",
            json={"experiment_ids": [exp["experiment_id"]], "max_results": 50},
        )
        runs = runs_resp.json().get("runs", [])
        assert len(runs) > 0, "[ERROR] No runs found"
        print(f"[OK] {len(runs)} run(s) found")

        eval_run = next((r for r in runs if "evaluate" in r["info"]["run_name"]), None)
        assert eval_run is not None, "[ERROR] No evaluation run found"
        metrics = {m["key"]: m["value"] for m in eval_run["data"].get("metrics", [])}
        assert "eval_roc_auc" in metrics, "[ERROR] Metric eval_roc_auc missing"
        print(f"[OK] Evaluation run found - eval_roc_auc = {metrics['eval_roc_auc']:.4f}")

        # 3. Artifacts check
        artifacts_resp = requests.get(
            f"{BASE_URL}/api/2.0/mlflow/artifacts/list",
            params={"run_id": eval_run["info"]["run_id"]},
        )
        artifacts = artifacts_resp.json().get("files", [])
        artifact_names = {a["path"] for a in artifacts}
        expected = {"roc_curve.png", "pr_curve.png", "confusion_matrix.png"}
        missing = expected - artifact_names
        assert not missing, f"[ERROR] Missing artifacts: {missing}"
        print(f"[OK] Artifacts present: {sorted(artifact_names)}")

        # 4. Model Registry check
        models_resp = requests.get(
            f"{BASE_URL}/api/2.0/mlflow/registered-models/search",
            params={"max_results": 20},
        )
        models = models_resp.json().get("registered_models", [])
        model = next((m for m in models if m["name"] == model_name), None)
        assert model is not None, f"[ERROR] Model '{model_name}' missing from Registry"
        latest = model["latest_versions"][0]
        print(f"[OK] Model '{model_name}' v{latest['version']} - status: {latest['status']}")

        print("\nAll MLflow UI components verified successfully.")

    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
