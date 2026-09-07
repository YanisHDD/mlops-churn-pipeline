"""
Script to download the Telco Customer Churn dataset.
Works out of the box even without pyyaml installed.
"""

import os
import urllib.request

DEFAULT_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DEFAULT_DEST = "data/raw.csv"


def download_dataset(config_path: str = "configs/config.yaml"):
    url = DEFAULT_URL
    dest = DEFAULT_DEST

    if os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                url = cfg.get("data", {}).get("raw_url", DEFAULT_URL)
                dest = cfg.get("data", {}).get("csv_path", DEFAULT_DEST)
        except (ImportError, OSError, KeyError):
            pass

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest):
        print(f"Dataset already exists at {dest}")
        return

    print(f"Downloading dataset from {url} to {dest}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Dataset successfully downloaded ({os.path.getsize(dest)} bytes).")


if __name__ == "__main__":
    download_dataset()
