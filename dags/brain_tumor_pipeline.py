import csv
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

ROOT = Path(__file__).resolve().parents[1]


def validate_datasets():
    split_file = ROOT / "data/processed/classification_split.csv"

    with split_file.open(newline="") as f:
        rows = list(csv.DictReader(f))

    for split, expected in [
        ("train", 2739),
        ("val", 587),
        ("test", 587),
    ]:
        subset = [r for r in rows if r["split"] == split]
        assert len(subset) == expected, (
            f"{split}: expected {expected}, found {len(subset)}"
        )
        assert all(
            (ROOT / r["image_path"]).is_file() for r in subset
        ), f"Missing classification images in {split}"
        print(f"Classification {split}: {len(subset)} images")

    yolo_root = ROOT / "data/processed/yolo_binary"
    for split, expected in [
        ("train", 2581),
        ("val", 559),
        ("test", 550),
    ]:
        images = list((yolo_root / "images" / split).glob("*.jpg"))
        labels = yolo_root / "labels" / split

        assert len(images) == expected, (
            f"YOLO {split}: expected {expected}, found {len(images)}"
        )
        assert all(
            (labels / f"{image.stem}.txt").is_file()
            for image in images
        )
        print(f"YOLO {split}: {len(images)} image-label pairs")


def verify_models():
    weights = [
        ROOT / "artifacts/googlenet_baseline/best.weights.h5",
        ROOT / "artifacts/yolo_detection/yolov8n_baseline/weights/best.pt",
    ]
    for path in weights:
        assert path.is_file() and path.stat().st_size > 0, path
        print(f"Checkpoint verified: {path.name}")


def validate_metrics():
    files = [
        ROOT / "artifacts/googlenet_baseline/validation_evaluation.json",
        ROOT / "artifacts/googlenet_baseline/test_evaluation.json",
        ROOT / "artifacts/yolov8n_baseline_metrics.json",
    ]
    for path in files:
        with path.open() as f:
            metrics = json.load(f)
        assert isinstance(metrics, dict) and metrics, path
        print(f"Metrics verified: {path.name}")


def log_to_mlflow():
    executable = ROOT / ".venv-mlflow/bin/python"
    script = ROOT / "log_experiments.py"

    subprocess.run(
        [str(executable), str(script)],
        cwd=str(ROOT),
        check=True,
        timeout=180,
    )
    print("MLflow logging complete")


with DAG(
    dag_id="brain_tumor_pipeline",
    start_date=datetime(2026, 11, 1),
    schedule_interval=None,
    catchup=False,
    default_args={
        "owner": "brain-tumor-ai",
        "retries": 1,
        "retry_delay": timedelta(seconds=10),
    },
    tags=["brain-tumor", "research"],
) as dag:
    data_task = PythonOperator(
        task_id="validate_datasets",
        python_callable=validate_datasets,
    )

    models_task = PythonOperator(
        task_id="verify_models",
        python_callable=verify_models,
    )

    metrics_task = PythonOperator(
        task_id="validate_metrics",
        python_callable=validate_metrics,
    )

    tracking_task = PythonOperator(
        task_id="log_to_mlflow",
        python_callable=log_to_mlflow,
    )

    data_task >> models_task >> metrics_task >> tracking_task
