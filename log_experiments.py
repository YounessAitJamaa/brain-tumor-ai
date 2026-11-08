import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("Brain Tumor AI - Baselines")


def read_json(path):
    with path.open() as file:
        return json.load(file)


classification = ROOT / "artifacts/googlenet_baseline"
detection = ROOT / "artifacts/yolov8n_baseline_metrics.json"

val = read_json(classification / "validation_evaluation.json")
test = read_json(classification / "test_evaluation.json")
yolo = read_json(detection)

with mlflow.start_run(run_name="GoogLeNet-Inception-v1"):
    mlflow.log_param("model", "GoogLeNet Inception v1")
    mlflow.log_param("task", "4-class classification")
    mlflow.log_param("image_size", 224)
    mlflow.log_metric("validation_accuracy", val["accuracy"])
    mlflow.log_metric("validation_macro_f1", val["macro_f1"])
    mlflow.log_metric("test_accuracy", test["accuracy"])
    mlflow.log_metric("test_macro_f1", test["macro_f1"])
    mlflow.log_artifact(str(classification / "validation_evaluation.json"))
    mlflow.log_artifact(str(classification / "test_evaluation.json"))

with mlflow.start_run(run_name="YOLOv8n-Tumor-Detection"):
    for key, value in yolo["training"].items():
        mlflow.log_param(key, value)

    mlflow.log_param("model", "YOLOv8n")
    mlflow.log_param("task", "binary tumor detection")

    for split in ("validation", "test"):
        for metric, value in yolo[f"{split}_metrics"].items():
            mlflow.log_metric(f"{split}_{metric}", value)

    mlflow.log_artifact(str(detection))

print("MLflow experiments recorded successfully!")
