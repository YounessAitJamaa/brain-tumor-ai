# Brain Tumor AI — Classification and Detection

Educational deep-learning research prototype for brain MRI analysis.

## Features

- GoogLeNet / Inception v1: four-class MRI classification
- YOLOv8n: binary tumor bounding-box detection
- FastAPI: REST inference endpoints
- Streamlit: MRI upload and visualization
- MLflow: experiment tracking
- Apache Airflow: dataset, artifact and experiment verification

## Results

| Model | Held-out test metric | Result |
|---|---|---|
| GoogLeNet | Accuracy | 83.13% |
| GoogLeNet | Macro F1 | 84.12% |
| YOLOv8n | Precision | 93.2% |
| YOLOv8n | Recall | 88.8% |
| YOLOv8n | mAP@50 | 95.9% |
| YOLOv8n | mAP@50–95 | 74.4% |

## Local setup

Python 3.10 is used for the two independent model environments.

Create the application environment:

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements-api.txt
```

Create the YOLO environment:

```bash
uv venv --python 3.10 .venv-yolo
uv pip install --python .venv-yolo/bin/python -r requirements-yolo.txt
```

The trained checkpoints and MRI datasets must be restored separately;
they are intentionally excluded from GitHub.

Expected checkpoints:

- `artifacts/googlenet_baseline/best.weights.h5`
- `artifacts/yolo_detection/yolov8n_baseline/weights/best.pt`

## Run the application

Terminal 1:

```bash
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

FastAPI docs: http://127.0.0.1:8000/docs

Streamlit: http://localhost:8501

## MLflow

Install MLflow in its own Python environment, then:

```bash
.venv-mlflow/bin/python log_experiments.py
.venv-mlflow/bin/python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The logged experiments reflect previously completed training runs.

## Airflow

Using Apache Airflow 2.10.5 and Python 3.10:

```bash
source airflow_env.sh
.venv-airflow/bin/airflow db migrate
.venv-airflow/bin/airflow dags test brain_tumor_pipeline 2026-11-08
```

The DAG checks datasets, checkpoints, metrics and MLflow logging.
It does not retrain the models.

## Limitations

- Research and education only; not for clinical diagnosis.
- MRI bounding-box annotations are not clinically verified.
- Patient-level independence across splits is not established.
- Model confidence is not a calibrated medical probability.
- Local application prototype; not a production deployment.
