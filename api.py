from io import BytesIO
from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from classification_inference import predict_classification


app = FastAPI(
    title="Brain Tumor AI",
    description="Research prototype — not for clinical diagnosis",
)

MAX_IMAGE_BYTES = 10 * 1024 * 1024


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/classification")
async def classify_image(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=415,
            detail="Please upload a JPEG or PNG image",
        )

    contents = await file.read(MAX_IMAGE_BYTES + 1)

    if not contents:
        raise HTTPException(status_code=400, detail="Empty image")

    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")

    try:
        with Image.open(BytesIO(contents)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp.write(contents)
        image_path = Path(temp.name)

    try:
        return predict_classification(image_path)
    finally:
        image_path.unlink(missing_ok=True)


# YOLOv8 runs in a separate Python environment.
import json
import subprocess

ROOT = Path(__file__).resolve().parent
YOLO_PYTHON = ROOT / ".venv-yolo/bin/python"
YOLO_SCRIPT = ROOT / "detection_inference.py"


@app.post("/predict/detection")
async def detect_tumor(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(
            status_code=415,
            detail="Please upload a JPEG or PNG image",
        )

    contents = await file.read(MAX_IMAGE_BYTES + 1)

    if not contents:
        raise HTTPException(status_code=400, detail="Empty image")

    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")

    try:
        with Image.open(BytesIO(contents)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image")

    with tempfile.NamedTemporaryFile(
        suffix=".png", delete=False
    ) as temp:
        temp.write(contents)
        image_path = Path(temp.name)

    try:
        result = subprocess.run(
            [str(YOLO_PYTHON), str(YOLO_SCRIPT), str(image_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        return json.loads(result.stdout)

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Detection timed out",
        )
    except (subprocess.CalledProcessError, OSError, ValueError):
        raise HTTPException(
            status_code=500,
            detail="Detection inference failed",
        )
    finally:
        image_path.unlink(missing_ok=True)
