import argparse
import json
import sys
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS = (
    ROOT
    / "artifacts/yolo_detection/yolov8n_baseline/weights/best.pt"
)


@lru_cache(maxsize=1)
def load_detection_model():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing YOLO weights: {WEIGHTS}")

    return YOLO(str(WEIGHTS))


def predict_detection(image_path, confidence=0.25):
    model = load_detection_model()

    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=confidence,
        device="cpu",
        verbose=False
    )[0]

    height, width = result.orig_shape
    detections = []

    for box in result.boxes:
        class_id = int(box.cls.item())

        detections.append({
            "class_name": result.names[class_id],
            "confidence": round(float(box.conf.item()), 4),
            "bbox_xyxy": [
                round(float(value), 1)
                for value in box.xyxy[0].tolist()
            ]
        })

    return {
        "image_width": width,
        "image_height": height,
        "detections": detections,
        "num_detections": len(detections)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    args = parser.parse_args()

    # Keep standard output clean for FastAPI JSON parsing.
    with redirect_stdout(sys.stderr):
        prediction = predict_detection(args.image_path)

    print(json.dumps(prediction))
