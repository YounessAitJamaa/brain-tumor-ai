import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / (
    "data/processed/yolo_binary/images/val/"
    "00040409d120315633f74805ce0ae401.jpg"
)
URL = "http://127.0.0.1:8765"

assert IMAGE.is_file(), f"Missing image: {IMAGE}"


def check(name, response, expected):
    assert response.status_code == expected, (
        f"{name}: expected {expected}, "
        f"got {response.status_code}: {response.text[:300]}"
    )
    print(f"PASS: {name}")


env = os.environ.copy()
env["TF_CPP_MIN_LOG_LEVEL"] = "3"

with open("/tmp/brain-tumor-api-test.log", "w") as log:
    server = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "api:app",
            "--host", "127.0.0.1", "--port", "8765"
        ],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )

    try:
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError("API server exited unexpectedly")
            try:
                response = requests.get(URL + "/health", timeout=1)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("API server did not start")

        check("Health endpoint", response, 200)
        assert response.json()["status"] == "ok"

        with IMAGE.open("rb") as f:
            response = requests.post(
                URL + "/predict/classification",
                files={"file": ("sample.jpg", f, "image/jpeg")},
                timeout=180,
            )
        check("GoogLeNet classification", response, 200)

        classification = response.json()
        probabilities = classification["probabilities"]
        assert len(probabilities) == 4
        assert abs(sum(probabilities.values()) - 1) < 0.001
        assert classification["predicted_class"] in probabilities

        with IMAGE.open("rb") as f:
            response = requests.post(
                URL + "/predict/detection",
                files={"file": ("sample.jpg", f, "image/jpeg")},
                timeout=180,
            )
        check("YOLOv8 detection", response, 200)

        detection = response.json()
        assert detection["num_detections"] >= 1
        assert len(detection["detections"]) == detection["num_detections"]

        for box in detection["detections"]:
            x1, y1, x2, y2 = box["bbox_xyxy"]
            assert 0 <= x1 < x2 <= detection["image_width"]
            assert 0 <= y1 < y2 <= detection["image_height"]

        response = requests.post(
            URL + "/predict/classification",
            files={"file": ("bad.txt", b"test", "text/plain")},
            timeout=30,
        )
        check("Unsupported file type", response, 415)

        response = requests.post(
            URL + "/predict/detection",
            files={"file": ("bad.jpg", b"invalid", "image/jpeg")},
            timeout=30,
        )
        check("Corrupted image rejection", response, 400)

        response = requests.post(
            URL + "/predict/classification",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
            timeout=30,
        )
        check("Empty image rejection", response, 400)

        print("\nALL API REGRESSION TESTS PASSED!")

    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
