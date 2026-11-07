from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Brain Tumor AI",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Brain Tumor AI")
st.write("MRI classification and tumor localization")
st.warning(
    "Research prototype only — not for clinical diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    contents = uploaded_file.getvalue()
    image = Image.open(BytesIO(contents)).convert("RGB")

    st.image(image, caption="Uploaded MRI", width=350)

    if st.button("Analyze MRI", type="primary"):
        files = {
            "file": (
                uploaded_file.name,
                contents,
                uploaded_file.type or "image/jpeg",
            )
        }

        try:
            with st.spinner("Analyzing MRI..."):
                classification_response = requests.post(
                    f"{API_URL}/predict/classification",
                    files=files,
                    timeout=120,
                )
                classification_response.raise_for_status()
                classification = classification_response.json()

                detection_response = requests.post(
                    f"{API_URL}/predict/detection",
                    files=files,
                    timeout=120,
                )
                detection_response.raise_for_status()
                detection = detection_response.json()

            left, right = st.columns(2)

            with left:
                st.subheader("Classification")
                st.metric(
                    "Predicted class",
                    classification["predicted_class"],
                )
                st.write(
                    f"Model confidence: "
                    f"{classification['confidence']:.1%}"
                )
                st.write("Class probabilities")
                st.bar_chart(classification["probabilities"])

            with right:
                st.subheader("Tumor detection")

                annotated = image.copy()
                draw = ImageDraw.Draw(annotated)

                for box in detection["detections"]:
                    x1, y1, x2, y2 = box["bbox_xyxy"]
                    draw.rectangle(
                        (x1, y1, x2, y2),
                        outline="lime",
                        width=3,
                    )
                    draw.text(
                        (x1, max(0, y1 - 15)),
                        f"Tumor {box['confidence']:.1%}",
                        fill="lime",
                    )

                st.image(
                    annotated,
                    caption="YOLOv8 predicted bounding boxes",
                    use_container_width=True,
                )
                st.write(
                    "Detected regions:",
                    detection["num_detections"],
                )

        except requests.RequestException as error:
            st.error(
                "Could not communicate with FastAPI. "
                "Check that the backend is running."
            )
            st.code(str(error))
        except (KeyError, ValueError) as error:
            st.error(f"Unexpected API response: {error}")
