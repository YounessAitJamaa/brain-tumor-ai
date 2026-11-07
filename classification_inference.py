import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from class_mapping import CLASS_NAMES
from preprocessing import preprocess_for_model
from train_googlenet import MODEL_URL


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "artifacts/googlenet_baseline"


@lru_cache(maxsize=1)
def load_classification_model():
    with (MODEL_DIR / "classes.json").open() as file:
        saved_classes = json.load(file)

    if saved_classes != CLASS_NAMES:
        raise ValueError("Classification class order mismatch")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        hub.KerasLayer(MODEL_URL, trainable=False),
        tf.keras.layers.Dense(
            len(CLASS_NAMES),
            activation="softmax"
        ),
    ])

    model.load_weights(str(MODEL_DIR / "best.weights.h5"))
    return model


def predict_classification(image_path):
    image = preprocess_for_model(image_path)
    batch = np.expand_dims(image, axis=0)

    model = load_classification_model()
    probabilities = model(batch, training=False).numpy()[0]

    predicted_id = int(np.argmax(probabilities))

    return {
        "predicted_class": CLASS_NAMES[predicted_id],
        "confidence": float(probabilities[predicted_id]),
        "probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(CLASS_NAMES)
        }
    }
