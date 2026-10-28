
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_hub as hub

from class_mapping import CLASS_NAMES


MODEL_URL = (
    "https://www.kaggle.com/models/google/inception-v1/"
    "TensorFlow2/feature-vector/2"
)


print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# Load pretrained GoogLeNet feature extractor
feature_extractor = hub.KerasLayer(
    MODEL_URL,
    trainable=False
)

# Build our four-class model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    feature_extractor,
    tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax"
    )
])

# Test with one artificial MRI-shaped input
test_image = tf.zeros((1, 224, 224, 3))

predictions = model(test_image)

print("Output shape:", predictions.shape)
print("Probabilities:", predictions.numpy())

assert predictions.shape == (1, 4)

print("GoogLeNet classifier test passed!")
