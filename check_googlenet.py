import os

# Use legacy Keras for TensorFlow Hub compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_hub as hub

MODEL_URL = (
    "https://tfhub.dev/google/imagenet/"
    "inception_v1/feature_vector/4"
)

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

# Load pretrained GoogLeNet / Inception v1
model = hub.load(MODEL_URL)

print("Pretrained GoogLeNet loaded successfully!")

# Create one artificial RGB image
test_image = tf.zeros((1, 224, 224, 3), dtype=tf.float32)

# Extract image features
features = model(test_image)

print("Input shape:", test_image.shape)
print("Feature shape:", features.shape)
