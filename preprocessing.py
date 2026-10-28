from PIL import Image, ImageOps
import numpy as np


def preprocess_for_model(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        image = ImageOps.pad(
            image,
            (224, 224),
            method=Image.Resampling.BILINEAR,
            color=(0, 0, 0)
        )

        image_array = np.array(image, dtype=np.float32)

    return image_array / 255.0
