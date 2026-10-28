import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from class_mapping import CLASS_NAMES, CLASS_TO_ID
from preprocessing import preprocess_for_model

MODEL_URL = (
    "https://www.kaggle.com/models/google/inception-v1/"
    "TensorFlow2/feature-vector/2"
)


class MRIBatches(tf.keras.utils.Sequence):
    """Load and preprocess one small batch of MRI images at a time."""

    def __init__(self, rows, batch_size=32, shuffle=False):
        super().__init__()
        self.rows = rows
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.order = np.arange(len(rows))
        self.random = np.random.default_rng(42)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.rows) / self.batch_size)

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        indices = self.order[start:start + self.batch_size]
        images = np.stack([
            preprocess_for_model(self.rows[i]["image_path"])
            for i in indices
        ])
        labels = np.array([
            CLASS_TO_ID[self.rows[i]["class"]]
            for i in indices
        ], dtype=np.int32)
        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            self.random.shuffle(self.order)


def main():
    tf.keras.utils.set_random_seed(42)
    output_dir = Path("artifacts/googlenet_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open("data/processed/classification_split.csv", newline="") as file:
        rows = list(csv.DictReader(file))

    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    assert len(train_rows) == 2739 and len(val_rows) == 587

    train_batches = MRIBatches(train_rows, shuffle=True)
    val_batches = MRIBatches(val_rows)

    print(f"Training: {len(train_rows)} | Validation: {len(val_rows)}")
    images, labels = train_batches[0]
    print("First batch:", images.shape, labels.shape, images.dtype)
    assert images.shape[1:] == (224, 224, 3)

    extractor = hub.KerasLayer(MODEL_URL, trainable=False)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        extractor,
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint = output_dir / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss",
            save_best_only=True, save_weights_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(output_dir / "history.csv")),
    ]
    with open(output_dir / "classes.json", "w") as file:
        json.dump(CLASS_NAMES, file, indent=2)

    model.fit(train_batches, validation_data=val_batches,
              epochs=5, callbacks=callbacks)
    print("Best model weights saved to:", checkpoint)


if __name__ == "__main__":
    main()
