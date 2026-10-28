import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from class_mapping import CLASS_NAMES, CLASS_TO_ID
from train_googlenet import MODEL_URL, MRIBatches


OUTPUT_DIR = Path("artifacts/googlenet_baseline")
CHECKPOINT = OUTPUT_DIR / "best.weights.h5"


def main():
    tf.keras.utils.set_random_seed(42)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing trained weights: {CHECKPOINT}")

    saved_classes_path = OUTPUT_DIR / "classes.json"
    with saved_classes_path.open() as file:
        saved_classes = json.load(file)
    if saved_classes != CLASS_NAMES:
        raise ValueError("Saved class order does not match class_mapping.py")

    with Path("data/processed/classification_split.csv").open(newline="") as file:
        all_rows = list(csv.DictReader(file))
    val_rows = [row for row in all_rows if row["split"] == "val"]
    if len(val_rows) != 587:
        raise ValueError(f"Expected 587 validation rows; found {len(val_rows)}")

    missing = [row["image_path"] for row in val_rows
               if not Path(row["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing validation image: {missing[0]}")

    batches = MRIBatches(val_rows, batch_size=32, shuffle=False)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        hub.KerasLayer(MODEL_URL, trainable=False),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    model.load_weights(str(CHECKPOINT))
    print("Loaded trained checkpoint:", CHECKPOINT)

    probabilities = model.predict(batches, verbose=1)
    if probabilities.shape != (len(val_rows), len(CLASS_NAMES)):
        raise ValueError(f"Unexpected prediction shape: {probabilities.shape}")

    actual = np.array([CLASS_TO_ID[row["class"]] for row in val_rows])
    predicted = probabilities.argmax(axis=1)
    count = len(CLASS_NAMES)
    confusion = np.zeros((count, count), dtype=np.int64)
    for true_label, predicted_label in zip(actual, predicted):
        confusion[true_label, predicted_label] += 1

    correct = int(np.trace(confusion))
    accuracy = correct / len(actual)
    print(f"\nValidation accuracy: {accuracy:.4%} ({correct}/{len(actual)})")
    print("Confusion matrix: rows=actual, columns=predicted")
    print("Class order:", CLASS_NAMES)
    print(confusion)

    results = []
    print("\nClass                 Precision  Recall     F1        Support")
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_total = int(confusion[:, index].sum())
        precision = tp / predicted_total if predicted_total else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results.append({"class": name, "precision": precision, "recall": recall,
                        "f1": f1, "support": support})
        print(f"{name:<21} {precision:>8.3f}   {recall:>7.3f}   {f1:>7.3f}   {support:>5}")

    macro_f1 = float(np.mean([entry["f1"] for entry in results]))
    print(f"\nMacro F1: {macro_f1:.4f}")

    output_path = OUTPUT_DIR / "validation_evaluation.json"
    with output_path.open("w") as file:
        json.dump({"split": "val", "samples": len(val_rows),
                   "accuracy": accuracy, "macro_f1": macro_f1,
                   "class_names": CLASS_NAMES,
                   "confusion_matrix": confusion.tolist(),
                   "per_class": results}, file, indent=2)
    print("Saved validation metrics to:", output_path)


if __name__ == "__main__":
    main()
