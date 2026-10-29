import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path


SOURCE_TO_CLASS_ID = {"Glioma": 0, "Meningioma": 1, "Pituitary": 3}
SPLITS = ("train", "val", "test")


def inspect_positive_label(label_file, expected_id):
    """Return normalized binary-label lines or a reason to exclude an image."""
    if not label_file.is_file():
        return None, "missing_positive_label"

    lines = [line.strip() for line in label_file.read_text().splitlines() if line.strip()]
    if not lines:
        return None, "empty_positive_label"

    binary_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            return None, "malformed_positive_label"

        try:
            class_id = int(parts[0])
            x, y, w, h = (float(value) for value in parts[1:])
        except ValueError:
            return None, "malformed_positive_label"

        if not all(math.isfinite(value) for value in (x, y, w, h)):
            return None, "invalid_box_geometry"
        if not (0 < w <= 1 and 0 < h <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
            return None, "invalid_box_geometry"
        if not (0 <= x - w / 2 and x + w / 2 <= 1):
            return None, "invalid_box_geometry"
        if not (0 <= y - h / 2 and y + h / 2 <= 1):
            return None, "invalid_box_geometry"

        if class_id != expected_id:
            return None, "source_class_mismatch"

        # Preserve box coordinates exactly; only the box class changes.
        binary_lines.append("0 " + " ".join(parts[1:]))

    return binary_lines, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path,
        default=Path("data/processed/classification_split.csv"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/processed/yolo_binary"),
    )
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(
            f"Output {args.out} already contains files. "
            "Move or remove the old generated output before running again."
        )

    with args.csv.open(newline="") as source:
        rows = list(csv.DictReader(source))

    if not rows:
        raise SystemExit("Input CSV is empty")
    required = {"split", "class", "image_path", "hash"}
    if not required.issubset(rows[0]):
        raise SystemExit(f"CSV requires fields: {sorted(required)}")

    if len({row["hash"] for row in rows}) != len(rows):
        raise SystemExit("Image hashes are not unique; stop to avoid leakage")

    # First validate ALL input image paths before copying anything.
    for row in rows:
        if row["split"] not in SPLITS:
            raise SystemExit(f"Unknown split: {row['split']}")
        if row["class"] not in (*SOURCE_TO_CLASS_ID, "No Tumor"):
            raise SystemExit(f"Unknown class: {row['class']}")
        if not Path(row["image_path"]).is_file():
            raise SystemExit(f"Image file not found: {row['image_path']}")

    for split in SPLITS:
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {split: Counter() for split in SPLITS}
    exclusions = []
    background_with_source_boxes = 0
    background_without_source_label = 0

    for row in rows:
        split = row["split"]
        image = Path(row["image_path"])
        label = image.parent.parent / "labels" / (image.stem + ".txt")
        counts[split]["source_images"] += 1

        if row["class"] == "No Tumor":
            binary_lines = []
            if not label.is_file():
                background_without_source_label += 1
            elif label.read_text().strip():
                background_with_source_boxes += 1
        else:
            binary_lines, reason = inspect_positive_label(
                label, SOURCE_TO_CLASS_ID[row["class"]]
            )
            if reason:
                counts[split]["excluded_positive"] += 1
                counts[split][reason] += 1
                exclusions.append({
                    "split": split,
                    "image": str(image),
                    "label": str(label),
                    "reason": reason,
                })
                continue

        # Use unique image hashes to avoid duplicate filenames across classes.
        stem = row["hash"]
        output_image = args.out / "images" / split / (stem + image.suffix.lower())
        output_label = args.out / "labels" / split / (stem + ".txt")
        if output_image.exists() or output_label.exists():
            raise SystemExit(f"Output collision for {stem}")

        shutil.copy2(image, output_image)
        output_label.write_text("\n".join(binary_lines) + ("\n" if binary_lines else ""))
        counts[split]["exported_images"] += 1
        if binary_lines:
            counts[split]["positive_images"] += 1
            counts[split]["tumor_boxes"] += len(binary_lines)
        else:
            counts[split]["background_images"] += 1

    # Use the data.yaml folder as the dataset root.
    (args.out / "data.yaml").write_text(
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: Tumor\n"
    )

    audit = {
        "policy": "Binary tumor localization; exclude questionable positive annotations",
        "input_images": len(rows),
        "splits": {split: dict(counts[split]) for split in SPLITS},
        "negative_images_with_original_boxes_ignored": background_with_source_boxes,
        "negative_images_missing_original_labels": background_without_source_label,
        "excluded_images": exclusions,
        "note": "Source bounding boxes not clinically verified; patient independence unknown",
    }
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    print("Binary YOLO dataset generated:", args.out)
    for split in SPLITS:
        print(split, dict(counts[split]))
    print("Source No Tumor boxes ignored:", background_with_source_boxes)
    print("Missing source No Tumor labels treated as backgrounds:", background_without_source_label)
    print("Excluded positive images:", len(exclusions))
    print("Training config:", args.out / "data.yaml")
    print("Audit report:", args.out / "audit.json")


if __name__ == "__main__":
    main()
