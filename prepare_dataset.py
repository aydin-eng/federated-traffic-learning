
from pathlib import Path
import random
import shutil

# Source COCO128 dataset
SOURCE = Path("coco128")
OUTPUT = Path("vehicle_dataset_clean")

# Original COCO class IDs -> our class IDs
CLASS_MAP = {
    2: 0,  # car
    3: 1,  # motorcycle
    5: 2,  # bus
    7: 3,  # truck
}

CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SEED = 42

random.seed(SEED)


def main():
    images_dir = SOURCE / "images" / "train2017"
    labels_dir = SOURCE / "labels" / "train2017"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            "COCO128 images or labels folder was not found."
        )

    # Match each image with its label file.
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    pairs = []

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            pairs.append((image_path, label_path))

    # Check label files without matching images.
    image_stems = {image.stem for image, _ in pairs}
    unmatched_labels = [
        p for p in labels_dir.glob("*.txt")
        if p.stem not in image_stems
    ]

    print("Matched image-label pairs:", len(pairs))
    print("Images without labels:", len(image_files) - len(pairs))
    print("Labels without images:", len(unmatched_labels))

    # Split matched pairs into train, validation and test.
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * 0.70)
    val_end = train_end + int(total * 0.15)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    # Create a fresh output folder.
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. "
            "Rename or inspect it before running this script again."
        )

    for split in splits:
        (OUTPUT / "images" / split).mkdir(parents=True)
        (OUTPUT / "labels" / split).mkdir(parents=True)

    object_counts = {name: 0 for name in CLASS_NAMES}
    invalid_lines = []

    for split, split_pairs in splits.items():
        for image_path, label_path in split_pairs:
            new_lines = []

            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                parts = line.split()

                if len(parts) != 5:
                    invalid_lines.append(
                        (label_path.name, line_number, "Expected 5 values")
                    )
                    continue

                try:
                    original_class = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    invalid_lines.append(
                        (label_path.name, line_number, "Invalid number")
                    )
                    continue

                if any(value < 0 or value > 1 for value in coords):
                    invalid_lines.append(
                        (label_path.name, line_number, "Coordinate out of range")
                    )
                    continue

                if original_class in CLASS_MAP:
                    new_class = CLASS_MAP[original_class]
                    new_lines.append(
                        f"{new_class} " + " ".join(parts[1:])
                    )
                    object_counts[CLASS_NAMES[new_class]] += 1

            # Copy images and save only selected vehicle labels.
            shutil.copy2(
                image_path,
                OUTPUT / "images" / split / image_path.name,
            )

            output_label = OUTPUT / "labels" / split / label_path.name
            output_label.write_text(
                "\n".join(new_lines) + ("\n" if new_lines else ""),
                encoding="utf-8",
            )

    # Write dataset configuration.
    yaml_text = (
        f"path: {OUTPUT.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + "".join(
            f"  {index}: {name}\n"
            for index, name in enumerate(CLASS_NAMES)
        )
    )

    (OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("\nSplit sizes:")
    for split, split_pairs in splits.items():
        print(f"{split}: {len(split_pairs)} images")

    print("\nSelected vehicle objects:")
    for name, count in object_counts.items():
        print(f"{name}: {count}")

    print("\nInvalid label lines:", len(invalid_lines))
    print("Dataset YAML:", OUTPUT / "data.yaml")
    print("\nDataset preparation finished.")


if __name__ == "__main__":
    main()
