from pathlib import Path
from collections import Counter
import random
import shutil

SOURCE = Path("vehicle_dataset_clean")
OUTPUT = Path("vehicle_dataset_balanced_v2")
CLASSES = [0, 1, 2, 3]
NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
rng = random.Random(42)

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Existing files were not changed."
    )

# Read every image and its labels from the current dataset.
records = []
for split in ["train", "val", "test"]:
    image_dir = SOURCE / "images" / split
    label_dir = SOURCE / "labels" / split

    for image in sorted(image_dir.iterdir()):
        if not image.is_file():
            continue

        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            raise FileNotFoundError(f"Missing label file: {label}")

        ids = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                cls = int(parts[0])
                if cls in CLASSES:
                    ids.append(cls)

        records.append({
            "image": image,
            "label": label,
            "ids": ids,
            "classes": set(ids),
        })

total = len(records)
sizes = {
    "train": int(total * 0.70),
    "val": int(total * 0.15),
}
sizes["test"] = total - sizes["train"] - sizes["val"]

# Count how many distinct images contain each class.
class_image_counts = Counter(
    cls for record in records for cls in record["classes"]
)

# Target number of images containing each class in each split.
targets = {
    split: {
        cls: class_image_counts[cls] * FRACTIONS[split]
        for cls in CLASSES
    }
    for split in FRACTIONS
}

# Rare-class images are considered first.
rng.shuffle(records)
records.sort(
    key=lambda r: (
        min(
            (class_image_counts[c] for c in r["classes"]),
            default=10**9
        ),
        -len(r["classes"])
    )
)

assigned = {split: [] for split in FRACTIONS}
assigned_counts = {split: Counter() for split in FRACTIONS}

for record in records:
    available = [
        split for split in FRACTIONS
        if len(assigned[split]) < sizes[split]
    ]

    def score(split):
        # Prefer the split with the largest relative class deficit.
        class_score = sum(
            max(0.0, targets[split][cls] - assigned_counts[split][cls])
            / max(1.0, targets[split][cls])
            for cls in record["classes"]
        )
        capacity = (sizes[split] - len(assigned[split])) / sizes[split]
        return class_score + capacity * 0.01

    chosen = max(available, key=score)
    assigned[chosen].append(record)
    assigned_counts[chosen].update(record["classes"])

# Copy files into a new dataset.
for split, items in assigned.items():
    (OUTPUT / "images" / split).mkdir(parents=True)
    (OUTPUT / "labels" / split).mkdir(parents=True)

    for record in items:
        shutil.copy2(
            record["image"],
            OUTPUT / "images" / split / record["image"].name
        )
        shutil.copy2(
            record["label"],
            OUTPUT / "labels" / split / record["label"].name
        )

yaml_text = (
    f"path: {OUTPUT.resolve().as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    "  0: car\n"
    "  1: motorcycle\n"
    "  2: bus\n"
    "  3: truck\n"
)
(OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

# Report instance counts and image counts for each class.
print("Dataset created:", OUTPUT.resolve())

for split, items in assigned.items():
    instance_counts = Counter(
        cls for record in items for cls in record["ids"]
    )
    image_counts = Counter(
        cls for record in items for cls in record["classes"]
    )
    empty_images = sum(not record["ids"] for record in items)

    print(f"\n--- {split.upper()} ---")
    print("Images:", len(items))
    print("Images without target objects:", empty_images)

    for cls in CLASSES:
        print(
            f"{NAMES[cls]}: {instance_counts[cls]} objects "
            f"in {image_counts[cls]} images"
        )

    print("Total objects:", sum(instance_counts.values()))
