from pathlib import Path
from collections import Counter
import random
import shutil

SOURCE = Path("vehicle_dataset_clean")
OUTPUT = Path("vehicle_dataset_balanced")
NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
RNG = random.Random(42)

if OUTPUT.exists():
    raise FileExistsError(f"{OUTPUT} already exists; nothing was overwritten.")

records = []
for split in ["train", "val", "test"]:
    image_dir = SOURCE / "images" / split
    label_dir = SOURCE / "labels" / split
    for image in sorted(image_dir.iterdir()):
        if not image.is_file():
            continue
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            continue

        ids = []
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 5:
                try:
                    cls = int(parts[0])
                    if cls in NAMES:
                        ids.append(cls)
                except ValueError:
                    pass

        records.append((image, label, ids))

RNG.shuffle(records)
class_image_counts = Counter(
    cls for _, _, ids in records for cls in set(ids)
)

# Assign images with rare classes first.
records.sort(
    key=lambda r: (
        -sum(1 / class_image_counts[c] for c in set(r[2])),
        -len(set(r[2])),
        RNG.random()
    )
)

total = len(records)
sizes = {
    "train": int(total * 0.70),
    "val": int(total * 0.15),
}
sizes["test"] = total - sizes["train"] - sizes["val"]

splits = {name: [] for name in sizes}
counts = {name: Counter() for name in sizes}

for image, label, ids in records:
    candidates = [s for s in sizes if len(splits[s]) < sizes[s]]

    def score(s):
        # Prefer the split with the greatest unmet class needs.
        deficit = sum(
            max(0, sizes[s] * 0.0 + (1 if class_image_counts[c] >= 3 else 0)
                - counts[s][c])
            for c in set(ids)
        )
        return (deficit, sizes[s] - len(splits[s]))

    chosen = max(candidates, key=score)
    splits[chosen].append((image, label, ids))
    counts[chosen].update(ids)

for split, items in splits.items():
    (OUTPUT / "images" / split).mkdir(parents=True)
    (OUTPUT / "labels" / split).mkdir(parents=True)

    for image, label, _ in items:
        shutil.copy2(image, OUTPUT / "images" / split / image.name)
        shutil.copy2(label, OUTPUT / "labels" / split / label.name)

yaml = (
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
(OUTPUT / "data.yaml").write_text(yaml, encoding="utf-8")

print("Created:", OUTPUT.resolve())
for split, items in splits.items():
    object_counts = Counter(c for _, _, ids in items for c in ids)
    print(f"\n--- {split.upper()} ---")
    print("Images:", len(items))
    print("Images with no target labels:", sum(not ids for _, _, ids in items))
    for cls, name in NAMES.items():
        print(f"{name}: {object_counts[cls]}")
    print("Total objects:", sum(object_counts.values()))
