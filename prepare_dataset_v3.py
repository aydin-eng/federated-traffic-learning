from pathlib import Path
from collections import Counter
import random
import shutil

random.seed(42)

SOURCE_IMAGES = Path("coco128/images/train2017")
SOURCE_LABELS = Path("coco128/labels/train2017")
OUTPUT = Path("vehicle_dataset_v3")

CLASS_MAP = {2: 0, 3: 1, 5: 2, 7: 3}
CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]
SPLITS = ["train", "val", "test"]

images = []
for image in SOURCE_IMAGES.iterdir():
    if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    label = SOURCE_LABELS / (image.stem + ".txt")
    if label.exists():
        images.append((image, label))

records = []
for image, label in images:
    lines = []
    present = set()

    for line in label.read_text().splitlines():
        parts = line.split()
        if parts and int(parts[0]) in CLASS_MAP:
            class_id = CLASS_MAP[int(parts[0])]
            lines.append(" ".join([str(class_id)] + parts[1:]))
            present.add(class_id)

    records.append({
        "image": image,
        "text": "\n".join(lines) + ("\n" if lines else ""),
        "present": present
    })

n = len(records)
capacity = {
    "train": round(n * 0.70),
    "val": round(n * 0.15)
}
capacity["test"] = n - capacity["train"] - capacity["val"]

class_images = Counter()
for r in records:
    class_images.update(r["present"])

assigned = {s: [] for s in SPLITS}
used = set()

# Reserve examples for every class in every split.
# Rare classes are handled first.
class_order = sorted(range(4), key=lambda c: class_images[c])
split_order = ["val", "test", "train"]

for c in class_order:
    for split in split_order:
        if any(c in r["present"] for r in assigned[split]):
            continue

        candidates = [
            r for r in records
            if id(r) not in used and c in r["present"]
        ]

        if not candidates or len(assigned[split]) >= capacity[split]:
            continue

        # Prefer an image that also contains other rare classes.
        candidates.sort(
            key=lambda r: (
                -sum(1 / max(class_images[x], 1) for x in r["present"]),
                random.random()
            )
        )

        chosen = candidates[0]
        assigned[split].append(chosen)
        used.add(id(chosen))

# Assign all remaining vehicle-containing images.
remaining = [r for r in records if id(r) not in used]
remaining.sort(
    key=lambda r: (
        not bool(r["present"]),
        -sum(1 / max(class_images[x], 1) for x in r["present"]),
        random.random()
    )
)

for r in remaining:
    available = [s for s in SPLITS if len(assigned[s]) < capacity[s]]
    if not available:
        raise RuntimeError("No available split capacity.")

    def score(s):
        class_need = sum(
            max(class_images[c] * {"train": .70, "val": .15, "test": .15}[s]
                - sum(c in x["present"] for x in assigned[s]), 0)
            / max(class_images[c], 1)
            for c in r["present"]
        )
        size_need = capacity[s] - len(assigned[s])
        return class_need + size_need / max(capacity[s], 1)

    chosen_split = max(available, key=score)
    assigned[chosen_split].append(r)

# Recreate the output folders.
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

for split in SPLITS:
    (OUTPUT / "images" / split).mkdir(parents=True)
    (OUTPUT / "labels" / split).mkdir(parents=True)

    for r in assigned[split]:
        shutil.copy2(r["image"], OUTPUT / "images" / split / r["image"].name)
        (OUTPUT / "labels" / split / (r["image"].stem + ".txt")).write_text(r["text"])

yaml = (
    f"path: {OUTPUT.resolve().as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n\n"
    "names:\n"
    "  0: car\n"
    "  1: motorcycle\n"
    "  2: bus\n"
    "  3: truck\n"
)
(OUTPUT / "data.yaml").write_text(yaml)

print(f"Matched image-label pairs: {n}")
print(f"Target sizes: {capacity}")

for split in SPLITS:
    subset = assigned[split]
    objects = Counter()
    class_image_counts = Counter()
    empty = 0

    for r in subset:
        if not r["present"]:
            empty += 1
        class_image_counts.update(r["present"])
        for line in r["text"].splitlines():
            if line:
                objects[CLASS_NAMES[int(line.split()[0])]] += 1

    print(f"\n{split.upper()}: {len(subset)} images")
    print(f"Images without selected vehicles: {empty}")
    for name in CLASS_NAMES:
        print(f"  {name}: {class_image_counts[name]} images, {objects[name]} objects")

print("\nCreated:", OUTPUT.resolve())
