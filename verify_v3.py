from pathlib import Path
from collections import Counter

root = Path("vehicle_dataset_v3")
names = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}

for split in ["train", "val", "test"]:
    folder = root / "labels" / split
    image_counts = Counter()
    object_counts = Counter()

    for file in folder.glob("*.txt"):
        found = set()

        for line in file.read_text().splitlines():
            parts = line.split()
            if parts and int(parts[0]) in names:
                cls = int(parts[0])
                found.add(cls)
                object_counts[names[cls]] += 1

        for cls in found:
            image_counts[names[cls]] += 1

    print(f"\n{split.upper()}")
    for name in names.values():
        print(f"{name}: {image_counts[name]} images, {object_counts[name]} objects")
