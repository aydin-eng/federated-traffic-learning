from pathlib import Path
from collections import Counter

root = Path("vehicle_dataset_clean")
names = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}

for split in ["train", "val", "test"]:
    labels_dir = root / "labels" / split
    counts = Counter()
    empty = 0

    for path in labels_dir.glob("*.txt"):
        lines = path.read_text(encoding="utf-8").splitlines()
        valid_lines = [line for line in lines if line.strip()]

        if not valid_lines:
            empty += 1

        for line in valid_lines:
            parts = line.split()
            if len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    if class_id in names:
                        counts[class_id] += 1
                except ValueError:
                    pass

    print(f"\n--- {split.upper()} ---")
    print("Images:", len(list((root / "images" / split).glob("*"))))
    print("Images with no target labels:", empty)

    for class_id, name in names.items():
        print(f"{name}: {counts[class_id]}")

    print("Total objects:", sum(counts.values()))
