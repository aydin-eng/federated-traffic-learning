from pathlib import Path
from collections import Counter

p = Path("coco128/labels/train2017")
mapping = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

image_counts = Counter()
object_counts = Counter()
empty = 0

for f in p.glob("*.txt"):
    found = set()

    for line in f.read_text().splitlines():
        parts = line.split()

        if parts and int(parts[0]) in mapping:
            name = mapping[int(parts[0])]
            object_counts[name] += 1
            found.add(name)

    if found:
        for name in found:
            image_counts[name] += 1
    else:
        empty += 1

print("\nImages containing each class:")
for name in mapping.values():
    print(f"{name}: {image_counts[name]} images, {object_counts[name]} objects")

print(f"\nImages with no selected vehicle: {empty}")
