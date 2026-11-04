from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import yaml

root = Path("vehicle_dataset_clean")
image_path = root / "images" / "val" / "000000000471.jpg"
label_path = root / "labels" / "val" / "000000000471.txt"
yaml_path = root / "data.yaml"

with open(yaml_path, encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data["names"]
names = {int(k): v for k, v in names.items()}

if not image_path.exists() or not label_path.exists():
    raise FileNotFoundError("Bus image or label file was not found.")

image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)
width, height = image.size

for line in label_path.read_text(encoding="utf-8").splitlines():
    parts = line.split()
    cls = int(parts[0])
    xc, yc, w, h = map(float, parts[1:])

    x1 = (xc - w / 2) * width
    y1 = (yc - h / 2) * height
    x2 = (xc + w / 2) * width
    y2 = (yc + h / 2) * height

    name = names.get(cls, f"class_{cls}")
    draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
    draw.text((x1, max(0, y1 - 22)), name, fill="red")

output = Path("bus_label_check.jpg")
image.save(output)
print("Image size:", image.size)
print("Class names:", names)
print("Saved:", output.resolve())
