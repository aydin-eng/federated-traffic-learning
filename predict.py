
from pathlib import Path
from ultralytics import YOLO

# Load the trained model
MODEL_PATH = "runs/detect/train-4/weights/best.pt"

model = YOLO(MODEL_PATH)

# Ask the user for an image path
image_path = input("Enter the image path: ").strip().strip('"')

if not Path(image_path).is_file():
    print("Error: Image file not found.")
else:
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25,
        save=True
    )

    print("\nDetection completed.")
    print("The result image is saved in the runs/detect folder.")

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            print(
                f"Class: {class_name}, "
                f"Confidence: {confidence:.2f}"
            )
