import os
import shutil
import random
import cv2
import numpy as np
from collections import Counter

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

SOURCE_DIR = r"C:\Users\Aydin.DESKTOP-PKIS5JG\OneDrive\Masaüstü\federated-traffic-learning\vehicle_dataset_v3"
TARGET_DIR = r"C:\Users\Aydin.DESKTOP-PKIS5JG\OneDrive\Masaüstü\federated-traffic-learning\vehicle_dataset_v3_augmented"

CLASS_NAMES = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}

# ----------------- AUGMENTATION FUNCTIONS -----------------

def flip_horizontal(img, boxes):
    # img: [H, W, C]
    # boxes: list of [cls_id, x, y, w, h]
    flipped_img = cv2.flip(img, 1)
    new_boxes = []
    for cls_id, x, y, w, h in boxes:
        new_x = 1.0 - x
        new_boxes.append([cls_id, new_x, y, w, h])
    return flipped_img, new_boxes

def adjust_lighting(img, alpha=1.0, beta=0):
    # alpha: contrast [0.5 - 1.5], beta: brightness [-50, +50]
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def apply_motion_blur(img, kernel_size=9):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)

def apply_fog_blur(img, ksize=7):
    # Gaussian blur + contrast softening for foggy/rainy traffic lens
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return cv2.addWeighted(blurred, 0.85, np.full_like(img, 200), 0.15, 0)

def adjust_hsv(img, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

def add_partial_occlusion(img, boxes, max_cutouts=2):
    # Add random small cutout rectangles that simulate roadside signs / partial occlusion
    out = img.copy()
    h, w = img.shape[:2]
    num_cuts = random.randint(1, max_cutouts)
    for _ in range(num_cuts):
        ch = random.randint(int(h * 0.05), int(h * 0.12))
        cw = random.randint(int(w * 0.05), int(w * 0.12))
        cy = random.randint(0, h - ch)
        cx = random.randint(0, w - cw)
        color = (random.randint(40, 90), random.randint(40, 90), random.randint(40, 90))
        out[cy:cy+ch, cx:cx+cw] = color
    return out, boxes

def slight_zoom(img, boxes, zoom_factor=1.10):
    # Slight center zoom (scale=1.10)
    h, w = img.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    cropped = img[y1:y1+new_h, x1:x1+new_w]
    scaled_img = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    new_boxes = []
    for cls_id, bx, by, bw, bh in boxes:
        # Convert norm to pixel
        px = bx * w
        py = by * h
        pw = bw * w
        ph = bh * h
        
        # Shift relative to crop
        npx = (px - x1) * zoom_factor
        npy = (py - y1) * zoom_factor
        npw = pw * zoom_factor
        nph = ph * zoom_factor
        
        # Back to norm
        nbx = npx / w
        nby = npy / h
        nbw = npw / w
        nbh = nph / h
        
        # Check boundary
        if 0 < nbx < 1 and 0 < nby < 1 and nbw > 0.01 and nbh > 0.01:
            new_boxes.append([cls_id, min(max(nbx, 0.01), 0.99), min(max(nby, 0.01), 0.99), min(nbw, 1.0), min(nbh, 1.0)])
    return scaled_img, new_boxes

# ----------------- PIPELINES DEFINITIONS -----------------

def pipeline_hflip(img, boxes):
    return flip_horizontal(img, boxes)

def pipeline_night(img, boxes):
    # Darken for dusk/night camera conditions
    return adjust_lighting(img, alpha=0.65, beta=-35), boxes

def pipeline_sunlight(img, boxes):
    # High glare daylight
    return adjust_lighting(img, alpha=1.2, beta=25), boxes

def pipeline_rain_fog(img, boxes):
    return apply_fog_blur(img, ksize=7), boxes

def pipeline_motion(img, boxes):
    return apply_motion_blur(img, kernel_size=9), boxes

def pipeline_hsv(img, boxes):
    return adjust_hsv(img), boxes

def pipeline_occlusion(img, boxes):
    return add_partial_occlusion(img, boxes)

def pipeline_zoom_light(img, boxes):
    z_img, z_boxes = slight_zoom(img, boxes, zoom_factor=1.12)
    return adjust_lighting(z_img, alpha=0.9, beta=10), z_boxes

def pipeline_combo_night_flip(img, boxes):
    f_img, f_boxes = flip_horizontal(img, boxes)
    return adjust_lighting(f_img, alpha=0.6, beta=-30), f_boxes

def pipeline_combo_fog_flip(img, boxes):
    f_img, f_boxes = flip_horizontal(img, boxes)
    return apply_fog_blur(f_img, ksize=5), f_boxes

def cv2_imread(path):
    with open(path, 'rb') as f:
        bytes_arr = bytearray(f.read())
        np_arr = np.asarray(bytes_arr, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def cv2_imwrite(path, img):
    ext = os.path.splitext(path)[1]
    is_success, buffer = cv2.imencode(ext, img)
    if is_success:
        with open(path, 'wb') as f:
            f.write(buffer)
        return True
    return False

# ----------------- MAIN AUGMENTATION RUNNER -----------------

def read_yolo_labels(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes

def save_yolo_labels(path, boxes):
    with open(path, 'w', encoding='utf-8') as f:
        for b in boxes:
            f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

def main():
    print("=" * 65)
    print("  VEHICLE DATASET AUGMENTATION PIPELINE (YOLO FORMAT)  ")
    print("=" * 65)

    # Recreate target directory
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
        
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(TARGET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, 'labels', split), exist_ok=True)

    # 1. Copy val and test untouched
    for split in ['val', 'test']:
        src_img_dir = os.path.join(SOURCE_DIR, 'images', split)
        src_lbl_dir = os.path.join(SOURCE_DIR, 'labels', split)
        tgt_img_dir = os.path.join(TARGET_DIR, 'images', split)
        tgt_lbl_dir = os.path.join(TARGET_DIR, 'labels', split)
        
        for f in os.listdir(src_img_dir):
            shutil.copy2(os.path.join(src_img_dir, f), os.path.join(tgt_img_dir, f))
        for f in os.listdir(src_lbl_dir):
            shutil.copy2(os.path.join(src_lbl_dir, f), os.path.join(tgt_lbl_dir, f))
            
    print("[1/3] Validation & Test sets copied untouched (Ground Truth maintained).")

    # 2. Process train split with class-aware augmentation
    train_src_img = os.path.join(SOURCE_DIR, 'images', 'train')
    train_src_lbl = os.path.join(SOURCE_DIR, 'labels', 'train')
    train_tgt_img = os.path.join(TARGET_DIR, 'images', 'train')
    train_tgt_lbl = os.path.join(TARGET_DIR, 'labels', 'train')

    before_counts = Counter()
    after_counts = Counter()
    total_original_images = 0
    total_augmented_created = 0

    image_files = [f for f in os.listdir(train_src_img) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_original_images = len(image_files)

    for img_name in image_files:
        stem = os.path.splitext(img_name)[0]
        ext = os.path.splitext(img_name)[1]
        
        img_path = os.path.join(train_src_img, img_name)
        lbl_path = os.path.join(train_src_lbl, stem + '.txt')
        
        img = cv2_imread(img_path)
        if img is None:
            continue
        boxes = read_yolo_labels(lbl_path)
        
        # Track initial counts
        for b in boxes:
            before_counts[CLASS_NAMES[b[0]]] += 1
            after_counts[CLASS_NAMES[b[0]]] += 1

        # Save original into target
        shutil.copy2(img_path, os.path.join(train_tgt_img, img_name))
        save_yolo_labels(os.path.join(train_tgt_lbl, stem + '.txt'), boxes)

        # Determine classes present
        classes_present = {b[0] for b in boxes}
        has_rare_class = any(c in classes_present for c in [1, 2, 3]) # motorcycle, bus, truck
        has_car = 0 in classes_present

        # Select recipes
        selected_recipes = []
        if has_rare_class:
            # Heavy augmentation to balance underrepresented vehicle classes
            selected_recipes = [
                ("hflip", pipeline_hflip),
                ("night", pipeline_night),
                ("sunlight", pipeline_sunlight),
                ("rainfog", pipeline_rain_fog),
                ("motion", pipeline_motion),
                ("hsv", pipeline_hsv),
                ("occlusion", pipeline_occlusion),
                ("zoom", pipeline_zoom_light),
                ("nightflip", pipeline_combo_night_flip),
                ("fogflip", pipeline_combo_fog_flip)
            ]
        elif has_car:
            # Moderate augmentation for cars
            selected_recipes = [
                ("hflip", pipeline_hflip),
                ("night", pipeline_night),
                ("rainfog", pipeline_rain_fog)
            ]
        else:
            # Background image: 1 environmental variant to double negative samples safely
            selected_recipes = [
                ("night", pipeline_night)
            ]

        # Apply each recipe
        for recipe_name, recipe_fn in selected_recipes:
            aug_img, aug_boxes = recipe_fn(img.copy(), [b.copy() for b in boxes])
            new_img_name = f"{stem}_aug_{recipe_name}{ext}"
            new_lbl_name = f"{stem}_aug_{recipe_name}.txt"
            
            cv2_imwrite(os.path.join(train_tgt_img, new_img_name), aug_img)
            save_yolo_labels(os.path.join(train_tgt_lbl, new_lbl_name), aug_boxes)
            
            for b in aug_boxes:
                after_counts[CLASS_NAMES[b[0]]] += 1
            total_augmented_created += 1

    # 3. Create data.yaml
    data_yaml_content = """path: vehicle_dataset_v3_augmented
train: images/train
val: images/val
test: images/test

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
"""
    with open(os.path.join(TARGET_DIR, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    print("[2/3] Data augmentation completed.")
    print("[3/3] Created vehicle_dataset_v3_augmented/data.yaml")
    
    total_train_final = total_original_images + total_augmented_created
    print("\n" + "=" * 65)
    print("  AUGMENTATION SUMMARY RESULTS")
    print("=" * 65)
    print(f"Original Train Images:     {total_original_images}")
    print(f"New Augmented Images:      {total_augmented_created}")
    print(f"Total Train Images:        {total_train_final} (Artış: ~%{int((total_train_final/total_original_images)*100)})")
    print("\nClass-Level Bounding Box Distribution (Nesne Sayıları):")
    print(f"{'Class Name':<15} | {'Original Train':<16} | {'Augmented Train':<16} | {'Multiplier':<10}")
    print("-" * 65)
    for c_id, c_name in CLASS_NAMES.items():
        b = before_counts[c_name]
        a = after_counts[c_name]
        mult = f"{a/b:.1f}x" if b > 0 else "0x"
        print(f"{c_name:<15} | {b:<16} | {a:<16} | {mult:<10}")
    print("=" * 65)

if __name__ == "__main__":
    main()
