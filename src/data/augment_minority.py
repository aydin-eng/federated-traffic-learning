"""Minority-class targeted augmentation for YOLO traffic detection (train split only).

Generates traffic-realistic augmented copies ONLY for frames containing rare classes
(motorcycle=1 or truck=3) so car counts are not re-inflated and val/test remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.prepare_balanced_dataset import (
    CLASS_NAMES,
    cv2_imread,
    cv2_imwrite,
    read_yolo_labels,
    write_metadata_csv,
    write_yolo_labels,
)

RARE_CLASS_IDS = {1, 3}  # motorcycle, truck


def aug_hflip(img: np.ndarray, boxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    flipped = cv2.flip(img, 1)
    out_boxes = [[int(b[0]), max(0.0, min(1.0, 1.0 - float(b[1]))), float(b[2]), float(b[3]), float(b[4])] for b in boxes]
    return flipped, out_boxes


def aug_night(img: np.ndarray, boxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    dark = cv2.convertScaleAbs(img, alpha=0.68, beta=-30)
    return dark, [list(b) for b in boxes]


def aug_sunlight(img: np.ndarray, boxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    bright = cv2.convertScaleAbs(img, alpha=1.18, beta=20)
    return bright, [list(b) for b in boxes]


def aug_rainfog(img: np.ndarray, boxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    fog = cv2.addWeighted(blurred, 0.86, np.full_like(img, 200), 0.14, 0)
    return fog, [list(b) for b in boxes]


def aug_hsv(img: np.ndarray, boxes: List[List[float]], rng: Optional[random.Random] = None) -> Tuple[np.ndarray, List[List[float]]]:
    r = rng or random
    gains = [1.0 + r.uniform(-0.02, 0.02), 1.0 + r.uniform(-0.35, 0.35), 1.0 + r.uniform(-0.25, 0.25)]
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=np.int16)
    lut_h = ((x * gains[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * gains[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * gains[2], 0, 255).astype(np.uint8)
    merged = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR), [list(b) for b in boxes]


RECIPE_CATALOG: List[Tuple[str, Callable[..., Tuple[np.ndarray, List[List[float]]]]]] = [
    ("hflip", aug_hflip),
    ("night", aug_night),
    ("rainfog", aug_rainfog),
    ("sunlight", aug_sunlight),
    ("hsv", aug_hsv),
]


def load_metadata_lookup(meta_path: str) -> Dict[str, Dict[str, str]]:
    """Load existing dataset_metadata.csv indexed by filename."""
    lookup: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(meta_path):
        return lookup
    with open(meta_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["filename"]] = row
    return lookup


def augment_minority_train(
    dataset_dir: str,
    rare_copies: int = 2,
    seed: int = 42,
) -> Tuple[Counter, Counter, int]:
    """Augment train images containing motorcycle or truck; return (before, after, created)."""
    train_img_dir = os.path.join(dataset_dir, "images", "train")
    train_lbl_dir = os.path.join(dataset_dir, "labels", "train")
    if not os.path.isdir(train_img_dir) or not os.path.isdir(train_lbl_dir):
        raise FileNotFoundError(f"Train images/labels not found under {dataset_dir}")

    meta_path = os.path.join(dataset_dir, "dataset_metadata.csv")
    meta_lookup = load_metadata_lookup(meta_path)
    rng = random.Random(seed)

    before_counts: Counter = Counter()
    after_counts: Counter = Counter()
    created_images = 0
    new_meta_rows: List[Dict[str, object]] = []

    image_files = sorted(
        f for f in os.listdir(train_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    for img_name in image_files:
        stem, ext = os.path.splitext(img_name)
        lbl_path = os.path.join(train_lbl_dir, stem + ".txt")
        boxes = read_yolo_labels(lbl_path)
        for b in boxes:
            before_counts[int(b[0])] += 1
            after_counts[int(b[0])] += 1

        # Never re-augment already augmented copies
        if "_aug_" in stem or rare_copies <= 0:
            continue

        present_classes = {int(b[0]) for b in boxes}
        if not (present_classes & RARE_CLASS_IDS):
            continue

        img = cv2_imread(os.path.join(train_img_dir, img_name))
        if img is None:
            continue

        # Ensure hflip is included first when >= 1 copy, then deterministic weather/lighting recipe
        recipes_pool = list(RECIPE_CATALOG[1:])
        rng.shuffle(recipes_pool)
        chosen = [RECIPE_CATALOG[0]] + recipes_pool
        chosen = chosen[: min(rare_copies, len(chosen))]

        orig_meta = meta_lookup.get(img_name, {})
        for rname, rfn in chosen:
            if rname == "hsv":
                aug_img, aug_boxes = aug_hsv(img.copy(), [list(b) for b in boxes], rng=rng)
            else:
                aug_img, aug_boxes = rfn(img.copy(), [list(b) for b in boxes])

            new_img_name = f"{stem}_aug_{rname}{ext}"
            new_lbl_name = f"{stem}_aug_{rname}.txt"
            if not cv2_imwrite(os.path.join(train_img_dir, new_img_name), aug_img):
                continue
            write_yolo_labels(os.path.join(train_lbl_dir, new_lbl_name), aug_boxes)

            cls_c = Counter(int(b[0]) for b in aug_boxes)
            after_counts.update(cls_c)
            created_images += 1
            new_meta_rows.append(
                {
                    "filename": new_img_name,
                    "split": "train",
                    "source": orig_meta.get("source", "augmented"),
                    "sequence": orig_meta.get("sequence", "unknown"),
                    "weather": f"{orig_meta.get('weather', 'unknown')}_{rname}",
                    "camera_state": orig_meta.get("camera_state", "unknown"),
                    "num_car": cls_c.get(0, 0),
                    "num_motorcycle": cls_c.get(1, 0),
                    "num_bus": cls_c.get(2, 0),
                    "num_truck": cls_c.get(3, 0),
                }
            )

    if new_meta_rows and os.path.exists(meta_path):
        write_metadata_csv(meta_path, new_meta_rows, append=True)

    print("=" * 72)
    print("  MINORITY CLASS AUGMENTATION SUMMARY (TRAIN ONLY)")
    print("=" * 72)
    print(f"New augmented train images created: {created_images:,}")
    print(f"{'Class':<15} | {'Before':>10} | {'After':>10} | {'Gain':>8}")
    print("-" * 52)
    for cid in sorted(CLASS_NAMES):
        b_cnt = before_counts.get(cid, 0)
        a_cnt = after_counts.get(cid, 0)
        gain = f"{a_cnt / b_cnt:.2f}x" if b_cnt > 0 else "0.00x"
        print(f"{CLASS_NAMES[cid]:<15} | {b_cnt:>10,} | {a_cnt:>10,} | {gain:>8}")

    car_n = after_counts.get(0, 0)
    for cid in (1, 2, 3):
        cnt = after_counts.get(cid, 0)
        ratio = f"{car_n / cnt:.1f}:1" if cnt > 0 else "inf"
        print(f"Imbalance ratio car:{CLASS_NAMES[cid]} -> {ratio}")
    print("=" * 72)
    return before_counts, after_counts, created_images


def run_self_test() -> None:
    """Validate hflip mirroring, car-only skipping, and val immutability."""
    tmp_root = tempfile.mkdtemp(prefix="aug_selftest_")
    try:
        for split in ("train", "val"):
            os.makedirs(os.path.join(tmp_root, "images", split), exist_ok=True)
            os.makedirs(os.path.join(tmp_root, "labels", split), exist_ok=True)

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Train car-only image -> should get 0 copies
        cv2_imwrite(os.path.join(tmp_root, "images", "train", "car_only.jpg"), dummy_img)
        write_yolo_labels(os.path.join(tmp_root, "labels", "train", "car_only.txt"), [[0, 0.25, 0.5, 0.1, 0.1]])

        # Train rare image (truck + motorcycle) -> should get 2 copies
        cv2_imwrite(os.path.join(tmp_root, "images", "train", "rare_mix.jpg"), dummy_img)
        write_yolo_labels(
            os.path.join(tmp_root, "labels", "train", "rare_mix.txt"),
            [[3, 0.2, 0.4, 0.2, 0.2], [1, 0.8, 0.6, 0.1, 0.1]],
        )

        # Val image -> must stay untouched
        cv2_imwrite(os.path.join(tmp_root, "images", "val", "val_truck.jpg"), dummy_img)
        write_yolo_labels(os.path.join(tmp_root, "labels", "val", "val_truck.txt"), [[3, 0.5, 0.5, 0.2, 0.2]])

        before, after, created = augment_minority_train(tmp_root, rare_copies=2, seed=42)
        assert created == 2, f"Expected 2 created images, got {created}"
        assert before == Counter({0: 1, 1: 1, 3: 1})
        assert after == Counter({0: 1, 1: 3, 3: 3}), f"Car should stay 1, rare 3x: {after}"
        assert len(os.listdir(os.path.join(tmp_root, "images", "val"))) == 1, "Val was modified!"

        hflip_lbl = read_yolo_labels(os.path.join(tmp_root, "labels", "train", "rare_mix_aug_hflip.txt"))
        assert abs(hflip_lbl[0][1] - 0.8) < 1e-5 and abs(hflip_lbl[1][1] - 0.2) < 1e-5
        print("PASS: augment_minority.py self-test succeeded.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment minority classes (motorcycle, truck) in train split.")
    parser.add_argument("--dataset-dir", default="balanced_traffic_dataset", help="Dataset root directory.")
    parser.add_argument("--rare-copies", type=int, default=2, help="Augmented copies per rare-class train image.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic unit self-test and exit.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        run_self_test()
    else:
        augment_minority_train(args.dataset_dir, rare_copies=args.rare_copies, seed=args.seed)
