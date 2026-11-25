"""VisDrone2019-DET minority vehicle fusion into the 4-class UA-DETRAC schema.

Imports motorcycle (VisDrone category 10 -> class 1), truck (category 6 -> class 3),
and bus (category 9 -> class 2) examples from elevated traffic-camera angles, masks
VisDrone ignored regions (category 0), and appends records to dataset_metadata.csv.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.prepare_balanced_dataset import (
    CLASS_NAMES,
    cv2_imread,
    cv2_imwrite,
    mask_ignored_and_restore_targets,
    write_metadata_csv,
    write_yolo_labels,
)

VISDRONE_URLS: Dict[str, str] = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
}

# Raw 1-indexed VisDrone2019-DET category IDs -> project 4-class schema
VISDRONE_CATEGORY_MAP: Dict[int, int] = {
    4: 0,   # car -> car
    5: 0,   # van -> car
    10: 1,  # motor -> motorcycle
    9: 2,   # bus -> bus
    6: 3,   # truck -> truck
}

MIN_VISDRONE_SIDE = 8.0
MIN_VISDRONE_AREA = 80.0


def download_and_extract_visdrone(split: str, cache_dir: str) -> str:
    """Download and extract VisDrone2019-DET archive if not already cached."""
    os.makedirs(cache_dir, exist_ok=True)
    url = VISDRONE_URLS[split]
    zip_name = os.path.basename(url)
    zip_path = os.path.join(cache_dir, zip_name)
    extract_dir = os.path.join(cache_dir, os.path.splitext(zip_name)[0])

    if not os.path.exists(extract_dir):
        if not os.path.exists(zip_path):
            print(f"Downloading {zip_name} from Ultralytics assets...")
            urllib.request.urlretrieve(url, zip_path)
        print(f"Extracting {zip_name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)

    # Locate folder containing images/ and annotations/
    if os.path.isdir(os.path.join(extract_dir, "images")) and os.path.isdir(
        os.path.join(extract_dir, "annotations")
    ):
        return extract_dir

    for root, dirs, _ in os.walk(cache_dir):
        if "images" in dirs and "annotations" in dirs and split in os.path.basename(root).lower():
            return root
    raise FileNotFoundError(f"Could not locate VisDrone {split} images/annotations under {cache_dir}")


def parse_visdrone_annotation(
    ann_path: str,
) -> Tuple[List[Tuple[int, float, float, float, float]], List[Tuple[float, float, float, float]]]:
    """Parse one raw VisDrone annotation file into (target_boxes_xywh, ignored_boxes_xywh)."""
    targets: List[Tuple[int, float, float, float, float]] = []
    ignored: List[Tuple[float, float, float, float]] = []

    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().rstrip(",").split(",")
            if len(parts) < 6:
                continue
            try:
                x = float(parts[0])
                y = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])
                score = int(float(parts[4]))
                category = int(float(parts[5]))
            except ValueError:
                continue

            if category == 0:
                if w > 0 and h > 0:
                    ignored.append((x, y, w, h))
                continue

            if score == 0 or category not in VISDRONE_CATEGORY_MAP:
                continue

            if w < MIN_VISDRONE_SIDE or h < MIN_VISDRONE_SIDE or (w * h) < MIN_VISDRONE_AREA:
                continue

            targets.append((VISDRONE_CATEGORY_MAP[category], x, y, w, h))

    return targets, ignored


def select_minority_samples(
    split_dir: str,
    max_samples: int,
    seed: int = 42,
) -> List[Tuple[str, str, List[Tuple[int, float, float, float, float]], List[Tuple[float, float, float, float]]]]:
    """Select VisDrone images prioritizing motorcycle (1) and truck (3), then bus (2)."""
    ann_dir = os.path.join(split_dir, "annotations")
    img_dir = os.path.join(split_dir, "images")
    ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.txt")))

    tier1_motor_or_truck = []
    tier2_bus_only = []

    for ann_p in ann_files:
        stem = os.path.splitext(os.path.basename(ann_p))[0]
        img_p = next(
            (
                cand
                for ext in (".jpg", ".jpeg", ".png")
                for cand in (os.path.join(img_dir, stem + ext),)
                if os.path.exists(cand)
            ),
            None,
        )
        if img_p is None:
            continue

        targets, ignored = parse_visdrone_annotation(ann_p)
        if not targets:
            continue

        cls_set = {t[0] for t in targets}
        if 1 in cls_set or 3 in cls_set:
            tier1_motor_or_truck.append((stem, img_p, targets, ignored))
        elif 2 in cls_set:
            tier2_bus_only.append((stem, img_p, targets, ignored))

    rng = random.Random(seed)
    rng.shuffle(tier1_motor_or_truck)
    rng.shuffle(tier2_bus_only)
    combined = tier1_motor_or_truck + tier2_bus_only
    return combined[:max_samples]


def fuse_visdrone_split(
    split_name: str,
    visdrone_split_dir: str,
    dataset_dir: str,
    max_samples: int,
    mask_ignored: bool = True,
    seed: int = 42,
) -> Tuple[int, Counter, List[Dict[str, object]]]:
    """Convert selected VisDrone samples into YOLO labels and write into dataset_dir."""
    selected = select_minority_samples(visdrone_split_dir, max_samples=max_samples, seed=seed)
    out_img_dir = os.path.join(dataset_dir, "images", split_name)
    out_lbl_dir = os.path.join(dataset_dir, "labels", split_name)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    added_imgs = 0
    box_counts: Counter = Counter()
    metadata_rows: List[Dict[str, object]] = []

    for stem, img_p, targets, ignored in selected:
        img = cv2_imread(img_p)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        yolo_boxes: List[Tuple[int, float, float, float, float]] = []
        target_xyxy: List[Tuple[int, int, int, int]] = []
        frame_cls: Counter = Counter()

        for cid, x, y, w, h in targets:
            x1 = max(0.0, min(float(img_w), x))
            y1 = max(0.0, min(float(img_h), y))
            x2 = max(0.0, min(float(img_w), x + w))
            y2 = max(0.0, min(float(img_h), y + h))
            bw = x2 - x1
            bh = y2 - y1
            if bw < MIN_VISDRONE_SIDE or bh < MIN_VISDRONE_SIDE:
                continue

            xc = max(0.0, min(1.0, (x1 + bw / 2.0) / img_w))
            yc = max(0.0, min(1.0, (y1 + bh / 2.0) / img_h))
            wn = max(0.001, min(1.0, bw / img_w))
            hn = max(0.001, min(1.0, bh / img_h))
            yolo_boxes.append((cid, xc, yc, wn, hn))
            target_xyxy.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
            frame_cls[cid] += 1

        if not yolo_boxes:
            continue

        if mask_ignored and ignored:
            img = mask_ignored_and_restore_targets(img, ignored, target_xyxy)

        out_fname = f"visdrone_{stem}.jpg"
        if not cv2_imwrite(os.path.join(out_img_dir, out_fname), img):
            continue
        write_yolo_labels(os.path.join(out_lbl_dir, f"visdrone_{stem}.txt"), yolo_boxes)

        added_imgs += 1
        box_counts.update(frame_cls)
        metadata_rows.append(
            {
                "filename": out_fname,
                "split": split_name,
                "source": "visdrone",
                "sequence": "visdrone",
                "weather": "visdrone",
                "camera_state": "elevated",
                "num_car": frame_cls.get(0, 0),
                "num_motorcycle": frame_cls.get(1, 0),
                "num_bus": frame_cls.get(2, 0),
                "num_truck": frame_cls.get(3, 0),
            }
        )

    return added_imgs, box_counts, metadata_rows


def run_self_test() -> None:
    """Validate VisDrone category mapping, box bounds, ignored masking, and metadata output."""
    tmp_root = tempfile.mkdtemp(prefix="visdrone_selftest_")
    try:
        vis_train = os.path.join(tmp_root, "visdrone_src", "train")
        os.makedirs(os.path.join(vis_train, "images"), exist_ok=True)
        os.makedirs(os.path.join(vis_train, "annotations"), exist_ok=True)

        # Image 1: contains ignored (0), pedestrian (1 -> skip), car (4 -> 0), motor (10 -> 1), score=0 (skip)
        img1 = np.full((200, 300, 3), 240, dtype=np.uint8)
        cv2_imwrite(os.path.join(vis_train, "images", "sample1.jpg"), img1)
        with open(os.path.join(vis_train, "annotations", "sample1.txt"), "w", encoding="utf-8") as f:
            f.write("10,10,50,50,0,0,0,0\n")     # ignored region -> masked to (114, 114, 114)
            f.write("20,20,30,40,1,1,0,0\n")     # pedestrian -> dropped
            f.write("80,50,40,30,1,4,0,0\n")     # car -> class 0
            f.write("150,80,25,25,1,10,0,0\n")   # motor -> class 1
            f.write("200,100,40,40,0,6,0,0\n")   # score=0 -> dropped

        # Image 2: contains van (5 -> 0), truck (6 -> 3), bus (9 -> 2)
        img2 = np.full((200, 300, 3), 220, dtype=np.uint8)
        cv2_imwrite(os.path.join(vis_train, "images", "sample2.jpg"), img2)
        with open(os.path.join(vis_train, "annotations", "sample2.txt"), "w", encoding="utf-8") as f:
            f.write("10,20,40,30,1,5,0,0\n")     # van -> class 0
            f.write("70,40,60,50,1,6,0,0\n")     # truck -> class 3
            f.write("160,60,70,50,1,9,0,0\n")    # bus -> class 2

        ds_dir = os.path.join(tmp_root, "dataset")
        added, counts, rows = fuse_visdrone_split(
            "train", vis_train, ds_dir, max_samples=10, mask_ignored=True, seed=42
        )
        assert added == 2, f"Expected 2 fused images, got {added}"
        assert counts == Counter({0: 2, 1: 1, 2: 1, 3: 1}), f"Unexpected counts: {counts}"
        assert len(rows) == 2

        # Verify label bounds and exact masked pixel values on sample1
        out_img1 = cv2_imread(os.path.join(ds_dir, "images", "train", "visdrone_sample1.jpg"))
        assert out_img1 is not None
        assert np.mean(np.abs(out_img1[25, 25].astype(np.int32) - np.array([114, 114, 114]))) < 5
        assert np.mean(np.abs(out_img1[90, 160].astype(np.int32) - np.array([240, 240, 240]))) < 5

        with open(os.path.join(ds_dir, "labels", "train", "visdrone_sample1.txt"), "r", encoding="utf-8") as f:
            lines = [ln.strip().split() for ln in f if ln.strip()]
        assert [int(x[0]) for x in lines] == [0, 1], f"Unexpected class IDs: {lines}"
        for ln in lines:
            for val in map(float, ln[1:]):
                assert 0.0 <= val <= 1.0, f"Out-of-bounds coord: {val}"

        write_metadata_csv(os.path.join(ds_dir, "dataset_metadata.csv"), rows, append=True)
        assert os.path.exists(os.path.join(ds_dir, "dataset_metadata.csv"))
        print("PASS: fuse_external_vehicles.py self-test succeeded.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse VisDrone2019-DET minority vehicles into traffic dataset.")
    parser.add_argument("--dataset-dir", default="balanced_traffic_dataset", help="Target YOLO dataset directory.")
    parser.add_argument("--cache-dir", default="visdrone_cache", help="Directory to cache VisDrone archives.")
    parser.add_argument("--max-train", type=int, default=2000, help="Max VisDrone images to fuse into train.")
    parser.add_argument("--max-val", type=int, default=350, help="Max VisDrone images to fuse into val.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-ignored", dest="mask_ignored", action="store_true", default=True)
    parser.add_argument("--no-mask-ignored", dest="mask_ignored", action="store_false")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic unit self-test and exit.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    all_rows: List[Dict[str, object]] = []
    print("=" * 72)
    print("  VISDRONE2019-DET MINORITY VEHICLE FUSION")
    print("=" * 72)
    for split_name, max_n in (("train", args.max_train), ("val", args.max_val)):
        if max_n <= 0:
            continue
        vis_dir = download_and_extract_visdrone(split_name, args.cache_dir)
        added, counts, rows = fuse_visdrone_split(
            split_name,
            vis_dir,
            args.dataset_dir,
            max_samples=max_n,
            mask_ignored=args.mask_ignored,
            seed=args.seed,
        )
        all_rows.extend(rows)
        summary = ", ".join(f"{CLASS_NAMES[cid]}={counts.get(cid, 0)}" for cid in sorted(CLASS_NAMES))
        print(f"[{split_name.upper():<5}] Fused {added:,} VisDrone images -> {summary}")

    if all_rows:
        meta_path = os.path.join(args.dataset_dir, "dataset_metadata.csv")
        write_metadata_csv(meta_path, all_rows, append=True)
        print(f"Updated metadata CSV: {meta_path} (+{len(all_rows)} rows)")
    print("=" * 72)


if __name__ == "__main__":
    main()
