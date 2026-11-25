"""Balanced UA-DETRAC dataset builder with stratified video splits and ignored-region masking.

Fixes class imbalance by replacing uniform frame striding with class-aware adaptive sampling,
masks UA-DETRAC <ignored_region> annotations to prevent false-positive background penalties,
and exports sequence/weather metadata for downstream Federated Learning simulations.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CLASS_NAMES: Dict[int, str] = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
}

DETRAC_VEHICLE_MAP: Dict[str, int] = {
    "car": 0,
    "van": 0,
    "bus": 2,
    "others": 3,
}

METADATA_COLUMNS: List[str] = [
    "filename",
    "split",
    "source",
    "sequence",
    "weather",
    "camera_state",
    "num_car",
    "num_motorcycle",
    "num_bus",
    "num_truck",
]

DEFAULT_IMG_W = 960.0
DEFAULT_IMG_H = 540.0
MIN_BOX_PX = 5.0
PAD_GRAY = (114, 114, 114)


@dataclass
class PixelBox:
    """Bounding box in pixel coordinates."""

    cls_id: int
    left: float
    top: float
    width: float
    height: float

    def to_xyxy(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(img_w, int(round(self.left))))
        y1 = max(0, min(img_h, int(round(self.top))))
        x2 = max(0, min(img_w, int(round(self.left + self.width))))
        y2 = max(0, min(img_h, int(round(self.top + self.height))))
        return x1, y1, x2, y2

    def to_yolo(self, img_w: float, img_h: float) -> Tuple[int, float, float, float, float]:
        x1 = max(0.0, min(img_w, self.left))
        y1 = max(0.0, min(img_h, self.top))
        x2 = max(0.0, min(img_w, self.left + self.width))
        y2 = max(0.0, min(img_h, self.top + self.height))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        xc = max(0.0, min(1.0, (x1 + bw / 2.0) / img_w))
        yc = max(0.0, min(1.0, (y1 + bh / 2.0) / img_h))
        wn = max(0.001, min(1.0, bw / img_w))
        hn = max(0.001, min(1.0, bh / img_h))
        return (self.cls_id, xc, yc, wn, hn)


@dataclass
class SequenceData:
    """Parsed UA-DETRAC XML sequence."""

    name: str
    weather: str
    camera_state: str
    ignored_regions: List[Tuple[float, float, float, float]]
    frames: Dict[int, List[PixelBox]]
    class_counts: Counter


def cv2_imread(path: str) -> Optional[np.ndarray]:
    """Unicode-safe OpenCV image reader (supports non-ASCII paths on Windows)."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def cv2_imdecode_bytes(raw: bytes) -> Optional[np.ndarray]:
    """Decode raw image bytes with OpenCV."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def cv2_imwrite(path: str, img: np.ndarray) -> bool:
    """Unicode-safe OpenCV image writer."""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def mask_ignored_and_restore_targets(
    img: np.ndarray,
    ignored_xywh: Iterable[Tuple[float, float, float, float]],
    target_xyxy: Iterable[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Mask ignored regions with neutral pad gray while preserving labeled target pixels."""
    regions = list(ignored_xywh)
    if not regions:
        return img
    h, w = img.shape[:2]
    masked = img.copy()
    for left, top, width, height in regions:
        x1 = max(0, min(w, int(round(left))))
        y1 = max(0, min(h, int(round(top))))
        x2 = max(0, min(w, int(round(left + width))))
        y2 = max(0, min(h, int(round(top + height))))
        if x2 > x1 and y2 > y1:
            masked[y1:y2, x1:x2] = PAD_GRAY

    # Restore labeled vehicles in case an ignored box touches or overlaps a valid target
    for tx1, ty1, tx2, ty2 in target_xyxy:
        if tx2 > tx1 and ty2 > ty1:
            masked[ty1:ty2, tx1:tx2] = img[ty1:ty2, tx1:tx2]
    return masked


def read_yolo_labels(path: str) -> List[List[float]]:
    """Read YOLO format label file into [[cls_id, x, y, w, h], ...]."""
    boxes: List[List[float]] = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(
                    [
                        int(float(parts[0])),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    ]
                )
    return boxes


def write_yolo_labels(path: str, boxes: Sequence[Sequence[float]]) -> None:
    """Write normalized YOLO bounding boxes to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write(f"{int(b[0])} {float(b[1]):.6f} {float(b[2]):.6f} {float(b[3]):.6f} {float(b[4]):.6f}\n")


def write_metadata_csv(path: str, rows: Sequence[Dict[str, object]], append: bool = False) -> None:
    """Write or append dataset_metadata.csv rows."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    mode = "a" if (append and file_exists) else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_data_yaml(out_dir: str) -> str:
    """Write Ultralytics YOLO data.yaml for the 4-class traffic schema."""
    yaml_path = os.path.join(out_dir, "data.yaml")
    norm_path = os.path.abspath(out_dir).replace("\\", "/")
    content = (
        "# Balanced Multi-Source Traffic Vehicle Detection Dataset\n"
        f"path: {norm_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 4\n"
        "names:\n"
        "  0: car\n"
        "  1: motorcycle\n"
        "  2: bus\n"
        "  3: truck\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return yaml_path


def parse_detrac_xml(xml_path: str) -> SequenceData:
    """Parse one UA-DETRAC XML annotation file including weather and ignored regions."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    seq_name = root.attrib.get("name", os.path.splitext(os.path.basename(xml_path))[0]).strip()

    seq_attr = root.find("sequence_attribute")
    weather = "unknown"
    camera_state = "unknown"
    if seq_attr is not None:
        # Official DETRAC XMLs spell this attribute 'sence_weather'
        weather = (
            seq_attr.attrib.get("sence_weather")
            or seq_attr.attrib.get("scene_weather")
            or "unknown"
        ).strip().lower()
        camera_state = seq_attr.attrib.get("camera_state", "unknown").strip().lower()

    ignored_regions: List[Tuple[float, float, float, float]] = []
    ir_elem = root.find("ignored_region")
    if ir_elem is not None:
        for bx in ir_elem.findall("box"):
            try:
                l = float(bx.attrib.get("left", 0.0))
                t = float(bx.attrib.get("top", 0.0))
                w = float(bx.attrib.get("width", 0.0))
                h = float(bx.attrib.get("height", 0.0))
                if w > 0 and h > 0:
                    ignored_regions.append((l, t, w, h))
            except (TypeError, ValueError):
                continue

    frames: Dict[int, List[PixelBox]] = {}
    class_counts: Counter = Counter()
    for fe in root.iter("frame"):
        n_str = fe.attrib.get("num")
        if not n_str:
            continue
        fnum = int(n_str)
        boxes: List[PixelBox] = []
        tl = fe.find("target_list")
        if tl is not None:
            for tgt in tl.findall("target"):
                bx = tgt.find("box")
                att = tgt.find("attribute")
                if bx is None or att is None:
                    continue
                vt = att.attrib.get("vehicle_type", "car").strip().lower()
                if vt not in DETRAC_VEHICLE_MAP:
                    continue
                cid = DETRAC_VEHICLE_MAP[vt]
                try:
                    l = float(bx.attrib.get("left", 0.0))
                    t = float(bx.attrib.get("top", 0.0))
                    w = float(bx.attrib.get("width", 0.0))
                    h = float(bx.attrib.get("height", 0.0))
                except (TypeError, ValueError):
                    continue
                if w < MIN_BOX_PX or h < MIN_BOX_PX:
                    continue
                boxes.append(PixelBox(cls_id=cid, left=l, top=t, width=w, height=h))
                class_counts[cid] += 1
        if boxes:
            frames[fnum] = boxes

    return SequenceData(
        name=seq_name,
        weather=weather,
        camera_state=camera_state,
        ignored_regions=ignored_regions,
        frames=frames,
        class_counts=class_counts,
    )


def stratified_video_split(
    sequences: List[SequenceData],
    val_ratio: float = 0.18,
    seed: int = 42,
) -> Tuple[List[SequenceData], List[SequenceData]]:
    """Split video sequences into train/val with zero overlap and stratified rare classes."""
    rng = random.Random(seed)
    truck_seqs: List[SequenceData] = []
    bus_seqs: List[SequenceData] = []
    car_seqs: List[SequenceData] = []

    for s in sorted(sequences, key=lambda x: x.name):
        if s.class_counts.get(3, 0) > 0:
            truck_seqs.append(s)
        elif s.class_counts.get(2, 0) > 0:
            bus_seqs.append(s)
        else:
            car_seqs.append(s)

    train_out: List[SequenceData] = []
    val_out: List[SequenceData] = []

    for bucket, min_val in ((truck_seqs, 4), (bus_seqs, 3), (car_seqs, 1)):
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        if not shuffled:
            continue
        n_val = max(min_val if len(shuffled) > min_val else 0, int(round(len(shuffled) * val_ratio)))
        n_val = min(n_val, max(0, len(shuffled) - 1))
        val_out.extend(shuffled[:n_val])
        train_out.extend(shuffled[n_val:])

    return train_out, val_out


def select_adaptive_frames(
    seq: SequenceData,
    truck_step: int,
    bus_step: int,
    car_step: int,
    max_car_only: Optional[int],
) -> List[int]:
    """Select frame indices using higher sampling density for rare classes (truck, bus)."""
    sorted_nums = sorted(seq.frames.keys())
    truck_frames = [n for n in sorted_nums if any(b.cls_id == 3 for b in seq.frames[n])]
    bus_frames = [
        n
        for n in sorted_nums
        if any(b.cls_id == 2 for b in seq.frames[n]) and not any(b.cls_id == 3 for b in seq.frames[n])
    ]
    car_only_frames = [
        n
        for n in sorted_nums
        if all(b.cls_id == 0 for b in seq.frames[n])
    ]

    selected = set(truck_frames[:: max(1, truck_step)])
    selected.update(bus_frames[:: max(1, bus_step)])

    sampled_car = car_only_frames[:: max(1, car_step)]
    if max_car_only is not None and max_car_only >= 0:
        sampled_car = sampled_car[:max_car_only]
    selected.update(sampled_car)

    return sorted(selected)


def load_xml_sequences(annot_dir: str) -> Tuple[List[SequenceData], List[SequenceData]]:
    """Load all train and test XML sequences from DETRAC_annotations."""
    train_xmls = sorted(glob.glob(os.path.join(annot_dir, "train", "*.xml")))
    test_xmls = sorted(glob.glob(os.path.join(annot_dir, "test", "*.xml")))
    if not train_xmls:
        raise FileNotFoundError(f"No train XML files found under: {os.path.join(annot_dir, 'train')}")
    return [parse_detrac_xml(p) for p in train_xmls], [parse_detrac_xml(p) for p in test_xmls]


def build_balanced_dataset(args: argparse.Namespace) -> Dict[str, Counter]:
    """Run dry-run analysis or full balanced dataset extraction."""
    train_all, test_seqs = load_xml_sequences(args.annot_dir)
    train_seqs, val_seqs = stratified_video_split(train_all, val_ratio=args.val_ratio, seed=args.seed)

    # Leakage verification
    train_names = {s.name for s in train_seqs}
    val_names = {s.name for s in val_seqs}
    overlap = train_names & val_names
    if overlap:
        raise RuntimeError(f"Data leakage detected between train and val sequences: {overlap}")

    if args.max_sequences is not None:
        train_seqs = train_seqs[: args.max_sequences]
        val_seqs = val_seqs[: args.max_sequences]
        test_seqs = test_seqs[: args.max_sequences]

    split_configs = {
        "train": (train_seqs, args.train_truck_step, args.train_bus_step, args.train_car_step, args.max_car_only),
        "val": (val_seqs, args.val_truck_step, args.val_bus_step, args.val_car_step, args.max_val_car_only),
        "test": (test_seqs, args.test_step, args.test_step, args.test_step, None),
    }

    split_box_counts: Dict[str, Counter] = {k: Counter() for k in split_configs}
    split_img_counts: Counter = Counter()
    metadata_rows: List[Dict[str, object]] = []

    zf: Optional[zipfile.ZipFile] = None
    zip_keys: set = set()
    if not args.dry_run:
        if os.path.exists(args.out_dir):
            shutil.rmtree(args.out_dir)
        for split in split_configs:
            os.makedirs(os.path.join(args.out_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(args.out_dir, "labels", split), exist_ok=True)

        if args.images_zip:
            zf = zipfile.ZipFile(args.images_zip, "r")
            zip_keys = set(zf.namelist())
        elif not args.images_dir:
            raise ValueError("Specify --images-zip, --images-dir, or --dry-run.")

    try:
        for split_name, (seqs, t_step, b_step, c_step, c_cap) in split_configs.items():
            for seq in seqs:
                frame_nums = select_adaptive_frames(seq, t_step, b_step, c_step, c_cap)
                if args.max_frames_per_seq is not None:
                    frame_nums = frame_nums[: args.max_frames_per_seq]

                for fnum in frame_nums:
                    boxes = seq.frames[fnum]
                    if args.dry_run:
                        split_img_counts[split_name] += 1
                        for b in boxes:
                            split_box_counts[split_name][b.cls_id] += 1
                        continue

                    img: Optional[np.ndarray] = None
                    if zf is not None:
                        cand_keys = [
                            f"DETRAC-Images/{seq.name}/img{fnum:05d}.jpg",
                            f"{seq.name}/img{fnum:05d}.jpg",
                        ]
                        matched_key = next((k for k in cand_keys if k in zip_keys), None)
                        if matched_key is None:
                            continue
                        img = cv2_imdecode_bytes(zf.read(matched_key))
                    else:
                        cand_paths = [
                            os.path.join(args.images_dir, seq.name, f"img{fnum:05d}.jpg"),
                            os.path.join(args.images_dir, "DETRAC-Images", seq.name, f"img{fnum:05d}.jpg"),
                        ]
                        src_path = next((p for p in cand_paths if os.path.exists(p)), None)
                        if src_path is None:
                            continue
                        img = cv2_imread(src_path)

                    if img is None:
                        continue

                    img_h, img_w = img.shape[:2]
                    if args.mask_ignored and seq.ignored_regions:
                        target_xyxy = [b.to_xyxy(img_w, img_h) for b in boxes]
                        img = mask_ignored_and_restore_targets(img, seq.ignored_regions, target_xyxy)

                    fname = f"{seq.name}_img{fnum:05d}"
                    out_img = os.path.join(args.out_dir, "images", split_name, f"{fname}.jpg")
                    out_lbl = os.path.join(args.out_dir, "labels", split_name, f"{fname}.txt")

                    if not cv2_imwrite(out_img, img):
                        continue
                    yolo_boxes = [b.to_yolo(float(img_w), float(img_h)) for b in boxes]
                    write_yolo_labels(out_lbl, yolo_boxes)

                    frame_cls = Counter(b.cls_id for b in boxes)
                    split_img_counts[split_name] += 1
                    split_box_counts[split_name].update(frame_cls)
                    metadata_rows.append(
                        {
                            "filename": f"{fname}.jpg",
                            "split": split_name,
                            "source": "ua_detrac",
                            "sequence": seq.name,
                            "weather": seq.weather,
                            "camera_state": seq.camera_state,
                            "num_car": frame_cls.get(0, 0),
                            "num_motorcycle": frame_cls.get(1, 0),
                            "num_bus": frame_cls.get(2, 0),
                            "num_truck": frame_cls.get(3, 0),
                        }
                    )
    finally:
        if zf is not None:
            zf.close()

    if not args.dry_run:
        write_data_yaml(args.out_dir)
        write_metadata_csv(os.path.join(args.out_dir, "dataset_metadata.csv"), metadata_rows, append=False)

    # Print summary report
    print("=" * 72)
    print("  BALANCED UA-DETRAC DATASET SUMMARY" + (" [DRY-RUN]" if args.dry_run else ""))
    print("=" * 72)
    print(f"Train sequences: {len(train_seqs)} | Val sequences: {len(val_seqs)} | Test sequences: {len(test_seqs)}")
    print(f"Train/Val sequence overlap: {len(overlap)} (zero-leakage verified)")
    for split_name in ("train", "val", "test"):
        c = split_box_counts[split_name]
        total_b = sum(c.values())
        car_n = c.get(0, 0)
        bus_n = c.get(2, 0)
        truck_n = c.get(3, 0)
        ratio_ct = f"{car_n / truck_n:.1f}:1" if truck_n > 0 else "inf"
        ratio_cb = f"{car_n / bus_n:.1f}:1" if bus_n > 0 else "inf"
        print(
            f"[{split_name.upper():<5}] imgs={split_img_counts[split_name]:>5,} | "
            f"boxes={total_b:>6,} | car={car_n:>6,} | bus={bus_n:>5,} | truck={truck_n:>5,} | "
            f"car:truck={ratio_ct}, car:bus={ratio_cb}"
        )
    print("=" * 72)
    return split_box_counts


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build class-balanced UA-DETRAC YOLO dataset.")
    parser.add_argument("--annot-dir", default="DETRAC_annotations", help="Directory containing train/ and test/ XMLs.")
    parser.add_argument("--images-zip", default=None, help="Path to DETRAC-Images.zip.")
    parser.add_argument("--images-dir", default=None, help="Path to extracted DETRAC-Images directory.")
    parser.add_argument("--out-dir", default="balanced_traffic_dataset", help="Output dataset directory.")
    parser.add_argument("--val-ratio", type=float, default=0.18, help="Fraction of train sequences allocated to val.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified sequence split.")
    parser.add_argument("--train-truck-step", type=int, default=2, help="Stride over train frames containing truck.")
    parser.add_argument("--train-bus-step", type=int, default=15, help="Stride over train frames containing bus.")
    parser.add_argument("--train-car-step", type=int, default=120, help="Stride over train car-only frames.")
    parser.add_argument("--max-car-only", type=int, default=3, help="Max car-only frames per train sequence.")
    parser.add_argument("--val-truck-step", type=int, default=2, help="Stride over val frames containing truck.")
    parser.add_argument("--val-bus-step", type=int, default=15, help="Stride over val frames containing bus.")
    parser.add_argument("--val-car-step", type=int, default=120, help="Stride over val car-only frames.")
    parser.add_argument("--max-val-car-only", type=int, default=3, help="Max car-only frames per val sequence.")
    parser.add_argument("--test-step", type=int, default=20, help="Stride over test frames.")
    parser.add_argument("--mask-ignored", dest="mask_ignored", action="store_true", default=True)
    parser.add_argument("--no-mask-ignored", dest="mask_ignored", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Analyze splits and counts without extracting images.")
    parser.add_argument("--max-sequences", type=int, default=None, help="Cap sequences per split (for testing).")
    parser.add_argument("--max-frames-per-seq", type=int, default=None, help="Cap frames per sequence (for testing).")
    return parser.parse_args(argv)


if __name__ == "__main__":
    build_balanced_dataset(parse_args())
