"""
bootstrap_labels.py — Extract YOLO training data from labeled game footage.

For each game, samples frames from labeled windows (Offense / Transition O / 6v5),
runs pre-trained YOLOv8 person detection, and saves:

  yolo_bootstrap/images/<game>_frame_XXXXXXX.jpg   raw frames
  yolo_bootstrap/labels/<game>_frame_XXXXXXX.txt   YOLO format, class 0 = unknown_player
  yolo_bootstrap/previews/<game>_frame_XXXXXXX.jpg annotated (--test only)

After running on all games, upload images/ + labels/ to Roboflow and relabel each
detected player box as player_dark / player_light / goalkeeper.

Usage:
  python bootstrap_labels.py              # all 6 games, no previews
  python bootstrap_labels.py --test       # Cal only, first 10 min, saves previews
  python bootstrap_labels.py --games Cal,USC
"""

import argparse
import os
import re
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

DATASETS_DIR = Path("datasets")
OUT_DIR = Path("yolo_bootstrap")
DEFAULT_INTERVAL = 5      # seconds between sampled frames
DEFAULT_CONF = 0.3        # YOLO detection confidence threshold
COCO_PERSON_CLASS = 0     # COCO class index for "person"
STUB_CLASS = 0            # output class index (unknown_player — relabeled in Roboflow)
TEST_MAX_DURATION = 600.0 # seconds of video to process in --test mode

ALL_GAMES = ["Cal", "Ford", "LB", "SJSU", "UCLA", "USC"]
TARGET_SUFFIXES = ("Offense", "Transition O", "6v5")


def parse_xml(xml_path: Path) -> list:
    with open(xml_path, "rb") as f:
        raw = f.read()
    content = None
    for enc in ("utf-16", "utf-8"):
        try:
            content = raw.decode(enc)
            break
        except Exception:
            continue
    if content is None:
        raise ValueError(f"Cannot decode {xml_path}")

    instances = []
    for m in re.finditer(
        r"<instance>.*?<start>([\d.]+)</start>.*?<end>([\d.]+)</end>.*?<code>(.*?)</code>",
        content, re.DOTALL,
    ):
        code = m.group(3).strip()
        if any(code.endswith(s) for s in TARGET_SUFFIXES):
            instances.append({
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "code": code,
            })
    return instances


def sample_timestamps(instances: list, interval: float, max_dur: float = None) -> list:
    """Return sorted [(timestamp, code)] sampled every `interval` sec within labeled windows."""
    seen = set()
    result = []
    for inst in instances:
        start = inst["start"]
        end = inst["end"] if max_dur is None else min(inst["end"], max_dur)
        t = start
        while t <= end:
            key = round(t, 1)
            if key not in seen:
                seen.add(key)
                result.append((key, inst["code"]))
            t += interval
    return sorted(result, key=lambda x: x[0])


def extract_frame(cap: cv2.VideoCapture, ts: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
    ret, frame = cap.read()
    return frame if ret else None


def boxes_to_yolo_lines(boxes, img_w: int, img_h: int) -> list:
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{STUB_CLASS} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def draw_boxes(frame: np.ndarray, boxes) -> np.ndarray:
    img = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        conf = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{conf:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def process_game(game: str, model: YOLO, out_images: Path, out_labels: Path,
                 out_previews, interval: float, conf: float, max_dur: float = None) -> dict:
    xml_path = DATASETS_DIR / game / f"{game}.xml"
    mp4_path = DATASETS_DIR / game / f"{game}.mp4"

    if not xml_path.exists() or not mp4_path.exists():
        print(f"  [SKIP] {game} — missing XML or MP4")
        return {"frames": 0, "boxes": 0}

    instances = parse_xml(xml_path)
    if not instances:
        print(f"  [SKIP] {game} — no labeled instances found")
        return {"frames": 0, "boxes": 0}

    cap = cv2.VideoCapture(str(mp4_path))
    timestamps = sample_timestamps(instances, interval, max_dur=max_dur)

    frames_written = 0
    boxes_total = 0

    for ts, code in timestamps:
        frame = extract_frame(cap, ts)
        if frame is None:
            continue

        results = model(frame, classes=[COCO_PERSON_CLASS], conf=conf, verbose=False)
        person_boxes = results[0].boxes if results else []

        if len(person_boxes) == 0:
            continue

        h, w = frame.shape[:2]
        stem = f"{game}_frame_{int(ts * 10):07d}"

        cv2.imwrite(str(out_images / f"{stem}.jpg"), frame)

        label_lines = boxes_to_yolo_lines(person_boxes, w, h)
        with open(out_labels / f"{stem}.txt", "w") as f:
            f.write("\n".join(label_lines))

        if out_previews is not None:
            annotated = draw_boxes(frame, person_boxes)
            cv2.imwrite(str(out_previews / f"{stem}.jpg"), annotated)

        boxes_total += len(person_boxes)
        frames_written += 1

    cap.release()
    return {"frames": frames_written, "boxes": boxes_total}


def main():
    parser = argparse.ArgumentParser(description="Bootstrap YOLO training data from labeled game footage")
    parser.add_argument("--games", default="all",
                        help="Comma-separated game names or 'all' (default: all)")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: Cal only, first 10 min, saves annotated previews")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help=f"Seconds between sampled frames (default: {DEFAULT_INTERVAL})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"YOLO detection confidence threshold (default: {DEFAULT_CONF})")
    args = parser.parse_args()

    if args.test:
        games = ["Cal"]
        max_dur = TEST_MAX_DURATION
        save_previews = True
        print(f"TEST MODE — Cal only, first {int(TEST_MAX_DURATION / 60)} min, saving previews")
    elif args.games == "all":
        games = ALL_GAMES
        max_dur = None
        save_previews = False
    else:
        games = [g.strip() for g in args.games.split(",")]
        max_dur = None
        save_previews = False

    out_images = OUT_DIR / "images"
    out_labels = OUT_DIR / "labels"
    out_previews = OUT_DIR / "previews" if save_previews else None

    for d in [out_images, out_labels]:
        d.mkdir(parents=True, exist_ok=True)
    if out_previews:
        out_previews.mkdir(parents=True, exist_ok=True)

    print("Loading YOLOv8x (COCO pretrained)...")
    model = YOLO("yolov8x.pt")

    total_frames = 0
    total_boxes = 0

    for game in games:
        print(f"\nProcessing {game}...")
        stats = process_game(game, model, out_images, out_labels, out_previews,
                             interval=args.interval, conf=args.conf, max_dur=max_dur)
        print(f"  {stats['frames']} frames  |  {stats['boxes']} person detections")
        total_frames += stats["frames"]
        total_boxes += stats["boxes"]

    print(f"\n{'='*50}")
    print(f"Done.  {total_frames} frames  |  {total_boxes} total detections")
    print(f"Output: {OUT_DIR}/")
    if save_previews:
        print(f"\nReview annotated frames in {out_previews}/")
        print("Verify boxes are tight around players and the count per frame looks right.")
        print("If detection looks good, run without --test to process all games.")
    else:
        print("\nNext steps:")
        print("  1. Upload yolo_bootstrap/images/ + labels/ to Roboflow")
        print("  2. Relabel each box as: player_dark / player_light / goalkeeper")
        print("  3. Export as YOLOv8 format and run yolov8_train.py")


if __name__ == "__main__":
    main()
