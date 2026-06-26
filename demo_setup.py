"""
Creates a 2-minute demo clip from Cal.mp4 and a mock classifier
that returns ground-truth labels from Cal.xml for those timestamps.
Run once; output goes to demo/ folder.
"""
import os
import re
import cv2

BASE_DIR = os.path.dirname(__file__)
DEMO_DIR = os.path.join(BASE_DIR, "demo")
CLIP_PATH = os.path.join(DEMO_DIR, "demo_clip.mp4")

DEMO_START = 136.5
DEMO_END = 256.5
DATASET_XML = os.path.join(BASE_DIR, "datasets", "Cal", "Cal.xml")
DATASET_MP4 = os.path.join(BASE_DIR, "datasets", "Cal", "Cal.mp4")


def create_demo_clip():
    os.makedirs(DEMO_DIR, exist_ok=True)
    if os.path.exists(CLIP_PATH):
        return CLIP_PATH

    cap = cv2.VideoCapture(DATASET_MP4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(CLIP_PATH, fourcc, fps, (w, h))

    start_frame = int(DEMO_START * fps)
    end_frame = int(DEMO_END * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Demo clip written: {CLIP_PATH}")
    return CLIP_PATH


def load_ground_truth():
    """Parse Cal.xml and return instances clipped to demo window, offset to t=0."""
    with open(DATASET_XML, "rb") as f:
        content = f.read().decode("utf-16")

    instances = []
    for m in re.finditer(
        r"<start>([\d.]+)</start>.*?<end>([\d.]+)</end>.*?<code>(.*?)</code>",
        content, re.DOTALL
    ):
        start = float(m.group(1)) - DEMO_START
        end = float(m.group(2)) - DEMO_START
        code = m.group(3).strip()
        if end > 0 and start < (DEMO_END - DEMO_START):
            start = max(0.0, start)
            end = min(DEMO_END - DEMO_START, end)
            instances.append((start, end, code))

    return instances


def label_at_time(ts: float, instances: list, team1: str, team2: str):
    """
    Given a timestamp (relative to demo start) and ground-truth instances,
    return the most specific matching label from our 6 target codes.
    Priority: 6v5 > Transition O > Offense
    """
    # Remap team names: original is California vs Stanford
    name_map = {"California": team1, "Stanford": team2}

    matched = []
    for start, end, code in instances:
        if start <= ts <= end:
            matched.append(code)

    # Priority: 6v5 first
    for code in matched:
        for orig, mapped in name_map.items():
            if code == f"{orig} 6v5":
                return f"{mapped} 6v5"
    for code in matched:
        for orig, mapped in name_map.items():
            if code == f"{orig} Transition O":
                return f"{mapped} Transition O"
    for code in matched:
        for orig, mapped in name_map.items():
            if code == f"{orig} Offense":
                return f"{mapped} Offense"

    return "neutral"


if __name__ == "__main__":
    create_demo_clip()
    gt = load_ground_truth()
    print(f"Loaded {len(gt)} ground-truth instances in demo window")
    for s, e, c in gt:
        print(f"  {s:.1f}-{e:.1f}: {c}")
