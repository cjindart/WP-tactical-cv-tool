"""
Auto-extract few-shot example frames from labeled dataset XML files.
Run once as a setup step before processing new games.
Pulls EXAMPLES_PER_CLASS frames per code suffix, spread across all available games.
"""
import os
import re
import glob
import random
import xml.etree.ElementTree as ET
import cv2

DATASET_DIR  = os.path.join(os.path.dirname(__file__), "datasets")
FEW_SHOT_DIR = os.path.join(os.path.dirname(__file__), "few_shot_examples")

TARGET_SUFFIXES  = ["Offense", "Transition O", "6v5"]
EXAMPLES_PER_CLASS = 5   # how many frames to keep per suffix


def read_xml_utf16(path):
    with open(path, "rb") as f:
        raw = f.read()
    # Try UTF-16 first, fall back to UTF-8
    for enc in ("utf-16", "utf-8"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_instances(xml_content):
    """Return list of (start, end, code) tuples."""
    instances = []
    for m in re.finditer(
        r"<instance>.*?<start>([\d.]+)</start>.*?<end>([\d.]+)</end>.*?<code>(.*?)</code>",
        xml_content,
        re.DOTALL,
    ):
        start, end, code = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        if end > start:
            instances.append((start, end, code))
    return instances


def extract_frame(video_path, timestamp_sec, out_path, width=512):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False
    h, w = frame.shape[:2]
    new_h = int(h * width / w)
    frame = cv2.resize(frame, (width, new_h))
    cv2.imwrite(out_path, frame)
    return True


def code_suffix(code):
    for suffix in TARGET_SUFFIXES:
        if code.endswith(suffix):
            return suffix
    return None


def candidate_timestamps(start, end):
    """Sample at 25%, 50%, 75% of the instance — three diverse moments."""
    dur = end - start
    return [start + dur * f for f in (0.25, 0.5, 0.75)]


def run():
    # Collect candidate (mp4_path, timestamp, code, game_name) per suffix from every game
    pool = {s: [] for s in TARGET_SUFFIXES}

    for team_dir in sorted(os.listdir(DATASET_DIR)):
        team_path = os.path.join(DATASET_DIR, team_dir)
        if not os.path.isdir(team_path):
            continue
        xml_files = glob.glob(os.path.join(team_path, "*.xml"))
        mp4_files = glob.glob(os.path.join(team_path, "*.mp4"))
        if not xml_files or not mp4_files:
            continue
        xml_path = xml_files[0]
        mp4_path = mp4_files[0]
        print(f"Scanning {team_dir}...")

        content  = read_xml_utf16(xml_path)
        instances = parse_instances(content)

        for start, end, code in instances:
            suffix = code_suffix(code)
            if suffix is None:
                continue
            for ts in candidate_timestamps(start, end):
                pool[suffix].append((mp4_path, ts, code, team_dir))

    # For each suffix, pick EXAMPLES_PER_CLASS spread evenly across games then random
    for suffix in TARGET_SUFFIXES:
        out_dir = os.path.join(FEW_SHOT_DIR, suffix)
        os.makedirs(out_dir, exist_ok=True)

        candidates = pool[suffix]
        if not candidates:
            print(f"  WARNING: no examples found for '{suffix}'")
            continue

        # Prefer diversity: pick up to 1 per game first, then fill randomly
        by_game = {}
        for item in candidates:
            by_game.setdefault(item[3], []).append(item)

        selected = []
        # Round-robin across games until we hit target
        game_lists = [random.sample(v, len(v)) for v in by_game.values()]
        i = 0
        while len(selected) < EXAMPLES_PER_CLASS and any(game_lists):
            lst = game_lists[i % len(game_lists)]
            if lst:
                selected.append(lst.pop(0))
            i += 1

        # Write frames (clear old files first so stale examples don't linger)
        for old in glob.glob(os.path.join(out_dir, "example_*.jpg")):
            os.remove(old)

        written = 0
        for idx, (mp4_path, ts, code, game) in enumerate(selected, 1):
            out_path = os.path.join(out_dir, f"example_{idx}.jpg")
            txt_path = os.path.join(out_dir, f"example_{idx}.txt")
            ok = extract_frame(mp4_path, ts, out_path)
            status = "OK" if ok else "FAILED"
            print(f"  {suffix}/example_{idx}.jpg  [{game} · {code} @ {ts:.1f}s]  {status}")
            if ok:
                # Store only the situation suffix, not the team name —
                # few-shot examples teach visual patterns, not team tendencies
                with open(txt_path, "w") as tf:
                    tf.write(suffix)
                written += 1

        print(f"  → {written}/{EXAMPLES_PER_CLASS} examples written for '{suffix}'")

    print("\nFew-shot extraction complete.")


if __name__ == "__main__":
    run()
