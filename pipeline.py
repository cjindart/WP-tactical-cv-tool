"""
Core water polo film tagging pipeline.
"""
import os
import re
import base64
import json
import time
import logging
import math
from typing import Callable

import cv2
import yaml
import anthropic

BASE_DIR = os.path.dirname(__file__)
FEW_SHOT_DIR = os.path.join(BASE_DIR, "few_shot_examples")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
REVIEW_LOG = os.path.join(BASE_DIR, "review.log")

DEFAULT_COLOR1 = "#cc0000"
DEFAULT_COLOR2 = "#0044cc"


def hex_to_sportscode(hex_color: str):
    """Convert #rrggbb to Sportscode 0-65535 RGB tuple."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c*2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    scale = 65535 / 255
    return (int(r * scale), int(g * scale), int(b * scale))


def hex_to_color_name(hex_color: str) -> str:
    """Approximate a hex color as a human-readable name for the Claude prompt."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c*2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    mx = max(r, g, b)
    if mx < 50:
        return "black"
    if min(r, g, b) > 200:
        return "white"
    hue_vals = [(r, "red"), (g, "green"), (b, "blue")]
    dom = max(hue_vals, key=lambda x: x[0])
    if dom[0] < 80:
        return "dark " + dom[1]
    if r > 200 and g > 150 and b < 80:
        return "yellow"
    if r > 200 and g < 100 and b > 150:
        return "pink/magenta"
    return dom[1]


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Frame diff scan
# ---------------------------------------------------------------------------

def compute_frame_diffs(video_path: str, progress_cb: Callable = None):
    """Return dict mapping second -> mean absolute diff to previous sampled frame."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    sample_every = max(1, int(fps))  # 1 frame per second for diff scan
    diffs = {}
    prev_gray = None
    frame_idx = 0
    second = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = float(cv2.absdiff(gray, prev_gray).mean())
                diffs[second] = diff
            prev_gray = gray
            second += 1
            if progress_cb and second % 30 == 0:
                progress_cb("diff_scan", second / max(duration, 1))
        frame_idx += 1

    cap.release()
    return diffs, duration


# ---------------------------------------------------------------------------
# Smart frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, diffs: dict, duration: float,
                   config: dict, progress_cb: Callable = None):
    """
    Extract frames at variable rate:
      - stable zones (low diff): 1 frame / stable_sample_interval sec
      - motion zones (high diff): 1 frame / motion_sample_interval sec
    Returns list of (timestamp_sec, jpeg_bytes).
    """
    cfg = config
    threshold = cfg["frame_diff_threshold"]
    stable_iv = cfg.get("stable_sample_interval", 15)
    motion_iv = cfg.get("motion_sample_interval", 5)

    # Short clips: sample at motion rate throughout — too few frames otherwise
    if duration < 600:
        stable_iv = motion_iv

    # Decide which seconds are "motion"
    motion_seconds = {s for s, d in diffs.items() if d >= threshold}

    # Build sample timestamps
    timestamps = []
    t = 0.0
    while t < duration:
        sec = int(t)
        iv = motion_iv if sec in motion_seconds else stable_iv
        timestamps.append(t)
        t += iv

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frames = []
    total = len(timestamps)
    for i, ts in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        new_w = 512
        new_h = int(h * new_w / w)
        frame = cv2.resize(frame, (new_w, new_h))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frames.append((ts, buf.tobytes()))
        if progress_cb and i % 10 == 0:
            progress_cb("extract_frames", i / max(total, 1))

    cap.release()
    return frames


def extract_single_frame(video_path: str, ts: float):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    h, w = frame.shape[:2]
    new_w = 512
    new_h = int(h * new_w / w)
    frame = cv2.resize(frame, (new_w, new_h))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Few-shot image loading
# ---------------------------------------------------------------------------

def load_few_shot_images():
    """Return dict: suffix -> list of base64 jpeg strings."""
    result = {}
    for suffix in ["Offense", "Transition O", "6v5"]:
        d = os.path.join(FEW_SHOT_DIR, suffix)
        imgs = []
        if os.path.isdir(d):
            for fname in sorted(os.listdir(d)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    with open(os.path.join(d, fname), "rb") as f:
                        imgs.append(base64.standard_b64encode(f.read()).decode())
        result[suffix] = imgs
    return result


# ---------------------------------------------------------------------------
# Claude classifier
# ---------------------------------------------------------------------------

def build_system_prompt(team1: str, team2: str, few_shot: dict,
                        color1: str = None, color2: str = None):
    color1_name = hex_to_color_name(color1) if color1 else None
    color2_name = hex_to_color_name(color2) if color2 else None

    cap_hint1 = f" (wearing {color1_name} caps)" if color1_name else ""
    cap_hint2 = f" (wearing {color2_name} caps)" if color2_name else ""

    lines = [
        "You are a water polo tactical analyst. Classify each video frame into exactly one label.",
        "",
        f"Teams: Team1 = '{team1}'{cap_hint1}, Team2 = '{team2}'{cap_hint2}",
    ]
    if color1_name or color2_name:
        lines += [
            "",
            "CRITICAL — use cap color as your PRIMARY signal for team identification:",
        ]
        if color1_name:
            lines.append(f"  {team1} players wear {color1_name} caps.")
        if color2_name:
            lines.append(f"  {team2} players wear {color2_name} caps.")
        lines += [
            "  To classify possession: identify which team's caps are clustered in the attacking half.",
            "  To classify 6v5: count caps by color — the team with 6 caps in the frame is the attacking team.",
            f"  If you see 6 {color1_name} caps and 5 {color2_name} caps attacking, label '{team1} 6v5'.",
            f"  If you see 6 {color2_name} caps and 5 {color1_name} caps attacking, label '{team2} 6v5'.",
            "  Do NOT guess team identity from jersey numbers or context — only use cap color.",
        ]

    lines += [
        "",
        "Labels and visual cues:",
        f"  '{team1} Offense': {team1} in half-court set, players positioned at 2-meter and perimeter, deliberate movement, goalie set.",
        f"  '{team1} Transition O': {team1} players sprinting up pool, disorganized defense, fast ball movement toward goal.",
        f"  '{team1} 6v5': {team1} has numerical advantage — 6 attackers vs 5 defenders, spread formation, player isolated at 2-meter. Always occurs within an Offense window.",
        f"  '{team2} Offense': {team2} in half-court set.",
        f"  '{team2} Transition O': {team2} players sprinting up pool.",
        f"  '{team2} 6v5': {team2} has 6v5 man-up advantage.",
        "  'neutral': dead ball, sprint at center, referee interaction, or ambiguous — no instance written.",
        "",
        "Return a JSON array — one object per frame in the order given:",
        '  [{"label": "<label>", "confidence": <0.0-1.0>}, ...]',
        "",
        "Only output valid JSON. No markdown, no explanation.",
    ]
    return "\n".join(lines)


def build_user_message(frames_b64: list, few_shot: dict, team1: str, team2: str):
    """Build Anthropic message content with optional few-shot images then query frames."""
    content = []

    # Few-shot block — show what each situation type looks like visually.
    # Labels are situation suffixes only (Offense / Transition O / 6v5), never team names,
    # so Claude learns water polo patterns, not team tendencies.
    # Cap color in the system prompt handles team identification separately.
    for suffix, imgs in few_shot.items():
        if not imgs:
            continue
        content.append({"type": "text", "text": f"Example frames labeled '{suffix}':"})
        for b64 in imgs:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
            })

    content.append({
        "type": "text",
        "text": f"Now classify these {len(frames_b64)} frames in order:",
    })
    for b64 in frames_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    return content


def classify_batch(client: anthropic.Anthropic, frames_b64: list,
                   system_prompt: str, few_shot: dict,
                   team1: str, team2: str, model="claude-sonnet-4-6"):
    content = build_user_message(frames_b64, few_shot, team1, team2)
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    text = response.content[0].text.strip()
    # Extract JSON array from response
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array in response: {text[:200]}")
    results = json.loads(m.group(0))
    usage = response.usage
    return results, usage


def classify_frames(client: anthropic.Anthropic, frames: list,
                    team1: str, team2: str, config: dict,
                    few_shot: dict, color1: str = None, color2: str = None,
                    progress_cb: Callable = None):
    """
    Classify all frames. Returns list of (timestamp, label, confidence).
    Also returns (total_api_calls, input_tokens, output_tokens).
    """
    batch_size = config.get("batch_size", 4)
    system_prompt = build_system_prompt(team1, team2, few_shot, color1, color2)
    conf_threshold = config.get("confidence_threshold", 0.6)

    results = []
    total_calls = 0
    total_input = 0
    total_output = 0

    review_log = open(REVIEW_LOG, "w")
    raw_log_path = REVIEW_LOG.replace("review.log", "raw_predictions.log")
    raw_log = open(raw_log_path, "w")
    raw_log.write(f"System prompt:\n{system_prompt}\n\n{'='*60}\n\n")

    batches = [frames[i:i+batch_size] for i in range(0, len(frames), batch_size)]
    for bi, batch in enumerate(batches):
        timestamps = [f[0] for f in batch]
        frames_b64 = [base64.standard_b64encode(f[1]).decode() for f in batch]

        try:
            preds, usage = classify_batch(client, frames_b64, system_prompt, few_shot, team1, team2)
        except Exception as e:
            logging.warning(f"Batch {bi} failed: {e}, defaulting to neutral")
            preds = [{"label": "neutral", "confidence": 0.0}] * len(batch)
            usage = type("u", (), {"input_tokens": 0, "output_tokens": 0})()

        total_calls += 1
        total_input += getattr(usage, "input_tokens", 0)
        total_output += getattr(usage, "output_tokens", 0)

        for ts, pred in zip(timestamps, preds):
            label = pred.get("label", "neutral")
            conf = float(pred.get("confidence", 0.0))
            raw_log.write(f"t={ts:7.2f}s  raw={label:<30} conf={conf:.2f}\n")
            if conf < conf_threshold and label != "neutral":
                review_log.write(f"LOW_CONF t={ts:.2f} label={label} conf={conf:.2f}\n")
                label = "neutral"
            results.append((ts, label, conf))

        if progress_cb:
            progress_cb("classify", (bi + 1) / max(len(batches), 1))

    review_log.close()
    raw_log.write(f"\n{'='*60}\nTotal frames: {len(results)}\n")
    raw_log.close()
    return results, (total_calls, total_input, total_output)


# ---------------------------------------------------------------------------
# Transition resampling
# ---------------------------------------------------------------------------

def resample_transitions(video_path: str, predictions: list,
                         team1: str, team2: str, config: dict,
                         client: anthropic.Anthropic, few_shot: dict,
                         color1: str = None, color2: str = None,
                         progress_cb: Callable = None):
    """
    At each label-change point, densely resample ±transition_window sec at
    motion_sample_interval. Classify those extra frames and merge into predictions.
    """
    cfg = config
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()

    # Short clips are already fully sampled — resampling only adds noise
    if duration < 600:
        return predictions, (0, 0, 0)

    window = cfg.get("transition_window", 30)
    dense_iv = cfg.get("motion_sample_interval", 5)
    conf_threshold = cfg.get("confidence_threshold", 0.6)
    batch_size = cfg.get("batch_size", 4)
    system_prompt = build_system_prompt(team1, team2, few_shot, color1, color2)

    # Find transition timestamps
    transition_ts = []
    for i in range(1, len(predictions)):
        if predictions[i][1] != predictions[i-1][1]:
            transition_ts.append(predictions[i][0])

    if not transition_ts:
        return predictions, (0, 0, 0)

    # Build dense sample timestamps near transitions
    existing_ts = {p[0] for p in predictions}
    extra_ts = set()
    raw_log_path = os.path.join(BASE_DIR, "raw_predictions.log")
    raw_log = open(raw_log_path, "a")

    for t in transition_ts:
        for dt in range(-window, window + 1, dense_iv):
            candidate = max(0.0, min(duration, t + dt))
            if candidate not in existing_ts:
                extra_ts.add(round(candidate, 3))

    extra_ts = sorted(extra_ts)
    extra_frames = []
    for ts in extra_ts:
        data = extract_single_frame(video_path, ts)
        if data:
            extra_frames.append((ts, data))

    if not extra_frames:
        return predictions, (0, 0, 0)

    total_calls = 0
    total_input = 0
    total_output = 0
    extra_preds = []

    batches = [extra_frames[i:i+batch_size] for i in range(0, len(extra_frames), batch_size)]
    for bi, batch in enumerate(batches):
        timestamps = [f[0] for f in batch]
        frames_b64 = [base64.standard_b64encode(f[1]).decode() for f in batch]
        try:
            preds, usage = classify_batch(client, frames_b64, system_prompt, few_shot, team1, team2)
        except Exception as e:
            logging.warning(f"Transition batch {bi} failed: {e}")
            preds = [{"label": "neutral", "confidence": 0.0}] * len(batch)
            usage = type("u", (), {"input_tokens": 0, "output_tokens": 0})()
        total_calls += 1
        total_input += getattr(usage, "input_tokens", 0)
        total_output += getattr(usage, "output_tokens", 0)
        for ts, pred in zip(timestamps, preds):
            label = pred.get("label", "neutral")
            conf = float(pred.get("confidence", 0.0))
            raw_log.write(f"RESAMPLE t={ts:7.2f}s  raw={label:<30} conf={conf:.2f}\n")
            if conf < conf_threshold:
                label = "neutral"
            extra_preds.append((ts, label, conf))
        if progress_cb:
            progress_cb("resample", (bi + 1) / max(len(batches), 1))

    raw_log.write(f"\nResample added {len(extra_preds)} frames (window={window:.1f}s)\n")
    raw_log.close()
    all_preds = sorted(predictions + extra_preds, key=lambda x: x[0])
    return all_preds, (total_calls, total_input, total_output)


# ---------------------------------------------------------------------------
# Segment merging + smoothing
# ---------------------------------------------------------------------------

def smooth_predictions(predictions: list, window: int = 2):
    """
    Multi-pass smoothing: replace any run of ≤window frames that is
    surrounded on both sides by the same label. Repeat until stable.
    """
    preds = list(predictions)
    changed = True
    while changed:
        changed = False
        i = 1
        while i < len(preds) - 1:
            # Find the extent of a contiguous run with a different label
            run_label = preds[i][1]
            prev_label = preds[i - 1][1]
            if run_label == prev_label:
                i += 1
                continue
            # Find end of this run
            j = i
            while j < len(preds) and preds[j][1] == run_label:
                j += 1
            run_len = j - i
            next_label = preds[j][1] if j < len(preds) else None
            if run_len <= window and prev_label == next_label:
                for k in range(i, j):
                    preds[k] = (preds[k][0], prev_label, preds[k][2])
                changed = True
            i = j
    return preds


def merge_segments(predictions: list, min_duration: float = 8.0):
    """Merge consecutive same-label frames into (start, end, label) segments.

    Each segment's end is set to the next different-label frame's timestamp so
    single-frame segments get a real duration instead of 0.
    """
    if not predictions:
        return []
    raw = []
    start_ts, cur_label, _ = predictions[0]

    for i in range(1, len(predictions)):
        ts, label, conf = predictions[i]
        if label != cur_label:
            # Segment ends where the next label begins
            if cur_label != "neutral":
                raw.append((start_ts, ts, cur_label))
            start_ts = ts
            cur_label = label

    # Last segment: extend to end of video isn't available, use the final timestamp
    if cur_label != "neutral":
        raw.append((start_ts, predictions[-1][0], cur_label))

    return [(s, e, l) for s, e, l in raw if e - s >= min_duration]


# ---------------------------------------------------------------------------
# XML writer
# ---------------------------------------------------------------------------

def write_xml(segments: list, team1: str, team2: str,
              color1: str = None, color2: str = None):
    """
    Build Sportscode XML string (UTF-8) from segments.
    For 6v5 segments, write both a [Team] 6v5 instance AND a [Team] Offense instance.
    """
    instance_id = 1
    instances_xml = []

    for start, end, label in segments:
        if end <= start:   # skip zero-duration artifacts
            continue
        # Determine if this is a 6v5 label
        is_6v5 = label.endswith("6v5")

        if is_6v5:
            # Offense instance first, then 6v5
            team = team1 if label.startswith(team1) else team2
            offense_code = f"{team} Offense"
            instances_xml.append(_instance_xml(instance_id, start, end, offense_code))
            instance_id += 1
            instances_xml.append(_instance_xml(instance_id, start, end, label))
            instance_id += 1
        else:
            instances_xml.append(_instance_xml(instance_id, start, end, label))
            instance_id += 1

    rows_xml = _rows_xml(team1, team2, color1, color2)

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<file>\n"
        "<ALL_INSTANCES>\n"
        + "".join(instances_xml)
        + "</ALL_INSTANCES>\n"
        + rows_xml
        + "</file>"
    )
    return xml


def _instance_xml(id_: int, start: float, end: float, code: str):
    return (
        "<instance>\n"
        f"<ID>{id_}</ID>\n"
        f"<start>{start:.3f}</start>\n"
        f"<end>{end:.3f}</end>\n"
        f"<code>{_escape(code)}</code>\n"
        "</instance>\n"
    )


def _escape(s: str):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _rows_xml(team1: str, team2: str, color1: str = None, color2: str = None):
    c1 = hex_to_sportscode(color1) if color1 else hex_to_sportscode(DEFAULT_COLOR1)
    c2 = hex_to_sportscode(color2) if color2 else hex_to_sportscode(DEFAULT_COLOR2)
    codes = [
        (f"{team1} Offense",     c1),
        (f"{team1} Transition O", c1),
        (f"{team1} 6v5",         c1),
        (f"{team2} Offense",     c2),
        (f"{team2} Transition O", c2),
        (f"{team2} 6v5",         c2),
    ]
    rows = ["<ROWS>\n"]
    for code, (r, g, b) in codes:
        rows.append(
            f"<row>\n"
            f"<code>{_escape(code)}</code>\n"
            f"<R>{r}</R>\n"
            f"<G>{g}</G>\n"
            f"<B>{b}</B>\n"
            f"</row>\n"
        )
    rows.append("</ROWS>\n")
    return "".join(rows)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(video_path: str, team1: str, team2: str,
                 api_key: str, color1: str = None, color2: str = None,
                 progress_cb: Callable = None):
    """
    Main entry point. Returns (xml_str, stats_dict).
    """
    config = load_config()
    client = anthropic.Anthropic(api_key=api_key)

    # Setup few-shot examples (extract if needed)
    from extract_few_shot import run as extract_few_shot_run
    if not any(
        os.path.isdir(os.path.join(FEW_SHOT_DIR, s)) and
        any(f.endswith(".jpg") for f in os.listdir(os.path.join(FEW_SHOT_DIR, s)))
        for s in ["Offense", "Transition O", "6v5"]
        if os.path.isdir(os.path.join(FEW_SHOT_DIR, s))
    ):
        if progress_cb:
            progress_cb("few_shot_extraction", 0.0)
        extract_few_shot_run()

    few_shot = load_few_shot_images()

    # 1. Frame diff scan
    if progress_cb:
        progress_cb("diff_scan", 0.0)
    diffs, duration = compute_frame_diffs(video_path, progress_cb)

    # 2. Smart frame extraction
    if progress_cb:
        progress_cb("extract_frames", 0.0)
    frames = extract_frames(video_path, diffs, duration, config, progress_cb)

    # 3. Batched Claude classification
    if progress_cb:
        progress_cb("classify", 0.0)
    predictions, (calls1, in1, out1) = classify_frames(
        client, frames, team1, team2, config, few_shot,
        color1=color1, color2=color2, progress_cb=progress_cb
    )

    # 4. Transition resampling
    if progress_cb:
        progress_cb("resample", 0.0)
    predictions, (calls2, in2, out2) = resample_transitions(
        video_path, predictions, team1, team2, config, client, few_shot,
        color1=color1, color2=color2, progress_cb=progress_cb
    )

    # 5. Smooth + merge
    # Smoothing window: 1 frame for short clips, 2 for full games (avoids over-smoothing sparse samples)
    smooth_window = 1 if duration < 300 else 2
    # Min segment duration: ~3.5% of video length, floor of 5s, cap of 10s
    min_dur = max(5.0, min(10.0, duration * 0.035))
    predictions = smooth_predictions(predictions, window=smooth_window)
    segments = merge_segments(predictions, min_duration=min_dur)

    # 6. Write XML
    xml = write_xml(segments, team1, team2, color1=color1, color2=color2)

    total_calls = calls1 + calls2
    total_input = in1 + in2
    total_output = out1 + out2
    # Estimate cost: claude-sonnet-4-6 pricing (input $3/M, output $15/M)
    estimated_cost = (total_input / 1_000_000 * 3.0) + (total_output / 1_000_000 * 15.0)

    stats = {
        "frames_analyzed": len(predictions),
        "segments_found": len(segments),
        "total_api_calls": total_calls,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "estimated_cost_usd": round(estimated_cost, 4),
    }

    return xml, stats


# ---------------------------------------------------------------------------
# Demo pipeline (no API key — uses ground-truth labels from Cal.xml)
# ---------------------------------------------------------------------------

def run_demo_pipeline(team1: str, team2: str, color1: str = None, color2: str = None,
                      progress_cb: Callable = None):
    """
    Runs the full pipeline on a 2-minute demo clip using ground-truth
    labels from Cal.xml as the mock classifier. No API key needed.
    Returns (xml_str, stats_dict).
    """
    import time as _time
    from demo_setup import create_demo_clip, load_ground_truth, label_at_time

    config = load_config()

    # Step 0: ensure demo clip exists
    if progress_cb:
        progress_cb("diff_scan", 0.0)
    video_path = create_demo_clip()

    # Step 1: Frame diff scan
    diffs, duration = compute_frame_diffs(video_path, progress_cb)

    # Step 2: Smart frame extraction
    if progress_cb:
        progress_cb("extract_frames", 0.0)
    frames = extract_frames(video_path, diffs, duration, config, progress_cb)

    # Step 3: Mock classification using ground-truth
    if progress_cb:
        progress_cb("classify", 0.0)
    gt_instances = load_ground_truth()
    predictions = []
    for i, (ts, jpeg_bytes) in enumerate(frames):
        label = label_at_time(ts, gt_instances, team1, team2)
        predictions.append((ts, label, 1.0))
        if progress_cb and i % 4 == 0:
            progress_cb("classify", (i + 1) / max(len(frames), 1))

    # Step 4: Transition resampling (mock — dense resample uses ground truth too)
    if progress_cb:
        progress_cb("resample", 0.0)
    transition_window = config.get("transition_window", 30)
    dense_iv = config.get("motion_sample_interval", 5)

    transition_ts = []
    for i in range(1, len(predictions)):
        if predictions[i][1] != predictions[i-1][1]:
            transition_ts.append(predictions[i][0])

    existing_ts = {p[0] for p in predictions}
    extra = []
    for t in transition_ts:
        for dt in range(-transition_window, transition_window + 1, dense_iv):
            candidate = round(max(0.0, min(duration, t + dt)), 3)
            if candidate not in existing_ts:
                extra.append(candidate)
                existing_ts.add(candidate)

    for i, ts in enumerate(sorted(extra)):
        label = label_at_time(ts, gt_instances, team1, team2)
        predictions.append((ts, label, 1.0))
        if progress_cb and i % 5 == 0:
            progress_cb("resample", (i + 1) / max(len(extra), 1))

    predictions.sort(key=lambda x: x[0])

    # Step 5: Smooth + merge (demo clip is ~120s, use light smoothing + low min duration)
    predictions = smooth_predictions(predictions, window=1)
    segments = merge_segments(predictions, min_duration=5.0)

    # Step 6: Write XML
    xml = write_xml(segments, team1, team2, color1=color1, color2=color2)

    stats = {
        "frames_analyzed": len(predictions),
        "segments_found": len(segments),
        "total_api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost_usd": 0.0,
        "demo": True,
    }
    return xml, stats
