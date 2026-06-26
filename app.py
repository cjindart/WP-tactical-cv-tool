"""
Flask web app for water polo film tagging.
"""
import os
import base64
import threading
import uuid
import logging
import random

import cv2
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, send_file

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024 * 1024  # 10 GB

jobs = {}
jobs_lock = threading.Lock()
sessions = {}       # session_id -> {video_path, used_timestamps, is_demo}
sessions_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def update_job(job_id, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


def extract_frame_b64(video_path: str, ts: float, width=800) -> str:
    cap = cv2.VideoCapture(video_path)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    ts = min(ts, max(0, dur - 5))
    cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return ""
    h, w = frame.shape[:2]
    new_h = int(h * width / w)
    frame = cv2.resize(frame, (width, new_h))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def pick_sample_timestamp(video_path: str, used: list) -> float:
    cap = cv2.VideoCapture(video_path)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    candidates = [120, 180, 240, 300, 360, dur * 0.15, dur * 0.25, dur * 0.35, dur * 0.5, dur * 0.65]
    candidates = [round(c, 0) for c in candidates if 5 <= c < dur - 5]
    untried = [c for c in candidates if c not in used]
    return untried[0] if untried else random.uniform(5, dur * 0.5)


def make_progress_cb(job_id):
    STAGE_WEIGHTS = {
        "few_shot_extraction": (0, 3),
        "diff_scan":           (3, 20),
        "extract_frames":      (20, 35),
        "classify":            (35, 70),
        "resample":            (70, 85),
        "finalize":            (85, 100),
    }
    def cb(stage, fraction):
        lo, hi = STAGE_WEIGHTS.get(stage, (0, 100))
        label_map = {
            "few_shot_extraction": "Extracting few-shot examples",
            "diff_scan":           "Scanning frame differences",
            "extract_frames":      "Extracting frames",
            "classify":            "Classifying frames with Claude",
            "resample":            "Refining transition boundaries",
            "finalize":            "Writing XML",
        }
        update_job(job_id, progress=round(lo + (hi - lo) * fraction, 1),
                   stage=label_map.get(stage, stage))
    return cb


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def run_pipeline_job(job_id, video_path, team1, team2, color1, color2, api_key, is_demo=False):
    from pipeline import run_pipeline
    try:
        update_job(job_id, status="running", progress=0, stage="Starting")
        xml, stats = run_pipeline(
            video_path, team1, team2, api_key,
            color1=color1, color2=color2, progress_cb=make_progress_cb(job_id)
        )
        if is_demo:
            stats["demo"] = True
        out_path = os.path.join(OUTPUT_DIR, f"{job_id}.xml")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(xml)
        update_job(job_id, status="done", progress=100, stage="Complete",
                   output_path=out_path, stats=stats)
    except Exception as e:
        logging.exception(f"Job {job_id} failed")
        update_job(job_id, status="error", error=str(e))
    finally:
        if not is_demo:   # never delete the shared demo clip
            try:
                os.remove(video_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prefill-key")
def prefill_key():
    return jsonify({"key": os.environ.get("ANTHROPIC_API_KEY", "")})


@app.route("/upload-video", methods=["POST"])
def upload_video():
    video = request.files.get("video")
    if not video or not video.filename.lower().endswith(".mp4"):
        return jsonify({"error": "Please upload an .mp4 file"}), 400

    session_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{session_id}.mp4")
    video.save(video_path)

    ts = 120.0
    frame_b64 = extract_frame_b64(video_path, ts)
    with sessions_lock:
        sessions[session_id] = {"video_path": video_path, "used_timestamps": [ts], "is_demo": False}

    return jsonify({"session_id": session_id, "frame": frame_b64, "timestamp": ts})


@app.route("/demo-start", methods=["POST"])
def demo_start():
    """Create a session backed by the pre-cut demo clip and return a preview frame."""
    from demo_setup import create_demo_clip
    clip_path = create_demo_clip()
    ts = 60.0   # ~1 min into the 2-min clip — good chance of seeing both teams
    frame_b64 = extract_frame_b64(clip_path, ts)
    session_id = str(uuid.uuid4())
    with sessions_lock:
        sessions[session_id] = {"video_path": clip_path, "used_timestamps": [ts], "is_demo": True}
    return jsonify({"session_id": session_id, "frame": frame_b64, "timestamp": ts})


@app.route("/resample-frame/<session_id>", methods=["POST"])
def resample_frame(session_id):
    with sessions_lock:
        sess = sessions.get(session_id)
    if not sess:
        return jsonify({"error": "Session not found"}), 404
    ts = pick_sample_timestamp(sess["video_path"], sess["used_timestamps"])
    sess["used_timestamps"].append(ts)
    return jsonify({"frame": extract_frame_b64(sess["video_path"], ts), "timestamp": ts})


@app.route("/process", methods=["POST"])
def process():
    session_id = request.form.get("session_id", "").strip()
    team1   = request.form.get("team1", "").strip()
    team2   = request.form.get("team2", "").strip()
    color1  = request.form.get("color1", "#cc0000").strip()
    color2  = request.form.get("color2", "#0044cc").strip()
    api_key = request.form.get("api_key", "").strip()

    if not team1 or not team2:
        return jsonify({"error": "Both team names are required"}), 400
    if not api_key:
        return jsonify({"error": "Anthropic API key is required"}), 400

    with sessions_lock:
        sess = sessions.pop(session_id, None)
    if not sess:
        return jsonify({"error": "Session expired — please re-upload the video"}), 400

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0, "stage": "Queued",
                        "team1": team1, "team2": team2}

    threading.Thread(
        target=run_pipeline_job,
        args=(job_id, sess["video_path"], team1, team2, color1, color2, api_key, sess["is_demo"]),
        daemon=True,
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/demo-ground-truth")
def demo_ground_truth():
    """Return the ground-truth XML for the demo window, filtered to our 6 codes."""
    from demo_setup import load_ground_truth
    from pipeline import write_xml
    VALID_SUFFIXES = ("Offense", "Transition O", "6v5")
    instances = load_ground_truth()  # list of (start, end, code) tuples, offset to t=0
    segments = [inst for inst in instances if any(inst[2].endswith(s) for s in VALID_SUFFIXES)]
    xml = write_xml(segments, "Stanford", "California")
    return xml, 200, {"Content-Type": "application/xml; charset=utf-8"}


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Job not complete"}), 404
    fname = f"{job.get('team1','team1')}_vs_{job.get('team2','team2')}_tags.xml"
    return send_file(job["output_path"], as_attachment=True, download_name=fname,
                     mimetype="application/xml")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, port=port, host="0.0.0.0")
