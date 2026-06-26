# WP TactSit — Automated Water Polo Film Tagger

Upload a raw `.mp4` game film, get back a Sportscode-compatible `.xml` with six key codes automatically tagged.

**The six codes:**
- `[Team] Offense` — half-court set possession
- `[Team] Transition O` — fast break
- `[Team] 6v5` — man-up power play (written alongside a concurrent `Offense` instance)

---

## Architecture

The pipeline is being migrated from Claude vision to a local CV model. Current state:

### Stage 1 (in development) — YOLO player detection
Fine-tune YOLOv8 on water polo footage to detect per frame:
- `player_dark` — dark-cap player
- `player_light` — light-cap player
- `goalkeeper`

### Stage 2 (planned) — Feature extraction
Per-frame signals: dark/light cap counts, centroids, goalie position, zoom level, optical flow.

### Stage 3 (planned) — Temporal classifier
LSTM or sliding-window model trained on frame features → outputs Offense / Transition O / 6v5 / neutral.

### Current fallback — Claude vision pipeline
The Flask app (`app.py`) runs the original Claude-based classifier while the YOLO pipeline is being trained. It samples frames, sends them to Claude Sonnet in batches with few-shot examples, and produces a Sportscode XML. Cost: ~$0.50–1.50 per 90-minute game.

---

## Setup

Requires Python 3.9+.

```bash
pip install flask anthropic opencv-python pyyaml python-dotenv ultralytics tqdm
```

---

## Running the web app (Claude vision pipeline)

```bash
python app.py
```

Open `http://localhost:5050`. Upload an `.mp4`, enter team names, sample cap colors, paste your Anthropic API key, and click **Tag Film**.

Add an `.env` file to pre-fill your API key:
```
ANTHROPIC_API_KEY=sk-ant-...
```

Demo mode is available without a game file — uses a pre-cut 2-minute Cal clip with ground-truth labels.

---

## Training the YOLO model

### Step 1 — Bootstrap labels

Extract frames from labeled game footage and run pretrained YOLO person detection:

```bash
# Test on Cal, first 10 min — check yolo_bootstrap/previews/ to verify detection quality
python bootstrap_labels.py --test

# Run on all 6 games once satisfied
python bootstrap_labels.py
```

Output lands in `yolo_bootstrap/images/` and `yolo_bootstrap/labels/` (all boxes stubbed as class 0).

### Step 2 — Roboflow review

1. Upload `yolo_bootstrap/images/` + `yolo_bootstrap/labels/` to [Roboflow](https://roboflow.com)
2. Relabel each detected player box as `player_dark`, `player_light`, or `goalkeeper`
3. Export as **YOLOv8 format** and place `dataset.yaml` at the project root

### Step 3 — Fine-tune

```bash
python yolov8_train.py
```

Model checkpoints saved to `runs/train/waterpolo/`.

---

## Dataset

Six labeled games in `datasets/`:

| Game | File |
|------|------|
| Cal  | Cal.mp4 + Cal.xml |
| Ford | Ford.mp4 + Ford.xml |
| LB   | LB.mp4 + LB.xml |
| SJSU | SJSU.mp4 + SJSU.xml |
| UCLA | UCLA.mp4 + UCLA.xml |
| USC  | USC.mp4 + USC.xml |

Each XML is a Sportscode timeline (UTF-16) with `[Team] Offense`, `[Team] Transition O`, and `[Team] 6v5` instances.

---

## Camera constraints

- Camera zooms into the attacking end — full pool is not always in frame
- Pool orientation varies by broadcast; teams switch sides at half-time
- The classifier uses cap count ratio, optical flow, zoom level, and goalie position as orientation-invariant signals rather than pool geometry

---

## Output format

Sportscode-compatible UTF-8 XML. Open via **File → Open** in Sportscode. 6v5 possessions appear as two overlapping instances — `[Team] Offense` and `[Team] 6v5` — matching the convention in hand-tagged files.
