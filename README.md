# WP TactSit — Automated Water Polo Film Tagger

Upload a raw `.mp4` game film, get back a Sportscode-compatible `.xml` with your six key codes automatically tagged.

## What it does

Water polo coaches spend hours manually tagging possessions in Sportscode. This tool uses computer vision and Claude AI to do it automatically.

**The six codes it tags:**
- `[Team] Offense` — half-court set possession
- `[Team] Transition O` — fast break up the pool
- `[Team] 6v5` — man-up power play (always written alongside a concurrent `Offense` instance, matching Sportscode's overlap convention)

**The pipeline, in plain English:**
1. Scans the video for motion zones vs. stable zones (no API cost)
2. Extracts ~100 representative frames, sampling more densely during action
3. Sends frames to Claude in batches of 4 with few-shot example images
4. Does a second dense pass around every label-change boundary to pin down exact transition times
5. Smooths out noise, merges consecutive frames into instances
6. Writes a `.xml` you can open directly in Sportscode

**Cost:** roughly **$0.50–$1.50 per 90-minute game** in Anthropic API usage.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/WP-TactSit.git
cd WP-TactSit
```

### 2. Install dependencies

Requires Python 3.9+.

```bash
pip install flask anthropic opencv-python pyyaml python-dotenv
```

### 3. Add your Anthropic API key

Get a key at [console.anthropic.com](https://console.anthropic.com). Then either:

**Option A — paste it in the web UI** (simplest, no setup needed)

**Option B — create a `.env` file** so it pre-fills automatically:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Add your dataset (optional but recommended)

The tool auto-extracts few-shot example images from labeled games you already have. Place them in:
```
datasets/
  OpponentName/
    OpponentName.mp4
    OpponentName.xml   ← Sportscode XML, UTF-16 encoded
```

If you don't have any labeled games, the tool will still run — it just won't have few-shot examples.

### 5. Run

```bash
python app.py
```

Open `http://localhost:5050` in your browser.

---

## How to use

1. **Drop your `.mp4`** into the upload zone. The server extracts a frame from ~2 minutes in.

2. **Enter team names** — Team 1 (your team) and Team 2 (opponent).

3. **Sample cap colors** — click on a player's cap in the extracted frame for each team. This helps Claude count caps to correctly identify which team has the 6v5 advantage. Hit "Show a different frame" if both teams aren't visible.

4. **Enter your API key** (or it pre-fills from `.env`).

5. **Click Tag Film.** A progress bar walks through each pipeline stage. When done, a color-coded timeline appears and the XML downloads automatically.

---

## Demo mode

Don't have a game ready? Two demo options appear alongside the Tag Film button:

- **⚡ Demo + Claude** — runs the real AI pipeline on a built-in 2-minute clip (Cal vs Stanford). Requires an API key. Costs ~$0.05–0.10.
- **▶ Mock Demo** — same clip, labels pulled from the ground-truth XML instead of Claude. Free, completes in ~10 seconds. Good for testing the UI and XML output format.

---

## Configuration

Edit `config.yaml` to tune the pipeline:

```yaml
frame_diff_threshold: 30.0     # motion sensitivity (lower = more frames extracted)
confidence_threshold: 0.6      # Claude predictions below this are logged to review.log, not written to XML
stable_sample_interval: 15     # seconds between frames in calm zones
motion_sample_interval: 5      # seconds between frames in active zones
transition_window: 30          # seconds around each label boundary to resample densely
batch_size: 4                  # frames per Claude API call
```

Low-confidence frames are written to `review.log` for manual review.

---

## Output format

The `.xml` output is UTF-8 and Sportscode-compatible. Open it in Sportscode via **File → Open**. The six codes appear color-coded by team (using the cap colors you sampled). Each instance has exact start/end timestamps in seconds.

6v5 possessions appear as two overlapping instances — `[Team] Offense` and `[Team] 6v5` with identical timestamps — matching the convention used in manually-tagged Sportscode files.
