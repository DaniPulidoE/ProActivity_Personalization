# ProActivity / ProVoice

## Description

Forked from https://github.com/LouisSY/ProActivity, will be adding a layer of personalization on top of the predictions.

## Prerequisites

Before you begin, ensure you have:

### Required Software
- **Python 3.12** (`>=3.12,<3.13`)
- **CARLA Simulator 0.10.0** - [Installation Guide](https://carla.readthedocs.io/en/latest/start_quickstart/)
  - The 0.10 Python wheel bundled in `wheels/` is for CPython 3.12 on Windows. For Linux, install the matching wheel from `<CARLA_ROOT>/PythonAPI/carla/dist/`.
- **uv Package Manager** - [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)

### System Requirements
- **OS**: Windows 11 (the bundled CARLA 0.10 Python wheel is Windows-only; `tool.uv.environments` in `pyproject.toml` is scoped to `win32`). To use Linux/macOS, drop in the matching `carla-0.10.0-*-linux_x86_64.whl` from `<CARLA_ROOT>/PythonAPI/carla/dist/` and widen `tool.uv.environments`.
- **GPU**: Dedicated GPU recommended for better performance

### Platform-Specific Setup
- **Windows**: Standard installation works out of the box
- **macOS (Apple Silicon)**: See [Mac Setup Guide](docs/README_macOS_carla_setup.md) (untested with CARLA 0.10)
- **Linux**: Standard installation works once the Linux carla wheel is wired up (see above)

## Installation

### Step 1: Clone and Setup Environment
```bash
cd proactivity-main
uv sync  # Install dependencies (required on first run)
```

### Step 2: Manually install CARLA python package 0.10.0
Install the package through `wheels/carla-0.10.0-cp312-cp312-win_amd64.whl`

### Step 3: Start CARLA Simulator
```bash
# Windows
CarlaUnreal.exe -quality-level=Low

# macOS/Linux
./CarlaUnreal.sh -quality-level=Low
```

> **Note**:
> Use `-quality-level=Low` for better performance if you have limited resources. 
> Use `-RenderOffScreen` for better performance if you have limited resources



## Quick Start

### Basic Manual Driving

Start the driving simulator in test mode (clean interface, basic controls only):
```bash
python -m src.drive.drive_improved --control test
```

For full controls including weather, cameras, and telemetry:
```bash
python -m src.drive.drive_improved --control full
```

> For detailed options, please refer to the [Control Modes](docs/README_DRIVE_CONTROL_MODES.md) section.



In a **separate terminal**, run:

#### Option 1: Using UV
```bash
uv run provoice \
  participantid=001 \
  environment=city \
  secondary_task=none \
  functionname="Adjust seat positioning" \
  modeltype=combined \
  state_model=xlstm \
  w_fcd=0.7
```

#### Option 2: Using Python Directly
```bash
python src/ProVoice/main.py \
  participantid=001 \
  environment=city \
  secondary_task=none \
  functionname="Adjust seat positioning" \
  modeltype=combined \
  state_model=xlstm \
  w_fcd=0.7
```

### Logging and Training Data

- `data/decisions.csv` is the **system decision log** written by ProVoice.
- `data/user_loa_labels.csv` is the **user label log** written by the driving UI every 20 seconds.
- `data/raw_data.jsonl` stores the raw multimodal context samples.

A driver may mark **more than one acceptable LoA** per window. Multiple marks are
written to `user_selected_loa` as a `;`-joined list; a single mark stays a bare
integer, so labels recorded before multi-select still parse unchanged:

| `user_selected_loa` | Meaning |
|---|---|
| `2` | only LoA 2 acceptable |
| `2;3` | LoA 2 **and** 3 both acceptable |

For best alignment across the two processes, use the same `session_id` in both commands.

#### Building the training dataset from logged sessions

`scripts/build_loa_dataset.py` turns logged sessions into trainable files by
joining each frame in `raw_data.jsonl` to the driver's chosen LoA in
`user_loa_labels.csv` (matched on `session_id` + the frame timestamp falling
inside a 20 s label window). The driver's `user_selected_loa` becomes the
ground-truth label — **not** the system's own predicted `LoA`.

```bash
# raw_data.jsonl + user_loa_labels.csv  ->  data/labeled_data.jsonl
#                                           data/processed_data/fcd_out.csv
python scripts/build_loa_dataset.py

# train on the driver's real labels
python -m ProVoice.train_XLSTM --in data/labeled_data.jsonl --out trained_models/state_xlstm.pt --context-length 256
python -m ProVoice.train_fcd_loa     # reads data/processed_data/fcd_out.csv -> trained_models/fcd_levels.pkl
```

`Level_1..5` is a **multi-hot** encoding of the levels the driver marked (LoA
0–4 → `Level_1..5`). One marked level gives the familiar one-hot, so existing
datasets are unaffected. If your labels contain multi-marks, pick a loss that
can represent them — see [State Model](#state-model-xlstm) below.

For the join to work, drive and ProVoice must share a `session_id` (use
`start_experiment.py` or `PV_SESSION_ID`), and the camera must see the driver's
face so the per-frame state features are populated. To dry-run the whole chain on
synthetic data, run `scripts/make_test_dataset.py` (writes clearly-labelled fake
data under `data/testdata/`) and point `build_loa_dataset.py` at it.

#### Quick Start (Recommended)

Use `start_experiment.py` to generate a session ID and launch every process with
shared parameters:

```bash
python start_experiment.py --participantid 001 --environment city --secondary-task none \
  --functionname "Adjust seat positioning" --modeltype combined --state-model xlstm --w-fcd 0.7
```

This script will:
1. Generate a unique `session_id` and save it to `.session_id`
2. Start NPC traffic, then `drive_improved.py`, as child processes (their output
   is interleaved into this terminal — no extra windows are opened)
3. **Wait for Drive to spawn the ego vehicle and publish `vehicle_id.txt`**
4. Launch ProVoice, passing that vehicle id explicitly
5. Monitor all children and shut the rest down if any one exits

Step 3 replaces a fixed sleep that merely assumed the vehicle existed by then.
Drive writes `vehicle_id.txt` only after the world tick, so its appearance is a
real "Drive is initialised" signal. Any stale `vehicle_id.txt` is deleted before
Drive starts, so the wait cannot be satisfied by the previous run's id, and if
Drive dies during startup the launcher reports it immediately instead of waiting
out the timeout.

Launcher options:

| Option | Default | Purpose |
|---|---|---|
| `--fullscreen` | off | Run the drive window fullscreen at the desktop resolution |
| `--res WIDTHxHEIGHT` | `1280x720` | Drive window size; ignored with `--fullscreen` |
| `--no-popup` | off | Suppress the LoA selection popups for the whole session (see below) |
| `--fixed` | off | Always spawn the ego at the same map spawn point (calibration runs) |
| `--test-drive` | off | Launch NPC traffic and Drive only, without ProVoice |
| `--calibration-only` | off | Run ProVoice's 180 s calibration, store the baseline, then stop everything |
| `--data-collection` | off | ProVoice records raw data only: no decision engine, no live calibration |
| `--vehicle-id-timeout` | `120` | Seconds to wait for Drive to publish the vehicle id |

> **Note**: Please do activate the correct Python environment before running this script.

#### Manual Launch (Alternative)

If you prefer to launch manually in two separate terminal windows, export the session ID first:

**macOS/Linux:**
```bash
export PV_SESSION_ID=$(uuidgen)
cd proactivity-main
# In first terminal:
python -m src.drive.drive_improved --control test --session-id "$PV_SESSION_ID" --participantid 001 --environment city --secondary-task none --functionname "Adjust seat positioning" --modeltype combined --state-model xlstm --w-fcd 0.7

# In second terminal:
uv run provoice session_id=$PV_SESSION_ID participantid=001 environment=city secondary_task=none functionname="Adjust seat positioning" modeltype=combined state_model=xlstm w_fcd=0.7
```

**Windows (PowerShell):**
```powershell
$env:PV_SESSION_ID = [guid]::NewGuid().ToString()
cd proactivity-main
# In first PowerShell window:
python -m src.drive.drive_improved --control test --session-id $env:PV_SESSION_ID --participantid 001 --environment city --secondary-task none --functionname "Adjust seat positioning" --modeltype combined --state-model xlstm --w-fcd 0.7

# In second PowerShell window:
uv run provoice session_id=$env:PV_SESSION_ID participantid=001 environment=city secondary_task=none functionname="Adjust seat positioning" modeltype=combined state_model=xlstm w_fcd=0.7
```

### Access Dashboard

Open your browser and navigate to:
```
http://127.0.0.1:8001
```

The web UI dashboard displays real-time metrics and analysis.

## Project Structure

```
proactivity-main/
├── start_experiment.py        # Launcher script (recommended for starting both processes)
├── scripts/
│   ├── build_loa_dataset.py    # Logged sessions -> labelled training data
│   └── map_wheel_buttons.py    # One-off: map steering wheel buttons for LoA input
├── src/
│   ├── drive/                  # Driving simulation module
│   │   ├── drive_improved.py   # Enhanced CARLA manual control
│   │   ├── drive.py            # Basic driving interface
│   │
│   └── ProVoice/               # AI assistant module
│       ├── main.py             # Entry point
│       ├── decision_engine.py   # AI decision making
│       ├── data_collector.py    # Data collection
│       ├── perception.py        # EAR/MAR (MediaPipe) + YOLO object-detection distraction
│       ├── train_distraction.py # Fine-tune YOLO26 on a custom distraction dataset
│       ├── train_fcd_loa.py     # Model training (FCD)
│       ├── train_XLSTM.py       # State→LoA training (official nx-ai/xlstm, xlstm==2.0.5)
│       └── webui/               # Dashboard interface
│
├── data/                       # Data storage
│   ├── decisions.csv          # System decision logs
│   ├── user_loa_labels.csv    # User LoA labels (every 20s)
│   └── raw_data.jsonl         # Raw event data
│
├── docs/                       # Documentation
│   ├── README_macOS_carla_setup.md
│   └── README_original.md
│
└── README.md                  # This file
```

## State Model (xLSTM)

The State→LoA model is a **real xLSTM sequence classifier** built on the
official [`nx-ai/xlstm`](https://github.com/NX-AI/xlstm) package
(`xlstm==2.0.5`), trained via `src/ProVoice/train_XLSTM.py`. It consumes the
per-frame state-feature sequence of a segment and predicts the preferred
Level of Automation over 5 classes (LoA 0–4).

### Choosing a loss (`--loss`)

| `--loss` | Head | Ordinal? | Accepts several marked LoAs? |
|---|---|---|---|
| `ce` (default) | softmax, 5 logits | no (nominal) | **yes** — the marked set becomes a uniform distribution |
| `corn` | K−1 conditional logits | yes | **no** |
| `coral` | shared weight + K−1 biases | yes | **yes** — the target is a cumulative vector |

**`coral` is the option to use for multi-marked labels while keeping the ordinal
structure.** Its loss is a sum of binary cross-entropies over the cumulative
outputs, so the target only has to lie in [0, 1]: the driver's marked set is
encoded as `q_k = P(y > k)` of a uniform distribution over that set. Marking
`{2,3}` gives `[1, 1, 0.5, 0]` and `{0,4}` gives `[0.5, 0.5, 0.5, 0.5]`, so
"two adjacent levels are fine" and "either extreme but nothing between" stay
distinguishable — unlike independent per-class losses, where both are just two
bits set. The mapping is invertible, so nothing about which levels were marked
is lost, and a single mark reproduces the standard 0/1 extended label exactly.

**`corn` cannot represent a marked set** — it partitions samples into hard
conditional subsets. It does not error on a soft target either, it silently
rounds it, so the trainer **refuses** multi-marked data on this path rather than
training on a corrupted label.

```bash
python -m ProVoice.train_XLSTM --in data/labeled_data.jsonl \
    --out trained_models/state_xlstm.pt --loss coral
```

The choice is baked into the checkpoint and picked up automatically by
`fine_tune_XLSTM.py` and the decision engine. Note `--laplace` fine-tuning
requires a **CORN** checkpoint and rejects the others.

Training reports two extra metrics alongside the usual ones: **`set-acc`**
(fraction of predictions the driver marked acceptable) and a **`MAE`** measured
to the *nearest* marked level. Both reduce exactly to plain accuracy and MAE
when every window marks a single level, so numbers stay comparable on
single-label data — but checkpoint selection now optimises "nearest acceptable
level" once multi-marks are present.

It uses the CPU-compatible mLSTM `xLSTMBlockStack` path (pure PyTorch); the
triton-based `xlstm.xlstm_large` / `mlstm_kernels` path is **not** used
(triton is unavailable on Windows). xLSTM inference therefore runs on CPU.
If `trained_models/state_xlstm.pt` is absent, the decision engine falls back
to FCD / LoA 0.

## Driver Perception (EAR / MAR / Distraction)

`src/ProVoice/data_collector.py` no longer depends on the upstream
`yolov5-deepsort-driverdistracted-driving-behavior-detection` package
(which pinned the project to Python 3.10 via its bundled `dlib` wheels
and a custom YOLOv5 codebase). It now uses the in-tree module
`src/ProVoice/perception.py`, which stacks two modern libraries:

| Signal | Implementation |
|---|---|
| Eye / mouth aspect ratio (`eye_ar`, `mar`) | MediaPipe FaceMesh |
| Distraction objects (`phone`, `drink`) | Ultralytics YOLO **object detection** (default) |
| Looking away (`gaze_distracted`) | MediaPipe gaze score |

Distraction runs in one of two modes, selected by the `PROVOICE_DISTRACTION_MODE`
environment variable:

- **`detect` (default)** — a COCO object detector (auto-downloaded by Ultralytics:
  `yolo26n.pt`, falling back to `yolo11n.pt`) localises `cell phone` → `phone`
  and `bottle`/`cup` → `drink` **as objects**. This works at any distance and can
  report several objects in the same frame. "Looking away" comes separately from
  the MediaPipe gaze score (`gaze_distracted`) — so the old single-label failure
  mode (everything collapsing to `distracted`, phone only seen near the face) is
  gone.
- **`classify`** — the legacy single-label model fine-tuned on the
  [State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection)
  dataset, downloaded from the Hugging Face Hub
  ([`maco018/in-car-distraction-yolo26`](https://huggingface.co/maco018/in-car-distraction-yolo26))
  and cached locally. One label per frame (`safe`/`phone`/`drink`/`distracted`);
  it is biased toward `distracted` on out-of-domain cameras, so it is opt-in.

Weights resolution precedence (first match wins):
1. `weights=` arg passed to `DistractionDetector(...)`
2. `PROVOICE_YOLO_WEIGHTS` env var — absolute path to a local `.pt` (offline use)
3. mode default — in `detect`: `PROVOICE_DETECT_WEIGHTS` (default `yolo26n.pt`);
   in `classify`: the Hugging Face download of `PROVOICE_YOLO_VARIANT`
   (`n`/`s`/`m`/`l`/`x`, default `l`) from `PROVOICE_YOLO_REPO`
   (default `maco018/in-car-distraction-yolo26`)

`face` is set whenever MediaPipe detects a face (independent of the detector).
Detection runs at imgsz 640; the classifier at imgsz 224 (auto-detected from the
checkpoint).

### Retraining the classify-mode model

This pipeline is only needed for `PROVOICE_DISTRACTION_MODE=classify` (the
default `detect` mode uses a stock COCO detector and needs no training). It is
kept in-repo so the classifier can be regenerated:

```bash
# 1. Download the State Farm dataset from Kaggle into
#    datasets/state-farm-distracted-driver-detection/, then build the
#    subject-aware YOLO classification split:
uv run python scripts/build_statefarm_dataset.py

# 2a. Fine-tune a single variant:
uv run --no-sync python -m ProVoice.train_distraction \
    --task classify --data datasets/distraction_sf \
    --weights yolo26l-cls.pt --epochs 50 --imgsz 224 --cos-lr --device 0

# 2b. ...or train the whole n/s/m/l/x series and package each for upload:
uv run --no-sync python scripts/train_yolo26_series.py --variants n,s,m,l,x

# 3. Upload the packaged exports/ folder to Hugging Face:
huggingface-cli upload maco018/in-car-distraction-yolo26 \
    exports/provoice-distraction-yolo26 . --repo-type model
```

> On an NVIDIA GPU, run `python scripts/setup_cuda_torch.py` once first to
> overlay the CUDA build of PyTorch (see the script header for why).

## Advanced Options

### Drive Script Options

```bash
python -m src.drive.drive_improved --help
```

Common options:
- `--control test|full` - Control mode (test: basic only, full: all controls)
- `--host` - CARLA server host (default: 127.0.0.1)
- `--port` - CARLA server port (default: 2000)
- `--res WIDTHxHEIGHT` - Window resolution (default: 1280x720)
- `--fullscreen` - Run fullscreen at the desktop resolution (overrides `--res`)
- `--no-wheel` - Ignore an attached steering wheel and force keyboard control
- `--fixed` - Spawn the ego at a fixed map spawn point instead of a random one, so
  every run starts from an identical position. Intended for calibration; leave it off
  for normal test drives, which keep the usual random spawn.
- `--no-popup` - Skip the LoA selection popups for the whole session. The scene is
  never frozen and nothing is appended to `data/user_loa_labels.csv`; use it for free
  driving, familiarisation runs and debugging. Popups are on by default, so omitting
  the flag keeps the normal 20 s prompt cadence.
- `--sync` - Enable synchronous mode
- `--autopilot` - Enable autopilot

The window size also drives the CARLA camera sensor resolution, so a larger
window costs frame rate — the driver-state pipeline is what pays. If it drops
noticeably, `--res 1920x1080` is a reasonable middle ground.

### Steering Wheel

A steering wheel is used automatically when one is attached; the keyboard is the
fallback. The wheel also answers the LoA prompt, so a participant never reaches
for the keyboard mid-drive:

- **Right / left paddle** — move the cursor up / down the LoA list
- **Front button** — tick or untick the level under the cursor
- **Front button on the `CONFIRM` row** — submit
- **Other front button** — close the simulation (same as the window's X)

Button indices are device-specific and must be mapped once per rig:

```bash
python scripts/map_wheel_buttons.py
```

Paste its output into the constants at the top of `src/drive/drive_improved.py`.
Until then the wheel cannot answer the prompt, and the popup says so on screen.

See **[Steering Wheel Setup](docs/README_STEERING_WHEEL.md)** for axis layouts,
compatibility vs native mode, and troubleshooting.

### Camera Options

The camera source is a **command-line argument** to `src/ProVoice/main.py`, not a
variable to edit. Accepted values:

| `camera_source` | Resolves to |
|---|---|
| `local` | local device index `0` |
| a digit, e.g. `1` | that local device index |
| `udp` | the stream at `camera_url` |
| anything else (default `front`) | local device index `0` |

```bash
# local webcam (default)
python src/ProVoice/main.py camera_source=local

# a second local camera
python src/ProVoice/main.py camera_source=1

# UDP stream (default port 8554)
python src/ProVoice/main.py camera_source=udp camera_url=udp://127.0.0.1:8554
```

> **`start_experiment.py` does not forward these.** The launcher passes no camera
> arguments, so ProVoice always falls back to local device index `0`. To use a
> streamed camera you must start `src/ProVoice/main.py` yourself (see
> [Manual Launch](#manual-launch-alternative)).

To feed a UDP stream, run `ffmpeg` on the machine holding the camera. Note
`udp://127.0.0.1:8554` only works when sender and receiver are the same machine;
across machines, the receiver must listen on all interfaces (`udp://@:8554`)
and the sender targets the receiver's LAN address.

```bash
# Windows (DirectShow) — list devices with: ffmpeg -list_devices true -f dshow -i dummy
ffmpeg -f dshow -i video="HD Pro Webcam C920" -vcodec mpeg4 -f mpegts udp://127.0.0.1:8554

# macOS
ffmpeg -f avfoundation -framerate 30 -i "0" -vcodec mpeg4 -f mpegts udp://127.0.0.1:8554

# Linux
ffmpeg -f v4l2 -framerate 30 -i /dev/video0 -vcodec mpeg4 -f mpegts udp://127.0.0.1:8554
```

Streaming adds encode/decode latency and jitter to a pipeline that samples driver
state continuously, so prefer a directly-attached camera for data collection.



## Documentation

### Setup Guides
- **[Steering Wheel Setup](docs/README_STEERING_WHEEL.md)** - Wheel detection, button mapping, LoA input from the wheel
- **[Drive Control Modes](docs/README_DRIVE_CONTROL_MODES.md)** - Keyboard and wheel bindings per `--control` mode
- **[macOS Apple Silicon Setup](docs/README_macOS_carla_setup.md)** - Detailed macOS installation
- **[Docker Setup](docs/README_macOS_docker_setup.md)** - Docker-based deployment
- **[Personalization Notes](docs/PERSONALIZATION_NOTES.md)** - Approach ladder for per-driver adaptation
- **[Original Documentation](docs/README_original.md)** - Archived original guide

### Additional Resources
- [CARLA Documentation](https://carla-ue5.readthedocs.io)
- [CARLA Python API Reference](https://carla-ue5.readthedocs.io/en/latest/python_api/)


## License

See [LICENSE](LICENSE) for details.



