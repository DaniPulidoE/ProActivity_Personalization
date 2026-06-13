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

For best alignment across the two processes, use the same `session_id` in both commands.

#### Quick Start (Recommended)

Use `start_experiment.py` to automatically generate a session ID and launch both processes in separate terminal windows with shared parameters:

```bash
python start_experiment.py --participantid 001 --environment city --secondary-task none \
  --functionname "Adjust seat positioning" --modeltype combined --state-model xlstm --w-fcd 0.7
```

This script will:
1. Generate a unique `session_id` and save it to `.session_id`
2. Open two new terminal windows (PowerShell on Windows, Terminal/gnome-terminal on macOS/Linux)
3. Launch `drive_improved.py` in the first window
4. Launch `provoice` in the second window

Both processes will automatically use the same session ID for data alignment.

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
├── start_both.py              # Launcher script (recommended for starting both processes)
├── src/
│   ├── drive/                  # Driving simulation module
│   │   ├── drive_improved.py   # Enhanced CARLA manual control
│   │   ├── drive.py            # Basic driving interface
│   │
│   └── ProVoice/               # AI assistant module
│       ├── main.py             # Entry point
│       ├── decision_engine.py   # AI decision making
│       ├── data_collector.py    # Data collection
│       ├── perception.py        # EAR/MAR (MediaPipe) + YOLO26 distraction detection
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
Level of Automation as a **single-label 5-class** output (LoA 0–4).

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
| Distraction labels (`safe`, `phone`, `drink`, `distracted`) | Ultralytics YOLO26 (classification) |

**The distraction model is not stored in this repo.** It is downloaded on
first use from the Hugging Face Hub
([`maco018/in-car-distraction-yolo26`](https://huggingface.co/maco018/in-car-distraction-yolo26))
and cached locally by `huggingface_hub` (so it only downloads once). The
repo hosts the full **YOLO26 classification series** (`n`/`s`/`m`/`l`/`x`)
fine-tuned on the
[State Farm Distracted Driver Detection](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection)
dataset — real in-cabin, driver-facing frames. The default variant is
`l` (best accuracy, 94.6% top-1 on held-out drivers).

Weights resolution precedence (first match wins):
1. `weights=` arg passed to `DistractionDetector(...)`
2. `PROVOICE_YOLO_WEIGHTS` env var — absolute path to a local `.pt` (offline use)
3. Hugging Face download of `PROVOICE_YOLO_VARIANT` (`n`/`s`/`m`/`l`/`x`, default `l`)
   from `PROVOICE_YOLO_REPO` (default `maco018/in-car-distraction-yolo26`)

`face` is set whenever MediaPipe detects a face (independent of YOLO).
The classifier runs at **imgsz 224** (its training resolution) — this is
auto-detected from the checkpoint.

### Retraining (e.g. when a newer YOLO release lands)

The training pipeline is kept in-repo so the models can be regenerated:

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
- `--sync` - Enable synchronous mode
- `--autopilot` - Enable autopilot

### Camera Options

You can adjust the camera source by modifying the `camera_source` variable in `src/ProVoice/main.py`.

##### Use the default camera (front-facing camera)
```python
python src/ProVoice/main.py camera_source=local
```
##### Use UDP Streaming:
```python
python src/ProVoice/main.py camera_source=udp
```
You can specify the UDP streaming port (default port: 8554)
```python
python src/ProVoice/main.py camera_source=udp camera_url=udp://127.0.0.1:8554
```

To enable UDP streaming, run `ffmpeg` in a separate terminal on macOS or Linux:
```bash
ffmpeg -f avfoundation -framerate 30 -i "0" -vcodec mpeg4 -f mpegts udp://127.0.0.1:8554
```



## Documentation

### Setup Guides
- **[macOS Apple Silicon Setup](docs/README_macOS_carla_setup.md)** - Detailed macOS installation
- **[Docker Setup](docs/README_macOS_docker_setup.md)** - Docker-based deployment
- **[Original Documentation](docs/README_original.md)** - Archived original guide

### Additional Resources
- [CARLA Documentation](https://carla-ue5.readthedocs.io)
- [CARLA Python API Reference](https://carla-ue5.readthedocs.io/en/latest/python_api/)


## License

See [LICENSE](LICENSE) for details.



