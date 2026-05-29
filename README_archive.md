# ProActivity / ProVoice

## Prerequisites

Before you begin, ensure you have:

### Required Software
- **Python >= 3.10, < 3.11**
- **CARLA Simulator 0.9.16** - [Installation Guide](https://carla.readthedocs.io/en/latest/start_quickstart/)
  - ⚠️ **Important**: CARLA 0.9.16 is required; other versions may not be compatible
- **uv Package Manager** - [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)

### System Requirements
- **OS**: macOS (Apple Silicon recommended), Windows 11, or Linux
- **GPU**: Dedicated GPU recommended for better performance

### Platform-Specific Setup
- **macOS (Apple Silicon)**: See [Mac Setup Guide](docs/README_macOS_carla_setup.md)
- **Windows**: Standard installation should work
- **Linux**: Standard installation should work

## Installation

### Step 1: Clone and Setup Environment
```bash
cd proactivity-main
uv sync  # Install dependencies (required on first run)
```

### Step 2: Start CARLA Simulator
```bash
# Windows
CarlaUE4.exe -quality-level=Low

# macOS/Linux
./CarlaUE4.sh -quality-level=Low
```

> **Note**: Use `-quality-level=Low` for better performance if you have limited resources
> 
> **Note**: Use `-RenderOffScreen` for better performance if you have limited resources



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

#### Option 1: Using UV (Recommended)
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

Use `start_both.py` to automatically generate a session ID and launch both processes in separate terminal windows with shared parameters:

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
│   │   └── wheel.py            # Wheel controller support
│   │
│   └── ProVoice/               # AI assistant module
│       ├── main.py             # Entry point
│       ├── decision_engine.py   # AI decision making
│       ├── data_collector.py    # Data collection
│       ├── train_fcd_loa.py     # Model training (FCD)
│       ├── train_XLSTM.py       # Model training (XLSTM)
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
- [CARLA Documentation](https://carla.readthedocs.io/)
- [CARLA Python API Reference](https://carla.readthedocs.io/en/latest/python_api/)


## License

See [LICENSE](LICENSE) for details.



