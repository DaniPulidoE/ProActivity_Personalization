# ProVoice - Robust Personalized Proactivity

## Description

ProVoice is a driving-simulator research platform for studying
**proactive in-vehicle assistants**. While a participant drives in CARLA simulator, the
system watches the driver (webcam: gaze, blinks, yawns, emotion, remote heart
rate; vehicle: speed, braking, junctions, headway) and predicts a **Level of
Autonomy (LoA, 0–4)** — how autonomously the assistant should act on a given
in-vehicle task, from *do nothing* to *act without asking*. Every 20 s the driver
is asked which LoAs they would have accepted, and those answers are the ground
truth.

This repository adds a **personalization layer** on top of the base LoA model
and evaluates it in two studies:

1. **Population study** — 12 drivers × 2 sessions, ~1,450 labels. Trains the
   base model (an xLSTM over a 20 s driver-state window with a rank-consistent
   ordinal head) and compares two ways of adapting it to an individual driver
   from a handful of their labels: **head fine-tuning with an L2-SP anchor** vs.
   **ANIL-style meta-learning (iMAML)**, in leave-one-driver-out cross-validation.
   The collected dataset is on Hugging Face:
   [`danipulidoe/proactivity_preference_dataset`](https://huggingface.co/datasets/danipulidoe/proactivity_preference_dataset)
   (schema: [`docs/study1_dataset_schema.md`](docs/study1_dataset_schema.md)).
2. **Live follow-up study** — the same drivers return and handle a simulated
   incoming phone call under three amounts K of personalization labels (K = 0, K=4,
   K=8), to measure how much data it takes before a driver is satisfied.

The runtime (CARLA + Drive UI + ProVoice) is a Windows-only study instrument.
The parts meant to be reused elsewhere are the offline pieces:
`src/ProVoice/models/` (`xlstm_model`, `head_adapt`, `laplace_head`,
`xlstm_maml`), `src/ProVoice/hr_filter.py`, `src/ProVoice/training_scripts/`,
and the dataset above.

Forked from: https://github.com/LouisSY/ProActivity.

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
cd ProActivity_Personalization
uv sync  # Install dependencies (required on first run)
```

### Step 2: Manually install CARLA python package 0.10.0
Install the package through `wheels/carla-0.10.0-cp312-cp312-win_amd64.whl` (only for Linux, installed via uv sync on Windows)

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
uv run python -m src.drive.drive_improved --control test
```

For full controls including weather, cameras, and telemetry:
```bash
uv run python -m src.drive.drive_improved --control full
```

> For detailed options, please refer to the [Control Modes](docs/README_DRIVE_CONTROL_MODES.md) section.



In a **separate terminal**, run (in the root of the project):

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
- `data/raw_data.jsonl` stores the raw multimodal context samples written by ProVoice.

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
uv run python scripts/build_loa_dataset.py

# train on the driver's real labels
uv run python -m ProVoice.models.train_XLSTM --in data/labeled_data.jsonl --out trained_models/state_xlstm.pt
uv run python -m ProVoice.train_fcd_loa     # reads data/processed_data/fcd_out.csv -> trained_models/fcd_levels.pkl
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

#### Starting the experimental pipeline

Use `start_experiment.py` to generate a session ID and launch every process with
shared parameters:

```bash
uv run python start_experiment.py --participantid 001 --environment city --secondary-task none \
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

#### The study rig: two machines

Both studies ran on **two machines**, and `start_experiment.py` is run on each
of them:

| Machine | Runs | Started with |
|---|---|---|
| **CARLA machine** | CARLA, NPC traffic, the Drive UI (LoA popups, phone-call event), and two small HTTP bridges: `scripts/vehicle_state_server.py` (:8080, vehicle state → ProVoice) and `scripts/provoice_status_server.py` (:8081, ProVoice lifecycle → Drive) | `--experiment-*-carla-remote` presets |
| **ProVoice machine** | ProVoice only (camera, perception, decision engine), reading vehicle state and the participant/session ids from the CARLA machine's bridge | `--experiment-*-provoice-remote` presets |

The link address is the constant `CARLA_MACHINE_IP` at the top of
`start_experiment.py` (or `PV_BRIDGE_URL`); nothing else is typed twice, the
participant id is entered once on the CARLA side. Network setup for the link is
in [`docs/remote_setup.md`](docs/remote_setup.md). 

A **single-machine** run is the development path: the same launcher starts
everything as child processes, with ProVoice reading vehicle state through the
file bridge (`--vehicle-bridge`) instead of HTTP.

#### Session recipes

The launcher has ~60 flags (`uv run python start_experiment.py --help`); the
`--experiment-*` presets are exact aliases for the flag combinations in the
right-hand column and accept extra flags (`--webcam`, `--res`, …) on the same
line.

| Phase | Rig (two machines) | Single machine |
|---|---|---|
| Teach the LoA popup (nothing logged) | CARLA: `--experiment-popup` | `--test-popup --fullscreen` |
| Familiarisation drive (no ProVoice) | CARLA: `--experiment-adaptation` | `--test-drive --no-popup --fullscreen` |
| Calibration (180 s baseline, then stop) | CARLA: `--experiment-calibration-carla-remote --participantid 001`<br>ProVoice: `--experiment-calibration-provoice-remote` | `--fixed --no-popup --calibration-only --fullscreen --participantid 001` |
| Population data collection (15 min) | CARLA: `--experiment-data-collection-carla-remote --participantid 001 --run 1`<br>ProVoice: `--experiment-data-collection-provoice-remote` | `--data-collection --random-function --fullscreen --participantid 001 --run 1` |
| Live follow-up study (one block) | CARLA: `--study-satisfaction-carla-remote --participantid 001 --condition <0\|1\|2> --block-idx <1\|2\|3>`<br>ProVoice: `--study-satisfaction-provoice-remote` | `--study-satisfaction --participantid 001 --condition <0\|1\|2> --block-idx <1\|2\|3> --fullscreen` |

Every cell is prefixed with `uv run python start_experiment.py`..

Flags worth knowing when composing your own command:

| Option | Purpose |
|---|---|
| `--participantid ID` | Participant id written into every log row |
| `--run 1\|2` | Which of the participant's two data-collection sessions this is (selects the planned traffic scenario) |
| `--functionname NAME` / `--random-function` | One in-vehicle function for the whole run, or draw one per popup from the study pool (mutually exclusive) |
| `--modeltype`, `--state-model`, `--w-fcd` | Decision strategy, state model (`xlstm`), and fusion weight — see [Decision Strategies](#decision-strategies) |
| `--fullscreen` / `--res WxH` | Drive window size (also the CARLA camera resolution) |
| `--no-popup` | Free driving: no LoA prompts, nothing appended to `user_loa_labels.csv` |
| `--keyboard-input` (default) / `--wheel-input` | How the LoA popup is answered |
| `--webcam` | Use camera index 1 (external USB) instead of the built-in camera |
| `--traffic-seed N` | Override the planned traffic scenario (normally derived from participant + run) |
| `--vehicle-bridge` | Run the CARLA vehicle-state file bridge as a supervised child process so ProVoice never holds a CARLA client (see "Vehicle-State Bridge") |
| `--vehicle-id-timeout S` | Seconds to wait for Drive to publish `vehicle_id.txt` (default 120) |

> **Note**: Please do activate the correct Python environment before running this script (if not using uv run).

#### Manual Launch (Alternative)

If you prefer to launch manually in two separate terminal windows, export the session ID first:

**macOS/Linux:**
```bash
export PV_SESSION_ID=$(uuidgen)
cd ProActivity_Personalization
# In first terminal:
uv run python -m src.drive.drive_improved --control test --session-id "$PV_SESSION_ID" --participantid 001 --environment city --secondary-task none --functionname "Adjust seat positioning" --modeltype combined --state-model xlstm --w-fcd 0.7

# In second terminal:
uv run provoice session_id=$PV_SESSION_ID participantid=001 environment=city secondary_task=none functionname="Adjust seat positioning" modeltype=combined state_model=xlstm w_fcd=0.7
```

**Windows (PowerShell):**
```powershell
$env:PV_SESSION_ID = [guid]::NewGuid().ToString()
cd ProActivity_Personalization
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
ProActivity_Personalization/
├── start_experiment.py        # Launcher script (recommended for starting both processes)
├── scripts/
│   ├── build_loa_dataset.py    # Logged sessions -> labelled training data
│   ├── map_wheel_buttons.py    # One-off: map steering wheel buttons for LoA input
|   ├──
|   ├── make_test_dataset.py # makes a mock dataset (to verify pipeline works)
|   ├── provoice_status_server.py # ProVoice -> CARLA bridge
|   ├── setup_cuda_torch.py # update environment so models are run on CUDA
|   ├── train_yolo26_series.py # train in-cabin object detection model
|   ├── upload_study1_hf.py # upload dataset from data collection study to huggingface
|   └── vehicle_state_server.py # CARLA -> ProVoice HTTP bridge
├── src/
│   ├── rPPG/ # handles the rPPG model
│   ├── drive/                  # Driving simulation module
│   │   ├── drive_improved.py   # Enhanced CARLA manual control
│   │   ├── ambience.py # manage background noise
│   │   ├── call_event.py # code to spawn calls in the UI
│   │   └── fixed_npc_traffic.py # spawns and controls NPC traffic in CARLA
│   │   ├── questionnaire.py # spawns the questionnaire for the satisfaction study
│   │   ├── study_session.py # manages a whole run of the live follow-up study
│   │
│   └── ProVoice/               # AI assistant module
│       ├── agents/ # currently not in use
│       ├── data/ # currently not in use
│       ├── models/ 
│       │    ├── fine_tune_XLSTM.py # fine-tune state model on driver-specific data (calls head_adapt.py)
│       │    ├── head_adapt.py # mechanism to adapt head to user's labels
│       │    ├── laplace_head.py # computes Laplace approximation of the head
│       │    ├── train_XLSTM.py # train state model
│       │    ├── xlstm_maml.py # train MAML version of state model
│       │    └── xlstm_model.py # define state model architecture
│       ├── main.py             # Entry point
│       ├── decision_engine.py   # AI decision making
│       ├── data_collector.py    # Data collection
│       ├── hr_filter.py # filter for the heart rate readings
│       ├── perception.py        # EAR/MAR (MediaPipe) + YOLO object-detection distraction
│       ├── train_distraction.py # Fine-tune YOLO26 on a custom distraction dataset
│       ├── train_fcd_loa.py     # Model training (FCD)
│       ├── study_bridge.py # thread that reads the data from the HTTP bridge connecting both machines (in the two-machine setup)
│       ├── training_analysis # analysis of the results of the hyperparameter sweeps
│       ├── training_scripts # code for the hyperparameter sweeps and the training of the models for each driver 
│       └── webui/               # Dashboard interface
│
├── data/                       # Data storage (created at runtime)
│   ├── decisions.csv          # System decision logs
│   ├── user_loa_labels.csv    # User LoA labels (every 20s)
│   └── raw_data.jsonl         # Raw event data
│
├── data_analysis/  # code to analyze results of the data collection and the live follow-up study 
│
├── docs/                       # Documentation
│   ├── README_macOS_carla_setup.md
│   └── README_original.md
│
├── trained_models/ # decision engine models
│   ├── user_study/ # models used for the live follow-up study
│       └── xlstm_p<xxx>_k<y> # model for participant xxx in condition y
│
├── wheels/ # stores carla 0.10.0 wheel 
│
└── README.md                  # This file
```

## Decision Strategies

ProVoice's decision engine (`src/ProVoice/decision_engine.py`) turns the current
context into a 5-class LoA distribution, decodes it to one LoA, and hands that to
the actuator. Which inputs it uses is chosen with `--modeltype`:

| `--modeltype` | Inputs | Model | Notes |
|---|---|---|---|
| `fcd` | the 12 **Functional Context Dimensions** of the active task (`src/ProVoice/fcd_config.py`; static per function, 1–5 scale) | XGBoost (`trained_models/fcd_levels.pkl`) | Task context only, independent of the driver |
| `state` | the driver-state stream | `--state-model xlstm` (default): xLSTM over the last 20 s resampled to 10 Hz, 33 features (`trained_models/state_xlstm.pt`); `--state-model classic`: MLP on per-frame features (`trained_models/state_levels.pkl`) | **What the live study serves** (`--modeltype state --state-model xlstm`), so the personalized head is the only thing that moves the output |
| `combined` (default) | both | `LoA = w_fcd · P_fcd + (1 − w_fcd) · P_state`, `--w-fcd` default 0.7 | Note FCD is static per task, so at 0.7 it dominates: the state model cannot move the served LoA by more than 0.3 |
| `collection` | — | none | Data collection only: ProVoice records `raw_data.jsonl` and makes no decisions |

There is no `--modeltype xlstm`: the xLSTM is a *state* model, so serving it alone
is `--modeltype state --state-model xlstm`. If a model file is missing the
strategy falls back to LoA 0 and marks the row `fallback=True` in `decisions.csv`.

How the 5-class distribution is decoded to one LoA is controlled by environment
variables: `PV_DECISION_METHOD` (`argmax` default, `expected`, or `quantile`),
`PV_QUANTILE_TAU` (0.65) and `PV_TEMP` (softmax temperature, 1.0).

### LoA → action

The mapping is fixed (`decision_engine._LOA_POLICY`) and identical for every
participant and condition — making it switchable would confound the studies.

| LoA | Action | Meaning for the phone-call event |
|---|---|---|
| 0 | `none` | the phone rings; the assistant does nothing |
| 1 | `suggest` | the assistant recommends answer / decline; the driver decides |
| 2 | `ask_approval` | the assistant proposes an action and waits for yes / no |
| 3 | `auto_with_veto` | the assistant announces the action and acts unless vetoed within a short window |
| 4 | `auto` | the assistant acts and informs the driver |

## State Model (xLSTM)

The State→LoA model is a **real xLSTM sequence classifier** built on the
official [`nx-ai/xlstm`](https://github.com/NX-AI/xlstm) package
(`xlstm==2.0.5`), trained via `src/ProVoice/models/train_XLSTM.py`. It consumes the
per-frame state-feature sequence of a segment and predicts the preferred
Level of Automation over 5 classes (LoA 0–4).

### Choosing a loss (`--loss`)

| `--loss` | Head | Ordinal? | Accepts several marked LoAs? |
|---|---|---|---|
| `ce` | softmax, 5 logits | no (nominal) | **yes** — the marked set becomes a uniform distribution |
| `corn` (default) | K−1 conditional logits | yes | **yes** |

The `corn` option is soft-CORN, an extension to `corn` allowing for multiple marked labels at the same time.

```bash
uv run python -m ProVoice.models.train_XLSTM --in data/labeled_data.jsonl \
    --out trained_models/state_xlstm.pt --loss corn
```

The choice is baked into the checkpoint and picked up automatically by
`fine_tune_XLSTM.py` and the decision engine. Note `--laplace` fine-tuning
requires a **SOFT-CORN** checkpoint and rejects the others (the `ce`version is not implemented in this codebase).

Training reports two extra metrics alongside the usual ones: **`set-acc`**
(fraction of predictions the driver marked acceptable) and a **`MAE`** measured
to the *nearest* marked level. Both reduce exactly to plain accuracy and MAE
when every window marks a single level, so numbers stay comparable on
single-label data — but checkpoint selection now optimises "nearest acceptable
level" once multi-marks are present.

It uses the CPU-compatible mLSTM `xLSTMBlockStack` path (pure PyTorch); the
triton-based `xlstm.xlstm_large` / `mlstm_kernels` path is **not** used
(triton is unavailable on Windows). xLSTM inference therefore runs on CPU.
If `trained_models/state_xlstm.pt` is absent, the decision engine falls back to FCD / LoA 0.

> **Note on the committed `trained_models/state_xlstm.pt`.** It is not a model
> trained on all 12 drivers: it is a copy of participant 001's *unadapted*
> live-study checkpoint (`user_study/xlstm_p001_k0.pt`, i.e. the LODO model
> trained on the other 11 drivers, with the FCD-augmented 76-wide head), placed
> there so the default serving path has something to load. Its `arch['study']`
> names participant `001`, so the decision engine's provenance check refuses to
> serve it to any other `--participantid` and the session then runs on the
> fallback. Retrain with the command above to get a genuine population model.

## Reproducing the offline results

Everything below runs without CARLA or a camera. Input is
`data/labeled_data.jsonl` (built by `scripts/build_loa_dataset.py` from the
logged sessions, or reconstructed from the Hugging Face dataset). All commands
are `uv run python -m …` from the repo root; a CUDA GPU is strongly recommended
(`scripts/setup_cuda_torch.py` once, then launch with the venv's Python directly
rather than `uv run` — see the script header).

Download `danipulidoe/proactivity_preference_dataset`  into data/study1_data/.

```
data/labeled_data.jsonl
        │
        ▼
 ① run_population_pipeline ──► results/pop_pipeline/corn_w10/selected_population.json
        │                        (hyperparameter sweep; also the CE ablation)
        ▼
 ② run_lodo_population ──────► trained_models/lodo/pop_heldout_<pid>.pt  ×12
        │                        results/lodo/lodo_population.csv  (the unpersonalized floor)
        ▼
 ③ sweep_l2sp_tau ───────────► results/l2sp_sweep/selected_tau.json  (the ONE L2-SP hyperparameter)
        │                        per-driver learning curves, MAE/QWK vs. K
        ▼
 Ⓐ sweep_anil_hparams ───────► results/anil_sweep/selected_anil.json
        ▼
 Ⓑ run_lodo_anil ────────────► trained_models/lodo_anil/anil_heldout_<pid>.pt  ×12
        ▼
 Ⓒ compare_arms_k_curve ─────► results/arm_comparison/  (primary result: L2-SP vs. ANIL, paired per driver)
        ▼
 Ⓓ phone_call_k_curve ───────► results/phone_call_k_curve/  (same curve, restricted to the study's function)
        ▼
 Ⓔ build_study_checkpoints ──► trained_models/user_study/xlstm_p<pid>_k<0|1|2>.pt  (served in the live study)
```

Stages ①–③ are the L2-SP arm, Ⓐ–Ⓒ the ANIL arm plus the comparison; every
stage is leave-one-driver-out (`training_scripts/folds.py`), so a held-out
driver's checkpoint never saw that driver's data — including the K = 0 model
served in the live study.

```bash
# ① population hyperparameters (four sweeps; resumable with the identical command)
uv run python -m ProVoice.training_scripts.run_population_pipeline \
    --in data/labeled_data.jsonl --outdir results/pop_pipeline

# ② 12 held-out population models at the selected configuration
uv run python -m ProVoice.training_scripts.run_lodo_population \
    --in data/labeled_data.jsonl \
    --selected results/pop_pipeline/corn_w10/selected_population.json

# ③ L2-SP arm: select tau, produce the per-driver learning curves
uv run python -m ProVoice.training_scripts.sweep_l2sp_tau \
    --in data/labeled_data.jsonl --ckpt-dir trained_models/lodo

# Ⓐ ANIL arm: meta-learning hyperparameters (shares tau and the stage-② checkpoints)
uv run python -m ProVoice.training_scripts.sweep_anil_hparams \
    --in data/labeled_data.jsonl \
    --selected-tau results/l2sp_sweep/selected_tau.json \
    --selected-population results/pop_pipeline/corn_w10/selected_population.json \
    --outdir results/anil_sweep

# Ⓑ 12 held-out meta-initializations
uv run python -m ProVoice.training_scripts.run_lodo_anil \
    --in data/labeled_data.jsonl \
    --selected results/anil_sweep/selected_anil.json \
    --pop-ckpt-dir trained_models/lodo --ckpt-dir trained_models/lodo_anil

# Ⓒ the arm comparison (MAE / QWK vs. K, paired per driver)
uv run python -m ProVoice.training_scripts.compare_arms_k_curve \
    --l2sp-ckpt-dir trained_models/lodo --anil-ckpt-dir trained_models/lodo_anil \
    --outdir results/arm_comparison

# Ⓓ same curve for the live study's function only ("Respond to a phone call")
uv run python -m ProVoice.training_scripts.phone_call_k_curve \
    --l2sp-ckpt-dir trained_models/lodo --anil-ckpt-dir trained_models/lodo_anil \
    --embed-fcd --steps 6000 --outdir results/phone_call_k_curve

# Ⓔ mint the checkpoints the live study serves
uv run python -m ProVoice.training_scripts.build_study_checkpoints \
    --in-data data/labeled_data.jsonl --ckpt-dir trained_models/lodo \
    --outdir trained_models/user_study \
    --verify-against results/phone_call_k_curve/phone_call_k_curve.csv
```

Single-driver tools behind these stages, usable on their own:
`ProVoice.models.fine_tune_XLSTM` (adapt one head; `--laplace` adds the
posterior over it), `scripts/sweep_train_frac.py` (one driver's quality-vs-K
curve), `ProVoice.models.xlstm_maml` (meta-train one fold).

### What is in `results/`

The **summary tables** every thesis table is built from are tracked: each
sweep's `*_results.csv` / `l2sp_tau_sweep.csv` and its `selected_*.json`, the
K-curve and paired-difference CSVs under `arm_comparison_v2/` and
`phone_call_k_curve*/`, `lodo/lodo_population.csv` (the K=0 floor per driver)
and `user_study_checkpoints.csv` (provenance of every served head). Per-run
epoch curves, plots and draft runs are not tracked.

The **per-epoch training curves** that two of the analysis notebooks read are
shipped as one archive instead, `results/results_runs.zip` (4 MB, 660 CSVs):

| archive path | read by |
|---|---|
| `anil_sweep_v2/runs/` | `src/ProVoice/training_analysis/anil_sweep_analysis.ipynb` |
| `pop_adapt_full/{corn_w10,corn_w20,ce_w20}/runs/`, `pop_adapt_full_ext/corn_w20/runs/` | `src/ProVoice/training_analysis/pop_model_training_analysis.ipynb` |

Unpack it in place before running either notebook — the paths inside the
archive are relative to `results/`, so this restores exactly the layout the
sweeps wrote:

```bash
# from the repo root
uv run python -c "import zipfile; zipfile.ZipFile('results/results_runs.zip').extractall('results')"
```

The other notebooks (`l2sp_sweep_analysis`, `arm_comparison_analysis`,
`phone_call_k_curve_analysis`) need only the tracked CSVs plus the study-1 data
from Hugging Face.

## Driver Perception (EAR / MAR / Distraction)

`src/ProVoice/data_collector.py` no longer depends on the upstream
`yolov5-deepsort-driverdistracted-driving-behavior-detection` package
(which pinned the project to Python 3.10 via its bundled `dlib` wheels
and a custom YOLOv5 codebase). It now uses the in-tree module
`src/ProVoice/perception.py`, which stacks two modern libraries:

| Signal | Implementation |
|---|---|
| Eye / mouth aspect ratio (`eye_ar`, `mar`) | MediaPipe Tasks FaceLandmarker |
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
uv run python -m src.drive.drive_improved --help
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
uv run python scripts/map_wheel_buttons.py
```

Paste its output into the constants at the lines 629-636 of `src/drive/drive_improved.py`.
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
- **[Personalization Notes](docs/PERSONALIZATION_NOTES.md)** - Approach ladder for per-driver adaptation (historical)
- **[Original Documentation](docs/README_original.md)** - Archived original guide

### Additional Resources
- [CARLA Documentation](https://carla-ue5.readthedocs.io)
- [CARLA Python API Reference](https://carla-ue5.readthedocs.io/en/latest/python_api/)


## License

See [LICENSE](LICENSE) for details.



