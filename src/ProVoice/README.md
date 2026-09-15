# ProVoice — the decision engine of ProActivity Personalization

`src/ProVoice` is the half of the study rig that watches the driver and decides
how autonomously the in-vehicle assistant should act. It samples the driver's
face and the CARLA vehicle state, predicts a **Level of Autonomy (LoA, 0–4)**
for the active in-vehicle task, logs everything, and — in the live follow-up
study — serves a **per-driver personalized head** to the Drive UI over a bridge.
The other half (`src/drive/`) owns the simulator, the labelling pop-ups and the
phone-call event; `start_experiment.py` at the repository root launches both.

This file is the module-level map. Installation, the session recipes and the
offline reproduction pipeline are in the repository [README](../../README.md);
design decisions and data contracts are in [CLAUDE.md](../../CLAUDE.md).

## What it does, end to end

```text
webcam ──► DataCollector (~20 Hz collection loop, worker threads for capture,
           face box, YOLO, vehicle state and the decision engine)
            ├─ MediaPipe Tasks FaceLandmarker → EAR, MAR, gaze_score,
            │    blink_rate, yawn_rate, perclos, drowsiness_alert
            ├─ EmotiEffLib (AffectNet) → emotion, emotion_prob
            ├─ MMRPhys rPPG (SCAMPS LEF 72×72) → heart_rate, hr_delta
            │    (respiratory_rate is still logged but no longer a model input)
            ├─ YOLO distraction → lab (phone / drink / face)
            └─ vehicle-state bridge → speed, brake, steer, throttle,
                 is_junction, lead_distance_m, …   (raw CARLA units)
                   │
                   ▼
           Decision engine (own thread, --decision-hz 4)
            ├─ fcd:      XGBoost on the task's 12 FCD dimensions
            ├─ state:    xLSTM over the last 10 s resampled to 10 Hz (33 dims)
            └─ combined: w_fcd · P_fcd + (1 − w_fcd) · P_state
                   │
                   ▼
           LoA → ProVoiceActuator → data/decisions.csv, data/raw_data.jsonl,
                 dashboard (port 8001), optional study bridge to the CARLA machine
```

A **180 s calibration** at session start establishes the per-driver baselines
(gaze, EAR, MAR, heart rate, blink rate, PERCLOS) that every physiological
feature is normalised against. The baseline is persisted per participant and
reused by later sessions.

## Running it

ProVoice never holds a CARLA client of its own; vehicle state arrives through a bridge
process. The launcher wires all of that up, so this is the normal entry point:

```bash
uv run python start_experiment.py --participantid 001 --environment city \
    --secondary-task none --functionname "Adjust seat positioning" \
    --modeltype combined --state-model xlstm --w-fcd 0.7
```

ProVoice on its own (`python -m ProVoice.main`, or the `provoice` script) takes
the same flags; `key=value` spellings are rewritten to `--key value`:

| Flag | Meaning |
| --- | --- |
| `--participantid` | Files the calibration baseline and tags every row |
| `--functionname` | The active in-vehicle task; must match a name in `fcd_config.py` exactly (a paraphrase resolves to the neutral all-3s FCD vector and warns `[fcd][warn]`) |
| `--modeltype` | `fcd` / `state` / `combined` (default) / `collection`. **There is no `xlstm` modeltype**: the xLSTM is a state model, so serving it alone is `--modeltype state --state-model xlstm` |
| `--state-model` | `xlstm` (default) or `classic` (the MLP in `trained_models/state_levels.pkl`) |
| `--w-fcd` | Fusion weight under `combined` (default 0.7) |
| `--decision-hz` | Decision-thread rate (default 4). Keep it fixed across participants |
| `--xlstm-model` | Checkpoint to serve (default `trained_models/state_xlstm.pt`) |
| `--window-seconds` | Time span of the xLSTM input; unset inherits the checkpoint's contract |
| `--calibration-only` | Run the 180 s calibration, store the baseline, exit |
| `--data-collection` | Record `raw_data.jsonl` only: no calibration, no model, no decisions (`--data-collection-timeout` caps it) |
| `--vehicle-state-file` / `--vehicle-state-url` | Local file bridge (default on one machine) or the HTTP bridge on a remote CARLA machine |
| `--webcam` | Use camera index 1 (the external driver-facing camera) |
| `--study-bridge`, `--status-url`, `--study-checkpoint-id` | Live study only: publish every decision to the CARLA machine and record which head served it |

Decoding of the 5-class distribution to a single LoA is controlled by
`PV_DECISION_METHOD` (`argmax` default, `expected`, `quantile`),
`PV_QUANTILE_TAU` and `PV_TEMP`. A CORN head is decoded with its rank rule
(the PMF's median), the same function the trainers and sweeps use.

## Module layout

| Path | Role |
| --- | --- |
| `main.py` | Entry point: argument parsing, session/participant identity (adopted from the bridge under `--remote`), model loading, dashboard server |
| `data_collector.py` | Multimodal sampling loop, calibration, and the capture / face-box / YOLO / vehicle-state / decision worker threads |
| `decision_engine.py` | `XGBoostLoAStrategy`, `StateLevelsLoAStrategy`, `StateXLSTMLoAStrategy`, `CombinedFusionStrategy`; the fixed LoA→action policy; the loader that refuses a study checkpoint held out for a different participant |
| `provoice_actuator.py` | LoA → action and the console/log trace of decisions |
| `logger.py` | Dual-stream logging: `raw_data.jsonl` (every frame) + `decisions.csv` (fixed schema, one row per decision) |
| `fcd_config.py` | The 12-dimensional FCD vector for each of the 14 known functions |
| `perception.py` | EAR/MAR geometry from face landmarks, the YOLO `DistractionDetector`, dashboard overlays |
| `hr_filter.py` | The ONE definition of the rPPG cleaning algorithm (octave/harmonic handling, baseline statistic), shared with `data_preprocessing/heart_rate_preprocessing.py` |
| `study_bridge.py` | Fire-and-forget publisher of each decision to the CARLA machine during a live-study block |
| `webui/` | FastAPI + Dash + Socket.IO dashboard at `http://127.0.0.1:8001` |
| `models/xlstm_model.py` | Single source of truth for the xLSTM architecture and the 33-feature encoding (`FEATURE_NAMES`, sentinels, aliases, 10 Hz resampling grid), soft-CORN loss and decoding |
| `models/train_XLSTM.py` | Population model training (`--loss corn` default, `--window-seconds 10`) |
| `models/head_adapt.py` | The ONE per-driver head adaptation optimizer (full-batch, K-independent budget, L2-SP anchor specified as prior precision τ) |
| `models/fine_tune_XLSTM.py` | Per-driver head fine-tuning on a frozen backbone; produces the head that gets served |
| `models/laplace_head.py` | Laplace posterior over the adapted CORN head (offline uncertainty analysis) |
| `models/xlstm_maml.py` | ANIL / iMAML meta-training — the comparison arm of the offline study |
| `training_scripts/` | The offline pipeline: population hyperparameter sweeps, leave-one-driver-out models, τ selection, the L2-SP vs. ANIL comparison, and `build_study_checkpoints.py`, which mints the checkpoints the live study serves |
| `training_analysis/` | Notebooks reading the sweep and comparison results |
| `train_fcd_loa.py` | XGBoost FCD → LoA (`trained_models/fcd_levels.pkl`) |
| `train_distraction.py` | Fine-tune YOLO26 on the in-cabin distraction dataset |
| `eval.py` | Offline evaluation report for a labelled dataset |
| `agents/`, `data/`, `demo.py`, `read.py`, `test.py` | Inherited from the original ProVoice codebase; not used by the current pipeline |

`src/rPPG/rppg_infer_simple.py` (outside this package) wraps the MMRPhys
estimator the collector feeds.

## Data it writes

| File | Content |
| --- | --- |
| `data/raw_data.jsonl` | One dict per collection tick: every perception and vehicle field, plus the LoA/FCD *in force* at that frame |
| `data/decisions.csv` | One row per decision, fixed schema, `timestamp` = the frame the decision was computed from (so it joins onto `raw_data.jsonl`) |
| `data/calibration_data/calibration_<pid>.json` | The stored per-driver baseline (+ a per-tick log under `calibration_logs/`) |

Ground-truth LoA labels are written by the Drive UI (`data/user_loa_labels.csv`,
one prompt per 20 s window, two under `--random-function`); the system's own
prediction is never a training label. `scripts/build_loa_dataset.py` aligns the
two into `data/labeled_data.jsonl`.

## Models

```text
trained_models/
├── fcd_levels.pkl                 XGBoost FCD → LoA
├── state_levels.pkl               classic MLP state → LoA
├── state_xlstm.pt                 population xLSTM (arch dict carries head_type,
│                                  context_length, window_seconds, resample_hz)
├── lodo/pop_heldout_<pid>.pt      leave-one-driver-out population models (offline)
└── user_study/xlstm_p<pid>_k<c>.pt  the live study's served heads, condition c ∈ {0,1,2}
```

Retrain the population model with

```bash
uv run python -m ProVoice.models.train_XLSTM --in data/labeled_data.jsonl \
    --out trained_models/state_xlstm.pt --loss corn
```

and adapt a head to one driver with `python -m ProVoice.models.fine_tune_XLSTM`.
The label is a **set** of acceptable LoAs per window (multi-hot), trained with
soft-CORN; metrics are set-aware (set-MAE, set-accuracy, QWK) and reduce to the
single-label forms when one level is marked. xLSTM inference runs on CPU (the
pure-PyTorch `xLSTMBlockStack` path; no triton). If `state_xlstm.pt` is
missing, the state strategy falls back and the row is marked `fallback=True`.

The YOLO distraction weights, the EmotiEffLib emotion model and the MMRPhys
checkpoint are downloaded on first use and cached; the MediaPipe landmarker
task file lands in `src/ProVoice/trained_models/`.

## Dashboard

`main.py` serves the dashboard at `http://127.0.0.1:8001` while it runs. It
shows the annotated camera frame, the current action, fatigue (blink rate, yawn
rate, PERCLOS, drowsiness), gaze, emotion, distraction labels, EAR/MAR and the
heart-rate / respiration trends, refreshed over a WebSocket from the collector's
latest frame.
