# ProActivity / ProVoice — Senior Engineering Review

_Date: 2026-06-13 — full pass over all source, config, packaging and docs._

This is a CARLA driving-simulator experiment platform with an AI co-pilot
("ProVoice") that fuses driver perception (face/eye/mouth, emotion, rPPG heart
rate, distraction) with task-feature (FCD) and driver-state models to decide a
Level of Automation (LoA), shown on a live web dashboard.

The architecture is reasonable. The **runtime quality is uneven**: the
perception/training utilities (`perception.py`, `train_distraction.py`, the
`scripts/`) are clean and well-documented, while the core runtime
(`decision_engine.py`, `data_collector.py`, `logger.py`, `main.py`, the
`data/` ETL) is riddled with a single systemic bug that disables almost all
error handling, plus several genuine correctness and performance defects.

Findings are ordered by severity. Each has a concrete location and fix.

---

## 🔴 Critical — correctness (fix before any data collection you trust)

### C1. `except NotImplementedError` everywhere — 55 occurrences, all dead handlers
Across 12 files the code wraps real operations in `try: … except NotImplementedError`.
`joblib.load`, `torch.load`, `model.predict_proba`, `float(...)`, file I/O,
`cv2.VideoCapture`, MediaPipe init, etc. **never raise `NotImplementedError`** —
they raise `FileNotFoundError`, `ValueError`, `RuntimeError`, `OSError`, `TypeError`.
So every one of these handlers is dead code, and the intended fallbacks never run.

Consequences:
- [main.py](src/ProVoice/main.py:173) — a corrupt/missing model raises, is *not*
  caught, and the `LoAZeroFallback` is never installed; the process crashes
  instead of degrading.
- [decision_engine.py](src/ProVoice/decision_engine.py:95) — if
  `predict_proba` throws (e.g. feature-count mismatch), it is not caught, the
  exception propagates out of …
- [data_collector.py](src/ProVoice/data_collector.py:525) — …the `_run_loop`
  `except NotImplementedError`, which also doesn't catch it, so **the data-collector
  thread dies silently** while the web server keeps serving stale data. The user
  sees a frozen dashboard with no error.
- [`_as01`](src/ProVoice/decision_engine.py:132), [extract.py `to_float`](src/ProVoice/data/extract.py:64),
  [merge.py `coerce_hr`](src/ProVoice/data/merge.py:56) — a non-numeric / `None`
  value crashes instead of returning the default.

**Fix:** global replace `except NotImplementedError` → `except Exception` (or,
better, the specific exceptions). This single change restores the entire
fallback design. Affected: `main.py`, `decision_engine.py`, `data_collector.py`,
`logger.py`, `eval.py`, `train_fcd_loa.py`, `train_XLSTM.py`, and all of `data/*.py`.

### C2. xLSTM is run with **untrained, randomly-initialised** recurrent weights
[train_XLSTM.py](src/ProVoice/train_XLSTM.py:169) trains a custom `XLSTM` whose
sub-modules are `fwd.*`, `bwd.*`, `proj.*`, `head.*` and saves
`model.state_dict()`. But the inference class
[StateXLSTMLoAStrategy](src/ProVoice/decision_engine.py:181) rebuilds a **different**
architecture — a stock `nn.LSTM` named `self.lstm` — and loads weights with:

```python
self.lstm.load_state_dict({k.replace("lstm.", ""): v for k,v in sd.items()
                           if k.startswith("lstm.")}, strict=False)
```

The checkpoint has **no keys starting with `lstm.`** (they start with `fwd.`/`bwd.`),
so this loads *nothing* and `strict=False` hides it. `proj`/`head` happen to match
and load, but the recurrent core stays random. **Every xLSTM prediction is noise.**
Because `state_model=xlstm` is the default in `start_experiment.py` and the README,
this is the path most people will actually run.

**Fix:** import and instantiate the real `XLSTM` class from `train_XLSTM` in the
strategy and `load_state_dict(..., strict=True)`, or save/load a TorchScript/`torch.save`
of the whole module. Add a `strict=True` load and assert no missing keys.

### C3. xLSTM trained as multi-label (sigmoid/BCE) but used as single-label (softmax)
Training optimises `BCEWithLogitsLoss` over 5 independent `Level_*` binaries
([train_XLSTM.py:253](src/ProVoice/train_XLSTM.py:253), labels one-hot→multi-hot),
but inference does `torch.softmax(logits)` + argmax
([decision_engine.py:249](src/ProVoice/decision_engine.py:249)). Even with C2 fixed,
the output semantics don't match the training objective.

### C4. Feature-encoding order differs between training and inference
- Inference: `_STATE_CAT = ['emotion','lab','environment','secondary_task']`
  ([decision_engine.py:99](src/ProVoice/decision_engine.py:99))
- Training:  `CAT = ['environment','secondary_task','lab','emotion']`
  ([train_XLSTM.py:14](src/ProVoice/train_XLSTM.py:14))

The categorical features land in different vector positions at train vs serve
time. (Moot for xLSTM until C2/C3 are fixed, but it will silently corrupt the
classic `StateLevelsLoAStrategy` too.) Define the feature schema **once** in a
shared module and import it in both places.

### C5. `wheel.py` never builds the control object in the wheel branch
In [wheel.py](src/drive/wheel.py:352) the `if wheel is not None:` branch computes
`steer/throttle/brake` but the `ctrl = carla.VehicleControl()` and
`vehicle.apply_control` wiring only exists inside the `else` (keyboard fallback)
branch ([lines 417-421](src/drive/wheel.py:417)). With a real wheel connected,
`ctrl` is undefined at [line 424](src/drive/wheel.py:424) → `NameError` on the
first non-interrupted frame (or it re-applies a stale full-brake `ctrl` from an
interruption frame). **Wheel/pedal driving does not work.** Move the `ctrl`
construction below the if/else so both paths populate it.

### C6. Logger corrupts the decisions CSV when columns change
[logger.py](src/ProVoice/logger.py:33) writes the header once from the first row's
keys, but later rows can introduce new keys (`fallback_reason`, `sub`, `hr_delta`,
…). When that happens it extends `self._processed_fieldnames` and writes wider
rows, but the header already on disk is narrower → **column misalignment** in
`data/decisions.csv`, which then feeds `eval.py` and the drive-side label join.
Use a fixed, exhaustive field list (you already have `DEFAULT_DECISION_COLUMNS` in
[drive_improved.py:156](src/drive/drive_improved.py:156)) or `extrasaction='ignore'`
with a stable schema, and rewrite the header when it grows.

---

## 🟠 High — performance (the 20 Hz loop cannot keep up as written)

The whole driver-state pipeline runs **serially in one thread** targeting 20 Hz
([data_collector._run_loop](src/ProVoice/data_collector.py:472)). Per frame it does:

1. MediaPipe FaceMesh for gaze ([get_gaze_score](src/ProVoice/data_collector.py:226))
2. **A second** MediaPipe FaceMesh for EAR/MAR inside `perception.frametest`
   ([perception.py:131](src/ProVoice/perception.py:131)) — confirmed two
   `FaceMesh.process()` calls + two BGR→RGB conversions per frame.
3. A Haar cascade face detect ([data_collector.py:318](src/ProVoice/data_collector.py:318))
   — a **third** face detector.
4. YOLO26-**l** classification `.predict` ([perception.py:331](src/ProVoice/perception.py:331))
5. The Keras mini-XCEPTION emotion CNN ([data_collector.py:183](src/ProVoice/data_collector.py:183))
6. rPPG frame ingest.

With `torch` pinned to the **CPU** wheel by default ([pyproject.toml:55](pyproject.toml:55)),
YOLO26-l + a Keras CNN at 20 Hz is not achievable; the loop will silently run far
slower than 20 Hz, skewing every rate/PERCLOS computation that assumes the sample
interval. **Recommendations:**
- Share **one** FaceMesh instance (compute gaze + EAR/MAR from the same landmarks);
  drop the Haar cascade and reuse FaceMesh's face box for emotion/rPPG ROI.
- Run YOLO at a lower cadence (e.g. every Nth frame) and/or default to the `n`/`s`
  variant for live use; the `l` model is for offline accuracy.
- Decouple capture / inference / decision into separate threads (or process the
  newest frame and drop stale ones) so a slow model can't stall capture.

### H1. Debug `print` spam in the hot loop
[data_collector.py:329-330](src/ProVoice/data_collector.py:329) prints
`!!!!!!!!!!hr/rr` every frame; [perception.py:371-372](src/ProVoice/perception.py:371)
prints labels/confs per detection box; [data_collector.py:450](src/ProVoice/data_collector.py:450)
and [538](src/ProVoice/data_collector.py:538) print cached speed / polled state each
tick. At ~20 Hz this is heavy console I/O. Replace with the `logging` module at
DEBUG and remove the `!!!!` lines.

### H2. Dashboard re-encodes and re-emits at 50 Hz
[webui/app.py:139](src/ProVoice/webui/app.py:139) loops every 20 ms and calls
`get_latest_frame()` → JPEG-encodes + base64 the frame **every** iteration, even
though new frames arrive ≤20 Hz. That re-encodes the same frame 2-3× and pushes it
to every socket client at 50 Hz. Emit only when the frame actually changed
(track a frame id), and drop the rate to ~15-20 Hz. Also wrap the emitter body in
try/except — one exception currently kills the emitter task permanently.

### H3. `decisions.csv` fully re-read every 20 s
[_load_latest_system_decision_snapshot](src/drive/drive_improved.py:254) scans the
entire growing CSV each time the LoA popup is answered (O(n) per label, O(n²) per
session). Keep a tail handle or have ProVoice expose the last decision over the
existing HTTP/socket channel.

### H4. Per-sample file open/close
[logger.py](src/ProVoice/logger.py:13) opens+closes both the JSONL and CSV on every
sample (~40 opens/s). Keep the handles open for the session and flush periodically.

---

## 🟡 Medium — robustness & design

- **Fabricated physiological/vehicle data silently fed to the model.** When rPPG
  yields nothing, `heart_rate` is set to `random.randint(60,100)`
  ([data_collector.py:435](src/ProVoice/data_collector.py:435)); with no CARLA link,
  speed is `random.randint(0,120)` ([:452](src/ProVoice/data_collector.py:452)).
  Random inputs to a decision model are dangerous for an experiment — emit `None`/NaN
  and let the model/UI mark "unavailable" instead.
- **The "personalization" layer is a no-op.** `adjust_fcd_by_state`
  ([fcd_config.py:65](src/ProVoice/fcd_config.py:65)) ignores its `_state` arg and
  only clamps. The README's headline ("adding a layer of personalization on top of
  the predictions") is not implemented anywhere I can find.
- **rPPG sampling-rate mismatch.** `OnlineRPPG(frame_rate=10)`
  ([data_collector.py:127](src/ProVoice/data_collector.py:127)) but frames are pushed
  at the loop rate (up to 20 Hz, realistically variable). HR/RR are derived from a
  wrong assumed sampling rate → biased estimates.
- **`OnlineRPPG` overwrites its own method with a Thread.**
  [`self.inference_thread = threading.Thread(target=self.inference_thread)`](src/rPPG/rppg_infer_simple.py:37)
  rebinds the method name to the thread object. It works by luck (RHS evaluated
  first) but is fragile; rename the thread attribute.
- **`__del__` → `stop()`** ([data_collector.py:169](src/ProVoice/data_collector.py:169))
  runs thread joins and hardware release during GC/interpreter shutdown — unreliable
  and can throw. Use explicit lifecycle / context manager only.
- **`client.set_timeout(2000.0)`** ([drive_improved.py:1594](src/drive/drive_improved.py:1594))
  — 2000 **seconds**. A hung CARLA call won't surface for 33 minutes. Likely meant
  20.0; pick something sane (and document if the large value is intentional for ngrok).
- **Shared mutable FCD dict.** `get_fcd_for_function`
  ([fcd_config.py:62](src/ProVoice/fcd_config.py:62)) returns the *same* dict stored
  in `BASE_FCD_CONFIG`; a caller that mutates it corrupts the global table. Return a copy.
- **`start_experiment.py` has no health monitoring / `finally` cleanup.** If a child
  crashes the launcher loops forever; if `main` throws (not KeyboardInterrupt)
  children leak. Poll `p.poll()` and move `stop_all()` into `finally`.
- **Global CARLA settings tug-of-war.** `fixed_npc_traffic.py`, `drive.py`,
  `wheel.py` and `drive_improved.py` each call `world.apply_settings(...)` (sync
  mode on/off, `fixed_delta_seconds`). Run concurrently they fight over global
  world state; e.g. NPC forces async while a `--sync` drive forces sync. Decide one
  owner of world settings.
- **No validation of CLI/env inputs.** `w_fcd`, `port`, `window`, etc. are cast with
  bare `float()/int()` in [main.py](src/ProVoice/main.py:147); a typo crashes at
  startup with a raw traceback.

---

## 🟢 Packaging / dependencies / deployment

- **`asyncio` listed as a dependency** ([pyproject.toml:31](pyproject.toml:31),
  resolved to a third-party `asyncio 4.0.0` in `uv.lock`). `asyncio` is part of the
  **standard library**; depending on a PyPI package of the same name can shadow it
  and is never correct. Remove it.
- **`requirements.txt` is stale and inconsistent with `pyproject.toml`.** It is
  missing runtime deps that the app needs — `fastapi`, `python-socketio`,
  `dash`, `dash-bootstrap-components`, `xgboost`, `joblib`, `scikit-learn`,
  `huggingface-hub`, `pyttsx3` — so a `pip install -r requirements.txt` install
  can't run the decision engine or the web UI. It also pins **both**
  `opencv-python` and `opencv-contrib-python` ([requirements.txt:40-41](requirements.txt:41)),
  the exact `cv2` clobber that [setup_cuda_torch.py](scripts/setup_cuda_torch.py:42)
  exists to undo. Pick a single source of truth (the `pyproject.toml`/`uv.lock`) and
  delete or regenerate `requirements.txt`.
- **`pandas-stubs` is a runtime dependency** ([pyproject.toml:26](pyproject.toml:26))
  — it's a type-stub package; move to a dev/optional group.
- **`psutil` is imported but not declared** ([gpu_deadline_watcher.py:41](scripts/gpu_deadline_watcher.py:41)).
- **Dockerfile/compose are broken/stale.** The lock is `required-environments =
  win32/AMD64` with a Windows-only CARLA wheel ([pyproject.toml:43](pyproject.toml:43),
  [:62](pyproject.toml:62)), so `uv sync --locked` on `linux/amd64`
  ([Dockerfile:3](Dockerfile:3)) cannot resolve CARLA. `docker-compose.yml` still
  references `carlasim/carla:0.9.16` (project is on 0.10.0) and `src/ProVoice/drive.py`
  (file lives at `src/drive/drive.py`). `EXPOSE` ports (8050/2002/8000/8002) don't
  match the actual server port (8001). Either fix the Linux path end-to-end or
  remove Docker support and say so.
- **~124 MB of model weights committed, duplicated across two trees.** `trained_models/`
  (62 MB) and `src/ProVoice/trained_models/` (62 MB) are **identical copies** (27+
  `.hdf5` emotion checkpoints each, plus `.pkl`/`.xml`). `.gitignore` *intends* to
  exclude models ("models … are NOT committed") and ignores `*.pt`, but **not**
  `*.hdf5`/`*.pkl`/`*.xml`, so they slipped in. Keep one copy of the one emotion
  model you actually load ([data_collector.py:68](src/ProVoice/data_collector.py:68)
  uses `fer2013_mini_XCEPTION.102-0.66.hdf5`), delete the rest and the duplicate
  tree, and extend `.gitignore` (or use Git LFS / HF Hub as you already do for YOLO).

---

## ⚪ Maintainability / cleanup

- **Dead / scratch code:**
  - `src/ProVoice/tools/{datasets,inference,preprocessor}.py` — vendored from
    `face_classification`, unused at runtime, and broken on the pinned stack
    (`DataFrame.as_matrix()` removed in pandas ≥1.0
    [datasets.py:71](src/ProVoice/tools/datasets.py:71); `keras.preprocessing.image`
    grayscale arg gone in keras 3 [inference.py:10](src/ProVoice/tools/inference.py:10);
    `imageio`/`PIL` not even declared deps [preprocessor.py:2](src/ProVoice/tools/preprocessor.py:2)).
  - [read.py](src/ProVoice/read.py:1) — 6-line scratch script that runs TTS on
    *import* and reads a non-existent `new.txt`. Delete or guard under `__main__`.
  - `demo.py`, `simcall_simulation.py` — standalone demos; mark clearly or move to an `examples/` dir.
- **`test.py` is not a test** ([src/ProVoice/test.py](src/ProVoice/test.py:1)) — it's
  a manual webcam viewer that pytest would try to collect. Rename (e.g. `gaze_demo.py`).
  It also divides by `left_eye_width` with no zero guard ([:33](src/ProVoice/test.py:33)).
- **Duplicated logic:** `FCD_NAMES` is redefined in 4 files (`fcd_config`, `eval`,
  `merge`, `extract`); `compute_gaze_score` exists in both `data_collector.py` and
  `test.py`. Centralise.
- **Misleading capability flags:** `HAS_CV2/HAS_NP/HAS_MP/HAS_MYFRAME` are hard-coded
  `True` ([data_collector.py:21-25](src/ProVoice/data_collector.py:21)) because the
  imports above them are unconditional — the guards that check them are dead.
- **Mixed Chinese/English** comments and user-facing strings throughout
  (`logger.py` "写处理后数据失败", `eval.py`/`extract.py`/`data/*` headers,
  actuator's full-width parentheses `（…）`). Pick one language for shipped strings.
- **No tests, linting, type-checking, or CI.** Type hints exist but nothing enforces
  them; adding `ruff` + `mypy` would have caught C1 and C5 immediately. A handful of
  unit tests around `decision_engine` (probs→LoA mapping, fusion, fallback) and the
  logger schema would protect the experiment data.
- **README drift:** mentions `start_both.py` (doesn't exist; it's `start_experiment.py`),
  describes the launcher as opening terminal windows / using `PV_SESSION_ID` while
  the current `start_experiment.py` uses `subprocess` + a `.session_id` file, and
  documents `camera_source=front` though the code only special-cases
  `udp`/`local`/digit ([main.py:311](src/ProVoice/main.py:311)).

---

## ⚠️ Security / safety notes (low urgency, research context)

- [gpu_deadline_watcher.py](scripts/gpu_deadline_watcher.py:54) terminates **other
  users'** Python processes by interpreter-path heuristic. Fine as a personal
  one-off, but it shouldn't live in a shared repo without a very loud warning.
- The dashboard binds `0.0.0.0:8001` with socket.io `cors_allowed_origins="*"`
  ([webui/app.py:24](src/ProVoice/webui/app.py:24), [main.py:342](src/ProVoice/main.py:342)) —
  it streams a live webcam of the driver to any host on the network. Bind to
  `127.0.0.1` unless remote access is required, and restrict CORS.
- The vehicle-state bridge ships speed/location over plain HTTP via ngrok
  ([vehicle_state_server.py](scripts/vehicle_state_server.py:1)) with no auth.

---

## Suggested order of attack (highest value first)

1. **C1**: replace `except NotImplementedError` → `except Exception` repo-wide. (Trivial, unblocks everything.)
2. **C5**: fix `wheel.py` control construction (if a wheel is used at all).
3. **C2/C3/C4**: make xLSTM train/serve architecture, loss/output, and feature
   order match — or disable the xLSTM path until it does, since it's the default.
4. **C6**: stabilise the decisions-CSV schema.
5. **H-series**: collapse to one FaceMesh, drop hot-loop prints, throttle the
   dashboard emitter, move YOLO off the per-frame CPU path.
6. **Packaging**: drop `asyncio`, reconcile `requirements.txt`↔`pyproject.toml`,
   de-duplicate/ignore the committed model weights.
7. Add `ruff`+`mypy`+a few unit tests so these don't regress.
