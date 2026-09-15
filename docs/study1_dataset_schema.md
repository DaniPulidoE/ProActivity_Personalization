# Study 1 dataset — column schema for the Hugging Face release

Column inventory of the population data collection (12 drivers × 2 sessions,
2026-08), computed over every row of `data/study1_data/` on 2026-09-13.
Types are the **declared types** — written into the dataset card as
`dataset_info.features` so `load_dataset` casts instead of inferring. Where the
source file is mixed the JSON types are given in parentheses; the release
normalises them (int `0` → `false`, int → float) because Arrow inference fails
on these files (`hr_repair_method` null→string) or is order-dependent
(`is_night`/`is_junction` bool/int). The trainers are indifferent to the
normalisation — see "Verification" below.

Files in the release:

| HF config | Source file | Published as | Rows | Cols | Unit of one row |
| --- | --- | --- | --- | --- | --- |
| `frames_raw` | `raw_data.jsonl` | `data/frames_raw.jsonl` | 380,990 | 63 | one DataCollector tick (~20 Hz), HR as filtered live |
| `frames_preprocessed` | `preprocessed_data.jsonl` | `data/frames_preprocessed.jsonl` | 380,990 | 65 | same frames, HR rebuilt offline by `heart_rate_preprocessing.py` (+2 provenance cols) |
| `labels` | `user_loa_labels.csv` | `data/labels.jsonl` **and** `data/labels.csv` | 1,446 | 22 | one driver prompt (two per 20 s window) |
| `labeled` | `labeled_data.jsonl` | `data/labeled.jsonl` | 508,282 | 74 | one (frame, label) pair — the training file; **frames are duplicated once per label**, see §C |
| — | `calibration_data_study/` | `calibration/calibration_<pid>.json`, `calibration/calibration_logs/log_calibration_<pid>.csv` | 12 + 12 files | — | per-driver 60 s calibration baseline and per-tick log — inputs of the HR repair |

**The release is built for full reproducibility**, which fixes the formats:
files are published in exactly the shapes the pipeline reads (JSONL frames,
CSV labels), so `heart_rate_preprocessing.py`, `build_loa_dataset.py` and every
trainer run on them unchanged. Two consequences:

- The Hub applies a single file format to all configs of one repo, so the
  `labels` config is `data/labels.jsonl`; `data/labels.csv` is the identical
  table in the CSV form `build_loa_dataset.py` takes (not a config; the build
  checks the two agree row for row).
- `calibration/` is included because `frames_raw` alone cannot regenerate
  `frames_preprocessed`: the HR repair needs each driver's calibration baseline
  and per-tick calibration log.

`scripts/upload_study1_hf.py` stages the release, refuses to upload if the
row/column counts differ from this table, and with `--verify-chain` re-runs
the pipeline on the staged inputs — the results are recorded in the card:

**Verification (2026-09-13, code `14a3be2`).** `heart_rate_preprocessing.py`
on the released `frames_raw` + `calibration/` reproduces the released
`frames_preprocessed` on all 380,990 rows and every published column;
`build_loa_dataset.py` on that output + `labels.csv` reproduces the released
`labeled` on all 508,282 rows; `train_XLSTM.normalize_row` applied to the
original `labeled_data.jsonl` and to the released `labeled.jsonl` is identical
on every row, so a model trained on the release is the model trained on the
original. Each config also loads through `datasets` with the declared types
and `participantid` keeps its zero padding.

The two frame files are byte-identical in every column outside the rPPG block
(0 differing cells over 380,990 rows); only `heart_rate`, `hr_delta` and the
`hr_repair_*` columns differ.

---

## A. Frame files — `frames_raw` (63 cols) / `frames_preprocessed` (65 cols)

Columns marked **P** exist only in `frames_preprocessed`.

### A.1 Identity / session context

| Column | Type | Null | Notes |
| --- | --- | --- | --- |
| `timestamp` | string | 0 % | Wall-clock `HH:MM:SS.mmm`, local time, **no date**. Join to `labels` via `session_id` + `window_start_timestamp`/`window_end_timestamp`. |
| `session_id` | string | 0 % | UUID; 24 distinct |
| `participantid` | string | 0 % | `001`–`012` |
| `traffic_seed` | int | 0 % | CARLA traffic-scenario seed (`TRAFFIC_SEED_PLAN` in `start_experiment.py`) |
| `environment` | string | 0 % | constant `city` |
| `secondary_task` | string | 0 % | constant `none` |
| `modeltype` | string | 0 % | constant `combined` (run config) |
| `state_model` | string | 0 % | constant `xlstm` (run config) |
| `w_fcd` | float | 0 % | constant `0.7` (run config) |

### A.2 Face / driver state (MediaPipe FaceLandmarker, EmotiEffLib, YOLO26)

| Column | Type | Null | Notes |
| --- | --- | --- | --- |
| `face_present` | bool | 0 % | landmarker found a face |
| `eye_ar` | float | 0 % | eye aspect ratio (model feature `ear`) |
| `mar` | float | 0 % | mouth aspect ratio |
| `gaze_score` | float | 0 % | z-scored against the 180 s calibration baseline |
| `gaze_score_raw` | float | 0 % | uncalibrated gaze score |
| `gaze_distracted` | bool | 0 % | `gaze_score` above calibrated threshold (mean + 2.5·std) |
| `blink_rate` | float | 0 % | normalized (Poisson) against calibration mean |
| `blink_rate_raw` | float | 0 % | blinks · min⁻¹ |
| `yawn_rate` | float | 0 % | normalized |
| `yawn_rate_raw` | float | 0 % | yawns · min⁻¹ |
| `perclos` | float | 0 % | z-scored |
| `perclos_raw` | float | 0 % | fraction of time eyes closed |
| `drowsiness_alert` | bool | 0 % | PERCLOS + MAR rule |
| `emotion` | string | 0.2 % | one of `angry, disgust, fear, happy, sad, surprise, neutral`; `null` = no reading (no face / classifier failure) |
| `emotion_prob` | float | 0.2 % | confidence of `emotion`; `null` iff `emotion` is null |
| `lab` | list\<string\> | 0 % | YOLO26 distraction classes present: subset of `face`, `phone`, `drink`; may be empty |
| `facebox_misses` | int | 0 % | face-box worker detector misses (cumulative) |
| `facebox_consec_misses` | int | 0 % | consecutive misses; box goes stale at 4 s |

### A.3 rPPG heart rate / respiration (MMRPhys, SCAMPS LEF 72×72)

| Column | Type | Null | Notes |
| --- | --- | --- | --- |
| `heart_rate` | float | 1.6 % | bpm. `frames_raw`: live-filtered reading. `frames_preprocessed`: offline-repaired — differs on 80,156 frames (21 %) |
| `heart_rate_raw` | float | 1.6 % | unfiltered estimator output |
| `hr_delta` | float | 1.6 % | `heart_rate` standardized against the per-driver baseline (median / SD of cleaned calibration readings, floor 5 bpm); recomputed in `frames_preprocessed` |
| `hr_rejected` | bool | 1.6 % | live 2f-harmonic filter rejected `heart_rate_raw` |
| `respiratory_rate` | float | 1.6 % | breaths · min⁻¹. **Not a model input** — RGB respiration from the synthetic-trained checkpoint was judged noise |
| `rr_delta` | float | 1.6 % | standardized RR (median / MAD baseline) |
| `rppg_gaps` | int | 0 % | look-away discontinuities > 1 s that spliced the model window |
| `rppg_dropped` | int | 0 % | frames lost to a full rPPG queue |
| `rppg_suppressed` | int | 0 % | readings flagged as gap-contaminated (never discarded in study 1) |
| `rppg_harmonic_rejects` | int | 0 % | probable 2f rejections |
| **P** `hr_repaired` | bool | 0 % | `heart_rate` was rewritten offline |
| **P** `hr_repair_method` | string | 79 % | `folded` (45,023 — 2f harmonic halved), `interpolated` (33,597 — outlier replaced), `carried` (1,536); null when not repaired |

### A.4 Vehicle / world (CARLA 0.10, via the vehicle-state bridge)

| Column | Type | Null | Notes |
| --- | --- | --- | --- |
| `speed_kmh` | float | 0 % | ego speed |
| `speed_limit_kmh` | float (int/float) | 0 % | `0` on the first 20 frames of each session (bridge not yet connected), else `30.0` |
| `speed_ratio_max` | float | 0 % | `speed_kmh / 150` |
| `speed_ratio_limit` | float (int/float) | 0 % | `speed_kmh / speed_limit_kmh`; `-1` when the limit is unknown |
| `throttle` | float | 0 % | [0, 1] |
| `brake` | float (int/float) | 0 % | [0, 1] |
| `steer` | float (int/float) | 0 % | [−1, 1] |
| `acceleration` | float | 0 % | magnitude, m · s⁻² |
| `gear` | int | 0 % | |
| `reverse` | bool | 0 % | |
| `hand_brake` | bool | 0 % | constant `False` |
| `is_junction` | bool (bool/int) | 0 % | ego waypoint is in a junction; int `0` only on the first 20 frames of each session |
| `is_night` | bool (bool/int) | 0 % | constant `False` (sun above horizon in every session) |
| `traffic_light_state` | string | 0.1 % | `Red` / `Yellow` / `Green` |
| `lead_distance_m` | float (int/float) | 0 % | distance to lead vehicle in ego lane, **scaled `/100`** (so ≈ [0, 1]); `-1` = no lead vehicle within 100 m (73 % of frames). Sentinel, not zero — 0 would mean contact |
| `headway_s` | float | 79 % | time headway, s; `null` = no lead vehicle, or ego below walking pace |
| `precipitation` | float (int/float) | 0 % | constant `0` (no weather in CARLA 0.10) |
| `fog_density` | float | 0 % | constant `0.0` |
| `headlight` | bool | 0 % | constant `False` |
| `fog_light` | bool | 0 % | constant `False` |
| `left_indicator` | bool | 0 % | constant `False` |
| `right_indicator` | bool | 0 % | constant `False` |

### A.5 Pipeline timing

Present on most frames, absent on a few early ticks per session (hence two key
sets in the JSONL).

| Column | Type | Notes |
| --- | --- | --- |
| `frame_dt_ms` | float | inter-frame gap |
| `collect_ms` | float | perception loop time for this tick |
| `fps_inst` | float | instantaneous achieved rate |
| `fps_avg` | float | running mean achieved rate — use to identify participant 001's ~4 Hz warm-up (session `77b516f6`, windows 1–12) |

---

## B. `labels` — `user_loa_labels.csv` (1,446 rows, 22 cols)

One row per prompt. Each 20 s window carries two prompts (`--random-function`),
so 723 windows → 1,446 rows.

| Column | Type | Notes |
| --- | --- | --- |
| `session_id` | string | join key to frames |
| `participantid` | string | `001`–`012` |
| `window_idx` | int | 1-based window index within the session |
| `prompt_in_window` | int | `1` or `2` |
| `window_start_ms` | int | window start, CARLA sim time |
| `window_end_ms` | int | window end, CARLA sim time (= start + 20,000) |
| `window_start_timestamp` | string (ISO 8601) | wall-clock window start — join to frame `timestamp` |
| `window_end_timestamp` | string (ISO 8601) | wall-clock window end |
| `selection_timestamp` | string (ISO 8601) | when the driver submitted the answer |
| `selection_frame` | int | CARLA frame at submission |
| `selection_sim_time` | float | CARLA sim time at submission, s |
| `selection_speed_kmh` | float | ego speed at submission |
| `functionname` | string | **the prompted task**, one of five: `Provide traffic news` (307), `Respond to a text message` (296), `Respond to a phone call` (292), `Change song` (276), `Provide weather update` (275) |
| `user_selected_loa` | string | **ground truth.** Single LoA `0`–`4` (1,348 rows), or a `;`-joined set of acceptable LoAs (98 rows, 6.8 %): `1;2` ×40, `0;1` ×30, `2;3` ×17, `3;4` ×5, `0;1;2;3` ×4, `1;2;3` ×1, and one non-contiguous `0;3` |
| `ambient_gain` | float | ambient-audio config (constant `0.35`) |
| `ambient_seed` | int | ambient-audio config (constant `0`) |
| `ambient_source` | string | ambient-audio config (constant) |
| `environment` | string | constant `city` |
| `secondary_task` | string | constant `none` |
| `modeltype` | string | constant `combined` |
| `state_model` | string | constant `xlstm` |
| `w_fcd` | float | constant `0.7` |

---

## C. `labeled` — `labeled_data.jsonl` (508,282 rows, 74 cols)

This is the file the xLSTM population model and every per-driver adaptation
were trained on. It is **fully derived** from `frames_preprocessed` and
`labels` by `scripts/build_loa_dataset.py`; it is included so the training
input is available as-is, without requiring the join to be reproduced.

### C.1 How it was generated

1. Each row of `labels` defines a 20 s window
   `[window_start_timestamp, window_end_timestamp]` within one `session_id`.
2. Every frame of `frames_preprocessed` whose `session_id` matches and whose
   wall-clock `timestamp` falls inside that window is attached to the label.
   Frames outside every window (calibration, the gaps between windows) are
   dropped: 254,141 of 380,990 frames are labelled.
3. The driver's `user_selected_loa` becomes the target — never the system's
   own `LoA` (which is null here anyway, and would be circular in a served
   session).
4. `functionname` and `FCD` are taken from the **label row**, not the frame.
   The collector stamps the CLI default (`Adjust seat positioning`) on every
   frame; the label records the task the driver was actually asked about, and
   its FCD vector is looked up in `fcd_config.py`.

```bash
python scripts/build_loa_dataset.py \
    --raw data/preprocessed_data.jsonl \
    --labels data/user_loa_labels.csv \
    --out-jsonl data/labeled_data.jsonl \
    --out-fcd data/processed_data/fcd_out.csv
```

### C.2 Why rows are duplicated

Under `--random-function` the drive UI asks **two prompts per 20 s window**,
about two different tasks, and the driver answers each separately. Both label
rows share the same window bounds, so they select the same frames. Each
frame is therefore emitted **once per label**: identical driver-state and
vehicle features, different `functionname`, `FCD`, `user_loa` and
`segment_id`. Taking only the first match would discard half the labels.

Consequences for anyone using the file:

- **The unit of analysis is `segment_id`, not the row.** 1,446 segments =
  1,446 labels; 508,282 rows = 254,141 distinct frames × 2. Row counts
  double-count frames; `(session_id, timestamp)` identifies a physical frame.
- Frames per segment: min 80, median 381, max 396 (the 80-frame segments are
  participant 001's ~4 Hz warm-up windows, see §A.5).
- A segment carries one label; the frames within it are one time series.
  Train/validation/test splits must be made at the **segment** (or window)
  level — splitting by row leaks a window's frames across splits.
- The two segments of one window are not independent samples of driver state:
  they differ only in the task asked about. Treat them as such in any
  analysis of the state features.

### C.3 Columns

All 65 `frames_preprocessed` columns (§A) are passed through verbatim, plus
two columns taken from the **label row** (the frame files do not carry them —
see §D for why the collector's own `functionname`/`FCD` were dropped):

| Column | Type | Notes |
| --- | --- | --- |
| `functionname` | string | the prompted task (five values, as in `labels.functionname`) |
| `FCD` | struct\<12 × int\> | the task's FCD vector, keys `Safety Risk, Increased Safety, Relevance, Magicality, Privacy, Trust, Time Consumption, Repetitiveness, Situational Context, Social Risk, Urgency, Complexity`, values 1–5; identical for all rows of a segment |

plus seven columns added by the builder:

| Column | Type | Notes |
| --- | --- | --- |
| `segment_id` | string | `<session_id>\|winNNNpM` (`NNN` = window index, `M` = prompt 1/2) — the unit of one label. **Count these, not rows.** |
| `user_loa` | string | copy of `labels.user_selected_loa` for this segment (`;`-joined sets preserved) |
| `Level_1` … `Level_5` | int | multi-hot of `user_loa`; `Level_k` = 1 iff LoA `k−1` is in the set |


---

## D. Columns excluded from the release

Present in the on-disk files, deliberately not uploaded.

| File | Column(s) | Why |
| --- | --- | --- |
| frames (both) | `functionname` | Constant `Adjust seat positioning` — the CLI default ProVoice was started with, **not** the task the driver was prompted about. Non-null and plausible-looking, so more misleading than an empty column. The prompted task is `labels.functionname` (and `labeled.functionname`, which is taken from the label row). |
| frames (both) | `LoA`, `FCD` | 100 % null — no model was served in study 1. The collector emits them because in a served session they hold the decision in force at that frame. |
| frames_preprocessed, labeled | `hr_repair_reason` | Free-text audit string for each HR repair (the rule that fired, e.g. `2f harmonic of session 52`). Redundant with `hr_repaired` + `hr_repair_method` for any downstream use; kept only in the local files. |
| labeled | `LoA` | 100 % null, passed through from the frames (see above). `FCD` is **kept** in `labeled` — the builder overwrites it with the prompted task's vector (§C.3). |
| labels | `emotion`, `system_action`, `system_level`, `system_loa`, `system_message`, `system_probs`, `system_profile`, `system_fallback`, `system_fallback_reason`, `system_fcd` | 100 % empty — no system prediction was shown to the driver in study 1. |

Under `calibration/`, only the 12 study participants' `calibration_<pid>.json`
and `calibration_logs/log_calibration_<pid>.csv` are released: the
`calibration_<pid>_preprocessed.json` files are *outputs* of
`heart_rate_preprocessing.py` (the reproduction chain regenerates them), and
participant `998` is a rig test, not part of the study.