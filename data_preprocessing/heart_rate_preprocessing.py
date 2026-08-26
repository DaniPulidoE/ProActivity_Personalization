"""Rebuild the heart-rate channel of ``raw_data.jsonl`` offline.

The rPPG estimator (MMRPhys) reports one heart rate every ~6 s by taking the
largest FFT peak in 36-198 bpm. A real pulse waveform carries genuine energy at
twice its rate, so for any driver under 99 bpm that 2f peak is also inside the
search band and the estimator sometimes locks onto it -- reporting ~140 for a
true ~70. It can also lose the pulse entirely and settle on a slower rhythm.
Neither failure is fully recoverable at serving time, because the live filter can
only compare a reading against the ones before it. This script redoes the job
with the whole session visible.

  IN   ``data/raw_data.jsonl``                     (frames, never modified)
       ``data/calibration_data/calibration_logs/`` (per-tick calibration logs)
  OUT  ``data/preprocessed_data.jsonl``            (same frames, HR rebuilt)
       ``data/calibration_data/calibration_<pid>_preprocessed.json``

**The raw column is intact, which is what makes this possible.** ``DataCollector``
writes every reading to ``heart_rate_raw`` BEFORE the live filter judges it
(``data_collector.py:2148``), alongside ``hr_rejected`` recording the verdict.
``heart_rate`` is the filtered column -- the last ACCEPTED reading, carried
forward -- and it is what ``hr_delta`` was derived from. This script reads
``heart_rate_raw`` and rewrites ``heart_rate`` / ``hr_delta``.


THE PIPELINE
============

**1. Recover the readings** (``reading_spans``). A reading is repeated on every
frame until the next arrives, so 380,990 frames collapse to ~2,500 distinct
readings. A new one begins wherever ``(heart_rate_raw, hr_rejected,
rppg_harmonic_rejects)`` changes -- the reject counter is monotonic, which is
what makes a REJECTED reading detectable even though ``heart_rate`` never moved
for it. Everything downstream works on readings, and is broadcast back to frames
at the end.

**2. Settle the octave, once per participant** (``session_fundamental``). This
must happen before anything local, because a driver can be bimodal: 012's first
session splits 53/55 between a cluster near 60 and one near 120, so the session
median (86) lands in the empty gap and is a rate they never had. Inside the high
cluster a local window has a HARMONIC majority, and any drift-tracking reference
computed there would take the harmonic as truth and flag the correct readings
instead. An earlier version of this script did exactly that, "repairing" 012's
true 56-60 bpm readings UP to ~110-135.

The tie is broken by physics, not preference: the estimator can report a rate
DOUBLED, but nothing in that pipeline can report one HALVED. Given modes at x and
~2x, the fundamental is x. Mechanically this is harmonic folding, the standard
octave-error fix in pitch tracking -- find the dominant mode, then test whether a
mode also exists near half of it. ``--octave-min-support`` is what stops the
preference running away: halving ANY rate explains the same readings just as
well, purely as harmonics of nothing, so 001's single tight mode at 88 would
otherwise return 44. The low mode must EXIST before it can win.

The vote pools the calibration log with every session of that participant -- same
driver, same rig, same afternoon. ``VERIFIED_FUNDAMENTAL`` overrides it where the
answer is known from outside the pipeline (010, confirmed by the participant's
own smartwatch).

**3. Flag the bad readings** (``flag_readings``), in two stages:

  a. *Octave*, against the participant's ``f0``: anything in ``[1.7*f0, 2.3*f0]``
     is a 2f lock, anything above ``2.3*f0`` is not a rate this driver had, and
     anything outside 40-180 bpm is not a pulse at all.
  b. *Drift*, against the median of each reading's ``--window`` neighbours --
     because a pulse genuinely moves over a 30-minute drive (007 spans 58-100)
     and one session-wide number would flag real physiology as error.

Stage (b) carries two guards that exist because of measured failures. The gate is
CAPPED (``--max-gate``): scaled purely by the local MAD it is self-defeating,
widened by the very contamination it should reject -- 009's local MAD of 13.5
inflated it to 70 bpm, admitting everything from 20 to 150 around a reference of
81. And a reading that fails the gate is re-labelled a 2f lock when HALVING
brings it closer to the local reference, which catches harmonics stage (a)
structurally cannot: ``f0`` is one number for a driver whose pulse drifts, so
009's 130/132/137 (2f of 65/66/68, rates they demonstrably have) fall below the
[138, 186] band that f0=81 implies.

There is deliberately NO subharmonic rule anywhere. With the octave settled it is
unnecessary, and it was the rule that inverted on 012.

**4. Repair, two ways** (``repair``), because "wrong" comes in two kinds:

  * an identified 2f lock is FOLDED (halved) -- we know not merely that it is
    wrong but what it is, so halving recovers the measurement;
  * everything else is INTERPOLATED, the mean of the nearest GOOD reading before
    and after ("nearest good", not adjacent, so a bad run is never averaged
    against itself; a single neighbour is carried at a session edge).

The two agree except where a lock is sustained. For ten of twelve participants
the longest bad run is 4 readings (~24 s) and the choice barely matters; for 010
it is 22 readings, where averaging two unrelated endpoints produced a 2.2-minute
flat plateau worse than the readings it replaced. Folded values then count as
good neighbours for the interpolation of the rest. Folding is safe only because
step 2 settled the octave first; ``--session-repair interpolate`` restores the
pure neighbour rule.

Neither rule smooths: a reading that passes is never modified, so genuine
excursions keep their full amplitude.

**5. Rebuild the baseline, then ``hr_delta``.**
``hr_delta = (heart_rate - mean) / std`` is the ONLY path the heart rate takes
into the model (``STATE_NUM`` in ``xlstm_model.py``), which makes the baseline
part of this filter rather than an input to it. The stored
``calibration_<pid>.json`` baselines were computed from UNFILTERED readings on
whatever octave they landed: 010's is 111.5 bpm, the 2f of a pulse near 55. Left
alone it would subtract 111 from a ~55 bpm series and pin ``hr_delta`` at a
constant, undoing steps 2-4 at the last moment.

So the calibration readings go through the same filter on the same ``f0``
(``clean_calibration``). There harmonics are folded rather than dropped, and no
alternative is worth having: 28 readings have no neighbours to interpolate from,
and dropping 010's leaves TWO. The calibration stays the baseline -- it is a
dedicated per-driver measurement, it is what the live system loads, and it is
recorded WHILE DRIVING (``data_collector.py:919``), so it measures the same
activity as the sessions it normalizes. Only the noise is removed.

Location is the median of the surviving readings; scale is their sample SD,
floored at ``--min-scale``. The floor matters because the calibration std is a
180 s estimate from ~25 noisy readings that only weakly predicts a driver's real
range (Spearman 0.55, n=12) while underestimating it ~1.9x -- below the floor it
is measuring estimator noise, and ``hr_delta`` divides by it. A driver who
genuinely varies more keeps their own std. ``hr_delta`` is finally clipped to
``+-_HR_DELTA_CLIP``; unclipped it reached +-12 on the one ``STATE_NUM`` column
not already inside [0,1].

``--baseline session`` -- and the automatic fallback when a calibration log is too
thin -- sources the baseline from the participant's own preprocessed readings
instead. 001 is in that state: its log holds a single reading, because
``_open_calibration_log`` opens with mode ``'w'`` and a later aborted run
truncated it. That changes what the baseline MEANS, so it is reported, never
silent.


WHAT IS WRITTEN
===============

Every input frame is written out in its original order. ``heart_rate`` and
``hr_delta`` are rewritten; ``heart_rate_raw`` and ``hr_rejected`` are preserved
untouched, so the live filter's verdict stays auditable and this script can be
re-run on its own output. Three provenance keys are added: ``hr_repaired``,
``hr_repair_reason`` and ``hr_repair_method``
(``folded`` / ``interpolated`` / ``carried``).

``calibration_<pid>.json`` is never modified -- derived baselines go to the
``_preprocessed`` copy, validated against the live schema first, because a file
that fails ``load_calibration`` would silently trigger a 180 s live
recalibration at session start.

The report prints per-participant counts and, separately, the things that must
not pass silently: an undecided octave, a baseline still too spread to be a
pulse, a stored baseline that is a harmonic of the one found here, a
session-sourced baseline, and any session whose median ``hr_delta`` sits too far
from zero to be a real difference between the calibration window and the drive.


RELATIONSHIP TO THE LIVE PATHS
==============================

There are three callers of the heart-rate filter, and only two of them can be
made equivalent. The algorithm and every constant it uses live in
``src/ProVoice/hr_filter.py``, which this script and ``data_collector.py`` both
IMPORT -- not a copy, not an AST-parsed set of numbers, the same functions.

**(A) This script.** Sees every session of every participant at once. Resolves
the octave over all of a participant's readings, and repairs a non-harmonic hole
from the readings on BOTH sides of it.

**(B) ``DataCollector.compute_calibration``.** Computes the stored baseline at
the END of the calibration phase. Crucially this is a BATCH step -- it runs once,
with all ~28 readings already collected -- so it runs the SAME
``clean_calibration`` and the SAME ``baseline_from_readings`` this script does.

  Measured: feeding both the identical calibration readings, the live baseline
  reproduces the dataset's baseline EXACTLY for 10 of 12 participants. The two
  exceptions are the two degenerate ones -- 001, whose log holds a single usable
  reading, and 010, discussed below.

  What (B) still lacks is EVIDENCE, not algorithm. This script pools the
  calibration readings with every session reading before voting on the octave
  (~530 values); (B) has only the ~28 calibration readings. Measured over the
  study cohort with the external override disabled, the octave voted from the
  calibration alone agrees with the pooled vote for **10 of 11** participants.
  The exception is 010, whose calibration is 26/28 harmonics: alone it votes
  114 bpm, pooled it votes 52. So a driver whose CALIBRATION is itself mostly 2f
  can still be resolved to the wrong octave live, and only a post-hoc run of this
  script fixes it -- it rewrites ``calibration_<pid>_preprocessed.json``, and the
  next session loads the corrected value.

**(C) ``DataCollector._capture_loop``.** Per reading, during a session. STRICTLY
CAUSAL, and therefore permanently different:

  * it cannot resolve an octave -- that needs the whole recording at once;
  * it cannot interpolate a hole -- that needs the readings after it, so it folds
    what it can identify and carries the previous value forward otherwise.

  It does share the range gate, the 2f band and the fold rule, and it seeds its
  reference from the stored (octave-corrected) baseline so it is anchored from
  the first reading rather than disarmed for three.

  The residual gap is measurable: of the 646 readings this script flags, the live
  filter caught 173 under the code that recorded the study data. About a fifth of
  readings therefore reach a served model as values training never saw -- the
  same class of train/serve skew that hid in ``ear`` for weeks (see CLAUDE.md).
  That figure predates the fold/range-gate/baseline-seed changes and has not been
  re-measured on data recorded under them.

Usage::

    python data_preprocessing/heart_rate_preprocessing.py --dry-run
    python data_preprocessing/heart_rate_preprocessing.py
    python data_preprocessing/heart_rate_preprocessing.py --participants 010 --verbose

--participants scopes processing AND reporting to the listed drivers, but never
writes --out-data -- it would otherwise overwrite the full cohort file with a copy
where every other driver's frames are unrepaired. Use it to inspect one driver's
flags/report; re-run without --participants to actually write the repaired file.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_COLLECTOR = REPO_ROOT / "src" / "ProVoice" / "data_collector.py"

# Fields the SERVING path indexes (DataCollector._CALIBRATION_SCHEMA). A file
# written by this script must satisfy the same contract or load_calibration()
# rejects it and silently falls back to a live 180 s calibration.
REQUIRED_SCHEMA: Dict[str, Tuple[str, ...]] = {
    "gaze_score": ("mean", "std", "threshold"),
    "ear": ("mean", "std", "threshold"),
    "mar": ("mean", "std", "threshold"),
    "bpm": ("mean", "std"),
    "rr": ("mean", "std"),
    "blink_rate": ("mean",),
    "perclos": ("mean", "std"),
}

# The cleaning algorithm and every constant it uses live in ProVoice.hr_filter,
# which DataCollector.compute_calibration also imports. That shared import is
# what makes the offline baseline and the live baseline the same computation
# rather than two that happen to agree. Importing a ProVoice submodule is cheap
# (PEP 562 lazy __init__): measured 0.03 s and no torch, cv2 or fastapi.
sys.path.insert(0, str(REPO_ROOT / "src"))
from ProVoice.hr_filter import (           # noqa: E402
    DEFAULT_WINDOW,
    HARMONIC_HI,
    HARMONIC_LO,
    HR_DELTA_CLIP,
    HR_FOLD_MARGIN,
    HR_MAD_K,
    HR_MAX_GATE,
    HR_MIN_DEVIATION,
    HR_MIN_SCALE,
    HR_RANGE,
    HR_REF_MIN,
    MAD_TO_SIGMA,
    VERIFIED_FUNDAMENTAL,
    baseline_from_readings,
    clean_calibration,
    flag_readings,
    session_fundamental,
    standardized_delta,
)


def read_calibration_log(path: Path) -> Tuple[List[float], List[float]]:
    """Return the (bpm, rr) readings a calibration log recorded, in order.

    The columns are blank on every tick that produced no new estimate, so a
    non-blank cell is exactly one reading -- the same list
    ``DataCollector.compute_calibration`` aggregated into the stored baseline.
    """
    bpm: List[float] = []
    rr: List[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            for column, sink in (("bpm", bpm), ("rr", rr)):
                raw = (row.get(column) or "").strip()
                if raw:
                    try:
                        sink.append(float(raw))
                    except ValueError:
                        pass
    return bpm, rr


def validate_calibration(cal: Dict[str, Any]) -> Optional[str]:
    """Mirror DataCollector._validate_calibration; return None when usable.

    Run before writing, because a calibration file that fails validation is
    worse than none: load_calibration() would reject it at session start and
    fall back to a live 180 s calibration, which looks like a UI hang.
    """
    for key, fields in REQUIRED_SCHEMA.items():
        entry = cal.get(key)
        if not isinstance(entry, dict):
            return f"missing or malformed '{key}' entry"
        for field in fields:
            v = entry.get(field)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return f"'{key}.{field}' is not a number ({v!r})"
            if not math.isfinite(float(v)):
                return f"'{key}.{field}' is not finite ({v!r})"
    for key in ("gaze_score", "ear", "mar"):
        if float(cal[key]["threshold"]) <= 0.0:
            return f"'{key}.threshold' is not positive"
        if float(cal[key]["mean"]) <= 0.0:
            return f"'{key}.mean' is not positive"
    if float(cal["perclos"]["std"]) <= 0.0:
        return "'perclos.std' is not positive"
    return None




# ── reading extraction ────────────────────────────────────────────────────────

def iter_frames(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield ``(line_number, frame)`` for every well-formed JSON line."""
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError:
                print(f"[hr][warn] {path.name}:{lineno} is not valid JSON -- skipped")


def reading_spans(frames: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse a session's frames into the distinct rPPG readings behind them.

    One reading surfaces every ``inference_interval`` frames and is then carried
    forward across ~115 frames, so consecutive frames repeat it. A new reading
    starts wherever ``(heart_rate_raw, hr_rejected, rppg_harmonic_rejects)``
    changes -- ``rppg_harmonic_rejects`` is a monotonic counter that ticks on
    every rejection, which is what makes a rejected reading detectable even
    though ``heart_rate`` does not move for it.

    Known conflation, harmless: two consecutive readings with an IDENTICAL value
    and identical accept/reject status merge into one span. Nothing downstream
    distinguishes them either -- the filter would classify them the same way and
    they interpolate to the same number -- so this costs a duplicate, not a fact.

    ``bpm_history`` would have separated even those, but the logger drops it
    before ``raw_data.jsonl`` (it is a ``DataCollector`` field, not a logged
    column), so it is not available offline.
    """
    spans: List[Dict[str, Any]] = []
    prev_key: Any = object()
    for i, f in enumerate(frames):
        raw = f.get("heart_rate_raw")
        if raw is None:
            prev_key = object()          # a staleness hole ends the current span
            continue
        key = (raw, f.get("hr_rejected"), f.get("rppg_harmonic_rejects"))
        if key != prev_key:
            spans.append({
                "value": float(raw),
                "start": i,
                "end": i + 1,
                "live_rejected": bool(f.get("hr_rejected")),
            })
            prev_key = key
        else:
            spans[-1]["end"] = i + 1
    return spans


# ── offline filter ────────────────────────────────────────────────────────────

def repair(
    values: Sequence[float],
    good: Sequence[bool],
    reason: Sequence[Optional[str]],
    args: argparse.Namespace,
) -> Tuple[List[Optional[float]], List[Optional[str]]]:
    """Recover each flagged reading; return ``(value, method)`` per reading.

    Two repairs, applied in this order, because they answer different questions.

    **Fold (halve) an identified 2f lock.** We do not merely know this reading is
    wrong, we know WHAT it is: twice the truth. Halving recovers the measurement;
    interpolating discards it and invents a number in its place. That distinction
    is worth little when bad readings come one or two at a time -- and for ten of
    the twelve participants the longest run is 4, about 24 s -- but it decides
    the outcome where a lock is sustained. Participant 010 has 183 of 244
    readings flagged, 108 of them identified 2f, in runs up to 22 readings: the
    two "good" endpoints of a 2.2-minute run are unrelated to each other, so
    their average is a flat plateau rather than an estimate, and it was visibly
    worse than the readings it replaced (p95 |hr_delta| rose 6.5 -> 9.5).

    Folding is safe only because the octave is settled per participant BEFORE
    this runs -- and for 010 by external measurement. Halve before that and a
    mis-diagnosed fold injects a wrong value at full confidence.

    A folded value is then re-checked against the local drift gate, and only the
    FOLDED indices adopt that second verdict; the rest keep the first pass's, so
    a fold cannot cascade into re-flagging readings that were already fine. A
    fold that still does not fit its neighbourhood falls through to:

    **Interpolate everything else** -- the mean of the nearest GOOD reading
    before and after. "Nearest good", not adjacent, because a bad run must not be
    averaged against itself. Readings recovered by folding count as good
    neighbours here, which is most of what makes the plateaus disappear. At a
    session edge only one side exists and that neighbour is carried.

    ``--session-repair interpolate`` disables folding and restores the pure
    neighbour-average rule. Returns ``None`` values when the session has no good
    readings at all -- the caller leaves such a session untouched.
    """
    n = len(values)
    if not any(good):
        return [None] * n, [None] * n

    out = [float(v) for v in values]
    ok = list(good)
    method: List[Optional[str]] = [None] * n

    if args.session_repair == "fold":
        folded = [i for i, (g, w) in enumerate(zip(good, reason))
                  if not g and w and w.startswith("2f harmonic")]
        for i in folded:
            out[i] = round(values[i] / 2.0, 1)
            ok[i] = True
        if folded:
            # Octave already answered by the fold, so this pass is the drift gate
            # only (f0=None). Its verdict is adopted for folded indices ONLY.
            recheck, _ = flag_readings(
                out, None, window=args.window, hr_range=(args.hr_min, args.hr_max),
                mad_k=args.mad_k, min_deviation=args.min_deviation,
                max_gate=args.max_gate, fold_margin=args.fold_margin)
            for i in folded:
                ok[i] = recheck[i]
                if ok[i]:
                    method[i] = "folded"
                else:
                    out[i] = float(values[i])      # fold rejected; fall through

    prev_good: List[Optional[float]] = [None] * n
    last: Optional[float] = None
    for i in range(n):
        prev_good[i] = last
        if ok[i]:
            last = out[i]

    next_good: List[Optional[float]] = [None] * n
    nxt: Optional[float] = None
    for i in range(n - 1, -1, -1):
        next_good[i] = nxt
        if ok[i]:
            nxt = out[i]

    for i in range(n):
        if ok[i]:
            continue
        a, b = prev_good[i], next_good[i]
        if a is not None and b is not None:
            out[i] = round((a + b) / 2.0, 1)
            method[i] = "interpolated"
        elif a is not None or b is not None:
            out[i] = float(a if a is not None else b)   # one side only: carry
            method[i] = "carried"
        else:
            method[i] = "interpolated"
    return [float(v) for v in out], method


# ── hr_delta ──────────────────────────────────────────────────────────────────
#
# ``hr_delta = (heart_rate - baseline_mean) / baseline_std`` is a MODEL INPUT
# (STATE_NUM in xlstm_model.py), and it is the only place the heart rate reaches
# the model at all. That makes the baseline and this filter one decision, not
# two: the stored ``calibration_XXX.json`` baseline was computed by
# ``compute_calibration`` from UNFILTERED rPPG readings, on whatever octave those
# happened to land, while this script may put the session on the other one.
#
# Participant 010 is the case in point -- stored baseline 111.5 bpm, and the
# octave analysis puts the session's fundamental at roughly half that. Keeping
# the stored mean would subtract ~111 from a ~55 bpm series and divide by 9.64,
# pinning hr_delta near -6 for the WHOLE session: not a driver-state feature any
# more, just a per-driver constant offset that the model would read as "this
# driver is permanently bradycardic". The normalization has to be re-derived on
# the same octave as the data it normalizes, or the channel is broken.
#
# So the baseline is recomputed here from the calibration log, through the SAME
# filter and the SAME fundamental as the session, and the result is checked
# against the preprocessed session (``centring`` below) rather than assumed.


def load_stored_baselines(calib_dir: Path, suffix: str) -> Dict[str, Tuple[float, float]]:
    """Per-participant ``(bpm mean, bpm std)`` as STORED -- the live baseline."""
    out: Dict[str, Tuple[float, float]] = {}
    for p in sorted(calib_dir.glob(f"calibration_*{suffix}.json")):
        stem = p.stem
        if suffix:
            stem = stem[: -len(suffix)]
        pid = stem.split("_", 1)[1]
        try:
            cal = json.loads(p.read_text(encoding="utf-8"))
            bpm = cal["bpm"]
            out[pid] = (float(bpm["mean"]), float(bpm["std"]))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"[hr][warn] {p.name}: no usable bpm baseline ({e})")
    return out


def derive_baseline(
    repaired: Sequence[float],
    calib_readings: Sequence[float],
    f0: Optional[float],
    args: argparse.Namespace,
) -> Tuple[Optional[Tuple[float, float]], str]:
    """Return ``((mean, std), source)`` for the bpm baseline, on ``f0``'s octave.

    **The calibration is the baseline** (``--baseline calibration``, default).
    It is a dedicated per-driver measurement, it is what the live system used,
    and -- importantly -- it is recorded WHILE DRIVING, not at rest (see the
    comment at ``data_collector.py:919``, where the EAR baseline is made robust
    to mid-blink frames for exactly that reason). So it is measuring the same
    activity as the session it normalizes, and there is no resting-vs-driving
    mismatch to correct for.

    What it does need is this script's filter. The stored baselines were
    computed by ``compute_calibration`` from UNFILTERED readings on whatever
    octave they happened to land: participant 010's is 111.5 bpm, the 2f of a
    pulse their smartwatch puts near 55. Running the calibration readings
    through ``clean_calibration`` first is what makes the calibration usable as
    a baseline rather than a source of a constant offset.

    ``--baseline session`` falls back to the participant's own preprocessed
    readings, and so does the automatic fallback when the calibration log yields
    too few readings to estimate from. Participant 001 is in that state: their
    log holds a single reading against a stored baseline of 92.0/2.9652 over
    3472 ticks, because ``_open_calibration_log`` opens with mode ``'w'`` and a
    later aborted run truncated it. That substitution changes what the baseline
    MEANS, from a dedicated measurement to a session average, so it is labelled
    in the report rather than made silently.

    Either way the scale is the sample SD, not the MAD: once the bad readings
    are gone the MAD is a coarse, low-efficiency view of the same dispersion,
    and on ~25 integer readings it lands on a lattice fine enough to have
    produced a 0.74 bpm figure for participant 003 -- an estimator artefact that
    ``hr_delta`` would then divide by.
    """
    cleaned = clean_calibration(
        calib_readings, f0, window=args.window,
        hr_range=(args.hr_min, args.hr_max), mad_k=args.mad_k,
        min_deviation=args.min_deviation, max_gate=args.max_gate,
        fold_margin=args.fold_margin)
    order = (("calibration", cleaned), ("session", repaired))
    if args.baseline == "session":
        order = order[::-1]
    for source, readings in order:
        readings = [v for v in readings if v is not None]
        if len(readings) < args.min_kept:
            continue
        stats = baseline_from_readings(
            readings, min_scale=args.min_scale,
            degenerate_scale=args.degenerate_scale)
        if stats is None:
            continue
        return stats, source
    return None, "none"


def write_calibration_baseline(
    pid: str,
    baseline: Tuple[float, float],
    source: str,
    f0: Optional[float],
    n_kept: int,
    args: argparse.Namespace,
) -> Tuple[Optional[Path], Optional[str]]:
    """Persist the derived bpm baseline into ``calibration_<pid><suffix>.json``.

    Returns ``(path_written, error)``; ``path_written`` is None on a dry run.

    The document is built on the EXISTING ``_preprocessed`` file when there is
    one, so anything an earlier pass put there (the floored ``perclos.std``, in
    particular) survives; only the ``bpm`` entry is replaced. Falling back to
    the original ``calibration_<pid>.json`` keeps this script usable on its own,
    but note that a file rebuilt that way carries the UNFLOORED PERCLOS std.

    Validated with the calibration script's own ``validate`` before writing,
    because a file that fails ``DataCollector._validate_calibration`` is worse
    than no file at all: ``load_calibration`` would reject it at session start
    and silently fall back to running a live 180 s calibration.

    ``bpm.threshold`` is deliberately not carried over. The live schema for bpm
    is ``('mean', 'std')`` -- the 0.55 sitting in that field is the MAR default
    leaking through a shared code path, and it is read by nothing.
    """
    src = args.calib_dir / f"calibration_{pid}{args.calib_suffix}.json"
    if not src.exists():
        src = args.calib_dir / f"calibration_{pid}.json"
    if not src.exists():
        return None, f"no calibration file to build on ({src.name})"
    try:
        cal = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return None, f"{src.name} unreadable ({e})"

    # Provenance records the LIVE baseline, read from the unsuffixed original,
    # not from whatever this script wrote on a previous run -- otherwise a
    # re-run would quietly overwrite the history with its own output.
    original = args.calib_dir / f"calibration_{pid}.json"
    old: Dict[str, Any] = {}
    if original.exists():
        try:
            old = dict(json.loads(original.read_text(encoding="utf-8")).get("bpm", {}))
        except (OSError, json.JSONDecodeError):
            pass
    cal["bpm"] = {"mean": float(baseline[0]), "std": float(baseline[1])}
    cal["hr_preprocessing"] = {
        "script": Path(__file__).name,
        "fundamental_bpm": f0,
        "octave_verified_externally": pid in VERIFIED_FUNDAMENTAL,
        "baseline_source": source,
        "readings_used": n_kept,
        "harmonics_folded": True,
        "original": {k: old.get(k) for k in ("mean", "std")},
    }
    reason = validate_calibration(cal)
    if reason is not None:
        return None, f"would fail validation ({reason})"

    dest = args.calib_dir / f"calibration_{pid}{args.calib_suffix}.json"
    if args.dry_run:
        return None, None
    dest.write_text(json.dumps(cal, indent=4) + "\n", encoding="utf-8")
    return dest, None


# ── per-session driver ────────────────────────────────────────────────────────

def process_session(
    frames: List[Dict[str, Any]],
    pid: Optional[str],
    spans: List[Dict[str, Any]],
    good: List[bool],
    reason: List[Optional[str]],
    fixed: List[Optional[float]],
    methods: List[Optional[str]],
    baseline: Optional[Tuple[float, float]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Rewrite ``heart_rate``/``hr_delta`` in place; return a one-row report.

    Filtering and repair happened earlier, per participant: the baseline is
    derived from the repaired readings, so they have to exist before it does.
    ``f0`` and ``baseline`` are likewise participant-level, not session-level --
    a driver's fundamental rate does not change between two sessions recorded
    the same afternoon, and pooling both sessions plus the calibration log gives
    the octave vote three times the evidence. That also puts both sessions and
    the baseline on one octave by construction, which is what ``hr_delta``
    depends on.
    """
    values = [s["value"] for s in spans]
    n_repaired = 0
    for span, ok, why, value, how in zip(spans, good, reason, fixed, methods):
        if value is None:
            continue
        for f in frames[span["start"]:span["end"]]:
            f["heart_rate"] = value
            f["hr_repaired"] = not ok
            f["hr_repair_reason"] = why
            f["hr_repair_method"] = how
            if baseline is not None:
                f["hr_delta"] = standardized_delta(value, *baseline, clip=args.clip_delta)
        if not ok:
            n_repaired += 1

    # Frames outside every span never had a reading to begin with (rPPG had not
    # locked yet, or the reading went stale). Their heart_rate/hr_delta are left
    # exactly as logged; the provenance keys are still written so that every
    # frame in the output carries the same key set.
    covered = set()
    for span in spans:
        covered.update(range(span["start"], span["end"]))
    for i, f in enumerate(frames):
        if i not in covered:
            f.setdefault("hr_repaired", False)
            f.setdefault("hr_repair_reason", None)
            f.setdefault("hr_repair_method", None)

    live_rejects = sum(1 for s in spans if s["live_rejected"])
    repaired_values = [v for v in fixed if v is not None]
    # The centring check the whole baseline argument rests on: after
    # preprocessing, the median hr_delta over the session should sit near 0. A
    # large value means the baseline and the data are not on the same footing --
    # most likely opposite octaves -- and that the feature has degenerated into
    # a per-driver constant. Computed from what was actually written, so it
    # cannot agree with the code and disagree with the file.
    centring = None
    if baseline is not None and repaired_values:
        centring = float(st.median(
            [standardized_delta(v, *baseline, clip=args.clip_delta)
             for v in repaired_values]))
    return {
        "pid": pid,
        "frames": len(frames),
        "readings": len(spans),
        "session_median": float(st.median(values)) if values else None,
        "repaired_median": float(st.median(repaired_values)) if repaired_values else None,
        "live_rejected": live_rejects,
        "flagged": sum(1 for g in good if not g),
        "repaired": n_repaired,
        "caught_by_offline_only": sum(
            1 for s, g in zip(spans, good) if not g and not s["live_rejected"]),
        "centring": centring,
        "folded": sum(1 for m in methods if m == "folded"),
        "interpolated": sum(1 for m in methods if m in ("interpolated", "carried")),
        "drops": [(values[i], reason[i]) for i in range(len(values)) if not good[i]],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in-data", type=Path, default=REPO_ROOT / "data" / "raw_data.jsonl")
    ap.add_argument("--out-data", type=Path,
                    default=REPO_ROOT / "data" / "preprocessed_data.jsonl")
    ap.add_argument("--calib-dir", type=Path,
                    default=REPO_ROOT / "data" / "calibration_data")
    ap.add_argument("--calib-suffix", default="_preprocessed",
                    help="suffix of the calibration file the derived baseline is "
                         "written to (and built on when it exists). The unsuffixed "
                         "calibration_XXX.json is never modified")
    ap.add_argument("--no-write-calibration", dest="write_calibration",
                    action="store_false",
                    help="derive baselines but do not persist them to the "
                         "calibration files")
    ap.add_argument("--participants", nargs="*", default=None,
                    help="participant ids to process (default: all)")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help="readings on EACH side forming a reading's reference")
    ap.add_argument("--octave", choices=("auto", "none"), default="auto",
                    help="'auto' (default) resolves each session's fundamental by "
                         "harmonic folding before filtering; 'none' skips that and "
                         "leaves only the local drift gate, which cannot handle a "
                         "bimodal session (see session_fundamental)")
    ap.add_argument("--octave-min-support", type=float, default=0.15,
                    help="fraction of readings that must sit in the LOWER mode "
                         "before it is allowed to overturn the dominant one as the "
                         "fundamental. Guards the octave-down preference from "
                         "running away on a single-mode session. Participant 010 is "
                         "the only one in this cohort sensitive to it at all")
    ap.add_argument("--baseline", choices=("calibration", "session"),
                    default="calibration",
                    help="'calibration' (default) derives the baseline from the "
                         "participant's calibration log, filtered and harmonic-"
                         "folded onto their octave; 'session' uses their own "
                         "preprocessed session readings instead")
    ap.add_argument("--session-repair", choices=("fold", "interpolate"),
                    default="fold",
                    help="'fold' (default) halves an identified 2f lock, recovering "
                         "the measurement; 'interpolate' replaces every flagged "
                         "reading with the mean of its nearest good neighbours. "
                         "They agree except where a lock is sustained (see repair)")
    ap.add_argument("--octave-margin", type=float, default=0.10,
                    help="fraction of readings by which the winning fundamental "
                         "must beat the best non-octave rival before the call is "
                         "reported as decided")
    ap.add_argument("--hr-min", type=float, default=HR_RANGE[0])
    ap.add_argument("--hr-max", type=float, default=HR_RANGE[1])
    ap.add_argument("--mad-k", type=float, default=HR_MAD_K)
    ap.add_argument("--min-deviation", type=float, default=HR_MIN_DEVIATION)
    ap.add_argument("--max-gate", type=float, default=HR_MAX_GATE,
                    help="ceiling on the MAD-scaled outlier gate. Without it a "
                         "noisy participant's own scatter widens the gate until it "
                         "rejects nothing (009 reached 70 bpm)")
    ap.add_argument("--fold-margin", type=float, default=HR_FOLD_MARGIN,
                    help="bpm by which halving must beat the raw value's distance "
                         "to the local reference before a gate-flagged reading is "
                         "treated as a 2f lock")
    ap.add_argument("--min-kept", type=int, default=5,
                    help="surviving readings a source needs before it may define "
                         "the baseline")
    ap.add_argument("--min-scale", type=float, default=HR_MIN_SCALE,
                    help="floor on the baseline std. hr_delta DIVIDES by it, so a "
                         "scale that measures the calibration window's noise "
                         "rather than the driver's range inflates the feature "
                         "without adding information (see HR_MIN_SCALE)")
    ap.add_argument("--clip-delta", type=float, default=HR_DELTA_CLIP,
                    help="bound on the emitted hr_delta, both directions. MUST "
                         "match _HR_DELTA_CLIP in data_collector.py, which is "
                         "where the default is read from")
    ap.add_argument("--max-scale", type=float, default=15.0,
                    help="warn when the derived baseline std exceeds this -- a wide "
                         "spread after repair means the rPPG never locked")
    ap.add_argument("--degenerate-scale", type=float, default=5.0,
                    help="scale used when every surviving reading is identical")
    ap.add_argument("--centring-tol", type=float, default=3.0,
                    help="warn when the median hr_delta of a preprocessed session "
                         "is further than this from 0. A modest offset is REAL -- "
                         "the calibration window and the drive are different "
                         "stretches of driving. A large one means the baseline and "
                         "the data are on different octaves or scales")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing")
    ap.add_argument("--verbose", action="store_true",
                    help="list every flagged reading with its reason")
    args = ap.parse_args(argv)

    if not args.in_data.is_file():
        print(f"[hr] no such file: {args.in_data}")
        return 2
    if args.out_data.resolve() == args.in_data.resolve():
        # The input is the only copy of heart_rate_raw; overwriting it in place
        # would destroy the column every re-run depends on.
        print("[hr] --out-data must differ from --in-data")
        return 2

    # Always the UNSUFFIXED originals: this is the live baseline the sessions
    # actually served on, and the thing worth warning about when it disagrees.
    stored = load_stored_baselines(args.calib_dir, "")
    if not stored:
        print(f"[hr][warn] no calibration baselines under {args.calib_dir}")

    frames = [f for _, f in iter_frames(args.in_data)]
    if not frames:
        print(f"[hr] {args.in_data.name} holds no frames")
        return 1

    # Group by (participant, session) while keeping each frame OBJECT in the
    # flat `frames` list, so the output can be written back in input order.
    # Sessions are contiguous in the file today, but nothing guarantees that and
    # a reordered output would silently break anything joining on line number.
    groups: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for f in frames:
        groups.setdefault((f.get("participantid"), f.get("session_id")), []).append(f)
    if args.participants:
        groups = {k: v for k, v in groups.items() if k[0] in args.participants}
    if not groups:
        print("[hr] nothing processed")
        return 1

    # ── pass 1: per-participant octave + baseline ─────────────────────────────
    # Both are participant-level facts, not session-level ones, and deciding
    # them once over ALL of a participant's evidence is what keeps their two
    # sessions and their baseline on a single octave.
    by_pid: Dict[Any, List[Tuple[Tuple[Any, Any], List[Dict[str, Any]]]]] = {}
    for key, session_frames in groups.items():
        by_pid.setdefault(key[0], []).append((key, session_frames))

    spans_by_key: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for key, session_frames in groups.items():
        spans_by_key[key] = reading_spans(session_frames)

    pid_state: Dict[Any, Dict[str, Any]] = {}
    filtered: Dict[Tuple[Any, Any], Tuple[List[bool], List[Optional[str]],
                                          List[Optional[float]],
                                          List[Optional[str]]]] = {}
    for pid, sessions in by_pid.items():
        session_values = [s["value"] for key, _ in sessions for s in spans_by_key[key]]
        calib_readings: List[float] = []
        log_path = args.calib_dir / "calibration_logs" / f"log_calibration_{pid}.csv"
        if log_path.exists():
            calib_readings, _rr = read_calibration_log(log_path)
        # The octave vote pools the calibration log with every session, since
        # all of it is the same driver on the same rig within the same afternoon.
        f0, share, margin = session_fundamental(
            list(calib_readings) + session_values, (args.hr_min, args.hr_max),
            min_support=args.octave_min_support,
            verified=VERIFIED_FUNDAMENTAL.get(str(pid)))
        if args.octave == "none":
            f0 = None

        repaired_all: List[Optional[float]] = []
        for key, _ in sessions:
            vals = [s["value"] for s in spans_by_key[key]]
            good, reason = flag_readings(
                vals, f0, window=args.window, hr_range=(args.hr_min, args.hr_max),
                mad_k=args.mad_k, min_deviation=args.min_deviation,
                max_gate=args.max_gate, fold_margin=args.fold_margin)
            fixed, methods = repair(vals, good, reason, args)
            filtered[key] = (good, reason, fixed, methods)
            repaired_all.extend(fixed)

        baseline, source = derive_baseline(repaired_all, calib_readings, f0, args)
        n_kept = len(clean_calibration(
            calib_readings, f0, window=args.window,
            hr_range=(args.hr_min, args.hr_max), mad_k=args.mad_k,
            min_deviation=args.min_deviation, max_gate=args.max_gate,
            fold_margin=args.fold_margin))
        pid_state[pid] = {
            "f0": f0, "share": share, "margin": margin,
            "baseline": baseline, "baseline_source": source,
            "stored": stored.get(pid or ""),
            "calib_readings": len(calib_readings),
            "calib_kept": n_kept,
            "written": None, "write_error": None,
        }
        if baseline is not None and args.write_calibration and pid is not None:
            dest, err = write_calibration_baseline(
                str(pid), baseline, source, f0,
                n_kept if source == "calibration" else len(repaired_all), args)
            pid_state[pid]["written"] = dest
            pid_state[pid]["write_error"] = err

    # ── pass 2: rewrite each session's frames ─────────────────────────────────
    reports: List[Dict[str, Any]] = []
    for key, session_frames in groups.items():
        stt = pid_state[key[0]]
        good, reason, fixed, methods = filtered[key]
        reports.append(process_session(
            session_frames, key[0], spans_by_key[key], good, reason, fixed,
            methods, stt["baseline"], args))

    if args.dry_run:
        print(f"[hr][dry-run] would write {len(frames)} frames to {args.out_data}")
    elif args.participants:
        # `frames`/`groups` only ever held the OTHER participants' data in
        # memory unfiltered -- writing here would overwrite --out-data's
        # existing (possibly already-repaired) copy of them with untouched
        # data, silently undoing prior repair work for everyone not in
        # --participants. A scoped run reports what it found; it does not
        # write until re-run without --participants over the full cohort.
        print(f"[hr] --participants={sorted(args.participants)} scopes this run to "
              f"{len(groups)} session(s); skipping the write to {args.out_data} so "
              f"the other participants' repaired data isn't overwritten with "
              f"unrepaired frames. Re-run without --participants to write the full "
              f"cohort file.")
    else:
        args.out_data.parent.mkdir(parents=True, exist_ok=True)
        with args.out_data.open("w", encoding="utf-8", newline="\n") as fh:
            for f in frames:
                fh.write(json.dumps(f) + "\n")

    # ── report ────────────────────────────────────────────────────────────────
    per_pid: Dict[Any, List[Dict[str, Any]]] = {}
    for r in reports:
        per_pid.setdefault(r["pid"], []).append(r)

    def _f(v: Any, spec: str = "6.1f") -> str:
        return format(v, spec) if isinstance(v, (int, float)) else f"{'-':>6}"

    print()
    print(f"{'pid':>4} {'sess':>5} {'reads':>6} {'live':>5} {'flag':>5} {'fold':>5} "
          f"{'interp':>6} {'f0':>6} {'med raw':>8} {'med fix':>8} {'baseline':>16} "
          f"{'src':>5} {'centred':>8}")
    print("-" * 112)
    warnings: List[str] = []
    notes: List[str] = []
    for pid in sorted(per_pid, key=lambda p: (p is None, p)):
        rs = per_pid[pid]
        stt = pid_state[pid]
        b = stt["baseline"]
        bs = f"{b[0]:6.1f} +- {b[1]:5.2f}" if b else f"{'(none)':>16}"
        cent = next((r["centring"] for r in rs if r["centring"] is not None), None)
        print(f"{str(pid):>4} {len(rs):>5} {sum(r['readings'] for r in rs):>6} "
              f"{sum(r['live_rejected'] for r in rs):>5} "
              f"{sum(r['flagged'] for r in rs):>5} "
              f"{sum(r['folded'] for r in rs):>5} "
              f"{sum(r['interpolated'] for r in rs):>6} "
              f"{_f(stt['f0'])} "
              f"{_f(next((r['session_median'] for r in rs), None), '8.1f')} "
              f"{_f(next((r['repaired_median'] for r in rs), None), '8.1f')} "
              f"{bs} {stt['baseline_source'][:5]:>5} {_f(cent, '8.2f')}")
        if args.verbose:
            for r in rs:
                for v, why in r["drops"]:
                    print(f"{'':>16}flagged {v:6.1f}  ({why})")

        # ── the three things that must not pass silently ──────────────────────
        if str(pid) in VERIFIED_FUNDAMENTAL:
            notes.append(
                f"{pid}: fundamental fixed at {stt['f0']:.0f} bpm by external "
                f"verification (VERIFIED_FUNDAMENTAL), not by the support vote")
        elif stt["f0"] is not None and stt["margin"] < args.octave_margin:
            warnings.append(
                f"{pid}: octave undecided -- support for the half-rate mode sits "
                f"within {100 * stt['margin']:.0f} points of the "
                f"{100 * args.octave_min_support:.0f}% threshold that chose "
                f"{stt['f0']:.0f} bpm; a nudge to --octave-min-support flips this "
                f"participant to the other octave, so check the raw BVP")
        if b and b[1] > args.max_scale:
            warnings.append(
                f"{pid}: baseline spread is {b[1]:.1f} bpm after repair -- the "
                f"series still is not a coherent pulse, so hr_delta carries little "
                f"for this participant whichever octave is chosen")
        st_b = stt["stored"]
        if st_b and b and stt["f0"] is not None and (
                HARMONIC_LO * stt["f0"] <= st_b[0] <= HARMONIC_HI * stt["f0"]):
            warnings.append(
                f"{pid}: the STORED calibration baseline ({st_b[0]:.1f} bpm) is a 2f "
                f"harmonic of the fundamental found here ({stt['f0']:.0f} bpm). The "
                f"live session served hr_delta against the harmonic; this output "
                f"re-derives it from {stt['baseline_source']} readings "
                f"({b[0]:.1f} bpm) so the channel is centred again")
        if b is None:
            warnings.append(
                f"{pid}: no usable baseline from either the calibration log or the "
                f"session -- hr_delta left exactly as logged for this participant")
        if stt["baseline_source"] == "session":
            warnings.append(
                f"{pid}: calibration log yielded only {stt['calib_kept']} usable "
                f"reading(s), so the baseline comes from their SESSION readings "
                f"instead -- a session average, not a dedicated measurement")
        if stt["write_error"]:
            warnings.append(
                f"{pid}: calibration baseline not written -- {stt['write_error']}")
        for r in rs:
            if r["centring"] is not None and abs(r["centring"]) > args.centring_tol:
                warnings.append(
                    f"{pid}: preprocessed session sits at median hr_delta "
                    f"{r['centring']:+.2f}, beyond +-{args.centring_tol} -- too far "
                    f"to be a real difference between the calibration window and "
                    f"the drive; check the baseline is on the same octave and scale "
                    f"as the session")

    n_read = sum(r["readings"] for r in reports)
    n_flag = sum(r["flagged"] for r in reports)
    n_only = sum(r["caught_by_offline_only"] for r in reports)
    n_live = sum(r["live_rejected"] for r in reports)
    print("-" * 112)
    for n in notes:
        print(f"[hr][note] {n}")
    for w in warnings:
        print(f"[hr][warn] {w}")
    if warnings or notes:
        print("-" * 112)
    n_fold = sum(r["folded"] for r in reports)
    n_interp = sum(r["interpolated"] for r in reports)
    print(f"{len(per_pid)} participants / {len(reports)} sessions | "
          f"{n_flag}/{n_read} readings flagged ({100 * n_flag / max(n_read, 1):.1f}%) | "
          f"live filter caught {n_live}, offline found {n_only} more | "
          f"repaired: {n_fold} folded, {n_interp} interpolated")
    n_written = sum(1 for s in pid_state.values() if s["written"])
    if args.dry_run:
        print("dry run -- no files written")
    else:
        print(f"wrote {len(frames)} frames to {args.out_data}")
        if args.write_calibration:
            print(f"wrote {n_written} calibration baseline(s) to "
                  f"{args.calib_dir}\\calibration_*{args.calib_suffix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
