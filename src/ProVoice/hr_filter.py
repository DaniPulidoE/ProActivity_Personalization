"""The heart-rate cleaning algorithm -- ONE definition, used live and offline.

The rPPG estimator (MMRPhys) takes the largest FFT peak in 36-198 bpm. A real
pulse waveform carries genuine energy at twice its rate, so for any driver under
99 bpm that 2f peak is also inside the search band and the estimator sometimes
locks onto it -- reporting ~140 for a true ~70. It can also lose the pulse and
settle on a slower rhythm. This module decides which readings are wrong and what
their value should have been.

**Three callers, and why they are not all equivalent.**

1. ``data_preprocessing/heart_rate_preprocessing.py`` -- rebuilds the whole
   dataset offline. Has every session at once, so it can resolve the octave over
   all of a participant's evidence and interpolate a hole from BOTH sides.
2. ``DataCollector.compute_calibration`` -- computes the stored baseline at the
   END of the calibration phase. This is a BATCH operation: it runs once, with
   all ~28 calibration readings already in hand, so it can run this module's
   filter in full. It is therefore equivalent to (1) restricted to the
   calibration readings -- the only thing it lacks is the session readings that
   (1) additionally pools into the octave vote.
3. ``DataCollector._capture_loop`` -- per-reading, during a session. STRICTLY
   CAUSAL: it sees only the past, so it cannot resolve an octave or interpolate.
   It uses ``_HR_HARMONIC_LO/HI`` against a running median and folds; the rest of
   this module does not apply to it.

Keeping the constants and the functions here rather than duplicating them is
what stops (1) and (2) from drifting apart. Before this module existed the
offline script AST-parsed ``data_collector.py`` to read the constants, which
kept the *numbers* in sync but not the *algorithm*: the live baseline aggregated
raw readings while the offline one aggregated filtered ones, and participant
010's stored baseline came out at 111.5 bpm -- the 2f harmonic of a pulse their
own smartwatch puts near 55.

This module deliberately imports nothing heavier than the standard library, so
importing it from a trainer, a sweep or an offline analysis costs milliseconds
and pulls in no torch, no cv2 and no fastapi.
"""
from __future__ import annotations

import math
import statistics as st
from typing import Dict, List, Optional, Sequence, Tuple

# -- constants shared by the live and offline paths ---------------------------

# MAD -> sigma for Gaussian data (1 / Phi^-1(0.75)). Without it the MAD reports
# ~0.6745*sigma and every scale would be ~33% too small.
_MAD_TO_SIGMA = 1.4826

# Floor on the dispersion stored for bpm/rr. Below it the estimate is measuring
# estimator noise over a 180 s window rather than the driver, and hr_delta
# DIVIDES by it. A driver who genuinely varies more keeps their own scale.
_ROBUST_SCALE_FLOOR = {"bpm": 5.0, "rr": 2.0}

# Bound on the z-scores _standardized_delta emits (hr_delta, rr_delta), both
# directions. Unclipped, hr_delta reached +-12 on the one STATE_NUM column that
# is not already inside [0,1].
_HR_DELTA_CLIP = 5.0

# Band, as a multiple of the reference, in which a reading is treated as the 2nd
# harmonic. +-15% around 2x: the toolbox FFT resolution is 3 bpm, so a true 2f
# peak always lands well inside.
_HR_HARMONIC_LO = 1.7
_HR_HARMONIC_HI = 2.3

# Physiologically defensible range for a seated adult. Outside it the estimator
# is not measuring a pulse, whatever it reported.
_HR_RANGE = (40.0, 180.0)

# Accepted readings needed before a reference is trusted enough to reject.
_HR_REF_MIN = 3

# Consecutive folds after which the live filter stops trusting its own reference
# and re-seeds from the raw value. Used only by the causal path (caller 3).
_HR_REJECT_RUN_MAX = 5

# Public aliases -- the offline script reads these names.
MAD_TO_SIGMA: float = _MAD_TO_SIGMA
HARMONIC_LO: float = _HR_HARMONIC_LO
HARMONIC_HI: float = _HR_HARMONIC_HI
HR_REF_MIN: int = _HR_REF_MIN
HR_DELTA_CLIP: float = _HR_DELTA_CLIP
HR_RANGE: Tuple[float, float] = _HR_RANGE
HR_MIN_SCALE: float = _ROBUST_SCALE_FLOOR["bpm"]

# Absolute floor on the outlier gate. Without it, a driver whose readings are
# nearly constant gets a MAD near zero and every small wobble is "an outlier".
HR_MIN_DEVIATION = 15.0
# Deviation from the reference, in MADs, beyond which a reading is dropped.
HR_MAD_K = 3.5
# CEILING on that MAD-scaled gate. Uncapped the rule is self-defeating: it is
# widened by the very contamination it exists to reject -- participant 009's
# local MAD of 13.5 inflated it to 70 bpm, admitting everything from 20 to 150
# around a reference of 81.
HR_MAX_GATE = 25.0
# A gate-flagged reading is treated as a 2f lock when halving brings it at least
# this many bpm CLOSER to the local reference.
HR_FOLD_MARGIN = 5.0

# Readings on each side that form a reading's reference window. Readings land
# every ~6 s, so +-8 spans ~1.5 min -- long enough to average out estimator
# jitter, short enough to follow a real trend.
DEFAULT_WINDOW = 8

# Octave calls confirmed against evidence OUTSIDE this pipeline. Recorded rather
# than left to the support threshold, because a tuning parameter that happens to
# land on the right answer is not the same as knowing the answer.
#
#   010 -- the low mode is the true pulse, confirmed by the participant's own
#          smartwatch. Also the ONLY participant in the study cohort whose
#          octave depends on the support threshold at all.
VERIFIED_FUNDAMENTAL: Dict[str, str] = {"010": "low"}


def session_fundamental(
    values: Sequence[float],
    hr_range: Tuple[float, float] = HR_RANGE,
    rel_tol: float = 0.25,
    min_support: float = 0.15,
    verified: Optional[str] = None,
) -> Tuple[Optional[float], float, float]:
    """Pick the session's fundamental rate; return ``(f0, share, margin)``.

    **Why this cannot be left to a local median.** Sessions 010 and 012 are
    BIMODAL -- a cluster near 55-65 and another near 110-130 -- and in 012's
    first session the split is 53/55, so the session median (86) lands in the
    empty gap between the modes and is not a rate the driver ever had. A local
    window inside the high cluster then has a harmonic majority, makes the
    harmonic its reference, and flags the CORRECT low readings as subharmonics.
    Before the octave is settled, no drift-tracking reference can be trusted.

    **The tie is broken by physics, not preference.** The toolbox takes the
    largest FFT peak in 36-198 bpm. A real BVP carries genuine energy at 2f, so
    a true rate below 99 bpm can be reported DOUBLED -- but nothing in that
    pipeline can report a rate HALVED, since there is no subharmonic to lock on
    to. Given two modes at x and ~2x, the fundamental is therefore x.

    Mechanically this is harmonic folding, the standard octave-error fix in
    pitch tracking: each candidate f0 is scored by how many readings it explains
    either as itself or as its 2f, and the best-scoring candidate wins with ties
    broken DOWNWARD (the octave-down convention, which is the same physical
    argument).

    **The octave-down preference needs a floor, or it runs away.** Halving ANY
    candidate explains the same readings just as well, purely as 2f: participant
    001's readings are a single tight mode at ~88 bpm, and an unconstrained
    search happily returns 44 -- a rate the driver never had, explaining 100% of
    the session as harmonics of nothing. A candidate is therefore only eligible
    if its own fundamental band holds at least ``min_support`` of the readings.
    The low mode has to EXIST before it can win.

    ``share`` is the fraction of readings explained; ``margin`` is the lead over
    the best candidate that is not an octave-relative of the winner. A small
    margin means the session did not actually decide, and the caller warns.
    """
    lo, hi = hr_range
    vals = [v for v in values if math.isfinite(v) and lo <= v <= hi]
    if len(vals) < HR_REF_MIN:
        return None, 0.0, 0.0

    def at(centre: float, v: float) -> bool:
        return abs(v - centre) <= rel_tol * centre

    def mode_at(centre: float) -> Tuple[int, float]:
        """Population and refined centre of the mode around ``centre``."""
        members = [v for v in vals if at(centre, v)]
        return len(members), (float(st.median(members)) if members else centre)

    # Step 1: the dominant mode -- the best-supported rate in the session.
    # Refined to the MEDIAN of its own members, not left at whichever reading
    # seeded it: with a +-25% tolerance many seeds cover the same mode, and
    # taking the smallest of them would put f0 at the mode's lower edge (it put
    # participant 003 at 84 bpm for a mode centred on 98).
    n_dom, dominant = max((mode_at(c)[0], c) for c in vals)
    _, dominant = mode_at(dominant)

    # Step 2: the octave question, and ONLY the octave question. Is there also a
    # mode near half the dominant rate? If so it is the fundamental and the
    # dominant mode is its 2f, because the estimator can double a rate but has
    # no mechanism to halve one. The support floor is what stops this from
    # running away: halving ANY rate explains the same readings just as well as
    # harmonics of nothing, so participant 001's single tight mode at 88 would
    # otherwise return 44. The low mode has to EXIST before it can win.
    n_half, half = mode_at(dominant / 2.0)
    if verified == "low" and half >= lo:
        f0, n_f0 = half, n_half          # external ground truth; no vote needed
    elif verified == "high":
        f0, n_f0 = dominant, n_dom
    elif half >= lo and n_half >= min_support * len(vals) and n_half > 0:
        f0, n_f0 = half, n_half
    else:
        f0, n_f0 = dominant, n_dom

    # How decisive the octave call was = how far the low mode's support sits
    # from the threshold that decided it. Comparing the two hypotheses' total
    # explanatory power instead would call every session undecided, since both
    # explain the SAME readings -- one as fundamentals, the other as 2f. The
    # support level is the quantity the decision actually turned on.
    margin = abs(n_half / len(vals) - min_support)
    return f0, n_f0 / len(vals), margin


def _local_reference(
    values: Sequence[float], good: Sequence[bool], i: int, window: int,
) -> Tuple[Optional[float], float]:
    """Median and MAD of the good readings around ``i``, excluding ``i`` itself.

    Excluding the candidate is what makes the test a test: included, a lone
    harmonic contributes to the very reference it is being compared against, and
    at small windows that is enough to pull the median toward it.
    """
    lo, hi = max(0, i - window), min(len(values), i + window + 1)
    neigh = [values[j] for j in range(lo, hi) if j != i and good[j]]
    if len(neigh) < HR_REF_MIN:
        return None, 0.0
    ref = float(st.median(neigh))
    mad = float(st.median([abs(v - ref) for v in neigh]))
    return ref, mad


def flag_readings(
    values: Sequence[float],
    f0: Optional[float],
    window: int = DEFAULT_WINDOW,
    hr_range: Tuple[float, float] = HR_RANGE,
    mad_k: float = HR_MAD_K,
    min_deviation: float = HR_MIN_DEVIATION,
    max_gate: float = HR_MAX_GATE,
    fold_margin: float = HR_FOLD_MARGIN,
    max_iter: int = 10,
) -> Tuple[List[bool], List[Optional[str]]]:
    """Return ``(good, reason)`` per reading -- index-aligned with ``values``.

    Index alignment is the point: a baseline only needs a filtered LIST of
    readings, but repairing a time series needs to know *where* each hole is so
    the neighbours either side of it can be found.

    Two stages, in this order for a reason:

    1. **Octave, decided once per session against ``f0``.** Anything in
       ``[1.7*f0, 2.3*f0]`` is a 2f lock and anything above ``2.3*f0`` is not a
       rate this driver had. Settling this globally is what stops a harmonic
       burst from capturing a local window (see ``session_fundamental``).
    2. **Drift, decided locally among the survivors.** A driver's pulse really
       does move over a 30-minute drive (007 spans 58-100), so the remaining
       spikes are judged against the median of their ``window`` neighbours, not
       against ``f0``. Only stage-1 survivors form that reference.

       The gate is CAPPED at ``max_gate``. Uncapped it is self-defeating: it is
       widened by the very contamination it exists to reject, and 009's local MAD
       of 13.5 inflated it to 70 bpm, admitting everything from 20 to 150 around
       a reference of 81. A reading that fails the gate is then re-labelled a 2f
       lock when HALVING brings it closer to the local reference. That is how
       harmonics of the low end of a drifting driver's range are caught: stage 1
       cannot see them, because its band is anchored on a single participant-wide
       ``f0``, so 009's 130/132/137 (2f of 65/66/68, rates they demonstrably
       have) fall below the [138, 186] band that f0=81 implies.

    There is deliberately NO subharmonic rule, unlike the calibration pass. For
    a location estimate, discarding a low outlier costs nothing; for a time
    series it was the rule that inverted -- in 012's bimodal session it flagged
    the true 56-60 bpm readings and repaired them UP to ~110-135, laundering the
    harmonic into the output. With the octave settled the rule is also
    unnecessary: a genuine low reading is now simply a stage-2 spike.

    Iterative for the same reason as the calibration pass: removing a spike
    moves the local medians, which can expose readings the first pass hid. Flags
    are only ever ADDED, so the good-set shrinks monotonically and the loop
    terminates; ``max_iter`` only bounds pathological input.
    """
    n = len(values)
    good = [True] * n
    reason: List[Optional[str]] = [None] * n

    lo, hi = hr_range
    for i, v in enumerate(values):
        if not math.isfinite(v):
            good[i], reason[i] = False, "not finite"
        elif not (lo <= v <= hi):
            good[i], reason[i] = False, f"outside {lo:.0f}-{hi:.0f} bpm"
        elif f0 is not None and HARMONIC_LO * f0 <= v <= HARMONIC_HI * f0:
            good[i], reason[i] = False, f"2f harmonic of session {f0:.0f}"
        elif f0 is not None and v > HARMONIC_HI * f0:
            good[i], reason[i] = False, f"{v:.0f} > 2.3x session {f0:.0f}"

    for _ in range(max_iter):
        newly: List[Tuple[int, str]] = []
        for i, v in enumerate(values):
            if not good[i]:
                continue
            ref, mad = _local_reference(values, good, i, window)
            if ref is None or ref <= 0.0:
                continue                 # too few neighbours to judge against
            gate = min(max(min_deviation, mad_k * MAD_TO_SIGMA * mad), max_gate)
            if abs(v - ref) > gate:
                # Does HALVING explain it? The stage-1 band is anchored on one
                # participant-level f0, but a driver's pulse drifts, so a 2f lock
                # on the LOW end of their range falls under that band: 009's f0
                # is 81, giving a band of [138, 186], while their 130/132/137
                # readings are 2f of 65/66/68 -- rates they demonstrably have.
                # Comparing against the LOCAL reference catches those, and lets
                # them be folded (recovering the measurement) rather than
                # interpolated (discarding it).
                half = v / 2.0
                if half >= lo and abs(half - ref) < abs(v - ref) - fold_margin:
                    newly.append((i, f"2f harmonic of local {ref:.0f}"))
                else:
                    newly.append((i, f"|{v:.0f}-{ref:.0f}| > {gate:.0f}"))
        if not newly:
            break
        for i, why in newly:
            good[i], reason[i] = False, why
    return good, reason


def clean_calibration(
    readings: Sequence[float],
    f0: Optional[float],
    window: int = DEFAULT_WINDOW,
    hr_range: Tuple[float, float] = HR_RANGE,
    mad_k: float = HR_MAD_K,
    min_deviation: float = HR_MIN_DEVIATION,
    max_gate: float = HR_MAX_GATE,
    fold_margin: float = HR_FOLD_MARGIN,
) -> List[float]:
    """Filter the calibration readings onto ``f0``'s octave, FOLDING harmonics.

    Unlike the session pass, a reading identified as a 2f lock is HALVED rather
    than discarded. The two cases differ in what is available to replace a bad
    reading with: a session has ~200 readings and good neighbours either side of
    any hole, so dropping and interpolating loses nothing. A calibration has
    ~28 readings and no neighbours worth borrowing from, so dropping is
    expensive -- and for participant 010 it is fatal. Their calibration is
    almost entirely 2f, and discarding it leaves TWO usable readings (47 and
    49), below any threshold at which a baseline means anything.

    Halving is only safe because the octave is settled FIRST, per participant,
    over all of their evidence at once -- and for 010 by external measurement.
    Halve before that is decided and a mis-diagnosed fold injects a wrong value
    at full confidence, which is why the session pass still drops rather than
    folds. Measured effect: 010 goes from 2 usable readings to 18, at
    54.5 +- 3.05 bpm; eight participants are unchanged to the decimal; 009 and
    012 improve slightly.

    A second pass then catches whatever the fold did not explain, using the
    local drift gate only -- the octave question is already answered.
    """
    vals = [float(v) for v in readings if v is not None]
    if not vals:
        return []
    good, why = flag_readings(
        vals, f0, window=window, hr_range=hr_range,
        mad_k=mad_k, min_deviation=min_deviation,
        max_gate=max_gate, fold_margin=fold_margin)
    folded: List[float] = []
    for v, ok, reason in zip(vals, good, why):
        if ok:
            folded.append(v)
        elif reason and reason.startswith("2f harmonic"):
            folded.append(v / 2.0)
    if len(folded) < 2:
        return folded
    good2, _ = flag_readings(
        folded, None, window=window, hr_range=hr_range,
        mad_k=mad_k, min_deviation=min_deviation,
        max_gate=max_gate, fold_margin=fold_margin)
    return [v for v, ok in zip(folded, good2) if ok]


def standardized_delta(value: float, mean: float, std: float,
                       clip: float = HR_DELTA_CLIP) -> float:
    """``DataCollector._standardized_delta``, reproduced for the bpm channel.

    Same degenerate-case collapse to a unit scale, the same clip, and the same
    rounding to one decimal -- ``hr_delta`` is a model input, so what this writes
    must be bit-for-bit what the serving path would emit for the same reading.
    The clip bound itself is AST-read from ``data_collector.py`` (see
    ``HR_DELTA_CLIP``) so the two cannot be changed independently.
    """
    baseline = mean if mean > 0.0 else 0.0
    scale = std if std > 0.0 else 1.0
    z = (value - baseline) / scale
    return round(max(-clip, min(clip, z)), 1)


def baseline_from_readings(
    readings: Sequence[float],
    min_scale: float = HR_MIN_SCALE,
    degenerate_scale: float = 5.0,
) -> Optional[Tuple[float, float]]:
    """``(location, scale)`` for a bpm baseline, or None when there is nothing.

    Median for location, population SD for scale, floored at ``min_scale``.

    The SD rather than the MAD, deliberately, and ONLY because the readings
    handed in here have already been through ``clean_calibration``. Measured on
    the study cohort over 4,000 bootstrap resamples per participant:

      * on RAW readings the SD is catastrophic -- 15.5, 15.2, 22.1 and 25.0 bpm
        for participants 007, 009, 012 and 010, because the harmonics define it.
        That is why the CAUSAL path, which cannot clean, must not use it.
      * on CLEANED readings the SD is the steadier estimate for 11 of 11
        participants, by 2-4x in coefficient of variation. The readings are
        integer-quantised upstream by mmrphys (``hr = np.round(hr, 0)``), so the
        MAD can only land on a half-integer lattice; resampled, participant
        003's MAD flips between 0.5 and 1.0 for a CV of 0.95. It also assigns
        the identical 2.9652 to six different drivers whose SDs run 2.27-4.22.
      * after cleaning, the tails the MAD guards against are gone: excess
        kurtosis is NEGATIVE for 7 of 11 participants.

    So the estimator choice is not a style question -- it follows from whether
    the caller has cleaned its input. Clean first, then SD.
    """
    vals = [float(v) for v in readings if v is not None]
    if not vals:
        return None
    mean = float(st.median(vals))
    std = float(st.pstdev(vals))
    if std <= 0.0:
        std = degenerate_scale
    return mean, max(std, min_scale)
