from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import math, os
import numpy as np
from ProVoice.fcd_config import FCD_NAMES, get_fcd_for_function, resolve_function_key, adjust_fcd_by_state
from ProVoice.models.xlstm_model import (
    encode_and_resample,
    DEFAULT_CONTEXT_LENGTH,
    load_checkpoint,
    log_encoded_frames,
    logits_to_probs,
    probs_to_label,
    secs_of_day as _secs_of_day,
)
import torch


# LoA -> (action, level). One fixed policy: at every level the system asks
# before it acts, and only full autonomy at LoA 4.
#
# There used to be a second, more aggressive mapping selected by a
# `conservative` parameter that every caller passed as True and no CLI flag
# ever set — dead configuration that made the policy look variable when it was
# not. Removed rather than wired up: a study comparing personalization arms
# needs the LoA->action mapping held FIXED across participants and arms, so a
# switch here would be a confound, not a feature.
_LOA_POLICY: Dict[int, Tuple[str, str]] = {
    0: ("none",           "low"),
    1: ("suggest",        "low"),
    2: ("ask_approval",   "medium"),
    3: ("auto_with_veto", "high"),
    4: ("auto",           "high"),
}


def _policy_from_loa(loa: int) -> Tuple[str, str]:
    return _LOA_POLICY[max(0, min(int(loa), 4))]

def _softmax(logits: List[float]) -> List[float]:
    m = max(logits); exps = [math.exp(x - m) for x in logits]; S = sum(exps); return [x / S for x in exps]

_EPS = 1e-12
def _apply_temp_bias_probs(probs: List[float], temperature: float = 1.0, class_bias: Optional[List[float]] = None) -> List[float]:
    logits = [math.log(max(p, _EPS)) for p in probs]
    if class_bias and len(class_bias) == 5:
        logits = [g + b for g, b in zip(logits, class_bias)]
    if abs(temperature - 1.0) > 1e-6:
        logits = [g / temperature for g in logits]
    return _softmax(logits)

def _decide_from_probs(probs: List[float], method: str = "argmax", expected_shift: float = 0.0, quantile_tau: float = 0.65) -> int:
    m = (method or "argmax").lower()
    if m == "expected":
        mu = sum(i * p for i, p in enumerate(probs)) - float(expected_shift or 0.0)
        return max(0, min(4, int(round(mu))))
    if m == "quantile":
        tau = max(0.0, min(1.0, float(quantile_tau or 0.65))); s = 0.0
        for k, p in enumerate(probs):
            s += p
            if s >= tau: return k
        return 4
    return int(max(range(len(probs)), key=lambda i: probs[i]))

# _secs_of_day now lives in xlstm_model (imported above as _secs_of_day): the
# resampler needs the same parser, and xlstm_model is the lower module — this
# one imports it, not the other way round.


def truncate_frames_by_seconds(seq: List[Dict[str, Any]], window_seconds: float) -> List[Dict[str, Any]]:
    """Keep only the trailing frames whose timestamps lie within ``window_seconds``
    of the newest frame.

    The DataCollector's buffer is capped in FRAMES, so its time span depends on
    the achieved sampling rate (measured ~4 Hz, not the nominal 20 Hz) — a
    frame-count cap alone would feed the model ~100 s of history when it was
    trained on 20 s label windows. Truncating by time keeps train and serve
    aligned regardless of the actual rate.

    Frames without a parseable timestamp are kept (their age is unknowable);
    if the NEWEST frame has no parseable timestamp, the sequence is returned
    unchanged (frame-count capping still applies upstream). Handles midnight
    wrap-around, since frame timestamps are time-of-day only.
    """
    if not seq or window_seconds is None or window_seconds <= 0:
        return seq
    t_last = _secs_of_day(seq[-1].get("timestamp"))
    if t_last is None:
        return seq
    kept: List[Dict[str, Any]] = []
    for frame in reversed(seq):
        t = _secs_of_day(frame.get("timestamp"))
        if t is not None:
            age = t_last - t
            if age < -43200.0:      # more than half a day "in the future" → midnight wrap
                age += 86400.0
            elif age < 0:           # small backward clock jitter → treat as same-instant
                age = 0.0
            if age > window_seconds:
                break  # buffer is chronological: everything earlier is older still
        kept.append(frame)
    kept.reverse()
    return kept


def _loa0_result(reason: str, fn_key_or_name: str, fcd: Optional[Dict[str, int]] = None) -> Dict[str, Any]:

    fn_key = resolve_function_key(fn_key_or_name)
    fcd = fcd or get_fcd_for_function(fn_key)
    action, level = _policy_from_loa(0)
    return {
        "action": action, "level": level, "LoA": 0,
        "message": f"fallback LoA0: {reason}",
        "fcd": fcd, "probs": [1.0, 0.0, 0.0, 0.0, 0.0],
        "profile": fn_key, "fallback": True, "fallback_reason": reason
    }


class BaseStrategy(ABC):
    @abstractmethod
    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class XGBoostLoAStrategy(BaseStrategy):
    def __init__(self, model_path: Optional[str], default_function: str,
                 temperature: float = 1.0, class_bias: Optional[List[float]] = None,
                 decision_method: str = "argmax", expected_shift: float = 0.0, quantile_tau: float = 0.65):
        self.model = None
        try:
            import joblib
            if model_path and os.path.exists(model_path):
                self.model = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading FCD model: {e}")
            self.model = None
        self.default_key = resolve_function_key(default_function)
        self.temperature = float(os.getenv("PV_TEMP", str(temperature)))
        self.class_bias = class_bias if class_bias is not None else [0.0]*5
        self.decision_method = (os.getenv("PV_DECISION_METHOD", decision_method)).lower()
        self.expected_shift = float(os.getenv("PV_EXPECTED_SHIFT", str(expected_shift)))
        self.quantile_tau = float(os.getenv("PV_QUANTILE_TAU", str(quantile_tau)))

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            fn = state.get("functionname") or self.default_key
            fcd = adjust_fcd_by_state(get_fcd_for_function(fn))
            if self.model is None:
                return _loa0_result("FCD model unavailable", fn, fcd)
            x = np.array([[fcd[k] for k in FCD_NAMES]], dtype=float)
            if hasattr(self.model, "predict_proba"):
                raw = self.model.predict_proba(x)[0]
                probs = raw[:5].tolist() if len(raw) >= 5 else (raw.tolist() + [0.0]*(5-len(raw)))
            else:
                k = int(self.model.predict(x)[0])
                probs = [0.0]*5; probs[max(0, min(4, k))] = 1.0
            probs_pp = _apply_temp_bias_probs(probs, self.temperature, self.class_bias)
            loa = _decide_from_probs(probs_pp, self.decision_method, self.expected_shift, self.quantile_tau)
            action, level = _policy_from_loa(loa)
            return {"action": action, "level": level, "LoA": loa, "message": "FCD xgboost", "fcd": fcd, "probs": probs_pp, "profile": resolve_function_key(fn), "fallback": False}
        except Exception as e:
            fn = state.get("functionname") or self.default_key
            return _loa0_result(f"FCD decide error: {e}", fn)

_STATE_CAT = ['emotion','lab','environment','secondary_task']
_STATE_NUM = ['drowsiness_alert','gaze_distracted','heart_rate']

class StateLevelsLoAStrategy(BaseStrategy):
    def __init__(self, model_path: Optional[str], default_function: str,
                 prob_threshold: Optional[float] = None, fcd_fallback: Optional[BaseStrategy] = None,
                 temperature: float = 1.0, class_bias: Optional[List[float]] = None,
                 decision_method: str = "argmax", expected_shift: float = 0.0, quantile_tau: float = 0.65):
        self.model = None
        try:
            import joblib
            if model_path and os.path.exists(model_path):
                self.model = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading state model: {e}")
            self.model = None
        self.default_key = resolve_function_key(default_function)
        self.prob_threshold = float(prob_threshold) if prob_threshold is not None else None
        self.fcd_fallback = fcd_fallback
        self.temperature = temperature
        self.class_bias = class_bias if class_bias is not None else [0.0]*5
        self.decision_method = decision_method
        self.expected_shift = expected_shift
        self.quantile_tau = quantile_tau

    @staticmethod
    def _as01(x: Any) -> float:
        if isinstance(x, bool): return 1.0 if x else 0.0
        s = str(x).strip().lower()
        if s in ("true","1","t","yes","y"): return 1.0
        if s in ("false","0","f","no","n","","none","nan","null"): return 0.0
        try: return float(s)
        except Exception:
            return 0.0

    def _extract_features(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        # `or ""` not `.get(k, "")`: emotion is now None when the classifier
        # produced no reading, and str(None) would feed the literal "None" to
        # the categorical encoder as though it were a class it had been
        # trained on. Absent and empty must encode identically here.
        cats = [str(state.get(k) or "") for k in _STATE_CAT]
        has_any = any(cats) or any(k in state for k in _STATE_NUM)
        if not has_any: return None
        fn = state.get("functionname") or self.default_key
        fcd = get_fcd_for_function(fn)
        num = [self._as01(state.get(k)) for k in _STATE_NUM]
        catv = []
        for c in cats:
            catv.extend([0.0 if c else 1.0, min(len(c)/16.0, 1.0)])
        return np.asarray([*(fcd[k] for k in FCD_NAMES), *num, *catv], dtype=float).reshape(1, -1)

    def _try_fcd_fallback(self, state: Dict[str, Any], fn: str, fcd: Dict[str, int]) -> Dict[str, Any]:
        if isinstance(self.fcd_fallback, BaseStrategy):
            out = self.fcd_fallback.decide(state)
            return out if not out.get("fallback") else _loa0_result("State fallback->FCD fallback", fn, fcd)
        return _loa0_result("State fallback no FCD provided", fn, fcd)

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            fn = state.get("functionname") or self.default_key
            fcd = get_fcd_for_function(fn)
            if self.model is None:
                return self._try_fcd_fallback(state, fn, fcd)
            X = self._extract_features(state)
            if X is None:
                return self._try_fcd_fallback(state, fn, fcd)
            proba = self.model.predict_proba(X)
            if isinstance(proba, list):
                probs = [float(p[:,1][0]) if p.shape[1]>1 else float(p[:,-1][0]) for p in proba]
            else:
                probs = [float(v) for v in proba[0]]
            if self.prob_threshold is not None and max(probs) < self.prob_threshold:
                return self._try_fcd_fallback(state, fn, fcd)
            probs_pp = _apply_temp_bias_probs(probs, self.temperature, self.class_bias)
            loa = _decide_from_probs(probs_pp, self.decision_method, self.expected_shift, self.quantile_tau)
            action, level = _policy_from_loa(loa)
            return {"action": action, "level": level, "LoA": loa, "message": "State model", "fcd": fcd, "probs": probs_pp, "profile": resolve_function_key(fn), "fallback": False}
        except Exception as e:
            print(f"Error in state model: {e}")
            fn = state.get("functionname") or self.default_key
            fcd = get_fcd_for_function(fn)
            return self._try_fcd_fallback(state, fn, fcd)


class StateXLSTMLoAStrategy(BaseStrategy):
    def __init__(self, model_path: Optional[str], default_function: str, window: int = 256,
                 fcd_fallback: Optional[BaseStrategy] = None, log_path: Optional[str] = None,
                 window_seconds: Optional[float] = None,
                 participantid: Optional[str] = None):
        # FIRST statement: __del__ runs on any partially-constructed instance,
        # and it reads this attribute. `load_checkpoint` below can raise (a
        # corrupt or arch-incompatible .pt), and while that is caught here, an
        # uncaught failure anywhere in __init__ would otherwise leave __del__
        # raising AttributeError during garbage collection — where exceptions
        # are printed and swallowed, masking the real error with a confusing
        # second one.
        self._log_fh = None
        self.model = None
        # `window` is only a fallback; the authoritative sequence length is
        # `context_length` taken from the loaded checkpoint's arch.
        self.context_length = int(window) if window else DEFAULT_CONTEXT_LENGTH
        self.ok = False
        self.head_type = "softmax"
        arch: Dict[str, Any] = {}
        try:
            if model_path and os.path.exists(model_path):
                self.model, arch = load_checkpoint(model_path)
                # Live-study checkpoints (minted by build_study_checkpoints.py)
                # carry provenance in arch['study'], including which driver
                # they were held out for. Serving one to the wrong driver is
                # silent and unrecoverable once the session has run, so this
                # must be a hard refusal, not a warning -- see CLAUDE.md
                # "Checkpoint naming and provenance". Checkpoints with no
                # 'study' key (population / plain LODO checkpoints) carry no
                # such claim and are unaffected by this check.
                study_meta = arch.get("study")
                if study_meta:
                    held_out = str(study_meta.get("held_out_pid"))
                    pid_str = str(participantid) if participantid else ""
                    if not pid_str or held_out != pid_str:
                        raise RuntimeError(
                            f"REFUSING to serve {model_path!r}: this checkpoint "
                            f"was held out for participant {held_out!r}, but this "
                            f"session's participantid is {pid_str!r}. Serving it "
                            f"anyway would silently give this driver another "
                            f"driver's personalized head."
                        )
                self.context_length = int(arch.get("context_length", self.context_length))
                # 'corn' checkpoints emit K-1 conditional logits; logits_to_probs
                # turns either head type into a 5-class PMF. Old checkpoints
                # lack the key -> softmax.
                self.head_type = str(arch.get("head_type", "softmax"))
                # State what was actually loaded. The head's WIDTH is the only
                # thing that distinguishes an FCD-augmented checkpoint from a
                # plain one -- forward() switches on it, no flag is stored, and
                # both load without complaint. So a narrow checkpoint served by
                # mistake in place of an --embed-fcd one is silent: it runs, it
                # predicts, and nothing says the task block is missing. One line
                # at load time is what makes that visible in a session log.
                width = self.model.head.in_features
                print(f"[xlstm] loaded {os.path.basename(model_path)}: "
                      f"head {width}-dim "
                      f"({'z+FCD, AUGMENTED' if self.model.head_uses_fcd() else 'z only, plain'}), "
                      f"embedding_dim={self.model.embedding_dim}, "
                      f"head_type={self.head_type}, "
                      f"context_length={self.context_length}")
                self.ok = True
        except Exception as e:
            print(f"Error loading state model: {e}")
            self.model = None
            self.ok = False
        self.window = int(window)
        # Time-based cap on the input window (seconds), matching how segments
        # were cut at training. Precedence: explicit arg > checkpoint arch >
        # 20 s (label-window default, for legacy checkpoints without the key).
        # <=0 disables the cap, leaving only the frame-count cap (context_length).
        if window_seconds is not None:
            ws = float(window_seconds)
        elif "window_seconds" in arch:
            ws = arch["window_seconds"]  # may be None = trained with cap disabled
        else:
            ws = 20.0
        self.window_seconds = float(ws) if ws and float(ws) > 0 else None
        # Fixed-grid resampling rate. Taken ONLY from the checkpoint — unlike
        # window_seconds there is no sane operator override, because serving on a
        # different grid than the model was trained on is precisely the failure
        # this is here to prevent. Legacy checkpoints lack the key -> None -> the
        # raw frames are used, which is the behaviour they were trained with.
        _rs = arch.get("resample_hz")
        self.resample_hz = float(_rs) if _rs and float(_rs) > 0 else None
        self.default_key = resolve_function_key(default_function)
        self.fcd_fallback = fcd_fallback
        try:
            self._log_fh = open(log_path, "a", encoding="utf-8") if log_path else None
        except Exception as e:
            print(f"Error opening log file: {e}")
            self._log_fh = None

    def close(self) -> None:
        # getattr, not self._log_fh: belt-and-braces for the same reason the
        # attribute is initialised first — close() must be safe to call on an
        # instance whose __init__ did not finish.
        fh = getattr(self, "_log_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            self._log_fh = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass   # never raise from a finalizer

    def _try_fcd_fallback(self, state: Dict[str, Any], fn: str) -> Dict[str, Any]:
        fcd = get_fcd_for_function(fn)
        if isinstance(self.fcd_fallback, BaseStrategy):
            out = self.fcd_fallback.decide(state)
            return out if not out.get("fallback") else _loa0_result("XLSTM fallback->FCD fallback", fn, fcd)
        return _loa0_result("XLSTM fallback no FCD provided", fn, fcd)

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            fn = state.get("functionname") or self.default_key
            if not self.ok:
                return self._try_fcd_fallback(state, fn)
            seq: List[Dict[str, Any]] = state.get("sequence") or []
            if not seq:
                return self._try_fcd_fallback(state, fn)
            # Truncate by TIME first (match the 20 s label-window semantics the
            # model was trained on), then resample onto the checkpoint's fixed
            # grid so the step count matches training regardless of the rate this
            # session achieved, then cap by frame count as a safety bound.
            # No padding at inference: batch size is 1, the model is causal, and
            # forward() without `lengths` reads the last timestep — which the
            # resampler anchors on the most recent real frame. Short
            # early-session windows are fine, they just yield fewer steps.
            if self.window_seconds:
                seq = truncate_frames_by_seconds(seq, self.window_seconds)
            X = encode_and_resample(
                seq,
                resample_hz=self.resample_hz,
                window_seconds=self.window_seconds,
                functionname=fn,
            )[-self.context_length:]
            if self._log_fh is not None:
                # Log only the last frame (most recent driver state) to keep
                # the file manageable at 20 Hz.
                log_encoded_frames(
                    self._log_fh, "infer",
                    state.get("timestamp", ""),
                    X[-1:],
                )
            with torch.no_grad():
                xb = torch.from_numpy(X)[None, ...]
                logits = self.model(xb)
                pmf = logits_to_probs(logits, self.head_type)
                # Decode with the head's canonical rule — for CORN that is Shi
                # et al.'s rank rule sum_k 1[q_k > 0.5] (the PMF's median), not
                # argmax (its mode). Same function the trainers and the sweep
                # use, so the LoA served is the LoA those curves measured.
                loa = int(probs_to_label(pmf, self.head_type)[0])
                probs = pmf.cpu().numpy()[0].tolist()
            action, level = _policy_from_loa(loa)
            fcd = get_fcd_for_function(fn)
            return {"action": action, "level": level, "LoA": loa, "message": "State XLSTM", "fcd": fcd, "probs": probs, "profile": resolve_function_key(fn), "fallback": False}
        except Exception as e:
            print(f"Error in state model: {e}")
            fn = state.get("functionname") or self.default_key
            return self._try_fcd_fallback(state, fn)


class CombinedFusionStrategy(BaseStrategy):
    def __init__(self, fcd_strategy: BaseStrategy, state_strategy: BaseStrategy, w_fcd: float = 0.7,
                 decision_method: str = "argmax", expected_shift: float = 0.0, quantile_tau: float = 0.65):
        self.fcd_strategy = fcd_strategy
        self.state_strategy = state_strategy
        self.w_fcd = float(w_fcd); self.w_state = 1.0 - float(w_fcd)
        self.decision_method = decision_method
        self.expected_shift = expected_shift
        self.quantile_tau = quantile_tau

    def _warn_degraded(self, reason):
        """Say it ONCE per distinct reason, on the console, as it happens.

        decisions.csv is read after the session; a block that ran without the
        state model cannot be recovered by then. At 4 Hz an unthrottled warning
        would also bury everything else in the log.
        """
        if reason in self._degraded_seen:
            return
        self._degraded_seen.add(reason)
        print("[decision] DEGRADED: %s. Every decision from here is missing "
              "half the model." % reason)

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        fn = state.get("functionname", "")
        try:
            of = self.fcd_strategy.decide(state)
        except Exception as e:
            of = _loa0_result(f"Fusion FCD error: {e}", fn)
        if not hasattr(self, "_degraded_seen"):
            self._degraded_seen = set()
        try:
            os_ = self.state_strategy.decide(state)
        except Exception as e:
            os_ = _loa0_result(f"Fusion State error: {e}", fn)
        of_fb = bool(of.get("fallback")); os_fb = bool(os_.get("fallback"))
        # Degrade gracefully: if only one model is unavailable, use the other
        # instead of collapsing the whole decision to LoA0. (Common case: no
        # trained xLSTM checkpoint -> fall back to the FCD model alone.)
        if of_fb and os_fb:
            return _loa0_result("Fusion: both FCD and state unavailable", fn)
        # A HALF-DEAD FUSION IS STILL A FALLBACK, and must say so in the
        # top-level columns.
        #
        # These two branches used to spread `of` / `os_` unchanged, and those
        # carry "fallback": False -- so a run in which the state model fell back
        # on EVERY tick wrote fallback=False, fallback_reason="" to all of
        # decisions.csv, with the truth buried in the nested `sub` JSON. That is
        # how a session that served zero model predictions looked clean: seen
        # 2026-08-21, where a missing state_xlstm.pt left the FCD XGBoost
        # deciding alone on a static per-function vector, and the served LoA was
        # therefore constant for the whole run.
        #
        # `fallback` here means "this decision is not what the configuration
        # asked for", not "LoA 0" -- the LoA is still the surviving model's, and
        # `sub` still holds both halves.
        if os_fb:
            reason = ("state unavailable (%s); FCD-only"
                      % (os_.get("fallback_reason") or "no reason given"))
            self._warn_degraded(reason)
            return {**of,
                    "message": (str(of.get("message", "")) + " (state unavailable; FCD-only)").strip(),
                    "fallback": True, "fallback_reason": reason,
                    "sub": {"fcd": of, "state": os_}}
        if of_fb:
            reason = ("FCD unavailable (%s); state-only"
                      % (of.get("fallback_reason") or "no reason given"))
            self._warn_degraded(reason)
            return {**os_,
                    "message": (str(os_.get("message", "")) + " (FCD unavailable; state-only)").strip(),
                    "fallback": True, "fallback_reason": reason,
                    "sub": {"fcd": of, "state": os_}}
        pf = of.get("probs", [0.2]*5); ps = os_.get("probs", [0.2]*5)
        probs = [self.w_fcd*pf[i] + self.w_state*ps[i] for i in range(5)]
        S = sum(probs); probs = [p / S if S>0 else 0.2 for p in probs]
        loa = _decide_from_probs(probs, self.decision_method, self.expected_shift, self.quantile_tau)
        action, level = _policy_from_loa(loa)
        return {"action": action, "level": level, "LoA": loa, "message": f"fusion w_fcd={self.w_fcd:.2f}", "fcd": of.get("fcd"), "probs": probs, "sub": {"fcd": of, "state": os_}, "fallback": False}
