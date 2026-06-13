from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import math, os
import numpy as np
from ProVoice.fcd_config import FCD_NAMES, get_fcd_for_function, resolve_function_key, adjust_fcd_by_state
from ProVoice.models.xlstm_model import (
    encode_frame,
    D_IN,
    DEFAULT_CONTEXT_LENGTH,
    load_checkpoint,
)

def _policy_from_loa(loa: int, conservative: bool = True) -> Tuple[str, str]:
    loa = max(0, min(int(loa), 4))
    if conservative:
        mapping = {0: ("none", "low"), 1: ("suggest", "low"), 2: ("ask_approval", "medium"), 3: ("auto_with_veto", "high"), 4: ("auto", "high")}
    else:
        mapping = {0: ("none", "low"), 1: ("suggest", "medium"), 2: ("auto_with_veto", "medium"), 3: ("auto", "high"), 4: ("auto", "high")}
    return mapping[loa]

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

def _loa0_result(reason: str, fn_key_or_name: str, fcd: Optional[Dict[str, int]] = None) -> Dict[str, Any]:

    fn_key = resolve_function_key(fn_key_or_name)
    fcd = fcd or get_fcd_for_function(fn_key)
    action, level = _policy_from_loa(0, conservative=True)
    return {
        "action": action, "level": level, "LoA": 0,
        "message": f"fallback LoA0: {reason}",
        "fcd": fcd, "probs": [1.0, 0.0, 0.0, 0.0, 0.0],
        "profile": fn_key, "fallback": True, "fallback_reason": reason
    }


class BaseStrategy:
    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class XGBoostLoAStrategy(BaseStrategy):
    def __init__(self, model_path: Optional[str], default_function: str, conservative: bool = True,
                 temperature: float = 1.0, class_bias: Optional[List[float]] = None,
                 decision_method: str = "argmax", expected_shift: float = 0.0, quantile_tau: float = 0.65):
        self.model = None
        try:
            import joblib
            if model_path and os.path.exists(model_path):
                self.model = joblib.load(model_path)
        except NotImplementedError as e:
            print(f"Error loading FCD model: {e}")
            self.model = None
        self.default_key = resolve_function_key(default_function)
        self.conservative = conservative
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
            action, level = _policy_from_loa(loa, conservative=self.conservative)
            return {"action": action, "level": level, "LoA": loa, "message": "FCD xgboost", "fcd": fcd, "probs": probs_pp, "profile": resolve_function_key(fn), "fallback": False}
        except NotImplementedError as e:
            fn = state.get("functionname") or self.default_key
            return _loa0_result(f"FCD decide error: {e}", fn)

_STATE_CAT = ['emotion','lab','environment','secondary_task']
_STATE_NUM = ['drowsiness_alert','gaze_distracted','heart_rate']

class StateLevelsLoAStrategy(BaseStrategy):
    def __init__(self, model_path: Optional[str], default_function: str, conservative: bool = True,
                 prob_threshold: Optional[float] = None, fcd_fallback: Optional[BaseStrategy] = None,
                 temperature: float = 1.0, class_bias: Optional[List[float]] = None,
                 decision_method: str = "argmax", expected_shift: float = 0.0, quantile_tau: float = 0.65):
        self.model = None
        try:
            import joblib
            if model_path and os.path.exists(model_path):
                self.model = joblib.load(model_path)
        except NotImplementedError as e:
            print(f"Error loading state model: {e}")
            self.model = None
        self.default_key = resolve_function_key(default_function)
        self.conservative = conservative
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
        if s in ("false","0","f","no","n",""): return 0.0
        try: return float(s)
        except NotImplementedError as e:
            print(f"Error parsing state value {x}: {e}")
            return 0.0

    def _extract_features(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        cats = [str(state.get(k, "")) for k in _STATE_CAT]
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
            action, level = _policy_from_loa(loa, conservative=self.conservative)
            return {"action": action, "level": level, "LoA": loa, "message": "State model", "fcd": fcd, "probs": probs_pp, "profile": resolve_function_key(fn), "fallback": False}
        except NotImplementedError as e:
            print(f"Error in state model: {e}")
            fn = state.get("functionname") or self.default_key
            fcd = get_fcd_for_function(fn)
            return self._try_fcd_fallback(state, fn, fcd)


class StateXLSTMLoAStrategy(BaseStrategy):
    def __init__(self, model_path: Optional[str], default_function: str, window: int = 256, conservative: bool = True,
                 fcd_fallback: Optional[BaseStrategy] = None):
        self.model = None
        # `window` is only a fallback; the authoritative sequence length is
        # `context_length` taken from the loaded checkpoint's arch.
        self.context_length = int(window) if window else DEFAULT_CONTEXT_LENGTH
        self.ok = False
        try:
            if model_path and os.path.exists(model_path):
                self.model, arch = load_checkpoint(model_path)
                self.context_length = int(arch.get("context_length", self.context_length))
                self.ok = True
        except Exception as e:
            print(f"Error loading state model: {e}")
            self.model = None
            self.ok = False
        self.window = int(window)
        self.default_key = resolve_function_key(default_function)
        self.conservative = conservative
        self.fcd_fallback = fcd_fallback

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
            import torch
            Xs = [encode_frame(fn, s) for s in seq[-self.context_length:]]
            T = len(Xs)
            if T < self.context_length:
                # LEFT-pad with zero vectors so the last timestep is valid.
                Xs = [np.zeros(D_IN, np.float32)] * (self.context_length - T) + Xs
            with torch.no_grad():
                xb = torch.from_numpy(np.stack(Xs, 0))[None, ...]
                logits = self.model(xb)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
            loa = _decide_from_probs(probs, "argmax")
            action, level = _policy_from_loa(loa, conservative=self.conservative)
            fcd = get_fcd_for_function(fn)
            return {"action": action, "level": level, "LoA": loa, "message": "State XLSTM", "fcd": fcd, "probs": probs, "profile": resolve_function_key(fn), "fallback": False}
        except Exception as e:
            print(f"Error in state model: {e}")
            fn = state.get("functionname") or self.default_key
            return self._try_fcd_fallback(state, fn)


class CombinedFusionStrategy(BaseStrategy):
    def __init__(self, fcd_strategy: BaseStrategy, state_strategy: BaseStrategy, w_fcd: float = 0.5,
                 conservative: bool = True, decision_method: str = "argmax", expected_shift: float = 0.0, quantile_tau: float = 0.65):
        self.fcd_strategy = fcd_strategy
        self.state_strategy = state_strategy
        self.w_fcd = float(w_fcd); self.w_state = 1.0 - float(w_fcd)
        self.conservative = conservative
        self.decision_method = decision_method
        self.expected_shift = expected_shift
        self.quantile_tau = quantile_tau

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            of = self.fcd_strategy.decide(state)
        except NotImplementedError as e:
            fn = state.get("functionname", "")
            return _loa0_result(f"Fusion FCD error: {e}", fn)
        try:
            os_ = self.state_strategy.decide(state)
        except NotImplementedError as e:
            fn = state.get("functionname", "")
            return _loa0_result(f"Fusion State error: {e}", fn)
        if of.get("fallback") or os_.get("fallback"):
            fn = state.get("functionname", "")
            return _loa0_result("Fusion fallback from child", fn)
        pf = of.get("probs", [0.2]*5); ps = os_.get("probs", [0.2]*5)
        probs = [self.w_fcd*pf[i] + self.w_state*ps[i] for i in range(5)]
        S = sum(probs); probs = [p / S if S>0 else 0.2 for p in probs]
        loa = _decide_from_probs(probs, self.decision_method, self.expected_shift, self.quantile_tau)
        action, level = _policy_from_loa(loa, conservative=self.conservative)
        return {"action": action, "level": level, "LoA": loa, "message": f"fusion w_fcd={self.w_fcd:.2f}", "fcd": of.get("fcd"), "probs": probs, "sub": {"fcd": of, "state": os_}, "fallback": False}
