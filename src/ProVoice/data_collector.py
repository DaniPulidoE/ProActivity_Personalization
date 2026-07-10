from __future__ import annotations

import base64
import datetime
import os
import threading
import time
import urllib.request
import json
from typing import Any, Dict, Optional, Tuple
from collections import deque

import math
import statistics
import cv2
import numpy as np
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh if mp else None

from rPPG.rppg_infer_simple import OnlineRPPG
from ProVoice import perception as _perception  # in-tree replacement for yolov5-deepsort

HAS_CV2 = True
HAS_MYFRAME = True  # kept for backward-compat; gates the perception.frametest call
HAS_RPPG = True
HAS_NP = True
HAS_MP = True


try:
    os.environ["KERAS_BACKEND"] = "torch"
    from keras.models import load_model  # type: ignore
    HAS_KERAS = True
except Exception as e:
    print(e, "Error loading keras")
    load_model = None  # type: ignore
    HAS_KERAS = False

_emotion_model = None
_face_detector = None
_emotion_input_size: Optional[Tuple[int, int]] = None

def _load_emotion_model(path: str) -> None:
    global _emotion_model, _face_detector, _emotion_input_size
    if not (HAS_KERAS and HAS_CV2):
        return
    if _emotion_model is not None:
        return
    if not os.path.exists(path):
        # The default path is CWD-relative; when launched from another directory, fall back to the copy shipped next to this module.
        fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'trained_models', os.path.basename(path))
        if os.path.exists(fallback):
            path = fallback
        else:
            print(f"[_load_emotion_model] FER model not found at {path!r} "
                  f"(nor {fallback!r}); emotion detection disabled.")
            return
    try:
        _emotion_model = load_model(path, compile=False)  # type: ignore
    except Exception as e:
        print(e, f"Error loading emotion model from {path}; emotion detection disabled")
        _emotion_model = None
        return
    print(f"[_load_emotion_model] Emotion model loaded successfully from {path}")
    _face_detector = cv2.CascadeClassifier(  # type: ignore
        os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    )
    _emotion_input_size = _emotion_model.input_shape[1:3]


class DataCollector:
    def __init__(
        self,
        visual: bool = True,
        physiological: bool = True,
        context: bool = True,
        sample_rate: float = 20.0,
        logger: Optional[Any] = None,
        decision_engine: Optional[Any] = None,
        actuator: Optional[Any] = None,
        function_name: str = "fatigue_alert",
        fer_model_path: str = './src/ProVoice/trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5',
        cam_index: int | str = 0,
        static_context: Optional[Dict[str, Any]] = None,  
        carla_vehicle: Optional[Any] = None,
        window_size: int = 400, # 20 seconds at 20Hz (as user inputs label each 20 seconds)
        vehicle_state_url: Optional[str] = None,
    ) -> None:
        # Everything stop()/__del__ touches is set FIRST, so a constructor
        # that fails part-way still leaves a safely destructible object.
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state_poll_thread: Optional[threading.Thread] = None
        self.rppg_estimator = None
        self.cap = None

        self.visual_enabled = bool(visual and HAS_CV2)
        self.phys_enabled = bool(physiological)
        self.context_enabled = bool(context)
        self.sampling_interval = max(0.02, 1.0 / float(sample_rate))

        self.logger = logger
        self.decision_engine = decision_engine
        self.actuator = actuator

        self.functionname = function_name or "fatigue_alert"

        self.cam_index = cam_index
        self.static_context: Dict[str, Any] = dict(static_context or {})

        self.face_mesh = None
        self.carla_vehicle = carla_vehicle
        self.vehicle_state_url = vehicle_state_url.rstrip("/") if vehicle_state_url else None
        self._cached_speed: int = 0
        self._cached_steer: int = 0
        self._cached_brake: int = 0
        self._cached_precipitation: int = 0
        self._cached_speed_limit: int = 0
        self._cached_night: int = 0
        self._cached_junction: int = 0
        self._cached_throttle: float = 0.0
        self._cached_gear: int = 0
        self._cached_hand_brake: bool = False
        self._cached_reverse: bool = False
        self._cached_acceleration: float = 0.0
        self._cached_fog_density: float = 0.0
        self._cached_traffic_light_state: Optional[str] = None
        self._cached_headlight: bool = False
        self._cached_fog_light: bool = False
        self._cached_left_indicator: bool = False
        self._cached_right_indicator: bool = False

        # If a CARLA actor is present, attempt to retrieve the vehicle_id
        self.vehicle_id = None
        if self.carla_vehicle is not None:
            try:
                self.vehicle_id = getattr(self.carla_vehicle, "id", None)
            except Exception as e:
                print("[DataCollector] Error getting vehicle_id from CARLA actor:", e)
                
        if self.visual_enabled:
            try:
                self.cap = cv2.VideoCapture(self.cam_index)  # type: ignore
                print(f"Connecting: {self.cam_index} ...")
                print(f"Camera opened: {self.cap.isOpened()}")

            except Exception as e:
                print(e, "Error opening camera")
                self.cap = None
                self.visual_enabled = False
            if HAS_MP:
                try:
                    print("[DataCollector] Initializing MediaPipe Face Mesh...")
                    self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)  # type: ignore
                    print("[DataCollector] MediaPipe Face Mesh loaded successfully.")
                except Exception as e:
                    print(e, "Error initializing face mesh")
                    self.face_mesh = None

            if HAS_MYFRAME:
                print("[DataCollector] Distraction/fatigue perception module detected (MediaPipe + Ultralytics YOLO26).")

            if HAS_RPPG and OnlineRPPG is not None:
                try:
                    self.rppg_estimator = OnlineRPPG(frame_rate=10, crop_size=72)  # type: ignore
                except Exception as e:
                    print(e, "Error initializing rPPG estimator")
                    self.rppg_estimator = None
                    #raise e

        if HAS_KERAS and HAS_CV2:
            _load_emotion_model(fer_model_path)

        self.latest_frame = None  # BGR
        self.latest_data: Dict[str, Any] = {}
        self.bpm_history: deque = deque(maxlen=80)
        self.rr_history: deque = deque(maxlen=80)
        # rPPG carry-forward
        self._last_hr: Optional[float] = None
        self._last_rr: Optional[float] = None
        self._last_hr_t: float = 0.0
        self._last_rr_t: float = 0.0
        self._rppg_staleness_s: float = 15.0
        self.data_history: deque = deque(maxlen=window_size)
        self.window_size = window_size
        self.blink_count = 0
        self.yawn_count = 0
        self.perclos = 0.0
        self.drowsiness_alert = False
        self.mCOUNTER = 0
        # PERCLOS over a fixed TIME window (rate-independent)
        self._perclos_buf: deque = deque()
        self._perclos_window_s: float = 10.0
        self._eye_closed_since: float = 0.0  # monotonic time when EAR first dropped
        # yawn and blink rates (instead of raw counts) should act as better predictors
        self.blink_times = []
        self.blink_rate = 0.0
        self.yawn_times = []
        self.yawn_rate = 0.0


        self._lock = threading.Lock()

        # calibration state
        self.calibrated = False
        self.calibrate: Dict[str, Any] = {}      # populated by compute_calibration()
        self._session_start_t: float = 0.0       # set when calibration completes
        self._calibration_data = dict({'gaze_score': [], 'ear': [], 'mar': [], 'bpm': [], 'rr': []})
        self._calibration_ear_ts = [] # timestamps of EAR values to callibrate blink rate/perclos
        self._calibration_mar_ts = [] # timestamps of MAR values to callibrate perclos

        # Calibration: 60 s time-based, collects gaze/EAR/MAR/HR/RR
        self._calibration_duration_s: float = 60.0
        self._calibration_start_t: float = 0.0
        # Transient camera-read failures (warm-up frames, momentary USB glitches)
        # must NOT abort calibration; only give up after this many seconds of
        # UNBROKEN failures. Resets on any successful read.
        self._read_fail_since: float = 0.0
        self._read_fail_grace_s: float = 5.0

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass  # never raise during interpreter teardown

    def detect_emotion(self, faces, gray) -> Optional[Dict[str, Any]]:
        if _emotion_model is None or _emotion_input_size is None:
            return None
        try:
            if len(faces) == 0:
                return None
            x, y, w, h = faces[0]
            face = gray[y:y + h, x:x + w]
            face_em = cv2.resize(face, _emotion_input_size)  # type: ignore
            face_em = face_em.astype('float32') / 255.0
            face_em = face_em[None, ..., None]  # (1,H,W,1)
            preds = _emotion_model.predict(face_em, verbose=0)[0]  # type: ignore
            arg = int(preds.argmax())
            conf = float(preds[arg])
            label = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}.get(arg, 'neutral')
            return {'emotion': label, 'emotion_prob': round(conf, 3)}
        except Exception as e:
            print(e, "Error detecting emotion")
            return None

    @staticmethod
    def compute_gaze_score(landmarks, image_width: int, image_height: int) -> float:
        if not HAS_NP:
            return 0.0
        try:
            left_pts = [landmarks[i] for i in [468, 469, 470, 471]]
            right_pts = [landmarks[i] for i in [473, 474, 475, 476]]

            def avg_point(pts):
                xs = [p.x for p in pts]
                ys = [p.y for p in pts]
                return np.array([np.mean(xs) * image_width, np.mean(ys) * image_height])

            left_center = avg_point(left_pts)
            right_center = avg_point(right_pts)
            left_outer = landmarks[33]
            left_inner = landmarks[133]
            right_inner = landmarks[362]
            right_outer = landmarks[263]
            left_eye_center = avg_point([left_outer, left_inner])
            right_eye_center = avg_point([right_outer, right_inner])
            left_eye_width = np.linalg.norm(
                (np.array([left_outer.x, left_outer.y]) - np.array([left_inner.x, left_inner.y])) * np.array([image_width, image_height])
            )
            right_eye_width = np.linalg.norm(
                (np.array([right_outer.x, right_outer.y]) - np.array([right_inner.x, right_inner.y])) * np.array([image_width, image_height])
            )
            left_score = np.linalg.norm(left_center - left_eye_center) / max(left_eye_width, 1e-6)
            right_score = np.linalg.norm(right_center - right_eye_center) / max(right_eye_width, 1e-6)
            return float((left_score + right_score) / 2.0)
        except Exception as e:
            print(e, "Error computing gaze score")
            return 0.0

    
    def get_gaze_score(self, frame, face_mesh_results) -> float:
        if not face_mesh_results:
            return 0.0
        try:
            if not face_mesh_results.multi_face_landmarks:
                return 0.0
            lm = face_mesh_results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            return self.compute_gaze_score(lm, w, h)
        except Exception as e:
            print(e, "Error computing gaze score")
            return 0.0
    
    def compute_face_mesh(self, frame):
        if not self.face_mesh or not HAS_MP:
            return None
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
            results = self.face_mesh.process(img_rgb)
            return results
        except Exception as e:
            print(e, "Error computing face mesh")
            return None

    def calibrate_step(self) -> None:
        if not self.visual_enabled or self.cap is None:
            # No camera at all — no baseline is obtainable; finish immediately
            # with defaults so the context/CARLA pipeline can proceed.
            print("[Calibration] Visual disabled or camera unavailable — "
                  "finishing calibration with default baselines.")
            self.compute_calibration()
            return
        ok, frame = self.cap.read()
        now = time.monotonic()
        if not ok:
            # Tolerate transient read failures (camera warm-up, brief glitches):
            # keep trying until the grace window elapses, then give up loudly.
            if self._read_fail_since == 0.0:
                self._read_fail_since = now
            self.latest_frame = None
            failed_for = now - self._read_fail_since
            if failed_for >= self._read_fail_grace_s:
                print(f"[Calibration] Camera read failed for "
                      f"{failed_for:.1f}s (>{self._read_fail_grace_s:.0f}s grace) — "
                      f"finishing calibration with whatever baselines exist.")
                self.compute_calibration()
            else:
                print(f"[Calibration] Camera read failed ({failed_for:.1f}s) — "
                      f"waiting for frames...")
            return
        self._read_fail_since = 0.0  # good frame — reset the failure timer

        if self._calibration_start_t == 0.0:
            self._calibration_start_t = now
            print("[Calibration] Starting 60 s calibration.")

        # ── 60-second calibration ─────────────────────────────────────────────
        face_mesh_results = self.compute_face_mesh(frame)

        gaze_score = self.get_gaze_score(frame, face_mesh_results)
        if gaze_score > 0.0:
            self._calibration_data['gaze_score'].append(gaze_score)

        if HAS_MYFRAME:
            try:
                ret, frame_annot = _perception.frametest(frame, face_mesh_results)
                lab, eye, mouth = ret
                # Only fold EAR/MAR into the baseline when a face is actually present. 
                if "face" in (lab or []):
                    self._calibration_data['ear'].append(eye)
                    self._calibration_data['mar'].append(mouth)
                    self._calibration_ear_ts.append((now, eye))
                    self._calibration_mar_ts.append((now, mouth))
            except Exception as e:  # noqa: BLE001
                print(e, "Error computing perception.frametest")
                frame_annot = frame
        else:
            frame_annot = frame

        # Face detector — needed for both rPPG and emotion (mirrors _visual_process)
        if _face_detector is not None and (
                self.rppg_estimator is not None or
                (_emotion_model is not None and _emotion_input_size is not None)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore
            faces = _face_detector.detectMultiScale(gray, 1.3, 5)  # type: ignore
        else:
            gray = None
            faces = []

        # Feed rPPG — collect HR/RR once the model starts producing values
        if self.rppg_estimator is not None and len(faces) > 0:
            x, y, w, h = faces[0]
            hr, rr = self.rppg_estimator.add_frame(frame[y:y + h, x:x + w])
            print(f"[Calibration] rPPG: HR={hr}, RR={rr}")
            if hr is not None and not (hr != hr):  # excludes NaN
                self._calibration_data['bpm'].append(float(hr))
            if rr is not None and not (rr != rr):
                self._calibration_data['rr'].append(float(rr))


        self.latest_frame = frame_annot
        print(f"[Calibration] {now - self._calibration_start_t:.1f}/{self._calibration_duration_s:.1f} s elapsed. ")
        if (now - self._calibration_start_t) >= self._calibration_duration_s:
            self.compute_calibration()

    def compute_calibration(self):
        # compute mean and std for each metric, set calibrated flag
        self.calibrate = dict()
        for key, values in self._calibration_data.items():
            if values:
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                # 0.65 threshold for MAR based on literature (doesn't make sense to calibrate based on a closed mouth)
                # EAR "eyes-closed" threshold: a fraction of the open-eye baseline
                # (with an absolute floor), which is robust to a tiny calibration
                # std. mean - 2.5*std sat right inside the normal operating range
                # and made PERCLOS/drowsiness fire almost constantly.
                self.calibrate[key] = {'mean': mean, 'std': std, 'threshold': mean + std * 2.5 if key in ['gaze_score'] else max(0.15, mean * 0.6) if key in ['ear'] else 0.65}
            else:
                # default values originally in the script
                thres = 0.2 if key in ['gaze_score', 'ear'] else 0.65
                self.calibrate[key] = {'mean': 0.0, 'std': 0.0, 'threshold': thres}
                
        # blink rate
        threshold_ear = self.calibrate['ear']['threshold']
        blink_count = 0
        eye_closed_since = 0.0
        for ts, ear in self._calibration_ear_ts:
            if ear < threshold_ear:
                if eye_closed_since == 0.0:
                    eye_closed_since = ts
            else:
                if eye_closed_since > 0.0:
                    duration_ms = (ts - eye_closed_since) * 1000
                    if 100 <= duration_ms <= 500:
                        blink_count += 1
                eye_closed_since = 0.0
        self.calibrate['blink_rate'] = {'mean': blink_count / (self._calibration_duration_s / 60)}  # blinks/min
        
        # perclos baseline: perclos over sliding window
        threshold_mar = self.calibrate['mar']['threshold']
        perclos_series = []
        if self._calibration_ear_ts:
            t0 = self._calibration_ear_ts[0][0]
            buf: deque = deque()
            for (ts, ear), (_, mar) in zip(self._calibration_ear_ts, self._calibration_mar_ts):
                buf.append((ts, ear < threshold_ear, mar > threshold_mar))
                while buf and ts - buf[0][0] > self._perclos_window_s:
                    buf.popleft()
                if ts - t0 < self._perclos_window_s:
                    continue  # window not yet full — skip warm-up
                nb = len(buf)
                eye_frac = sum(1 for _, e, _ in buf if e) / nb
                mouth_frac = sum(1 for _, _, m in buf if m) / nb
                perclos_series.append(eye_frac + mouth_frac * 0.2)
        mean_perclos = float(np.mean(perclos_series)) if perclos_series else 0.0
        if len(perclos_series) > 1:
            std_perclos = max(float(np.std(perclos_series)), 1e-3)
        else:  # no full-window samples collected -> std=1.0 (not 1e-6)
            std_perclos = 1.0
        self.calibrate['perclos'] = {'mean': mean_perclos, 'std': std_perclos}

        self.calibrated = True
        self._session_start_t = time.monotonic()
        
        print("Calibration completed:", self.calibrate)    
    

    def _standardized_delta(self, value: float, cal_key: str, history) -> float:
        """(value - baseline) / std as a z-score.

        Baseline and std come from the 60 s calibration for ``cal_key`` when it
        produced samples; otherwise they fall back to the rolling ``history``
        stats. Every degenerate case (no calibration, short history, zero std)
        collapses to a unit std, so the result is the raw deviation — never a
        divide-by-zero or an exploded value.
        """
        cal = self.calibrate.get(cal_key, {}) if hasattr(self, 'calibrate') else {}
        cal_mean = cal.get('mean', 0.0)
        baseline = cal_mean if cal_mean > 0.0 else (sum(history) / len(history) if history else 0.0)
        cal_std = cal.get('std', 0.0)
        if cal_std > 0.0:
            std = cal_std
        elif len(history) > 1:
            std = statistics.stdev(history)
        else:
            std = 1.0
        if std == 0.0:
            std = 1.0
        return round((value - baseline) / std, 1)

    def _visual_process(self, data: Dict[str, Any]) -> None:
        if not self.visual_enabled or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            print("not okay")
            self.latest_frame = None
            return

        # reuse face mesh in gaze score and emotion detection to avoid double processing
        face_mesh_results = self.compute_face_mesh(frame) 
        # gaze score
        gaze_score = self.get_gaze_score(frame, face_mesh_results)
        # standardize gaze score based on calibration mean and std, if available
        normalized_gaze_score = (gaze_score - self.calibrate['gaze_score']['mean']) / self.calibrate['gaze_score']['std'] if self.calibrate['gaze_score']['std'] != 0 else 0.0
        data['gaze_score'] = round(normalized_gaze_score, 3)
        data['gaze_distracted'] = bool(gaze_score > self.calibrate['gaze_score']['threshold'])

        if _face_detector is not None and (
                self.rppg_estimator is not None or (_emotion_model is not None and _emotion_input_size is not None)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore
            faces = _face_detector.detectMultiScale(gray, 1.3, 5)  # type: ignore
        else:
            gray = None
            faces = []


        if self.rppg_estimator is not None:  # type: ignore
            # rPPG: heart rate and respiratory rate are computed from the face region, if detected
            now = time.monotonic()
            # Only feed rPPG (and update the held reading) when a face is present.
            # rPPG emits a fresh value only every ~45 face frames; on the frames
            # in between it returns None and the held value is left untouched.
            if len(faces) > 0:
                x, y, w, h = faces[0]
                hr, rr = self.rppg_estimator.add_frame(frame[y:y + h, x:x + w])
                #print(f"rPPG: HR={hr}, RR={rr}")
                if hr is not None and not (hr != hr):   # fresh, non-NaN reading
                    self._last_hr = round(float(hr), 1)
                    self._last_hr_t = now
                    self.bpm_history.append(self._last_hr)
                if rr is not None and not (rr != rr):
                    self._last_rr = round(float(rr), 1)
                    self._last_rr_t = now
                    self.rr_history.append(self._last_rr)

            # Carry the last reading forward while it is fresh (avoids most inputs being null)
            if self._last_hr is not None and (now - self._last_hr_t) <= self._rppg_staleness_s:
                data['heart_rate'] = self._last_hr
                # snapshot as list: deque is not JSON-serializable (dashboard
                # emit), and a live reference would mutate under stored frames
                data['bpm_history'] = list(self.bpm_history)
                data['hr_delta'] = self._standardized_delta(self._last_hr, 'bpm', self.bpm_history)
            if self._last_rr is not None and (now - self._last_rr_t) <= self._rppg_staleness_s:
                data['respiratory_rate'] = self._last_rr
                data['rr_history'] = list(self.rr_history)
                data['rr_delta'] = self._standardized_delta(self._last_rr, 'rr', self.rr_history)

        emo = self.detect_emotion(faces, gray)
        if emo:
            data.update(emo)  # emotion, emotion_prob

        if HAS_MYFRAME:
            try:
                ret, frame_annot = _perception.frametest(frame, face_mesh_results)
                lab, eye, mouth = ret
            except Exception as e:  # noqa: BLE001 (perception code crosses C extensions)
                print(e, "Error computing perception.frametest")
                frame_annot = frame
                lab, eye, mouth = ([], 0.3, 0.5)
        else:
            frame_annot = frame
            lab, eye, mouth = ([], 0.3, 0.5)

        MOUTH_AR_CONSEC_FRAMES = 3
        BLINK_MIN_MS = 100   # below this is EAR noise, not a blink (this is equivalent to 2 frames at 20Hz)
        BLINK_MAX_MS = 500  # above this is eye closure / drowsiness, not a blink

        now = time.monotonic()
        eye_closed = eye < self.calibrate['ear']['threshold']
        mouth_open = mouth > self.calibrate['mar']['threshold']

        # blink detection (duration-gated; unchanged)
        if eye_closed:
            if self._eye_closed_since == 0.0:
                self._eye_closed_since = now
        else:
            if self._eye_closed_since > 0.0:
                duration_ms = (now - self._eye_closed_since) * 1000
                if BLINK_MIN_MS <= duration_ms <= BLINK_MAX_MS:
                    self.blink_count += 1
                    self.blink_times.append(now)
            self._eye_closed_since = 0.0

        # yawn detection (consecutive-frame gated; unchanged)
        if mouth_open:
            self.mCOUNTER += 1
        else:
            if self.mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                self.yawn_count += 1
                self.yawn_times.append(now)
            self.mCOUNTER = 0

        # PERCLOS over a fixed TIME window
        self._perclos_buf.append((now, eye_closed, mouth_open))
        while self._perclos_buf and now - self._perclos_buf[0][0] > self._perclos_window_s:
            self._perclos_buf.popleft()
        n_buf = len(self._perclos_buf)
        if n_buf > 0:
            eye_frac = sum(1 for _, e, _ in self._perclos_buf if e) / n_buf
            mouth_frac = sum(1 for _, _, m in self._perclos_buf if m) / n_buf
            self.perclos = eye_frac + mouth_frac * 0.2
            self.drowsiness_alert = self.perclos > 0.38

        # yawn and blink rates
        now = time.monotonic()
        blink_window = 30.0 # 30 second window for blinks
        self.blink_times = [t for t in self.blink_times if now - t <= blink_window]
        elapsed_s = min(now - self._session_start_t, blink_window)
        self.blink_rate = len(self.blink_times) / (elapsed_s / 60) if elapsed_s > 0 else 0.0  # blinks per minute
        yawn_window = 180 # 3 minute window for yawns
        self.yawn_times = [t for t in self.yawn_times if now - t <= yawn_window]
        self.yawn_rate = len(self.yawn_times) / (yawn_window / 60)  # yawns per minute


        self.latest_frame = frame_annot
        
        # Add the computed metrics to the data dictionary (normalized)
        # normalize blink rate (it is a Poisson divided by a specific value - Variance=X/n)
        normalized_blink_rate = (self.blink_rate - self.calibrate.get('blink_rate', {}).get('mean', 0.0)) / math.sqrt((self.calibrate.get('blink_rate', {}).get('mean', 1.0)/(elapsed_s / 60))) if (self.calibrate.get('blink_rate', {}).get('mean', 0.0) > 0 and elapsed_s > 0) else 0.0
        data['blink_rate'] = round(float(normalized_blink_rate), 3)
        # Anscombe transform to normalize yawn rate
        normalized_yawn_rate = 2*math.sqrt((self.yawn_rate*(yawn_window / 60)) + 3/8) 
        # normalize perclos
        normalized_perclos = (self.perclos - self.calibrate.get('perclos', {}).get('mean', 0.0)) / self.calibrate.get('perclos', {}).get('std', 1.0) if self.calibrate.get('perclos', {}).get('std', 1.0) > 0 else 0.0
        data['perclos'] = round(float(normalized_perclos), 3)
        data['yawn_rate'] = round(float(normalized_yawn_rate), 3)   
        data['drowsiness_alert'] = bool(self.drowsiness_alert)
        data['eye_ar'] = float(eye)
        data['mar'] = float(mouth)
        data['lab'] = [str(x) for x in (lab or [])]
        # Raw (non-normalized) values for the dashboard display only; the model
        # reads the normalized keys above. EAR/MAR are already raw. These extras
        # are ignored by encode_frame and dropped from decisions.csv's schema.
        data['gaze_score_raw'] = round(float(gaze_score), 3)          # raw gaze deviation
        data['blink_rate_raw'] = round(float(self.blink_rate), 2)     # blinks/min
        data['yawn_rate_raw'] = round(float(self.yawn_rate), 2)       # yawns/min
        data['perclos_raw'] = round(float(self.perclos), 3)           # fraction [0,1]
        
    def _carla_process(self, data: Dict[str, Any]):
        # Logging-only extras default to None; overwritten below when data is available.
        throttle = gear = hand_brake = reverse = acceleration = None
        fog_density = traffic_light_state = None
        headlight = fog_light = left_indicator = right_indicator = None

        if self.carla_vehicle is not None:
            try:
                vel = self.carla_vehicle.get_velocity()
                speed = int((vel.x**2 + vel.y**2 + vel.z**2)**0.5 * 3.6)

                control = self.carla_vehicle.get_control()
                brake = control.brake
                steer = control.steer
                throttle = control.throttle
                gear = control.gear
                hand_brake = bool(control.hand_brake)
                reverse = bool(control.reverse)

                acc = self.carla_vehicle.get_acceleration()
                acceleration = round((acc.x**2 + acc.y**2 + acc.z**2)**0.5, 3)

                speed_lim = self.carla_vehicle.get_speed_limit()

                world = self.carla_vehicle.get_world()
                weather = world.get_weather()
                precipitation = round(weather.precipitation / 100.0, 3)
                is_night = bool(weather.sun_altitude_angle < 0)
                fog_density = round(weather.fog_density / 100.0, 3)

                location = self.carla_vehicle.get_location()
                waypoint = world.get_map().get_waypoint(location)
                is_junction = bool(waypoint.is_junction)

                # traffic light state (Red/Yellow/Green/Off/Unknown)
                try:
                    traffic_light_state = str(self.carla_vehicle.get_traffic_light_state()).split('.')[-1]
                except Exception:
                    pass

                # vehicle light state (bitmask: LowBeam=2, HighBeam=4, RightBlinker=16, LeftBlinker=32, Fog=128)
                try:
                    ls = int(self.carla_vehicle.get_light_state())
                    headlight = bool(ls & (2 | 4))
                    fog_light = bool(ls & 128)
                    left_indicator = bool(ls & 32)
                    right_indicator = bool(ls & 16)
                except Exception:
                    pass

                self._cached_speed = speed
                self._cached_steer = steer
                self._cached_brake = brake
                self._cached_speed_limit = speed_lim
                self._cached_precipitation = precipitation
                self._cached_night = is_night
                self._cached_junction = is_junction
                self._cached_throttle = throttle
                self._cached_gear = gear
                self._cached_hand_brake = hand_brake
                self._cached_reverse = reverse
                self._cached_acceleration = acceleration
                self._cached_fog_density = fog_density
                if traffic_light_state is not None:
                    self._cached_traffic_light_state = traffic_light_state
                if headlight is not None:
                    self._cached_headlight = headlight
                    self._cached_fog_light = fog_light
                    self._cached_left_indicator = left_indicator
                    self._cached_right_indicator = right_indicator

            except Exception as e:
                print("[DataCollector] Error reading vehicle state:", e)
                speed = self._cached_speed
                steer = self._cached_steer
                brake = self._cached_brake
                speed_lim = self._cached_speed_limit
                precipitation = self._cached_precipitation
                is_night = self._cached_night
                is_junction = self._cached_junction
                throttle = self._cached_throttle
                gear = self._cached_gear
                hand_brake = self._cached_hand_brake
                reverse = self._cached_reverse
                acceleration = self._cached_acceleration
                fog_density = self._cached_fog_density
                traffic_light_state = self._cached_traffic_light_state
                headlight = self._cached_headlight
                fog_light = self._cached_fog_light
                left_indicator = self._cached_left_indicator
                right_indicator = self._cached_right_indicator

        elif self.vehicle_state_url is not None:
            # Updated by _poll_vehicle_state background thread — just read cache.
            speed = self._cached_speed
            brake = self._cached_brake
            steer = self._cached_steer
            speed_lim = self._cached_speed_limit
            precipitation = self._cached_precipitation
            is_night = self._cached_night
            is_junction = self._cached_junction
            throttle = self._cached_throttle
            gear = self._cached_gear
            hand_brake = self._cached_hand_brake
            reverse = self._cached_reverse
            acceleration = self._cached_acceleration
            fog_density = self._cached_fog_density
            traffic_light_state = self._cached_traffic_light_state
            headlight = self._cached_headlight
            fog_light = self._cached_fog_light
            left_indicator = self._cached_left_indicator
            right_indicator = self._cached_right_indicator

        else:
            # No CARLA actor or bridge — minimal env-var fallback.
            pv = os.getenv('PV_SPEED')
            speed = int(pv) if pv not in (None, '') else None
            brake = None
            steer = None
            speed_lim = None
            precipitation = None
            is_night = None
            is_junction = None

        MAX_SPEED = 150
        data['speed_ratio_max']   = speed / MAX_SPEED if speed is not None else None
        data['speed_ratio_limit'] = speed / speed_lim if (speed_lim is not None and speed_lim > 0) else -1
        data['brake']             = brake
        data['steer']             = steer
        data['precipitation']     = precipitation
        data['is_night']          = is_night
        data['is_junction']       = is_junction
        data['speed_kmh']           = speed
        data['speed_limit_kmh']     = speed_lim
        data['throttle']            = throttle
        data['gear']                = gear
        data['hand_brake']          = hand_brake
        data['reverse']             = reverse
        data['acceleration']        = acceleration
        data['fog_density']         = fog_density
        data['traffic_light_state'] = traffic_light_state
        data['headlight']           = headlight
        data['fog_light']           = fog_light
        data['left_indicator']      = left_indicator
        data['right_indicator']     = right_indicator

    def collect_data(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        
        if self.visual_enabled:
            self._visual_process(data)

        if self.phys_enabled:
            # No live rPPG reading this cycle — report unknown rather than
            # fabricating a value that pollutes the model input and dashboard.
            if 'heart_rate' not in data:
                data['heart_rate'] = None
                data['hr_delta'] = None
            
            if 'respiratory_rate' not in data:
                data['respiratory_rate'] = None
                data['rr_delta'] = None


        if self.context_enabled:
            self._carla_process(data)

        data['timestamp'] = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

        ctx = dict(self.static_context)
        ctx['functionname'] = ctx.get('functionname') or self.functionname
        for k, v in ctx.items():
            if v not in (None, ""):
                data[k] = v

        if 'emotion' not in data or not data['emotion']:
            data['emotion'] = 'neutral'

        with self._lock:
            self.latest_data = dict(data)

        return data


    def _run_loop(self) -> None:
        # Old: frame-count calibration counter
        # calibration_counter = 0

        next_t = time.monotonic()
        while self._running:
            try:
                if self.calibrated is False:
                    # Two-phase time-based calibration — completion handled inside calibrate_step()
                    self.calibrate_step()
                else:
                    data = self.collect_data()
                    
                    # Add sequence for LSTM models.
                    seq_entry = dict(data)
                    seq_entry.pop('bpm_history', None)
                    seq_entry.pop('rr_history', None)
                    with self._lock:
                        self.data_history.append(seq_entry) # no need to pop as we use deque with maxlen now

                    action = None
                    if self.decision_engine:
        
                        data['functionname'] = data.get('functionname', self.functionname)
                        # separate dict to avoid adding the whole history to each sequence entry
                        data_for_decision = dict(data)
                        data_for_decision['sequence'] = list(self.data_history)

                        action = self.decision_engine.decide(dict(data_for_decision))
                        if self.logger and isinstance(action, dict):
                            action_for_log = dict(action)
                            for key in ('timestamp', 'session_id', 'participantid', 'environment', 'secondary_task', 'functionname', 'emotion', 'modeltype', 'state_model', 'w_fcd', 'hr_delta', 'rr_delta'):
                                value = data.get(key)
                                if value not in (None, ''):
                                    action_for_log.setdefault(key, value)
                            self.logger.log_processed(action_for_log)
                        data['last_action'] = action
                        # collect_data() snapshotted latest_data BEFORE the
                        # decision existed; re-publish so the dashboard/get_latest
                        # actually shows the LoA/action.
                        with self._lock:
                            self.latest_data = dict(data)


                    if self.logger:
                        raw_with_decision = dict(data)
                        if isinstance(action, dict):
                            raw_with_decision['LoA'] = action.get('LoA')
                            fcd = action.get('fcd') or action.get('fcd_scores')
                            if isinstance(fcd, dict):
                                raw_with_decision['FCD'] = fcd
                        raw_with_decision.pop('bpm_history', None)
                        raw_with_decision.pop('rr_history', None)
                        self.logger.log_raw(raw_with_decision)


                    if self.actuator and action is not None:
                        self.actuator.execute(action)

            except Exception as e:
                import traceback; traceback.print_exc()
                print('[DataCollector] loop error:', e)

            next_t += self.sampling_interval
            now = time.monotonic()
            if next_t < now: # fell behind — skip missed ticks
                next_t = now
            time.sleep(max(0.0, next_t - now))

        print("data collector stopped!")

    def _poll_vehicle_state(self, interval: float = 0.5) -> None:
        """Background thread: fetch vehicle state from the bridge every `interval` seconds."""
        while self._running:
            try:
                with urllib.request.urlopen(self.vehicle_state_url, timeout=2.0) as resp:
                    state = json.loads(resp.read())
                self._cached_speed               = int(state.get("speed_kmh", self._cached_speed))
                self._cached_brake               = float(state.get("brake", self._cached_brake))
                self._cached_steer               = float(state.get("steer", self._cached_steer))
                self._cached_speed_limit         = float(state.get("speed_limit_kmh", self._cached_speed_limit))
                self._cached_precipitation       = float(state.get("precipitation", self._cached_precipitation))
                self._cached_night               = bool(state.get("is_night", self._cached_night))
                self._cached_junction            = bool(state.get("is_junction", self._cached_junction))
                self._cached_throttle            = float(state.get("throttle", self._cached_throttle))
                self._cached_gear                = int(state.get("gear", self._cached_gear))
                self._cached_hand_brake          = bool(state.get("hand_brake", self._cached_hand_brake))
                self._cached_reverse             = bool(state.get("reverse", self._cached_reverse))
                self._cached_acceleration        = float(state.get("acceleration", self._cached_acceleration))
                self._cached_fog_density         = float(state.get("fog_density", self._cached_fog_density))
                self._cached_traffic_light_state = state.get("traffic_light_state", self._cached_traffic_light_state)
                self._cached_headlight           = bool(state.get("headlight", self._cached_headlight))
                self._cached_fog_light           = bool(state.get("fog_light", self._cached_fog_light))
                self._cached_left_indicator      = bool(state.get("left_indicator", self._cached_left_indicator))
                self._cached_right_indicator     = bool(state.get("right_indicator", self._cached_right_indicator))
            except Exception:
                pass  # keep using last cached values on any error
            time.sleep(interval)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if self.vehicle_state_url:
            self._state_poll_thread = threading.Thread(target=self._poll_vehicle_state, daemon=True)
            self._state_poll_thread.start()
            print(f"[DataCollector] Vehicle state polling started → {self.vehicle_state_url}")

    def stop(self) -> None:
        # Halt the loop BEFORE stopping the rPPG estimator / releasing the
        # camera, so no tick can touch an already-stopped resource.
        self._running = False
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            if self._state_poll_thread and self._state_poll_thread.is_alive():
                self._state_poll_thread.join(timeout=2.0)
        except Exception:
            pass
        if self.rppg_estimator is not None:
            try:
                self.rppg_estimator.stop()
            except Exception as e:
                print(e, "Error stopping rPPG estimator")
        self.release()

    def release(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception as e:
            print(e, "Error releasing camera")
            pass
        self.cap = None

    def get_latest_data(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.latest_data)

    def get_latest_frame(self) -> Optional[str]:
        frame = None
        with self._lock:
            frame = self.latest_frame
        if frame is None or not HAS_CV2:
            return None
        try:
            _, buffer = cv2.imencode('.jpg', frame)  # type: ignore
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(e, "Error encoding latest frame")
            return None

    def get_latest(self) -> Dict[str, Any]:
        with self._lock:
            data = dict(self.latest_data)
            frame = self.latest_frame
        img_b64 = None
        if frame is not None and HAS_CV2:
            try:
                _, buffer = cv2.imencode('.jpg', frame)  # type: ignore
                img_b64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(e, "Error encoding latest frame")
                img_b64 = None
        return {"data": data, "frame_b64": img_b64}
