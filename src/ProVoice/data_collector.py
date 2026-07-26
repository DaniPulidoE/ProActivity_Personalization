from __future__ import annotations

import base64
import datetime
import os
import re
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
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from rPPG.rppg_infer_simple import OnlineRPPG
from ProVoice import perception as _perception  # in-tree replacement for yolov5-deepsort

HAS_CV2 = True
HAS_MYFRAME = True  # kept for backward-compat
HAS_RPPG = True
HAS_NP = True
HAS_MP = True

# ── Emotion: EmotiEffLib (maintained successor to HSEmotion) ──────────────────
try:
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer  # type: ignore
    HAS_EMOTIEFFLIB = True
except Exception as e:
    print(e, "Error importing EmotiEffLib; emotion detection disabled")
    EmotiEffLibRecognizer = None  # type: ignore
    HAS_EMOTIEFFLIB = False

_emotion_recognizer = None
_EMOTIEFFLIB_MODEL = "enet_b0_8_best_afew"
# EmotiEffLib emits 8 AffectNet classes; map to the pipeline's 7-class
# EMOTION_VOCAB so D_IN stays 40 and existing xLSTM checkpoints keep loading.
# 'contempt' has no FER2013 counterpart -> mapped to neutral (most conservative).
_EMOTIEFFLIB_TO_VOCAB = {
    'anger': 'angry', 'contempt': 'neutral', 'disgust': 'disgust', 'fear': 'fear',
    'happiness': 'happy', 'neutral': 'neutral', 'sadness': 'sad', 'surprise': 'surprise',
}


def _load_emotion_recognizer() -> None:
    """Load the EmotiEffLib ONNX recognizer once (downloads weights on first use)."""
    global _emotion_recognizer
    if not HAS_EMOTIEFFLIB or _emotion_recognizer is not None:
        return
    try:
        _emotion_recognizer = EmotiEffLibRecognizer(  # type: ignore
            engine="onnx", model_name=_EMOTIEFFLIB_MODEL, device="cpu")
        print(f"[emotion] EmotiEffLib '{_EMOTIEFFLIB_MODEL}' loaded successfully.")
    except Exception as e:
        print(e, "Error loading EmotiEffLib; emotion detection disabled")
        _emotion_recognizer = None


# ── Face landmarks: MediaPipe Tasks FaceLandmarker (maintained API) ───────────
_FACE_LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                        "face_landmarker/float16/1/face_landmarker.task")


def _resolve_face_landmarker_model() -> Optional[str]:
    """Return a local path to face_landmarker.task, downloading it once if absent."""
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    p = os.path.join(d, 'face_landmarker.task')
    if os.path.exists(p):
        return p
    try:
        urllib.request.urlretrieve(_FACE_LANDMARKER_URL, p)
        print(f"[face_landmarker] downloaded model -> {p}")
        return p
    except Exception as e:
        print(e, "Error downloading face_landmarker.task; face landmarks disabled")
        return None


# ── Per-driver calibration persistence ───────────────────────────────────────
# Baselines are stable across sessions for the same driver, so a stored file
# lets a returning participant skip the 60 s routine. Loading is only allowed
# when the file carries EVERY field the serving path reads — see
# _validate_calibration(); anything missing/degenerate falls back to a live
# calibration rather than silently serving zeros.
_CALIBRATION_DIR = os.path.join('.', 'data', 'calibration_data')
# key -> required numeric fields (mirrors what compute_calibration() writes)
_CALIBRATION_SCHEMA: Dict[str, Tuple[str, ...]] = {
    'gaze_score': ('mean', 'std', 'threshold'),
    'ear':        ('mean', 'std', 'threshold'),
    'mar':        ('mean', 'std', 'threshold'),
    'bpm':        ('mean', 'std', 'threshold'),
    'rr':         ('mean', 'std', 'threshold'),
    'blink_rate': ('mean',),
    'perclos':    ('mean', 'std'),
}


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
        calibration_dir: str = _CALIBRATION_DIR,
        rppg_fps: float = 30.0,
        face_box_hz: float = 5.0,
    ) -> None:
        # Everything stop()/__del__ touches is set FIRST, so a constructor
        # that fails part-way still leaves a safely destructible object.
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state_poll_thread: Optional[threading.Thread] = None
        self._yolo_thread: Optional[threading.Thread] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._face_box_thread: Optional[threading.Thread] = None
        self.rppg_estimator = None
        self.cap = None

        # ── Camera ownership ──────────────────────────────────────────────────
        # The capture worker is the ONLY caller of cap.read(); every other
        # consumer (main loop, YOLO26, face-box worker) reads the frame it
        # publishes here. cv2.VideoCapture cannot be read from two threads, and
        # rPPG needs UNIFORMLY sampled frames at ~30 fps while the main decision
        # loop runs an order of magnitude slower and with heavy jitter — so the
        # two cadences have to be decoupled.
        self._cap_lock = threading.Lock()
        self._cap_frame = None            # latest raw BGR frame
        self._cap_frame_id: int = 0       # bumped per frame; consumers skip stale
        self._cap_started: bool = False   # first successful read happened
        self._rppg_fps_target: float = float(rppg_fps)
        self._rppg_fps: float = float(rppg_fps)   # overwritten with the achieved rate
        self._face_box_interval: float = 1.0 / max(1e-3, float(face_box_hz))

        # YOLO distraction runs OFF the main loop (its ~45 ms inference is the
        # loop's biggest cost). The worker reads the latest raw frame and writes
        # the label list; the main loop reads that list. All shared under a lock.
        self._yolo_lock = threading.Lock()
        self._yolo_frame = None          # latest raw BGR frame (main writes)
        self._yolo_frame_id: int = 0     # bumped each new frame; worker skips stale
        self._yolo_lab: list = []        # latest distraction labels (worker writes)
        self._yolo_dets: list = []       # latest detection boxes (worker writes) for the overlay
        self._distraction = None         # DistractionDetector, created in the worker

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

        self.face_landmarker = None
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
                # Request the capture format MMRPhys's own demos use. The driver
                # may refuse any of these, so read back what it actually granted:
                # the achieved rate becomes the rPPG sampling rate (it sets both
                # the Butterworth cutoffs and the FFT frequency axis, so getting
                # it wrong scales every HR/RR estimate).
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # type: ignore
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)   # type: ignore
                self.cap.set(cv2.CAP_PROP_FPS, self._rppg_fps_target)  # type: ignore
                _fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)    # type: ignore
                if not (5.0 < _fps < 240.0):   # driver reported nothing usable
                    _fps = self._rppg_fps_target
                self._rppg_fps = _fps
                print(f"[DataCollector] Capture: "
                      f"{self.cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"      # type: ignore
                      f"{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "     # type: ignore
                      f"@ {self._rppg_fps:.1f} fps (rPPG sampling rate)")
            except Exception as e:
                print(e, "Error opening camera")
                self.cap = None
                self.visual_enabled = False
            if HAS_MP:
                try:
                    print("[DataCollector] Initializing MediaPipe Tasks FaceLandmarker...")
                    _model = _resolve_face_landmarker_model()
                    if _model:
                        _opts = mp_vision.FaceLandmarkerOptions(
                            base_options=mp_python.BaseOptions(model_asset_path=_model),
                            running_mode=mp_vision.RunningMode.IMAGE,
                            num_faces=1,
                            output_face_blendshapes=False,
                            output_facial_transformation_matrixes=False,
                        )
                        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(_opts)
                        print("[DataCollector] FaceLandmarker loaded successfully.")
                except Exception as e:
                    print(e, "Error initializing FaceLandmarker")
                    self.face_landmarker = None

            if HAS_MYFRAME:
                print("[DataCollector] Distraction perception ready (Ultralytics YOLO26, decoupled thread).")

            if HAS_RPPG and OnlineRPPG is not None:
                try:
                    # crop_size=72 selects MMRPhysLEF, matching the x180x72
                    # checkpoint OnlineRPPG loads. frame_rate is the ACHIEVED
                    # capture rate, not a nominal one.
                    self.rppg_estimator = OnlineRPPG(frame_rate=int(round(self._rppg_fps)),
                                                     crop_size=72)  # type: ignore
                except Exception as e:
                    print(e, "Error initializing rPPG estimator")
                    self.rppg_estimator = None
                    #raise e

        _load_emotion_recognizer()

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
        # rPPG face crop: MMRPhys is trained on a SQUARED face-detector box
        # enlarged by LARGE_BOX_COEF=1.5 (see _rppg_crop_box for the exact
        # reproduction), then EMA-smoothed here to kill the per-frame crop jitter
        # that corrupts the pulse signal (rPPG is spectral, jitter-sensitive).
        # EMA rather than Kalman: the head moves slowly, so a velocity model buys
        # nothing and EMA mirrors the toolbox's own near-static box. Both tunable.
        self._rppg_box_coef: float = 1.5                        # = LARGE_BOX_COEF
        self._rppg_box_alpha: float = 0.25                      # EMA weight (lower = smoother)
        self._rppg_box_ema: Optional[np.ndarray] = None         # smoothed [cx, cy, w, h]
        # Shared square crop box, written by the face-box worker and read by the
        # capture worker at full frame rate (see _face_box_loop / _capture_loop).
        self._box_lock = threading.Lock()
        self._rppg_box: Optional[Tuple[int, int, int, int]] = None   # (x, y, w, h)
        self._rppg_box_t: float = 0.0                                # monotonic stamp
        self._rppg_box_staleness_s: float = 2.0                      # drop the box after this
        self._box_landmarker = None                                  # worker-owned detector
        # rPPG results, published by the capture worker, read by the main loop.
        self._rppg_lock = threading.Lock()
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
        # Timestamps of the last rPPG readings folded into the baseline, so each
        # estimate is sampled exactly once (see calibrate_step).
        self._cal_last_hr_t: float = 0.0
        self._cal_last_rr_t: float = 0.0
        self._calibration_ear_ts = [] # timestamps of EAR values to callibrate blink rate/perclos
        self._calibration_mar_ts = [] # timestamps of MAR values to callibrate perclos

        # Calibration: 60 s time-based, collects gaze/EAR/MAR/HR/RR
        self._calibration_duration_s: float = 60.0
        self._calibration_start_t: float = 0.0
        # Persisted per-driver baselines: the loop tries the stored file ONCE
        # before starting the live routine (see _run_loop).
        self.calibration_dir = calibration_dir or _CALIBRATION_DIR
        self._calibration_load_attempted = False
        # Transient camera-read failures (warm-up frames, momentary USB glitches)
        # must NOT abort calibration; only give up after this many seconds of
        # UNBROKEN failures. Resets on any successful read.
        self._read_fail_since: float = 0.0
        self._read_fail_grace_s: float = 5.0

        # --- DEBUG: frame-rate instrumentation (safe to delete) ---
        self._dbg_last_tick_t: float = 0.0
        self._dbg_last_print_t: float = 0.0
        self._dbg_dt_hist: deque = deque(maxlen=100)

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass  # never raise during interpreter teardown

    def detect_emotion(self, face_rgb) -> Optional[Dict[str, Any]]:
        """Classify emotion on an RGB face crop via EmotiEffLib.

        Returns the label mapped to the pipeline's 7-class EMOTION_VOCAB plus the
        top-class probability, or None when the recognizer or crop is missing.
        """
        if _emotion_recognizer is None or face_rgb is None or getattr(face_rgb, 'size', 0) == 0:
            return None
        try:
            emos, scores = _emotion_recognizer.predict_emotions(face_rgb, logits=False)
            raw = str(emos[0]).strip().lower()
            label = _EMOTIEFFLIB_TO_VOCAB.get(raw, 'neutral')
            conf = float(np.max(np.asarray(scores)))
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

    
    def get_gaze_score(self, frame, landmarks) -> float:
        has_face = bool(landmarks)
        if not has_face:
            # No face detected. Post-calibration, fall back to the per-driver
            # gaze baseline so the standardized score is neutral (zero deviation)
            # rather than a spurious value. During calibration the baseline does
            # not exist yet and the caller filters on >0.0 to skip the frame, so
            # keep returning 0.0 there.
            if self.calibrated:
                return float(self.calibrate.get('gaze_score', {}).get('mean', 0.0))
            return 0.0
        try:
            h, w, _ = frame.shape
            return self.compute_gaze_score(landmarks, w, h)
        except Exception as e:
            print(e, "Error computing gaze score")
            return 0.0

    def compute_face_landmarks(self, frame):
        """Return the list of 478 normalized face landmarks (or None) via the
        maintained MediaPipe Tasks FaceLandmarker. Each landmark exposes .x/.y/.z
        in the same canonical indexing as the legacy solutions FaceMesh, so gaze,
        EAR/MAR, and the bounding box are computed identically."""
        if self.face_landmarker is None or not HAS_MP:
            return None
        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # type: ignore
            res = self.face_landmarker.detect(mp_img)
            if res.face_landmarks:
                return res.face_landmarks[0]
            return None
        except Exception as e:
            print(e, "Error computing face landmarks")
            return None

    @staticmethod
    def _bbox_from_landmarks(landmarks, w: int, h: int, margin: float = 0.10):
        """Axis-aligned face box (x, y, bw, bh) in pixels from normalized
        landmarks — replaces the Haar cascade crop for rPPG and emotion. Reuses
        the landmarks already computed for gaze/EAR/MAR, so it costs ~nothing."""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        mx, my = margin * (x1 - x0), margin * (y1 - y0)
        X = max(0, int((x0 - mx) * w))
        Y = max(0, int((y0 - my) * h))
        X2 = min(w, int((x1 + mx) * w))
        Y2 = min(h, int((y1 + my) * h))
        return X, Y, max(0, X2 - X), max(0, Y2 - Y)

    def _rppg_crop_box(self, landmarks, w: int, h: int):
        """SQUARE, enlarged + EMA-smoothed face box (x, y, bw, bh) for rPPG.

        Reproduces the training-time crop geometry of MMRPhys exactly
        (BaseLoader.face_detection + crop_face_resize):

          1. take the face box,
          2. SQUARE it: ``side = max(w, h)``, re-centred on the box centre,
          3. enlarge by ``_rppg_box_coef`` (= LARGE_BOX_COEF = 1.5) about that
             same centre, so the crop takes in forehead and cheeks.

        Squaring matters: the crop is later resized to 72x72, so a non-square box
        anisotropically stretches the face relative to every frame the model was
        trained on. The source box here is the MediaPipe landmark hull rather than
        YOLO5Face/WiderFace, which is the one remaining approximation.

        The EMA then low-pass filters centre and size to damp the per-frame
        min/max jitter that would otherwise smear the pulse spectrum (rPPG is a
        spectral estimate, so crop jitter is noise). Clamped to the frame.
        """
        x, y, bw, bh = self._bbox_from_landmarks(landmarks, w, h, margin=0.0)
        cx, cy = x + bw / 2.0, y + bh / 2.0
        side = max(bw, bh) * self._rppg_box_coef
        box = np.array([cx, cy, side, side], dtype=np.float32)
        if self._rppg_box_ema is None:
            self._rppg_box_ema = box
        else:
            a = self._rppg_box_alpha
            self._rppg_box_ema = a * box + (1.0 - a) * self._rppg_box_ema
        scx, scy, sw, sh = self._rppg_box_ema
        X = max(0, int(round(scx - sw / 2.0)))
        Y = max(0, int(round(scy - sh / 2.0)))
        X2 = min(w, int(round(scx + sw / 2.0)))
        Y2 = min(h, int(round(scy + sh / 2.0)))
        return X, Y, max(0, X2 - X), max(0, Y2 - Y)

    def calibrate_step(self) -> None:
        if not self.visual_enabled or self.cap is None:
            # No camera at all — no baseline is obtainable; finish immediately
            # with defaults so the context/CARLA pipeline can proceed.
            print("[Calibration] Visual disabled or camera unavailable — "
                  "finishing calibration with default baselines.")
            self.compute_calibration()
            return
        # Frames come from the capture worker, which is already feeding rPPG at
        # full rate (see _capture_loop) — calibration must NOT enqueue frames of
        # its own or it would interleave duplicates into the 181-frame window.
        frame = self._get_capture_frame()
        ok = frame is not None
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
        h_img, w_img = frame.shape[:2]
        landmarks = self.compute_face_landmarks(frame)
        face_present = bool(landmarks)

        gaze_score = self.get_gaze_score(frame, landmarks)
        if gaze_score > 0.0:
            self._calibration_data['gaze_score'].append(gaze_score)

        # EAR/MAR from the SAME landmarks (only when a face is actually present).
        if face_present:
            try:
                eye, mouth = _perception.eye_mouth_aspect_ratios(landmarks, w_img, h_img)
                self._calibration_data['ear'].append(eye)
                self._calibration_data['mar'].append(mouth)
                self._calibration_ear_ts.append((now, eye))
                self._calibration_mar_ts.append((now, mouth))
            except Exception as e:  # noqa: BLE001
                print(e, "Error computing EAR/MAR")

        # Sample the rPPG readings the capture worker publishes. Keyed on the
        # reading TIMESTAMP so each estimate is counted once — polling the held
        # value every tick would append the same number over and over (a reading
        # lands every 6 s, the loop ticks 20x/s) and collapse the baseline std.
        if self.rppg_estimator is not None:
            with self._rppg_lock:
                last_hr, last_hr_t = self._last_hr, self._last_hr_t
                last_rr, last_rr_t = self._last_rr, self._last_rr_t
            if last_hr is not None and last_hr_t > self._cal_last_hr_t:
                self._cal_last_hr_t = last_hr_t
                self._calibration_data['bpm'].append(float(last_hr))
                print(f"[Calibration] rPPG: HR={last_hr}")
            if last_rr is not None and last_rr_t > self._cal_last_rr_t:
                self._cal_last_rr_t = last_rr_t
                self._calibration_data['rr'].append(float(last_rr))
                print(f"[Calibration] rPPG: RR={last_rr}")

        self.latest_frame = frame
        #print(f"[Calibration] {now - self._calibration_start_t:.1f}/{self._calibration_duration_s:.1f} s elapsed. ")
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
                self.calibrate[key] = {'mean': mean, 'std': std, 'threshold': mean + std * 2.5 if key in ['gaze_score'] else max(0.15, mean * 0.75) if key in ['ear'] else 0.55}
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

        self.save_calibration()

    # ── Calibration persistence ──────────────────────────────────────────────
    def calibration_file(self) -> Optional[str]:
        """Path of this participant's stored calibration, or None without an ID.

        The participant ID reaches us through ``static_context`` (set by
        ``main.py`` from ``--participantid``); it is sanitized because it ends
        up in a filename.
        """
        pid = str(self.static_context.get('participantid') or '').strip()
        if not pid:
            return None
        safe_pid = re.sub(r'[^A-Za-z0-9_.-]', '_', pid)
        return os.path.join(self.calibration_dir, f"calibration_{safe_pid}.json")

    @staticmethod
    def _validate_calibration(cal: Any) -> Optional[str]:
        """Return None when ``cal`` is usable, else why it is not.

        Guards the serving path, which indexes ``self.calibrate`` directly
        (``calibrate['gaze_score']['std']``, ``calibrate['ear']['threshold']``,
        …): every key/field of the schema must be present and hold a finite,
        non-boolean number. Beyond mere presence we reject the degenerate file
        ``compute_calibration()`` writes when no face was ever seen (zero
        gaze/EAR/MAR means, zero PERCLOS std) — that carries no per-driver
        information, so a live calibration is strictly better. bpm/rr means may
        legitimately be 0.0 (rPPG can fail to lock) and stay allowed.
        """
        if not isinstance(cal, dict):
            return f"expected a JSON object, got {type(cal).__name__}"
        for key, fields in _CALIBRATION_SCHEMA.items():
            entry = cal.get(key)
            if not isinstance(entry, dict):
                return f"missing or malformed '{key}' entry"
            for field in fields:
                v = entry.get(field)
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    return f"'{key}.{field}' is not a number ({v!r})"
                if not math.isfinite(float(v)):
                    return f"'{key}.{field}' is not finite ({v!r})"
        for key in ('gaze_score', 'ear', 'mar'):
            if float(cal[key]['threshold']) <= 0.0:
                return f"'{key}.threshold' is not positive ({cal[key]['threshold']!r})"
            if float(cal[key]['mean']) <= 0.0:
                return (f"'{key}.mean' is {cal[key]['mean']!r} — no face samples "
                        f"were collected for this driver")
        if float(cal['perclos']['std']) <= 0.0:
            return f"'perclos.std' is not positive ({cal['perclos']['std']!r})"
        return None

    def load_calibration(self) -> bool:
        """Adopt this participant's stored baselines, skipping the 60 s routine.

        Returns True only when a well-formed file was found and applied; on any
        other outcome the caller runs the live calibration instead.
        """
        path = self.calibration_file()
        if path is None:
            print("[Calibration] No participantid in context — cannot reuse a "
                  "stored calibration; running the 60 s calibration.")
            return False
        if not os.path.exists(path):
            print(f"[Calibration] No stored calibration at {path} — "
                  f"running the 60 s calibration.")
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cal = json.load(f)
        except Exception as e:  # unreadable / invalid JSON
            print(e, f"Error reading {path}; running the 60 s calibration")
            return False
        reason = self._validate_calibration(cal)
        if reason is not None:
            print(f"[Calibration] Ignoring {path} ({reason}) — "
                  f"running the 60 s calibration.")
            return False

        # Copy field-by-field as floats: keeps `calibrate` exactly the shape the
        # live routine produces (no stray keys from an older file leaking in).
        self.calibrate = {
            key: {field: float(cal[key][field]) for field in fields}
            for key, fields in _CALIBRATION_SCHEMA.items()
        }
        self.calibrated = True
        self._session_start_t = time.monotonic()
        print(f"[Calibration] Loaded calibration from {path} — "
              f"skipping the 60 s calibration.")
        print("Calibration loaded:", self.calibrate)
        return True

    def save_calibration(self) -> Optional[str]:
        """Persist the freshly computed baselines for this participant."""
        path = self.calibration_file()
        if path is None:
            print("[Calibration] No participantid in context — calibration not saved.")
            return None
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.calibrate, f, indent=4)
        except Exception as e:  # a failed save must never abort the session
            print(e, f"Error saving calibration to {path}")
            return None
        print(f"[Calibration] Saved calibration to {path}")
        return path


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

    def _visual_process(self, data: Dict[str, Any]) -> bool:
        """Populate ``data`` with visual driver-state features.

        Returns ``False`` only when the camera read itself failed (a transient
        hardware glitch) so the caller can skip the whole tick instead of
        emitting a frame with no driver-state features. Returns ``True``
        otherwise — including when visual is disabled, where there is simply
        nothing to read.
        """
        if not self.visual_enabled or self.cap is None:
            return True
        # Frames come from the capture worker (sole owner of cap.read()); this
        # loop takes whatever is newest. It may occasionally see the same frame
        # twice when it out-runs the camera — harmless, since blink/yawn/PERCLOS
        # are all gated on wall-clock time rather than frame counts.
        frame = self._get_capture_frame()
        if frame is None:
            self.latest_frame = None
            return False

        # MediaPipe Tasks FaceLandmarker — the SINGLE face detector for the whole
        # pipeline: gaze, EAR/MAR, and the rPPG/emotion crop box all key off these
        # landmarks. Replaces both the legacy solutions FaceMesh and the separate
        # Haar cascade, so there is no longer a second detector to disagree with.
        h_img, w_img = frame.shape[:2]
        landmarks = self.compute_face_landmarks(frame)
        face_present = bool(landmarks)
        data['face_present'] = face_present

        # Publish the raw frame for the decoupled YOLO worker (see _yolo_loop).
        with self._yolo_lock:
            self._yolo_frame = frame
            self._yolo_frame_id += 1

        # gaze score — get_gaze_score returns the per-driver baseline when no face
        # is present, so the standardized value below is neutral (0.0 deviation).
        gaze_score = self.get_gaze_score(frame, landmarks)
        normalized_gaze_score = (gaze_score - self.calibrate['gaze_score']['mean']) / self.calibrate['gaze_score']['std'] if self.calibrate['gaze_score']['std'] != 0 else 0.0
        data['gaze_score'] = round(normalized_gaze_score, 3)
        data['gaze_distracted'] = bool(gaze_score > self.calibrate['gaze_score']['threshold'])

        # Emotion crop from the landmarks (no Haar): a tight, roughly MTCNN-style
        # box, which is what EmotiEffLib/AffectNet expects. The rPPG crop is NOT
        # taken here — the capture worker cuts it from every frame at full rate
        # using the box published by _face_box_loop.
        emotion_crop_bgr = None
        if face_present:
            ex, ey, ew, eh = self._bbox_from_landmarks(landmarks, w_img, h_img, margin=0.10)
            ec = frame[ey:ey + eh, ex:ex + ew]
            if ec.size > 0:
                emotion_crop_bgr = ec

        if self.rppg_estimator is not None:  # type: ignore
            # rPPG readings are produced by the capture worker (a fresh value
            # every inference_interval frames, None in between). Read the held
            # values and carry them forward while fresh, so most frames carry a
            # heart rate instead of a null.
            now = time.monotonic()
            with self._rppg_lock:
                last_hr, last_hr_t = self._last_hr, self._last_hr_t
                last_rr, last_rr_t = self._last_rr, self._last_rr_t
                bpm_hist = list(self.bpm_history)
                rr_hist = list(self.rr_history)
            if last_hr is not None and (now - last_hr_t) <= self._rppg_staleness_s:
                data['heart_rate'] = last_hr
                # snapshot as list: deque is not JSON-serializable (dashboard
                # emit), and a live reference would mutate under stored frames
                data['bpm_history'] = bpm_hist
                data['hr_delta'] = self._standardized_delta(last_hr, 'bpm', bpm_hist)
            if last_rr is not None and (now - last_rr_t) <= self._rppg_staleness_s:
                data['respiratory_rate'] = last_rr
                data['rr_history'] = rr_hist
                data['rr_delta'] = self._standardized_delta(last_rr, 'rr', rr_hist)

        # Emotion — EmotiEffLib on the RGB tight face crop.
        if emotion_crop_bgr is not None:
            emo = self.detect_emotion(cv2.cvtColor(emotion_crop_bgr, cv2.COLOR_BGR2RGB))
            if emo:
                data.update(emo)  # emotion, emotion_prob

        # EAR/MAR from the landmarks (sentinels when no face, as before).
        if face_present:
            try:
                eye, mouth = _perception.eye_mouth_aspect_ratios(landmarks, w_img, h_img)
            except Exception as e:  # noqa: BLE001
                print(e, "Error computing EAR/MAR")
                eye, mouth = 0.3, 0.2
        else:
            eye, mouth = 0.3, 0.2

        # Distraction labels + detection boxes from the decoupled YOLO worker.
        with self._yolo_lock:
            lab = list(self._yolo_lab)
            yolo_dets = list(self._yolo_dets)
        if face_present and "face" not in lab:
            lab.insert(0, "face")

        # Dashboard overlay: eye/mouth boxes (landmarks) + object boxes (YOLO).
        # Drawn on a copy — the raw frame published to the YOLO worker is untouched.
        frame_annot = _perception.draw_overlays(frame, landmarks, yolo_dets)

        MOUTH_AR_CONSEC_FRAMES = 3
        BLINK_MIN_MS = 100   # below this is EAR noise, not a blink (this is equivalent to 2 frames at 20Hz)
        BLINK_MAX_MS = 500  # above this is eye closure / drowsiness, not a blink
        blink_window = 30.0  # 30 second window for blinks
        yawn_window = 180    # 3 minute window for yawns

        now = time.monotonic()
        eye_closed = eye < self.calibrate['ear']['threshold']
        mouth_open = mouth > self.calibrate['mar']['threshold']
        # Time-only, so valid whether or not a face is visible this frame; drives
        # the blink-rate std in the normalization below.
        elapsed_s = min(now - self._session_start_t, blink_window)

        if face_present:
            # blink detection (duration-gated)
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

            # yawn detection (consecutive-frame gated)
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

            # yawn and blink rates (over their respective time windows)
            self.blink_times = [t for t in self.blink_times if now - t <= blink_window]
            self.blink_rate = len(self.blink_times) / (elapsed_s / 60) if elapsed_s > 0 else 0.0  # blinks per minute
            self.yawn_times = [t for t in self.yawn_times if now - t <= yawn_window]
            self.yawn_rate = len(self.yawn_times) / (yawn_window / 60)  # yawns per minute
        else:
            # Face absent → not a driver-state measurement. Leave the blink/yawn/
            # PERCLOS buffers untouched so self.perclos, self.blink_rate and
            # self.yawn_rate keep their last face-present values (carried
            # forward). Reset the in-progress blink/yawn timers so a detection
            # cannot span the face-absence gap and fire as a false event.
            self._eye_closed_since = 0.0
            self.mCOUNTER = 0


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
        return True

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

    def collect_data(self) -> Tuple[Dict[str, Any], bool]:
        """Collect one multimodal frame.

        Returns ``(data, read_ok)``. ``read_ok`` is ``False`` when the camera
        read failed this tick, in which case ``data`` is incomplete and the
        caller should skip the tick rather than log/decide on a driver-state-less
        frame. ``latest_data`` is left untouched on failure so the dashboard
        keeps showing the last good frame.
        """
        data: Dict[str, Any] = {}

        # Declare the physiological keys up front so heart_rate/respiratory_rate
        # (and their deltas) always occupy the SAME position in every logged
        # frame. When a fresh rPPG reading exists, _visual_process overwrites
        # these in place (updating an existing dict key preserves its insertion
        # order); when it doesn't, they stay None here — reported as unknown
        # rather than fabricated. Without this, a fresh reading was inserted
        # inside _visual_process (early) while a null was inserted later in a
        # phys fallback, so the field jumped position between frames.
        if self.phys_enabled:
            data['heart_rate'] = None
            data['hr_delta'] = None
            data['respiratory_rate'] = None
            data['rr_delta'] = None

        if self.visual_enabled:
            if not self._visual_process(data):
                return data, False

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

        return data, True


    def _run_loop(self) -> None:
        # Old: frame-count calibration counter
        # calibration_counter = 0

        next_t = time.monotonic()
        while self._running:
            try:
                if not self.calibrated and not self._calibration_load_attempted:
                    # Once per loop start: reuse this driver's stored baselines
                    # if a valid file exists, otherwise fall through to the live
                    # routine below.
                    self._calibration_load_attempted = True
                    self.load_calibration()

                if self.calibrated is False:
                    # Two-phase time-based calibration — completion handled inside calibrate_step()
                    self.calibrate_step()
                else:
                    _dbg_t0 = time.monotonic()  # DEBUG fps (safe to delete)
                    data, read_ok = self.collect_data()
                    # --- DEBUG: achieved frame interval + collect() cost (safe to delete) ---
                    _dbg_collect_ms = (time.monotonic() - _dbg_t0) * 1000.0
                    if self._dbg_last_tick_t:
                        _dbg_dt_ms = (_dbg_t0 - self._dbg_last_tick_t) * 1000.0
                        self._dbg_dt_hist.append(_dbg_dt_ms)
                        _dbg_avg = sum(self._dbg_dt_hist) / len(self._dbg_dt_hist)
                        data['frame_dt_ms'] = round(_dbg_dt_ms, 1)       # ms since previous tick
                        data['collect_ms']  = round(_dbg_collect_ms, 1)  # perception cost this tick
                        data['fps_inst']    = round(1000.0 / _dbg_dt_ms, 2) if _dbg_dt_ms > 0 else 0.0
                        data['fps_avg']     = round(1000.0 / _dbg_avg, 2) if _dbg_avg > 0 else 0.0
                        if _dbg_t0 - self._dbg_last_print_t >= 1.0:  # throttle console to ~1 Hz
                            self._dbg_last_print_t = _dbg_t0
                            print(f"[DataCollector][fps] dt={_dbg_dt_ms:6.1f}ms "
                                  f"collect={_dbg_collect_ms:6.1f}ms "
                                  f"fps_inst={data['fps_inst']:5.2f} fps_avg={data['fps_avg']:5.2f} "
                                  f"face={data.get('face_present')}")
                    self._dbg_last_tick_t = _dbg_t0
                    # --- end DEBUG ---

                    if not read_ok:
                        # Transient camera read failure: skip the whole tick — no
                        # sequence entry, no log, no decision/actuation — so we
                        # never emit a zero-driver-state frame. Stay quiet for a
                        # short grace window (a momentary USB glitch), then warn
                        # once the failures persist (mirrors calibrate_step).
                        now_m = time.monotonic()
                        if self._read_fail_since == 0.0:
                            self._read_fail_since = now_m
                        failed_for = now_m - self._read_fail_since
                        if failed_for >= self._read_fail_grace_s:
                            print(f"[DataCollector] Camera read failing for "
                                  f"{failed_for:.1f}s (>{self._read_fail_grace_s:.0f}s grace) — "
                                  f"camera likely disconnected; skipping ticks.")
                    else:
                        self._read_fail_since = 0.0  # good read — reset the timer

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

    def _get_capture_frame(self):
        """Latest frame published by the capture worker, or None before the first
        successful read. Consumers get whatever is newest — only the rPPG path
        (inside the capture worker itself) needs every frame."""
        with self._cap_lock:
            return self._cap_frame

    def _capture_loop(self) -> None:
        """Background thread: sole owner of the camera.

        Reads at the camera's native cadence (``cap.read()`` blocks until the
        next frame, so the device clock — not a sleep — paces this loop and Δt
        stays uniform), publishes each frame for the other consumers, and feeds
        EVERY frame to the rPPG estimator using the box maintained by
        ``_face_box_loop``.

        Uniform sampling is the point: the FFT that turns the BVP/RSP waveform
        into HR/RR assumes a constant Δt, and the main decision loop's interval
        swings with MediaPipe/emotion/decision-engine cost (it even skips missed
        ticks). Sampling rPPG off that loop smeared the pulse peak.
        """
        print(f"[DataCollector] Capture worker started "
              f"(target {self._rppg_fps:.1f} fps).")
        fail_since = 0.0
        n_frames = 0
        t_first = 0.0
        while self._running:
            cap = self.cap
            if cap is None:
                break
            try:
                ok, frame = cap.read()
            except Exception as e:  # noqa: BLE001 — camera yanked mid-read
                print(e, "Error reading camera in capture worker")
                ok, frame = False, None
            now = time.monotonic()
            if not ok or frame is None:
                # Publish the failure by clearing the frame so consumers skip
                # their tick, but keep retrying: the grace-window warning is the
                # main loop's job (it owns _read_fail_since).
                if fail_since == 0.0:
                    fail_since = now
                with self._cap_lock:
                    self._cap_frame = None
                time.sleep(0.01)
                continue
            fail_since = 0.0

            with self._cap_lock:
                self._cap_frame = frame
                self._cap_frame_id += 1
                self._cap_started = True

            # --- rPPG: every frame, no drops ---------------------------------
            if self.rppg_estimator is not None:
                with self._box_lock:
                    box = self._rppg_box
                    box_t = self._rppg_box_t
                if box is not None and (now - box_t) <= self._rppg_box_staleness_s:
                    x, y, bw, bh = box
                    crop = frame[y:y + bh, x:x + bw]
                    if crop.size > 0:
                        try:
                            hr, rr = self.rppg_estimator.add_frame(crop)
                        except Exception as e:  # noqa: BLE001
                            print(e, "Error feeding rPPG estimator")
                            hr = rr = None
                        # A reading surfaces only every inference_interval frames;
                        # None in between. Publish for the main loop to read.
                        if hr is not None and not (hr != hr):   # excludes NaN
                            with self._rppg_lock:
                                self._last_hr = round(float(hr), 1)
                                self._last_hr_t = now
                                self.bpm_history.append(self._last_hr)
                        if rr is not None and not (rr != rr):
                            with self._rppg_lock:
                                self._last_rr = round(float(rr), 1)
                                self._last_rr_t = now
                                self.rr_history.append(self._last_rr)

            # Achieved-rate telemetry: the configured filters/FFT assume
            # self._rppg_fps, so a large drift here invalidates HR/RR.
            n_frames += 1
            if t_first == 0.0:
                t_first = now
            elif n_frames % 300 == 0:
                measured = (n_frames - 1) / max(1e-6, now - t_first)
                if abs(measured - self._rppg_fps) > 0.15 * self._rppg_fps:
                    print(f"[DataCollector][capture] achieved {measured:.1f} fps vs "
                          f"assumed {self._rppg_fps:.1f} fps — HR/RR will be "
                          f"scaled by {measured / self._rppg_fps:.2f}x")
        print("[DataCollector] Capture worker stopped.")

    def _face_box_loop(self) -> None:
        """Background thread: maintain the square rPPG crop box.

        Runs its own FaceLandmarker at ``face_box_hz`` (MediaPipe task objects
        are not safe to share across threads, so this is a second instance — the
        main loop keeps its own for gaze/EAR/MAR). Publishing the box from here
        rather than from the main loop means the capture worker never crops with
        a box that went stale because the decision engine blocked.

        Cadence mirrors the reference implementations: MMRPhys re-detects every
        DYNAMIC_DETECTION_FREQUENCY frames (~1 s) and the browser demo every 3-6
        frames, reusing a cached box in between.
        """
        if not HAS_MP:
            print("[DataCollector] MediaPipe unavailable; face-box worker exiting.")
            return
        try:
            _model = _resolve_face_landmarker_model()
            if not _model:
                print("[DataCollector] No landmarker model; face-box worker exiting.")
                return
            _opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_model),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._box_landmarker = mp_vision.FaceLandmarker.create_from_options(_opts)
        except Exception as e:  # noqa: BLE001
            print(e, "Error initializing face-box landmarker; worker exiting")
            return
        print(f"[DataCollector] Face-box worker started "
              f"({1.0 / self._face_box_interval:.1f} Hz).")
        last_id = -1
        next_t = time.monotonic()
        while self._running:
            with self._cap_lock:
                frame = self._cap_frame
                fid = self._cap_frame_id
            if frame is not None and fid != last_id:
                last_id = fid
                h_img, w_img = frame.shape[:2]
                try:
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,       # type: ignore
                                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    res = self._box_landmarker.detect(mp_img)
                    landmarks = res.face_landmarks[0] if res.face_landmarks else None
                except Exception as e:  # noqa: BLE001
                    print(e, "Error detecting face box")
                    landmarks = None
                if landmarks:
                    # Square + 1.5x + EMA, exactly as at training time.
                    box = self._rppg_crop_box(landmarks, w_img, h_img)
                    if box[2] > 0 and box[3] > 0:
                        with self._box_lock:
                            self._rppg_box = box
                            self._rppg_box_t = time.monotonic()
                # No face -> leave the last box in place; the capture worker
                # drops it once _rppg_box_staleness_s elapses.
            next_t += self._face_box_interval
            now = time.monotonic()
            if next_t < now:
                next_t = now
            time.sleep(max(0.0, next_t - now))
        try:
            if self._box_landmarker is not None:
                self._box_landmarker.close()
        except Exception:
            pass
        print("[DataCollector] Face-box worker stopped.")

    def _yolo_loop(self) -> None:
        """Background thread: run YOLO26 distraction OFF the main loop.

        Reads the latest raw frame published by ``_visual_process`` and writes the
        distraction label list to ``self._yolo_lab`` under ``self._yolo_lock``. The
        main loop reads that list each tick. This removes YOLO's ~45 ms/frame from
        the critical path (its biggest cost) — the main loop no longer blocks on it.
        """
        try:
            self._distraction = _perception.DistractionDetector()
        except Exception as e:  # noqa: BLE001
            print(e, "Error initializing distraction detector; YOLO thread exiting")
            return
        if not getattr(self._distraction, "available", False):
            print("[DataCollector] YOLO distraction unavailable; worker exiting.")
            return
        print("[DataCollector] YOLO distraction worker started.")
        last_id = -1
        while self._running:
            with self._yolo_lock:
                frame = self._yolo_frame
                fid = self._yolo_frame_id
            if frame is None or fid == last_id:   # nothing new yet
                time.sleep(0.01)
                continue
            last_id = fid
            try:
                labels, dets = self._distraction(frame)
            except Exception as e:  # noqa: BLE001
                print(e, "Error in YOLO distraction")
                labels, dets = [], []
            with self._yolo_lock:
                self._yolo_lab = [str(x) for x in (labels or [])]
                self._yolo_dets = list(dets or [])
        print("[DataCollector] YOLO distraction worker stopped.")

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
        if self.visual_enabled and self.cap is not None:
            # Camera owner first: every other visual consumer reads the frames it
            # publishes, and the face-box worker needs one before it can detect.
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            # Wait for the first frame so calibration does not open with a
            # spurious "camera read failed" while the device warms up.
            _deadline = time.monotonic() + 3.0
            while time.monotonic() < _deadline:
                with self._cap_lock:
                    if self._cap_started:
                        break
                time.sleep(0.02)
            else:
                print("[DataCollector] No frame within 3 s of starting capture — "
                      "continuing anyway (camera may still be warming up).")
            self._face_box_thread = threading.Thread(target=self._face_box_loop, daemon=True)
            self._face_box_thread.start()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if self.visual_enabled:
            self._yolo_thread = threading.Thread(target=self._yolo_loop, daemon=True)
            self._yolo_thread.start()
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
            if self._yolo_thread and self._yolo_thread.is_alive():
                self._yolo_thread.join(timeout=2.0)
            if self._face_box_thread and self._face_box_thread.is_alive():
                self._face_box_thread.join(timeout=2.0)
            # Capture worker last among the visual threads: it must have exited
            # before release() frees the VideoCapture it is reading from.
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=2.0)
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
