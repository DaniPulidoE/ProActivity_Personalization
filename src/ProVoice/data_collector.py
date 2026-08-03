from __future__ import annotations

import base64
import csv
import datetime
import os
import re
import threading
import time
import http.client
import urllib.request
import urllib.parse
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
# Standalone module (src/vehicle_state_io.py), deliberately outside the
# ProVoice package so the lightweight bridge process can import it without
# dragging in torch/cv2/mediapipe.
from vehicle_state_io import VehicleStateChannel

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

class _Phase:
    """Context manager that records how long a loop section took, in ms.

    Overhead is two perf_counter() calls (~100 ns) against sections measured in
    milliseconds, so it is safe to leave enabled during a live session.
    """
    __slots__ = ('store', 'name', 't0')

    def __init__(self, store, name: str) -> None:
        self.store = store
        self.name = name
        self.t0 = 0.0

    def __enter__(self):
        if self.store is not None:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.store is not None:
            dt = (time.perf_counter() - self.t0) * 1000.0
            buf = self.store.get(self.name)
            if buf is None:
                buf = self.store[self.name] = deque(maxlen=200)
            buf.append(dt)
        return False


_emotion_recognizer = None
_EMOTIEFFLIB_MODEL = "enet_b0_8_best_afew"
# EmotiEffLib emits 8 AffectNet classes; map to the pipeline's 7-class EMOTION_VOCAB (map contempt -> neutral)
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
# Store per-driver baselines after calibration - to be reused in other runs
_CALIBRATION_DIR = os.path.join('.', 'data', 'calibration_data')
# Per-tick raw measures collected during calibration (one CSV row per tick),
# distinct from _CALIBRATION_DIR which holds only the aggregated baseline.
_CALIBRATION_LOG_DIR = os.path.join(_CALIBRATION_DIR, 'calibration_logs')
_CALIBRATION_LOG_FIELDS = ('timestamp', 'elapsed_s', 'gaze_score', 'ear', 'mar', 'bpm', 'rr')
# key -> required numeric fields (mirrors what compute_calibration() writes)
_CALIBRATION_SCHEMA: Dict[str, Tuple[str, ...]] = {
    'gaze_score': ('mean', 'std', 'threshold'),
    'ear':        ('mean', 'std', 'threshold'),
    'mar':        ('mean', 'std', 'threshold'),
    'bpm':        ('mean', 'std'),
    'rr':         ('mean', 'std'),
    'blink_rate': ('mean',),
    'perclos':    ('mean', 'std'),
}

# Anscombe transform of a zero count: 2*sqrt(0 + 3/8) = 1.224745.
# The Anscombe transform is a variance-stabilizing transform for Poisson data, 
# and the zero-count value is used to avoid taking the square root of zero.
_YAWN_ANSCOMBE_ZERO = 2.0 * math.sqrt(3.0 / 8.0)

# ── Robust baseline statistics for the rPPG signals ───────────────────────────
# HR/RR readings contain occasional harmonic-lock outliers (see _hr_is_harmonic),
# so the baseline uses median + MAD rather than mean + std.
# This value approximates the std by the median
_MAD_TO_SIGMA = 1.4826
# Floor in dispersion for bpm and rr:

#_ROBUST_SCALE_FLOOR = {'bpm': 5.0, 'rr': 2.0}

# ── HR harmonic rejection ─────────────────────────────────────────────────────
# Band, as a multiple of the running reference, in which a reading is treated as
# the 2nd harmonic rather than the pulse. +-15% around 2x: the toolbox's FFT
# resolution is 30/600 Hz = 3 bpm, so a true 2f peak always lands well inside.
_HR_HARMONIC_LO = 1.7
_HR_HARMONIC_HI = 2.3
# Accepted readings needed before the band is trusted enough to reject anything.
_HR_REF_MIN = 3
# Consecutive rejections after which the REFERENCE is presumed wrong rather than
# the readings, and is re-seeded. Bounds the worst case: without it a bad
# reference could suppress every reading for the rest of the session.
_HR_REJECT_RUN_MAX = 5


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
        vehicle_state_path: Optional[str] = None,
        state_poll_hz: float = 2.0,
        calibration_dir: str = _CALIBRATION_DIR,
        rppg_fps: float = 30.0,
        face_box_hz: float = 1.0,
        decision_hz: float = 4.0,
        calibration_only: bool = False,
        data_collection: bool = False,
        traffic_seed: Optional[int] = None,
    ) -> None:
        # Everything stop()/__del__ touches is set FIRST, so a constructor
        # that fails part-way still leaves a safely destructible object.
        self._running = False
        self._stopped = False
        self._thread: Optional[threading.Thread] = None
        self._state_poll_thread: Optional[threading.Thread] = None
        self._yolo_thread: Optional[threading.Thread] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._face_box_thread: Optional[threading.Thread] = None
        self._decision_thread: Optional[threading.Thread] = None
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

        # ── Decision cadence ──────────────────────────────────────────────────
        # The decision engine runs on its OWN thread (see _decision_loop): it
        # snapshots the shared state the collection loop publishes and never
        # blocks it. Two reasons this is not just a speed-up:
        #   * the decision rate becomes an explicit, reportable parameter rather
        #     than an emergent property of perception latency and CPU load — so
        #     decisions.csv row density is no longer confounded with machine load
        #     across participants;
        #   * an xLSTM forward over the whole window is the single largest
        #     remaining cost on the collection path.
        self.decision_interval = 1.0 / max(1e-3, float(decision_hz))
        # Last decision, published for the dashboard and for the raw log's LoA /
        # FCD fields. Written by the decision thread, read by the collection
        # thread and the web UI — always under _lock.
        self._last_action: Optional[Dict[str, Any]] = None
        # Frame timestamp the last decision was computed from. Guarantees at most
        # one decisions.csv row per distinct collected frame, so every row still
        # joins exactly onto one raw_data.jsonl line.
        self._last_decided_ts: Optional[str] = None

        self.functionname = function_name or "fatigue_alert"

        self.cam_index = cam_index
        self.static_context: Dict[str, Any] = dict(static_context or {})

        self.face_landmarker = None
        self.carla_vehicle = carla_vehicle
        # Cached per session — see the note at the get_weather() call site.
        self._carla_world = None
        self._carla_map = None
        self.vehicle_state_url = vehicle_state_url.rstrip("/") if vehicle_state_url else None
        # File published by scripts/vehicle_state_file_bridge.py. Preferred over
        # the URL for a same-machine setup: reading a small file avoids the ~20
        # TCP connections/second the HTTP bridge generated, which is the kernel
        # path suspected in the two machine bugchecks of 2026-07-28.
        self.vehicle_state_path = vehicle_state_path or None
        # Seconds between bridge polls. Every vehicle field is held constant
        # between polls, so this is the real sampling rate of speed/steer/brake
        # when the bridge is in use — it should track the collection rate, not
        # sit at the old 2 Hz default.
        self._state_poll_interval = 1.0 / max(1e-3, float(state_poll_hz))
        # ONE HTTP connection, reused for every poll of the remote bridge (see
        # _http_get_state). urllib.request.urlopen, which this used to call,
        # opens and tears down a TCP connection per request: at 20 Hz that is
        # ~36,000 connect/teardown cycles in a 30-minute session, and that
        # kernel churn is the suspected trigger of the 2026-07-28 machine
        # bugchecks. Held here rather than in the poller's locals so stop()
        # can be sure it is closed.
        self._http_conn = None
        self._http_path = "/"
        # Comfortably longer than one poll interval, short enough that a dead
        # bridge is noticed within a couple of frames rather than stalling the
        # thread for the default socket timeout.
        self._state_http_timeout = max(2.0, 3.0 * self._state_poll_interval)
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
        # None rather than 0, unlike every cached field above. These two are
        # computed by the bridge (scripts/vehicle_state_file_bridge.py:
        # lead_and_headway) and None is a real reading -- no leader within
        # range, or a headway that is undefined because the ego is stopped.
        # Zero would say the opposite of both: bumper to bumper, closing now.
        self._cached_lead_distance: Optional[float] = None
        self._cached_headway_s: Optional[float] = None
        self._warned_no_headway: bool = False

        # Which traffic scenario this session ran. Set by the launcher, or read
        # off the bridge's /session under --remote; None means nobody told this
        # process, which is a real state and is logged as null rather than
        # guessed at. Do NOT default it to 42 here: a wrong seed recorded
        # confidently is worse than an absent one, because only the absent one
        # is visible as a problem later.
        self.traffic_seed: Optional[int] = traffic_seed

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
                # Request 640x480 (as MMRPhys's own demos do) at the rate the
                # deployed checkpoint was trained on: the SCAMPS configs specify
                # FS: 30, so the model's temporal filters were learned at 30 fps.
                # (Switch this to 25 if the BP4D checkpoint is ever made the
                # default -- BP4D is a 25 fps dataset.)
                # The driver may refuse any of these, so read back what it
                # actually granted: the achieved rate becomes the rPPG sampling
                # rate (it sets both the Butterworth cutoffs and the FFT
                # frequency axis, so getting it wrong scales every HR/RR value).
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
        # Harmonic-rejection state. Touched only by the rPPG worker thread, so no
        # lock; the counters are ints, read atomically by the main loop for the
        # logged diagnostic.
        self._hr_accepted: deque = deque(maxlen=5)  # recent ACCEPTED readings -> reference
        self._hr_reject_run: int = 0                # consecutive rejections
        self._hr_harmonic_rejects: int = 0          # session total, logged
        # UNFILTERED reading as the toolbox produced it, kept whether or not the
        # harmonic filter accepted it, and logged as heart_rate_raw. The live
        # filter is a guard on the actuator, NOT the source of truth: dropping a
        # reading from the log would make the filter irreversible and unauditable,
        # and the offline pipeline needs the raw series to re-derive HR with a
        # two-sided (Hampel-style) filter that the causal live one cannot match.
        self._last_hr_raw: Optional[float] = None
        self._last_hr_raw_t: float = 0.0
        self._last_hr_rejected: bool = False
        self._rppg_staleness_s: float = 15.0
        # rPPG face crop: MMRPhys is trained on a SQUARED face-detector box
        # enlarged by LARGE_BOX_COEF=1.5 (see _rppg_crop_box for the exact
        # reproduction), then EMA-smoothed here to kill the per-frame crop jitter
        # that corrupts the pulse signal (rPPG is spectral, jitter-sensitive).
        # EMA rather than Kalman: the head moves slowly, so a velocity model buys
        # nothing and EMA mirrors the toolbox's own near-static box. Both tunable.
        self._rppg_box_coef: float = 1.5                        # = LARGE_BOX_COEF
        self._rppg_box_alpha: float = 1.0                       # 1.0 = no smoothing (reference)
        self._rppg_box_ema: Optional[np.ndarray] = None         # smoothed [cx, cy, w, h]
        # Shared square crop box, written by the face-box worker and read by the
        # capture worker at full frame rate (see _face_box_loop / _capture_loop).
        self._box_lock = threading.Lock()
        self._rppg_box: Optional[Tuple[int, int, int, int]] = None   # (x, y, w, h)
        self._rppg_box_t: float = 0.0                                # monotonic stamp
        # Drop the box after this. 4 s, not 2: the box-refresh period is set by
        # how long a detection takes, which inflates under load, and at 2 s it
        # sat right on the threshold -- so the box intermittently expired and
        # rPPG feeding stalled even though the detector never failed. This is a
        # tolerance for a SLOW detector, not for an absent face: a real
        # look-away is caught by the miss messages and the gap counter.
        self._rppg_box_staleness_s: float = 4.0
        self._box_landmarker = None                                  # worker-owned detector
        # rPPG results, published by the capture worker, read by the main loop.
        self._rppg_lock = threading.Lock()
        # Continuity tracking. Frames are fed only while a fresh face box exists,
        # so a look-away punches a hole in the stream and the model window
        # splices across it — the FFT then treats the join as continuous, giving
        # a silently wrong HR.
        #
        # Suppression is OFF by default: in practice gaps fire often enough that
        # discarding every affected reading leaves too little data. The counters
        # below stay live, so `rppg_gaps` still marks which frames sit near a
        # discontinuity and post-hoc analysis can filter on it. Set
        # _rppg_gap_suppress = True to refuse contaminated readings outright.
        self._rppg_gap_suppress: bool = False
        # 30 frames (= 1 s at 30 fps), not 5. At 5 the counter fired on ordinary
        # scheduler jitter: the main loop never sleeps (its per-tick decision +
        # logging work exceeds sampling_interval, so `next_t < now` always), so
        # the capture worker gets descheduled for a few hundred ms at a time and
        # trips any threshold near one frame interval. Those stalls are real
        # frame loss, but counting them tells you nothing you can act on --
        # 1 s isolates actual look-aways, which is what the flag is for.
        self._rppg_gap_frames: float = 30.0   # gap > this many frame intervals = discontinuity
        self._rppg_last_fed_t: float = 0.0
        self._rppg_contig: int = 0            # gap-free frames fed since the last hole
        self._rppg_gap_count: int = 0         # discontinuities seen this session
        self._rppg_suppressed: int = 0        # readings discarded as contaminated
        # Face-detector misses. Every failed detection is reported: a miss stops
        # the crop box being refreshed, and once it goes stale the capture worker
        # stops feeding rPPG altogether — so this is the upstream cause of most
        # rPPG dropouts, and the check that tells you whether YOLO5Face actually
        # works on your camera and lighting.
        self._face_box_misses: int = 0        # total failed detections
        self._face_box_consec_misses: int = 0 # current unbroken run
        self._face_box_last_ok_t: float = 0.0 # monotonic stamp of last success
        self.data_history: deque = deque(maxlen=window_size)
        self.window_size = window_size
        self.blink_count = 0
        self.yawn_count = 0
        self.perclos = 0.0
        self.drowsiness_alert = False
        # PERCLOS over a fixed TIME window (rate-independent). 30 s follows the
        # literature (Wierwille / Dinges & Mallis define PERCLOS over 60 s; 30 s
        # is the common shorter variant). The previous 10 s was below anything
        # published and, at ~10 fps, quantized the fraction to ~1% per frame.
        # One attribute, read by BOTH compute_calibration() and the serving loop,
        # so the baseline and the served statistic can never describe different
        # window lengths.
        self._perclos_buf: deque = deque()
        self._perclos_window_s: float = 30.0
        self._eye_closed_since: float = 0.0  # monotonic time when EAR first dropped
        self._mouth_open_since: float = 0.0  # monotonic time when MAR first rose
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
        # Timestamp of EVERY calibration tick, face present or not — the two
        # lists above only grow when a face is visible, so they measure face
        # detection, not loop cadence. See compute_calibration() for why the
        # achieved rate is worth recording.
        self._calibration_tick_ts: List[float] = []
        # Per-tick raw log (see calibration_log_file()); opened lazily on the
        # first tick of a live calibration, closed once compute_calibration()
        # finishes.
        self.calibration_log_dir = _CALIBRATION_LOG_DIR
        self._calibration_log_fh = None
        self._calibration_log_writer = None

        # Calibration: 180 s time-based, collects gaze/EAR/MAR/HR/RR.
        # 3 minutes is set by the BLINK-RATE baseline, not by rPPG: a blink count
        # is Poisson, so 60 s at ~15 blinks/min gives SE = sqrt(15)/15 ≈ 26% on
        # the baseline rate, versus ~15% at 3 min. (rPPG is no longer the binding
        # constraint — at 30 fps it emits a reading every ~1.5 s, i.e. ~120 per
        # 3 min.) 3 min also averages over road/traffic variability, which matters
        # now that the baseline is collected while driving.
        self._calibration_duration_s: float = 180.0
        self._calibration_start_t: float = 0.0
        # Persisted per-driver baselines: the loop tries the stored file ONCE
        # before starting the live routine (see _run_loop).
        self.calibration_dir = calibration_dir or _CALIBRATION_DIR
        self._calibration_load_attempted = False
        # Run modes. Mutually exclusive; rejected in main.py before reaching here.
        #   calibration_only: measure a fresh baseline, store it, then exit.
        #   data_collection:  record raw data and nothing else — no live
        #                     calibration (reuse the stored baseline or fall back
        #                     to neutral defaults, see _run_loop) and no
        #                     inference. main.py also passes decision_engine=None
        #                     so no model is even loaded; the guard in start() is
        #                     what makes the mode hold regardless of the caller.
        self.calibration_only = calibration_only
        self.data_collection = data_collection
        # Invoked once the baseline is stored under calibration_only. This class
        # cannot shut the process down itself (main.py owns the uvicorn server),
        # so the owner installs a callback that does.
        self.on_calibration_complete: Optional[Any] = None
        # Invoked ONCE, right after the first frame reaches raw_data.jsonl.
        # Same hand-off as above: this class knows when logging starts, and the
        # owner decides what that means. In a two-machine run main.py uses it to
        # tell Drive on the OTHER machine that its LoA windows can begin --
        # locally Drive watches this same first line in the file itself.
        self.on_first_frame_logged: Optional[Any] = None
        self._first_frame_logged = False
        # Transient camera-read failures (warm-up frames, momentary USB glitches)
        # must NOT abort calibration; only give up after this many seconds of
        # UNBROKEN failures. Resets on any successful read.
        self._read_fail_since: float = 0.0
        self._read_fail_grace_s: float = 5.0

        # --- DEBUG: frame-rate instrumentation (safe to delete) ---
        self._dbg_last_tick_t: float = 0.0
        self._dbg_last_print_t: float = 0.0
        self._dbg_dt_hist: deque = deque(maxlen=100)
        # Per-section timing. Set to None to disable all phase instrumentation;
        # the _Phase context manager then short-circuits to two attribute reads.
        self._prof: Optional[Dict[str, deque]] = {}
        self._prof_period_s: float = 5.0     # how often the breakdown is printed
        self._prof_last_print_t: float = 0.0

    def _phase(self, name: str) -> _Phase:
        """``with self._phase('decide'): ...`` — times one section of the loop."""
        return _Phase(self._prof, name)

    def _print_profile(self, dt_avg_ms: float) -> None:
        """Print the per-section breakdown, slowest first.

        ``dt`` is the whole COLLECTION tick; the sections below are the parts of
        it that are instrumented, so ``dt - sum(sections)`` is time the loop
        spent descheduled or in uninstrumented code.

        Phases named ``decide.*`` run on the decision thread, not this one, so
        they are reported separately and excluded from the tick's accounting —
        folding them in would inflate `accounted` past the tick itself and make
        `unaccounted` go negative.
        """
        if not self._prof:
            return
        # Snapshot: the decision thread creates new keys in this dict as its own
        # phases fire for the first time, and iterating it live raises
        # "dictionary changed size during iteration".
        snapshot = list(self._prof.items())
        loop_rows, dec_rows = [], []
        for name, buf in snapshot:
            vals = sorted(buf)
            n = len(vals)
            if n == 0:
                continue
            row = (sum(vals) / n, vals[min(n - 1, int(n * 0.95))], name)
            (dec_rows if name.startswith('decide') else loop_rows).append(row)
        if not loop_rows and not dec_rows:
            return
        loop_rows.sort(reverse=True)
        dec_rows.sort(reverse=True)
        n_samples = max((len(b) for _, b in snapshot), default=0)
        accounted = sum(r[0] for r in loop_rows if '.' not in r[2])
        print(f"[DataCollector][prof] tick={dt_avg_ms:7.1f}ms  "
              f"accounted={accounted:7.1f}ms  "
              f"unaccounted={dt_avg_ms - accounted:7.1f}ms  "
              f"(mean/p95 over last {n_samples} ticks)")
        for mean, p95, name in loop_rows:
            share = (100.0 * mean / dt_avg_ms) if dt_avg_ms > 0 else 0.0
            indent = "    " if '.' in name else "  "
            print(f"{indent}{name:<22s} {mean:7.1f} ms  p95 {p95:7.1f} ms  {share:5.1f}%")
        if dec_rows:
            budget_ms = self.decision_interval * 1000.0
            spent = sum(r[0] for r in dec_rows)
            print(f"  [decision thread]      budget {budget_ms:7.1f} ms  "
                  f"spent {spent:7.1f} ms  "
                  f"({'OVER — decisions are pacing themselves' if spent > budget_ms else 'within budget'})")
            for mean, p95, name in dec_rows:
                share = (100.0 * mean / budget_ms) if budget_ms > 0 else 0.0
                print(f"    {name:<22s} {mean:7.1f} ms  p95 {p95:7.1f} ms  {share:5.1f}%")

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

    def _rppg_crop_box(self, x: float, y: float, bw: float, bh: float,
                       w: int, h: int):
        """SQUARE, enlarged face box (x, y, bw, bh) for rPPG.

        Reproduces the training-time crop geometry of MMRPhys
        (BaseLoader.face_detection + crop_face_resize) given a raw detector box:

          1. SQUARE it: ``side = max(w, h)``, re-centred on the box centre,
          2. enlarge by ``_rppg_box_coef`` (= LARGE_BOX_COEF = 1.5) about that
             same centre, so the crop takes in forehead and cheeks,
          3. clamp to the frame.

        Squaring matters: the crop is later resized to 72x72, so a non-square box
        anisotropically stretches the face relative to every frame the model was
        trained on.

        ``_rppg_box_alpha`` = 1.0 reproduces the reference exactly: no temporal
        smoothing, so the box is piecewise constant between detections (the
        upstream loader uses the box from detection ``i // detection_freq``
        verbatim, with USE_MEDIAN_FACE_BOX False). Lower it to re-enable EMA
        smoothing of centre and size -- worthwhile only if the detection rate is
        raised well above the reference 1 Hz, since the EMA time constant is
        ~1/alpha detections.
        """
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

    def _hr_is_harmonic(self, hr: float) -> bool:
        """True when ``hr`` looks like the 2nd harmonic of the current pulse rate.

        The toolbox picks the largest FFT peak in 0.6-3.3 Hz (36-198 bpm) with no
        harmonic disambiguation. A real BVP waveform carries genuine energy at 2f
        (sharp systolic upstroke, dicrotic notch), and for any true rate below
        99 bpm that 2f peak is still inside the band — so every seated
        participant is exposed. The symptom is a reading that jumps to ~2x the
        established rate and back (measured: ~150 against a true ~75).

        Compared against a running median of recent ACCEPTED readings. The
        reference decides ONLY accept/reject; the published value is the raw
        reading, so genuine excursions keep their full amplitude — unlike a
        rolling median, which would flatten any excursion shorter than its own
        half-width.

        A GRADUAL rise never triggers this: intermediate readings are accepted,
        the reference tracks upward, and the band moves with it. Only an
        effectively instantaneous near-doubling matches, which is not reachable
        while seated. That is why persistence is not the discriminator here —
        a sustained harmonic and a sustained real change look identical in
        duration, and only the *transition* distinguishes them.
        """
        if len(self._hr_accepted) < _HR_REF_MIN:
            return False                       # reference not established yet
        ref = float(np.median(self._hr_accepted))
        if ref <= 0.0:
            return False
        return _HR_HARMONIC_LO * ref <= hr <= _HR_HARMONIC_HI * ref

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
            print(f"[Calibration] Starting {self._calibration_duration_s:.0f} s calibration.")
        # One entry per tick that actually sampled a frame. Failed reads return
        # above without appending, which matches how a session's rate is
        # measured from raw_data.jsonl (only logged frames count).
        self._calibration_tick_ts.append(now)

        # ── time-based calibration (see _calibration_duration_s) ──────────────
        h_img, w_img = frame.shape[:2]
        landmarks = self.compute_face_landmarks(frame)
        face_present = bool(landmarks)

        gaze_score = self.get_gaze_score(frame, landmarks)
        log_gaze = None
        if gaze_score > 0.0:
            self._calibration_data['gaze_score'].append(gaze_score)
            log_gaze = gaze_score

        # EAR/MAR from the SAME landmarks (only when a face is actually present).
        log_ear = None
        log_mar = None
        if face_present:
            try:
                eye, mouth = _perception.eye_mouth_aspect_ratios(landmarks, w_img, h_img)
                self._calibration_data['ear'].append(eye)
                self._calibration_data['mar'].append(mouth)
                self._calibration_ear_ts.append((now, eye))
                self._calibration_mar_ts.append((now, mouth))
                log_ear, log_mar = eye, mouth
            except Exception as e:  # noqa: BLE001
                print(e, "Error computing EAR/MAR")

        # Sample the rPPG readings the capture worker publishes. Keyed on the
        # reading TIMESTAMP so each estimate is counted once — polling the held
        # value every tick would append the same number over and over (a reading
        # lands every 6 s, the loop ticks 20x/s) and collapse the baseline std.
        log_bpm = None
        log_rr = None
        if self.rppg_estimator is not None:
            with self._rppg_lock:
                last_hr, last_hr_t = self._last_hr, self._last_hr_t
                last_rr, last_rr_t = self._last_rr, self._last_rr_t
            if last_hr is not None and last_hr_t > self._cal_last_hr_t:
                self._cal_last_hr_t = last_hr_t
                self._calibration_data['bpm'].append(float(last_hr))
                log_bpm = float(last_hr)
                print(f"[Calibration] rPPG: HR={last_hr}")
            if last_rr is not None and last_rr_t > self._cal_last_rr_t:
                self._cal_last_rr_t = last_rr_t
                self._calibration_data['rr'].append(float(last_rr))
                log_rr = float(last_rr)
                print(f"[Calibration] rPPG: RR={last_rr}")

        self._log_calibration_sample(now, log_gaze, log_ear, log_mar, log_bpm, log_rr)

        self.latest_frame = frame
        #print(f"[Calibration] {now - self._calibration_start_t:.1f}/{self._calibration_duration_s:.1f} s elapsed. ")
        if (now - self._calibration_start_t) >= self._calibration_duration_s:
            self.compute_calibration()

    def compute_calibration(self, save: bool = True):
        # compute mean and std for each metric, set calibrated flag
        # ``save=False`` is used by the --data-collection fallback below: with no
        # samples this writes the neutral defaults, and persisting those would
        # leave a zeroed file masquerading as this driver's stored baseline.
        self.calibrate = dict()
        for key, values in self._calibration_data.items():
            if values:
                if key == 'ear':
                    # The quantity we actually want is the OPEN-eye level, but the
                    # baseline is now collected while driving, so ~6% of frames are
                    # mid-blink (~15 blinks/min x ~250 ms). Those sit at the bottom
                    # of the distribution and drag a plain mean down, which would
                    # lower the threshold and make eye closures fire LESS often.
                    # The MEDIAN is unaffected: 6% low-side contamination shifts it
                    # by only ~-0.08 sigma (<1%), far inside its 50% breakdown
                    # point, so the 0.75 multiplier below means what the literature
                    # intends by it (Ersoy et al. 2026 use 0.75 of a static
                    # eyes-open calibration). A trimmed mean would also work but
                    # sits ~0.35 sigma high, silently turning 0.75 into ~0.78.
                    # NOTE: for 'ear' this field holds the MEDIAN, not the mean.
                    mean = float(np.median(values))
                    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                elif key in ('bpm', 'rr'):
                    # Fully robust: median for location, MAD for scale. The rPPG
                    # readings entering here still contain the occasional 2f
                    # harmonic that slipped past _hr_is_harmonic (and, for 'rr',
                    # no harmonic filter at all — see the note in that method).
                    # The MEAN would be pulled by them, but the STD far more so:
                    # a handful of ~150s among ~75s inflates it enormously, and
                    # _standardized_delta divides hr_delta by it.
                    mean = float(np.median(values))
                    mad = float(np.median([abs(x - mean) for x in values]))
                    #std = max(_MAD_TO_SIGMA * mad, _ROBUST_SCALE_FLOOR[key])
                    std = _MAD_TO_SIGMA * mad
                else:
                    mean = sum(values) / len(values)
                    # Dispersion about the location estimate above. Not read at
                    # serving for 'mar' (only 'threshold' is); kept for the record
                    # and for the gaze_score normalization, which does use it.
                    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                # MAR: fixed 0.55 rather than a fraction of the baseline. A driving
                # baseline contains speech, which inflates a calibrated mouth
                # baseline and would make yawns be missed for the whole session;
                # a fixed threshold plus the 2 s duration gate in _visual_process
                # fails more safely. NOTE: 0.55 comes from dlib-derived studies
                # while perception.py computes MAR on MediaPipe landmarks — verify
                # it separates yawns from speech on recorded footage.
                # EAR "eyes-closed" threshold: a fraction of the open-eye baseline
                # (with an absolute floor), which is robust to a tiny calibration
                # std. mean - 2.5*std sat right inside the normal operating range
                # and made PERCLOS/drowsiness fire almost constantly.
                self.calibrate[key] = {'mean': mean, 'std': std, 'threshold': mean + std * 2.5 if key in ['gaze_score'] else max(0.15, mean * 0.75) if key in ['ear'] else 0.55}
            else:
                # default values originally in the script
                thres = 0.2 if key in ['gaze_score', 'ear'] else 0.55
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
        # Floor the scale at a PHYSIOLOGICAL value, not at epsilon. During an alert
        # baseline PERCLOS barely moves, so its std measures measurement noise
        # rather than the driver's dynamic range; dividing by it turns the served
        # feature into a signal-to-noise ratio and it explodes. With a measured
        # baseline (mean 0.043, std 0.016) the old 1e-3 floor let PERCLOS 0.38 --
        # the drowsiness_alert threshold -- normalize to z = 21.
        # This is the same failure that killed the EAR "mean - 2.5*std" rule; the
        # per-driver MEAN is worth keeping (baseline PERCLOS varies with eye shape
        # and blink rate) but the per-driver STD is not a meaningful scale.
        # 0.1 is chosen so the alert->drowsy span (~0.04 -> 0.38) maps to ~3.4
        # units. A driver whose baseline PERCLOS genuinely varies more than this
        # still gets their own std, so it degrades gracefully.
        PERCLOS_MIN_SCALE = 0.1
        if len(perclos_series) > 1:
            std_perclos = max(float(np.std(perclos_series)), PERCLOS_MIN_SCALE)
        else:  # no full-window samples collected -> fall back to the same scale
            std_perclos = PERCLOS_MIN_SCALE
        self.calibrate['perclos'] = {'mean': mean_perclos, 'std': std_perclos}

        # Achieved loop rate during calibration. Recorded because the baselines
        # above are NOT all rate-invariant: 'blink_rate.mean' comes from the same
        # duration-gated detector the session uses (see _visual_process), whose
        # yield depends on the sample interval relative to the 100/500 ms gate.
        # When calibration and session run at the same rate that dependence
        # cancels in the z-score; when they differ it does not, and shows up as a
        # constant offset on a feature the model reads.
        #
        # The two rates differ SYSTEMATICALLY, not incidentally: calibration runs
        # a much lighter tick than collect_data() — no emotion inference, no YOLO
        # (the worker idles until _visual_process publishes a frame), no decision
        # engine (its loop gates on self.calibrated), no CARLA queries and no
        # logging. So calibration sits nearer the nominal cap and the session
        # falls below it. Storing the rate makes that gap measurable instead of
        # invisible: nothing else records it, since the [fps] print and the
        # profiler both live in the post-calibration branch of _run_loop.
        #
        # This is diagnostic METADATA and deliberately NOT in _CALIBRATION_SCHEMA:
        # that schema is the contract for fields the SERVING path indexes, and
        # nothing reads these. Adding it there would invalidate every existing
        # calibration file for a value no consumer needs.
        self.calibrate['sample_rate'] = self._calibration_rate_stats()

        self.calibrated = True
        self._session_start_t = time.monotonic()

        print("Calibration completed:", self.calibrate)
        _sr = self.calibrate['sample_rate']
        print(f"[Calibration] achieved rate: median {_sr['median_hz']:.1f} Hz, "
              f"mean {_sr['mean_hz']:.1f} Hz, slow tail (p05) {_sr['p05_hz']:.1f} Hz "
              f"over {_sr['n_ticks']:.0f} ticks / {_sr['duration_s']:.0f} s. "
              f"Compare against the [fps] line once collection starts.")

        if save:
            self.save_calibration()

        self._close_calibration_log()

    def _calibration_rate_stats(self) -> Dict[str, float]:
        """Loop-rate statistics over the calibration ticks.

        Median rather than mean for the headline figure: a handful of long
        stalls drag the mean well below the rate the loop actually held (18.5 vs
        15.0 Hz on a measured session). ``p05_hz`` is the slow tail, which is
        what matters for the blink gate — a blink is lost once the sample
        interval approaches its 100-500 ms duration window. All-zero when fewer
        than two ticks were recorded, so the field is always present and finite.
        """
        ts = self._calibration_tick_ts
        if len(ts) < 2:
            return {'median_hz': 0.0, 'mean_hz': 0.0, 'p05_hz': 0.0,
                    'n_ticks': float(len(ts)), 'duration_s': 0.0}
        span = float(ts[-1] - ts[0])
        d = np.diff(np.asarray(ts, dtype=np.float64))
        # Drop non-positive intervals (clock jitter) and multi-second pauses,
        # which are stalls rather than a sampling cadence.
        d = d[(d > 0.0) & (d < 5.0)]
        if d.size == 0:
            return {'median_hz': 0.0, 'mean_hz': 0.0, 'p05_hz': 0.0,
                    'n_ticks': float(len(ts)), 'duration_s': round(span, 2)}
        return {
            'median_hz': round(float(1.0 / np.median(d)), 2),
            'mean_hz': round(float((len(ts) - 1) / span), 2) if span > 0 else 0.0,
            'p05_hz': round(float(1.0 / np.percentile(d, 95)), 2),
            'n_ticks': float(len(ts)),
            'duration_s': round(span, 2),
        }

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

    def calibration_log_file(self) -> Optional[str]:
        """Path of this participant's per-tick calibration log, or None without an ID.

        Distinct from ``calibration_file()``: that file holds the aggregated
        mean/std/threshold baseline; this one holds the raw per-tick samples
        (gaze score, EAR, MAR, HR, RR) the baseline was computed from.
        """
        pid = str(self.static_context.get('participantid') or '').strip()
        if not pid:
            return None
        safe_pid = re.sub(r'[^A-Za-z0-9_.-]', '_', pid)
        return os.path.join(self.calibration_log_dir, f"log_calibration_{safe_pid}.csv")

    def _open_calibration_log(self) -> None:
        """Lazily open the per-tick calibration log for writing (idempotent)."""
        if self._calibration_log_writer is not None:
            return
        path = self.calibration_log_file()
        if path is None:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._calibration_log_fh = open(path, 'w', newline='', encoding='utf-8')
            self._calibration_log_writer = csv.writer(self._calibration_log_fh)
            self._calibration_log_writer.writerow(_CALIBRATION_LOG_FIELDS)
        except Exception as e:  # a failed log must never abort calibration
            print(e, f"Error opening calibration log at {path}")
            self._calibration_log_fh = None
            self._calibration_log_writer = None

    def _log_calibration_sample(
        self, now: float, gaze_score: Optional[float], ear: Optional[float],
        mar: Optional[float], bpm: Optional[float], rr: Optional[float],
    ) -> None:
        """Append one row of raw calibration measures; blank where not sampled this tick."""
        self._open_calibration_log()
        if self._calibration_log_writer is None:
            return
        try:
            self._calibration_log_writer.writerow([
                now, round(now - self._calibration_start_t, 3),
                gaze_score, ear, mar, bpm, rr,
            ])
        except Exception as e:  # noqa: BLE001
            print(e, "Error writing calibration log row")

    def _close_calibration_log(self) -> None:
        if self._calibration_log_fh is not None:
            try:
                self._calibration_log_fh.close()
            except Exception:
                pass
        self._calibration_log_fh = None
        self._calibration_log_writer = None

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
        """Adopt this participant's stored baselines, skipping the live routine.

        Returns True only when a well-formed file was found and applied; on any
        other outcome the caller runs the live calibration instead.
        """
        path = self.calibration_file()
        if path is None:
            print("[Calibration] No participantid in context — cannot reuse a "
                  f"stored calibration; running the {self._calibration_duration_s:.0f} s calibration.")
            return False
        if not os.path.exists(path):
            print(f"[Calibration] No stored calibration at {path} — "
                  f"running the {self._calibration_duration_s:.0f} s calibration.")
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                cal = json.load(f)
        except Exception as e:  # unreadable / invalid JSON
            print(e, f"Error reading {path}; running the "
                     f"{self._calibration_duration_s:.0f} s calibration")
            return False
        reason = self._validate_calibration(cal)
        if reason is not None:
            print(f"[Calibration] Ignoring {path} ({reason}) — "
                  f"running the {self._calibration_duration_s:.0f} s calibration.")
            return False

        # Copy field-by-field as floats: keeps `calibrate` exactly the shape the
        # live routine produces (no stray keys from an older file leaking in).
        self.calibrate = {
            key: {field: float(cal[key][field]) for field in fields}
            for key, fields in _CALIBRATION_SCHEMA.items()
        }
        # Diagnostic metadata, outside the schema (see compute_calibration), so
        # the comprehension above would drop it. Carried through and echoed
        # because the rate this file was recorded at is exactly what a later
        # session has to be compared against — and files written before this
        # existed simply have no entry.
        _sr = cal.get('sample_rate')
        if isinstance(_sr, dict):
            self.calibrate['sample_rate'] = {
                k: float(v) for k, v in _sr.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
        self.calibrated = True
        self._session_start_t = time.monotonic()
        print(f"[Calibration] Loaded calibration from {path} — "
              f"skipping the live calibration.")
        print("Calibration loaded:", self.calibrate)
        _sr = self.calibrate.get('sample_rate')
        if _sr and _sr.get('median_hz'):
            print(f"[Calibration] this baseline was recorded at "
                  f"{_sr['median_hz']:.1f} Hz (median) — compare against the "
                  f"[fps] line once collection starts; a large gap biases the "
                  f"normalized blink_rate.")
        else:
            print("[Calibration] this baseline predates sample-rate recording, "
                  "so the rate it was measured at is unknown.")
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

        Baseline and std come from the live calibration for ``cal_key`` when it
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
        with self._phase('visual.frame'):
            frame = self._get_capture_frame()
        if frame is None:
            self.latest_frame = None
            return False

        # MediaPipe Tasks FaceLandmarker — the SINGLE face detector for the whole
        # pipeline: gaze, EAR/MAR, and the rPPG/emotion crop box all key off these
        # landmarks. Replaces both the legacy solutions FaceMesh and the separate
        # Haar cascade, so there is no longer a second detector to disagree with.
        h_img, w_img = frame.shape[:2]
        with self._phase('visual.landmarks'):
            landmarks = self.compute_face_landmarks(frame)
        face_present = bool(landmarks)
        data['face_present'] = face_present

        # Publish the raw frame for the decoupled YOLO worker (see _yolo_loop).
        with self._yolo_lock:
            self._yolo_frame = frame
            self._yolo_frame_id += 1

        # gaze score — get_gaze_score returns the per-driver baseline when no face
        # is present, so the standardized value below is neutral (0.0 deviation).
        with self._phase('visual.gaze'):
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
                last_hr_raw, last_hr_raw_t = self._last_hr_raw, self._last_hr_raw_t
                last_hr_rejected = self._last_hr_rejected
                bpm_hist = list(self.bpm_history)
                rr_hist = list(self.rr_history)
            # Unfiltered HR + whether the live filter rejected it. Post-hoc only —
            # NOT model inputs (encode_frame ignores them, and they are dropped
            # from the decisions.csv schema). These are what the offline pipeline
            # re-derives hr_delta from, so they are logged even on frames where
            # the filtered heart_rate is absent.
            if last_hr_raw is not None and (now - last_hr_raw_t) <= self._rppg_staleness_s:
                data['heart_rate_raw'] = last_hr_raw
                data['hr_rejected'] = bool(last_hr_rejected)
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

            # rPPG data quality (dashboard/post-hoc only; not model inputs).
            # rppg_gaps counts look-aways that broke the model's input window;
            # rppg_dropped counts frames lost to a full queue. Both silently
            # invalidate the FFT's uniform-sampling assumption, so a session
            # with a high count should be treated with suspicion.
            data['rppg_gaps'] = int(self._rppg_gap_count)
            data['rppg_dropped'] = int(getattr(self.rppg_estimator, 'dropped_frames', 0))
            data['rppg_suppressed'] = int(self._rppg_suppressed)
            # Readings dropped as probable 2f harmonics. A high count means the
            # BVP fundamental is weak (crop, lighting or motion) — worth checking
            # before trusting the session's HR at all.
            data['rppg_harmonic_rejects'] = int(self._hr_harmonic_rejects)
            # Detector misses are the upstream cause of most rPPG dropouts.
            data['facebox_misses'] = int(self._face_box_misses)
            data['facebox_consec_misses'] = int(self._face_box_consec_misses)

        # Emotion — EmotiEffLib on the RGB tight face crop.
        if emotion_crop_bgr is not None:
            with self._phase('visual.emotion'):
                emo = self.detect_emotion(cv2.cvtColor(emotion_crop_bgr, cv2.COLOR_BGR2RGB))
            if emo:
                data.update(emo)  # emotion, emotion_prob

        # EAR/MAR from the landmarks (sentinels when no face, as before).
        if face_present:
            try:
                with self._phase('visual.earmar'):
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
        with self._phase('visual.overlay'):
            frame_annot = _perception.draw_overlays(frame, landmarks, yolo_dets)

        # Yawn gate in MILLISECONDS, not frames. The old `MOUTH_AR_CONSEC_FRAMES = 3`
        # meant 750 ms at the pipeline's former ~4 fps but only 300 ms once the loop
        # was parallelized to ~10 fps — i.e. the gate silently became 2.5x more
        # permissive, and 300 ms of mouth opening is ordinary speech. A yawn runs
        # 4-7 s, so 2 s of sustained opening separates it from talking and is
        # immune to any future change in frame rate.
        MOUTH_OPEN_MIN_MS = 2000
        BLINK_MIN_MS = 100   # below this is EAR noise, not a blink (this is equivalent to 2 frames at 20Hz)
        BLINK_MAX_MS = 500  # above this is eye closure / drowsiness, not a blink
        blink_window = 60.0  # 60 second window for blinks
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

            # yawn detection (duration-gated, mirrors the blink detector above)
            if mouth_open:
                if self._mouth_open_since == 0.0:
                    self._mouth_open_since = now
            else:
                if self._mouth_open_since > 0.0:
                    if (now - self._mouth_open_since) * 1000 >= MOUTH_OPEN_MIN_MS:
                        self.yawn_count += 1
                        self.yawn_times.append(now)
                self._mouth_open_since = 0.0

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
            self._mouth_open_since = 0.0


        self.latest_frame = frame_annot
        
        # Add the computed metrics to the data dictionary (normalized)
        # normalize blink rate (it is a Poisson divided by a specific value - Variance=X/n)
        normalized_blink_rate = (self.blink_rate - self.calibrate.get('blink_rate', {}).get('mean', 0.0)) / math.sqrt((self.calibrate.get('blink_rate', {}).get('mean', 1.0)/(elapsed_s / 60))) if (self.calibrate.get('blink_rate', {}).get('mean', 0.0) > 0 and elapsed_s > 0) else 0.0
        data['blink_rate'] = round(float(normalized_blink_rate), 3)
        # Anscombe transform to normalize yawn rate. Anscombe stabilizes the
        # VARIANCE of a Poisson count (Var ~ 1) but does not CENTER it, so the
        # raw transform bottoms out at 2*sqrt(3/8) = 1.2247 when no yawns have
        # occurred and only ever rises from there — a large constant offset on a
        # feature that sits next to z-scored ones. Subtracting the transform of
        # the baseline count re-centers it: 0 yawns -> 0.0, 1 -> 1.12, 2 -> 1.86.
        # No division: Var[A(X)] ~ 1 already, which is the point of Anscombe.
        # Baseline count is fixed at 0 rather than calibrated — an alert driver
        # yawns ~0 times in the 3 min baseline, so a measured lambda_0 would carry
        # ~100% relative error and add noise instead of removing it.
        yawn_count_in_window = self.yawn_rate * (yawn_window / 60)
        normalized_yawn_rate = 2*math.sqrt(yawn_count_in_window + 3/8) - _YAWN_ANSCOMBE_ZERO
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
        # Bridge-only, so None here covers the direct-CARLA path and the
        # no-CARLA fallback; the bridge branch below overwrites both. See the
        # warning in the direct path for why it is not computed there.
        lead_distance_m = headway_s = None

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

                # World and Map are CACHED, not re-fetched per frame.
                #
                # get_world() built a fresh World wrapper every tick and
                # get_map() re-downloaded and re-parsed the entire OpenDRIVE
                # description behind it — at 20 Hz, twenty full map parses a
                # second, purely to read one `is_junction` flag. The project's
                # own bridge already caches exactly this
                # (scripts/vehicle_state_server.py: "get_map() is expensive to
                # call at 20 Hz"); the in-process path never got the same fix.
                #
                # Neither object changes for the life of a session (the map is
                # only reloaded by loading a new town, which this experiment
                # never does), so caching is safe as well as far cheaper.
                if self._carla_world is None:
                    self._carla_world = self.carla_vehicle.get_world()
                    self._carla_map = self._carla_world.get_map()
                    print("[DataCollector] Cached CARLA world and map "
                          "(previously re-fetched every frame).")
                world = self._carla_world
                weather = world.get_weather()
                precipitation = round(weather.precipitation / 100.0, 3)
                is_night = bool(weather.sun_altitude_angle < 0)
                fog_density = round(weather.fog_density / 100.0, 3)

                location = self.carla_vehicle.get_location()
                waypoint = self._carla_map.get_waypoint(location)
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

                # Headway is NOT computed on this path, and the gap is loud
                # rather than silent because a null column looks like "the road
                # was clear for the whole session" instead of "this was never
                # measured".
                #
                # It lives in the bridge (vehicle_state_file_bridge.py:
                # lead_and_headway) because that is where the ONE definition of
                # the record lives -- the same reason collect() is shared
                # between the file and HTTP bridges. Reimplementing it here
                # would give the same column two definitions that could drift.
                # This path is also the one that "reliably corrupts its heap"
                # (see --vehicle-bridge in start_experiment.py), so it is not
                # the path a data-collection run should be on: adding a
                # per-frame world snapshot and lane walk to it would be adding
                # CARLA traffic to the process that must not hold a CARLA
                # client.
                if not self._warned_no_headway:
                    self._warned_no_headway = True
                    print("[DataCollector] WARNING: reading CARLA directly, so "
                          "lead_distance_m and headway_s will be null for this "
                          "whole session. Use --vehicle-bridge for any run "
                          "whose data will be kept.")

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
        
        elif self.vehicle_state_url is not None or self.vehicle_state_path is not None:
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
            lead_distance_m = self._cached_lead_distance
            headway_s = self._cached_headway_s

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
        # Traffic context. null means "no vehicle ahead in this lane within
        # 100 m" for the distance, and additionally "ego below walking pace" for
        # the headway -- neither is a missing value to be imputed, and neither
        # should be filled with 0 (a 0 s headway is a collision).
        data['lead_distance_m']     = lead_distance_m
        data['headway_s']           = headway_s
        # Static for the session, repeated per frame so a raw_data.jsonl line is
        # self-describing: which traffic scenario produced it is not recoverable
        # from any other field, and joining to a separate manifest is one more
        # thing that can be lost between the two machines.
        data['traffic_seed']        = self.traffic_seed

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
            # Unfiltered HR + live-filter verdict, for post-hoc re-derivation.
            data['heart_rate_raw'] = None
            data['hr_rejected'] = None

        if self.visual_enabled:
            with self._phase('visual'):
                ok = self._visual_process(data)
            if not ok:
                return data, False

        if self.context_enabled:
            with self._phase('carla'):
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
                    # calibration_only always measures fresh — adopting the
                    # stored file would exit before recording anything.
                    loaded = False if self.calibration_only else self.load_calibration()
                    if not loaded and self.data_collection:
                        # Live routine waived and no baseline to reuse. Take the
                        # neutral defaults compute_calibration() already falls
                        # back to when no samples exist, unsaved, so the driver's
                        # real calibration is never overwritten by zeros.
                        print("[Calibration] --data-collection: no stored baseline to "
                              "reuse, starting with neutral defaults. Features that "
                              "are normalized per driver will be uncalibrated.")
                        self.compute_calibration(save=False)

                if self.calibrated is False:
                    # Two-phase time-based calibration — completion handled inside calibrate_step()
                    self.calibrate_step()
                    if self.calibrated and self.calibration_only:
                        # calibrate_step() just finished and save_calibration()
                        # has run. Nothing else in this process is wanted.
                        print("[Calibration] --calibration-only: baseline stored, "
                              "shutting down.")
                        self._running = False
                        if self.on_calibration_complete is not None:
                            try:
                                self.on_calibration_complete()
                            except Exception as e:
                                print(f"[Calibration] completion callback failed: {e}")
                        break
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
                        if (self._prof is not None
                                and _dbg_t0 - self._prof_last_print_t >= self._prof_period_s):
                            self._prof_last_print_t = _dbg_t0
                            self._print_profile(_dbg_avg)
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
                        with self._phase('history'):
                            seq_entry = dict(data)
                            seq_entry.pop('bpm_history', None)
                            seq_entry.pop('rr_history', None)
                            with self._lock:
                                self.data_history.append(seq_entry) # no need to pop as we use deque with maxlen now

                        # Deciding, logging the decision and actuating all happen
                        # on the decision thread (_decision_loop), which reads the
                        # snapshot published above. This loop only records the
                        # frame.
                        if self.logger:
                            with self._phase('log_raw'):
                                raw_with_decision = dict(data)
                                # LoA/FCD now mean "the decision in force as this
                                # frame was recorded" rather than "computed from
                                # this frame" — stale by at most one decision
                                # period. Nothing in training reads them
                                # (build_loa_dataset.py labels from
                                # user_selected_loa, and FCD is static per task),
                                # so this costs the pipeline nothing.
                                #
                                # Both keys are written UNCONDITIONALLY, even
                                # before the first decision exists: omitting them
                                # would shift every following field's position in
                                # frames logged at session start, the same
                                # ordering problem the phys keys are pre-declared
                                # to avoid in collect_data().
                                with self._lock:
                                    action = self._last_action
                                raw_with_decision['LoA'] = (
                                    action.get('LoA') if isinstance(action, dict) else None)
                                fcd = (action.get('fcd') or action.get('fcd_scores')
                                       if isinstance(action, dict) else None)
                                raw_with_decision['FCD'] = fcd if isinstance(fcd, dict) else None
                                raw_with_decision.pop('bpm_history', None)
                                raw_with_decision.pop('rr_history', None)
                                self.logger.log_raw(raw_with_decision)
                            if not self._first_frame_logged:
                                # AFTER the write, so the signal can never
                                # precede the data it announces: a Drive that
                                # started its windows on this signal would
                                # otherwise label a window whose first frames
                                # are not in the file yet.
                                self._first_frame_logged = True
                                if self.on_first_frame_logged is not None:
                                    try:
                                        self.on_first_frame_logged()
                                    except Exception as e:  # noqa: BLE001
                                        print("[DataCollector] first-frame "
                                              f"callback failed: {e}")

            except Exception as e:
                import traceback; traceback.print_exc()
                print('[DataCollector] loop error:', e)

            next_t += self.sampling_interval
            now = time.monotonic()
            if next_t < now: # fell behind — skip missed ticks
                next_t = now
            # 'sleep' near zero means the loop never idles: it is running flat
            # out and every other thread (capture, face-box, YOLO) is competing
            # with it for slices.
            with self._phase('sleep'):
                time.sleep(max(0.0, next_t - now))

        print("data collector stopped!")

    def _decision_loop(self) -> None:
        """Background thread: run the decision engine off the collection loop.

        Snapshots the state the collection loop publishes (``latest_data`` plus
        the ``data_history`` window), runs the engine on it, then logs the
        decision and actuates. The collection loop never waits on any of this,
        so perception cadence is decoupled from inference cost.

        Ownership of the two log streams is split so each file keeps a SINGLE
        writer thread: ``log_raw`` stays on the collection loop, ``log_processed``
        lives here. Logger takes a lock regardless, but the split is what makes
        the invariant easy to keep.
        """
        if not self.decision_engine:
            print("[DataCollector] No decision engine; decision worker not started.")
            return
        print(f"[DataCollector] Decision worker started "
              f"({1.0 / self.decision_interval:.1f} Hz).")
        next_t = time.monotonic()
        while self._running:
            try:
                if self.calibrated:
                    # The lock is not optional here: list() iterates the deque and
                    # the collection loop appends to it from another thread — an
                    # unguarded copy raises "deque mutated during iteration".
                    with self._phase('decide.copy'):
                        with self._lock:
                            seq = list(self.data_history)
                            last_ts = self._last_decided_ts

                    # Both the per-frame context and the window come from the
                    # SAME source — the newest entry of the window itself, not
                    # latest_data. latest_data is published a moment before that
                    # frame reaches data_history, so reading it here could pair a
                    # frame-N timestamp with a window ending at frame N-1 and put
                    # a decisions.csv row on the wrong side of a 20 s label-window
                    # boundary. seq[-1] carries every field the engine and the CSV
                    # schema need (static context is merged in collect_data), so
                    # nothing is lost by preferring it.
                    snap = dict(seq[-1]) if seq else {}
                    ts = snap.get('timestamp')
                    # Skip when the collector has produced no new frame since the
                    # last decision (decision_hz above the achieved capture rate,
                    # or a stalled camera). Keeps decisions.csv at one row per
                    # distinct frame, so each row still joins exactly onto one
                    # raw_data.jsonl line.
                    if seq and (ts is None or ts != last_ts):
                        snap['functionname'] = snap.get('functionname') or self.functionname
                        snap['sequence'] = seq

                        with self._phase('decide.run'):
                            action = self.decision_engine.decide(snap)

                        with self._lock:
                            self._last_action = action if isinstance(action, dict) else None
                            self._last_decided_ts = ts

                        if self.logger and isinstance(action, dict):
                            with self._phase('decide.log'):
                                action_for_log = dict(action)
                                # Context comes from the SAME snapshot the
                                # decision was computed on, so the row's context
                                # can never disagree with the model's input.
                                # 'timestamp' stays the frame's, not the decision
                                # wall-clock: that is what keeps the join to
                                # raw_data.jsonl and to the 20 s label windows in
                                # build_loa_dataset.py exact.
                                for key in ('timestamp', 'session_id', 'participantid', 'environment', 'secondary_task', 'functionname', 'emotion', 'modeltype', 'state_model', 'w_fcd', 'hr_delta', 'rr_delta'):
                                    value = snap.get(key)
                                    if value not in (None, ''):
                                        action_for_log.setdefault(key, value)
                                self.logger.log_processed(action_for_log)

                        if self.actuator and action is not None:
                            with self._phase('decide.actuate'):
                                self.actuator.execute(action)

            except Exception as e:
                import traceback; traceback.print_exc()
                print('[DataCollector] decision loop error:', e)

            next_t += self.decision_interval
            now = time.monotonic()
            if next_t < now:      # fell behind — skip missed ticks
                next_t = now
            time.sleep(max(0.0, next_t - now))

        print("[DataCollector] Decision worker stopped.")

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
                crop = None
                if box is not None and (now - box_t) <= self._rppg_box_staleness_s:
                    x, y, bw, bh = box
                    crop = frame[y:y + bh, x:x + bw]
                    if crop.size == 0:
                        crop = None
                if crop is not None:
                    # --- continuity check (before feeding) ---------------
                    gap_limit = self._rppg_gap_frames / max(1e-6, self._rppg_fps)
                    if self._rppg_last_fed_t and (now - self._rppg_last_fed_t) > gap_limit:
                        self._rppg_gap_count += 1
                        self._rppg_contig = 0
                        if self._rppg_gap_count <= 5 or self._rppg_gap_count % 50 == 0:
                            print(f"[DataCollector][rppg] frame gap "
                                  f"{now - self._rppg_last_fed_t:.2f}s "
                                  f"(> {gap_limit:.2f}s) — window spans a "
                                  f"discontinuity (gap #{self._rppg_gap_count}"
                                  f"{'; suppressing' if self._rppg_gap_suppress else ''})")
                    else:
                        self._rppg_contig += 1
                    self._rppg_last_fed_t = now

                    try:
                        hr, rr = self.rppg_estimator.add_frame(crop)
                    except Exception as e:  # noqa: BLE001
                        print(e, "Error feeding rPPG estimator")
                        hr = rr = None
                    # A reading surfaces only every inference_interval frames;
                    # None in between. Publish for the main loop to read —
                    # but only once the model's whole input window is free of
                    # pre-gap frames.
                    window = int(getattr(self.rppg_estimator, 'num_frames', 181))
                    contaminated = self._rppg_contig < window
                    if (hr is not None or rr is not None) and contaminated:
                        self._rppg_suppressed += 1   # counted whether or not we act on it
                    clean = (not contaminated) or (not self._rppg_gap_suppress)
                    if clean and hr is not None and not (hr != hr):   # excludes NaN
                        hr_val = round(float(hr), 1)
                        rejected = (self._hr_is_harmonic(hr_val)
                                    and self._hr_reject_run < _HR_REJECT_RUN_MAX)
                        # Record the raw reading FIRST, unconditionally, so the
                        # live filter can never destroy data the offline pipeline
                        # needs (see _last_hr_raw).
                        with self._rppg_lock:
                            self._last_hr_raw = hr_val
                            self._last_hr_raw_t = now
                            self._last_hr_rejected = rejected
                        # Drop probable 2f locks, but never more than
                        # _HR_REJECT_RUN_MAX in a row — past that the reference
                        # is likelier to be wrong than the readings.
                        if rejected:
                            self._hr_reject_run += 1
                            self._hr_harmonic_rejects += 1
                            if (self._hr_harmonic_rejects <= 5
                                    or self._hr_harmonic_rejects % 25 == 0):
                                print(f"[DataCollector][rppg] dropped {hr_val} bpm as a "
                                      f"probable 2f harmonic of "
                                      f"~{float(np.median(self._hr_accepted)):.0f} bpm "
                                      f"(reject #{self._hr_harmonic_rejects})")
                        else:
                            if self._hr_reject_run >= _HR_REJECT_RUN_MAX:
                                print(f"[DataCollector][rppg] {self._hr_reject_run} "
                                      f"consecutive rejections — re-seeding the HR "
                                      f"reference from {hr_val} bpm")
                                self._hr_accepted.clear()
                            self._hr_reject_run = 0
                            self._hr_accepted.append(hr_val)
                            with self._rppg_lock:
                                self._last_hr = hr_val
                                self._last_hr_t = now
                                self.bpm_history.append(hr_val)
                    if clean and rr is not None and not (rr != rr):
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

    @staticmethod
    def _make_yolo5face():
        """The YOLO5Face detector MMRPhys was trained with, or None.

        Weights (Y5sF_WFRGB.pth, WiderFace) ship inside the installed mmrphys
        package. Importing it normally would execute
        ``mmrphys.dataset.data_loader.__init__``, which pulls in every dataset
        loader and therefore mat73 / skimage — heavy deps this project does not
        need. Pre-registering a stand-in package module with the right __path__
        makes Python skip that __init__ while still resolving the submodules.
        """
        import importlib
        import sys as _sys
        import types as _types
        pkg = "mmrphys.dataset.data_loader"
        if pkg not in _sys.modules:
            _ds = importlib.import_module("mmrphys.dataset")
            shim = _types.ModuleType(pkg)
            shim.__path__ = [os.path.join(os.path.dirname(_ds.__file__), "data_loader")]
            shim.__package__ = pkg
            _sys.modules[pkg] = shim
        mod = importlib.import_module(pkg + ".face_detector.YOLO5Face")
        y5f = mod.YOLO5Face("Y5F", None)   # None -> CUDA when available, else CPU
        # Detect at 320 rather than the default 640. detect_face() scales the
        # frame by img_size/max(h, w) and letterboxes to a square, then maps the
        # result back with scale_coords -- so the returned box is still in
        # full-frame coordinates and the box DISTRIBUTION is unchanged; only the
        # compute drops (measured 79.7 ms -> 33.0 ms per call, 2.4x).
        #
        # This matters because the worker refreshes the crop box: under load a
        # detection inflates ~10x, pushing the refresh period past
        # _rppg_box_staleness_s, which stalls rPPG feeding entirely. Halving the
        # cost keeps the period clear of that threshold. A driver's face fills a
        # large part of a 640x480 frame, so it stays far above the detector's
        # small-face limit. Raise back to 640 if detections start to be missed.
        y5f.img_size = 320
        return y5f

    def _face_box_loop(self) -> None:
        """Background thread: maintain the square rPPG crop box.

        Uses YOLO5Face — the same WiderFace detector MMRPhys was trained with —
        so the crop box comes from the same distribution as the training data
        instead of being approximated from a MediaPipe landmark hull. Falls back
        to a private FaceLandmarker instance if YOLO5Face cannot be loaded
        (MediaPipe task objects are not safe to share across threads, hence a
        second instance; the main loop keeps its own for gaze/EAR/MAR).

        Cadence matches the reference exactly: the upstream loader re-detects
        every DYNAMIC_DETECTION_FREQUENCY frames, which equals FS for every
        dataset -> 1 Hz, holding the box constant in between. That also keeps
        the cost sane: YOLO5Face is ~106 ms/call on CPU (~11% of one core at
        1 Hz, but ~53% at 5 Hz).

        Publishing the box from here rather than from the main loop means the
        capture worker never crops with a box that went stale because the
        decision engine blocked.
        """
        detect = None       # frame(BGR) -> (x, y, w, h) or None
        detector_name = "?"
        try:
            y5f = self._make_yolo5face()

            def detect(frame):                                   # noqa: F811
                # Upstream feeds RGB to the detector (its loaders convert on read).
                res = y5f.detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not res:
                    return None
                x1, y1, x2, y2 = res
                return x1, y1, x2 - x1, y2 - y1

            detector_name = "YOLO5Face"
            print("[DataCollector] Face-box worker: YOLO5Face (matches MMRPhys training).")
        except Exception as e:  # noqa: BLE001
            print(e, "YOLO5Face unavailable; face-box worker falling back to MediaPipe")

        if detect is None:
            if not HAS_MP:
                print("[DataCollector] No face detector available; face-box worker exiting.")
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

            def detect(frame):                                   # noqa: F811
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,   # type: ignore
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = self._box_landmarker.detect(mp_img)
                if not res.face_landmarks:
                    return None
                h_i, w_i = frame.shape[:2]
                # margin=0: the 1.5x enlargement below supplies the padding.
                return self._bbox_from_landmarks(res.face_landmarks[0], w_i, h_i, margin=0.0)

            detector_name = "MediaPipe"

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
                    raw = detect(frame)
                except Exception as e:  # noqa: BLE001
                    print(e, "Error detecting face box")
                    raw = None
                now_d = time.monotonic()
                if raw is not None:
                    # Square + 1.5x, exactly as at training time.
                    box = self._rppg_crop_box(raw[0], raw[1], raw[2], raw[3],
                                              w_img, h_img)
                    if box[2] > 0 and box[3] > 0:
                        if self._face_box_consec_misses:
                            print(f"[DataCollector][facebox] {detector_name} "
                                  f"re-acquired the face after "
                                  f"{self._face_box_consec_misses} miss(es) "
                                  f"({now_d - self._face_box_last_ok_t:.1f}s blind)")
                        self._face_box_consec_misses = 0
                        self._face_box_last_ok_t = now_d
                        with self._box_lock:
                            self._rppg_box = box
                            self._rppg_box_t = now_d
                else:
                    # No face: the box is NOT refreshed, so it ages. Once it
                    # passes _rppg_box_staleness_s the capture worker stops
                    # feeding rPPG entirely — report every miss, with enough
                    # context to tell "driver looked away" from "detector cannot
                    # see this face at all".
                    self._face_box_misses += 1
                    self._face_box_consec_misses += 1
                    blind = (f"{now_d - self._face_box_last_ok_t:.1f}s since last detection"
                             if self._face_box_last_ok_t
                             else "NEVER detected a face this session")
                    with self._box_lock:
                        box_age = now_d - self._rppg_box_t if self._rppg_box_t else None
                    stale = (box_age is None
                             or box_age > self._rppg_box_staleness_s)
                    print(f"[DataCollector][facebox] {detector_name} found NO face "
                          f"(consecutive {self._face_box_consec_misses}, "
                          f"total {self._face_box_misses}, "
                          f"{blind}) — "
                          + ("crop box STALE: rPPG is no longer being fed"
                             if stale else
                             f"reusing box, {self._rppg_box_staleness_s - (box_age or 0.0):.1f}s "
                             f"before rPPG stops"))
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

    def _close_state_connection(self) -> None:
        """Drop the pooled HTTP connection; the next poll opens a fresh one."""
        conn, self._http_conn = self._http_conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001 — already-dead socket
                pass

    def _http_get_state(self) -> "Tuple[Dict[str, Any], Optional[float]]":
        """One GET from the remote bridge over a PERSISTENT connection.

        Returns (record, age_seconds). The age comes from the bridge's
        X-State-Age header, NOT from the record's "ts": ts is the wall clock of
        the machine running CARLA, and between two machines those clocks
        disagree by seconds, which would make a perfectly fresh feed look stale
        (or a frozen one look current). X-State-Age is measured entirely on the
        bridge with a monotonic clock, so it survives both skew and an NTP
        step. None means the bridge did not send it (an older server), in which
        case the caller simply does not staleness-check.

        The connection is kept open across polls, so the steady state is one
        request on one socket. A bridge restart or an idle timeout closes it
        from the far end; that failure only shows up on the next use, so a
        connection we had already used is retried ONCE on a fresh socket rather
        than being reported as an outage.
        """
        last_err = None
        for _attempt in (0, 1):
            reused = self._http_conn is not None
            try:
                if self._http_conn is None:
                    parts = urllib.parse.urlsplit(self.vehicle_state_url)
                    path = parts.path or "/"
                    if parts.query:
                        path = f"{path}?{parts.query}"
                    self._http_path = path
                    cls = (http.client.HTTPSConnection if parts.scheme == "https"
                           else http.client.HTTPConnection)
                    self._http_conn = cls(parts.hostname,
                                          parts.port,
                                          timeout=self._state_http_timeout)
                self._http_conn.request(
                    "GET", self._http_path,
                    headers={"Accept": "application/json",
                             # Explicit, though it is HTTP/1.1's default: this
                             # header is the whole point of the exercise.
                             "Connection": "keep-alive"})
                resp = self._http_conn.getresponse()
                # Draining the body is what makes the connection reusable — an
                # unread response leaves the socket mid-message and the next
                # request would raise ResponseNotReady.
                body = resp.read()
                print(f"[DataCollector] HTTP GET {self.vehicle_state_url} "
                      f"-> {resp.status} {resp.reason} ({len(body)} bytes)")
                if resp.status != 200:
                    raise ValueError(f"bridge returned HTTP {resp.status} {resp.reason}")
                age_hdr = resp.getheader("X-State-Age")
                age = None
                if age_hdr is not None:
                    try: 
                        age = float(age_hdr)
                    except ValueError:
                        age = None
                return json.loads(body), age
            except ValueError:
                # A well-formed answer we do not like (503 while the bridge is
                # still connecting, or unparseable JSON). The connection itself
                # is fine and retrying immediately would not help.
                raise
            except (OSError, http.client.HTTPException) as e:
                last_err = e
                self._close_state_connection()
                if not reused:
                    # A brand-new connection failing is a real outage, not a
                    # stale pooled socket: report it instead of hammering.
                    raise
        raise last_err  # pragma: no cover — the loop above always returns or raises

    def _poll_vehicle_state(self, interval: Optional[float] = None) -> None:
        """Background thread: fetch vehicle state from the bridge.

        Cadence comes from state_poll_hz (see __init__); the argument is kept
        only so an explicit override still works.
        """
        interval = self._state_poll_interval if interval is None else interval
        source = self.vehicle_state_path or self.vehicle_state_url
        kind = "file" if self.vehicle_state_path else "http"
        print(f"[DataCollector] Vehicle-state poller started "
              f"({1.0 / interval:.1f} Hz, {kind} -> {source}).")
        _fail_streak = 0
        _stale_since = 0.0
        _channel = None
        # ISOLATION CONTRACT: nothing the bridge does may take ProVoice down.
        # The bridge is a separate process that can die, be restarted by the
        # launcher, hold the lock too long, publish garbage, or never start at
        # all. Every one of those raises inside this try, is logged, and leaves
        # the _cached_* fields at their last good values -- which _carla_process
        # reads. Collection, perception, decisions and logging carry on
        # untouched; only the vehicle columns go stale. This thread must never
        # propagate an exception, because that would kill the thread and freeze
        # the vehicle state silently for the rest of the session.
        while self._running:
            try:
                age = None
                if self.vehicle_state_path:
                    # read() takes the same advisory lock the bridge holds while
                    # writing, so a half-written record cannot be observed. A
                    # missing or empty file just means the bridge has not
                    # published yet, and raises into the handler below.
                    if _channel is None:
                        _channel = VehicleStateChannel(self.vehicle_state_path)
                    state = _channel.read()
                    _ts = state.get("ts")
                    if _ts is not None:
                        # Writer and reader share this machine's clock, so the
                        # record's own wall-clock stamp gives the true age.
                        age = time.time() - float(_ts)
                else:
                    # Remote bridge: ONE connection reused across polls, and an
                    # age the bridge measured itself — the two machines' wall
                    # clocks disagree, so "ts" cannot be used here.
                    state, age = self._http_get_state()

                # Stale means the bridge stopped updating: the fields below are
                # then frozen history being written into rows as if current, so
                # it has to be said out loud.
                if age is not None and age > max(2.0, 10.0 * interval):
                    if _stale_since == 0.0:
                        _stale_since = time.time()
                    if _fail_streak == 0:
                        print(f"[DataCollector] Vehicle state is STALE "
                              f"({age:.1f}s old) — the bridge has stopped "
                              f"publishing; values are frozen.")
                        _fail_streak = 1
                    raise ValueError(f"stale vehicle state ({age:.1f}s)")
                _stale_since = 0.0

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
                # NOT coerced, and NOT defaulted to the last value like the
                # fields above. None is a meaningful reading here -- "no vehicle
                # ahead in my lane" -- so it has to survive as None rather than
                # be replaced by the last real gap, which would leave a stale
                # leader in the data for the rest of the drive after the road
                # cleared. float(None) would also raise, taking the whole poll
                # with it. .get(key, cached) cannot express this: the key is
                # PRESENT and its value is null.
                if "lead_distance_m" in state:
                    v = state["lead_distance_m"]
                    self._cached_lead_distance = float(v) if v is not None else None
                if "headway_s" in state:
                    v = state["headway_s"]
                    self._cached_headway_s = float(v) if v is not None else None
                if _fail_streak:
                    print(f"[DataCollector] Vehicle-state bridge back after "
                          f"{_fail_streak} failed poll(s) "
                          f"(~{_fail_streak * interval:.1f}s of frozen state).")
                    _fail_streak = 0
            except Exception as e:  # noqa: BLE001 — keep the last cached values
                # Silence here would be dangerous: every vehicle field simply
                # STOPS UPDATING and the rows keep being written as if fine.
                # Report the start of an outage and then throttle, so a long
                # outage is visible in the log without flooding it at 20 Hz.
                _fail_streak += 1
                if _fail_streak == 1 or _fail_streak % 100 == 0:
                    print(f"[DataCollector] Vehicle-state poll FAILED "
                          f"(#{_fail_streak}, ~{_fail_streak * interval:.1f}s "
                          f"frozen): {type(e).__name__}: {e}")
                # Drop the cached handle so the next tick reopens. The launcher
                # restarts a dead bridge, and a stale handle must not stop us
                # picking the new one up.
                #
                # The HTTP connection is deliberately NOT dropped here:
                # _http_get_state() already closes it for every failure that
                # implicates the socket, and the failures that reach this point
                # with a healthy connection (a 503 while the bridge is still
                # connecting to CARLA, a stale record) would otherwise force a
                # reconnect per tick — recreating exactly the connection churn
                # the keep-alive path exists to remove.
                if _channel is not None:
                    try:
                        _channel.close()
                    except Exception:
                        pass
                    _channel = None
            except BaseException as e:  # noqa: BLE001
                # Belt and braces for the isolation contract above: even a
                # non-Exception escape (bar the shutdown signals) must not be
                # allowed to kill this thread and silently freeze vehicle state.
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                print(f"[DataCollector] Vehicle-state poller: unexpected "
                      f"{type(e).__name__}: {e} — continuing with cached values.")
                _channel = None
                # Unknown failure: the connection's state is unknown too, so
                # start over rather than reuse something possibly mid-message.
                self._close_state_connection()
            time.sleep(interval)

        # Owning thread closes what it opened, so the socket and the file handle
        # are released at stop() rather than at interpreter teardown.
        self._close_state_connection()
        if _channel is not None:
            try:
                _channel.close()
            except Exception:  # noqa: BLE001
                pass
        print("[DataCollector] Vehicle-state poller stopped.")

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
        if self.decision_engine and not self.data_collection:
            self._decision_thread = threading.Thread(target=self._decision_loop, daemon=True)
            self._decision_thread.start()
        elif self.data_collection:
            print("[DataCollector] --data-collection: inference disabled, "
                  "recording raw data only.")
        if self.visual_enabled:
            self._yolo_thread = threading.Thread(target=self._yolo_loop, daemon=True)
            self._yolo_thread.start()
        if self.vehicle_state_url or self.vehicle_state_path:
            self._state_poll_thread = threading.Thread(target=self._poll_vehicle_state, daemon=True)
            self._state_poll_thread.start()
            print(f"[DataCollector] Vehicle state polling started → "
                  f"{self.vehicle_state_path or self.vehicle_state_url}")

    def stop(self) -> None:
        # main.py calls stop() from the SIGINT handler and again from its
        # finally block, so make it idempotent — the second pass would
        # otherwise re-join dead threads and re-stop the rPPG worker, which is
        # where some of the Ctrl+C noise came from.
        if getattr(self, '_stopped', False):
            return
        self._stopped = True
        # Halt the loop BEFORE stopping the rPPG estimator / releasing the
        # camera, so no tick can touch an already-stopped resource.
        self._running = False
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            # Generous: a decision already in flight must finish before main.py's
            # finally block closes the Logger and the strategies' own handles.
            if self._decision_thread and self._decision_thread.is_alive():
                self._decision_thread.join(timeout=5.0)
            if self._yolo_thread and self._yolo_thread.is_alive():
                self._yolo_thread.join(timeout=2.0)
            if self._face_box_thread and self._face_box_thread.is_alive():
                self._face_box_thread.join(timeout=2.0)
            # Capture worker last among the visual threads: it MUST have exited
            # before release() frees the VideoCapture it is reading from, or
            # cv2 tears the device down under an in-flight cap.read() and the
            # backend raises on the way out (this is what made Ctrl+C noisy).
            # It only ever needs to finish one read, so allow a generous wait —
            # and if it somehow does not exit, skip release() entirely and let
            # process teardown reclaim the device.
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=5.0)
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
        # Never release the device while the capture worker could still be
        # inside cap.read() — that races the backend and surfaces as an error
        # during Ctrl+C. If the worker is still alive, drop our reference and
        # let process teardown reclaim the camera instead.
        cap_thread = self._capture_thread
        if cap_thread is not None and cap_thread.is_alive():
            print("[DataCollector] Capture worker still running — skipping "
                  "cap.release() to avoid racing an in-flight read.")
            self.cap = None
            return
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception as e:
            print(e, "Error releasing camera")
            pass
        self.cap = None

    def get_latest_data(self) -> Dict[str, Any]:
        # last_action is merged in here rather than stored inside latest_data:
        # the collection loop replaces latest_data wholesale every tick, which
        # would wipe the action between decisions. Merging on read also removes
        # the old race where the emitter (which dedupes on 'timestamp', see
        # webui/app.py) could pick up the pre-decision copy of a tick and show a
        # frame with no LoA at all.
        with self._lock:
            data = dict(self.latest_data)
            action = self._last_action
        if action is not None:
            data['last_action'] = action
        return data

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
            action = self._last_action
            frame = self.latest_frame
        if action is not None:
            data['last_action'] = action
        img_b64 = None
        if frame is not None and HAS_CV2:
            try:
                _, buffer = cv2.imencode('.jpg', frame)  # type: ignore
                img_b64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(e, "Error encoding latest frame")
                img_b64 = None
        return {"data": data, "frame_b64": img_b64}
