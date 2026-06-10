from __future__ import annotations

import base64
import datetime
import os
import random
import threading
import time
import urllib.request
import urllib.error
import json
from typing import Any, Dict, Optional, Tuple

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
except NotImplementedError as e:
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
        raise FileNotFoundError(path)
    _emotion_model = load_model(path, compile=False)  # type: ignore
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
        window_size: int = 256,
        vehicle_state_url: Optional[str] = None,
    ) -> None:
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

        self.cap = None
        self.face_mesh = None
        self.carla_vehicle = carla_vehicle
        self.vehicle_state_url = vehicle_state_url.rstrip("/") if vehicle_state_url else None
        self._cached_speed: int = 0
        # 如果有 CARLA actor，尝试获取 vehicle_id
        self.vehicle_id = None
        if self.carla_vehicle is not None:
            try:
                self.vehicle_id = getattr(self.carla_vehicle, "id", None)
            except NotImplementedError as e:
                print("[DataCollector] Error getting vehicle_id from CARLA actor:", e)
                
        if self.visual_enabled:
            try:
                self.cap = cv2.VideoCapture(self.cam_index)  # type: ignore
                print(f"Connecting: {self.cam_index} ...")
                print(f"Camera opened: {self.cap.isOpened()}")

            except NotImplementedError as e:
                print(e, "Error opening camera")
                self.cap = None
                self.visual_enabled = False
            if HAS_MP:
                try:
                    print("[DataCollector] Initializing MediaPipe Face Mesh...")
                    self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)  # type: ignore
                    print("[DataCollector] MediaPipe Face Mesh loaded successfully.")
                except NotImplementedError as e:
                    print(e, "Error initializing face mesh")
                    self.face_mesh = None

            if HAS_MYFRAME:
                print("[DataCollector] Distraction/fatigue perception module detected (MediaPipe + Ultralytics YOLO26).")

            if HAS_RPPG and OnlineRPPG is not None:
                try:
                    self.rppg_estimator = OnlineRPPG(frame_rate=10, crop_size=72)  # type: ignore
                except NotImplementedError as e:
                    print(e, "Error initializing rPPG estimator")
                    self.rppg_estimator = None
                    raise e
        else:
            self.rppg_estimator = None

        if HAS_KERAS and HAS_CV2:
            _load_emotion_model(fer_model_path)

        self.latest_frame = None  # BGR
        self.latest_data: Dict[str, Any] = {}
        self.bpm_history: list = []
        self.rr_history: list = []
        self.data_history: list = []
        self.window_size = window_size
        self.blink_count = 0
        self.yawn_count = 0
        self.perclos = 0.0
        self.drowsiness_alert = False
        self.Roll = 0
        self.Rolleye = 0
        self.Rollmouth = 0
        self.COUNTER = 0
        self.mCOUNTER = 0
        # yawn and blink rates (instead of raw counts) should act as better predictors
        self.blink_times = []
        self.blink_rate = 0.0
        self.yawn_times = []
        self.yawn_rate = 0.0


        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # calibration state
        self.calibrated = False
        self._calibration_data = dict({'gaze_score': [], 'ear': [], 'mar': [], 'bpm': [], 'rr': []})
        self._calibration_frames = 100 # subject to changes...

    def __del__(self):
        self.stop()

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
        except NotImplementedError as e:
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
        except NotImplementedError as e:
            print(e, "Error computing gaze score")
            return 0.0

    def get_gaze_score(self, frame) -> float:
        if not self.face_mesh or not HAS_MP:
            return 0.0
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
            results = self.face_mesh.process(img_rgb)
            if not results.multi_face_landmarks:
                return 0.0
            lm = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            return self.compute_gaze_score(lm, w, h)
        except NotImplementedError as e:
            print(e, "Error computing gaze score")
            return 0.0

    def calibrate_step(self) -> None:
        # check for errors (same as in collect_data)
        if not self.visual_enabled or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            print("not okay")
            self.latest_frame = None
            return
        
        # gaze score
        gaze_score = self.get_gaze_score(frame)
        if gaze_score > 0.0: # filter out frames where gaze score couldn't be computed to avoid skewing calibration
            self._calibration_data['gaze_score'].append(gaze_score)
        
        # load face detector
        if _face_detector is not None and (
                self.rppg_estimator is not None or (_emotion_model is not None and _emotion_input_size is not None)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore
            faces = _face_detector.detectMultiScale(gray, 1.3, 5)  # type: ignore
        else:
            gray = None
            faces = []
        
        # face data for EAR/MAR and Perclos
        if HAS_MYFRAME:
            try:
                ret, frame_annot = _perception.frametest(frame)
                lab, eye, mouth = ret
                if eye > 0.15: # filter out frames where eye is too closed to avoid skewing calibration
                    self._calibration_data['ear'].append(eye)
                self._calibration_data['mar'].append(mouth)
            except Exception as e:  # noqa: BLE001 (perception code crosses C extensions)
                print(e, "Error computing perception.frametest")
                frame_annot = frame
                lab, eye, mouth = ([], 0.3, 0.5)
        else:
            frame_annot = frame
            lab, eye, mouth = ([], 0.3, 0.5)

        self.latest_frame = frame_annot # so frame appears on webpage

    def compute_calibration(self):
        # compute mean and std for each metric, set calibrated flag
        self.calibrate = dict()
        for key, values in self._calibration_data.items():
            if values:
                mean = sum(values) / len(values)
                std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
                # 0.65 threshold for MAR based on literature (doesn't make sense to calibrate based on a closed mouth)
                self.calibrate[key] = {'mean': mean, 'std': std, 'threshold': mean + std * 2.5 if key in ['gaze_score'] else mean - std * 2.5 if key in ['ear'] else 0.65}
            else:
                # default values originally in the script
                thres = 0.2 if key in ['gaze_score', 'ear'] else 0.65
                self.calibrate[key] = {'mean': 0.0, 'std': 0.0, 'threshold': thres}
        self.calibrated = True
        
        print("Calibration completed:", self.calibrate)    
        

    def _visual_process(self, data: Dict[str, Any]) -> None:
        if not self.visual_enabled or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            print("not okay")
            self.latest_frame = None
            return

        # Gaze
        gaze_score = self.get_gaze_score(frame)
        data['gaze_score'] = round(float(gaze_score), 3)
        data['gaze_distracted'] = bool(gaze_score > self.calibrate['gaze_score']['threshold'])

        if _face_detector is not None and (
                self.rppg_estimator is not None or (_emotion_model is not None and _emotion_input_size is not None)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore
            faces = _face_detector.detectMultiScale(gray, 1.3, 5)  # type: ignore
        else:
            gray = None
            faces = []

        # rPPG
        if self.rppg_estimator is not None:  # type: ignore
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = frame[y:y + h, x:x + w]
                hr, rr = self.rppg_estimator.add_frame(face_roi)
                print(f"!!!!!!!!!!!!!!!!!!!!!hr: {hr}")
                print(f"!!!!!!!!!!!!!!!!!!!!!rr: {rr}")
                if hr is not None:
                    # Heart rate
                    data['bpm'] = round(float(hr), 1)
                    data['heart_rate'] = data['bpm']
                    self.bpm_history.append(data['bpm'])
                    if len(self.bpm_history) > 80:
                        self.bpm_history.pop(0)
                    data['bpm_history'] = self.bpm_history
                if rr is not None:
                    # Respiratory Rate
                    data['breaths-per-minute'] = round(float(rr), 1)
                    data['respiratory_rate'] = data['breaths-per-minute']
                    self.rr_history.append(data['breaths-per-minute'])
                    if len(self.rr_history) > 80:
                        self.rr_history.pop(0)
                    data['rr_history'] = self.rr_history
                    
                # set flag if there is an anomalous increase or decrease in hr
                # also add deviation from mean as a feature
                if len(self.bpm_history) >= 10:
                    hr_mean = sum(self.bpm_history) / len(self.bpm_history)
                    hr_std = (sum((x - hr_mean)**2 for x in self.bpm_history) / len(self.bpm_history)) ** 0.5
                    #data['hr_anomaly'] = bool(abs(data.get('bpm', hr_mean) - hr_mean) > 2.5 * hr_std)
                    data['hr_delta'] = round(data.get('bpm', hr_mean) - hr_mean, 1)
                # same for rr
                if len(self.rr_history) >= 10:
                    rr_mean = sum(self.rr_history) / len(self.rr_history)
                    rr_std = (sum((x - rr_mean)**2 for x in self.rr_history) / len(self.rr_history)) ** 0.5
                    #data['rr_anomaly'] = bool(abs(data.get('breaths-per-minute', rr_mean) - rr_mean) > 2.5 * rr_std)
                    data['rr_delta'] = round(data.get('breaths-per-minute', rr_mean) - rr_mean, 1)
                    
        emo = self.detect_emotion(faces, gray)
        if emo:
            data.update(emo)  # emotion, emotion_prob

        if HAS_MYFRAME:
            try:
                ret, frame_annot = _perception.frametest(frame)
                lab, eye, mouth = ret
            except Exception as e:  # noqa: BLE001 (perception code crosses C extensions)
                print(e, "Error computing perception.frametest")
                frame_annot = frame
                lab, eye, mouth = ([], 0.3, 0.5)
        else:
            frame_annot = frame
            lab, eye, mouth = ([], 0.3, 0.5)

        #EYE_AR_THRESH = 0.2
        EYE_AR_CONSEC_FRAMES = 2
        #MAR_THRESH = 0.65
        MOUTH_AR_CONSEC_FRAMES = 3

        if eye < self.calibrate['ear']['threshold']:
            self.COUNTER += 1
            self.Rolleye += 1
        else:
            if self.COUNTER >= EYE_AR_CONSEC_FRAMES:
                self.blink_count += 1
                self.blink_times.append(time.monotonic())

            self.COUNTER = 0

        if mouth > self.calibrate['mar']['threshold']:
            self.mCOUNTER += 1
            self.Rollmouth += 1
        else:
            if self.mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                self.yawn_count += 1
                self.yawn_times.append(time.monotonic())
            self.mCOUNTER = 0

        self.Roll += 1
        if self.Roll >= 50:
            self.perclos = (self.Rolleye / self.Roll) + (self.Rollmouth / self.Roll) * 0.2
            self.drowsiness_alert = self.perclos > 0.38
            self.Roll = 0
            self.Rolleye = 0
            self.Rollmouth = 0

        # yawn and blink rates )
        blink_window = 30.0 # 30 second window for blinks
        now = time.monotonic()
        self.blink_times = [t for t in self.blink_times if now - t <= blink_window]
        data['blink_rate'] = len(self.blink_times) / (blink_window / 60)  # blinks per minute
        yawn_window = 180 # 3 minute window for yawms
        self.yawn_times = [t for t in self.yawn_times if now - t <= yawn_window]
        data['yawn_rate'] = len(self.yawn_times) / (yawn_window / 60)  # yawns per minute
        
        self.latest_frame = frame_annot
        data['blink_count'] = int(self.blink_count)
        data['yawn_count'] = int(self.yawn_count)
        data['perclos'] = round(float(self.perclos), 3)
        data['drowsiness_alert'] = bool(self.drowsiness_alert)
        data['eye_ar'] = float(eye)
        data['mar'] = float(mouth)
        data['lab'] = [str(x) for x in (lab or [])]

    def collect_data(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        
        if self.visual_enabled:
            self._visual_process(data)

        if self.phys_enabled and 'heart_rate' not in data:
            data['heart_rate'] = random.randint(60, 100)

        if self.context_enabled:
            if self.carla_vehicle is not None:
                try:
                    vel = self.carla_vehicle.get_velocity()
                    speed = (vel.x**2 + vel.y**2 + vel.z**2)**0.5
                    speed = int(speed * 3.6)
                    self._cached_speed = speed
                except NotImplementedError as e:
                    print("[DataCollector] Error reading vehicle speed:", e)
                    speed = self._cached_speed
            elif self.vehicle_state_url is not None:
                try:
                    with urllib.request.urlopen(self.vehicle_state_url, timeout=0.15) as resp:
                        state = json.loads(resp.read())
                    speed = int(state.get("speed_kmh", self._cached_speed))
                    self._cached_speed = speed
                except (urllib.error.URLError, OSError, KeyError, ValueError):
                    speed = self._cached_speed
            else:
                speed = int(os.getenv('PV_SPEED', random.randint(0, 120)))
            data['speed'] = speed

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
        calibration_counter = 0

        next_t = time.monotonic()
        while self._running:
            try:
                if self.calibrated is False:
                    # run calibration for _calibration_frames frames 
                    self.calibrate_step()
                    calibration_counter += 1
                    if calibration_counter >= self._calibration_frames:
                        self.compute_calibration()
                else:
                    data = self.collect_data()
                    
                    # Add sequence for LSTM models
                    with self._lock:
                        self.data_history.append(dict(data))
                        if len(self.data_history) > self.window_size:
                            self.data_history.pop(0)

                    action = None
                    if self.decision_engine:
        
                        data['functionname'] = data.get('functionname', self.functionname)
                        # separate dict to avoid adding the whole history to each sequence entry
                        data_for_decision = dict(data)
                        data_for_decision['sequence'] = list(self.data_history)

                        action = self.decision_engine.decide(dict(data_for_decision))
                        if self.logger and isinstance(action, dict):
                            action_for_log = dict(action)
                            for key in ('session_id', 'participantid', 'environment', 'secondary_task', 'functionname', 'emotion', 'modeltype', 'state_model', 'w_fcd'):
                                value = data.get(key)
                                if value not in (None, ''):
                                    action_for_log.setdefault(key, value)
                            self.logger.log_processed(action_for_log)
                        data['last_action'] = action


                    if self.logger:
                        raw_with_decision = dict(data)
                        if isinstance(action, dict):
                            raw_with_decision['LoA'] = action.get('LoA')
                            fcd = action.get('fcd') or action.get('fcd_scores')
                            if isinstance(fcd, dict):
                                raw_with_decision['FCD'] = fcd
                        self.logger.log_raw(raw_with_decision)


                    if self.actuator and action is not None:
                        self.actuator.execute(action)

            except NotImplementedError as e:
                print('[DataCollector] loop error:', e)

            next_t += self.sampling_interval
            time.sleep(max(0.0, min(self.sampling_interval, next_t - time.monotonic())))
        print("data collector stopped!")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self.rppg_estimator:
            self.rppg_estimator.stop()
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.release()

    def release(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        except NotImplementedError as e:
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
        except NotImplementedError as e:
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
            except NotImplementedError as e:
                print(e, "Error encoding latest frame")
                img_b64 = None
        return {"data": data, "frame_b64": img_b64}
