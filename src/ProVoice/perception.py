"""In-tree replacement for ``yolov5_deepsort_driverdistracted_driving_behavior_detection.myframe``.

The upstream package pinned us to Python 3.10 and bundled dlib + an old
YOLOv5 codebase. This module reimplements the only function the project
actually used (``frametest(frame)``) on top of two modern,
Python-version-agnostic dependencies:

* **MediaPipe FaceMesh** -- eye aspect ratio (EAR) and mouth aspect ratio
  (MAR), replacing dlib's 68-point landmark predictor.
* **Ultralytics YOLO26** -- in-cabin driver-distraction classification.

Public API (drop-in replacement for ``myframe.frametest``)::

    ret, annotated = frametest(frame_bgr)
    labels, eye_ar, mouth_ar = ret

The distraction model is **not bundled in this repo**. It is downloaded
on first use from the Hugging Face Hub
(`maco018/in-car-distraction-yolo26 <https://huggingface.co/maco018/in-car-distraction-yolo26>`_)
and cached locally by ``huggingface_hub`` (so it only downloads once).
The five YOLO26 size variants (``n``/``s``/``m``/``l``/``x``) are
classification models producing the labels ``safe``, ``phone``,
``drink`` and ``distracted``.

Override the source via environment variables:

* ``PROVOICE_YOLO_WEIGHTS`` -- absolute path to a local ``.pt`` (highest
  priority; skips the download entirely).
* ``PROVOICE_YOLO_REPO``    -- Hugging Face repo id
  (default ``maco018/in-car-distraction-yolo26``).
* ``PROVOICE_YOLO_VARIANT`` -- which size to pull: ``n``/``s``/``m``/``l``/``x``
  (default ``l`` -- highest accuracy).

To retrain (e.g. when a newer YOLO release lands) see
``src/ProVoice/train_distraction.py`` and
``scripts/train_yolo26_series.py``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np

try:
    import mediapipe as mp  # type: ignore
    _mp_face_mesh = mp.solutions.face_mesh
    _HAS_MEDIAPIPE = True
except ImportError:  # pragma: no cover
    mp = None  # type: ignore
    _mp_face_mesh = None  # type: ignore
    _HAS_MEDIAPIPE = False

try:
    from ultralytics import YOLO  # type: ignore
    _HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore
    _HAS_ULTRALYTICS = False


# ---------------------------------------------------------------------------
# EAR / MAR via MediaPipe FaceMesh
# ---------------------------------------------------------------------------

# Canonical FaceMesh landmark indices for the six points the EAR formula
# expects (horizontal corners + two pairs of vertical points).
# Convention: ear = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
_LEFT_EYE_IDX = (33, 160, 158, 133, 153, 144)   # viewer's left (subject's right)
_RIGHT_EYE_IDX = (362, 385, 387, 263, 373, 380)  # viewer's right (subject's left)

# Mouth: outer-corner-to-corner as horizontal, upper-inner-lip to
# lower-inner-lip as vertical. Returns ratio in roughly the same range
# as the dlib 68-point MAR (~0.0 closed, ~0.7+ wide yawn).
_MOUTH_LEFT_CORNER = 78
_MOUTH_RIGHT_CORNER = 308
_MOUTH_TOP_INNER = 13
_MOUTH_BOTTOM_INNER = 14
# Additional vertical points for a more dlib-compatible MAR shape
_MOUTH_TOP_OUTER = 81
_MOUTH_BOTTOM_OUTER = 178


def _euclid(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def _ear_from_landmarks(lm: np.ndarray, idx: Tuple[int, int, int, int, int, int]) -> float:
    p1, p2, p3, p4, p5, p6 = (lm[i] for i in idx)
    horiz = _euclid(p1, p4)
    if horiz < 1e-6:
        return 0.0
    return (_euclid(p2, p6) + _euclid(p3, p5)) / (2.0 * horiz)


def _mar_from_landmarks(lm: np.ndarray) -> float:
    left = lm[_MOUTH_LEFT_CORNER]
    right = lm[_MOUTH_RIGHT_CORNER]
    horiz = _euclid(left, right)
    if horiz < 1e-6:
        return 0.0
    v_inner = _euclid(lm[_MOUTH_TOP_INNER], lm[_MOUTH_BOTTOM_INNER])
    v_outer = _euclid(lm[_MOUTH_TOP_OUTER], lm[_MOUTH_BOTTOM_OUTER])
    return (v_inner + v_outer) / (2.0 * horiz)


def _landmarks_to_pixels(lms, w: int, h: int) -> np.ndarray:
    arr = np.empty((len(lms), 2), dtype=np.float32)
    for i, p in enumerate(lms):
        arr[i, 0] = p.x * w
        arr[i, 1] = p.y * h
    return arr


class EARMARDetector:
    """EAR/MAR + facial landmark overlay on top of MediaPipe FaceMesh."""

    # Defaults are the same "no fatigue" sentinels the old code used when
    # detection failed.
    DEFAULT_EAR = 0.3
    DEFAULT_MAR = 0.5

    def __init__(self, max_num_faces: int = 1, refine_landmarks: bool = True) -> None:
        self._lock = threading.Lock()
        self._mesh = None
        if _HAS_MEDIAPIPE and _mp_face_mesh is not None:
            self._mesh = _mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    @property
    def available(self) -> bool:
        return self._mesh is not None

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, float, bool]:
        """Return ``(annotated_frame, eye_ar, mouth_ar, face_present)``."""
        if not self.available or frame_bgr is None:
            return frame_bgr, self.DEFAULT_EAR, self.DEFAULT_MAR, False

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame_bgr, self.DEFAULT_EAR, self.DEFAULT_MAR, False

        lm = _landmarks_to_pixels(results.multi_face_landmarks[0].landmark, w, h)
        ear_left = _ear_from_landmarks(lm, _LEFT_EYE_IDX)
        ear_right = _ear_from_landmarks(lm, _RIGHT_EYE_IDX)
        ear = (ear_left + ear_right) / 2.0
        mar = _mar_from_landmarks(lm)

        annotated = frame_bgr.copy()
        for idx in _LEFT_EYE_IDX + _RIGHT_EYE_IDX:
            p = lm[idx].astype(int)
            cv2.circle(annotated, tuple(p), 1, (0, 255, 0), -1)
        for idx in (_MOUTH_LEFT_CORNER, _MOUTH_RIGHT_CORNER, _MOUTH_TOP_INNER,
                    _MOUTH_BOTTOM_INNER, _MOUTH_TOP_OUTER, _MOUTH_BOTTOM_OUTER):
            p = lm[idx].astype(int)
            cv2.circle(annotated, tuple(p), 1, (0, 255, 0), -1)

        return annotated, round(float(ear), 3), round(float(mar), 3), True


# ---------------------------------------------------------------------------
# Distraction detection via Ultralytics YOLO26
# ---------------------------------------------------------------------------

# Map COCO class names produced by the default YOLO26n weights back to
# the original project labels. The original model had four classes
# (face, smoke, phone, drink); COCO covers two cleanly.
DEFAULT_COCO_CLASS_MAP: Dict[str, str] = {
    "cell phone": "phone",
    "cup": "drink",
    "bottle": "drink",
}


#: Hugging Face repo the fine-tuned distraction models live in.
DEFAULT_HF_REPO = "maco018/in-car-distraction-yolo26"
#: Which size variant to pull by default (l = best accuracy).
DEFAULT_HF_VARIANT = "l"


def _download_from_hf(repo: str, variant: str) -> str:
    """Download ``<variant>/model.pt`` from a Hugging Face model repo.

    Returns the local cached path. Raises if huggingface_hub is missing or
    the download fails (the caller decides how to degrade).
    """
    from huggingface_hub import hf_hub_download  # local import; optional dep
    return hf_hub_download(repo_id=repo, filename=f"{variant}/model.pt")


class DistractionDetector:
    """YOLO classifier/detector that returns the project distraction labels.

    By default the fine-tuned classification model is downloaded from the
    Hugging Face Hub (see module docstring) and cached locally. Detection
    models (bounding boxes) are still supported for backwards compatibility;
    the mode is auto-detected from the loaded weights.

    Resolution order for the weights:

    1. ``weights=`` argument (local path or any value ``YOLO()`` accepts)
    2. ``PROVOICE_YOLO_WEIGHTS`` env var (local path)
    3. Hugging Face download of ``PROVOICE_YOLO_VARIANT`` from
       ``PROVOICE_YOLO_REPO`` (defaults: variant ``l``, repo
       ``maco018/in-car-distraction-yolo26``)
    """

    # Full set of labels the runtime understands (classification models produce
    # safe/phone/drink/distracted; legacy detection models produce phone/drink).
    PROJECT_LABELS = ("face", "smoke", "phone", "drink", "distracted", "safe")

    def __init__(
        self,
        weights: Optional[str] = None,
        conf: float = 0.6,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        class_map: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.conf = conf
        self.iou = iou
        # imgsz is resolved AFTER the model loads (classification models must
        # run at their training resolution, e.g. 224; detection at 640).
        # An explicit value always wins.
        self._imgsz_override = imgsz
        self.imgsz = imgsz or 640
        self.device = device
        self._model = None
        self._labels: Dict[int, str] = {}
        self._is_custom = False
        self._lock = threading.Lock()

        if not _HAS_ULTRALYTICS:
            print("[perception] ultralytics not installed; distraction detection disabled.")
            return

        # Resolution order: explicit arg > PROVOICE_YOLO_WEIGHTS env >
        # download from the Hugging Face Hub (cached after first run).
        if weights is None:
            weights = os.getenv("PROVOICE_YOLO_WEIGHTS")
        if weights is None:
            repo = os.getenv("PROVOICE_YOLO_REPO", DEFAULT_HF_REPO)
            variant = os.getenv("PROVOICE_YOLO_VARIANT", DEFAULT_HF_VARIANT)
            try:
                weights = _download_from_hf(repo, variant)
                print(f"[perception] using {repo} variant '{variant}' -> {weights}")
            except Exception as exc:  # noqa: BLE001
                print(f"[perception] could not fetch distraction model from "
                      f"Hugging Face ({repo}:{variant}): {exc}. "
                      f"Distraction detection disabled (set PROVOICE_YOLO_WEIGHTS "
                      f"to use a local model offline).")
                self._model = None
                return
        try:
            self._model = YOLO(weights)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            print(f"[perception] Failed to load YOLO weights '{weights}': {exc}")
            self._model = None
            return

        names = getattr(self._model, "names", {}) or {}
        if isinstance(names, dict):
            self._labels = {int(k): str(v) for k, v in names.items()}
        else:
            self._labels = {i: str(v) for i, v in enumerate(names)}

        # Detect classification vs detection model from task metadata.
        task = getattr(self._model, "task", None) or ""
        self._is_classify = ("classif" in str(task).lower()
                              or "-cls" in str(weights).lower())

        # Resolve inference resolution. Classification models MUST run at
        # their training size (typically 224); running a 224-trained
        # classifier at 640 collapses its predictions. Prefer the imgsz
        # baked into the checkpoint, then fall back to task defaults.
        if self._imgsz_override is None:
            trained_imgsz = None
            try:
                args = getattr(self._model, "ckpt", {}).get("train_args", {}) or {}
                trained_imgsz = args.get("imgsz")
            except Exception:
                trained_imgsz = None
            if trained_imgsz:
                self.imgsz = int(trained_imgsz)
            else:
                self.imgsz = 224 if self._is_classify else 640

        # If model already produces project labels, bypass COCO remap.
        custom_overlap = set(self._labels.values()) & set(self.PROJECT_LABELS)
        if custom_overlap:
            self._is_custom = True
            self._class_map: Dict[str, str] = {n: n for n in self._labels.values()
                                                if n in self.PROJECT_LABELS}
        else:
            self._class_map = dict(class_map or DEFAULT_COCO_CLASS_MAP)

        print(f"[perception] DistractionDetector loaded '{weights}' "
              f"(mode={'classify' if self._is_classify else 'detect'}, "
              f"custom={self._is_custom}, classes={len(self._labels)})")

    @property
    def available(self) -> bool:
        return self._model is not None

    def __call__(self, frame_bgr: np.ndarray) -> Tuple[List[str], List[Tuple[str, float, Tuple[int, int, int, int]]]]:
        """Run inference. Returns ``(label_list, detection_list)``.

        ``label_list`` is the de-duplicated list of project labels that
        appear in this frame. ``detection_list`` items are
        ``(label, confidence, (x1, y1, x2, y2))`` for callers that want
        to draw boxes themselves.
        """
        if not self.available or frame_bgr is None:
            return [], []

        with self._lock:
            results = self._model.predict(  # type: ignore[union-attr]
                frame_bgr,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )

        labels: List[str] = []
        detections: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
        for res in results:
            if self._is_classify:
                # Classification mode: top-1 prediction from r.probs
                probs = getattr(res, "probs", None)
                if probs is None:
                    continue
                cid = int(probs.top1)
                conf = float(probs.top1conf)
                if conf < self.conf:
                    continue
                raw_name = self._labels.get(cid)
                if raw_name is None:
                    continue
                project_label = self._class_map.get(raw_name, raw_name
                                                     if raw_name in self.PROJECT_LABELS else None)
                if project_label and project_label not in labels:
                    labels.append(project_label)
                # No bounding box in classification mode; detections list stays empty.
            else:
                # Detection mode: iterate bounding boxes
                boxes = getattr(res, "boxes", None)
                if boxes is None or boxes.cls is None:
                    continue
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs   = boxes.conf.cpu().numpy().astype(float)
                xyxy    = boxes.xyxy.cpu().numpy().astype(int)
                for cid, conf, box in zip(cls_ids, confs, xyxy):
                    if conf < self.conf:
                        continue
                    raw_name = self._labels.get(int(cid))
                    if raw_name is None:
                        continue
                    project_label = self._class_map.get(raw_name)
                    if project_label is None:
                        continue
                    if project_label not in labels:
                        labels.append(project_label)
                    x1, y1, x2, y2 = (int(v) for v in box)
                    detections.append((project_label, float(conf), (x1, y1, x2, y2)))
        return labels, detections


# ---------------------------------------------------------------------------
# Drop-in API: frametest()
# ---------------------------------------------------------------------------

_default_earmar: Optional[EARMARDetector] = None
_default_distraction: Optional[DistractionDetector] = None
_default_lock = threading.Lock()


def _get_default_detectors() -> Tuple[EARMARDetector, DistractionDetector]:
    global _default_earmar, _default_distraction
    with _default_lock:
        if _default_earmar is None:
            _default_earmar = EARMARDetector()
        if _default_distraction is None:
            _default_distraction = DistractionDetector()
    return _default_earmar, _default_distraction


def frametest(frame_bgr: np.ndarray) -> Tuple[Tuple[List[str], float, float], np.ndarray]:
    """Drop-in replacement for ``myframe.frametest``.

    Returns ``((labels, eye_ar, mouth_ar), annotated_frame)``.
    Labels are restricted to the project-recognised set (``face``,
    ``smoke``, ``phone``, ``drink``). ``face`` is set whenever
    MediaPipe sees a face; the other three depend on the YOLO model.
    """
    earmar, distraction = _get_default_detectors()
    annotated, eye_ar, mouth_ar, face_present = earmar(frame_bgr)
    labels, detections = distraction(frame_bgr if annotated is None else annotated)

    if face_present and "face" not in labels:
        labels.insert(0, "face")

    if detections and annotated is not None:
        for lab, conf, (x1, y1, x2, y2) in detections:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(annotated, f"{lab} {conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return (labels, float(eye_ar), float(mouth_ar)), annotated
