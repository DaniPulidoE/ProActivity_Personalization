import threading
import types
import queue
from pathlib import Path
import site

import cv2
import numpy as np
from scipy import signal as scipy_signal
from mmrphys.tools.run_inference.infer_from_frames import RemoteVitalSigns



class OnlineRPPG(RemoteVitalSigns):
    """
    A simple real-time rPPG interface that returns BPM when enough frames
    are accumulated.

    See: https://physiologicailab.github.io/mmrphys-live/
    """
    # frame_rate defaults to 30 to match the SCAMPS checkpoint loaded below
    # (the SCAMPS configs specify FS: 30; BP4D would be 25). DataCollector
    # overrides it with the rate the camera actually achieved.
    def __init__(self, frame_rate: int = 30, crop_size: int = 72):
        self.ingest_count_frame = 0
        self.dropped_frames = 0     # frames lost to a full queue (see add_frame)
        sitepackage_paths = site.getsitepackages()
        # SCAMPS is the default, NOT BP4D, despite being synthetic. Reasons:
        #
        #  1. Reproducible preprocessing. Every BP4D config in the upstream repo
        #     sets DO_CROP_FACE: False against a pre-made "BP4D_72x72" dataset
        #     that no script in the repo produces, so the crop convention behind
        #     the BP4D checkpoints is unknown. The SCAMPS configs specify it in
        #     full (Y5F detector, square box, LARGE_BOX_COEF 1.5, re-detect every
        #     FS frames, INTER_AREA, FS 30) -- which is exactly what this
        #     pipeline now reproduces.
        #  2. Its reported number is measured under OUR deployment condition.
        #     Paper Table 3: MMRPhys+TSFM trained on SCAMPS and tested
        #     cross-dataset on iBVP + PURE + UBFC-rPPG gives HR MAE 1.54 bpm
        #     (Corr 0.900). The BP4D figures are within-dataset (Fold1/2/3), so
        #     they are optimistic and not comparable. The paper also validates
        #     SCAMPS -> BP4D+ transfer directly (SS6).
        #
        # Caveat to validate in the pilot: SCAMPS is rendered, and RGB
        # respiration comes from head motion / pulse-amplitude modulation, which
        # synthetic data models less faithfully than pulse. Trust HR first; check
        # RR against a reference before relying on rr_delta.
        #
        # The checkpoint MUST match ``crop_size``: infer_from_frames dispatches on
        # height (9 -> MMRPhysSEF, 36 -> MMRPhysMEF, 72 -> MMRPhysLEF), so a
        # "...x180x9" (SEF) file with crop_size=72 instantiates LEF instead. That
        # does NOT raise — all three variants share identical state_dict key names
        # and tensor shapes (87,600 params; they differ only in conv strides and
        # head geometry) — so 9x9-trained filters silently run on a 72x72 pyramid.
        # The paper's own ablation (§5) reports 72x72 ~= 36x36 > 9x9 for BOTH rPPG
        # and rRSP, so the x180x72 LEF checkpoint is the one to use.
        rel_candidates = [
            ("SCAMPS", "SCAMPS_MMRPhysLEF_BVP_RSP_RGBx180x72_SFSAM_Label_Epoch0.pth"),
            ("BP4D",   "BP4D_MMRPhysLEF_BVP_RSP_RGBx180x72_SFSAM_Label_Fold1_Epoch4.pth"),
        ]
        model_path = None
        for path in sitepackage_paths:
            for sub, fname in rel_candidates:
                cand = Path(path) / "mmrphys" / "final_model_release" / sub / fname
                if cand.exists():
                    model_path = cand
                    break
            if model_path is not None:
                break
        print(f"[OnlineRPPG] using checkpoint: {model_path}")
        config = {
            'model': {'path': model_path,
                      'type': 'torch', 'input_shape': {'num_frames': 181, 'channels': 3, 'height': crop_size, 'width': crop_size}},
            'video': {'sampling_rate': frame_rate},
            'processing': {
                'plot_duration': 20,  # seconds -> signal buffer = 20 s * fs
                # Frames between forward passes. MUST stay at num_frames - 1:
                # RemoteVitalSigns.inference_thread appends ALL (num_frames - 1)
                # output samples to the signal buffer, so any smaller interval
                # re-appends signal that is already there. Consecutive windows
                # would then overlap while the buffer is still read as uniformly
                # sampled at fs, corrupting the FFT's time axis -- the buffer
                # advances faster than real time and the HR peak is pulled off
                # (measured: 68.6 bpm for a true 72.0 at interval=45, and the
                # bias swings +-3.6 bpm with the interval value). Every upstream
                # demo config ships 181/180 for exactly this reason, and the
                # paper's protocol (SS4.3) likewise amalgamates NON-overlapping
                # segments before the FFT.
                #
                # Was 45 -- chosen when capture ran at ~10 fps, where 180 frames
                # meant an 18 s wait between readings. At 30 fps it is 6 s, so
                # the reason for shortening it no longer applies. One 181-frame
                # LEF pass costs ~96 ms on CPU => ~1.6% of one core.
                'inference_interval': 180
            }}

        super().__init__(config)


        self.signal_processor.init_filters()   # re-init with the patched method

        self._fs_lock = threading.Lock()

        # Old (naming collision — method overwritten by attribute):
        # self.inference_thread = threading.Thread(target=self.inference_thread)
        # self.inference_thread.start()
        self._thread = threading.Thread(target=self.inference_thread, daemon=True)
        self._thread.start()
        print("OnlineRPPG STARTED!")


    def stop(self):
        # Old (put() could block forever if queue full; no timeout on join):
        # self.frame_queue.put(None)
        # self.stop_event.set()
        # self.inference_thread.join()
        try:
            self.frame_queue.put(None, timeout=1.0)
        except Exception:
            pass
        self.stop_event.set()
        self._thread.join(timeout=3.0)

    def add_frame(self, face_frame: np.ndarray) -> tuple[float, float] | tuple[None, None]:
        """
        Add a single input frame and compute BPM when window is full.

        ``face_frame`` is a BGR face crop. To match the training pipeline the
        caller must supply a SQUARE crop enlarged 1.5x around the detector box
        (see DataCollector._rppg_crop_box); this method only handles the colour
        conversion, the downscale and the layout.

        Returns:
            bpm (float) or None
        """

        if self.stop_event.is_set():
            print("interrupt add_frame")
            return None, None

        # Resize to expected network input size.
        # INTER_AREA (not the cv2 default INTER_LINEAR) — this is what
        # BaseLoader.crop_face_resize uses during training, and it is the correct
        # choice for a large downscale: without area averaging, skin texture
        # aliases and head motion injects broadband noise into what is ultimately
        # a spectral estimate. Pixel SCALE is irrelevant (the model applies
        # torch.diff then InstanceNorm3d, so 0-255 vs 0-1 is cancelled).
        processed_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        processed_frame = cv2.resize(processed_frame, (self.width, self.height),
                                     interpolation=cv2.INTER_AREA)
        processed_frame = processed_frame[np.newaxis, :, :, :]
        processed_frame = processed_frame.transpose(0, 3, 1, 2)

        try:
            self.frame_queue.put((self.ingest_count_frame, face_frame, processed_frame, True),
                                 timeout=0.01)
        except queue.Full:
            # A dropped frame silently breaks the uniform-dt assumption the FFT
            # rests on, so count it rather than only printing: the caller
            # surfaces the total as a data-quality field.
            self.dropped_frames += 1
            print(f"Frame queue full! (dropped {self.dropped_frames} total)")
            return None, None

        self.ingest_count_frame += 1

        try:
            data = self.result_queue.get(block=False)
        except queue.Empty:
            return None, None
        if data is None:
            return None, None

        _, _, _, hr, rr, _ = data
        return hr, rr
