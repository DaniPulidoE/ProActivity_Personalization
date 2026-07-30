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

        # Ring buffer replacing the parent's O(num_frames) shift-and-append
        # window (RemoteVitalSigns.inference_thread rolls the WHOLE
        # (1, C, num_frames, H, W) array on every frame — measured at
        # ~6 ms/frame on this project's target hardware, ~13x the amortized
        # cost of the model forward pass itself, and the actual cause of the
        # frame-queue-full drops this class works around above). Writing one
        # slot is O(1); the sliding window is only assembled (O(num_frames))
        # right before a forward pass, i.e. once every inference_interval
        # frames. Verified bit-identical to the parent's buffer content and
        # model output at every trigger point before landing this change.
        # self.inference_frame_buffer (parent's init_buffers()) is left
        # allocated but unused by this override.
        self._ring = np.zeros(
            (self.num_frames, self.in_channels, self.height, self.width))
        self._ring_write_idx = 0

        # Old (naming collision — method overwritten by attribute):
        # self.inference_thread = threading.Thread(target=self.inference_thread)
        # self.inference_thread.start()
        self._thread = threading.Thread(target=self.inference_thread, daemon=True)
        self._thread.start()
        print("OnlineRPPG STARTED!")


    def inference_thread(self):
        """
        Drop-in replacement for RemoteVitalSigns.inference_thread (overrides
        the inherited method by name — self._thread targets self.inference_thread,
        which now resolves here via the MRO, no monkeypatching involved).

        Identical control flow/semantics to the parent (same trigger cadence,
        same result_queue payloads, same face_detected branch, same shutdown
        sentinel) — the only change is HOW the sliding window is maintained:
        a ring buffer (O(1) per frame) instead of the parent's full-array
        shift-and-append (O(num_frames) per frame). See the ring buffer
        comment in __init__ for the verification this relies on.
        """
        count_frame = 0
        measurements_recorded = 0

        print("Starting inference thread (ring-buffer)...")

        while not self.stop_event.is_set():
            data = self.frame_queue.get()
            if data is None:
                break

            frame_idx, face_frame, processed_frame, face_detected = data

            if face_detected and processed_frame is not None:
                # processed_frame is (1, C, H, W); ring slot is (C, H, W).
                self._ring[self._ring_write_idx] = processed_frame[0]
                write_idx = self._ring_write_idx
                self._ring_write_idx = (write_idx + 1) % self.num_frames

                if count_frame >= self.num_frames and count_frame % self.inference_interval == 0:
                    # Oldest-to-newest order — matches exactly what the
                    # parent's buf[:, :, i, :, :] shift-and-append produces.
                    next_write = self._ring_write_idx
                    if next_write == 0:
                        ordered = self._ring
                    else:
                        ordered = np.concatenate(
                            (self._ring[next_write:], self._ring[:next_write]), axis=0)
                    frame_buffer = ordered.transpose(1, 0, 2, 3)[np.newaxis, ...]

                    bvp, rsp = self.model_instance.infer_rphys(frame_buffer)
                    bvp, rsp = self.signal_processor.post_process(bvp, rsp)

                    self.bvp_buffer.extend(bvp.reshape(-1))
                    self.rsp_buffer.extend(rsp.reshape(-1))

                    hr, rr = self.signal_processor.compute_metrics(
                        np.array(list(self.bvp_buffer)),
                        np.array(list(self.rsp_buffer))
                    )

                    if not np.isnan(hr) and not np.isnan(rr):
                        measurements_recorded += 1
                        if measurements_recorded % 10 == 0:
                            print(f"Recorded {measurements_recorded} valid measurements. HR: {hr:.1f}, RR: {rr:.1f}")

                    self.result_queue.put((face_frame, np.array(list(self.bvp_buffer)),
                                        np.array(list(self.rsp_buffer)), hr, rr, face_detected))
            else:
                self.result_queue.put((None, np.array(list(self.bvp_buffer)) if self.bvp_buffer else np.zeros(self.signal_buffer_size),
                                    np.array(list(self.rsp_buffer)) if self.rsp_buffer else np.zeros(self.signal_buffer_size),
                                    np.nan, np.nan, False))

            count_frame += 1

        print(f"Inference thread ended. Recorded {measurements_recorded} valid measurements")
        self.result_queue.put(None)

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
