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
    def __init__(self, frame_rate: int = 10, crop_size: int = 72):
        self.ingest_count_frame = 0
        sitepackage_paths = site.getsitepackages()
        model_path = None
        for path in sitepackage_paths:
            model_path = Path(path) / "mmrphys" / "final_model_release" / "SCAMPS" / \
                         "SCAMPS_MMRPhysSEF_BVP_RSP_RGBx180x9_SFSAM_Label_Epoch0.pth"
            if model_path.exists():
                break
        config = {
            'model': {'path': model_path,
                      'type': 'torch', 'input_shape': {'num_frames': 181, 'channels': 3, 'height': crop_size, 'width': crop_size}},
            'video': {'sampling_rate': frame_rate},
            'processing': {
                'plot_duration': 20,  # seconds
                # Old: 'inference_interval': 180  # ~18s at 10fps, too infrequent
                'inference_interval': 45   # ~6s at 7fps / ~4.5s at 10fps
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
        Returns:
            bpm (float) or None
        """

        if self.stop_event.is_set():
            print("interrupt add_frame")
            return None, None

        # Resize to expected network input size
        processed_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        processed_frame = cv2.resize(processed_frame, (self.width, self.height))
        processed_frame = processed_frame[np.newaxis, :, :, :]
        processed_frame = processed_frame.transpose(0, 3, 1, 2)

        try:
            self.frame_queue.put((self.ingest_count_frame, face_frame, processed_frame, True),
                                 timeout=0.01)
        except queue.Full:
            print("Frame queue full!")
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
