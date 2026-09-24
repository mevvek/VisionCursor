"""
camera/camera_manager.py
Modular camera capture manager supporting local webcams and external streams.
"""

import cv2
import time
from typing import Optional, Tuple
import numpy as np

class CameraManager:
    def __init__(self, source: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.prev_frame_time = 0.0
        self.current_fps = 0.0

    def start(self) -> bool:
        """Initializes and opens the camera feed."""
        # Using cv2.CAP_DSHOW on Windows for fast device initialization if numeric
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            # Fallback to default backend if CAP_DSHOW fails
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a frame, flips it horizontally (mirror view), and tracks FPS."""
        if not self.cap or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # Mirror horizontally so natural head movement matches screen intuition
        frame = cv2.flip(frame, 1)

        # Calculate instantaneous FPS
        current_time = time.time()
        time_diff = current_time - self.prev_frame_time
        if time_diff > 0:
            self.current_fps = 1.0 / time_diff
        self.prev_frame_time = current_time

        return True, frame

    def release(self):
        """Safely release the video capture source."""
        if self.cap and self.cap.isOpened():
            self.cap.release()