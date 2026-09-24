"""
gaze/gaze_estimator.py
Estimates horizontal and vertical gaze directions from normalized iris-in-eye coordinates.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from vision.face_tracker import FaceTrackingResult
from vision.eye_tracker import EyeData
from vision.iris_tracker import IrisData
from config.settings import CONFIG

@dataclass
class GazeResult:
    direction: str = "UNKNOWN"          # "CENTER", "LEFT", "RIGHT", "UP", "DOWN"
    raw_x: float = 0.5
    raw_y: float = 0.5
    smooth_x: float = 0.5
    smooth_y: float = 0.5
    confidence: str = "LOW"             # "HIGH", "MEDIUM", "LOW"

class GazeEstimator:
    def __init__(self):
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None
        self.alpha = CONFIG.gaze.smoothing_factor

    def estimate_gaze(self, tracking_result: FaceTrackingResult) -> GazeResult:
        """Computes normalized gaze coordinates and direction classification."""
        if not tracking_result.face_detected:
            self.smooth_x = None
            self.smooth_y = None
            return GazeResult(direction="NO FACE", confidence="LOW")

        left_ratio = self._calculate_eye_ratio(tracking_result.left_eye, tracking_result.left_iris, is_left=True)
        right_ratio = self._calculate_eye_ratio(tracking_result.right_eye, tracking_result.right_iris, is_left=False)

        # Fuse eye readings
        if left_ratio is not None and right_ratio is not None:
            raw_x = (left_ratio[0] + right_ratio[0]) / 2.0
            raw_y = (left_ratio[1] + right_ratio[1]) / 2.0
            confidence = "HIGH"
        elif left_ratio is not None:
            raw_x, raw_y = left_ratio
            confidence = "MEDIUM"
        elif right_ratio is not None:
            raw_x, raw_y = right_ratio
            confidence = "MEDIUM"
        else:
            return GazeResult(direction="UNKNOWN", confidence="LOW")

        # Temporal Exponential Smoothing (EMA)
        if self.smooth_x is None or self.smooth_y is None:
            self.smooth_x = raw_x
            self.smooth_y = raw_y
        else:
            self.smooth_x = self.alpha * raw_x + (1.0 - self.alpha) * self.smooth_x
            self.smooth_y = self.alpha * raw_y + (1.0 - self.alpha) * self.smooth_y

        # Direction Classification based on dead-zones and thresholds
        direction = self._classify_direction(self.smooth_x, self.smooth_y)

        return GazeResult(
            direction=direction,
            raw_x=round(float(raw_x), 3),
            raw_y=round(float(raw_y), 3),
            smooth_x=round(float(self.smooth_x), 3),
            smooth_y=round(float(self.smooth_y), 3),
            confidence=confidence
        )

    def _calculate_eye_ratio(self, eye: Optional[EyeData], iris: Optional[IrisData], is_left: bool) -> Optional[Tuple[float, float]]:
        """Calculates normalized (0.0 to 1.0) iris position within eye frame."""
        if eye is None or iris is None or eye.width <= 0 or eye.height <= 0:
            return None

        # Determine horizontal extremes
        # Frame is mirrored horizontally in CameraManager:
        # Left side of screen is user's right side
        if is_left:
            # Left Eye: inner corner is medial (362), outer is lateral (263)
            x_min = min(eye.inner_corner[0], eye.outer_corner[0])
            x_max = max(eye.inner_corner[0], eye.outer_corner[0])
        else:
            x_min = min(eye.inner_corner[0], eye.outer_corner[0])
            x_max = max(eye.inner_corner[0], eye.outer_corner[0])

        y_min = min(eye.top_point[1], eye.bottom_point[1])
        y_max = max(eye.top_point[1], eye.bottom_point[1])

        # Clamp calculations within bounds
        h_span = max(float(x_max - x_min), 1.0)
        v_span = max(float(y_max - y_min), 1.0)

        ratio_x = float(iris.center[0] - x_min) / h_span
        ratio_y = float(iris.center[1] - y_min) / v_span

        # Constrain to reasonable [0.0, 1.0] range
        ratio_x = float(np.clip(ratio_x, 0.0, 1.0))
        ratio_y = float(np.clip(ratio_y, 0.0, 1.0))

        return ratio_x, ratio_y

    def _classify_direction(self, x: float, y: float) -> str:
        """Classifies normalized coordinates against configurable thresholds."""
        cfg = CONFIG.gaze

        # Priority 1: Vertical extremes
        if y <= cfg.vertical_up_thresh:
            return "UP"
        if y >= cfg.vertical_down_thresh:
            return "DOWN"

        # Priority 2: Horizontal extremes
        if x <= cfg.horizontal_left_thresh:
            return "LEFT"
        if x >= cfg.horizontal_right_thresh:
            return "RIGHT"

        return "CENTER"