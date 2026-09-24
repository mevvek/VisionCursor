"""
gaze/gaze_estimator.py
Estimates horizontal and vertical gaze directions from normalized iris-in-eye coordinates.
Uses anatomical medial-lateral landmarks for precise horizontal travel.
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
    direction: str = "UNKNOWN"
    raw_x: float = 0.5
    raw_y: float = 0.5
    smooth_x: float = 0.5
    smooth_y: float = 0.5
    confidence: str = "LOW"

class GazeEstimator:
    def __init__(self):
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None
        self.alpha = 0.30

    def estimate_gaze(self, tracking_result: FaceTrackingResult) -> GazeResult:
        if not tracking_result.face_detected:
            self.smooth_x = None
            self.smooth_y = None
            return GazeResult(direction="NO FACE", confidence="LOW")

        left_ratio = self._calculate_eye_ratio(tracking_result.left_eye, tracking_result.left_iris)
        right_ratio = self._calculate_eye_ratio(tracking_result.right_eye, tracking_result.right_iris)

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

        # Smooth output
        if self.smooth_x is None or self.smooth_y is None:
            self.smooth_x = raw_x
            self.smooth_y = raw_y
        else:
            self.smooth_x = self.alpha * raw_x + (1.0 - self.alpha) * self.smooth_x
            self.smooth_y = self.alpha * raw_y + (1.0 - self.alpha) * self.smooth_y

        direction = self._classify_direction(self.smooth_x, self.smooth_y)

        return GazeResult(
            direction=direction,
            raw_x=round(float(raw_x), 3),
            raw_y=round(float(raw_y), 3),
            smooth_x=round(float(self.smooth_x), 3),
            smooth_y=round(float(self.smooth_y), 3),
            confidence=confidence
        )

    def _calculate_eye_ratio(self, eye: Optional[EyeData], iris: Optional[IrisData]) -> Optional[Tuple[float, float]]:
        if eye is None or iris is None or len(eye.contour_points) < 6:
            return None

        # Horizontal: Leftmost pixel to Rightmost pixel of current eye contour
        pts = eye.contour_points
        min_x = float(np.min(pts[:, 0]))
        max_x = float(np.max(pts[:, 0]))
        eye_width = max(max_x - min_x, 1.0)

        # Vertical: Topmost to Bottommost
        min_y = float(np.min(pts[:, 1]))
        max_y = float(np.max(pts[:, 1]))
        eye_height = max(max_y - min_y, 1.0)

        # Iris relative to width
        rx = float(iris.center[0] - min_x) / eye_width
        ry = float(iris.center[1] - min_y) / eye_height

        return float(np.clip(rx, 0.0, 1.0)), float(np.clip(ry, 0.0, 1.0))

    def _classify_direction(self, x: float, y: float) -> str:
        if x < 0.44:
            return "LEFT"
        if x > 0.56:
            return "RIGHT"
        if y < 0.40:
            return "UP"
        if y > 0.60:
            return "DOWN"
        return "CENTER"