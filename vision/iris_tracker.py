"""
vision/iris_tracker.py
Extracts iris landmarks, calculates iris centers, and returns structured data.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
from vision.landmarks import (
    LEFT_IRIS_CENTER, LEFT_IRIS_POINTS,
    RIGHT_IRIS_CENTER, RIGHT_IRIS_POINTS,
    denormalize_landmark, denormalize_landmarks
)

@dataclass
class IrisData:
    center: Tuple[int, int]
    points: np.ndarray
    radius: float

class IrisTracker:
    def extract_iris(self, landmarks, frame_w: int, frame_h: int, is_left: bool) -> Optional[IrisData]:
        """Extracts iris center, circumference points, and approximate radius."""
        if not landmarks:
            return None

        center_idx = LEFT_IRIS_CENTER if is_left else RIGHT_IRIS_CENTER
        points_indices = LEFT_IRIS_POINTS if is_left else RIGHT_IRIS_POINTS

        center = denormalize_landmark(landmarks[center_idx], frame_w, frame_h)
        points = denormalize_landmarks(landmarks, points_indices, frame_w, frame_h)

        # Calculate approximate radius using distance to circumference points
        distances = [np.linalg.norm(np.array(center) - pt) for pt in points[1:]]
        radius = float(np.mean(distances)) if distances else 4.0

        return IrisData(center=center, points=points, radius=radius)