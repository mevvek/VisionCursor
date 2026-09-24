"""
vision/eye_tracker.py
Extracts eye contour, corners, and bounding metrics.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from vision.landmarks import (
    LEFT_EYE_CONTOUR, LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
    RIGHT_EYE_CONTOUR, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
    denormalize_landmark, denormalize_landmarks
)

@dataclass
class EyeData:
    contour_points: np.ndarray
    inner_corner: Tuple[int, int]
    outer_corner: Tuple[int, int]
    top_point: Tuple[int, int]
    bottom_point: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    width: float
    height: float

class EyeTracker:
    def extract_eye(self, landmarks, frame_w: int, frame_h: int, is_left: bool) -> Optional[EyeData]:
        """Extracts eye contour, extreme bounds, and geometric dimensions."""
        if not landmarks:
            return None

        if is_left:
            contour_idx = LEFT_EYE_CONTOUR
            inner_idx = LEFT_EYE_INNER
            outer_idx = LEFT_EYE_OUTER
            top_idx = LEFT_EYE_TOP
            bottom_idx = LEFT_EYE_BOTTOM
        else:
            contour_idx = RIGHT_EYE_CONTOUR
            inner_idx = RIGHT_EYE_INNER
            outer_idx = RIGHT_EYE_OUTER
            top_idx = RIGHT_EYE_TOP
            bottom_idx = RIGHT_EYE_BOTTOM

        contour_pts = denormalize_landmarks(landmarks, contour_idx, frame_w, frame_h)
        inner = denormalize_landmark(landmarks[inner_idx], frame_w, frame_h)
        outer = denormalize_landmark(landmarks[outer_idx], frame_w, frame_h)
        top = denormalize_landmark(landmarks[top_idx], frame_w, frame_h)
        bottom = denormalize_landmark(landmarks[bottom_idx], frame_w, frame_h)

        # Bounding box of eye contour
        x, y, w, h = 0, 0, 0, 0
        if len(contour_pts) > 0:
            xs = contour_pts[:, 0]
            ys = contour_pts[:, 1]
            x, y = int(np.min(xs)), int(np.min(ys))
            w, h = int(np.max(xs) - x), int(np.max(ys) - y)

        eye_width = float(np.linalg.norm(np.array(inner) - np.array(outer)))
        eye_height = float(np.linalg.norm(np.array(top) - np.array(bottom)))

        return EyeData(
            contour_points=contour_pts,
            inner_corner=inner,
            outer_corner=outer,
            top_point=top,
            bottom_point=bottom,
            bounding_box=(x, y, w, h),
            width=eye_width,
            height=eye_height
        )