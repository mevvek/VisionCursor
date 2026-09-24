"""
vision/landmarks.py
MediaPipe landmark indices and geometric helper functions.
"""

from typing import List, Tuple
import numpy as np

# Left Eye contour & key points
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

# Right Eye contour & key points
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

# Iris landmarks (Center + 4 circumference points via refine_landmarks=True)
LEFT_IRIS_CENTER = 468
LEFT_IRIS_POINTS = [468, 469, 470, 471, 472]

RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_POINTS = [473, 474, 475, 476, 477]

# Subtle face contour outline points for visual confirmation
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

def denormalize_landmark(landmark, frame_w: int, frame_h: int) -> Tuple[int, int]:
    """Converts a normalized landmark (0.0 to 1.0) into pixel coordinates."""
    return int(landmark.x * frame_w), int(landmark.y * frame_h)

def denormalize_landmarks(landmarks, indices: List[int], frame_w: int, frame_h: int) -> np.ndarray:
    """Converts a list of landmark indices into a NumPy array of integer (x, y) coordinates."""
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append([int(lm.x * frame_w), int(lm.y * frame_h)])
    return np.array(coords, dtype=np.int32)