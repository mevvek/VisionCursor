"""
vision/face_tracker.py
Extracts full face mesh landmarks, eyes, irises (with radius calculation), and nose-tip anchor.
Fully compatible with Python 3.13 & MediaPipe Tasks API.
"""

import os
import inspect
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from vision.eye_tracker import EyeTracker, EyeData
from vision.iris_tracker import IrisData

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "face_landmarker.task")

@dataclass
class FaceTrackingResult:
    face_detected: bool = False
    raw_pixel_landmarks: Optional[List[Tuple[int, int]]] = None
    left_eye: Optional[EyeData] = None
    right_eye: Optional[EyeData] = None
    left_iris: Optional[IrisData] = None
    right_iris: Optional[IrisData] = None
    nose_point: Optional[Tuple[float, float]] = None
    nose_px: Optional[Tuple[int, int]] = None

class FaceTracker:
    def __init__(self, model_path: Optional[str] = None):
        target_model_path = model_path or DEFAULT_MODEL_PATH

        if not os.path.exists(target_model_path):
            raise FileNotFoundError(f"[ERROR] Model file not found at: {target_model_path}")

        base_options = python.BaseOptions(model_asset_path=target_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.eye_tracker = EyeTracker()

    def _extract_eyes_safely(self, pixel_landmarks) -> Tuple[Optional[EyeData], Optional[EyeData]]:
        try:
            if hasattr(self.eye_tracker, "extract_eyes"):
                return self.eye_tracker.extract_eyes(pixel_landmarks)
            if hasattr(self.eye_tracker, "extract_left_eye") and hasattr(self.eye_tracker, "extract_right_eye"):
                return (
                    self.eye_tracker.extract_left_eye(pixel_landmarks),
                    self.eye_tracker.extract_right_eye(pixel_landmarks)
                )
            if hasattr(self.eye_tracker, "extract_eye"):
                sig = inspect.signature(self.eye_tracker.extract_eye)
                params = list(sig.parameters.keys())
                if len(params) >= 2:
                    return (
                        self.eye_tracker.extract_eye(pixel_landmarks, is_left=True),
                        self.eye_tracker.extract_eye(pixel_landmarks, is_left=False)
                    )
                else:
                    return self.eye_tracker.extract_eye(pixel_landmarks), None
        except Exception:
            return None, None
        return None, None

    def _build_iris_data(self, center_pt: Tuple[int, int], perimeter_pts: np.ndarray) -> IrisData:
        """Calculates mean radius and creates IrisData safely."""
        cx, cy = center_pt
        dists = [np.hypot(px - cx, py - cy) for px, py in perimeter_pts]
        avg_radius = float(np.mean(dists)) if dists else 5.0
        
        # Check constructor signature for optional arguments
        try:
            return IrisData(center=center_pt, points=perimeter_pts, radius=avg_radius)
        except TypeError:
            return IrisData(center=center_pt, points=perimeter_pts)

    def process_frame(self, frame: np.ndarray) -> FaceTrackingResult:
        h, w, _ = frame.shape
        rgb_frame = frame[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return FaceTrackingResult(face_detected=False)

        raw_landmarks = result.face_landmarks[0]

        # Landmark #4 is the physical nose tip
        nose_lm = raw_landmarks[4]
        nose_norm = (nose_lm.x, nose_lm.y)
        nose_px = (int(nose_lm.x * w), int(nose_lm.y * h))

        pixel_landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in raw_landmarks]

        left_eye, right_eye = self._extract_eyes_safely(pixel_landmarks)

        # Iris landmarks (468: Left center, 473: Right center)
        left_iris = None
        right_iris = None
        if len(pixel_landmarks) >= 478:
            left_center = pixel_landmarks[468]
            left_pts = np.array([pixel_landmarks[i] for i in [469, 470, 471, 472]], dtype=np.int32)
            left_iris = self._build_iris_data(left_center, left_pts)

            right_center = pixel_landmarks[473]
            right_pts = np.array([pixel_landmarks[i] for i in [474, 475, 476, 477]], dtype=np.int32)
            right_iris = self._build_iris_data(right_center, right_pts)

        return FaceTrackingResult(
            face_detected=True,
            raw_pixel_landmarks=pixel_landmarks,
            left_eye=left_eye,
            right_eye=right_eye,
            left_iris=left_iris,
            right_iris=right_iris,
            nose_point=nose_norm,
            nose_px=nose_px
        )

    def close(self):
        self.landmarker.close()