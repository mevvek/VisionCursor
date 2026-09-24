"""
vision/face_tracker.py
Face and Iris tracking using MediaPipe Tasks FaceLandmarker (Python 3.13+ / MediaPipe 1.0+ compatible).
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, List
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from vision.landmarks import FACE_OVAL, denormalize_landmarks
from vision.eye_tracker import EyeTracker, EyeData
from vision.iris_tracker import IrisTracker, IrisData

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

@dataclass
class FaceTrackingResult:
    face_detected: bool = False
    face_count: int = 0
    face_oval_points: Optional[np.ndarray] = None
    left_eye: Optional[EyeData] = None
    right_eye: Optional[EyeData] = None
    left_iris: Optional[IrisData] = None
    right_iris: Optional[IrisData] = None

class FaceTracker:
    def __init__(self, max_num_faces: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self._ensure_model_exists()

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)
        self.eye_tracker = EyeTracker()
        self.iris_tracker = IrisTracker()

    def _ensure_model_exists(self):
        """Downloads the official MediaPipe face landmarker model if not present locally."""
        if not os.path.exists(MODEL_PATH):
            print(f"[INFO] Downloading official MediaPipe FaceLandmarker model to {MODEL_PATH}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("[INFO] Model download complete.")

    def process_frame(self, frame: np.ndarray) -> FaceTrackingResult:
        """Processes a single BGR frame and returns structured tracking data."""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return FaceTrackingResult(face_detected=False, face_count=0)

        face_count = len(results.face_landmarks)
        primary_landmarks = self._select_primary_face(results.face_landmarks, w, h)

        face_oval = denormalize_landmarks(primary_landmarks, FACE_OVAL, w, h)
        left_eye = self.eye_tracker.extract_eye(primary_landmarks, w, h, is_left=True)
        right_eye = self.eye_tracker.extract_eye(primary_landmarks, w, h, is_left=False)
        left_iris = self.iris_tracker.extract_iris(primary_landmarks, w, h, is_left=True)
        right_iris = self.iris_tracker.extract_iris(primary_landmarks, w, h, is_left=False)

        return FaceTrackingResult(
            face_detected=True,
            face_count=face_count,
            face_oval_points=face_oval,
            left_eye=left_eye,
            right_eye=right_eye,
            left_iris=left_iris,
            right_iris=right_iris
        )

    def _select_primary_face(self, multi_landmarks: List, frame_w: int, frame_h: int):
        """Picks the largest face in the camera view to guarantee 1-user control."""
        if len(multi_landmarks) == 1:
            return multi_landmarks[0]

        max_area = 0.0
        selected = multi_landmarks[0]

        for face in multi_landmarks:
            pts = denormalize_landmarks(face, FACE_OVAL, frame_w, frame_h)
            x, y, w, h = cv2.boundingRect(pts)
            area = w * h
            if area > max_area:
                max_area = area
                selected = face

        return selected

    def close(self):
        self.detector.close()