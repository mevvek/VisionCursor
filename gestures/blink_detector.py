"""
gestures/blink_detector.py
Calculates Eye Aspect Ratio (EAR) directly from standard MediaPipe face mesh landmarks.
"""

import time
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple
from config.settings import CONFIG

@dataclass
class BlinkEvent:
    blink_detected: bool = False
    eye_state: str = "OPEN"
    left_ear: float = 0.0
    right_ear: float = 0.0
    average_ear: float = 0.0
    blink_duration: float = 0.0
    blink_timestamp: float = 0.0

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

class BlinkDetector:
    def __init__(self):
        cfg = getattr(CONFIG, "eye", None)
        self.closed_threshold = getattr(cfg, "ear_closed_threshold", 0.21)
        self.open_threshold = getattr(cfg, "ear_open_threshold", 0.25)
        self.min_duration = getattr(cfg, "min_blink_duration", 0.05)
        self.max_duration = getattr(cfg, "max_blink_duration", 0.65)
        self.cooldown = getattr(cfg, "blink_cooldown", 0.12)

        self.eye_state = "OPEN"
        self.closed_start_time: Optional[float] = None
        self.last_blink_time: float = 0.0

    @staticmethod
    def _compute_ear_from_indices(landmarks: List[Tuple[int, int]], indices: List[int]) -> float:
        if len(landmarks) < 468:
            return 0.30

        p1 = landmarks[indices[0]]
        p2 = landmarks[indices[1]]
        p3 = landmarks[indices[2]]
        p4 = landmarks[indices[3]]
        p5 = landmarks[indices[4]]
        p6 = landmarks[indices[5]]

        width = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
        if width <= 1e-4:
            return 0.30

        h1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
        h2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])

        return float((h1 + h2) / (2.0 * width))

    def update_with_landmarks(self, landmarks: Optional[List[Tuple[int, int]]], timestamp: Optional[float] = None) -> BlinkEvent:
        now = timestamp or time.monotonic()

        if not landmarks or len(landmarks) < 468:
            return BlinkEvent(eye_state="OPEN", left_ear=0.30, right_ear=0.30, average_ear=0.30)

        left_ear = self._compute_ear_from_indices(landmarks, LEFT_EYE_INDICES)
        right_ear = self._compute_ear_from_indices(landmarks, RIGHT_EYE_INDICES)
        avg_ear = (left_ear + right_ear) / 2.0

        is_closed = avg_ear <= self.closed_threshold
        blink_detected = False
        duration = 0.0

        if self.eye_state == "OPEN":
            if is_closed:
                self.eye_state = "CLOSED"
                self.closed_start_time = now

        elif self.eye_state == "CLOSED":
            if not is_closed:
                self.eye_state = "OPEN"
                if self.closed_start_time is not None:
                    duration = now - self.closed_start_time
                    time_since_last = now - self.last_blink_time

                    if (self.min_duration <= duration <= self.max_duration) and (time_since_last >= self.cooldown):
                        blink_detected = True
                        self.last_blink_time = now

                self.closed_start_time = None

        return BlinkEvent(
            blink_detected=blink_detected,
            eye_state=self.eye_state,
            left_ear=round(left_ear, 3),
            right_ear=round(right_ear, 3),
            average_ear=round(avg_ear, 3),
            blink_duration=round(duration, 3) if blink_detected else 0.0,
            blink_timestamp=now if blink_detected else 0.0
        )