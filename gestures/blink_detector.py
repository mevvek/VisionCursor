"""
gestures/blink_detector.py
Calculates Eye Aspect Ratio (EAR) and uses a state machine to detect
clean, single, duration-measured blink events.
Strictly NO mouse clicking or action execution.
"""

import time
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
from config.settings import CONFIG

@dataclass
class BlinkEvent:
    blink_detected: bool = False
    eye_state: str = "OPEN"           # "OPEN" or "CLOSED"
    left_ear: float = 0.0
    right_ear: float = 0.0
    average_ear: float = 0.0
    blink_duration: float = 0.0
    blink_timestamp: float = 0.0

class BlinkDetector:
    """
    Eye Aspect Ratio (EAR) based blink detector.
    Tracks state transitions: OPEN -> CLOSING/CLOSED -> OPENING -> BLINK_EVENT.
    """
    def __init__(self):
        cfg = getattr(CONFIG, "eye", None)
        self.closed_threshold = getattr(cfg, "ear_closed_threshold", 0.20)
        self.open_threshold = getattr(cfg, "ear_open_threshold", 0.24)
        self.min_duration = getattr(cfg, "min_blink_duration", 0.06)
        self.max_duration = getattr(cfg, "max_blink_duration", 0.45)
        self.cooldown = getattr(cfg, "blink_cooldown", 0.15)
        self.both_eyes_required = getattr(cfg, "both_eyes_required", True)

        self.eye_state = "OPEN"
        self.closed_start_time: Optional[float] = None
        self.last_blink_time: float = 0.0

    @staticmethod
    def calculate_ear(eye_contour: Optional[np.ndarray]) -> float:
        """
        Calculates Eye Aspect Ratio (EAR) from eyelid points:
        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
        """
        if eye_contour is None or len(eye_contour) < 6:
            return 0.30

        pts = eye_contour
        
        # Horizontal width (corners)
        p1 = pts[0]
        p4 = pts[3] if len(pts) > 3 else pts[-1]
        width = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
        if width <= 0:
            return 0.30

        # Vertical heights
        if len(pts) >= 6:
            p2 = pts[1]
            p6 = pts[5]
            p3 = pts[2]
            p5 = pts[4]
            h1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
            h2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
            ear = (h1 + h2) / (2.0 * width)
        else:
            min_y = float(np.min(pts[:, 1]))
            max_y = float(np.max(pts[:, 1]))
            ear = (max_y - min_y) / width

        return float(ear)

    def update(self, left_contour: Optional[np.ndarray], right_contour: Optional[np.ndarray], timestamp: Optional[float] = None) -> BlinkEvent:
        """
        Processes current frame contours and updates blink state machine.
        Returns a single BlinkEvent per closure cycle.
        """
        now = timestamp or time.time()

        left_ear = self.calculate_ear(left_contour) if left_contour is not None else 0.30
        right_ear = self.calculate_ear(right_contour) if right_contour is not None else 0.30

        # Decide closure condition
        if self.both_eyes_required and left_contour is not None and right_contour is not None:
            is_closed = (left_ear <= self.closed_threshold) and (right_ear <= self.closed_threshold)
            avg_ear = (left_ear + right_ear) / 2.0
        else:
            avg_ear = (left_ear + right_ear) / 2.0
            is_closed = avg_ear <= self.closed_threshold

        blink_detected = False
        duration = 0.0

        # State Machine
        if self.eye_state == "OPEN":
            if is_closed:
                # Transition OPEN -> CLOSED
                self.eye_state = "CLOSED"
                self.closed_start_time = now

        elif self.eye_state == "CLOSED":
            if not is_closed:
                # Transition CLOSED -> OPEN (Eyes have reopened)
                self.eye_state = "OPEN"
                if self.closed_start_time is not None:
                    duration = now - self.closed_start_time
                    time_since_last_blink = now - self.last_blink_time

                    # Validate valid blink window and cooldown
                    if (self.min_duration <= duration <= self.max_duration) and (time_since_last_blink >= self.cooldown):
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