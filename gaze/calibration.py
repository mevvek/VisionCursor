"""
gaze/calibration.py
Direct proportional mapping for Gaze to Screen space.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple
import numpy as np
from config.settings import CONFIG

@dataclass
class CalibrationPoint:
    screen_norm_x: float
    screen_norm_y: float
    screen_px_x: int
    screen_px_y: int
    gaze_x: float = 0.0
    gaze_y: float = 0.0

class CalibrationManager:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cfg = getattr(CONFIG, "calibration", None)

        self.stabilization_delay_sec = 0.7
        self.sample_count = 15
        self.grid_rows = 3
        self.grid_cols = 3
        self.screen_margin_ratio = 0.10

        self.calib_file = os.path.join(os.path.dirname(__file__), "..", "config", "calibration.json")

        self.points: List[CalibrationPoint] = []
        self.current_idx: int = 0
        self.state: str = "IDLE"
        self.state_start_time: float = 0.0
        self.current_samples: List[Tuple[float, float]] = []

        # Baseline ranges
        self.is_calibrated: bool = False
        self.gaze_min_x: float = 0.38
        self.gaze_max_x: float = 0.62
        self.gaze_min_y: float = 0.35
        self.gaze_max_y: float = 0.65

        self.load_calibration()

    def start_calibration(self):
        self.points = self._generate_9_points()
        self.current_idx = 0
        self.current_samples = []
        self.state = "STABILIZING"
        self.state_start_time = time.time()
        print(f"[CALIBRATION] Starting 9-point calibration...")

    def cancel_calibration(self):
        self.state = "IDLE"
        self.current_samples = []

    def update(self, raw_gaze_x: float, raw_gaze_y: float) -> str:
        if self.state not in ["STABILIZING", "COLLECTING"]:
            return self.state

        now = time.time()
        elapsed = now - self.state_start_time

        if self.state == "STABILIZING":
            if elapsed >= self.stabilization_delay_sec:
                self.state = "COLLECTING"
                self.current_samples = []
                self.state_start_time = now

        elif self.state == "COLLECTING":
            self.current_samples.append((raw_gaze_x, raw_gaze_y))

            if len(self.current_samples) >= self.sample_count:
                arr = np.array(self.current_samples)
                med_x = float(np.median(arr[:, 0]))
                med_y = float(np.median(arr[:, 1]))

                pt = self.points[self.current_idx]
                pt.gaze_x = med_x
                pt.gaze_y = med_y

                self.current_idx += 1
                if self.current_idx >= len(self.points):
                    self._finalize_calibration()
                    self.state = "COMPLETED"
                else:
                    self.state = "STABILIZING"
                    self.state_start_time = time.time()

        return self.state

    def _finalize_calibration(self):
        all_gx = [p.gaze_x for p in self.points]
        all_gy = [p.gaze_y for p in self.points]

        self.gaze_min_x = min(all_gx)
        self.gaze_max_x = max(all_gx)
        self.gaze_min_y = min(all_gy)
        self.gaze_max_y = max(all_gy)

        self.is_calibrated = True
        self.save_calibration()

    def map_gaze_to_screen(self, gaze_x: float, gaze_y: float) -> Tuple[int, int]:
        """Direct, highly sensitive mapping to guarantee full screen traversal."""
        # Agar calibration nahi bhi hui ho ya hui ho, hum eye ke actual 0.40 - 0.60 range ko 
        # Screen ke 0 to 1920 me amplify kar rahe hain:
        min_x = self.gaze_min_x if self.is_calibrated else 0.42
        max_x = self.gaze_max_x if self.is_calibrated else 0.58
        min_y = self.gaze_min_y if self.is_calibrated else 0.38
        max_y = self.gaze_max_y if self.is_calibrated else 0.62

        span_x = max(max_x - min_x, 0.05)
        span_y = max(max_y - min_y, 0.05)

        # Ratio: 0.0 (Far Left) to 1.0 (Far Right)
        norm_x = (gaze_x - min_x) / span_x
        norm_y = (gaze_y - min_y) / span_y

        # 1.5x Multiplier to reach edges comfortably
        norm_x = (norm_x - 0.5) * 1.6 + 0.5
        norm_y = (norm_y - 0.5) * 1.6 + 0.5

        norm_x = float(np.clip(norm_x, 0.0, 1.0))
        norm_y = float(np.clip(norm_y, 0.0, 1.0))

        px_x = int(norm_x * (self.screen_w - 1))
        px_y = int(norm_y * (self.screen_h - 1))

        return px_x, px_y

    def _generate_9_points(self) -> List[CalibrationPoint]:
        xs = np.linspace(0.1, 0.9, 3)
        ys = np.linspace(0.1, 0.9, 3)
        pts = []
        for y in ys:
            for x in xs:
                px = int(x * (self.screen_w - 1))
                py = int(y * (self.screen_h - 1))
                pts.append(CalibrationPoint(screen_norm_x=float(x), screen_norm_y=float(y), screen_px_x=px, screen_px_y=py))
        return pts

    def save_calibration(self):
        data = {
            "screen_width": self.screen_w,
            "screen_height": self.screen_h,
            "bounds": {
                "gaze_min_x": self.gaze_min_x,
                "gaze_max_x": self.gaze_max_x,
                "gaze_min_y": self.gaze_min_y,
                "gaze_max_y": self.gaze_max_y,
            },
            "points": [asdict(p) for p in self.points]
        }
        with open(self.calib_file, "w") as f:
            json.dump(data, f, indent=4)

    def load_calibration(self) -> bool:
        if not os.path.exists(self.calib_file):
            return False
        try:
            with open(self.calib_file, "r") as f:
                data = json.load(f)
            self.gaze_min_x = data["bounds"]["gaze_min_x"]
            self.gaze_max_x = data["bounds"]["gaze_max_x"]
            self.gaze_min_y = data["bounds"]["gaze_min_y"]
            self.gaze_max_y = data["bounds"]["gaze_max_y"]
            self.is_calibrated = True
            return True
        except Exception:
            return False