"""
gaze/calibration.py
Collects 9-point iris landmarks, calculates calibration mapping, and persists JSON data.
No cursor movement.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
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
        
        # Robust fallback configurations
        self.stabilization_delay_sec = getattr(self.cfg, "stabilization_delay_sec", 0.8)
        self.sample_count = getattr(self.cfg, "sample_count", 25)
        self.grid_rows = getattr(self.cfg, "grid_rows", 3)
        self.grid_cols = getattr(self.cfg, "grid_cols", 3)
        self.screen_margin_ratio = getattr(self.cfg, "screen_margin_ratio", 0.10)
        
        self.calib_file = getattr(
            self.cfg, "calibration_file_path", 
            getattr(self.cfg, "calibration_file", os.path.join(os.path.dirname(__file__), "..", "config", "calibration.json"))
        )

        self.points: List[CalibrationPoint] = []
        self.current_idx: int = 0

        # State tracking: IDLE, STABILIZING, COLLECTING, COMPLETED
        self.state: str = "IDLE"
        self.state_start_time: float = 0.0
        self.current_samples: List[Tuple[float, float]] = []

        # Mapping bounds
        self.is_calibrated: bool = False
        self.gaze_min_x: float = 0.0
        self.gaze_max_x: float = 1.0
        self.gaze_min_y: float = 0.0
        self.gaze_max_y: float = 1.0

        # Try loading existing calibration if present
        self.load_calibration()

    def start_calibration(self):
        """Prepares 9-point grid and initializes collection sequence."""
        self.points = self._generate_9_points()
        self.current_idx = 0
        self.current_samples = []
        self.state = "STABILIZING"
        self.state_start_time = time.time()
        print(f"[CALIBRATION] Started 9-point sequence for display: {self.screen_w}x{self.screen_h}")

    def cancel_calibration(self):
        """Cancels current sequence without corrupting previous valid calibration."""
        self.state = "IDLE"
        self.current_samples = []
        print("[CALIBRATION] Sequence cancelled by user.")

    def update(self, raw_gaze_x: float, raw_gaze_y: float) -> str:
        """State machine update called on each camera frame."""
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
                # Compute median to reject micro-flutter
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
        """Calculates mapping bounds and saves to JSON."""
        all_gx = [p.gaze_x for p in self.points]
        all_gy = [p.gaze_y for p in self.points]

        self.gaze_min_x = min(all_gx)
        self.gaze_max_x = max(all_gx)
        self.gaze_min_y = min(all_gy)
        self.gaze_max_y = max(all_gy)

        self.is_calibrated = True
        self.save_calibration()
        print("[CALIBRATION] Sequence successfully completed and saved.")

    def map_gaze_to_screen(self, gaze_x: float, gaze_y: float) -> Tuple[int, int]:
        """Maps live gaze coordinates to screen pixel coordinates (Without moving cursor)."""
        if not self.is_calibrated:
            return self.screen_w // 2, self.screen_h // 2

        # Normalize relative to user's calibrated gaze extremes
        span_x = max(self.gaze_max_x - self.gaze_min_x, 0.001)
        span_y = max(self.gaze_max_y - self.gaze_min_y, 0.001)

        norm_sx = (gaze_x - self.gaze_min_x) / span_x
        norm_sy = (gaze_y - self.gaze_min_y) / span_y

        norm_sx = float(np.clip(norm_sx, 0.0, 1.0))
        norm_sy = float(np.clip(norm_sy, 0.0, 1.0))

        px_x = int(norm_sx * (self.screen_w - 1))
        px_y = int(norm_sy * (self.screen_h - 1))

        return px_x, px_y

    def _generate_9_points(self) -> List[CalibrationPoint]:
        margin = self.screen_margin_ratio
        xs = np.linspace(margin, 1.0 - margin, self.grid_cols)
        ys = np.linspace(margin, 1.0 - margin, self.grid_rows)

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
        print(f"[CALIBRATION] Saved JSON configuration to: {self.calib_file}")

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
            print(f"[CALIBRATION] Loaded existing calibration from {self.calib_file}")
            return True
        except Exception as e:
            print(f"[WARNING] Could not parse calibration file: {e}")
            return False