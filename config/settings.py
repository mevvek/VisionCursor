"""
config/settings.py
Central configuration file for VISION CURSOR.
"""

import os
from dataclasses import dataclass, field

@dataclass
class CameraConfig:
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30

@dataclass
class CalibrationConfig:
    stabilization_delay_sec: float = 0.8
    sample_count: int = 25
    grid_rows: int = 3
    grid_cols: int = 3
    screen_margin_ratio: float = 0.10
    calibration_file_path: str = os.path.join(os.path.dirname(__file__), "calibration.json")

@dataclass
class GazeConfig:
    horizontal_left_thresh: float = 0.42
    horizontal_right_thresh: float = 0.58
    vertical_up_thresh: float = 0.38
    vertical_down_thresh: float = 0.65
    smoothing_factor: float = 0.25

@dataclass
class EyeTrackingConfig:
    ear_closed_threshold: float = 0.22
    ear_open_threshold: float = 0.26
    # --- FLEXIBLE DURATION ---
    # 0.20s se quick intentional second blink bhi drop nahi hoga
    min_blink_duration: float = 0.20          
    max_blink_duration: float = 1.20          
    blink_cooldown: float = 0.10
    both_eyes_required: bool = True

@dataclass
class GestureConfig:
    # --- RELAXED DOUBLE BLINK WINDOW ---
    double_blink_min_interval: float = 0.12   # Rapid tap allow karega
    decision_window_max: float = 1.45         # 1.45s ka aaram se pura waqt milega 2nd blink ke liye
    gesture_cooldown: float = 0.45            # Action confirm hone ke baad reset

@dataclass
class CursorConfig:
    control_enabled_on_startup: bool = False
    smoothing_alpha: float = 0.22
    deadzone_px: float = 7.0
    range_x: float = 0.13
    range_y: float = 0.10

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    eye: EyeTrackingConfig = field(default_factory=EyeTrackingConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)

CONFIG = AppConfig()