"""
config/settings.py
Central configuration file for VisionCursor.
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
    calibration_file: str = os.path.join(os.path.dirname(__file__), "calibration.json")

@dataclass
class GazeConfig:
    horizontal_left_thresh: float = 0.42
    horizontal_right_thresh: float = 0.58
    vertical_up_thresh: float = 0.38
    vertical_down_thresh: float = 0.65
    smoothing_factor: float = 0.25

@dataclass
class EyeTrackingConfig:
    # EAR Thresholds
    ear_closed_threshold: float = 0.21        # EAR <= 0.21 closed
    ear_open_threshold: float = 0.25          # EAR >= 0.25 open
    min_blink_duration: float = 0.06          # 60ms minimum
    max_blink_duration: float = 1.20          # 1.2 sec maximum (lambe blink bhi pakdega)
    blink_cooldown: float = 0.20              # Cooldown between clicks
    both_eyes_required: bool = True           # Accidental winks reject karega
    enable_blink_click: bool = True           # Blink = Left Mouse Click ON!

@dataclass
class CursorConfig:
    control_enabled_on_startup: bool = False
    smoothing_alpha: float = 0.22             # Stabilized smoothing (jitter khatam)
    deadzone_px: float = 7.0                  # Micro head shakes ko freeze karega
    range_x: float = 0.13                     # Horizontal sensitivity
    range_y: float = 0.10                     # Vertical sensitivity

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    eye: EyeTrackingConfig = field(default_factory=EyeTrackingConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)

CONFIG = AppConfig()