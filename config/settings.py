"""
config/settings.py
Central configuration file for VisionCursor.
"""

import os
from dataclasses import dataclass, field

@dataclass
class CameraConfig:
    device_id: int = 0                  # 0: Default laptop webcam
    width: int = 640                    # 640x480 for real-time 30+ FPS
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
class CursorConfig:
    control_enabled_on_startup: bool = False   # Safe startup: OFF by default
    smoothing_alpha: float = 0.22             # EMA weight for target cursor (0.15 - 0.35)
    movement_threshold_px: float = 6.0        # Ignore small eye tremors < 6 px
    max_cursor_step_px: float = 85.0          # Max pixels cursor can jump per frame
    emergency_pause_key: str = "ctrl+shift+p"

@dataclass
class EyeTrackingConfig:
    ear_closed_threshold: float = 0.20
    ear_open_threshold: float = 0.25
    min_blink_duration: float = 0.05
    max_blink_duration: float = 0.35
    double_blink_max_interval: float = 0.45
    click_cooldown: float = 0.8

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    eye: EyeTrackingConfig = field(default_factory=EyeTrackingConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)

CONFIG = AppConfig()