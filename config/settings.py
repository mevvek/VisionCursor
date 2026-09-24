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
    stabilization_delay_sec: float = 0.8     # Time to settle eyes before recording
    sample_count: int = 25                  # Frames to average per point
    grid_rows: int = 3                      # 3x3 = 9 calibration points
    grid_cols: int = 3
    screen_margin_ratio: float = 0.10       # 10% margin from display boundaries
    calibration_file_path: str = os.path.join(os.path.dirname(__file__), "calibration.json")
    calibration_file: str = os.path.join(os.path.dirname(__file__), "calibration.json")

@dataclass
class GazeConfig:
    # Normalized iris ratio boundaries (0.0 to 1.0)
    horizontal_left_thresh: float = 0.42    # x <= 0.42 -> LOOKING LEFT
    horizontal_right_thresh: float = 0.58   # x >= 0.58 -> LOOKING RIGHT
    vertical_up_thresh: float = 0.38        # y <= 0.38 -> LOOKING UP
    vertical_down_thresh: float = 0.65      # y >= 0.65 -> LOOKING DOWN
    smoothing_factor: float = 0.25          # EMA smoothing factor

@dataclass
class EyeTrackingConfig:
    ear_closed_threshold: float = 0.20
    ear_open_threshold: float = 0.25
    min_blink_duration: float = 0.05
    max_blink_duration: float = 0.35
    double_blink_max_interval: float = 0.45
    click_cooldown: float = 0.8

@dataclass
class CursorConfig:
    smoothing_factor: float = 0.25
    deadzone_radius_px: float = 8.0
    dwell_time_seconds: float = 0.8
    emergency_pause_key: str = "ctrl+shift+p"

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    eye: EyeTrackingConfig = field(default_factory=EyeTrackingConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)

CONFIG = AppConfig()