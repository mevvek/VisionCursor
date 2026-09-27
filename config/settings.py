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
    min_blink_duration: float = 0.20
    max_blink_duration: float = 1.20
    blink_cooldown: float = 0.10
    both_eyes_required: bool = True

@dataclass
class GestureConfig:
    double_blink_min_interval: float = 0.12
    decision_window_max: float = 1.45
    gesture_cooldown: float = 0.45

@dataclass
class CursorConfig:
    control_enabled_on_startup: bool = False
    smoothing_alpha: float = 0.13             # Exponential smoothing factor
    deadzone_px: float = 11.0                 # Suppresses micro-tremors when resting
    range_x: float = 0.12                     # Horizontal motion sensitivity
    range_y: float = 0.065                    # Vertical motion sensitivity
    vertical_gain: float = 1.25               # Vertical reach multiplier for distance use

@dataclass
class ClickConfig:
    gaze_stability_duration: float = 0.35     # Required fixation hold time before click validation
    gaze_stability_radius_px: float = 40.0    # Target fixation radius in pixels
    click_cooldown: float = 0.80              # Minimum delay between successive clicks

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    eye: EyeTrackingConfig = field(default_factory=EyeTrackingConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    click: ClickConfig = field(default_factory=ClickConfig)

CONFIG = AppConfig()