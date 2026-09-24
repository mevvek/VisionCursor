"""
config/settings.py
Central configuration file for VisionCursor.
"""

from dataclasses import dataclass, field

@dataclass
class CameraConfig:
    device_id: int = 0                  # 0: Default laptop webcam
    width: int = 640                    # 640x480 for real-time 30+ FPS
    height: int = 480
    fps: int = 30

@dataclass
class GazeConfig:
    # Normalized iris ratio boundaries (0.0 to 1.0)
    # Looking Right moves iris towards outer right corner
    # Looking Left moves iris towards outer left corner
    horizontal_left_thresh: float = 0.42    # x <= 0.42 -> LOOKING LEFT
    horizontal_right_thresh: float = 0.58   # x >= 0.58 -> LOOKING RIGHT

    # Looking Up moves iris towards top eyelid
    # Looking Down moves iris towards lower eyelid
    vertical_up_thresh: float = 0.38        # y <= 0.38 -> LOOKING UP
    vertical_down_thresh: float = 0.65      # y >= 0.65 -> LOOKING DOWN

    # Smoothing factor (alpha) for Exponential Moving Average
    # 0.1 = very smooth/slight delay, 0.4 = responsive/mild jitter
    smoothing_factor: float = 0.25

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
    gaze: GazeConfig = field(default_factory=GazeConfig)
    eye: EyeTrackingConfig = field(default_factory=EyeTrackingConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)

CONFIG = AppConfig()