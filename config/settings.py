"""
config/settings.py
Central configuration file for VisionCursor.
All sensitivity parameters, blink timings, and thresholds are configured here.
"""

from dataclasses import dataclass, field

@dataclass
class CameraConfig:
    device_id: int = 0                  # 0: Default laptop webcam
    width: int = 640                    # Keep 640x480 for fast, real-time FPS
    height: int = 480
    fps: int = 30

@dataclass
class EyeTrackingConfig:
    # Eye Aspect Ratio (EAR) Threshold for blink detection
    ear_closed_threshold: float = 0.20  # EAR <= 0.20 is treated as eye closed
    ear_open_threshold: float = 0.25    # EAR >= 0.25 is treated as eye fully opened

    # Blink timing (seconds)
    min_blink_duration: float = 0.05    # Filter out micro-flutter / sensor noise
    max_blink_duration: float = 0.35    # Longer than this is considered a long blink/close
    double_blink_max_interval: float = 0.45  # Max gap between 1st blink end and 2nd blink start
    click_cooldown: float = 0.8         # Cooldown window post-click to avoid double firing

@dataclass
class CursorConfig:
    # Cursor stabilization & smoothing
    smoothing_factor: float = 0.25      # Exponential smoothing factor (alpha: 0.1 to 0.4)
    deadzone_radius_px: float = 8.0     # Pixels of deadzone to eliminate micro-jitter
    dwell_time_seconds: float = 0.8     # Time gaze must hold still before gesture confirmation
    emergency_pause_key: str = "ctrl+shift+p"

@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    eye: EyeTrackingConfig = field(default_factory=EyeTrackingConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)

CONFIG = AppConfig()