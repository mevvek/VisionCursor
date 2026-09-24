"""
control/cursor_controller.py
Smooth and jitter-free cursor controller with micro-deadzone stabilization.
"""

from typing import Tuple, Optional
import math
import pyautogui
from config.settings import CONFIG

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.001

class CursorController:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h

        cfg = getattr(CONFIG, "cursor", None)
        self.range_x = getattr(cfg, "range_x", 0.13)
        self.range_y = getattr(cfg, "range_y", 0.10)
        self.alpha = getattr(cfg, "smoothing_alpha", 0.22)
        self.deadzone_px = getattr(cfg, "deadzone_px", 7.0)

        self.center_x: Optional[float] = None
        self.center_y: Optional[float] = None

        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None

        self.is_enabled: bool = False

    def toggle(self, current_norm_pt: Tuple[float, float]) -> bool:
        self.is_enabled = not self.is_enabled
        if self.is_enabled:
            self.recenter(current_norm_pt)
        return self.is_enabled

    def recenter(self, current_norm_pt: Tuple[float, float]):
        self.center_x = current_norm_pt[0]
        self.center_y = current_norm_pt[1]
        self.smooth_x = None
        self.smooth_y = None

    def update_position(self, norm_pt: Tuple[float, float]) -> Tuple[int, int]:
        if not self.is_enabled:
            mx, my = pyautogui.position()
            return int(mx), int(my)

        if self.center_x is None:
            self.recenter(norm_pt)

        # Deviation from center
        dx = norm_pt[0] - self.center_x
        dy = norm_pt[1] - self.center_y

        target_norm_x = 0.5 + (dx / (self.range_x * 2.0))
        target_norm_y = 0.5 + (dy / (self.range_y * 2.0))

        target_norm_x = max(0.0, min(1.0, target_norm_x))
        target_norm_y = max(0.0, min(1.0, target_norm_y))

        raw_px_x = target_norm_x * (self.screen_w - 1)
        raw_px_y = target_norm_y * (self.screen_h - 1)

        # First frame initialization
        if self.smooth_x is None or self.smooth_y is None:
            self.smooth_x = raw_px_x
            self.smooth_y = raw_px_y
            return int(self.smooth_x), int(self.smooth_y)

        # Micro-tremor Deadzone: If head moved less than deadzone_px, freeze cursor
        dist = math.hypot(raw_px_x - self.smooth_x, raw_px_y - self.smooth_y)
        if dist < self.deadzone_px:
            return int(round(self.smooth_x)), int(round(self.smooth_y))

        # Heavy stabilization smoothing
        self.smooth_x = self.alpha * raw_px_x + (1.0 - self.alpha) * self.smooth_x
        self.smooth_y = self.alpha * raw_px_y + (1.0 - self.alpha) * self.smooth_y

        final_x = int(round(self.smooth_x))
        final_y = int(round(self.smooth_y))

        try:
            pyautogui.moveTo(final_x, final_y)
        except Exception:
            pass

        return final_x, final_y