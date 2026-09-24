"""
control/cursor_controller.py
Smooth and responsive cursor controller based on nose-anchor navigation.
Corrected for natural front-camera orientation.
"""

from typing import Tuple, Optional
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.001

class CursorController:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h

        # Sensitivity box: kitna head tilt karne par edge tak jaye
        self.range_x = 0.12
        self.range_y = 0.09

        self.center_x: Optional[float] = None
        self.center_y: Optional[float] = None

        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None
        self.alpha = 0.35  # Smooth and snappy

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

        # FRONT WEBCAM FIX:
        # Looking left tilts face to your left, which moves camera pixel to the left (-dx)
        # So: screen_x moves left (subtracted), screen_y moves down/up naturally
        dx = norm_pt[0] - self.center_x
        dy = norm_pt[1] - self.center_y

        target_norm_x = 0.5 + (dx / (self.range_x * 2.0))
        target_norm_y = 0.5 + (dy / (self.range_y * 2.0))

        # Clamp inside boundaries
        target_norm_x = max(0.0, min(1.0, target_norm_x))
        target_norm_y = max(0.0, min(1.0, target_norm_y))

        raw_px_x = target_norm_x * (self.screen_w - 1)
        raw_px_y = target_norm_y * (self.screen_h - 1)

        # Exponential Moving Average Smoothing
        if self.smooth_x is None:
            self.smooth_x = raw_px_x
            self.smooth_y = raw_px_y
        else:
            self.smooth_x = self.alpha * raw_px_x + (1.0 - self.alpha) * self.smooth_x
            self.smooth_y = self.alpha * raw_px_y + (1.0 - self.alpha) * self.smooth_y

        final_x = int(round(self.smooth_x))
        final_y = int(round(self.smooth_y))

        try:
            pyautogui.moveTo(final_x, final_y)
        except Exception:
            pass

        return final_x, final_y