"""
control/cursor_controller.py
Cursor stabilization module utilizing adaptive exponential smoothing and deadzone filtering.
"""

import math
import pyautogui
from config.settings import CONFIG

class CursorController:
    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h

        cfg = getattr(CONFIG, "cursor", None)
        self.base_alpha = getattr(cfg, "smoothing_alpha", 0.13)
        self.deadzone = getattr(cfg, "deadzone_px", 11.0)
        self.range_x = getattr(cfg, "range_x", 0.12)
        self.range_y = getattr(cfg, "range_y", 0.065)
        self.v_gain = getattr(cfg, "vertical_gain", 1.25)

        self.center_x = None
        self.center_y = None
        self.prev_x = screen_w // 2
        self.prev_y = screen_h // 2
        self.is_enabled = False

    def toggle(self, current_nose_point=None):
        self.is_enabled = not self.is_enabled
        if self.is_enabled and current_nose_point:
            self.recenter(current_nose_point)
        return self.is_enabled

    def recenter(self, current_nose_point):
        self.center_x, self.center_y = current_nose_point
        cur_x, cur_y = pyautogui.position()
        self.prev_x = float(cur_x)
        self.prev_y = float(cur_y)

    def update_position(self, current_nose_point):
        if not self.is_enabled or self.center_x is None:
            return pyautogui.position()

        nx, ny = current_nose_point

        dx = nx - self.center_x
        dy = (ny - self.center_y) * self.v_gain

        norm_target_x = 0.5 + (dx / (self.range_x * 2.0))
        norm_target_y = 0.5 + (dy / (self.range_y * 2.0))

        # Clamp normalized coordinates to stay within screen bounds
        norm_target_x = max(0.0, min(1.0, norm_target_x))
        norm_target_y = max(0.0, min(1.0, norm_target_y))

        target_px_x = norm_target_x * self.screen_w
        target_px_y = norm_target_y * self.screen_h

        diff_x = target_px_x - self.prev_x
        diff_y = target_px_y - self.prev_y
        dist = math.hypot(diff_x, diff_y)

        # Micro-deadzone check to prevent jitter when holding target
        if dist < self.deadzone:
            smooth_x = self.prev_x
            smooth_y = self.prev_y
        else:
            # Dynamic alpha scaling based on movement speed
            speed_factor = min(1.0, (dist - self.deadzone) / 80.0)
            effective_alpha = self.base_alpha + (0.12 * speed_factor)

            smooth_x = self.prev_x + effective_alpha * diff_x
            smooth_y = self.prev_y + effective_alpha * diff_y

        smooth_x = max(0.0, min(float(self.screen_w - 1), smooth_x))
        smooth_y = max(0.0, min(float(self.screen_h - 1), smooth_y))

        int_x, int_y = int(smooth_x), int(smooth_y)
        pyautogui.moveTo(int_x, int_y, _pause=False)

        self.prev_x = smooth_x
        self.prev_y = smooth_y

        return int_x, int_y