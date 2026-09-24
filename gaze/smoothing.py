"""
gaze/smoothing.py
Fast-response Exponential Moving Average with adaptive leap-ahead.
"""

from typing import Tuple, Optional
import math

class CursorSmoother:
    def __init__(self, alpha: float = 0.35, deadzone_px: float = 5.0, max_step_px: float = 240.0):
        self.alpha = alpha
        self.deadzone_px = deadzone_px
        self.max_step_px = max_step_px

        self.current_x: Optional[float] = None
        self.current_y: Optional[float] = None

    def smooth(self, target_x: int, target_y: int) -> Tuple[int, int]:
        if self.current_x is None or self.current_y is None:
            self.current_x = float(target_x)
            self.current_y = float(target_y)
            return int(self.current_x), int(self.current_y)

        dx = target_x - self.current_x
        dy = target_y - self.current_y
        dist = math.hypot(dx, dy)

        # 1. Deadzone: Ignore micro eye flutter (< 5px)
        if dist < self.deadzone_px:
            return int(self.current_x), int(self.current_y)

        # 2. Adaptive Max Step: Allow larger jumps across screen (up to 240px)
        if dist > self.max_step_px:
            scale = self.max_step_px / dist
            target_x = int(self.current_x + dx * scale)
            target_y = int(self.current_y + dy * scale)

        # 3. EMA with 0.35 responsiveness
        self.current_x = self.alpha * target_x + (1.0 - self.alpha) * self.current_x
        self.current_y = self.alpha * target_y + (1.0 - self.alpha) * self.current_y

        return int(round(self.current_x)), int(round(self.current_y))

    def reset(self):
        self.current_x = None
        self.current_y = None