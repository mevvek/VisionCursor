"""
control/scroll_controller.py
Calibrated head-pitch scroll controller with comfortable sensitivity.
"""

import time
import ctypes
from dataclasses import dataclass
from typing import Optional
from config.settings import CONFIG

MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120

@dataclass
class ScrollStatus:
    zone: str = "NONE"             # "NONE", "UP", "DOWN"
    state: str = "IDLE"            # "IDLE", "WAITING", "ACTIVE"
    hold_duration: float = 0.0
    action: str = "NONE"
    current_offset: float = 0.0

class ScrollController:
    def __init__(self):
        cfg = getattr(CONFIG, "scroll", None)
        # Practical head-tilt threshold for 1-metre sitting posture
        self.tilt_threshold = 0.028
        self.activation_delay = 0.30              # 300ms deliberate hold
        self.interval = 0.15                      # Smooth scroll pace
        self.scroll_amount = 1                    # 1 notch per step

        self.current_direction = "NONE"
        self.tilt_start_time: Optional[float] = None
        self.last_scroll_time: float = 0.0
        self.state = "IDLE"

    def _native_scroll(self, clicks: int):
        wheel_amount = clicks * WHEEL_DELTA
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_amount, 0)

    def update(self, norm_y: float, center_y: Optional[float], enabled: bool, current_time: Optional[float] = None) -> ScrollStatus:
        now = current_time if current_time is not None else time.monotonic()

        if not enabled:
            self.reset()
            return ScrollStatus("NONE", "IDLE", 0.0, "NONE", 0.0)

        cy = center_y if center_y is not None else 0.5
        offset = norm_y - cy

        direction = "NONE"
        if offset < -self.tilt_threshold:
            direction = "UP"
        elif offset > self.tilt_threshold:
            direction = "DOWN"

        # Center neutral zone: completely freeze scrolling
        if direction == "NONE":
            self.reset()
            return ScrollStatus("NONE", "IDLE", 0.0, "NONE", round(offset, 3))

        if direction != self.current_direction:
            self.current_direction = direction
            self.tilt_start_time = now
            self.state = "WAITING"
            self.last_scroll_time = 0.0
            return ScrollStatus(self.current_direction, self.state, 0.0, "NONE", round(offset, 3))

        hold_time = now - self.tilt_start_time
        action = "NONE"

        if hold_time < self.activation_delay:
            self.state = "WAITING"
        else:
            self.state = "ACTIVE"
            if (now - self.last_scroll_time) >= self.interval:
                if self.current_direction == "UP":
                    self._native_scroll(self.scroll_amount)
                    action = "SCROLL_UP"
                elif self.current_direction == "DOWN":
                    self._native_scroll(-self.scroll_amount)
                    action = "SCROLL_DOWN"
                self.last_scroll_time = now

        return ScrollStatus(
            zone=self.current_direction,
            state=self.state,
            hold_duration=round(hold_time, 2),
            action=action,
            current_offset=round(offset, 3)
        )

    def reset(self):
        self.current_direction = "NONE"
        self.tilt_start_time = None
        self.last_scroll_time = 0.0
        self.state = "IDLE"