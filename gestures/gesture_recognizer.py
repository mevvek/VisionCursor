"""
gestures/gesture_recognizer.py
Dual-Branch Intentional Gesture Engine:
Decides whether an action is SINGLE or DOUBLE blink based on the decision window.
Auto-resets to WAITING state after confirmation.
"""

import time
from dataclasses import dataclass
from typing import Optional
from config.settings import CONFIG
from gestures.blink_detector import BlinkEvent

@dataclass
class GestureResult:
    action: str = "NONE"                    # "NONE", "SINGLE_CONFIRMED", "DOUBLE_CONFIRMED"
    state: str = "WAITING"                  # "WAITING", "WAITING FOR 2ND BLINK", "CONFIRMED"
    time_remaining: float = 0.0
    interval: float = 0.0
    timestamp: float = 0.0

class GestureRecognizer:
    def __init__(self):
        cfg = getattr(CONFIG, "gesture", None)
        self.min_interval = getattr(cfg, "double_blink_min_interval", 0.15)
        self.decision_window = getattr(cfg, "decision_window_max", 0.60)
        self.cooldown = getattr(cfg, "gesture_cooldown", 0.40)

        self.state = "WAITING"
        self.first_blink_time: Optional[float] = None
        self.last_confirmed_time: float = 0.0

    def update(self, blink_event: BlinkEvent, current_time: Optional[float] = None) -> GestureResult:
        now = current_time if current_time is not None else time.monotonic()
        action = "NONE"
        measured_interval = 0.0
        remaining_time = 0.0

        # Post-confirmation cooldown (Blank slate lock)
        if (now - self.last_confirmed_time) < self.cooldown:
            return GestureResult(action="NONE", state="WAITING", timestamp=now)

        if self.state == "WAITING":
            if blink_event.blink_detected:
                # Deliberate Blink 1 detected -> Start countdown window
                self.state = "WAITING FOR 2ND BLINK"
                self.first_blink_time = now

        elif self.state == "WAITING FOR 2ND BLINK":
            elapsed = now - self.first_blink_time
            remaining_time = max(0.0, self.decision_window - elapsed)

            # Scenario A: Second blink arrives inside window
            if blink_event.blink_detected:
                measured_interval = elapsed
                if self.min_interval <= measured_interval <= self.decision_window:
                    action = "DOUBLE_CONFIRMED"
                    self.last_confirmed_time = now
                    self.state = "WAITING"
                    self.first_blink_time = None
                else:
                    # Noise or flutter -> Restart first blink anchor
                    self.first_blink_time = now

            # Scenario B: Window expired with NO second blink -> Intentional SINGLE action confirmed
            elif elapsed >= self.decision_window:
                action = "SINGLE_CONFIRMED"
                self.last_confirmed_time = now
                self.state = "WAITING"
                self.first_blink_time = None

        return GestureResult(
            action=action,
            state=self.state,
            time_remaining=round(remaining_time, 2),
            interval=round(measured_interval, 2),
            timestamp=now
        )