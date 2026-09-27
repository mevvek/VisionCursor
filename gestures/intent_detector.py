"""
gestures/intent_detector.py
Evaluates spatial gaze stability and correlates with confirmed double-blink gestures.
Ensures that clicking only occurs on stable target fixation.
"""

import time
import math
from typing import Tuple, Optional
from dataclasses import dataclass
from config.settings import CONFIG

@dataclass
class GazeStabilityResult:
    is_stable: bool = False
    stable_duration: float = 0.0
    reference_px: Tuple[int, int] = (0, 0)

@dataclass
class IntentResult:
    trigger_click: bool = False
    gaze_stable: bool = False
    stable_duration: float = 0.0
    status_label: str = "SEARCHING"

class IntentDetector:
    def __init__(self):
        cfg = getattr(CONFIG, "click", None)
        self.req_duration = getattr(cfg, "gaze_stability_duration", 0.35)
        self.radius = getattr(cfg, "gaze_stability_radius_px", 35.0)

        self.ref_pos: Optional[Tuple[int, int]] = None
        self.stable_start_time: Optional[float] = None
        self.current_duration: float = 0.0
        self.is_stable: bool = False

    def update_stability(self, cursor_px: Tuple[int, int], current_time: Optional[float] = None) -> GazeStabilityResult:
        now = current_time if current_time is not None else time.monotonic()
        cx, cy = cursor_px

        if self.ref_pos is None:
            self.ref_pos = (cx, cy)
            self.stable_start_time = now
            self.current_duration = 0.0
            self.is_stable = False
            return GazeStabilityResult(False, 0.0, self.ref_pos)

        # Distance from stable anchor reference
        rx, ry = self.ref_pos
        dist = math.hypot(cx - rx, cy - ry)

        if dist <= self.radius:
            # Still dwelling within stability radius
            self.current_duration = now - self.stable_start_time
            if self.current_duration >= self.req_duration:
                self.is_stable = True
        else:
            # Gaze moved outside target tolerance - anchor shifts
            self.ref_pos = (cx, cy)
            self.stable_start_time = now
            self.current_duration = 0.0
            self.is_stable = False

        return GazeStabilityResult(self.is_stable, round(self.current_duration, 2), self.ref_pos)

    def evaluate_intent(self,
                        cursor_px: Tuple[int, int],
                        double_blink_confirmed: bool,
                        cursor_active: bool,
                        calibration_present: bool,
                        cooldown_finished: bool,
                        current_time: Optional[float] = None) -> IntentResult:
        """
        Validates the strict intersection:
        Cursor Active + Calibrated + Gaze Stable + Double Blink + Cooldown Finished -> Trigger Click
        """
        now = current_time if current_time is not None else time.monotonic()
        stab = self.update_stability(cursor_px, now)

        # Baseline safety guards
        if not calibration_present:
            return IntentResult(False, stab.is_stable, stab.stable_duration, "CALIBRATION REQ")

        if not cursor_active:
            return IntentResult(False, stab.is_stable, stab.stable_duration, "STANDBY")

        if not cooldown_finished:
            return IntentResult(False, stab.is_stable, stab.stable_duration, "COOLDOWN")

        # Status categorization
        status = "TARGET LOCKED" if stab.is_stable else "FIXATING..."

        # Intent verification rule
        trigger = False
        if double_blink_confirmed:
            if stab.is_stable:
                trigger = True
                status = "CLICK TRIGGERED"
                # Post-click stability reset: require fresh fixation
                self.reset_stability()
            else:
                status = "REJECTED (UNSTABLE)"

        return IntentResult(trigger, stab.is_stable, stab.stable_duration, status)

    def reset_stability(self):
        """Resets the stability anchor after a verified click action."""
        self.ref_pos = None
        self.stable_start_time = None
        self.current_duration = 0.0
        self.is_stable = False
