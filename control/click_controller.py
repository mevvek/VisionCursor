"""
control/click_controller.py
Encapsulates safe single left-click actions using PyAutoGUI.
Strictly disallows right-clicks, double-clicks, and scrolls.
"""

import time
import pyautogui
from config.settings import CONFIG

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.001

class ClickController:
    def __init__(self):
        cfg = getattr(CONFIG, "click", None)
        self.cooldown = getattr(cfg, "click_cooldown", 0.80)
        self.last_click_time: float = 0.0

    @property
    def can_click(self) -> bool:
        return (time.monotonic() - self.last_click_time) >= self.cooldown

    def left_click(self) -> bool:
        """Executes a single left mouse click at the current cursor coordinate."""
        now = time.monotonic()
        if (now - self.last_click_time) < self.cooldown:
            return False

        try:
            pyautogui.click()
            self.last_click_time = now
            return True
        except Exception as e:
            print(f"[ERROR] ClickController failed: {e}")
            return False