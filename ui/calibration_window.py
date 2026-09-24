"""
ui/calibration_window.py
Interactive Fullscreen Calibration Overlay.
Renders target dots, instructions, countdowns, and gaze mapping verification.
"""

import cv2
import numpy as np
from gaze.calibration import CalibrationManager

class CalibrationWindow:
    def __init__(self, manager: CalibrationManager, window_name: str = "VisionCursor Calibration"):
        self.manager = manager
        self.window_name = window_name

    def open(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def close(self):
        cv2.destroyWindow(self.window_name)

    def render(self, predicted_screen_pt: tuple = None) -> np.ndarray:
        w, h = self.manager.screen_w, self.manager.screen_h
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Header bar instructions
        cv2.putText(canvas, "VisionCursor - Calibration Mode", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, "Press 'c' to Start | 'ESC' to Cancel / Exit", (40, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        state = self.manager.state

        if state == "IDLE":
            msg = "Ready. Look naturally at the screen and press 'c' to start 9-point calibration."
            cv2.putText(canvas, msg, (w // 2 - 380, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.manager.is_calibrated and predicted_screen_pt is not None:
                # Live validation cursor simulation (Green dot, NOT the real mouse)
                cv2.circle(canvas, predicted_screen_pt, 12, (0, 255, 0), -1)
                cv2.circle(canvas, predicted_screen_pt, 22, (0, 255, 0), 2)
                cv2.putText(canvas, "Estimated Gaze Point (Validation Only)", (predicted_screen_pt[0] + 25, predicted_screen_pt[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        elif state in ["STABILIZING", "COLLECTING"]:
            idx = self.manager.current_idx
            total = len(self.manager.points)
            pt = self.manager.points[idx]

            # Progress banner
            cv2.putText(canvas, f"Target {idx + 1} of {total}", (w // 2 - 80, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

            if state == "STABILIZING":
                status = "Keep your gaze steady on the orange dot..."
                color = (0, 140, 255)  # Orange
                pulse_r = 18
            else:
                samples_got = len(self.manager.current_samples)
                sample_max = self.manager.cfg.sample_count
                status = f"Recording iris data... ({samples_got}/{sample_max})"
                color = (0, 255, 0)    # Green
                pulse_r = 14

            cv2.putText(canvas, status, (w // 2 - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            # Draw target dot
            cv2.circle(canvas, (pt.screen_px_x, pt.screen_px_y), pulse_r, color, -1)
            cv2.circle(canvas, (pt.screen_px_x, pt.screen_px_y), pulse_r + 10, (255, 255, 255), 2)

        elif state == "COMPLETED":
            cv2.putText(canvas, "Calibration Successful! Data saved to config/calibration.json",
                        (w // 2 - 360, h // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, "Look around to see green gaze validation point. Press ESC to continue.",
                        (w // 2 - 340, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            if predicted_screen_pt is not None:
                cv2.circle(canvas, predicted_screen_pt, 12, (0, 255, 0), -1)
                cv2.circle(canvas, predicted_screen_pt, 22, (0, 255, 0), 2)

        return canvas