"""
main.py - Phase 4 Calibration Verification
Integrates 9-point Fullscreen Calibration with live Gaze-to-Screen coordinate mapping.
Physical mouse cursor remains untouched.
"""

import sys
import cv2
import numpy as np
import pyautogui  # Used ONLY to read screen resolution safely

from config.settings import CONFIG
from camera.camera_manager import CameraManager
from vision.face_tracker import FaceTracker
from gaze.gaze_estimator import GazeEstimator
from gaze.calibration import CalibrationManager
from ui.calibration_window import CalibrationWindow

def main():
    print("=" * 60)
    print("VisionCursor: Initializing Phase 4 (Gaze Calibration System)...")
    print("=" * 60)

    # Detect primary display dimensions
    screen_w, screen_h = pyautogui.size()
    print(f"[DISPLAY] Detected Primary Screen Resolution: {screen_w} x {screen_h}")

    cam = CameraManager(
        source=CONFIG.camera.device_id,
        width=CONFIG.camera.width,
        height=CONFIG.camera.height,
        fps=CONFIG.camera.fps
    )

    if not cam.start():
        print("[ERROR] Camera failed to start.")
        sys.exit(1)

    tracker = FaceTracker()
    gaze_estimator = GazeEstimator()
    calib_mgr = CalibrationManager(screen_w, screen_h)
    calib_ui = CalibrationWindow(calib_mgr)

    calib_ui.open()
    print("\n[INSTRUCTIONS]:")
    print(" - Look at the Fullscreen Calibration window.")
    print(" - Press 'c' to begin the 9-point calibration.")
    print(" - Keep your gaze steady on each dot as it turns from Orange to Green.")
    print(" - Press 'ESC' to exit calibration or cancel anytime.")
    print(" - PHYSICAL MOUSE CURSOR WILL NOT MOVE.\n")

    try:
        while True:
            success, frame = cam.read_frame()
            if not success:
                break

            tracking_res = tracker.process_frame(frame)
            gaze_res = gaze_estimator.estimate_gaze(tracking_res)

            # Update calibration state machine with raw iris features
            calib_state = calib_mgr.update(gaze_res.raw_x, gaze_res.raw_y)

            # Predict screen position if calibrated (Visual confirmation only)
            predicted_pt = None
            if calib_mgr.is_calibrated:
                predicted_pt = calib_mgr.map_gaze_to_screen(gaze_res.smooth_x, gaze_res.smooth_y)

            # Render full-screen calibration UI
            ui_frame = calib_ui.render(predicted_pt)
            cv2.imshow(calib_ui.window_name, ui_frame)

            # Small diagnostic preview in camera window
            cv2.putText(frame, f"Calibration: {calib_state}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Gaze: ({gaze_res.raw_x:.2f}, {gaze_res.raw_y:.2f})", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("VisionCursor - Camera Preview", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                if calib_mgr.state in ["STABILIZING", "COLLECTING"]:
                    calib_mgr.cancel_calibration()
                else:
                    break
            elif key == ord('c'):
                calib_mgr.start_calibration()
            elif key == ord('q'):
                break

    finally:
        calib_ui.close()
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        print("\n[SUCCESS] Phase 4 completed cleanly.")

if __name__ == "__main__":
    main()