"""
main.py - Phase 3 Verification
Integrates camera, face/iris tracking, and gaze direction estimation with visual radar.
Zero cursor control / No PyAutoGUI.
"""

import cv2
import sys
import numpy as np
from config.settings import CONFIG
from camera.camera_manager import CameraManager
from vision.face_tracker import FaceTracker
from gaze.gaze_estimator import GazeEstimator, GazeResult

def draw_visual_overlay(frame, tracking_result):
    """Draws eye contours and iris markers."""
    if not tracking_result.face_detected:
        return frame

    if tracking_result.left_eye:
        cv2.polylines(frame, [tracking_result.left_eye.contour_points], isClosed=True, color=(255, 255, 0), thickness=1)
    if tracking_result.right_eye:
        cv2.polylines(frame, [tracking_result.right_eye.contour_points], isClosed=True, color=(255, 255, 0), thickness=1)

    for iris in [tracking_result.left_iris, tracking_result.right_iris]:
        if iris:
            cv2.polylines(frame, [iris.points], isClosed=True, color=(0, 255, 120), thickness=1)
            cv2.circle(frame, iris.center, 3, (0, 0, 255), -1)

    return frame

def draw_gaze_radar(frame, gaze_res: GazeResult, pos=(500, 30), size=(110, 110)):
    """Draws a mini 2D radar box in the top-right showing real-time iris travel."""
    x0, y0 = pos
    w, h = size

    # Background radar box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (100, 100, 100), 1)

    # Crosshairs
    cx = x0 + w // 2
    cy = y0 + h // 2
    cv2.line(frame, (cx, y0 + 10), (cx, y0 + h - 10), (60, 60, 60), 1)
    cv2.line(frame, (x0 + 10, cy), (x0 + w - 10, cy), (60, 60, 60), 1)

    # Normalized gaze dot
    dot_x = int(x0 + np.clip(gaze_res.smooth_x, 0.05, 0.95) * w)
    dot_y = int(y0 + np.clip(gaze_res.smooth_y, 0.05, 0.95) * h)

    # Color dot based on direction
    dot_color = (0, 255, 255) if gaze_res.direction == "CENTER" else (0, 120, 255)
    cv2.circle(frame, (dot_x, dot_y), 6, dot_color, -1)
    cv2.putText(frame, "Gaze Radar", (x0 + 12, y0 + h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

def draw_hud(frame, fps: float, gaze_res: GazeResult):
    """Draws top-left telemetry panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (260, 175), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "VisionCursor [Phase 3]", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Prominent Gaze Output
    dir_color = (0, 255, 0) if gaze_res.direction == "CENTER" else (0, 200, 255)
    cv2.putText(frame, f"Gaze: {gaze_res.direction}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.75, dir_color, 2)

    cv2.putText(frame, f"X (Norm): {gaze_res.smooth_x:.2f}", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(frame, f"Y (Norm): {gaze_res.smooth_y:.2f}", (20, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
    cv2.putText(frame, f"Confidence: {gaze_res.confidence}", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1)

    cv2.putText(frame, "Press 'q' to exit", (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

def main():
    print("=" * 60)
    print("VisionCursor: Initializing Phase 3 (Gaze Direction Detection)...")
    print("=" * 60)

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
    print("[OK] Face tracker and Gaze estimator ready.")

    try:
        while True:
            success, frame = cam.read_frame()
            if not success:
                break

            tracking_result = tracker.process_frame(frame)
            gaze_result = gaze_estimator.estimate_gaze(tracking_result)

            frame = draw_visual_overlay(frame, tracking_result)
            draw_hud(frame, cam.current_fps, gaze_result)
            draw_gaze_radar(frame, gaze_result)

            cv2.imshow("VisionCursor - Phase 3 Gaze Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        print("\n[SUCCESS] Phase 3 cleanly terminated.")

if __name__ == "__main__":
    main()