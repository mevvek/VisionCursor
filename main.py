"""
main.py - Phase 2 Verification
Integrates camera, face, eye, and iris tracking with a live visual debug overlay.
No cursor movement or PyAutoGUI actions are executed.
"""

import cv2
import sys
from config.settings import CONFIG
from camera.camera_manager import CameraManager
from vision.face_tracker import FaceTracker

def draw_visual_overlay(frame, tracking_result):
    """Draws face oval, eye contours, iris markers, and diagnostic HUD on the frame."""
    if not tracking_result.face_detected:
        return frame

    # 1. Subtle Face Oval (Dark grey/blue)
    if tracking_result.face_oval_points is not None:
        cv2.polylines(frame, [tracking_result.face_oval_points], isClosed=True, color=(80, 80, 80), thickness=1)

    # 2. Eye Contours (Cyan)
    if tracking_result.left_eye:
        cv2.polylines(frame, [tracking_result.left_eye.contour_points], isClosed=True, color=(255, 255, 0), thickness=1)
    if tracking_result.right_eye:
        cv2.polylines(frame, [tracking_result.right_eye.contour_points], isClosed=True, color=(255, 255, 0), thickness=1)

    # 3. Iris Contours & Center Markers (Red center + Green border)
    for iris in [tracking_result.left_iris, tracking_result.right_iris]:
        if iris:
            cv2.polylines(frame, [iris.points], isClosed=True, color=(0, 255, 120), thickness=1)
            cv2.circle(frame, iris.center, 3, (0, 0, 255), -1)

    return frame

def draw_hud(frame, fps: float, tracking_result):
    """Draws real-time diagnostic telemetry in the top-left corner."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 160), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "VisionCursor [Phase 2]", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    face_status = "Detected" if tracking_result.face_detected else "Not Detected"
    face_color = (0, 255, 0) if tracking_result.face_detected else (0, 0, 255)
    cv2.putText(frame, f"Face: {face_status}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)

    left_eye_status = "Detected" if tracking_result.left_eye else "None"
    right_eye_status = "Detected" if tracking_result.right_eye else "None"
    cv2.putText(frame, f"Left Eye: {left_eye_status}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Right Eye: {right_eye_status}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    iris_status = "Detected" if (tracking_result.left_iris and tracking_result.right_iris) else "None"
    cv2.putText(frame, f"Iris: {iris_status}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)

    if tracking_result.face_count > 1:
        cv2.putText(frame, f"Faces: {tracking_result.face_count} (Primary Selected)", (20, 153),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    cv2.putText(frame, "Press 'q' to exit", (20, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

def main():
    print("=" * 60)
    print("VisionCursor: Initializing Phase 2 (Face, Eye & Iris Tracking)...")
    print("=" * 60)

    cam = CameraManager(
        source=CONFIG.camera.device_id,
        width=CONFIG.camera.width,
        height=CONFIG.camera.height,
        fps=CONFIG.camera.fps
    )

    if not cam.start():
        print("[ERROR] Camera initialization failed.")
        sys.exit(1)

    tracker = FaceTracker()
    print("[OK] Face, Eye, and Iris tracking initialized.")

    try:
        while True:
            success, frame = cam.read_frame()
            if not success:
                print("[WARNING] Frame capture dropped.")
                break

            result = tracker.process_frame(frame)
            frame = draw_visual_overlay(frame, result)
            draw_hud(frame, cam.current_fps, result)

            cv2.imshow("VisionCursor - Phase 2 Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        print("\n[SUCCESS] Phase 2 terminated cleanly.")

if __name__ == "__main__":
    main()