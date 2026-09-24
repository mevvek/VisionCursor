"""
main.py - Dual Intentional Gesture Telemetry Engine
Recognizes both Single Intentional Blink and Double Intentional Blink.
Stateless and Auto-Resetting. Strictly NO mouse clicking executed.
"""

import sys
import time
import cv2
import numpy as np
import pyautogui

from config.settings import CONFIG
from camera.camera_manager import CameraManager
from vision.face_tracker import FaceTracker
from control.cursor_controller import CursorController
from gestures.blink_detector import BlinkDetector, BlinkEvent, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from gestures.gesture_recognizer import GestureRecognizer, GestureResult

FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]
CHEEK_TRIANGLES = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (10, 109), (109, 67), (67, 103), (103, 54), (54, 21), (21, 162),
    (168, 197), (197, 5), (5, 4), (4, 1),
    (127, 234), (234, 93), (93, 132), (132, 58), (58, 172), (172, 136),
    (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365),
    (1, 2), (2, 164), (164, 0), (0, 17), (17, 18), (18, 200), (200, 199), (199, 175), (175, 152)
]

def get_direction_label(dx: float, dy: float, deadzone: float = 0.015) -> str:
    if abs(dx) < deadzone and abs(dy) < deadzone:
        return "CENTER"
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    else:
        return "DOWN" if dy > 0 else "UP"

def draw_hud(frame, fps: float, cursor_active: bool, cursor_pos: tuple,
             direction: str, blink_evt: BlinkEvent, gesture_res: GestureResult,
             last_confirmed_action: str, last_action_time: float):
    overlay = frame.copy()
    x1, y1, x2, y2 = 14, 14, 385, 295

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    c_len, c_color = 16, (0, 230, 255)
    cv2.line(frame, (x1, y1), (x1 + c_len, y1), c_color, 2)
    cv2.line(frame, (x1, y1), (x1, y1 + c_len), c_color, 2)
    cv2.line(frame, (x2, y1), (x2 - c_len, y1), c_color, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + c_len), c_color, 2)
    cv2.line(frame, (x1, y2), (x1 + c_len, y2), c_color, 2)
    cv2.line(frame, (x1, y2), (x1 - c_len, y2), c_color, 2)
    cv2.line(frame, (x2, y2), (x2 - c_len, y2), c_color, 2)
    cv2.line(frame, (x2, y2), (x2, y2 - c_len), c_color, 2)

    cv2.putText(frame, "VISION CURSOR // SYS.ONLINE", (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1)
    fps_color = (0, 255, 120) if fps >= 25 else (0, 165, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (26, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, fps_color, 1)

    dir_color = (0, 255, 255) if direction == "CENTER" else (0, 180, 255)
    cv2.putText(frame, f"GAZE: {direction}", (26, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.60, dir_color, 2)
    cx, cy = cursor_pos
    cv2.putText(frame, f"CURSOR: ({cx}, {cy}) px", (26, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)

    # EAR Telemetry
    ear_str = f"L:{blink_evt.left_ear:.2f} | R:{blink_evt.right_ear:.2f} | AVG:{blink_evt.average_ear:.2f}"
    cv2.putText(frame, f"EAR : {ear_str}", (26, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    state_color = (0, 255, 120) if blink_evt.eye_state == "OPEN" else (0, 0, 255)
    cv2.putText(frame, f"EYE STATE: {blink_evt.eye_state}", (26, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.44, state_color, 1)

    # Live Gesture State
    if gesture_res.state == "WAITING FOR 2ND BLINK":
        state_txt = f"WAITING FOR 2ND BLINK ({gesture_res.time_remaining:.2f}s)"
        s_color = (0, 215, 255)
    else:
        state_txt = "WAITING FOR 1ST BLINK (READY)"
        s_color = (160, 160, 160)
    cv2.putText(frame, f"GESTURE: {state_txt}", (26, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.42, s_color, 1)

    # Confirmation Alert (Flashes for 0.4s then auto-resets)
    now = time.monotonic()
    if (now - last_action_time) < 0.45:
        if last_confirmed_action == "SINGLE_CONFIRMED":
            cv2.putText(frame, ">> SINGLE BLINK CONFIRMED <<", (26, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 2)
        elif last_confirmed_action == "DOUBLE_CONFIRMED":
            cv2.putText(frame, ">> DOUBLE BLINK CONFIRMED <<", (26, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "ACTION: ZERO CLICK (Detected Only)", (26, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 140, 150), 1)

    status_text = "ENGAGED" if cursor_active else "STANDBY"
    status_color = (0, 255, 120) if cursor_active else (0, 140, 255)
    cv2.circle(frame, (32, 240), 4, status_color, -1)
    cv2.putText(frame, f"MOUSE: {status_text}", (44, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_color, 1)

    cv2.putText(frame, "[E] Toggle | [C] Recenter | [Q] Exit", (26, 276),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 150, 160), 1)

def draw_sci_fi_visuals(frame, tracking_res):
    if not tracking_res.face_detected or not tracking_res.raw_pixel_landmarks:
        return frame

    pts = tracking_res.raw_pixel_landmarks
    num_pts = len(pts)

    mesh_overlay = frame.copy()
    for (i1, i2) in CHEEK_TRIANGLES:
        if i1 < num_pts and i2 < num_pts:
            cv2.line(mesh_overlay, pts[i1], pts[i2], (140, 110, 30), 1)

    for i in range(len(FACE_OVAL_IDX) - 1):
        i1, i2 = FACE_OVAL_IDX[i], FACE_OVAL_IDX[i + 1]
        if i1 < num_pts and i2 < num_pts:
            cv2.line(mesh_overlay, pts[i1], pts[i2], (180, 160, 20), 1)

    cv2.addWeighted(mesh_overlay, 0.45, frame, 0.55, 0, frame)

    left_eye_pts = np.array([pts[i] for i in LEFT_EYE_INDICES], dtype=np.int32)
    right_eye_pts = np.array([pts[i] for i in RIGHT_EYE_INDICES], dtype=np.int32)
    cv2.polylines(frame, [left_eye_pts], True, (255, 255, 0), 1)
    cv2.polylines(frame, [right_eye_pts], True, (255, 255, 0), 1)

    for iris in [tracking_res.left_iris, tracking_res.right_iris]:
        if iris:
            cv2.polylines(frame, [iris.points], isClosed=True, color=(0, 255, 120), thickness=1)
            cv2.circle(frame, iris.center, 5, (0, 0, 180), 1)
            cv2.circle(frame, iris.center, 3, (0, 0, 255), -1)

    if tracking_res.nose_px:
        nx, ny = tracking_res.nose_px
        cv2.circle(frame, (nx, ny), 8, (0, 210, 255), 1)
        cv2.circle(frame, (nx, ny), 2, (0, 255, 255), -1)
        cv2.line(frame, (nx - 14, ny), (nx - 4, ny), (0, 210, 255), 1)
        cv2.line(frame, (nx + 4, ny), (nx + 14, ny), (0, 210, 255), 1)
        cv2.line(frame, (nx, ny - 14), (nx, ny - 4), (0, 210, 255), 1)
        cv2.line(frame, (nx, ny + 4), (nx, ny + 14), (0, 210, 255), 1)

    return frame

def main():
    print("=" * 60)
    print("VisionCursor: Dual Gesture Disambiguation Loaded")
    print("=" * 60)

    screen_w, screen_h = pyautogui.size()
    cam = CameraManager(
        source=CONFIG.camera.device_id,
        width=CONFIG.camera.width,
        height=CONFIG.camera.height,
        fps=CONFIG.camera.fps
    )

    if not cam.start():
        print("[ERROR] Camera failed to initialize.")
        sys.exit(1)

    tracker = FaceTracker()
    cursor_ctrl = CursorController(screen_w, screen_h)
    blink_detector = BlinkDetector()
    gesture_recognizer = GestureRecognizer()

    last_confirmed_action = "NONE"
    last_action_time = 0.0

    print("\n[READY FOR TESTING]:")
    print("  - Single Deliberate Blink -> Waits 0.6s -> SINGLE BLINK CONFIRMED -> Resets.")
    print("  - Double Deliberate Blink -> Detects 2nd blink -> DOUBLE BLINK CONFIRMED -> Resets.")
    print("  - Mouse clicks remain disabled.\n")

    try:
        while True:
            success, frame = cam.read_frame()
            if not success:
                break

            now = time.monotonic()
            tracking_res = tracker.process_frame(frame)

            final_x, final_y = pyautogui.position()
            norm_x, norm_y = 0.5, 0.5
            direction = "CENTER"

            # 1. Cursor Navigation
            if tracking_res.face_detected and tracking_res.nose_point:
                norm_x, norm_y = tracking_res.nose_point
                final_x, final_y = cursor_ctrl.update_position(tracking_res.nose_point)

                if cursor_ctrl.center_x is not None:
                    dx = norm_x - cursor_ctrl.center_x
                    dy = norm_y - cursor_ctrl.center_y
                    direction = get_direction_label(dx, dy)

            # 2. Blink Detection
            landmarks = tracking_res.raw_pixel_landmarks
            blink_evt = blink_detector.update_with_landmarks(landmarks, timestamp=now)

            # 3. Dual Gesture Recognizer
            gesture_res = gesture_recognizer.update(blink_evt, current_time=now)

            if gesture_res.action != "NONE":
                last_confirmed_action = gesture_res.action
                last_action_time = now
                if gesture_res.action == "SINGLE_CONFIRMED":
                    print("[GESTURE] SINGLE BLINK CONFIRMED (Timeout passed - Ready for Next)")
                elif gesture_res.action == "DOUBLE_CONFIRMED":
                    print(f"[GESTURE] DOUBLE BLINK CONFIRMED ({gesture_res.interval:.2f}s interval - Ready for Next)")

            # Render
            frame = draw_sci_fi_visuals(frame, tracking_res)
            draw_hud(frame, cam.current_fps, cursor_ctrl.is_enabled, (final_x, final_y),
                     direction, blink_evt, gesture_res, last_confirmed_action, last_action_time)

            cv2.imshow("VisionCursor - Sys.Gaze Recognizer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break
            elif key in [ord('e'), ord('p')]:
                if tracking_res.nose_point:
                    cursor_ctrl.toggle(tracking_res.nose_point)
            elif key == ord('c'):
                if tracking_res.nose_point:
                    cursor_ctrl.recenter(tracking_res.nose_point)
                    print("[CURSOR] Center re-calibrated.")

    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        print("\n[SUCCESS] Terminated safely.")

if __name__ == "__main__":
    main()