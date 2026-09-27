"""
main.py - Safe VisionCursor execution loop.
Includes PyAutoGUI failsafe and zero-freeze guarantees.
"""

import sys
import time
import cv2
import numpy as np
import pyautogui
import ctypes

from config.settings import CONFIG
from camera.camera_manager import CameraManager
from vision.face_tracker import FaceTracker
from control.cursor_controller import CursorController
from control.click_controller import ClickController
from control.scroll_controller import ScrollController, ScrollStatus
from gestures.blink_detector import BlinkDetector, BlinkEvent, LEFT_EYE_INDICES, RIGHT_EYE_INDICES
from gestures.gesture_recognizer import GestureRecognizer, GestureResult
from gestures.intent_detector import IntentDetector

# Enable PyAutoGUI safety failsafe: Moving physical mouse to any screen corner kills execution safely
pyautogui.FAILSAFE = True

def is_key_pressed(vk_code: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)

VK_E = 0x45
VK_C = 0x43
VK_P = 0x50
VK_Q = 0x51
VK_ESCAPE = 0x1B

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
             intent_status: str, is_stable: bool, stable_dur: float,
             last_click_alert_time: float, scroll_status: ScrollStatus):
    overlay = frame.copy()
    x1, y1, x2, y2 = 14, 14, 395, 335

    cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    c_len, c_color = 16, (0, 230, 255)
    cv2.line(frame, (x1, y1), (x1 + c_len, y1), c_color, 2)
    cv2.line(frame, (x1, y1), (x1, y1 + c_len), c_color, 2)
    cv2.line(frame, (x2, y1), (x2 - c_len, y1), c_color, 2)
    cv2.line(frame, (x2, y1), (x2 + c_len, y1), c_color, 2)
    cv2.line(frame, (x1, y2), (x1 + c_len, y2), c_color, 2)
    cv2.line(frame, (x1, y2), (x1 - c_len, y2), c_color, 2)
    cv2.line(frame, (x2, y2), (x2 - c_len, y2), c_color, 2)
    cv2.line(frame, (x2, y2), (x2 - c_len, y2), c_color, 2)

    cv2.putText(frame, "VISION CURSOR // SYS.ONLINE", (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1)
    fps_color = (0, 255, 120) if fps >= 25 else (0, 165, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (26, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.44, fps_color, 1)

    dir_color = (0, 255, 255) if direction == "CENTER" else (0, 180, 255)
    cv2.putText(frame, f"GAZE: {direction}", (26, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.60, dir_color, 2)
    cx, cy = cursor_pos
    cv2.putText(frame, f"CURSOR: ({cx}, {cy}) px", (26, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)

    stab_txt = f"LOCKED ({stable_dur:.2f}s)" if is_stable else f"FIXATING ({stable_dur:.2f}s)"
    stab_color = (0, 255, 120) if is_stable else (0, 180, 255)
    cv2.putText(frame, f"TARGET: {stab_txt}", (26, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.42, stab_color, 1)

    ear_str = f"L:{blink_evt.left_ear:.2f} | R:{blink_evt.right_ear:.2f} | AVG:{blink_evt.average_ear:.2f}"
    cv2.putText(frame, f"EAR : {ear_str}", (26, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    if gesture_res.state == "WAITING FOR 2ND BLINK":
        state_txt = f"WAITING FOR 2ND BLINK ({gesture_res.time_remaining:.2f}s)"
        s_color = (0, 215, 255)
    else:
        state_txt = "WAITING FOR 1ST BLINK"
        s_color = (160, 160, 160)
    cv2.putText(frame, f"GESTURE: {state_txt}", (26, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.42, s_color, 1)

    if scroll_status.state == "ACTIVE":
        scr_txt = f"{scroll_status.zone} (SCROLLING)"
        scr_color = (0, 255, 120)
    elif scroll_status.state == "WAITING":
        scr_txt = f"{scroll_status.zone} (HOLD {scroll_status.hold_duration:.2f}s)"
        scr_color = (0, 215, 255)
    else:
        scr_txt = "NONE (IDLE)"
        scr_color = (160, 160, 160)
    cv2.putText(frame, f"SCROLL: {scr_txt}", (26, 194), cv2.FONT_HERSHEY_SIMPLEX, 0.42, scr_color, 1)

    now = time.monotonic()
    if (now - last_click_alert_time) < 0.50:
        cv2.putText(frame, ">> LEFT CLICK EXECUTED! <<", (26, 222),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)
    else:
        status_color = (0, 255, 120) if is_stable else (160, 180, 200)
        cv2.putText(frame, f"INTENT: {intent_status}", (26, 222),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_color, 1)

    status_text = "ENGAGED" if cursor_active else "STANDBY"
    status_color = (0, 255, 120) if cursor_active else (0, 140, 255)
    cv2.circle(frame, (32, 252), 4, status_color, -1)
    cv2.putText(frame, f"MOUSE: {status_text}", (44, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_color, 1)

    cv2.putText(frame, "[E] Toggle | [C] Recenter | [P] Emergency Pause", (26, 288),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 150, 160), 1)
    cv2.putText(frame, "[Q] Exit VisionCursor", (26, 310),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 150, 160), 1)

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
    print("VisionCursor: Clean & Safe Engine")
    print("=" * 60)

    screen_w, screen_h = pyautogui.size()
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
    cursor_ctrl = CursorController(screen_w, screen_h)
    blink_detector = BlinkDetector()
    gesture_recognizer = GestureRecognizer()
    intent_detector = IntentDetector()
    click_ctrl = ClickController()
    scroll_ctrl = ScrollController()

    last_click_alert_time = 0.0
    last_key_handled_time = 0.0

    print("\n[SAFETY CONTROLS]:")
    print("  - Cursor starts in STANDBY mode (Your physical mouse works 100% normally).")
    print("  - Press 'E' to engage Vision Cursor. Press 'P' for instant Emergency Stop.")
    print("  - Fail-safe: Slam physical mouse into any screen corner to immediately exit.")
    print("  - Press 'Q' or ESC in camera window to safely close.\n")

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

            # 1. Cursor Navigation (Only runs if manually ENGAGED via 'E')
            if tracking_res.face_detected and tracking_res.nose_point:
                norm_x, norm_y = tracking_res.nose_point
                if cursor_ctrl.is_enabled:
                    if (now - last_click_alert_time) > 0.15:
                        final_x, final_y = cursor_ctrl.update_position(tracking_res.nose_point)

                if cursor_ctrl.center_x is not None:
                    dx = norm_x - cursor_ctrl.center_x
                    dy = norm_y - cursor_ctrl.center_y
                    direction = get_direction_label(dx, dy)

            # 2. Canonical Eyelid EAR Blink Detection
            landmarks = tracking_res.raw_pixel_landmarks
            blink_evt = blink_detector.update_with_landmarks(landmarks, timestamp=now)

            # 3. Double-Blink Recognition
            gesture_res = gesture_recognizer.update(blink_evt, current_time=now)
            double_blink_confirmed = (gesture_res.action == "DOUBLE_CONFIRMED")

            # 4. Gaze Stability & Click Intent Arbitration
            intent_res = intent_detector.evaluate_intent(
                cursor_px=(final_x, final_y),
                double_blink_confirmed=double_blink_confirmed,
                cursor_active=cursor_ctrl.is_enabled,
                calibration_present=True,
                cooldown_finished=click_ctrl.can_click,
                current_time=now
            )

            # 5. Execute Safe Left Click
            if intent_res.trigger_click:
                click_executed = click_ctrl.left_click()
                if click_executed:
                    last_click_alert_time = now
                    print(f"[ACTION] LEFT CLICK EXECUTED at ({final_x}, {final_y}) px!")

            # 6. Smooth Tilt Scroll
            scroll_status = scroll_ctrl.update(
                norm_y=norm_y,
                center_y=cursor_ctrl.center_y,
                enabled=cursor_ctrl.is_enabled,
                current_time=now
            )
            # Render Visuals
            frame = draw_sci_fi_visuals(frame, tracking_res)
            draw_hud(frame, cam.current_fps, cursor_ctrl.is_enabled, (final_x, final_y),
                     direction, blink_evt, gesture_res, intent_res.status_label,
                     intent_res.gaze_stable, intent_res.stable_duration,
                     last_click_alert_time, scroll_status)

            cv2.imshow("VisionCursor - Sys.Gaze Recognizer", frame)

            # Global Hotkey Handling
            cv_key = cv2.waitKey(1) & 0xFF
            if (now - last_key_handled_time) > 0.30:
                if cv_key in [27, ord('q')] or is_key_pressed(VK_Q) or is_key_pressed(VK_ESCAPE):
                    break
                elif cv_key in [ord('e')] or is_key_pressed(VK_E):
                    last_key_handled_time = now
                    if tracking_res.nose_point:
                        cursor_ctrl.toggle(tracking_res.nose_point)
                        intent_detector.reset_stability()
                        scroll_ctrl.reset()
                        state_str = "ENGAGED" if cursor_ctrl.is_enabled else "STANDBY"
                        print(f"[HOTKEY] Cursor state: {state_str}")
                elif cv_key in [ord('p')] or is_key_pressed(VK_P):
                    last_key_handled_time = now
                    cursor_ctrl.is_enabled = False
                    intent_detector.reset_stability()
                    scroll_ctrl.reset()
                    print("[HOTKEY] Emergency Pause Activated (MOUSE: STANDBY).")
                elif cv_key in [ord('c')] or is_key_pressed(VK_C):
                    last_key_handled_time = now
                    if tracking_res.nose_point:
                        cursor_ctrl.recenter(tracking_res.nose_point)
                        intent_detector.reset_stability()
                        scroll_ctrl.reset()
                        print("[HOTKEY] Center re-calibrated.")

    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        print("\n[SUCCESS] VisionCursor cleanly terminated.")

if __name__ == "__main__":
    main()