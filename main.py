"""
main.py - Phase 5 (Cyberpunk Developer HUD & Facial Topology Visualizer)
Fully compatible with Python 3.13 & MediaPipe Tasks API.
"""

import sys
import cv2
import numpy as np
import pyautogui

from config.settings import CONFIG
from camera.camera_manager import CameraManager
from vision.face_tracker import FaceTracker
from control.cursor_controller import CursorController

# Predefined key facial contour connection indices (No legacy mp.solutions dependency)
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]
LIPS_OUTER_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 61]
LEFT_EYEBROW_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW_IDX = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
NOSE_BRIDGE_IDX = [168, 6, 197, 195, 5, 4, 1, 2]

# Canonical Delaunay triangulated mesh samples for cyber-lining effect
CHEEK_TRIANGLES = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (10, 109), (109, 67), (67, 103), (103, 54), (54, 21), (21, 162),
    (168, 197), (197, 5), (5, 4), (4, 1),
    (127, 234), (234, 93), (93, 132), (132, 58), (58, 172), (172, 136),
    (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365),
    (1, 2), (2, 164), (164, 0), (0, 17), (17, 18), (18, 200), (200, 199), (199, 175), (175, 152),
    (117, 118), (118, 119), (119, 120), (120, 121), (121, 128),
    (346, 347), (347, 348), (348, 349), (349, 350), (350, 357)
]

def get_direction_label(dx: float, dy: float, deadzone: float = 0.015) -> str:
    """Computes cardinal direction for developer telemetry."""
    if abs(dx) < deadzone and abs(dy) < deadzone:
        return "CENTER"
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    else:
        return "DOWN" if dy > 0 else "UP"

def draw_hud(frame, fps: float, cursor_active: bool, cursor_pos: tuple, norm_pt: tuple, direction: str):
    """Draws sleek sci-fi developer telemetry HUD."""
    overlay = frame.copy()
    x1, y1, x2, y2 = 14, 14, 340, 220

    # Glassmorphism dark background
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (10, 12, 16), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Cyberpunk corner brackets
    c_len = 16
    c_color = (0, 230, 255)
    cv2.line(frame, (x1, y1), (x1 + c_len, y1), c_color, 2)
    cv2.line(frame, (x1, y1), (x1, y1 + c_len), c_color, 2)
    cv2.line(frame, (x2, y1), (x2 - c_len, y1), c_color, 2)
    cv2.line(frame, (x2, y1), (x2, y1 + c_len), c_color, 2)
    cv2.line(frame, (x1, y2), (x1 + c_len, y2), c_color, 2)
    cv2.line(frame, (x1, y2), (x1, y2 - c_len), c_color, 2)
    cv2.line(frame, (x2, y2), (x2 - c_len, y2), c_color, 2)
    cv2.line(frame, (x2, y2), (x2, y2 - c_len), c_color, 2)

    # Title & FPS
    cv2.putText(frame, "VISION CURSOR // SYS.ONLINE", (26, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1)
    fps_color = (0, 255, 120) if fps >= 25 else (0, 165, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (26, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.46, fps_color, 1)

    # Gaze Direction
    dir_color = (0, 255, 255) if direction == "CENTER" else (0, 180, 255)
    cv2.putText(frame, f"GAZE: {direction}", (26, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, dir_color, 2)

    # Coordinates
    nx, ny = norm_pt
    cv2.putText(frame, f"NORM RATIO : [{nx:.3f}, {ny:.3f}]", (26, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 200), 1)
    cx, cy = cursor_pos
    cv2.putText(frame, f"CURSOR POS : ({cx}, {cy}) px", (26, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)

    # Controller Status
    status_text = "ENGAGED [TRACKING]" if cursor_active else "STANDBY [PAUSED]"
    status_color = (0, 255, 120) if cursor_active else (0, 140, 255)
    cv2.circle(frame, (32, 172), 5, status_color, -1)
    cv2.putText(frame, status_text, (44, 176), cv2.FONT_HERSHEY_SIMPLEX, 0.48, status_color, 1)

    # Keybind Info
    cv2.putText(frame, "[E] Toggle | [C] Recenter | [Q] Exit", (26, 204),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 150, 160), 1)

def draw_sci_fi_visuals(frame, tracking_res):
    """Renders facial wireframe mesh, red iris points, and reticle."""
    if not tracking_res.face_detected or not tracking_res.raw_pixel_landmarks:
        return frame

    pts = tracking_res.raw_pixel_landmarks
    num_pts = len(pts)

    # 1. Subtle Face Structure Lining
    mesh_overlay = frame.copy()
    
    # Cheek & Structure Lines
    for (i1, i2) in CHEEK_TRIANGLES:
        if i1 < num_pts and i2 < num_pts:
            cv2.line(mesh_overlay, pts[i1], pts[i2], (140, 110, 30), 1)

    # Face Oval
    for i in range(len(FACE_OVAL_IDX) - 1):
        i1, i2 = FACE_OVAL_IDX[i], FACE_OVAL_IDX[i + 1]
        if i1 < num_pts and i2 < num_pts:
            cv2.line(mesh_overlay, pts[i1], pts[i2], (180, 160, 20), 1)

    # Lips Outer
    for i in range(len(LIPS_OUTER_IDX) - 1):
        i1, i2 = LIPS_OUTER_IDX[i], LIPS_OUTER_IDX[i + 1]
        if i1 < num_pts and i2 < num_pts:
            cv2.line(mesh_overlay, pts[i1], pts[i2], (160, 130, 20), 1)

    # Eyebrows
    for i in range(len(LEFT_EYEBROW_IDX) - 1):
        i1, i2 = LEFT_EYEBROW_IDX[i], LEFT_EYEBROW_IDX[i + 1]
        if i1 < num_pts and i2 < num_pts:
            cv2.line(mesh_overlay, pts[i1], pts[i2], (160, 140, 20), 1)
    for i in range(len(RIGHT_EYEBROW_IDX) - 1):
        i1, i2 = RIGHT_EYEBROW_IDX[i], RIGHT_EYEBROW_IDX[i + 1]
        if i1 < num_pts and i2 < num_pts:
            cv2.line(mesh_overlay, pts[i1], pts[i2], (160, 140, 20), 1)

    # Nose Bridge Line
    for i in range(len(NOSE_BRIDGE_IDX) - 1):
        i1, i2 = NOSE_BRIDGE_IDX[i], NOSE_BRIDGE_IDX[i + 1]
        if i1 < num_pts and i2 < num_pts:
            cv2.line(mesh_overlay, pts[i1], pts[i2], (200, 180, 40), 1)

    # Blend wireframe with camera frame
    cv2.addWeighted(mesh_overlay, 0.45, frame, 0.55, 0, frame)

    # 2. Eye Contours (Cyan)
    if tracking_res.left_eye:
        cv2.polylines(frame, [tracking_res.left_eye.contour_points], True, (255, 255, 0), 1)
    if tracking_res.right_eye:
        cv2.polylines(frame, [tracking_res.right_eye.contour_points], True, (255, 255, 0), 1)

    # 3. Iris Tracking (Green Ring + Glowing Red Center Dot 🔴)
    for iris in [tracking_res.left_iris, tracking_res.right_iris]:
        if iris:
            cv2.polylines(frame, [iris.points], isClosed=True, color=(0, 255, 120), thickness=1)
            cv2.circle(frame, iris.center, 5, (0, 0, 180), 1)
            cv2.circle(frame, iris.center, 3, (0, 0, 255), -1)

    # 4. Nose Crosshair Target Reticle
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
    print("VisionCursor: Developer Topology Engine Loaded")
    print("=" * 60)

    screen_w, screen_h = pyautogui.size()
    print(f"[DISPLAY] Resolution: {screen_w} x {screen_h}")

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

    try:
        while True:
            success, frame = cam.read_frame()
            if not success:
                break

            tracking_res = tracker.process_frame(frame)

            final_x, final_y = pyautogui.position()
            norm_x, norm_y = 0.5, 0.5
            direction = "CENTER"

            if tracking_res.face_detected and tracking_res.nose_point:
                norm_x, norm_y = tracking_res.nose_point
                final_x, final_y = cursor_ctrl.update_position(tracking_res.nose_point)

                if cursor_ctrl.center_x is not None:
                    dx = norm_x - cursor_ctrl.center_x
                    dy = norm_y - cursor_ctrl.center_y
                    direction = get_direction_label(dx, dy)

            # 1. Render Face wireframe, red iris dots & crosshairs
            frame = draw_sci_fi_visuals(frame, tracking_res)

            # 2. Render Cyberpunk Developer HUD
            draw_hud(frame, cam.current_fps, cursor_ctrl.is_enabled, (final_x, final_y), (norm_x, norm_y), direction)

            cv2.imshow("VisionCursor - Phase 5", frame)

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