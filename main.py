"""
main.py - Phase 1 Verification
Tests CameraManager, OpenCV window display, and real-time FPS performance.
"""

import cv2
import sys
from config.settings import CONFIG
from camera.camera_manager import CameraManager

def main():
    print("=" * 60)
    print("VisionCursor: Initializing Phase 1 Verification...")
    print(f"Target Camera Source: {CONFIG.camera.device_id}")
    print(f"Target Resolution   : {CONFIG.camera.width}x{CONFIG.camera.height} @ {CONFIG.camera.fps} FPS")
    print("=" * 60)

    cam = CameraManager(
        source=CONFIG.camera.device_id,
        width=CONFIG.camera.width,
        height=CONFIG.camera.height,
        fps=CONFIG.camera.fps
    )

    if not cam.start():
        print("\n[ERROR] Could not open camera source. Troubleshooting steps:")
        print(" 1. Check if another application (Zoom, Teams, Browser) is using the webcam.")
        print(" 2. If using an external USB camera, change device_id to 1 in config/settings.py.")
        print(" 3. Check Windows Camera Privacy Settings and ensure desktop apps have access.")
        sys.exit(1)

    print("\n[OK] Camera opened successfully.")
    print("Press 'q' in the camera window to complete Phase 1 test.")

    try:
        while True:
            success, frame = cam.read_frame()
            if not success:
                print("[WARNING] Frame drop or camera disconnected.")
                break

            # Overlay FPS counter & Phase status
            fps_text = f"FPS: {cam.current_fps:.1f}"
            cv2.putText(frame, fps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "VisionCursor - Phase 1: Camera Ready", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to exit", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("VisionCursor Diagnostic (Phase 1)", frame)

            # Exit cleanly when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("\n[SUCCESS] Phase 1 test completed cleanly.")

if __name__ == "__main__":
    main()