"""Live check of the ArUco floor tracking: move the camera around and watch
the robot-frame grid stay glued to the floor.

Shows every frame with the detected markers, the grid in robot coordinates
(50 mm) and the robot origin/axes if in view. Hover the mouse to read the
robot (x, y) under the cursor; click to print it. The bottom line tells how
many markers are used and the fit residual (mm).

Optional accuracy check with the arm: press 'c' and put the gripper tip on
any floor point, Enter -> the arm's real (x, y) is compared with the
camera's estimate at the clicked pixel (click the same spot first).

Run: python3 track_aruco_floor.py --camera-index 0
     python3 track_aruco_floor.py --record3d
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from aruco_floor import DEFAULT_PATH, ArucoFloor, FrameOrienter, save_view_rotate

WINDOW = "ArUco floor tracking"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--record3d", action="store_true")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--calibration", default=DEFAULT_PATH)
    p.add_argument("--grid-mm", type=float, default=50.0)
    p.add_argument("--view-rotate", type=int, choices=(0, 90, 180, 270), default=None, help="extra turn of the view on top of robot-aligned (robot +x up). Press r in the window to cycle and it is saved for all scripts.")
    p.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    p.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "auto"))
    args = p.parse_args()

    floor = ArucoFloor(args.calibration)
    unmirror = FrameOrienter(floor, view_rotate=args.view_rotate)  # true + robot-aligned (+ preferred view turn)
    if args.view_rotate is not None:
        save_view_rotate(args.calibration, args.view_rotate)
        print(f"view rotate {args.view_rotate} deg saved to {args.calibration}")
    print(f"{args.calibration}: {len(floor.world_corners)}개 마커 {sorted(floor.world_corners)}")

    if args.record3d:
        from record3d_camera import Record3DCamera

        cam = Record3DCamera(device_index=args.device_index)
        grab, close = (lambda: cam.get_frame().rgb), cam.close
    else:
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            raise SystemExit(f"cannot open camera index {args.camera_index}")

        def grab():
            ok, f = cap.read()
            if not ok:
                raise RuntimeError("camera read failed")
            return f

        close = cap.release

    arm = None
    mouse = {"pos": None, "click": None}

    def on_mouse(event, x, y, flags, param):
        mouse["pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["click"] = (x, y)
            if floor.valid:
                wx, wy = floor.pixel_to_world(x, y)
                print(f"pixel ({x},{y}) -> robot ({wx:.1f}, {wy:.1f}) mm")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    t_prev, fps = time.time(), 0.0
    try:
        while True:
            frame = unmirror.fix(grab())
            t0 = time.time()
            floor.update(frame)
            dt_ms = (time.time() - t0) * 1000
            floor.draw(frame, args.grid_mm)
            if mouse["pos"] and floor.valid:
                x, y = mouse["pos"]
                wx, wy = floor.pixel_to_world(x, y)
                cv2.putText(frame, f"({wx:.0f}, {wy:.0f})", (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f"({wx:.0f}, {wy:.0f})", (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            if mouse["click"]:
                cv2.drawMarker(frame, mouse["click"], (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(1e-3, now - t_prev)
            t_prev = now
            line = f"{fps:.0f} fps  detect+solve {dt_ms:.1f} ms   r: rotate view (saved)   c: arm check at clicked point   q: quit"
            cv2.putText(frame, line, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, line, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                unmirror.view_k = (unmirror.view_k + 1) % 4
                deg = unmirror.view_k * 90
                save_view_rotate(args.calibration, deg)
                print(f"view rotate {deg} deg (saved -- the main pipeline uses the same)")
                continue
            if key == ord("c"):
                if not mouse["click"] or not floor.valid:
                    print("먼저 화면에서 점을 클릭하세요")
                    continue
                if arm is None:
                    from roarm_driver import RoArm

                    arm = RoArm(ip=args.ip, serial_port=args.serial_port or None)
                est = floor.pixel_to_world(*mouse["click"])
                print(f"토크 off. 그리퍼 끝을 클릭한 점 위에 대고 Enter")
                arm.torque(False)
                try:
                    input("> ")
                    pose = arm.feedback()
                finally:
                    arm.torque(True)
                if pose is None:
                    print("팔 피드백 없음")
                    continue
                err = float(np.hypot(pose.x - est[0], pose.y - est[1]))
                print(f"camera ({est[0]:.1f}, {est[1]:.1f})  arm ({pose.x:.1f}, {pose.y:.1f})  error {err:.1f} mm")
    finally:
        close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
