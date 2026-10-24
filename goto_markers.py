"""Drive the arm over each calibrated ArUco marker in turn -- a visual check
of a floor calibration: the open gripper should end up centred on every
marker. Saves a camera snapshot per marker (recordings/goto_<id>.jpg).

Run: python3 goto_markers.py                  # all markers in aruco_floor.json
     python3 goto_markers.py --ids 6 8 9 --dwell 3 --camera-index 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2

from pick_place import WORKSPACE, clamp, expand_workspace, load_floor_z, load_start_pose, load_workspace_calibration
from aruco_floor import ArucoFloor, FrameOrienter
from roarm_driver import Pose, RoArm


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration", default="aruco_floor.json")
    p.add_argument("--ids", type=int, nargs="*", default=None)
    p.add_argument("--dwell", type=float, default=3.0, help="seconds to pause over each marker")
    p.add_argument("--camera-index", type=int, default=0, help="-1 = no camera")
    p.add_argument("--floor-clearance-mm", type=float, default=15.0)
    p.add_argument("--hover-clearance-mm", type=float, default=120.0)
    p.add_argument("--start-pose", default="roarm_start_pose.json")
    p.add_argument("--floor-calibration", default="floor_calibration.json")
    p.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    p.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "auto"))
    args = p.parse_args()

    cal = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    centers = {int(k): (float(v["center"][0]), float(v["center"][1])) for k, v in cal["markers"].items()}
    ids = args.ids or sorted(centers)
    missing = [i for i in ids if i not in centers]
    if missing:
        print(f"캘리브레이션에 없는 마커라 건너뜀: {missing}")
    ids = [i for i in ids if i in centers]

    start_pose = load_start_pose(args.start_pose)
    expand_workspace(start_pose)
    ws = load_workspace_calibration("workspace_calibration.json")
    if ws is not None:
        WORKSPACE["x"], WORKSPACE["y"] = ws["x"], ws["y"]
    floor_z = load_floor_z(args.floor_calibration)
    if floor_z is None:
        raise SystemExit(f"{args.floor_calibration} not found")
    WORKSPACE["z"] = (min(WORKSPACE["z"][0], floor_z + 5.0), WORKSPACE["z"][1])
    grab_z, hover_z = floor_z + args.floor_clearance_mm, floor_z + args.hover_clearance_mm

    cap = cv2.VideoCapture(args.camera_index) if args.camera_index >= 0 else None
    unmirror = FrameOrienter(ArucoFloor(args.calibration))
    arm = RoArm(ip=args.ip, serial_port=args.serial_port or None)
    OPEN = RoArm.GRIPPER_OPEN

    def goto(x, y, z, speed=0.25):
        arm.move_to(Pose(clamp(x, "x"), clamp(y, "y"), clamp(z, "z"), OPEN), speed=speed)

    def wait(sec):
        end = time.time() + sec
        frame = None
        while time.time() < end:
            if cap is not None:
                ok, f = cap.read()
                if ok:
                    frame = f = unmirror.fix(f)
                    cv2.imshow("goto markers", f)
                    cv2.waitKey(1)
            else:
                time.sleep(0.05)
        return frame

    Path("recordings").mkdir(exist_ok=True)
    arm.torque(True)
    arm.open_gripper()
    goto(start_pose.x, start_pose.y, start_pose.z)
    wait(1.5)
    cx, cy = start_pose.x, start_pose.y
    try:
        for i in ids:
            x, y = centers[i]
            print(f"id {i}: -> ({x:.1f}, {y:.1f}) z={grab_z:.1f}")
            goto(cx, cy, hover_z)
            wait(0.8)
            goto(x, y, hover_z)
            wait(1.5)
            goto(x, y, grab_z, speed=0.2)
            frame = wait(args.dwell)
            fb = arm.feedback()
            if fb is not None:
                print(f"       feedback ({fb.x:.1f}, {fb.y:.1f}, {fb.z:.1f})")
            if frame is not None:
                cv2.imwrite(f"recordings/goto_{i}.jpg", frame)
            cx, cy = x, y
        goto(cx, cy, hover_z)
        wait(0.8)
        goto(start_pose.x, start_pose.y, start_pose.z)
        wait(1.5)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
