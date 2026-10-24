"""Calibrate the safe operating area by hand: move the arm to its 4 extremes.

Replaces the guessed WORKSPACE margin in pick_place.py with real, measured
bounds. Torque is turned off so you can push the arm by hand to each extreme
(front/back/left/right of the reachable area on the table) and press Enter
to record it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from roarm_driver import RoArm

OUT_PATH = "workspace_calibration.json"

POINTS = [
    ("front", "팔을 앞으로(작업 범위 x 최대) 최대한 뻗은 안전한 위치로 이동"),
    ("back", "팔을 뒤로(작업 범위 x 최소) 최대한 당긴 안전한 위치로 이동"),
    ("left", "팔을 왼쪽(작업 범위 y 최대)으로 최대한 이동"),
    ("right", "팔을 오른쪽(작업 범위 y 최소)으로 최대한 이동"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Hand-calibrate the safe x/y operating area.")
    parser.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    parser.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "auto"))
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)

    print("토크를 끕니다. 팔을 손으로 자유롭게 움직일 수 있습니다.")
    arm.torque(False)
    time.sleep(0.3)

    recorded = {}
    try:
        for name, prompt in POINTS:
            input(f"{prompt}. 준비되면 Enter (Ctrl+C로 취소): ")
            pose = arm.feedback()
            if pose is None:
                print("피드백을 읽지 못했습니다. 연결을 확인하세요.")
                arm.torque(True)
                return 1
            print(f"  {name}: x={pose.x:.1f}, y={pose.y:.1f}, z={pose.z:.1f}")
            recorded[name] = {"x": pose.x, "y": pose.y, "z": pose.z}
    except KeyboardInterrupt:
        print("\n취소합니다.")
        arm.torque(True)
        return 1

    print("토크를 다시 켭니다.")
    arm.torque(True)
    time.sleep(0.3)

    xs = [p["x"] for p in recorded.values()]
    ys = [p["y"] for p in recorded.values()]
    data = {
        "points": recorded,
        "x": [min(xs), max(xs)],
        "y": [min(ys), max(ys)],
    }
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"작업 범위: x={data['x']}, y={data['y']}")
    print(f"저장됨: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
