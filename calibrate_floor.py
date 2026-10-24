"""Calibrate the pick/place surface height by hand.

Torque off, physically move the gripper down to the table/floor surface,
press Enter, and the real z (and x, y) get saved for pick_place.py to use
instead of a guessed relative offset.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from roarm_driver import RoArm

OUT_PATH = "floor_calibration.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Hand-calibrate the pick/place surface height.")
    parser.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    parser.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "auto"))
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)

    print("토크를 끕니다. 팔을 손으로 자유롭게 움직일 수 있습니다.")
    arm.torque(False)
    time.sleep(0.3)

    try:
        input("그리퍼 끝을 바닥/집을 물체 표면 높이에 맞춘 뒤 Enter를 누르세요 (Ctrl+C로 취소): ")
    except KeyboardInterrupt:
        print("\n취소합니다.")
        arm.torque(True)
        return 1

    pose = arm.feedback()
    if pose is None:
        print("피드백을 읽지 못했습니다. 연결을 확인하세요.")
        arm.torque(True)
        return 1

    print(f"측정된 위치: x={pose.x:.1f}, y={pose.y:.1f}, z={pose.z:.1f}")

    print("토크를 다시 켭니다.")
    arm.torque(True)
    time.sleep(0.3)

    data = {"x": pose.x, "y": pose.y, "z": pose.z}
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장됨: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
