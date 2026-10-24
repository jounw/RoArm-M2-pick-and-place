"""Teach the start pose by hand.

Torque off, physically move the arm to where you want each run to begin
(and end up when holding nothing), press Enter, and the pose (x, y, z, t)
is saved to roarm_start_pose.json. pick_place.py loads this file as the
home position for every cycle.

Run: python3 teach_start_pose.py --serial-port /dev/cu.usbserial-120
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from roarm_driver import RoArm

OUT_PATH = "roarm_start_pose.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Hand-teach the start pose and save it.")
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
        input("팔을 '시작 위치'에 맞춘 뒤 Enter를 누르세요 (Ctrl+C로 취소): ")
    except KeyboardInterrupt:
        print("\n취소합니다.")
        arm.torque(True)
        return 1

    pose = arm.feedback()
    if pose is None:
        print("피드백을 읽지 못했습니다. 연결을 확인하세요.")
        arm.torque(True)
        return 1

    print(f"측정된 시작 위치: x={pose.x:.1f}, y={pose.y:.1f}, z={pose.z:.1f}, t={pose.t:.3f}")

    print("토크를 다시 켭니다.")
    arm.torque(True)
    time.sleep(0.3)

    data = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "serial_port": args.serial_port,
        "base_pose": {"x": pose.x, "y": pose.y, "z": pose.z, "t": pose.t},
    }
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장됨: {args.out}")
    print("이제 pick_place.py가 매 사이클 이 위치를 시작/복귀 자세로 사용합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
