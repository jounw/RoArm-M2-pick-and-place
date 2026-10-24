"""Teach-by-demonstration motions: record the arm while you move it by hand,
save under a number, replay later.

  python3 motions.py record 1          # torque off, move the arm (and gripper) by hand, Enter stops
  python3 motions.py play 1            # replay motion 1 (--speed 1.5 = faster)
  python3 motions.py list
  python3 motions.py delete 1

Files: motions/motion_<n>.json -- {"hz": 20, "samples": [[t_sec, x, y, z, grip], ...]}

Recording samples arm.feedback() at ~20 Hz with torque OFF, so the trajectory
and the gripper opening you make by hand are both captured. Playback turns
torque on, moves slowly to the first sample, then streams the samples with
T:1041 direct control at the recorded timing (gripper included).

In the command loop (pick_place_record3d_to_eih.py --place-mode command) just
type the number, or say "3번 동작" / "3번 해봐", and that motion plays.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from pathlib import Path

from roarm_driver import Pose, RoArm

MOTION_DIR = Path("motions")
RECORD_HZ = 20.0
_MOTION_RE = re.compile(r"^\s*(?:(\d{1,3})\s*번?\s*(?:동작|모션|행동)?\s*(?:해|해봐|해줘|실행|재생|틀어|플레이)?|(?:동작|모션)\s*(\d{1,3})\s*(?:번)?\s*(?:해|해봐|해줘|실행|재생)?)\s*[.!]?\s*$")


def motion_path(n: int) -> Path:
    return MOTION_DIR / f"motion_{n}.json"


def parse_motion_command(text: str) -> int | None:
    """'3', '3번', '3번 동작', '동작 3 해줘' -> 3; anything else -> None."""
    m = _MOTION_RE.match(text)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def list_motions() -> dict[int, dict]:
    out = {}
    if MOTION_DIR.exists():
        for p in sorted(MOTION_DIR.glob("motion_*.json")):
            try:
                n = int(p.stem.split("_")[1])
                data = json.loads(p.read_text(encoding="utf-8"))
                out[n] = {"path": str(p), "seconds": data["samples"][-1][0] if data["samples"] else 0.0,
                          "samples": len(data["samples"]), "recorded_at": data.get("recorded_at", "?"), "name": data.get("name", "")}
            except (ValueError, KeyError, IndexError):
                continue
    return out


def record_motion(arm: RoArm, n: int, name: str = "", stop_event: threading.Event | None = None,
                  max_seconds: float = 120.0, on_sample=None) -> Path:
    """Torque off, sample feedback at RECORD_HZ until Enter (or stop_event), save."""
    MOTION_DIR.mkdir(exist_ok=True)
    arm.torque(False)
    samples: list[list[float]] = []
    if stop_event is None:
        stop_event = threading.Event()
        threading.Thread(target=lambda: (input(), stop_event.set()), daemon=True).start()
        print(f"[motion {n}] 토크를 껐습니다. 팔(과 집게)을 손으로 움직이세요. 녹화 중... Enter를 누르면 끝납니다.")
    t0 = time.time()
    period = 1.0 / RECORD_HZ
    next_t = t0
    try:
        while not stop_event.is_set() and time.time() - t0 < max_seconds:
            fb = arm.feedback()
            now = time.time()
            if fb is not None:
                samples.append([round(now - t0, 3), round(fb.x, 2), round(fb.y, 2), round(fb.z, 2), round(fb.t, 3)])
                if on_sample is not None:
                    on_sample(fb, now - t0)
            next_t += period
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.time()
    finally:
        arm.torque(True)
    data = {"hz": RECORD_HZ, "name": name, "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), "samples": samples}
    path = motion_path(n)
    path.write_text(json.dumps(data), encoding="utf-8")
    dur = samples[-1][0] if samples else 0.0
    print(f"[motion {n}] 저장: {path} ({len(samples)} samples, {dur:.1f}s). 토크 다시 켬.")
    return path


def play_motion(arm: RoArm, n: int, speed: float = 1.0, on_tick=None, approach_speed: float = 0.2) -> bool:
    """Replay motion n. on_tick(i, total) is called per sample (HUD refresh).
    Returns False if the motion does not exist."""
    path = motion_path(n)
    if not path.exists():
        print(f"[motion {n}] 없음: {path}")
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"]
    if not samples:
        print(f"[motion {n}] 비어 있음")
        return False
    arm.torque(True)
    t, x, y, z, g = samples[0]
    print(f"[motion {n}] 시작 자세로 이동 ({x:.0f}, {y:.0f}, {z:.0f})")
    arm.move_to(Pose(x, y, z, g), speed=approach_speed)
    # wait until we are roughly there (or 4 s)
    deadline = time.time() + 4.0
    while time.time() < deadline:
        fb = arm.feedback()
        if fb is not None and abs(fb.x - x) + abs(fb.y - y) + abs(fb.z - z) < 15:
            break
        if on_tick is not None:
            on_tick(0, len(samples))
        time.sleep(0.05)
    print(f"[motion {n}] 재생 ({len(samples)} samples, {samples[-1][0] / speed:.1f}s, speed x{speed})")
    t_start = time.time()
    for i, (t, x, y, z, g) in enumerate(samples):
        target = t_start + t / speed
        while True:
            now = time.time()
            if now >= target:
                break
            if on_tick is not None:
                on_tick(i, len(samples))
            time.sleep(min(0.02, target - now))
        arm.servo_to(Pose(x, y, z, g))
    # hold the final pose with a regular move so the controller settles there
    t, x, y, z, g = samples[-1]
    arm.move_to(Pose(x, y, z, g), speed=0.3)
    print(f"[motion {n}] 끝")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("record", "play", "list", "delete"))
    p.add_argument("n", type=int, nargs="?")
    p.add_argument("--name", default="", help="record: optional label")
    p.add_argument("--speed", type=float, default=1.0, help="play: time scale (2 = twice as fast)")
    p.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    p.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "auto"))
    args = p.parse_args()

    if args.cmd == "list":
        ms = list_motions()
        if not ms:
            print("저장된 동작이 없습니다 (motions/).")
        for n, m in ms.items():
            print(f"  {n:3d}: {m['seconds']:.1f}s, {m['samples']} samples, {m['recorded_at']} {m['name']}")
        return 0
    if args.n is None:
        print("번호가 필요합니다 (예: record 1)")
        return 1
    if args.cmd == "delete":
        path = motion_path(args.n)
        if path.exists():
            path.unlink()
            print(f"삭제: {path}")
        return 0

    arm = RoArm(ip=args.ip, serial_port=args.serial_port or None)
    if args.cmd == "record":
        if motion_path(args.n).exists():
            print(f"(motion {args.n}이 이미 있어 덮어씁니다)")
        record_motion(arm, args.n, name=args.name)
        return 0
    return 0 if play_motion(arm, args.n, speed=args.speed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
