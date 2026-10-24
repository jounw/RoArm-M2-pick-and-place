"""Teach where each ArUco floor marker sits in robot coordinates.

One-time step for the movable top-down camera (see aruco_floor.py). After
this, the camera may be placed anywhere that sees >= 2 markers.

Procedure:
  1. Tape the printed markers (make_aruco_markers.py) around the work area.
  2. Run this with the top-down camera; the live view shows every detected
     marker with its id (green = taught, yellow = not yet).
  3. Press the marker's id key to teach it (0-9, or 'n' for the id typed
     next): torque is switched off, you put the gripper TIP on the marker's
     centre, press Enter in the terminal, and the arm's (x, y) is recorded.
     Teach at least 4, spread out; more = more robust.
  4. Press 's' to solve and save. From the taught centres a first
     homography is fitted, then each taught marker's four corner positions
     in robot coordinates are derived (the marker's size and its orientation
     in the image give them) and stored, so the runtime can use 4 points
     per marker.

Keys: 0-9 teach that id, n teach an id typed in the terminal (>= 10),
      d delete the last taught marker, s solve + save, q quit.

Guided mode (simplest): --auto walks through every visible marker in id
order (or --ids 6 7 8 9 10 11) -- for each one: torque off, put the tip on
it, press Enter -- then solves and saves by itself.

Nudge mode (most accurate, matches how picking works): --nudge drives the
arm itself to grab height near each marker; you steer it with keys while
watching the camera until the OPEN gripper is centred over the marker, then
Enter. The recorded (x, y) is exactly the command that puts the gripper on
that floor point, so the arm's wrist-vs-fingertip geometry cancels out
(hand-placing the "tip" leaves ~10%% radial scale error, measured).

  keys (camera window focused):  i/k = x +/-   j/l = y +/-   u/o = z +/-
                                 1 = 1 mm steps, 5 = 5 mm steps (default), 0 = 10 mm
                                 Enter = record, x = skip marker, q = quit

Run: python3 calibrate_aruco_floor.py --camera-index 0 --nudge
     python3 calibrate_aruco_floor.py --camera-index 0 --auto
     python3 calibrate_aruco_floor.py --camera-index 0          (interactive keys)
     python3 calibrate_aruco_floor.py --record3d --auto
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

import aruco_floor
from aruco_floor import DEFAULT_PATH, MirrorFix, detect, make_detector
from roarm_driver import RoArm

WINDOW = "ArUco floor calibration"


def open_camera(args):
    if args.record3d:
        from record3d_camera import Record3DCamera

        cam = Record3DCamera(device_index=args.device_index)
        return lambda: cam.get_frame().rgb, cam.close
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera index {args.camera_index}")

    def grab():
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return frame

    return grab, cap.release


def teach_marker(arm: RoArm, marker_id: int, restore_torque: bool = True) -> tuple[float, float] | None:
    """restore_torque=False leaves the arm limp after recording so it can be
    moved away by hand (the caller turns torque back on)."""
    print(f"\n[id {marker_id}] 토크를 끕니다. 그리퍼 끝을 마커 {marker_id}의 정중앙에 대고 Enter (취소: x + Enter)")
    arm.torque(False)
    try:
        ans = input("> ").strip().lower()
        if ans == "x":
            print("취소")
            return None
        pose = None
        for _ in range(5):
            pose = arm.feedback()
            if pose is not None:
                break
            time.sleep(0.2)
        if pose is None:
            print("팔 피드백을 못 받았습니다. 다시 시도하세요.")
            return None
        print(f"[id {marker_id}] robot (x, y) = ({pose.x:.1f}, {pose.y:.1f})  z={pose.z:.1f}")
        return float(pose.x), float(pose.y)
    finally:
        if restore_torque:
            arm.torque(True)


def solve(taught: dict[int, tuple[float, float]], pixel_corners: dict[int, np.ndarray], marker_size_mm: float) -> dict | None:
    """Fit pixel->world from the taught centres, then derive each marker's
    world corners: transform its pixel corners with that H and re-centre
    them on the taught centre (the taught point is the ground truth; the
    H only supplies the marker's orientation and perspective)."""
    ids = [i for i in taught if i in pixel_corners]
    if len(ids) < 4:
        print(f"저장하려면 현재 화면에 보이는 가르친 마커가 4개 이상이어야 합니다 (지금 {len(ids)}개)")
        return None
    src = np.array([pixel_corners[i].mean(axis=0) for i in ids], dtype=np.float64)
    dst = np.array([taught[i] for i in ids], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
        print("호모그래피를 풀지 못했습니다. 가르친 점들이 한 직선 위에 있거나 서로 겹칩니다:")
        for i, (px, wd) in enumerate(zip(src, dst)):
            print(f"  id{ids[i]}: 픽셀 ({px[0]:.0f}, {px[1]:.0f})  ->  로봇 ({wd[0]:.1f}, {wd[1]:.1f})")
        return None
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.sqrt(np.sum((proj - dst) ** 2, axis=1))
    print("마커별 잔차(mm): " + ", ".join(f"id{i}={e:.1f}" for i, e in zip(ids, err)))
    if err.max() > 15:
        print("경고: 잔차가 15mm를 넘는 마커가 있습니다. 그 마커를 'd'로 지우고 다시 가르치는 것을 권합니다.")

    markers = {}
    for i in ids:
        wc = cv2.perspectiveTransform(pixel_corners[i].reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
        wc = wc - wc.mean(axis=0) + np.array(taught[i])  # keep the taught centre exact
        # sanity: derived side length should be close to the printed size
        side = float(np.mean([np.linalg.norm(wc[k] - wc[(k + 1) % 4]) for k in range(4)]))
        if abs(side - marker_size_mm) > marker_size_mm * 0.25:
            print(f"경고: id{i}의 추정 한 변이 {side:.0f}mm (인쇄 {marker_size_mm:.0f}mm). 인쇄 배율이나 카메라 왜곡을 확인하세요.")
        markers[str(i)] = {"center": [taught[i][0], taught[i][1]], "corners": wc.round(2).tolist()}
    return markers


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-index", type=int, default=0, help="top-down webcam index (Iriun etc.)")
    p.add_argument("--record3d", action="store_true", help="use the Record3D iPad camera instead of a webcam")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--dictionary", default="DICT_4X4_50")
    p.add_argument("--marker-size-mm", type=float, default=50.0, help="printed black square side (make_aruco_markers.py --size-mm)")
    p.add_argument("--out", default=DEFAULT_PATH)
    p.add_argument("--auto", action="store_true", help="guided mode: go through the markers one by one (Enter each), then save")
    p.add_argument("--keep-old", action="store_true", help="--nudge: start from the markers already in the output file instead of fresh")
    p.add_argument("--nudge", action="store_true", help="arm drives itself to grab height at each marker; steer with keys until the gripper is centred, Enter records")
    p.add_argument("--start-pose", default="roarm_start_pose.json")
    p.add_argument("--floor-calibration", default="floor_calibration.json")
    p.add_argument("--floor-clearance-mm", type=float, default=15.0, help="--nudge: gripper height above the calibrated floor while aligning (same as picking)")
    p.add_argument("--hover-clearance-mm", type=float, default=120.0, help="--nudge: travel height between markers")
    p.add_argument("--ids", type=int, nargs="*", default=None, help="--auto: which ids, in order (default: all visible, ascending)")
    p.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    p.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "auto"))
    p.add_argument("--baudrate", type=int, default=115200)
    args = p.parse_args()

    arm = RoArm(ip=args.ip, serial_port=args.serial_port or None, baudrate=args.baudrate)
    raw_grab, close = open_camera(args)
    detector = make_detector(args.dictionary)
    unmirror = MirrorFix(detector)

    def grab():
        return unmirror.fix(raw_grab())  # frames are TRUE images from here on (see MirrorFix)

    taught: dict[int, tuple[float, float]] = {}
    order: list[int] = []
    if Path(args.out).exists() and args.nudge and not args.keep_old:
        print(f"({args.out}의 기존 티칭은 무시합니다 -- nudge 모드는 명령 좌표 기준이라 손으로 가르친 값과 섞지 않습니다. 이어서 하려면 --keep-old)")
    elif Path(args.out).exists():
        old = json.loads(Path(args.out).read_text(encoding="utf-8"))
        for k, v in old.get("markers", {}).items():
            taught[int(k)] = (float(v["center"][0]), float(v["center"][1]))
            order.append(int(k))
        print(f"{args.out}에서 기존 티칭 {len(taught)}개를 불러왔습니다: {sorted(taught)} (덮어쓰려면 그 id를 다시 가르치세요)")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    mirror = None

    def provisional_H(found):
        """pixel->world from the markers taught so far (>= 4 visible), for the
        on-screen check of where each recorded arm position lands."""
        ids = [i for i in taught if i in found]
        if len(ids) < 4:
            return None
        src = np.array([found[i].mean(axis=0) for i in ids], dtype=np.float64)
        dst = np.array([taught[i] for i in ids], dtype=np.float64)
        try:
            H, _ = cv2.findHomography(src, dst, 0)
        except cv2.error:
            return None
        if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
            return None  # degenerate (two identical teach points / collinear markers): skip the overlay
        return H

    def show(frame, found, highlight=None, line=""):
        disp = frame.copy()
        for i, c in found.items():
            color = (0, 0, 255) if i == highlight else (0, 220, 0) if i in taught else (0, 220, 255)
            pts = c.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(disp, [pts], True, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.polylines(disp, [pts], True, color, 4 if i == highlight else 3, cv2.LINE_AA)
            cx, cy = c.mean(axis=0).astype(int)
            label = f"{i}" + ("  taught" if i in taught else "")
            cv2.putText(disp, label, (cx - 18, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.putText(disp, label, (cx - 18, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
            cv2.drawMarker(disp, (cx, cy), color, cv2.MARKER_CROSS, 18, 2)
        # where the recorded arm positions land, once >= 4 taught markers are visible:
        # a magenta cross should sit on each taught marker's centre
        H = provisional_H(found)
        if H is not None:
            H_inv = np.linalg.inv(H)
            for i, (wx, wy) in taught.items():
                v = H_inv @ np.array([wx, wy, 1.0])
                if abs(v[2]) < 1e-9 or not np.all(np.isfinite(v)):
                    continue
                px, py = int(round(v[0] / v[2])), int(round(v[1] / v[2]))
                cv2.drawMarker(disp, (px, py), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 34, 3)
                if i in found:
                    err = float(np.hypot(*(found[i].mean(axis=0) - (px, py))))
                    cv2.putText(disp, f"{err:.0f}px", (px + 20, py + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2, cv2.LINE_AA)
        if aruco_floor.last_duplicates:
            line += f"   DUPLICATE ids {aruco_floor.last_duplicates} ignored -- remove the spare sheet!"
        cv2.putText(disp, line, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(disp, line, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW, disp)
        cv2.waitKey(1)

    def save(found) -> bool:
        markers = solve(taught, found, args.marker_size_mm)
        if markers is None:
            return False
        data = {
            "dictionary": args.dictionary,
            "marker_size_mm": args.marker_size_mm,
            "markers": markers,
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        # keep the user's preferred screen orientation across recalibrations
        if Path(args.out).exists():
            try:
                prev = json.loads(Path(args.out).read_text(encoding="utf-8"))
                if "view_rotate" in prev:
                    data["view_rotate"] = prev["view_rotate"]
            except (OSError, ValueError):
                pass
        Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"저장: {args.out} ({len(markers)}개 마커). 이제 track_aruco_floor.py로 확인하거나 메인 스크립트에 --floor aruco 를 주세요.")
        return True

    def collect(seconds: float, line: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Union of detections over `seconds` (a marker that flickers is still
        counted) with the last frame; keeps the window live meanwhile."""
        nonlocal mirror
        acc: dict[int, np.ndarray] = {}
        t_end = time.time() + seconds
        frame = None
        while time.time() < t_end or frame is None:
            frame = grab()
            found = detect(detector, frame, False)
            acc.update(found)
            show(frame, acc, line=line)
        return frame, acc

    if args.nudge:
        from pick_place import WORKSPACE, clamp, expand_workspace, load_floor_z, load_start_pose, load_workspace_calibration
        from roarm_driver import Pose

        start_pose = load_start_pose(args.start_pose)
        expand_workspace(start_pose)
        ws = load_workspace_calibration("workspace_calibration.json")
        if ws is not None:
            WORKSPACE["x"], WORKSPACE["y"] = ws["x"], ws["y"]
        floor_z = load_floor_z(args.floor_calibration)
        if floor_z is None:
            print(f"{args.floor_calibration}가 없습니다 (바닥 z가 필요). 먼저 바닥 높이를 캘리브레이션하세요.")
            return 1
        grab_z = floor_z + args.floor_clearance_mm
        hover_z = floor_z + args.hover_clearance_mm
        OPEN = RoArm.GRIPPER_OPEN
        print(f"nudge 모드: 집기 높이 z={grab_z:.1f} (바닥 {floor_z:.1f} + {args.floor_clearance_mm:.0f}), 이동 높이 z={hover_z:.1f}")

        WORKSPACE["z"] = (min(WORKSPACE["z"][0], floor_z + 5.0), WORKSPACE["z"][1])
        print(f"작업영역: x={WORKSPACE['x']} y={WORKSPACE['y']} z={WORKSPACE['z']} (이 밖으로는 안 보냅니다)")

        def goto(x, y, z, speed=0.25):
            arm.move_to(Pose(clamp(x, "x"), clamp(y, "y"), clamp(z, "z"), OPEN), speed=speed)

        def travel(x, y):
            """lift, move over, descend -- never drag along the floor"""
            fb = arm.feedback()
            cx, cy = (fb.x, fb.y) if fb is not None else (x, y)
            goto(cx, cy, hover_z)
            time.sleep(0.8)
            goto(x, y, hover_z)
            time.sleep(1.2)
            goto(x, y, grab_z, speed=0.2)
            time.sleep(1.0)

        try:
            arm.torque(True)
            arm.open_gripper()
            goto(start_pose.x, start_pose.y, start_pose.z)
            time.sleep(1.0)
            frame, found = collect(2.0, "nudge calibration: looking for markers...")
            if aruco_floor.last_duplicates:
                print(f"경고: id {aruco_floor.last_duplicates}가 두 번 보입니다 (여분 인쇄지?). 치우고 다시 실행하세요.")
                return 1
            ids = args.ids if args.ids else sorted(found)
            if not ids:
                print("화면에 마커가 하나도 안 보입니다.")
                return 1
            print(f"\n보이는 마커 {sorted(found)} -> 순서대로 {ids}")
            print("카메라 창을 클릭해 포커스를 준 뒤: i/k = x±, j/l = y±, u/o = z±, 1/5/0 = 스텝(mm), Enter = 기록, x = 건너뜀, q = 종료")
            print("열린 집게의 중심이 마커 정중앙 바로 위에 오게 맞추세요 (집을 때 물체가 놓일 자리).\n")
            step = 5.0
            cur = [start_pose.x, start_pose.y, grab_z]
            for n, marker_id in enumerate(ids, 1):
                # initial guess: provisional map if we have one, else the last taught point, else start pose
                H = provisional_H(found)
                if H is not None and marker_id in found:
                    c = found[marker_id].mean(axis=0)
                    v = H @ np.array([c[0], c[1], 1.0])
                    gx, gy = float(v[0] / v[2]), float(v[1] / v[2])
                elif marker_id in taught:
                    gx, gy = taught[marker_id]
                else:
                    gx, gy = cur[0], cur[1]
                print(f"[{n}/{len(ids)}] id {marker_id}: ({gx:.0f}, {gy:.0f})로 이동합니다. 화면을 보며 맞춘 뒤 Enter.")
                travel(gx, gy)
                cur = [gx, gy, grab_z]
                while True:
                    frame = grab()
                    found = detect(detector, frame, False)
                    show(frame, found, highlight=marker_id,
                         line=f"[{n}/{len(ids)}] marker {marker_id}: centre the open gripper over it  (step {step:.0f}mm)  i/k x  j/l y  u/o z  Enter=record  x=skip")
                    key = cv2.waitKey(30) & 0xFF
                    if key == 255:
                        continue
                    moved = True
                    if key == ord("i"):
                        cur[0] += step
                    elif key == ord("k"):
                        cur[0] -= step
                    elif key == ord("j"):
                        cur[1] += step
                    elif key == ord("l"):
                        cur[1] -= step
                    elif key == ord("u"):
                        cur[2] += step
                    elif key == ord("o"):
                        cur[2] -= step
                    elif key == ord("1"):
                        step, moved = 1.0, False
                    elif key == ord("5"):
                        step, moved = 5.0, False
                    elif key == ord("0"):
                        step, moved = 10.0, False
                    elif key in (13, 10):
                        fb = arm.feedback()
                        xy = (float(fb.x), float(fb.y)) if fb is not None else (cur[0], cur[1])
                        taught[marker_id] = xy
                        if marker_id in order:
                            order.remove(marker_id)
                        order.append(marker_id)
                        print(f"[id {marker_id}] 기록: commanded ({cur[0]:.1f}, {cur[1]:.1f})  feedback ({xy[0]:.1f}, {xy[1]:.1f})  z={cur[2]:.1f}")
                        break
                    elif key == ord("x"):
                        print(f"id {marker_id} 건너뜀")
                        break
                    elif key == ord("q"):
                        goto(cur[0], cur[1], hover_z)
                        return 1
                    else:
                        moved = False
                    if moved:
                        cur[0], cur[1], cur[2] = clamp(cur[0], "x"), clamp(cur[1], "y"), clamp(cur[2], "z")
                        goto(cur[0], cur[1], cur[2], speed=0.3)
                        print(f"  -> ({cur[0]:.0f}, {cur[1]:.0f}, {cur[2]:.0f})", end="\r")
                goto(cur[0], cur[1], hover_z)
                time.sleep(0.6)
            print("\n모두 끝. 시작 자세로 돌아간 뒤 저장합니다.")
            goto(cur[0], cur[1], hover_z)
            time.sleep(0.8)
            goto(start_pose.x, start_pose.y, start_pose.z)
            time.sleep(1.5)
            frame, found = collect(2.0, "solving...")
            return 0 if save(found) else 1
        finally:
            close()
            cv2.destroyAllWindows()

    if args.auto:
        try:
            frame, found = collect(2.0, "guided calibration: looking for markers...")
            if aruco_floor.last_duplicates:
                print(f"경고: id {aruco_floor.last_duplicates}가 두 번 보입니다 (여분 인쇄지?). 치우고 다시 실행하세요.")
                return 1
            ids = args.ids if args.ids else sorted(found)
            if not ids:
                print("화면에 마커가 하나도 안 보입니다.")
                return 1
            print(f"\n화면에 보이는 마커: {sorted(found)} -> 순서대로 {len(ids)}개를 가르칩니다: {ids}")
            print("마커마다 두 번 Enter: (1) 토크가 꺼지면 그리퍼 끝을 마커 정중앙에 대고 Enter, (2) 팔을 치운 뒤 Enter.")
            print("건너뛰려면 x + Enter. 가르칠 때 마커가 팔에 가려 안 보여도 괜찮습니다 -- 위치는 저장 때 다시 찾습니다.\n")
            for n, marker_id in enumerate(ids, 1):
                frame, found = collect(0.5, f"[{n}/{len(ids)}] marker {marker_id}: put the gripper tip on it, Enter")
                show(frame, found, highlight=marker_id,
                     line=f"[{n}/{len(ids)}] marker {marker_id}: put the gripper tip on its centre, then Enter in the terminal")
                if marker_id not in found:
                    print(f"[{n}/{len(ids)}] (id {marker_id}는 지금 화면에 안 보입니다 -- 팔에 가려진 것이면 그냥 진행하세요)")
                print(f"[{n}/{len(ids)}]", end="")
                xy = teach_marker(arm, marker_id, restore_torque=False)  # stays limp so it can be moved away
                if xy is None:
                    arm.torque(True)
                    continue
                same = [j for j, (tx, ty) in taught.items() if j != marker_id and abs(tx - xy[0]) + abs(ty - xy[1]) < 5.0]
                if same:
                    print(f"  경고: id {marker_id}의 위치가 id {same}와 거의 같습니다 (팔을 안 옮기고 Enter?). 끝나고 --nudge --keep-old --ids {marker_id} 로 다시 가르치세요.")
                taught[marker_id] = xy
                if marker_id in order:
                    order.remove(marker_id)
                order.append(marker_id)
                print(f"  -> 토크는 꺼진 채입니다. 팔을 마커 위에서 치우고 Enter (화면에 보라색 X가 마커 중심에 오는지 확인)")
                input("> ")
                arm.torque(True)
                frame, found = collect(1.0, f"marker {marker_id} taught -- check: magenta X should sit on each taught marker")
                if len([i for i in taught if i in found]) >= 4:
                    H = provisional_H(found)
                    errs = []
                    for i in taught:
                        if i in found and H is not None:
                            v = np.linalg.inv(H) @ np.array([*taught[i], 1.0])
                            errs.append(f"id{i}={float(np.hypot(*(found[i].mean(axis=0) - (v[0] / v[2], v[1] / v[2])))):.0f}px")
                    print("  화면 오차: " + ", ".join(errs))
            print("\n모두 끝. 토크를 끕니다 -- 팔이 마커를 가리지 않게 치운 뒤 Enter를 누르면 저장합니다.")
            arm.torque(False)
            input("> ")
            arm.torque(True)
            frame, found = collect(2.0, "solving...")
            return 0 if save(found) else 1
        finally:
            close()
            cv2.destroyAllWindows()

    print(__doc__.split("Keys:")[1].split("Guided mode")[0].strip())
    try:
        while True:
            frame = grab()
            found = detect(detector, frame, False)
            show(frame, found, line=f"visible {sorted(found)}  taught {sorted(taught)}   keys: 0-9/n teach, d undo, s save, q quit")
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return 0
            marker_id = None
            if ord("0") <= key <= ord("9"):
                marker_id = key - ord("0")
            elif key == ord("n"):
                try:
                    marker_id = int(input("가르칠 마커 id: ").strip())
                except ValueError:
                    continue
            if marker_id is not None:
                if marker_id not in found:
                    print(f"마커 {marker_id}가 지금 화면에 안 보입니다 (보이는 것: {sorted(found)})")
                    continue
                xy = teach_marker(arm, marker_id)
                if xy is not None:
                    taught[marker_id] = xy
                    if marker_id in order:
                        order.remove(marker_id)
                    order.append(marker_id)
            elif key == ord("d") and order:
                last = order.pop()
                taught.pop(last, None)
                print(f"id {last} 삭제")
            elif key == ord("s"):
                save(found)
    finally:
        close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
