"""RoArm-M2 pick-and-place using the iPad's Record3D app as a FIXED overview
camera, but using ONLY its RGB stream -- the LiDAR depth is never read.

This is pick_place_fixed_cam.py's logic (ground-plane homography from
calibrate_ground_homography.py) applied to the Record3D RGB frame instead of
a plain cv2.VideoCapture webcam. The iPad is just a (very good) fixed RGB
camera here; its depth sensor is completely unused.

Since there's no depth, height (z) cannot come from the object itself: exactly
like pick_place_fixed_cam.py, z comes from floor_calibration.json (the
calibrated floor height, + a clearance) -- this only works correctly for
objects that sit flat on that calibrated floor.

Loop, once per cycle:
  1. move to start pose, open gripper
  2. grab one Record3D frame, use ONLY its rgb (ignore depth_m entirely);
     detect the target (VLM / darkest-blob / manual box) on that RGB image
  3. map the detected box's center pixel -> real (x, y) via the homography
     (ground_homography.json, 4 or 9+ point calibration)
  4. move straight to that (x, y) at the calibrated floor height (+ clearance),
     close the gripper
  5. lift, carry to place location (taught pose OR click-to-place), open, lift
  6. return to start pose

Handles the iPad's raw-frame rotation the same way every other LiDAR script
in this project does (--rotate, default "left"), and reuses
displayed_to_original_xy so clicks/drags on the rotated preview map back
correctly onto the original RGB frame that the homography was calibrated
against. IMPORTANT: calibrate_ground_homography.py must be re-run with a
Record3D-based capture (or otherwise using the rotation/orientation you plan
to run this script with) so the pixel_points it saved match what this script
sees -- a homography calibrated against a different camera/orientation won't
line up.

Run: python3 pick_place_record3d_rgb.py --detector manual --place-mode click --serial-port /dev/cu.usbserial-120
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from darkest_detector import find_darkest_box
from record3d_camera import Record3DCamera, displayed_to_original_xy, rotate_for_display
from roarm_driver import Pose, RoArm
from vision import locate

FRAME_PATH = "vision_frame_record3d_rgb.jpg"
WINDOW_NAME = "RoArm Record3D RGB-only Pick-Place"


def load_start_pose(path: str) -> Pose:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    p = data["base_pose"]
    return Pose(x=float(p["x"]), y=float(p["y"]), z=float(p["z"]), t=float(p["t"]))


def load_place_pose(path: str) -> Pose | None:
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    p = data.get("base_pose", data)
    return Pose(x=float(p["x"]), y=float(p["y"]), z=float(p["z"]), t=float(p["t"]))


def load_floor_z(path: str) -> float:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found -- run calibrate_floor.py first")
    return float(json.loads(Path(path).read_text(encoding="utf-8"))["z"])


def load_homography(path: str) -> np.ndarray:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found -- run calibrate_ground_homography.py first")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.array(data["homography"], dtype=np.float64)


def pixel_to_world(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
    v = H @ np.array([px, py, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


def world_to_pixel(H_inv: np.ndarray, wx: float, wy: float) -> tuple[int, int]:
    v = H_inv @ np.array([wx, wy, 1.0])
    return int(round(v[0] / v[2])), int(round(v[1] / v[2]))


class GroundGridOverlay:
    """AR-style grid overlay of the calibrated floor area, drawn on the
    ORIGINAL (unrotated) frame -- rotate_for_display is applied to the whole
    annotated frame afterward, same pattern as pick_place_lidar.py."""

    def __init__(self, homography_path: str, grid_spacing_mm: float = 50.0):
        self.enabled = False
        self.corners_px: list[tuple[int, int]] = []
        self.grid_lines_px: list[tuple[tuple[int, int], tuple[int, int]]] = []
        if not Path(homography_path).exists():
            return
        data = json.loads(Path(homography_path).read_text(encoding="utf-8"))
        pixel_points = data.get("pixel_points")
        world_points = data.get("world_points")
        H = np.array(data["homography"], dtype=np.float64)
        if not pixel_points or not world_points or len(pixel_points) < 4:
            return
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return

        self.corners_px = [(int(x), int(y)) for x, y in pixel_points]

        xs = [w[0] for w in world_points]
        ys = [w[1] for w in world_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        lines = []
        n_x = max(1, int(round((x_max - x_min) / grid_spacing_mm)))
        n_y = max(1, int(round((y_max - y_min) / grid_spacing_mm)))
        for i in range(n_x + 1):
            wx = x_min + (x_max - x_min) * i / n_x
            lines.append((world_to_pixel(H_inv, wx, y_min), world_to_pixel(H_inv, wx, y_max)))
        for j in range(n_y + 1):
            wy = y_min + (y_max - y_min) * j / n_y
            lines.append((world_to_pixel(H_inv, x_min, wy), world_to_pixel(H_inv, x_max, wy)))
        self.grid_lines_px = lines
        self.enabled = True

    def draw(self, frame) -> None:
        if not self.enabled:
            return
        overlay = frame.copy()
        for p1, p2 in self.grid_lines_px:
            cv2.line(overlay, p1, p2, (0, 200, 255), 1, cv2.LINE_AA)
        if len(self.corners_px) >= 4:
            pts = np.array(self.corners_px, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 220, 0), thickness=2, lineType=cv2.LINE_AA)
            for x, y in self.corners_px:
                cv2.circle(overlay, (x, y), 5, (0, 220, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)


_ground_grid: GroundGridOverlay | None = None


def _draw_grid(frame) -> None:
    if _ground_grid is not None:
        _ground_grid.draw(frame)


def capture_rgb_for_vlm(rgb: np.ndarray) -> None:
    cv2.imwrite(FRAME_PATH, rgb)


def select_roi_manually(cam: Record3DCamera, rotate: str) -> tuple[int, int, int, int] | None:
    """Freeze on a live (rotated-for-display) frame, drag a box. Returns the
    box in ORIGINAL (unrotated) frame pixel coordinates -- same coordinate
    space the homography's pixel_points were recorded in."""
    print("  라이브 화면입니다 (RGB만 사용). SPACE/Enter로 화면을 잡은 뒤 물체에 박스를 드래그하세요 (취소: c)")
    orig_frame = None
    while True:
        f = cam.get_frame()
        orig_frame = f.rgb
        disp = rotate_for_display(orig_frame, rotate).copy()
        cv2.putText(disp, "SPACE/Enter: freeze & draw box   c: cancel", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
        cv2.putText(disp, "SPACE/Enter: freeze & draw box   c: cancel", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
        cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):
            break
        if key == ord("c"):
            return None
    orig_h, orig_w = orig_frame.shape[:2]
    disp_frame = rotate_for_display(orig_frame, rotate)
    dx, dy, dw, dh = (int(v) for v in cv2.selectROI(WINDOW_NAME, disp_frame, fromCenter=False, showCrosshair=True))
    if dw <= 0 or dh <= 0:
        return None
    corners = [(dx, dy), (dx + dw, dy), (dx, dy + dh), (dx + dw, dy + dh)]
    mapped = [displayed_to_original_xy(cx, cy, orig_w, orig_h, rotate) for cx, cy in corners]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    x, y, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return x, y, x2 - x, y2 - y


def normalized_box_to_pixel(box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    return int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h)


def detect_target_pixel_box(
    cam: Record3DCamera, rgb: np.ndarray, target: str | None, detector: str, min_area_frac: float, rotate: str
) -> tuple[int, int, int, int] | None:
    """Returns a pixel (x1, y1, x2, y2) box on the ORIGINAL rgb frame, or None."""
    if detector == "manual":
        roi = select_roi_manually(cam, rotate)
        if roi is None:
            return None
        x, y, w, h = roi
        return x, y, x + w, y + h
    if detector == "darkest":
        box = find_darkest_box(rgb, min_area_frac=min_area_frac)
        if box is None:
            return None
        x, y, w, h = box
        return x, y, x + w, y + h
    # vlm
    capture_rgb_for_vlm(rgb)
    det = locate(FRAME_PATH, target)
    print(f"  VLM: {det.answer!r}")
    if det.box is None:
        return None
    return normalized_box_to_pixel(det.box, rgb.shape)


def draw_label(frame, org: tuple[int, int], text: str, color: tuple[int, int, int] = (255, 80, 80)) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def show_preview(rgb, box, world_xy: tuple[float, float] | None, text: str, rotate: str) -> None:
    """box is in ORIGINAL frame coords -- draw on a copy of the original,
    then rotate the whole annotated image for display."""
    annotated = rgb.copy()
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 80, 80), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated, (cx, cy), 5, (255, 80, 80), -1)
        if world_xy is not None:
            wx, wy = world_xy
            draw_label(annotated, (cx + 10, cy - 10), f"({wx:.0f}, {wy:.0f})mm")
    _draw_grid(annotated)
    disp = rotate_for_display(annotated, rotate)
    cv2.putText(disp, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
    cv2.putText(disp, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
    cv2.imshow(WINDOW_NAME, disp)
    cv2.waitKey(1)


def show_live(cam: Record3DCamera, seconds: float, rotate: str) -> None:
    end = time.time() + seconds
    while time.time() < end:
        frame = cam.get_frame()
        annotated = frame.rgb.copy()
        _draw_grid(annotated)
        cv2.imshow(WINDOW_NAME, rotate_for_display(annotated, rotate))
        cv2.waitKey(30)


def wait_gripper_settled(arm: RoArm, cam: Record3DCamera, rotate: str, timeout: float = 2.5) -> None:
    deadline = time.time() + timeout
    last_t = None
    while time.time() < deadline:
        show_live(cam, 0.15, rotate)
        fb = arm.feedback()
        if fb is None:
            continue
        if last_t is not None and abs(fb.t - last_t) < 0.01:
            return
        last_t = fb.t


def pick(arm: RoArm, cam: Record3DCamera, carry_z: float, grip_settle_sec: float, grip_close_rad: float, rotate: str) -> None:
    arm.set_gripper(grip_close_rad)
    wait_gripper_settled(arm, cam, rotate, timeout=3.0)
    show_live(cam, grip_settle_sec, rotate)
    pose = arm.feedback()
    pose.z = max(pose.z, carry_z)
    arm.move_to(pose)
    show_live(cam, 1.0, rotate)


def select_point_click(cam: Record3DCamera, prompt_text: str, H: np.ndarray, rotate: str) -> tuple[int, int] | None:
    """Freeze on the live (rotated-for-display) feed, click one point.
    Returns (px, py) in ORIGINAL frame coordinates, or None if cancelled."""
    clicked_disp: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_disp.clear()
            clicked_disp.append((x, y))

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print(f"  {prompt_text} (클릭 후 c로 확정, 취소: q)")
    try:
        while True:
            frame = cam.get_frame()
            orig_h, orig_w = frame.rgb.shape[:2]
            annotated = frame.rgb.copy()
            _draw_grid(annotated)
            disp = rotate_for_display(annotated, rotate).copy()
            if clicked_disp:
                dx, dy = clicked_disp[0]
                px, py = displayed_to_original_xy(dx, dy, orig_w, orig_h, rotate)
                cv2.circle(disp, (dx, dy), 8, (0, 220, 0), -1)
                wx, wy = pixel_to_world(H, px, py)
                draw_label(disp, (dx + 12, dy - 12), f"({wx:.0f}, {wy:.0f})mm", color=(0, 220, 0))
            cv2.putText(disp, f"{prompt_text} (click, c: confirm, q: cancel)", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
            cv2.putText(disp, f"{prompt_text} (click, c: confirm, q: cancel)", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
            cv2.imshow(WINDOW_NAME, disp)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("c") and clicked_disp:
                dx, dy = clicked_disp[0]
                return displayed_to_original_xy(dx, dy, orig_w, orig_h, rotate)
    finally:
        cv2.setMouseCallback(WINDOW_NAME, lambda *a: None)


def pick_place_pose_from_click(cam: Record3DCamera, H: np.ndarray, place_z: float, rotate: str) -> Pose | None:
    pt = select_point_click(cam, "놓을 위치를 클릭하세요", H, rotate)
    if pt is None:
        return None
    px, py = pt
    wx, wy = pixel_to_world(H, px, py)
    print(f"  클릭 픽셀=({px},{py}) -> world=({wx:.1f},{wy:.1f})")
    return Pose(x=wx, y=wy, z=place_z, t=RoArm.GRIPPER_OPEN)


def place_at(arm: RoArm, cam: Record3DCamera, place_pose: Pose, carry_z: float, grip_close_rad: float, rotate: str) -> None:
    high_z = max(place_pose.z, carry_z)
    arm.move_to(Pose(place_pose.x, place_pose.y, high_z, grip_close_rad))
    show_live(cam, 1.3, rotate)
    arm.move_to(Pose(place_pose.x, place_pose.y, place_pose.z, grip_close_rad))
    show_live(cam, 1.0, rotate)
    arm.open_gripper()
    wait_gripper_settled(arm, cam, rotate, timeout=2.5)
    show_live(cam, 0.3, rotate)
    arm.move_to(Pose(place_pose.x, place_pose.y, high_z, RoArm.GRIPPER_OPEN))
    show_live(cam, 0.8, rotate)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RoArm-M2 pick-and-place using Record3D as an RGB-only fixed camera (no LiDAR depth used)")
    p.add_argument("--target", default=None, help="natural language description of the object (required for --detector vlm)")
    p.add_argument("--detector", choices=("vlm", "darkest", "manual"), default="manual")
    p.add_argument("--vlm-backend", choices=("gemini", "openai"), default=None)
    p.add_argument("--min-area-frac", type=float, default=0.01)
    p.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    p.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "/dev/cu.usbserial-120"))
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--device-index", type=int, default=0, help="Record3D device index, if multiple iOS devices are connected")
    p.add_argument("--start-pose", default="roarm_start_pose.json")
    p.add_argument(
        "--place-mode",
        choices=("taught", "click"),
        default="taught",
        help="'taught' uses --place-pose/place_pose.json (fixed spot); "
             "'click' picks up, holds in the air, and waits for you to click the drop point on the live feed each cycle",
    )
    p.add_argument("--place-pose", default="place_pose.json")
    p.add_argument("--place-x", type=float, default=None)
    p.add_argument("--place-y", type=float, default=None)
    p.add_argument("--place-z", type=float, default=None)
    p.add_argument("--click-place-z", type=float, default=None, help="z to set the object down at in --place-mode click (default: calibrated floor z + clearance)")
    p.add_argument("--homography", default="ground_homography.json", help="output of calibrate_ground_homography.py")
    p.add_argument("--grid-spacing-mm", type=float, default=50.0, help="AR overlay: grid line spacing in real mm on the calibrated floor area (0 disables the overlay)")
    p.add_argument("--floor-calibration", default="floor_calibration.json")
    p.add_argument("--floor-clearance-mm", type=float, default=5.0, help="gripper stops this far above the calibrated floor")
    p.add_argument(
        "--rotate",
        choices=("none", "left", "right", "180"),
        default="left",
        help="rotate the preview window (display only, clicks map back correctly): "
             "'left' = 90deg counter-clockwise (default, matches iPad held upright), 'right' = 90deg clockwise, '180' = upside down",
    )
    p.add_argument("--approach-speed", type=float, default=0.5, help="arm speed (0-1) for the move to the target")
    p.add_argument("--carry-z", type=float, default=None, help="absolute z to travel at while carrying (default: start pose height)")
    p.add_argument("--grip-settle-sec", type=float, default=0.5)
    p.add_argument("--grip-close-rad", type=float, default=3.25)
    p.add_argument("--max-cycles", type=int, default=0, help="0 = run until Ctrl+C or no object found")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.vlm_backend:
        os.environ["VLM_BACKEND"] = args.vlm_backend

    H = load_homography(args.homography)

    floor_z = load_floor_z(args.floor_calibration)
    grab_z = floor_z + args.floor_clearance_mm
    print(f"floor z={floor_z:.1f}, grab z={grab_z:.1f}")

    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)
    start_pose = load_start_pose(args.start_pose)

    place_pose = None
    if args.place_mode == "taught":
        place_pose = load_place_pose(args.place_pose)
        if place_pose is not None:
            print(f"place pose loaded: x={place_pose.x:.1f} y={place_pose.y:.1f} z={place_pose.z:.1f}")
        elif any(v is not None for v in (args.place_x, args.place_y, args.place_z)):
            place_pose = Pose(
                x=args.place_x if args.place_x is not None else start_pose.x,
                y=args.place_y if args.place_y is not None else start_pose.y + 100.0,
                z=args.place_z if args.place_z is not None else start_pose.z,
                t=RoArm.GRIPPER_OPEN,
            )
        else:
            print("놓을 위치가 없습니다 (teach_place.py 실행 권장); 한 번만 집고 시작 자세에서 들고 있습니다.")
    else:
        print("place-mode=click: 매 사이클 물체를 든 채 공중에서 대기하다가, 화면 클릭으로 놓을 위치를 지정합니다.")

    carry_z = args.carry_z if args.carry_z is not None else start_pose.z
    click_place_z = args.click_place_z if args.click_place_z is not None else grab_z

    print("Record3D 카메라에 연결 중... (iPad에서 Record3D 앱 -> USB Streaming mode 켜져 있어야 함, RGB만 사용합니다)")
    cam = Record3DCamera(device_index=args.device_index)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    global _ground_grid
    if args.grid_spacing_mm > 0:
        _ground_grid = GroundGridOverlay(args.homography, grid_spacing_mm=args.grid_spacing_mm)
        if _ground_grid.enabled:
            print(f"AR 그리드 오버레이 켜짐 (간격 {args.grid_spacing_mm:.0f}mm)")

    try:
        cycle = 0
        while True:
            cycle += 1
            print(f"===== cycle {cycle} =====")
            print("moving to start pose")
            arm.move_to(start_pose)
            show_live(cam, 1.0, args.rotate)
            arm.open_gripper()
            wait_gripper_settled(arm, cam, args.rotate, timeout=2.0)

            frame = cam.get_frame()  # depth_m is intentionally never used below
            box = detect_target_pixel_box(cam, frame.rgb, args.target, args.detector, args.min_area_frac, args.rotate)
            if box is None:
                show_preview(frame.rgb, None, None, "no target found", args.rotate)
                if cycle == 1:
                    print("failed to locate target, aborting")
                    return 1
                print("no more objects found; stopping loop")
                break

            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            wx, wy = pixel_to_world(H, cx, cy)
            show_preview(frame.rgb, box, (wx, wy), f"target pixel=({cx:.0f},{cy:.0f}) -> world=({wx:.1f},{wy:.1f})", args.rotate)
            print(f"target pixel center=({cx:.0f},{cy:.0f}) -> world (x,y)=({wx:.1f},{wy:.1f})")

            approach_pose = Pose(x=wx, y=wy, z=max(grab_z, carry_z), t=RoArm.GRIPPER_OPEN)
            print("moving above target")
            arm.move_to(approach_pose, speed=args.approach_speed)
            show_live(cam, 1.2, args.rotate)

            print("descending to grab height")
            arm.move_to(Pose(wx, wy, grab_z, RoArm.GRIPPER_OPEN), speed=args.approach_speed)
            show_live(cam, 1.0, args.rotate)

            print("picking")
            pick(arm, cam, carry_z, args.grip_settle_sec, args.grip_close_rad, args.rotate)

            if args.place_mode == "click":
                print("공중에서 대기 중: 놓을 위치를 화면에서 클릭하세요")
                click_place_pose = pick_place_pose_from_click(cam, H, click_place_z, args.rotate)
                if click_place_pose is None:
                    print("취소됨: 물체를 든 채 시작 자세로 복귀합니다 (그립 유지)")
                    arm.move_to(Pose(start_pose.x, start_pose.y, carry_z, args.grip_close_rad))
                    show_live(cam, 1.5, args.rotate)
                    arm.close_gripper()
                    show_live(cam, 0.3, args.rotate)
                    break
                print(f"placing at clicked point: x={click_place_pose.x:.1f} y={click_place_pose.y:.1f} z={click_place_pose.z:.1f}")
                place_at(arm, cam, click_place_pose, carry_z, args.grip_close_rad, args.rotate)
                print("returning to start pose (gripper open)")
                arm.move_to(Pose(start_pose.x, start_pose.y, start_pose.z, RoArm.GRIPPER_OPEN))
                show_live(cam, 1.0, args.rotate)
            elif place_pose is None:
                print("returning to start pose, holding the object (no place pose -> single cycle)")
                arm.move_to(start_pose)
                show_live(cam, 1.5, args.rotate)
                arm.close_gripper()
                show_live(cam, 0.3, args.rotate)
                break
            else:
                print("placing at taught pose")
                place_at(arm, cam, place_pose, carry_z, args.grip_close_rad, args.rotate)
                print("returning to start pose (gripper open)")
                arm.move_to(Pose(start_pose.x, start_pose.y, start_pose.z, RoArm.GRIPPER_OPEN))
                show_live(cam, 1.0, args.rotate)

            if args.max_cycles and cycle >= args.max_cycles:
                print(f"reached max cycles ({args.max_cycles}); stopping")
                break
    except KeyboardInterrupt:
        print("\ninterrupted, stopping")
    finally:
        cam.close()
        cv2.destroyWindow(WINDOW_NAME)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
