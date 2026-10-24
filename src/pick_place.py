"""RoArm-M2 VLM pick-and-place.

Top-down eye-in-hand camera. Loop:
  1. move to start pose, open gripper
  2. find the target (VLM / darkest-blob / manual box)
  3. keep it centered under the camera while descending to the calibrated floor height
  4. close the gripper, lift
  5. move to the place pose, open the gripper, lift
  6. return to start pose

Run: python3 pick_place.py --target "the red block" --serial-port /dev/cu.usbserial-120
Or without any API calls: python3 pick_place.py --detector manual --serial-port /dev/cu.usbserial-120
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2

from darkest_detector import find_darkest_box
from roarm_driver import Pose, RoArm
from tracker import Tracker, normalized_box_to_pixel_roi
from vision import ask, locate, route

FRAME_PATH = "vision_frame.jpg"
WINDOW_NAME = "RoArm Eye-in-Hand"

# Widened around the loaded start/place pose at runtime (see expand_workspace),
# this is just a safety fallback if that pose is somehow missing an axis.
WORKSPACE = {
    "x": (80.0, 260.0),
    "y": (-180.0, 180.0),
    "z": (20.0, 320.0),
}

# How far outward (mm) to widen WORKSPACE around each pose we actually plan to visit.
WORKSPACE_MARGIN = {"x": 120.0, "y": 150.0, "z": 120.0}


def clamp(value: float, axis: str) -> float:
    low, high = WORKSPACE[axis]
    return max(low, min(high, value))


def expand_workspace(pose: Pose) -> None:
    """Widen WORKSPACE so it comfortably contains `pose`.

    The default WORKSPACE box is an arbitrary guess; a real calibrated start
    pose can sit right at its edge (e.g. this rig's start pose has x=83, z=31,
    almost at the box's lower bound), which silently clamps every servo
    correction to a no-op. Widening around the poses we actually use avoids that.
    """
    for axis, margin in WORKSPACE_MARGIN.items():
        value = getattr(pose, axis)
        low, high = WORKSPACE[axis]
        WORKSPACE[axis] = (min(low, value - margin), max(high, value + margin))


def load_workspace_calibration(path: str) -> dict[str, tuple[float, float]] | None:
    """Load hand-measured x/y bounds from calibrate_workspace.py, if present."""
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"x": tuple(data["x"]), "y": tuple(data["y"])}


def load_start_pose(path: str) -> Pose:
    if not Path(path).exists():
        raise FileNotFoundError(f"start pose file not found: {path}")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pose = data["base_pose"]
    return Pose(x=float(pose["x"]), y=float(pose["y"]), z=float(pose["z"]), t=float(pose["t"]))


def load_place_pose(path: str) -> Pose | None:
    """Load the hand-taught place location from teach_place.py, if present."""
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    p = data.get("base_pose", data)
    return Pose(x=float(p["x"]), y=float(p["y"]), z=float(p["z"]), t=float(p["t"]))


class RecordingCapture:
    """cv2.VideoCapture wrapper that also writes every frame it reads to a video.

    Every code path (detection, tracking, settling, preview) reads through
    cap.read(), so recording here captures the entire run end to end. One file
    per run, timestamped, in the recordings/ directory.
    """

    def __init__(self, index: int, out_dir: str = "recordings", fps: float = 20.0):
        self.cap = cv2.VideoCapture(index)
        self.writer = None
        self.path = None
        self.fps = fps
        if self.cap.isOpened():
            Path(out_dir).mkdir(exist_ok=True)
            self.path = str(Path(out_dir) / time.strftime("run_%Y%m%d_%H%M%S.mp4"))

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def read(self):
        ok, frame = self.cap.read()
        if ok and self.path is not None:
            if self.writer is None:
                # The writer is created from the FIRST real frame's shape, not
                # CAP_PROP_FRAME_WIDTH/HEIGHT: on macOS (AVFoundation) the
                # reported capture size can differ from what read() actually
                # delivers, and a size mismatch makes VideoWriter silently
                # reject every frame ("FFmpeg: Failed to write frame").
                h, w = frame.shape[:2]
                self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            self.writer.write(frame)
        return ok, frame

    def release(self) -> None:
        self.cap.release()
        if self.writer is not None:
            self.writer.release()
            print(f"카메라 영상 저장됨: {self.path}")


def capture_frame(cap: cv2.VideoCapture):
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("failed to read a camera frame")
    cv2.imwrite(FRAME_PATH, frame)
    return frame


def sleep_showing(cap: cv2.VideoCapture, seconds: float) -> None:
    """Wait `seconds` while keeping the live camera feed on screen.

    Used during the pick/place moves so the window keeps streaming (and stays
    responsive) instead of freezing/looking off until the object is placed.
    """
    end = time.time() + seconds
    while time.time() < end:
        ok, frame = cap.read()
        if ok:
            cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(30)


def settle_tracking(cap: cv2.VideoCapture, tracker: Tracker | None, seconds: float) -> None:
    """Wait out a move while KEEPING THE TRACKER RUNNING on live frames.

    The arm shifting the camera is exactly when a tracker loses its target if we
    only feed it the before/after frames -- the object teleports across the view.
    Updating it every frame during the motion lets it follow the object smoothly,
    and the window keeps streaming instead of freezing.
    """
    end = time.time() + seconds
    while time.time() < end:
        ok, frame = cap.read()
        if ok:
            if tracker is not None:
                tracker.update(frame)
            cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(15)


def wait_gripper_settled(arm: RoArm, cap: cv2.VideoCapture, timeout: float = 2.5) -> None:
    """Block until the gripper has ACTUALLY finished moving before the arm moves on.

    Polls the gripper angle (feedback t) until it stops changing -- so we never
    start moving mid-open/mid-close. Keeps the camera feed live while waiting.
    """
    deadline = time.time() + timeout
    last_t = None
    while time.time() < deadline:
        sleep_showing(cap, 0.15)
        fb = arm.feedback()
        if fb is None:
            continue
        if last_t is not None and abs(fb.t - last_t) < 0.01:
            return  # gripper angle stable -> fully open/closed
        last_t = fb.t


def show_preview(
    frame, box: tuple[int, int, int, int] | None, deadband: int, aligned: bool, text: str,
    ref_x: int = 500, ref_y: int = 500,
) -> None:
    h, w = frame.shape[:2]
    cx, cy = int(ref_x / 1000 * w), int(ref_y / 1000 * h)
    color = (0, 220, 0) if aligned else (0, 180, 255)
    db_px = int(deadband / 1000 * min(w, h))
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color, 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color, 2)
    cv2.rectangle(frame, (cx - db_px, cy - db_px), (cx + db_px, cy + db_px), color, 1)
    if box is not None:
        x1, y1, x2, y2 = box
        px1, py1 = int(x1 / 1000 * w), int(y1 / 1000 * h)
        px2, py2 = int(x2 / 1000 * w), int(y2 / 1000 * h)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 80, 80), 2)
        cv2.circle(frame, ((px1 + px2) // 2, (py1 + py2) // 2), 5, (255, 80, 80), -1)
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
    cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
    cv2.imshow(WINDOW_NAME, frame)
    cv2.waitKey(1)


def pixel_box_to_normalized(box: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x, y, bw, bh = box
    return (int(x / w * 1000), int(y / h * 1000), int((x + bw) / w * 1000), int((y + bh) / h * 1000))


def reacquire_with_vlm(frame, target: str) -> tuple[int, int, int, int] | None:
    """One VLM call to (re)find the target. Returns a pixel ROI, or None if not visible."""
    det = locate(FRAME_PATH, target)
    print(f"  VLM: {det.answer!r}")
    if det.box is None:
        return None
    return normalized_box_to_pixel_roi(det.box, frame.shape)


def select_roi_manually(cap: cv2.VideoCapture, frame) -> tuple[int, int, int, int] | None:
    """Live-preview ROI selection: the window keeps streaming until the user
    freezes a frame (SPACE/Enter), then they drag a box on that frame."""
    print("  라이브 화면입니다. SPACE/Enter로 화면을 잡은 뒤 물체에 박스를 드래그하세요 (취소: c)")
    while True:
        ok, live = cap.read()
        if ok:
            frame = live
            disp = live.copy()
            cv2.putText(disp, "SPACE/Enter: freeze & draw box   c: cancel", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3)
            cv2.putText(disp, "SPACE/Enter: freeze & draw box   c: cancel", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
            cv2.imshow(WINDOW_NAME, disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):  # Enter / Space -> freeze the latest frame and draw
            break
        if key == ord("c"):
            return None
    x, y, w, h = (int(v) for v in cv2.selectROI(WINDOW_NAME, frame, fromCenter=False, showCrosshair=True))
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def prompt_for_target(cap: cv2.VideoCapture, voice: bool = False) -> str | None:
    """Conversational prompt: questions are answered by the VLM about the
    CURRENT camera view; a pick command returns the object to go grab.
    Text mode: '?'-suffixed inputs are questions, anything else is the target.
    Voice mode: speak; the VLM routes the utterance to answer-or-pick itself.
    Returns None when the user quits."""
    if voice:
        import speech

        speech.speak("무엇을 집을까요? 질문도 하실 수 있어요.")
        while True:
            heard = speech.listen()
            if not heard:
                continue
            print(f"[나] {heard}")
            if any(w in heard for w in ("종료", "그만해", "끝내", "멈춰")):
                speech.speak("종료합니다.")
                return None
            capture_frame(cap)  # fresh frame -> FRAME_PATH
            try:
                result = route(FRAME_PATH, heard)
            except Exception as e:
                speech.speak("잘 처리하지 못했어요. 다시 말씀해주세요.")
                print(f"(라우팅 실패: {e})")
                continue
            if result.get("action") == "pick" and result.get("target"):
                speech.speak(f"{result['target']} 집을게요.")
                return str(result["target"])
            speech.speak(str(result.get("text", "")))

    print("\n무엇이든 물어보세요 ('?'로 끝내면 질문, 예: 뭐가 보여?)")
    print("물체 설명을 입력하면 그걸 집으러 갑니다. 종료: q")
    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit", "종료"):
            return None
        if raw.endswith("?") or raw.endswith("？"):
            capture_frame(cap)  # fresh frame -> FRAME_PATH
            try:
                print(ask(FRAME_PATH, raw))
            except Exception as e:  # keep the conversation alive on API hiccups
                print(f"(질문 실패: {e})")
            continue
        return raw


def detect_target(cap: cv2.VideoCapture, frame, target: str, detector: str, min_area_frac: float) -> tuple[int, int, int, int] | None:
    """(Re)find the target. `detector` picks the strategy: "vlm", "darkest" (no API calls), or "manual"."""
    if detector == "manual":
        return select_roi_manually(cap, frame)
    if detector == "darkest":
        box = find_darkest_box(frame, min_area_frac=min_area_frac)
        print(f"  darkest-blob: {'found ' + str(box) if box else 'none found'}")
        return box
    return reacquire_with_vlm(frame, target)


def load_floor_z(path: str) -> float | None:
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(data["z"])


def load_camera_hfov(path: str) -> float | None:
    if not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(data["hfov_deg"])


def estimate_world_offset(
    ex: int, ey: int, height_mm: float, frame_shape: tuple[int, ...], hfov_deg: float
) -> tuple[float, float]:
    """Inverse-project a pixel offset onto the floor plane.

    Pinhole approximation: at height `height_mm` above the floor, a pixel at
    the image edge corresponds to a ray at half the camera's field of view.
    ex, ey are pixel offsets from image center in 0-1000 normalized units.
    Returns (lateral_mm, forward_mm) -- still just an estimate (unknown exact
    mount offset/tilt), meant to get close in one smooth move before the
    per-step visual fine-tuning takes over.
    """
    h_img, w_img = frame_shape[:2]
    half_hfov = math.radians(hfov_deg / 2)
    half_vfov = math.atan(math.tan(half_hfov) * (h_img / w_img))
    lateral_mm = height_mm * math.tan(half_hfov) * (ex / 500)
    forward_mm = height_mm * math.tan(half_vfov) * (ey / 500)
    return lateral_mm, forward_mm


def track_and_descend(
    arm: RoArm, cap: cv2.VideoCapture, target: str, floor_z: float | None, args: argparse.Namespace
) -> bool:
    """Look-then-move approach to a top-down target.

    Each iteration is a discrete "look, then move": capture a frame, locate the
    box, geometrically project its image-space offset onto the floor plane to
    get a real mm correction, then issue ONE interpolated move_to (T:104) that
    re-centers x/y and -- only once centered -- steps z down toward the floor.
    The arm fully settles between looks, so there is no continuous direct-control
    servo loop: a bad estimate costs a single clamped step (the target is always
    clamped into the reachable workspace) instead of running the arm into an
    unreachable pose. Once the object is centered and z has reached the floor
    clearance height, we stop and let the caller close the gripper.

    Only two situations call the VLM: the very first lookup, and reacquiring
    after the local tracker loses the object `args.track_lost_retries` times in
    a row -- everything else is local tracking (no API calls).
    """
    pose = arm.feedback() or load_start_pose(args.start_pose)
    target_z = clamp(floor_z + args.floor_clearance_mm, "z") if floor_z is not None else clamp(pose.z - args.pick_down_mm, "z")
    # "darkest" is a plain color/contrast detector -- it's cheap and stateless, so we just
    # re-run it every frame instead of tracking, which sidesteps tracker drift entirely
    # (a learned tracker like ViT-Track is tuned for natural scenes, not a flat-colored
    # object against a uniform background, and can lock onto the wrong thing and drift).
    use_tracker = args.detector != "darkest"
    tracker = Tracker() if use_tracker else None

    frame = capture_frame(cap)
    roi = detect_target(cap, frame, target, args.detector, args.min_area_frac)
    if roi is None:
        print("target not visible at start, aborting")
        return False
    if use_tracker:
        tracker.init(frame, roi)

    lost_streak = 0
    miss_streak = 0

    for step in range(args.max_iters):
        frame = capture_frame(cap)

        if use_tracker:
            # periodically re-anchor the tracker on a fresh VLM detection so drift or a
            # lock-on to the wrong object gets corrected instead of compounding
            reanchor = (
                args.detector == "vlm"
                and args.vlm_reanchor_every > 0
                and step > 0
                and step % args.vlm_reanchor_every == 0
            )
            if reanchor:
                fresh = detect_target(cap, frame, target, args.detector, args.min_area_frac)
                if fresh is not None:
                    tracker.init(frame, fresh)
                    ok, box = True, fresh
                    print(f"[{step}] re-anchored on VLM")
                else:
                    ok, box = tracker.update(frame)
            else:
                ok, box = tracker.update(frame)
            if not ok:
                lost_streak += 1
                print(f"[{step}] tracking lost ({lost_streak}/{args.track_lost_retries})")
                show_preview(frame, None, args.deadband, False, f"[{step}] tracking lost", args.ref_x, args.ref_y)
                if lost_streak < args.track_lost_retries:
                    continue
                roi = detect_target(cap, frame, target, args.detector, args.min_area_frac)
                lost_streak = 0
                if roi is None:
                    miss_streak += 1
                    if miss_streak >= args.vlm_retry_limit:
                        print("repeatedly failed to reacquire target, aborting")
                        return False
                    continue
                miss_streak = 0
                tracker.init(frame, roi)
                continue
        else:
            # stateless per-frame detection, no tracker to lose/reinit
            box = find_darkest_box(frame, min_area_frac=args.min_area_frac)
            if box is None:
                miss_streak += 1
                print(f"[{step}] darkest-blob not found ({miss_streak}/{args.vlm_retry_limit})")
                show_preview(frame, None, args.deadband, False, f"[{step}] not found", args.ref_x, args.ref_y)
                if miss_streak >= args.vlm_retry_limit:
                    print("repeatedly failed to find target, aborting")
                    return False
                continue
            miss_streak = 0

        box_norm = pixel_box_to_normalized(box, frame.shape)
        x1, y1, x2, y2 = box_norm
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # align the object to where the gripper tip actually appears in the eye-in-hand
        # view (args.ref_x/ref_y in 0-1000 space), NOT the image center: the camera sits
        # above/behind the gripper, so the tip shows up low in the frame (lower quarter)
        ex = cx - args.ref_x
        ey = cy - args.ref_y
        centered = abs(ex) <= args.deadband and abs(ey) <= args.deadband

        # sync to where the arm actually ended up (reflects clamping / a settled move)
        pose = arm.feedback() or pose
        # trigger once we've descended TO OR BELOW the grab height (the arm often
        # settles a bit lower than commanded, so a strict abs() window never fires)
        at_grab_height = pose.z <= target_z + 1.0
        status = f"[{step}] err=({ex:+d},{ey:+d}) z={pose.z:.1f}->{target_z:.1f} centered={centered} at_grab={at_grab_height}"
        print(status)
        show_preview(frame, box_norm, args.deadband, centered, status, args.ref_x, args.ref_y)

        if at_grab_height:
            # We're at grab height. Stop servoing -- near the surface a top-down camera
            # can no longer resolve the forward offset -- and lunge a fixed distance
            # forward to cover the residual camera->gripper offset, then let the caller
            # close the gripper. forward_sign encodes which x direction is "forward",
            # so -forward_sign * dist moves toward where a top-of-frame object sits.
            pose.x = clamp(pose.x - args.forward_sign * args.grab_forward_mm, "x")
            print(f"at grab height z={pose.z:.1f}: lunging forward {args.grab_forward_mm:.0f}mm (slow) -> x={pose.x:.1f}")
            arm.move_to(pose, speed=args.grab_speed)
            sleep_showing(cap, max(args.settle_time, 1.0))
            return True

        # project the pixel offset onto the floor plane -> real mm the object sits
        # away from directly under the gripper, at the current height above the floor
        height_mm = max(pose.z - floor_z, 20.0) if floor_z is not None else 150.0
        lateral_mm, forward_mm = estimate_world_offset(ex, ey, height_mm, frame.shape, args.camera_hfov_deg)
        dy = args.lateral_sign * lateral_mm * args.approach_gain
        dx = args.forward_sign * forward_mm * args.approach_gain
        # cap how far we move per look: a big jump shifts the object across the frame
        # faster than the tracker's search window can follow, so it loses the object
        # (worst on the first few large corrections). Move at most max_xy_step_mm and
        # let the next look continue -- keeps the target inside the tracker's window.
        step_mag = math.hypot(dx, dy)
        if step_mag > args.max_xy_step_mm:
            scale = args.max_xy_step_mm / step_mag
            dx *= scale
            dy *= scale
        pose.y = clamp(pose.y + dy, "y")
        pose.x = clamp(pose.x + dx, "x")
        # descend while re-centering, but slow the descent the further off-center the
        # object is. A top-down camera loses forward/lateral resolution as it nears the
        # surface (each mm of reach needs the object right at the image edge), so an
        # object far up in the view must be reached FIRST or we bottom out before we
        # ever get over it. err_frac 0 (centered) -> full step, ~1 (edge) -> 15% step.
        err_frac = min(1.0, max(abs(ex), abs(ey)) / 500.0)
        descend_frac = max(0.15, 1.0 - err_frac)
        if pose.z > target_z:
            pose.z = clamp(max(target_z, pose.z - args.descend_step_mm * descend_frac), "z")

        arm.move_to(pose, speed=args.approach_speed)
        # keep detecting/tracking DURING the move (camera shifts with the arm) so the
        # tracker follows the object through the motion instead of losing it between looks
        settle_tracking(cap, tracker if use_tracker else None, args.settle_time)

    print("gave up approaching target after max_iters")
    return False


def pick(arm: RoArm, cap: cv2.VideoCapture, carry_z: float, grip_settle_sec: float, grip_close_rad: float) -> None:
    """Assumes track_and_descend already put the gripper at the pick height."""
    arm.set_gripper(grip_close_rad)
    # wait until the gripper has actually finished closing (settled on the object)
    # before the arm moves, so a fast run can't lift mid-close and drop the object
    wait_gripper_settled(arm, cap, timeout=3.0)
    sleep_showing(cap, grip_settle_sec)  # brief extra dwell to seat the grip
    pose = arm.feedback()
    # lift straight up to the safe carry height so the object clears others in
    # transit -- keep COMMANDING the close angle: the feedback t is where the
    # fingers stalled on the object, and re-sending it relaxes the grip
    arm.move_to(Pose(pose.x, pose.y, clamp(max(pose.z, carry_z), "z"), grip_close_rad))
    sleep_showing(cap, 1.0)


def place_at(arm: RoArm, cap: cv2.VideoCapture, place_pose: Pose, carry_z: float, grip_close_rad: float) -> None:
    """Carry the held object to the hand-taught place pose, set it down, open the
    gripper, and lift clear.

    move_to's `t` field also drives the gripper, so we keep sending the firm-close
    value while transiting/descending (object stays gripped) and only open once we
    are down at the spot. The final lift is commanded with the gripper OPEN so it
    stays open on the way home.
    """
    high_z = clamp(max(place_pose.z, carry_z), "z")
    # transit high and descend, still gripping the object
    arm.move_to(Pose(place_pose.x, place_pose.y, high_z, grip_close_rad))
    sleep_showing(cap, 1.3)
    arm.move_to(Pose(place_pose.x, place_pose.y, place_pose.z, grip_close_rad))
    sleep_showing(cap, 1.0)
    # set down: release, and make sure the gripper is FULLY open before moving away
    # (otherwise it lifts mid-open and can drag/knock the object it just placed)
    arm.open_gripper()
    wait_gripper_settled(arm, cap, timeout=2.5)
    sleep_showing(cap, 0.3)
    # lift clear with the gripper open
    arm.move_to(Pose(place_pose.x, place_pose.y, high_z, RoArm.GRIPPER_OPEN))
    sleep_showing(cap, 0.8)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RoArm-M2 VLM pick-and-place")
    p.add_argument("--target", default=None, help="natural language description of the object to pick (required for --detector vlm)")
    p.add_argument(
        "--detector",
        choices=("vlm", "darkest", "manual"),
        default="manual",
        help=(
            "'vlm' asks the VLM what/where the target is; 'darkest' finds the largest dark blob against a "
            "light background (no API calls); 'manual' lets you drag a box on the camera window yourself"
        ),
    )
    p.add_argument("--voice", action="store_true", help="talk to the robot: mic (OpenAI whisper) in, spoken replies (OpenAI TTS) out; commands and scene questions are routed automatically")
    p.add_argument("--vlm-backend", choices=("gemini", "openai", "molmo"), default=None, help="which VLM does the locating: gemini (Robotics-ER, precise boxes; default when a Gemini key exists), openai, or molmo (fully local via Ollama, unverified accuracy -- see vision.py docstring)")
    p.add_argument("--min-area-frac", type=float, default=0.01, help="darkest detector: ignore blobs smaller than this fraction of the frame area")
    p.add_argument("--ip", default=os.getenv("ROARM_IP", "192.168.4.1"))
    p.add_argument("--serial-port", default=os.getenv("ROARM_SERIAL_PORT", "/dev/cu.usbserial-120"))
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--start-pose", default="roarm_start_pose.json")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--record", dest="record", action="store_true", default=True, help="save the full camera feed of each run to recordings/run_<timestamp>.mp4 (on by default)")
    p.add_argument("--no-record", dest="record", action="store_false")
    p.add_argument("--place-pose", default="place_pose.json", help="hand-taught place location from teach_place.py; if present, every picked object is dropped here")
    p.add_argument("--place-x", type=float, default=None)
    p.add_argument("--place-y", type=float, default=None)
    p.add_argument("--place-z", type=float, default=None)
    p.add_argument("--max-iters", type=int, default=80)
    p.add_argument("--max-cycles", type=int, default=0, help="how many pick-place cycles to run before stopping (0 = keep going until no object is found or Ctrl+C); requires a taught place pose")
    p.add_argument("--deadband", type=int, default=35, help="pixel deadband in 0-1000 space, informational only (shown as 'aligned')")
    p.add_argument("--ref-x", type=int, default=500, help="horizontal alignment target in 0-1000 image space (500 = center)")
    p.add_argument("--ref-y", type=int, default=750, help="vertical alignment target in 0-1000 image space: where the gripper tip appears in the eye-in-hand view (750 = lower quarter, camera sits above/behind the gripper)")
    p.add_argument("--track-lost-retries", type=int, default=3, help="consecutive tracker-lost frames before re-calling the VLM")
    p.add_argument("--vlm-reanchor-every", type=int, default=5, help="vlm mode: re-run the VLM every N steps to re-anchor the tracker and correct drift (0 disables)")
    p.add_argument("--vlm-retry-limit", type=int, default=3, help="consecutive failed VLM reacquire attempts before giving up")
    p.add_argument("--approach-speed", type=float, default=0.9, help="arm speed (0-1) for each interpolated look-then-move step")
    p.add_argument("--approach-gain", type=float, default=1.0, help="fraction of the projected mm offset to correct per look (<1 damps overshoot)")
    p.add_argument("--max-xy-step-mm", type=float, default=25.0, help="max x/y move per look; caps big jumps so the tracker doesn't lose the object when it shifts far across the frame")
    p.add_argument("--settle-time", type=float, default=0.1, help="seconds to let the arm settle after each move before looking again")
    p.add_argument("--descend-step-mm", type=float, default=30.0, help="z step per iteration while descending to the floor")
    p.add_argument("--grab-forward-mm", type=float, default=60.0, help="once at grab height, lunge this far forward (mm) to cover the camera->gripper offset before closing the gripper")
    p.add_argument("--grab-speed", type=float, default=0.1, help="arm speed (0-1) for the slow final forward lunge into the object")
    p.add_argument("--lateral-sign", type=float, default=-1.0)
    p.add_argument("--forward-sign", type=float, default=-1.0)
    p.add_argument("--pick-down-mm", type=float, default=25.0, help="used only if --floor-calibration file is missing")
    p.add_argument("--carry-z", type=float, default=None, help="absolute z height to lift to and travel at while carrying the object, so it clears other objects (default: the start pose height)")
    p.add_argument("--grip-settle-sec", type=float, default=0.5, help="wait this long for the gripper to fully close and grip before the arm lifts")
    p.add_argument("--grip-close-rad", type=float, default=3.25, help="how far to close the gripper when grabbing (1.57=open, 3.14=nominal closed, up to 3.4=firm)")
    p.add_argument("--place-down-mm", type=float, default=25.0, help="used only if --floor-calibration file is missing")
    p.add_argument("--floor-calibration", default="floor_calibration.json")
    p.add_argument("--workspace-calibration", default="workspace_calibration.json", help="output of calibrate_workspace.py")
    p.add_argument("--floor-clearance-mm", type=float, default=5.0, help="gripper stops this far above the calibrated floor/table")
    p.add_argument("--camera-hfov-deg", type=float, default=60.0, help="camera horizontal field of view; overridden by --camera-calibration if that file exists")
    p.add_argument("--camera-calibration", default="camera_calibration.json", help="output of calibrate_camera_fov.py")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.vlm_backend:
        os.environ["VLM_BACKEND"] = args.vlm_backend

    arm = RoArm(ip=args.ip, serial_port=args.serial_port, baudrate=args.baudrate)
    start_pose = load_start_pose(args.start_pose)

    # prefer the hand-taught place location (teach_place.py); fall back to explicit
    # --place-x/y/z if given; otherwise None -> just hold the object at the start pose
    place_pose = load_place_pose(args.place_pose)
    if place_pose is not None:
        print(f"place pose loaded (from {args.place_pose}): x={place_pose.x:.1f} y={place_pose.y:.1f} z={place_pose.z:.1f}")
    elif any(v is not None for v in (args.place_x, args.place_y, args.place_z)):
        place_pose = Pose(
            x=args.place_x if args.place_x is not None else start_pose.x,
            y=args.place_y if args.place_y is not None else start_pose.y + 100.0,
            z=args.place_z if args.place_z is not None else start_pose.z,
            t=RoArm.GRIPPER_OPEN,
        )
    else:
        print("no taught place pose found; will hold the object at the start pose (run teach_place.py to set one)")

    floor_z = load_floor_z(args.floor_calibration)
    if floor_z is not None:
        print(f"floor calibration loaded: z={floor_z:.1f} (from {args.floor_calibration})")
    else:
        print(f"no floor calibration found at {args.floor_calibration}, using --pick-down-mm/--place-down-mm instead")

    calibrated_hfov = load_camera_hfov(args.camera_calibration)
    if calibrated_hfov is not None:
        args.camera_hfov_deg = calibrated_hfov
        print(f"camera FOV calibration loaded: hfov={calibrated_hfov:.1f} deg (from {args.camera_calibration})")
    else:
        print(f"no camera FOV calibration found at {args.camera_calibration}, using --camera-hfov-deg={args.camera_hfov_deg}")

    # widen z (and, as a fallback, x/y) around the poses we actually plan to visit
    expand_workspace(start_pose)
    if place_pose is not None:
        expand_workspace(place_pose)

    measured_workspace = load_workspace_calibration(args.workspace_calibration)
    if measured_workspace is not None:
        WORKSPACE["x"] = measured_workspace["x"]
        WORKSPACE["y"] = measured_workspace["y"]
        print(f"workspace calibration loaded (from {args.workspace_calibration}), overriding guessed x/y margins")
    else:
        print(f"no workspace calibration found at {args.workspace_calibration}, using guessed x/y margins instead")

    if floor_z is not None:
        WORKSPACE["z"] = (min(WORKSPACE["z"][0], floor_z - 10.0), WORKSPACE["z"][1])
    print(f"workspace: x={WORKSPACE['x']}, y={WORKSPACE['y']}, z={WORKSPACE['z']}")

    carry_z = clamp(args.carry_z if args.carry_z is not None else start_pose.z, "z")
    print(f"carry height (transit z): {carry_z:.1f}")

    cap = RecordingCapture(args.camera_index) if args.record else cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera index {args.camera_index}")
    if args.record and cap.path:
        print(f"카메라 영상 녹화 중: {cap.path}")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        cycle = 0
        while True:
            cycle += 1
            print(f"===== cycle {cycle} =====")
            print("moving to start pose")
            arm.move_to(start_pose)
            sleep_showing(cap, 1.0)
            arm.open_gripper()
            wait_gripper_settled(arm, cap, timeout=2.0)

            # per-cycle conversational target: with no --target, chat with the VLM
            # about the scene and/or name the next object to grab
            cycle_target = args.target
            if args.detector == "vlm" and not cycle_target:
                cycle_target = prompt_for_target(cap, voice=args.voice)
                if cycle_target is None:
                    print("종료합니다")
                    break

            if not track_and_descend(arm, cap, cycle_target, floor_z, args):
                if cycle == 1:
                    print("failed to locate/descend to target, aborting")
                    return 1
                print("no more objects found; stopping loop")
                break

            print("picking")
            pick(arm, cap, carry_z, args.grip_settle_sec, args.grip_close_rad)

            if place_pose is None:
                # nothing taught -> hold the object at the start pose; can't loop
                # (the next cycle would open the gripper and drop it), so do one cycle
                print("returning to start pose, holding the object (no place pose -> single cycle)")
                arm.move_to(start_pose)
                sleep_showing(cap, 1.5)
                arm.close_gripper()  # make sure the grip is held, not opened by the move
                sleep_showing(cap, 0.3)
                break

            print("placing at taught pose")
            place_at(arm, cap, place_pose, carry_z, args.grip_close_rad)
            if args.voice:
                import speech

                speech.speak("옮겼어요.")
            print("returning to start pose (gripper open)")
            arm.move_to(Pose(start_pose.x, start_pose.y, start_pose.z, RoArm.GRIPPER_OPEN))
            sleep_showing(cap, 1.0)

            if args.max_cycles and cycle >= args.max_cycles:
                print(f"reached max cycles ({args.max_cycles}); stopping")
                break
    except KeyboardInterrupt:
        print("\ninterrupted, stopping")
    finally:
        cap.release()
        cv2.destroyWindow(WINDOW_NAME)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
