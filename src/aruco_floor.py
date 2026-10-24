"""Floor-plane homography from ArUco markers -- re-solved on every frame, so
the top-down camera may be moved, bumped or hand-held.

Setup (once): print the markers (make_aruco_markers.py), tape them around
the OUTSIDE of the work area, run calibrate_aruco_floor.py to record where
each marker's centre is in robot coordinates. That writes aruco_floor.json:

  {"dictionary": "DICT_4X4_50", "marker_size_mm": 50,
   "markers": {"0": {"center": [x, y], "corners": [[x, y] x4]}, ...}}

Runtime: ArucoFloor.update(frame) detects whatever markers are visible,
pairs their pixel corners with the stored world corners (4 points per
marker, so even 2 visible markers give 8 correspondences) and solves the
pixel -> robot (x, y) homography with RANSAC. `H` is updated IN PLACE, so
any code holding a reference to the same array (TopdownHUD, pixel_to_world
callers) sees the new mapping without being told. When too few markers are
visible the last good H is kept and `stale` turns True.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PATH = "aruco_floor.json"
MIN_MARKERS = 2  # 2 markers = 8 corner correspondences; 1 marker is too shaky


def get_dictionary(name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_detector(dict_name: str = "DICT_4X4_50"):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(get_dictionary(dict_name), params)


# ids seen more than once in the last detection (e.g. a spare printed sheet
# lying in view): ambiguous, so they are dropped -- and reported here
last_duplicates: list[int] = []


def _detect_raw(detector, gray: np.ndarray) -> dict[int, np.ndarray]:
    global last_duplicates
    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        last_duplicates = []
        return {}
    out: dict[int, np.ndarray] = {}
    dups: set[int] = set()
    for i, c in zip(ids.flatten(), corners):
        i = int(i)
        if i in out:
            dups.add(i)
        out[i] = c.reshape(4, 2).copy()
    for i in dups:
        out.pop(i, None)
    last_duplicates = sorted(dups)
    return out


def detect(detector, frame_bgr: np.ndarray, mirror: bool | None = None) -> dict[int, np.ndarray]:
    """{id: 4x2 float32 pixel corners} for every marker in the image.

    mirror: some phone-webcam apps (Iriun front camera) deliver a horizontally
    MIRRORED feed, in which ArUco patterns do not decode (only near-symmetric
    ids slip through). True = decode on the flipped image and map the corners
    back to the original pixel grid; None = try both and keep whichever finds
    more markers. The corner order per marker stays consistent between
    calibration and runtime because both go through this function."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    found = {} if mirror is True else _detect_raw(detector, gray)
    if mirror is False:
        return found
    flipped = _detect_raw(detector, cv2.flip(gray, 1))
    w = gray.shape[1]
    for c in flipped.values():
        c[:, 0] = (w - 1) - c[:, 0]
    if mirror is True or len(flipped) > len(found):
        return flipped
    return found


def guess_mirror(detector, frame_bgr: np.ndarray) -> bool | None:
    """True if the feed looks mirrored (more markers decode flipped), False if
    not, None if nothing decodes either way."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    n_plain = len(_detect_raw(detector, gray))
    n_flip = len(_detect_raw(detector, cv2.flip(gray, 1)))
    if n_plain == 0 and n_flip == 0:
        return None
    return n_flip > n_plain


class MirrorFix:
    """Un-mirror a webcam feed at the source. Some phone-webcam apps (Iriun)
    deliver a horizontally flipped image; shown raw it misleads people (left
    is right, printed ids unreadable, VLM left/right answers wrong). Decides
    once from the markers (needs >= 1 decodable marker) and then flips every
    frame so the whole pipeline sees a true image."""

    def __init__(self, detector=None):
        self.detector = detector or make_detector()
        self.mirror: bool | None = None

    def fix(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.mirror is None:
            self.mirror = guess_mirror(self.detector, frame_bgr)
            if self.mirror:
                print("(camera feed is MIRRORED -- flipping every frame so the image is true)")
        return cv2.flip(frame_bgr, 1) if self.mirror else frame_bgr


class FrameOrienter:
    """Make every frame TRUE and ROBOT-ALIGNED regardless of how the camera
    is mounted or whether the app mirrors it: after fix(), robot +x (forward)
    points UP on screen and robot +y (left) points LEFT.

    Mirror is decided by handedness: the pixel->robot homography of a real
    (unmirrored) top-down image has a negative Jacobian determinant (image y
    runs down, robot frame is right-handed seen from above); a mirrored feed
    flips the sign. Rotation (0/90/180/270) is then chosen so robot +x points
    up. Both are decided once from the first frames that show >= 2 taught
    markers and kept for the session (a frame size change mid-run would break
    the recording), so re-run if you turn the camera by 90 degrees."""

    ROTS = {0: None, 1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}

    def __init__(self, floor: "ArucoFloor", view_rotate: int | None = None):
        self.floor = floor
        self.mirror: bool | None = None
        self.rot_k: int | None = None  # number of 90-degree clockwise turns applied after the flip
        # extra turn on top of robot-aligned, for the human's preferred view
        # (0/90/180/270; from the calibration file unless given)
        deg = floor.view_rotate if view_rotate is None else view_rotate
        self.view_k = (int(deg) // 90) % 4

    @property
    def ready(self) -> bool:
        return self.mirror is not None and self.rot_k is not None

    @staticmethod
    def _solve(floor: "ArucoFloor", frame_bgr: np.ndarray, mirror: bool) -> np.ndarray | None:
        found = detect(floor.detector, frame_bgr, mirror)
        vis = {i: c for i, c in found.items() if i in floor.world_corners}
        if len(vis) < MIN_MARKERS:
            return None
        src = np.concatenate([vis[i] for i in sorted(vis)]).astype(np.float64)
        dst = np.concatenate([floor.world_corners[i] for i in sorted(vis)])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 15.0)
        return H if H is not None and inl is not None and int(inl.sum()) >= 6 else None

    def calibrate(self, raw_bgr: np.ndarray) -> bool:
        """Decide mirror + rotation from one raw frame. False if not enough markers yet."""
        guess = guess_mirror(self.floor.detector, raw_bgr)
        H = self._solve(self.floor, raw_bgr, bool(guess))
        if H is None:
            return False
        # Jacobian of pixel->robot at the image centre
        h, w = raw_bgr.shape[:2]

        def p2w(px, py):
            v = H @ np.array([px, py, 1.0])
            return np.array([v[0] / v[2], v[1] / v[2]])

        c = p2w(w / 2, h / 2)
        jx = p2w(w / 2 + 10, h / 2) - c  # robot delta for image +x
        jy = p2w(w / 2, h / 2 + 10) - c  # robot delta for image +y (down)
        det = jx[0] * jy[1] - jx[1] * jy[0]
        # a true image has det < 0. detect() returned corners in RAW pixel coords,
        # so H describes the raw image: det > 0 means the raw image is mirrored.
        self.mirror = det > 0
        if self.mirror != bool(guess):
            print(f"(orient: handedness says mirrored={self.mirror}, marker decoding said {guess} -- trusting handedness)")
        # direction of robot +x in the raw image, then after the flip
        Hinv = np.linalg.inv(H)

        def w2p(wx, wy):
            v = Hinv @ np.array([wx, wy, 1.0])
            return np.array([v[0] / v[2], v[1] / v[2]])

        d = w2p(c[0] + 50, c[1]) - w2p(c[0], c[1])
        if self.mirror:
            d[0] = -d[0]
        # choose k (90-degree CW turns) so that d points up (0, -1). CW: (dx, dy) -> (-dy, dx)
        best, best_k = -1e9, 0
        for k in range(4):
            v = d.copy()
            for _ in range(k):
                v = np.array([-v[1], v[0]])
            score = -v[1] / (np.linalg.norm(v) + 1e-9)
            if score > best:
                best, best_k = score, k
        self.rot_k = best_k
        print(f"(orient: mirror={self.mirror}, rotate {90 * self.rot_k} deg -> robot +x up, +y left)")
        return True

    def fix(self, raw_bgr: np.ndarray) -> np.ndarray:
        if not self.ready and not self.calibrate(raw_bgr):
            return raw_bgr  # pass through until markers are seen
        f = cv2.flip(raw_bgr, 1) if self.mirror else raw_bgr
        rot = self.ROTS[(self.rot_k + self.view_k) % 4]
        return cv2.rotate(f, rot) if rot is not None else f


class ArucoFloor:
    def __init__(self, path: str = DEFAULT_PATH, H: np.ndarray | None = None):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.path = path
        self.dict_name = data.get("dictionary", "DICT_4X4_50")
        self.marker_size_mm = float(data.get("marker_size_mm", 50.0))
        self.view_rotate = int(data.get("view_rotate", 0))  # extra display turn chosen by the user (see FrameOrienter)
        self.world_corners: dict[int, np.ndarray] = {
            int(k): np.array(v["corners"], dtype=np.float64) for k, v in data["markers"].items()
        }
        self.world_centers: dict[int, tuple[float, float]] = {
            int(k): (float(v["center"][0]), float(v["center"][1])) for k, v in data["markers"].items()
        }
        self.detector = make_detector(self.dict_name)
        self.mirror: bool | None = None  # decided from the first frame that decodes (see detect())
        # H is shared by reference: callers may hand in the array they already use
        self.H = H if H is not None else np.eye(3, dtype=np.float64)
        self.valid = H is not None
        self.stale = True
        self.visible: dict[int, np.ndarray] = {}
        self.unknown_ids: list[int] = []
        self.duplicate_ids: list[int] = []
        self.rms_px = float("nan")
        self.last_good_time: float | None = None
        self.n_updates = 0

    # ---- per frame -----------------------------------------------------
    def update(self, frame_bgr: np.ndarray) -> bool:
        """Detect markers and re-solve H. True if H was refreshed this frame."""
        if self.mirror is None:
            self.mirror = guess_mirror(self.detector, frame_bgr)
            if self.mirror:
                print("(aruco: camera feed is MIRRORED -- decoding on the flipped image)")
        found = detect(self.detector, frame_bgr, self.mirror)
        self.visible = {i: c for i, c in found.items() if i in self.world_corners}
        self.unknown_ids = sorted(i for i in found if i not in self.world_corners)
        self.duplicate_ids = list(last_duplicates)
        if len(self.visible) < MIN_MARKERS:
            self.stale = True
            return False
        src = np.concatenate([self.visible[i] for i in sorted(self.visible)]).astype(np.float64)
        dst = np.concatenate([self.world_corners[i] for i in sorted(self.visible)])
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 15.0)  # 15 mm reprojection gate
        if H is None or inliers is None or int(inliers.sum()) < 6:
            self.stale = True
            return False
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        self.rms_px = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))  # in mm, despite the name
        self.H[...] = H  # in place: everyone sharing this array sees it
        self.valid = True
        self.stale = False
        self.last_good_time = time.time()
        self.n_updates += 1
        return True

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        v = self.H @ np.array([px, py, 1.0])
        return float(v[0] / v[2]), float(v[1] / v[2])

    def world_to_pixel(self, wx: float, wy: float) -> tuple[int, int]:
        v = np.linalg.inv(self.H) @ np.array([wx, wy, 1.0])
        return int(round(v[0] / v[2])), int(round(v[1] / v[2]))

    # ---- overlay -------------------------------------------------------
    def draw(self, frame: np.ndarray, grid_spacing_mm: float = 50.0) -> None:
        """Marker outlines + ids, a robot-frame grid over the marker area and
        a status line. Green = fresh, orange = holding the last good H."""
        for i, c in self.visible.items():
            pts = c.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True, (0, 220, 0), 2, cv2.LINE_AA)
            cx, cy = c.mean(axis=0).astype(int)
            cv2.putText(frame, str(i), (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, str(i), (cx - 8, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if self.valid:
            try:
                H_inv = np.linalg.inv(self.H)
            except np.linalg.LinAlgError:
                H_inv = None
            if H_inv is not None and self.world_centers:
                xs = [p[0] for p in self.world_centers.values()]
                ys = [p[1] for p in self.world_centers.values()]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                color = (0, 200, 255) if not self.stale else (0, 140, 255)

                def w2p(wx, wy):
                    v = H_inv @ np.array([wx, wy, 1.0])
                    return int(round(v[0] / v[2])), int(round(v[1] / v[2]))

                overlay = frame.copy()
                for wx in np.arange(np.floor(x0 / grid_spacing_mm) * grid_spacing_mm, x1 + 1e-6, grid_spacing_mm):
                    cv2.line(overlay, w2p(wx, y0), w2p(wx, y1), color, 1, cv2.LINE_AA)
                for wy in np.arange(np.floor(y0 / grid_spacing_mm) * grid_spacing_mm, y1 + 1e-6, grid_spacing_mm):
                    cv2.line(overlay, w2p(x0, wy), w2p(x1, wy), color, 1, cv2.LINE_AA)
                # robot origin / axes if they fall in frame
                ox, oy = w2p(0, 0)
                h, w = frame.shape[:2]
                if 0 <= ox < w and 0 <= oy < h:
                    cv2.circle(overlay, (ox, oy), 6, (255, 80, 80), -1)
                    cv2.arrowedLine(overlay, (ox, oy), w2p(100, 0), (0, 0, 255), 2, tipLength=0.2)
                    cv2.arrowedLine(overlay, (ox, oy), w2p(0, 100), (0, 255, 0), 2, tipLength=0.2)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)
        status = (f"aruco: {len(self.visible)} markers  rms {self.rms_px:.1f}mm"
                  if not self.stale else
                  f"aruco: {len(self.visible)} markers -- HOLDING last H" if self.valid else
                  f"aruco: {len(self.visible)} markers -- NO homography yet")
        if self.unknown_ids:
            status += f"  (untaught ids: {self.unknown_ids})"
        if self.duplicate_ids:
            status += f"  DUPLICATE ids {self.duplicate_ids} ignored -- remove the spare sheet!"
        color = (0, 220, 0) if not self.stale else (0, 140, 255) if self.valid else (0, 0, 255)
        cv2.putText(frame, status, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, status, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def save_view_rotate(path: str, degrees: int) -> None:
    """Persist the preferred extra view rotation (0/90/180/270) in the calibration file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["view_rotate"] = int(degrees) % 360
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
