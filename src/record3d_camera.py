"""Thin wrapper around the Record3D iOS app's RGBD USB stream.

Lets an iPad/iPhone with a LiDAR (or TrueDepth) sensor act as a depth camera
for this project: every frame gives not just a color image but a per-pixel
distance in meters, so a detected object's real 3D position can be read
directly instead of assumed to lie on a flat calibrated floor plane.

Setup:
  1. pip install record3d --break-system-packages   (or without the flag on your own machine)
  2. Connect the iPad to the Mac via USB (or its own USB-C cable).
  3. Open the Record3D app -> Settings tab -> enable "USB Streaming mode".
  4. Leave the Record3D app in the foreground/streaming while you run scripts
     that use this wrapper.

Coordinate convention returned by get_frame():
  depth_m[y, x]  -- distance in meters along the camera's forward (Z) axis
                    for pixel (x, y) of the (already depth-aligned) rgb image
  intrinsics     -- 3x3 camera matrix [[fx,0,cx],[0,fy,cy],[0,0,1]] for the
                    returned rgb/depth resolution

pixel_depth_to_camera_point() turns a pixel + its depth into a 3D point in
the camera's own coordinate frame (meters, camera at the origin, Z forward).
That point is then meant to be converted into the robot's coordinate frame
via the rigid transform produced by calibrate_lidar_extrinsic.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event

import cv2
import numpy as np


@dataclass
class Frame:
    rgb: np.ndarray  # HxWx3 BGR (OpenCV convention), already flipped/aligned
    depth_m: np.ndarray  # HxW float32, meters, same resolution as rgb
    confidence: np.ndarray | None  # HxW, higher = more reliable, or None
    intrinsics: np.ndarray  # 3x3


class Record3DCamera:
    """Blocking, synchronous-feeling wrapper: call get_frame() whenever you
    want the latest frame; internally waits for the streaming thread's
    callback. Mirrors the RGBD info shown in marek-simonik/record3d's
    demo-main.py, but packaged so callers don't need to touch the raw
    Record3DStream/session API directly."""

    DEVICE_TYPE__TRUEDEPTH = 0
    DEVICE_TYPE__LIDAR = 1

    def __init__(self, device_index: int = 0):
        from record3d import Record3DStream

        self._event = Event()
        self._session = Record3DStream()
        self._session.on_new_frame = self._on_new_frame
        self._session.on_stream_stopped = self._on_stream_stopped

        devices = Record3DStream.get_connected_devices()
        if len(devices) <= device_index:
            raise RuntimeError(
                f"Record3D device #{device_index} not found ({len(devices)} device(s) connected). "
                "Connect the iPad via USB and enable 'USB Streaming mode' in the Record3D app."
            )
        dev = devices[device_index]
        print(f"Record3D: connecting to device UDID={dev.udid}")
        self._session.connect(dev)

    def _on_new_frame(self) -> None:
        # called from a non-main thread; just signal, do the real work in get_frame()
        self._event.set()

    def _on_stream_stopped(self) -> None:
        print("Record3D: stream stopped")

    @staticmethod
    def _intrinsic_matrix(coeffs) -> np.ndarray:
        return np.array([
            [coeffs.fx, 0, coeffs.tx],
            [0, coeffs.fy, coeffs.ty],
            [0, 0, 1],
        ], dtype=np.float64)

    def get_frame(self, timeout: float = 5.0) -> Frame:
        """Block until the next frame arrives (or timeout), return it."""
        if not self._event.wait(timeout=timeout):
            raise TimeoutError("timed out waiting for a Record3D frame -- is the app streaming?")

        depth = self._session.get_depth_frame()
        rgb = self._session.get_rgb_frame()
        try:
            confidence = self._session.get_confidence_frame()
        except Exception:
            confidence = None
        intrinsics = self._intrinsic_matrix(self._session.get_intrinsic_mat())

        if self._session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
            depth = cv2.flip(depth, 1)
            rgb = cv2.flip(rgb, 1)

        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_m = depth.astype(np.float32)

        # The depth stream can come back at a LOWER resolution than the RGB
        # stream (e.g. LiDAR depth ~256x192 vs RGB ~1920x1440) even though
        # both cover the same field of view. Upscale depth to match RGB
        # (nearest-neighbor: depth is piecewise-smooth, no need to blend) so
        # every RGB pixel coordinate can be used directly to index depth_m.
        # The intrinsics returned by the SDK correspond to this depth
        # resolution, so we scale fx/fy/cx/cy to match the upscaled size.
        if depth_m.shape[:2] != rgb_bgr.shape[:2]:
            dh, dw = depth_m.shape[:2]
            rh, rw = rgb_bgr.shape[:2]
            scale_x = rw / dw
            scale_y = rh / dh
            depth_m = cv2.resize(depth_m, (rw, rh), interpolation=cv2.INTER_NEAREST)
            intrinsics = intrinsics.copy()
            intrinsics[0, 0] *= scale_x  # fx
            intrinsics[1, 1] *= scale_y  # fy
            intrinsics[0, 2] *= scale_x  # cx
            intrinsics[1, 2] *= scale_y  # cy

        self._event.clear()
        return Frame(rgb=rgb_bgr, depth_m=depth_m, confidence=confidence, intrinsics=intrinsics)

    def close(self) -> None:
        try:
            self._session.disconnect()
        except Exception:
            pass


def pixel_depth_to_camera_point(px: float, py: float, depth_m: float, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project a pixel + its depth (meters) into a 3D point in the
    camera's own coordinate frame (meters, camera at origin, +Z forward)."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x = (px - cx) * depth_m / fx
    y = (py - cy) * depth_m / fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float64)


ROTATE_CODES = {
    "none": None,
    "left": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "right": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
}


def rotate_for_display(img: np.ndarray, rotate: str) -> np.ndarray:
    """Rotate a frame for display only. 'left' (default used throughout this
    project) matches an iPad held upright in portrait -- the raw Record3D
    frame comes out sideways, so rotating 90deg counter-clockwise puts it
    right-side up the way the person is actually holding/viewing the device."""
    code = ROTATE_CODES[rotate]
    return img if code is None else cv2.rotate(img, code)


def displayed_to_original_xy(dx: int, dy: int, orig_w: int, orig_h: int, rotate: str) -> tuple[int, int]:
    """Map a pixel clicked on the ROTATED display image back to (px, py) in
    the original (unrotated) frame, so depth/intrinsics (defined in the
    original frame) can be looked up correctly."""
    if rotate == "none":
        return dx, dy
    if rotate == "left":  # display = original rotated 90deg CCW
        return orig_w - 1 - dy, dx
    if rotate == "right":  # display = original rotated 90deg CW
        return dy, orig_h - 1 - dx
    if rotate == "180":
        return orig_w - 1 - dx, orig_h - 1 - dy
    raise ValueError(rotate)


def median_depth_around(depth_m: np.ndarray, px: int, py: int, radius: int = 3) -> float | None:
    """Robust depth read: median over a small window around (px, py), ignoring
    zero/invalid samples. A single-pixel depth read on LiDAR data is noisy
    and can hit a dropout (0.0); a small neighborhood is much more stable."""
    h, w = depth_m.shape[:2]
    x0, x1 = max(0, px - radius), min(w, px + radius + 1)
    y0, y1 = max(0, py - radius), min(h, py + radius + 1)
    patch = depth_m[y0:y1, x0:x1]
    valid = patch[(patch > 0.05) & (patch < 5.0) & np.isfinite(patch)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def average_frame(cam: "Record3DCamera", n: int = 5) -> Frame:
    """Grab n frames and return one with a temporally-denoised depth map
    (per-pixel median across frames, ignoring invalid samples). LiDAR depth
    has frame-to-frame jitter and occasional dropouts on a single read;
    holding still for a handful of frames and taking the median per pixel
    is a big accuracy win over trusting one frame. Uses the LAST frame's
    rgb/intrinsics (assumes the camera didn't move meaningfully during the
    n grabs) with the denoised depth swapped in."""
    n = max(1, n)
    frames = [cam.get_frame() for _ in range(n)]
    last = frames[-1]
    if n == 1:
        return last

    stack = np.stack([f.depth_m for f in frames], axis=0)  # (n, H, W)
    valid = (stack > 0.05) & (stack < 5.0) & np.isfinite(stack)
    stack_masked = np.where(valid, stack, np.nan)
    with np.errstate(all="ignore"):
        denoised = np.nanmedian(stack_masked, axis=0)
    denoised = np.nan_to_num(denoised, nan=0.0).astype(np.float32)

    return Frame(rgb=last.rgb, depth_m=denoised, confidence=last.confidence, intrinsics=last.intrinsics)


def estimate_object_color_mask(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    color_tolerance: float = 28.0,
    ref_color_lab: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment the object out of its detection box using COLOR alone (no
    depth), so the box itself can be tightened to the object's actual shape
    instead of being trusted as-is.

    Takes the median Lab color of a central sample region of the box as the
    object's reference color (unless ref_color_lab is given), then masks
    every pixel in the box within color_tolerance of that reference in Lab
    space. Lab is used because Euclidean distance in it tracks perceived
    color difference much better than raw BGR.

    Returns (mask, ref_color_lab): mask is a boolean array shaped like the
    box (y2-y1, x2-x1), True where the pixel is judged to belong to the
    object; ref_color_lab is the 3-vector reference color used (useful to
    reuse across frames of the same object).
    """
    x1, y1, x2, y2 = box
    h, w = rgb.shape[:2]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    rgb_patch = rgb[y1:y2, x1:x2]
    lab_patch = cv2.cvtColor(rgb_patch, cv2.COLOR_BGR2LAB).astype(np.float32)

    if ref_color_lab is None:
        ph, pw = lab_patch.shape[:2]
        # sample a central region (avoids edges of a loose box, which are
        # more likely to be background) to seed the reference color
        my0, my1 = int(ph * 0.3), int(ph * 0.7) + 1
        mx0, mx1 = int(pw * 0.3), int(pw * 0.7) + 1
        center_sample = lab_patch[max(0, my0):max(1, my1), max(0, mx0):max(1, mx1)]
        ref_color_lab = np.median(center_sample.reshape(-1, 3), axis=0)

    color_dist = np.linalg.norm(lab_patch - ref_color_lab[None, None, :], axis=2)
    mask = color_dist <= color_tolerance
    return mask, ref_color_lab


def refine_box_with_color(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    color_tolerance: float = 28.0,
    min_mask_frac: float = 0.05,
) -> tuple[int, int, int, int]:
    """Tighten a (possibly loose) detection box to the actual extent of the
    object's color mask inside it. Falls back to the original box if the
    mask is too small/empty (e.g. object color too close to background --
    color segmentation isn't reliable there, so don't risk shrinking to
    nothing)."""
    x1, y1, x2, y2 = box
    mask, _ = estimate_object_color_mask(rgb, box, color_tolerance=color_tolerance)
    if mask.sum() < min_mask_frac * mask.size:
        return box
    ys, xs = np.where(mask)
    return x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1


def estimate_object_surface_point(
    depth_m: np.ndarray,
    box: tuple[int, int, int, int],
    intrinsics: np.ndarray,
    near_percentile: float = 15.0,
    rgb: np.ndarray | None = None,
    color_tolerance: float = 28.0,
    color_weight: bool = True,
) -> np.ndarray | None:
    """Estimate the (x, y, z) of an object's TOP surface from all the depth
    samples inside its detected box, instead of trusting one center pixel --
    and, when rgb is given, using COLOR as the PRIMARY signal for which
    pixels belong to the object, with depth only used to pick its near
    (top) surface among those color-matched pixels.

    RGB-first pass (when rgb is provided, color_weight=True): the object's
    color mask (see estimate_object_color_mask) is computed over the whole
    box first. Only pixels inside that color mask are considered at all --
    background of a different color is excluded before depth ever comes
    into it. Depth is then used only to pick the near/top surface among
    those already-color-selected pixels (still useful since an object has
    some depth range across its visible surface). This lets RGB carry most
    of the "which pixels are the object" decision, with the LiDAR mainly
    supplying the z (and refining which of the object's own pixels are its
    top-facing surface vs. its side/shadow).

    Depth-only fallback (rgb is None, or color mask is too small, or
    color_weight=False): the object's visible surface is whatever's closest
    to the camera within the box. Taking the near_percentile-th percentile
    of valid depths in the box picks out that near surface robustly.

    The resulting world point is the CENTROID (mean x/y/z) of the surviving
    samples, weighted evenly -- this centers the point on the visible
    surface patch rather than on a single pixel that might be a dropout or
    an edge artifact.

    Returns None if there aren't enough valid samples in the box.
    """
    x1, y1, x2, y2 = box
    h, w = depth_m.shape[:2]
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    patch = depth_m[y1:y2, x1:x2]
    ys, xs = np.mgrid[y1:y2, x1:x2]
    valid_mask = (patch > 0.05) & (patch < 5.0) & np.isfinite(patch)
    if valid_mask.sum() < 5:
        return None

    # RGB-first: restrict to the object's own color before touching depth
    color_selected = None
    if color_weight and rgb is not None and rgb.shape[:2] == depth_m.shape[:2]:
        color_mask, _ = estimate_object_color_mask(rgb, (x1, y1, x2, y2), color_tolerance=color_tolerance)
        candidate = valid_mask & color_mask
        if candidate.sum() >= 5:
            color_selected = candidate

    base_mask = color_selected if color_selected is not None else valid_mask
    base_depths = patch[base_mask]

    near_cutoff = np.percentile(base_depths, near_percentile)
    # a small tolerance band around the near cutoff -- take samples AT the
    # near surface, not just the single closest pixel
    band = max(0.01, (base_depths.max() - base_depths.min()) * 0.05)
    surface_mask = base_mask & (patch <= near_cutoff + band)
    if surface_mask.sum() < 3:
        surface_mask = base_mask & (patch <= near_cutoff + 2 * band)
    if surface_mask.sum() == 0:
        surface_mask = base_mask

    # secondary refinement pass: even within the color-selected set, re-tighten
    # around the median color of the chosen near-surface pixels, in case the
    # initial box-wide color mask let in some off-color edge/background pixels
    if rgb is not None and rgb.shape[:2] == depth_m.shape[:2] and surface_mask.sum() >= 3:
        rgb_patch = rgb[y1:y2, x1:x2]
        lab_patch = cv2.cvtColor(rgb_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_color = np.median(lab_patch[surface_mask], axis=0)
        color_dist = np.linalg.norm(lab_patch - ref_color[None, None, :], axis=2)
        refined_mask = surface_mask & (color_dist <= color_tolerance)
        if refined_mask.sum() >= 3:
            surface_mask = refined_mask

    surf_ys = ys[surface_mask]
    surf_xs = xs[surface_mask]
    surf_depths = patch[surface_mask]

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    world_x = (surf_xs - cx) * surf_depths / fx
    world_y = (surf_ys - cy) * surf_depths / fy
    world_z = surf_depths

    return np.array([float(world_x.mean()), float(world_y.mean()), float(world_z.mean())], dtype=np.float64)
