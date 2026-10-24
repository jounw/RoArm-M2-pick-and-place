"""Object tracker for the eye-in-hand loop.

Primary: OpenCV's ViT-Track (transformer-based single-object tracker from
opencv_zoo) -- much more robust to rotation/scale/partial occlusion than
plain template matching, and still fast enough for this loop's pace.
Falls back to multi-scale template matching if the ONNX model file isn't
present (e.g. offline machine that never ran download_vittrack_model.py).

VLM gives the initial box; after that this tracker follows the object
frame-to-frame without any further API calls.
"""

from __future__ import annotations

import os

import cv2

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "vittrack.onnx")
DEFAULT_SCORE_THRESHOLD = 0.3


def normalized_box_to_pixel_roi(
    box: tuple[int, int, int, int], frame_shape: tuple[int, ...], pad_frac: float = 0.15
) -> tuple[int, int, int, int]:
    """Convert a 0-1000 normalized (x1,y1,x2,y2) box to a padded pixel (x,y,w,h) ROI."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    px1 = int(x1 / 1000 * w)
    py1 = int(y1 / 1000 * h)
    px2 = int(x2 / 1000 * w)
    py2 = int(y2 / 1000 * h)
    px1, px2 = sorted((max(0, min(w - 1, px1)), max(0, min(w - 1, px2))))
    py1, py2 = sorted((max(0, min(h - 1, py1)), max(0, min(h - 1, py2))))
    bw = max(1, px2 - px1)
    bh = max(1, py2 - py1)
    pad_x = int(bw * pad_frac)
    pad_y = int(bh * pad_frac)
    px1 = max(0, px1 - pad_x)
    py1 = max(0, py1 - pad_y)
    px2 = min(w - 1, px2 + pad_x)
    py2 = min(h - 1, py2 + pad_y)
    return px1, py1, max(1, px2 - px1), max(1, py2 - py1)


class TemplateTracker:
    """Fallback: multi-scale template matching, no model file needed."""

    def __init__(self, min_score: float = 0.32, search_scale: float = 3.0) -> None:
        self.box: tuple[int, int, int, int] | None = None
        self.template_gray = None
        self.min_score = min_score
        self.search_scale = search_scale
        self.scales = (0.65, 0.8, 1.0, 1.2, 1.45)

    def init(self, frame, roi: tuple[int, int, int, int]) -> bool:
        h_img, w_img = frame.shape[:2]
        x, y, w, h = (int(v) for v in roi)
        x = max(0, min(w_img - 2, x))
        y = max(0, min(h_img - 2, y))
        w = max(4, min(w, w_img - x))
        h = max(4, min(h, h_img - y))
        self.box = (x, y, w, h)
        template_bgr = frame[y : y + h, x : x + w]
        self.template_gray = cv2.GaussianBlur(cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        return self.template_gray.size > 0

    def update(self, frame) -> tuple[bool, tuple[int, int, int, int] | None]:
        if self.box is None or self.template_gray is None:
            return False, None

        img_h, img_w = frame.shape[:2]
        x, y, w, h = self.box
        cx, cy = x + w / 2, y + h / 2
        search_w = max(int(w * self.search_scale), w + 80)
        search_h = max(int(h * self.search_scale), h + 80)
        sx1 = max(0, int(cx - search_w / 2))
        sy1 = max(0, int(cy - search_h / 2))
        sx2 = min(img_w, int(cx + search_w / 2))
        sy2 = min(img_h, int(cy + search_h / 2))
        search = frame[sy1:sy2, sx1:sx2]
        if search.size == 0:
            return False, self.box

        search_gray = cv2.GaussianBlur(cv2.cvtColor(search, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        base_h, base_w = self.template_gray.shape[:2]

        best_score = -1.0
        best_box = None
        for scale in self.scales:
            tw = max(4, int(base_w * scale))
            th = max(4, int(base_h * scale))
            if tw >= search_gray.shape[1] or th >= search_gray.shape[0]:
                continue
            templ = cv2.resize(self.template_gray, (tw, th), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(search_gray, templ, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if score > best_score:
                best_score = float(score)
                best_box = (sx1 + loc[0], sy1 + loc[1], tw, th)

        if best_box is None or best_score < self.min_score:
            return False, self.box

        self.box = best_box
        return True, best_box


class Tracker:
    """ViT-Track when the model file is available, else TemplateTracker."""

    def __init__(self, score_threshold: float = DEFAULT_SCORE_THRESHOLD) -> None:
        self.score_threshold = score_threshold
        self.use_vit = os.path.exists(MODEL_PATH)
        self._vit = None
        self._fallback = TemplateTracker()
        self.box: tuple[int, int, int, int] | None = None
        if not self.use_vit:
            print(f"[tracker] {MODEL_PATH} not found, falling back to template matching. Run download_vittrack_model.py.")

    def init(self, frame, roi: tuple[int, int, int, int]) -> bool:
        x, y, w, h = (int(v) for v in roi)
        self.box = (x, y, w, h)
        if self.use_vit:
            params = cv2.TrackerVit_Params()
            params.net = MODEL_PATH
            self._vit = cv2.TrackerVit_create(params)
            self._vit.init(frame, (x, y, w, h))
            return True
        return self._fallback.init(frame, roi)

    def update(self, frame) -> tuple[bool, tuple[int, int, int, int] | None]:
        if self.use_vit:
            ok, box = self._vit.update(frame)
            box = tuple(int(v) for v in box) if box else self.box
            if ok:
                score = self._vit.getTrackingScore()
                ok = score >= self.score_threshold
            if box is not None:
                self.box = box
            return ok, self.box
        return self._fallback.update(frame)
