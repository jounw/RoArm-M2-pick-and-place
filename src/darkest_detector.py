"""No-VLM fallback: find the darkest blob against a light background.

Useful when the object is hard to describe/verify with a VLM (e.g. black
tape on a white table) -- this is plain OpenCV thresholding, no API calls.
"""

from __future__ import annotations

import cv2


def find_darkest_box(frame, min_area_frac: float = 0.01) -> tuple[int, int, int, int] | None:
    """Returns a pixel (x, y, w, h) box around the largest dark blob, or None."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area_frac * w * h:
        return None

    return cv2.boundingRect(best)  # x, y, w, h
