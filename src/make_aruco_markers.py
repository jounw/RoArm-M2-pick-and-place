"""Print sheet of ArUco markers for the movable top-down camera's floor
calibration (see calibrate_aruco_floor.py).

Markers: DICT_4X4_50, ids 0..N-1, square side --size-mm (default 50) with a
white quiet zone around each and the id printed below. Laid out on A4 at
300 dpi so that printing at 100% scale (no "fit to page") gives the exact
physical size -- the side length is what the calibration later relies on.

Put them around the OUTSIDE of the work area (4 corners + 2 mid-edges for
the default 6): objects and the arm then never cover more than a couple.

More markers than fit on one A4 go onto further pages (aruco_markers_p1.png,
aruco_markers_p2.png, ...).

Run: python3 make_aruco_markers.py            -> 12 markers, 2 pages
     python3 make_aruco_markers.py --count 6 --size-mm 40
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

DPI = 300
A4_MM = (210.0, 297.0)
DICT = cv2.aruco.DICT_4X4_50


def mm_to_px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=12, help="how many markers (ids 0..count-1)")
    p.add_argument("--size-mm", type=float, default=50.0, help="black square side length in mm")
    p.add_argument("--out", default="aruco_markers.png")
    args = p.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    page_h, page_w = mm_to_px(A4_MM[1]), mm_to_px(A4_MM[0])

    side = mm_to_px(args.size_mm)
    quiet = mm_to_px(max(10.0, args.size_mm * 0.25))  # white border: detection needs it
    cell = side + 2 * quiet + mm_to_px(8)  # + room for the id label
    cols = max(1, (page_w - mm_to_px(10)) // cell)
    rows = max(1, (page_h - mm_to_px(10)) // cell)
    per_page = cols * rows
    n_pages = -(-args.count // per_page)
    x0 = (page_w - cols * cell) // 2
    y0 = (page_h - rows * cell) // 2

    stem = args.out.rsplit(".", 1)[0]
    outs = []
    for pg in range(n_pages):
        page = np.full((page_h, page_w), 255, np.uint8)
        first = pg * per_page
        ids = range(first, min(first + per_page, args.count))
        for k, i in enumerate(ids):
            r, c = divmod(k, cols)
            marker = cv2.aruco.generateImageMarker(dictionary, i, side)
            x = x0 + c * cell + quiet
            y = y0 + r * cell + quiet
            page[y:y + side, x:x + side] = marker
            # thin cut line around the quiet zone and the id below the marker
            cv2.rectangle(page, (x - quiet + 2, y - quiet + 2), (x + side + quiet - 2, y + side + quiet - 2), 200, 1)
            cv2.putText(page, f"id {i}  {args.size_mm:.0f}mm", (x, y + side + quiet + mm_to_px(5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
        cv2.putText(page, f"ArUco DICT_4X4_50  ids {ids.start}-{ids.stop - 1}  side {args.size_mm:.0f}mm  --  "
                          f"print at 100% scale, A4  (page {pg + 1}/{n_pages})",
                    (mm_to_px(10), page_h - mm_to_px(6)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
        out = args.out if n_pages == 1 else f"{stem}_p{pg + 1}.png"
        cv2.imwrite(out, page)
        outs.append(out)
    print(f"saved {outs}: {args.count} markers, {args.size_mm:.0f}mm side, {per_page} per page, A4 @ {DPI}dpi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
