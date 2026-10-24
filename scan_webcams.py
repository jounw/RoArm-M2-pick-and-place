"""Probe webcam indices 0-7; save one frame from each index that opens.

Run on your actual machine (needs real camera hardware access):
    python3 scan_webcams.py

Saves webcam_0.jpg, webcam_1.jpg, ... for every index that successfully opens
and reads a frame, and prints a summary of which indices worked.
"""

from __future__ import annotations

import cv2

OUT_DIR = "."


def main() -> None:
    found = []
    for i in range(8):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"index {i}: 사용 불가")
            cap.release()
            continue

        # warm up a few frames -- some cameras return a black/garbage first frame
        ok, frame = False, None
        for _ in range(5):
            ok, frame = cap.read()

        if ok and frame is not None:
            path = f"{OUT_DIR}/webcam_{i}.jpg"
            cv2.imwrite(path, frame)
            print(f"index {i}: 캡처됨 -> {path} (shape={frame.shape})")
            found.append(i)
        else:
            print(f"index {i}: 열렸지만 프레임을 읽지 못함")

        cap.release()

    print("\n사용 가능한 인덱스:", found if found else "없음")


if __name__ == "__main__":
    main()
