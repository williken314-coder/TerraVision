"""Raster analysis for scanned sheets (no text layer), using OpenCV.

Finds line work and round symbols so a scanned sheet still reports what it
contains. It does not read numbers: labels on scanned sheets come only from
the vision reader, and they stay flagged for review.
"""
from __future__ import annotations

try:
    import cv2  # opencv-python-headless
    import numpy as np
    OPENCV = True
except Exception:  # pragma: no cover - optional dependency
    OPENCV = False


def analyze_png(png: bytes) -> dict:
    if not OPENCV:
        return {"available": False, "reason": "OpenCV is not installed (pip install opencv-python-headless)."}
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=4)
    blur = cv2.medianBlur(img, 3)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=12, param1=120, param2=28, minRadius=4, maxRadius=18)
    return {
        "available": True,
        "line_count": 0 if lines is None else int(len(lines)),
        "circle_count": 0 if circles is None else int(circles.shape[1]),
        "size_px": [int(img.shape[1]), int(img.shape[0])],
    }
