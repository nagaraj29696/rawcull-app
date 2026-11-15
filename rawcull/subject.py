"""Subject stage: find birds / animals / people and judge focus on the subject itself.

1. Full-frame detection on the 1600 px analysis image (finds anything reasonably large).
2. Small subjects: if nothing was found where the camera focused (the AF point), look there again
   in windows cut from the *native-resolution* pixels, at 1x and 2x zoom.  A bird that is 40 px wide
   in the preview is 150 sensor pixels wide in the RAW, which is plenty for the detector.
   Without an AF point the two sharpest regions of the frame are used as search centres instead.
3. For every detection, sharpness is measured on a native-resolution crop of the subject.

This module is imported lazily by analysis.py (it needs helpers from there).
"""
from __future__ import annotations

import cv2
import numpy as np

from .analysis import _edge_map, _edge_width
from .config import Settings

MEASURE_CAP = 1000      # native px: longest side of the crop used to judge sharpness
MAX_SUBJECTS = 4


def _ios(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (small + 1e-9)


def covers(box, pt, pad: float = 0.15) -> bool:
    """Is the normalised point inside the box (grown by `pad` of its size)?"""
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return x0 - px <= pt[0] <= x1 + px and y0 - py <= pt[1] <= y1 + py


def sharp_centers(wmap: np.ndarray, k: int = 2, grid=(8, 6)) -> list[tuple[float, float]]:
    """Centres of the k tiles with the crispest edges (where an in-focus subject probably is)."""
    h, w = wmap.shape
    gx, gy = grid
    tw, th = w // gx, h // gy
    found = []
    for j in range(gy):
        for i in range(gx):
            ew, _ = _edge_width(wmap, (i * tw, j * th, (i + 1) * tw, (j + 1) * th), min_edges=25)
            if ew is not None:
                found.append((ew, ((i + 0.5) * tw / w, (j + 0.5) * th / h)))
    found.sort(key=lambda t: t[0])
    centers: list[tuple[float, float]] = []
    for _, c in found:
        if all(abs(c[0] - o[0]) > 0.2 or abs(c[1] - o[1]) > 0.2 for o in centers):
            centers.append(c)
        if len(centers) >= k:
            break
    return centers or [(0.5, 0.5)]


def subject_sharpness(gray_u8: np.ndarray, s: Settings):
    """(edge width in px or None, number of edge pixels, Laplacian variance) inside the crop.
    Only an ellipse inscribed in the crop counts, so background edges in the corners of a
    bounding box do not make a soft bird look sharp."""
    h, w = gray_u8.shape
    g = cv2.GaussianBlur(gray_u8, (0, 0), s.subject_preblur)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (max(1, w // 2), max(1, h // 2)), 0, 0, 360, 255, -1)
    wmap = _edge_map(g)
    wmap[mask == 0] = 0
    ew, n = _edge_width(wmap, (0, 0, w, h), min_edges=20)
    lap = float(cv2.Laplacian(g, cv2.CV_32F)[mask > 0].var()) if (mask > 0).any() else 0.0
    return ew, n, lap


def _window_detect(src, detector, center, win: int, conf: float) -> list[dict]:
    """Run the detector on a native-pixel window centred on `center` (win px square)."""
    crop, ox, oy = src.window(center[0], center[1], win)
    if min(crop.shape[:2]) < 48:
        return []
    boxes, scores, classes = detector._infer(crop, conf)
    ch, cw = crop.shape[:2]
    out = []
    from .detect import COCO
    for b, sc, c in zip(boxes, scores, classes):
        # a box cut by the window border (that is not the image border) is partial; skip it
        cut = ((b[0] <= 3 and ox > 0) or (b[2] >= cw - 3 and ox + cw < src.nw)
               or (b[1] <= 3 and oy > 0) or (b[3] >= ch - 3 and oy + ch < src.nh))
        if cut:
            continue
        out.append({"cls": int(c), "name": COCO[int(c)], "conf": round(float(sc), 3), "src": "af_window",
                    "box": [float((b[0] + ox) / src.nw), float((b[1] + oy) / src.nh),
                        float((b[2] + ox) / src.nw), float((b[3] + oy) / src.nh)]})
    return out


def _measure(det: dict, src, bgr: np.ndarray, af, s: Settings) -> None:
    """Fill in native_w, edge_w, edge_n, lap, af_inside for one detection (in place)."""
    x0, y0, x1, y1 = det["box"]
    nw, nh = (src.nw, src.nh) if src is not None else (bgr.shape[1], bgr.shape[0])
    X0, X1, Y0, Y1 = x0 * nw, x1 * nw, y0 * nh, y1 * nh
    det["native_w"] = int(X1 - X0)
    det["af_inside"] = None if af is None else covers(det["box"], af)
    cx, cy = ((af[0] * nw, af[1] * nh) if (af is not None and det["af_inside"]) else ((X0 + X1) / 2, (Y0 + Y1) / 2))
    if X1 - X0 > MEASURE_CAP:                      # huge subject: judge the part around the focus point
        X0 = min(max(cx - MEASURE_CAP / 2, X0), X1 - MEASURE_CAP); X1 = X0 + MEASURE_CAP
    if Y1 - Y0 > MEASURE_CAP:
        Y0 = min(max(cy - MEASURE_CAP / 2, Y0), Y1 - MEASURE_CAP); Y1 = Y0 + MEASURE_CAP
    crop = src.crop_px(X0, Y0, X1, Y1) if src is not None else bgr[int(Y0):int(Y1), int(X0):int(X1)]
    if min(crop.shape[:2]) < 16:
        det.update(edge_w=None, edge_n=0, lap=0.0, native=src is not None)
        return
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ew, n, lap = subject_sharpness(gray, s)
    det.update(edge_w=ew, edge_n=int(n), lap=lap, native=src is not None)


def run_subject_stage(src, bgr: np.ndarray, af, centers, detector, s: Settings) -> list[dict]:
    """Returns up to MAX_SUBJECTS measured detections (all detectable classes)."""
    dets = []
    for d in detector.detect(bgr, conf=s.subject_conf, small=False):
        d["src"] = "full"
        dets.append(d)

    if s.small_subjects and src is not None:
        for c in centers:
            if any(covers(d["box"], c) for d in dets):
                continue                              # something is already known there
            for win in (640, 320):                    # native px: 1x and 2x zoom
                new = _window_detect(src, detector, c, win, s.subject_conf)
                for n in new:
                    if not any(d["cls"] == n["cls"] and _ios(d["box"], n["box"]) > 0.6 for d in dets):
                        dets.append(n)
                if any(covers(d["box"], c) for d in new):
                    break

    def prominence(d):
        x0, y0, x1, y1 = d["box"]
        return d["conf"] * np.sqrt(max((x1 - x0) * (y1 - y0), 1e-6)) * (1.6 if af is not None and covers(d["box"], af) else 1.0)

    dets.sort(key=prominence, reverse=True)
    dets = dets[:MAX_SUBJECTS]
    for d in dets:
        _measure(d, src, bgr, af, s)
    return dets
