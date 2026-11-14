"""Per-image measurements: sharpness where it matters, exposure, faces, perceptual hash.

`analyze_file` is a plain top-level function so it can run inside worker processes.
Everything it returns is JSON-serialisable.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

from . import exif as exif_mod
from .config import Settings
from .grouping import dhash_hex
from .loader import load_preview, map_sensor_point

_FACE = None
_EYE = None
_CASCADES_LOADED = False

# Some OpenCV wheels ship without the Haar cascade XML files; Ubuntu's `opencv-data`
# package installs them under /usr/share/opencv4/haarcascades.
_CASCADE_DIRS = [
    getattr(getattr(cv2, "data", None), "haarcascades", ""),
    "/usr/share/opencv4/haarcascades/",
    "/usr/share/opencv/haarcascades/",
    "/usr/local/share/opencv4/haarcascades/",
]


def _load_cascade(filename: str):
    for d in _CASCADE_DIRS:
        if d and os.path.isfile(os.path.join(d, filename)):
            c = cv2.CascadeClassifier(os.path.join(d, filename))
            if not c.empty():
                return c
    return None


def _cascades():
    """(face_cascade, eye_cascade); either may be None when the XML files are not installed.
    Face detection is optional: without it the AF point / sharpest-area logic still works."""
    global _FACE, _EYE, _CASCADES_LOADED
    if not _CASCADES_LOADED:
        _FACE = _load_cascade("haarcascade_frontalface_default.xml")
        _EYE = _load_cascade("haarcascade_eye.xml")
        _CASCADES_LOADED = True
    return _FACE, _EYE


def faces_available() -> bool:
    return _cascades()[0] is not None


def _lap_var(gray_blur: np.ndarray, rect=None) -> float:
    """Variance of the Laplacian (a standard focus measure) inside rect=(x0,y0,x1,y1)."""
    g = gray_blur
    if rect is not None:
        x0, y0, x1, y1 = rect
        g = g[y0:y1, x0:x1]
    if g.shape[0] < 8 or g.shape[1] < 8:
        return 0.0
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def _norm_to_px(rect, w, h):
    x, y, rw, rh = rect
    x0 = int(max(0, min(w - 1, round(x * w))))
    y0 = int(max(0, min(h - 1, round(y * h))))
    x1 = int(max(x0 + 1, min(w, round((x + rw) * w))))
    y1 = int(max(y0 + 1, min(h, round((y + rh) * h))))
    return x0, y0, x1, y1


def _peak_tile(gray_blur, grid=(8, 6)):
    """Fallback when there is no AF point and no face: use the sharpest area of the frame
    (the subject is normally the sharpest thing in a good photo)."""
    h, w = gray_blur.shape
    gx, gy = grid
    tw, th = w // gx, h // gy
    scored = []
    for j in range(gy):
        for i in range(gx):
            rect = (i * tw, j * th, (i + 1) * tw, (j + 1) * th)
            scored.append((_lap_var(gray_blur, rect), rect))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = [v for v, _ in scored[:3]]
    x0, y0, x1, y1 = scored[0][1]
    return float(np.mean(top)), (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)


def _detect_faces(gray: np.ndarray):
    """Haar-cascade face detection on a downscaled copy. Returns normalised (x,y,w,h) boxes."""
    face_c, _ = _cascades()
    if face_c is None:
        return []
    h, w = gray.shape
    scale = 800.0 / max(h, w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) \
        if scale < 1 else gray
    sh, sw = small.shape
    min_side = max(30, int(0.04 * sw))
    faces = face_c.detectMultiScale(small, scaleFactor=1.1, minNeighbors=6,
                                    minSize=(min_side, min_side))
    out = [(x / sw, y / sh, fw / sw, fh / sh) for (x, y, fw, fh) in faces]
    out.sort(key=lambda r: r[2] * r[3], reverse=True)
    return out[:6]


def _eye_hint(gray: np.ndarray, face_norm) -> int | None:
    """Number of eyes (0-2) found in the biggest face; None when the face is too small to say.
    This is a rough heuristic: 0 eyes on a frontal face often means closed eyes or a blink."""
    _, eye_c = _cascades()
    if eye_c is None:
        return None
    h, w = gray.shape
    x0, y0, x1, y1 = _norm_to_px(face_norm, w, h)
    if x1 - x0 < 60:
        return None
    crop = gray[y0:y0 + int((y1 - y0) * 0.62), x0:x1]
    f = 240.0 / crop.shape[1]
    crop = cv2.resize(crop, (240, max(1, int(crop.shape[0] * f))), interpolation=cv2.INTER_AREA)
    eyes = eye_c.detectMultiScale(crop, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return int(min(2, len(eyes)))


def _fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def analyze_file(job: dict) -> dict:
    """job = {path, exif, thumb_dir, settings(dict)}"""
    path = Path(job["path"])
    s = Settings.from_dict(job["settings"])
    exif = job.get("exif") or {}
    base = {"path": str(path), "name": path.name, "error": None,
            "fingerprint": _fingerprint(path), "mode": s.mode,
            "exif": {k: exif.get(k) for k in exif_mod.TAGS if exif.get(k) is not None},
            "capture_ts": exif_mod.capture_ts(exif, path)}
    try:
        prev = load_preview(path, s.mode)
        bgr = prev.bgr
        h0, w0 = bgr.shape[:2]
        scale = s.analysis_long_edge / max(h0, w0)
        if scale < 1:
            bgr = cv2.resize(bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)   # tames sensor noise before measuring detail

        # --- thumbnail for the UI -------------------------------------------------------
        tdir = Path(job["thumb_dir"])
        tdir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(f"{path}|{base['fingerprint']}".encode()).hexdigest()[:16]
        thumb_path = tdir / f"{key}.jpg"
        ts = s.thumb_size / max(h, w)
        thumb = cv2.resize(bgr, (max(1, int(w * ts)), max(1, int(h * ts))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, 82])

        # --- where is the subject? -------------------------------------------------------
        faces = _detect_faces(gray)
        af = exif_mod.focus_point(exif)
        global_var = _lap_var(blur)
        peak_var, peak_rect = _peak_tile(blur)

        if af is not None:
            fx, fy = map_sensor_point(af[0], af[1], prev.flip)
            bw = s.af_box_frac
            bh = bw * w / h
            rx = min(max(fx - bw / 2, 0.0), 1.0 - bw)
            ry = min(max(fy - bh / 2, 0.0), max(0.0, 1.0 - bh))
            roi = (rx, ry, bw, bh)
            source = "af_point"
        elif faces:
            roi, source = faces[0], "face"
        else:
            roi, source = peak_rect, "peak"
        focus_var = _lap_var(blur, _norm_to_px(roi, w, h)) if source != "peak" else peak_var

        face_var = eyes = None
        if faces:
            face_var = _lap_var(blur, _norm_to_px(faces[0], w, h))
            eyes = _eye_hint(gray, faces[0])

        # --- exposure --------------------------------------------------------------------
        clip_hi = float((bgr.max(axis=2) >= 252).mean())
        clip_lo = float((gray <= 5).mean())

        base.update({
            "w": int(w0), "h": int(h0), "flip": int(prev.flip),
            "thumb": str(thumb_path),
            "dhash": dhash_hex(gray),
            "focus_var": float(focus_var), "global_var": float(global_var),
            "roi": [float(v) for v in roi], "roi_source": source,
            "faces": [[float(v) for v in f] for f in faces],
            "face_var": None if face_var is None else float(face_var),
            "eyes_hint": eyes,
            "clip_hi": clip_hi, "clip_lo": clip_lo, "mean_luma": float(gray.mean()),
        })
    except Exception as exc:  # corrupt file, unsupported RAW, ...
        base["error"] = f"{type(exc).__name__}: {exc}"
    return base
