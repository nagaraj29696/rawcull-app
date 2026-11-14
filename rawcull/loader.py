"""Reading previews and full-resolution pixels from RAW files and ordinary images.

Everything returned by this module is an *upright* BGR uint8 image, so the rest of the
program never has to think about camera orientation.  The `flip` value (LibRaw convention:
0 none, 3 = 180, 5 = 90 CCW, 6 = 90 CW) is returned so that coordinates stored in the
sensor frame (like Sony's focus location) can be mapped into the upright frame.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

RAW_EXTS = {".arw", ".dng", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2", ".pef", ".srw"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

_FLIP_TO_CV = {
    3: cv2.ROTATE_180,
    5: cv2.ROTATE_90_COUNTERCLOCKWISE,
    6: cv2.ROTATE_90_CLOCKWISE,
}


def is_raw(path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTS


def list_images(folder, recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.iterdir()
    exts = RAW_EXTS | IMG_EXTS
    files = [p for p in it if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")]
    # If a RAW and a same-named JPEG both exist, keep only the RAW (the JPEG is its twin).
    raw_stems = {p.with_suffix("").name.lower() for p in files if is_raw(p)}
    files = [p for p in files if is_raw(p) or p.with_suffix("").name.lower() not in raw_stems]
    return sorted(files, key=lambda p: p.name.lower())


def rotate_upright(bgr: np.ndarray, flip: int) -> np.ndarray:
    code = _FLIP_TO_CV.get(flip)
    return cv2.rotate(bgr, code) if code is not None else bgr


def map_sensor_point(fx: float, fy: float, flip: int) -> tuple[float, float]:
    """Map a normalised point from the sensor frame to the upright frame."""
    if flip == 3:
        return 1.0 - fx, 1.0 - fy
    if flip == 6:
        return 1.0 - fy, fx
    if flip == 5:
        return fy, 1.0 - fx
    return fx, fy


@dataclass
class Preview:
    bgr: np.ndarray
    flip: int


def _pil_to_bgr(im: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(im.convert("RGB"))[:, :, ::-1])


def load_preview(path, mode: str = "fast", min_edge: int = 1000) -> Preview:
    path = Path(path)
    if not is_raw(path):
        with Image.open(path) as im:
            orient = im.getexif().get(274, 1)
            im = ImageOps.exif_transpose(im)
            flip = {3: 3, 6: 6, 8: 5}.get(orient, 0)
            return Preview(_pil_to_bgr(im), flip)

    import rawpy

    with rawpy.imread(str(path)) as raw:
        flip = int(raw.sizes.flip)
        sensor_landscape = raw.sizes.width >= raw.sizes.height
        if mode == "fast":
            try:
                thumb = raw.extract_thumb()
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                thumb = None
            if thumb is not None:
                arr = None
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    with Image.open(io.BytesIO(thumb.data)) as im:
                        arr = _pil_to_bgr(im)
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                    arr = np.ascontiguousarray(thumb.data[:, :, ::-1])
                if arr is not None and max(arr.shape[:2]) >= min_edge:
                    thumb_landscape = arr.shape[1] >= arr.shape[0]
                    if thumb_landscape == sensor_landscape:
                        # preview is still in sensor orientation -> make it upright
                        arr = rotate_upright(arr, flip)
                    return Preview(np.ascontiguousarray(arr), flip)
        # accurate mode, or no usable embedded preview: LibRaw output is already upright
        rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
        return Preview(np.ascontiguousarray(rgb[:, :, ::-1]), flip)


def load_fullres_rgb(path) -> np.ndarray:
    """Full-resolution upright RGB, for checking focus at 100% zoom."""
    path = Path(path)
    if not is_raw(path):
        with Image.open(path) as im:
            return np.ascontiguousarray(np.asarray(ImageOps.exif_transpose(im).convert("RGB")))
    import rawpy

    with rawpy.imread(str(path)) as raw:
        return np.ascontiguousarray(raw.postprocess(use_camera_wb=True, output_bps=8))
