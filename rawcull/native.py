"""Native-resolution pixels for judging focus on small subjects.

The 1600 px analysis image is far too coarse for a bird that fills 3% of the frame: a focus
miss of a few sensor pixels is invisible there.  `NativeSource` hands out crops at the
camera's own resolution instead:

* Bayer RAW (Sony ARW etc.): the raw mosaic is binned 2x2 into a half-resolution colour image
  (about 3000 px wide for the A6400).  No demosaicing, so a crop costs a few milliseconds.
* Other RAW formats: one half-size LibRaw decode, cached.
* JPEG / PNG / TIFF: the image itself at full resolution.

All coordinates handed in and out are *upright* (as the photo is displayed).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .loader import ensure_unpacked, is_raw, rotate_upright


def upright_to_sensor(ux: float, uy: float, flip: int) -> tuple[float, float]:
    """Inverse of loader.map_sensor_point."""
    if flip == 3:
        return 1.0 - ux, 1.0 - uy
    if flip == 6:
        return uy, 1.0 - ux
    if flip == 5:
        return 1.0 - uy, ux
    return ux, uy


def bin_bayer_crop(raw, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Half-resolution BGR uint8 crop of the visible raw area (sensor pixel coordinates).

    Each 2x2 block of any Bayer pattern holds R, G, G, B, so binning needs no knowledge of the
    pattern: colours come from `raw_colors_visible`."""
    x0, y0 = x0 & ~1, y0 & ~1
    w2, h2 = (x1 - x0) // 2, (y1 - y0) // 2
    x1, y1 = x0 + w2 * 2, y0 + h2 * 2
    img = raw.raw_image_visible[y0:y1, x0:x1].astype(np.float32)
    col = raw.raw_colors_visible[y0:y1, x0:x1]

    black = np.asarray(raw.black_level_per_channel, np.float32)
    img = np.maximum(img - black[col], 0.0) / max(float(raw.white_level) - float(black.mean()), 1.0)
    wb = list(raw.camera_whitebalance)
    g2 = wb[3] if len(wb) > 3 and wb[3] > 0 else wb[1]
    gains = np.array([wb[0], wb[1], wb[2], g2], np.float32)
    gains = gains / max(float(gains[1]), 1e-6)
    img = img * gains[col]

    blocks = img.reshape(h2, 2, w2, 2).transpose(0, 2, 1, 3).reshape(h2, w2, 4)
    cblk = col.reshape(h2, 2, w2, 2).transpose(0, 2, 1, 3).reshape(h2, w2, 4)

    def chan(mask):
        return (blocks * mask).sum(2) / np.maximum(mask.sum(2), 1)

    rgb = np.stack([chan(cblk == 2), chan((cblk == 1) | (cblk == 3)), chan(cblk == 0)], 2)   # B, G, R
    # normalise exposure per crop so edge-contrast thresholds behave the same in dark and bright scenes
    top = float(np.percentile(rgb, 99.5))
    rgb = rgb * min(0.9 / max(top, 0.02), 8.0)
    return (np.clip(rgb, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


class NativeSource:
    def __init__(self, path, raw=None):
        self.path = Path(path)
        self._raw = raw
        self._own_raw = False
        self._img: np.ndarray | None = None
        self.kind = "image"          # "bayer" | "image" (full upright BGR array in self._img)
        self.flip = 0
        self.nw = self.nh = 0         # upright native size in pixels
        self.scale = 1                # sensor pixels per native pixel
        self._open()

    # ---- opening -------------------------------------------------------------------------
    def _open(self):
        if not is_raw(self.path):
            with Image.open(self.path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                self._img = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
            long_edge = max(self._img.shape[:2])
            if long_edge > 4000:      # full-size camera JPEG: bring it to the half-resolution regime of RAW crops
                k = 3000.0 / long_edge
                self._img = cv2.resize(self._img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
                self.scale = 1.0 / k
            self.nh, self.nw = self._img.shape[:2]
            return
        import rawpy

        if self._raw is None:
            self._raw = rawpy.RawPy()
            self._raw.open_file(str(self.path))
            self._own_raw = True
        raw = self._raw
        ensure_unpacked(raw)
        self.flip = int(raw.sizes.flip)
        pattern = raw.raw_pattern
        if pattern is not None and pattern.shape == (2, 2):
            self.kind = "bayer"
            self.scale = 2
            H, W = raw.raw_image_visible.shape
            self._sw, self._sh = W, H
            self.nw, self.nh = (H // 2, W // 2) if self.flip in (5, 6) else (W // 2, H // 2)
        else:   # X-Trans etc.: a single half-size decode (slow, but correct)
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
            self._img = np.ascontiguousarray(rgb[:, :, ::-1])
            self.nh, self.nw = self._img.shape[:2]
            self.scale = 2

    def close(self):
        if self._own_raw and self._raw is not None:
            self._raw.close()
        self._raw = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- cropping ------------------------------------------------------------------------
    def crop_px(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        """Upright BGR crop; coordinates are upright native pixels.  Clamped to the image."""
        x0, x1 = sorted((max(0.0, min(self.nw, x0)), max(0.0, min(self.nw, x1))))
        y0, y1 = sorted((max(0.0, min(self.nh, y0)), max(0.0, min(self.nh, y1))))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return np.zeros((2, 2, 3), np.uint8)
        if self.kind == "image":
            return self._img[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
        # Bayer: upright native box -> sensor frame -> sensor pixels -> bin -> rotate upright
        ax, ay = upright_to_sensor(x0 / self.nw, y0 / self.nh, self.flip)
        bx, by = upright_to_sensor(x1 / self.nw, y1 / self.nh, self.flip)
        sx0, sx1 = sorted((ax, bx))
        sy0, sy1 = sorted((ay, by))
        crop = bin_bayer_crop(self._raw, int(sx0 * self._sw), int(sy0 * self._sh),
                              int(np.ceil(sx1 * self._sw)), int(np.ceil(sy1 * self._sh)))
        return rotate_upright(crop, self.flip)

    def window(self, cx: float, cy: float, size_px: int) -> tuple[np.ndarray, int, int]:
        """size_px square window (upright native px) centred on a normalised point.
        Returns (crop, x_origin, y_origin) in native pixels."""
        half = size_px / 2
        x0 = int(round(min(max(cx * self.nw - half, 0), max(self.nw - size_px, 0))))
        y0 = int(round(min(max(cy * self.nh - half, 0), max(self.nh - size_px, 0))))
        return self.crop_px(x0, y0, x0 + size_px, y0 + size_px), x0, y0
