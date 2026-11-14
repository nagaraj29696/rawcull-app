"""EXIF / maker-note reading through the `exiftool` command (optional but recommended).

exiftool gives us what the camera itself recorded: capture time, ISO/shutter/aperture, and,
for Sony bodies like the A6400, the autofocus point (`FocusLocation`).  If exiftool is not
installed everything still works; we just fall back to file times and to image-based
focus-area detection.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

TAGS = [
    "DateTimeOriginal", "ISO", "ExposureTime", "FNumber", "FocalLength",
    "Model", "LensModel", "FocusLocation", "FocusLocation2", "FocusMode", "AFAreaMode",
]


def have_exiftool() -> bool:
    return shutil.which("exiftool") is not None


def read_exif_batch(paths, chunk: int = 300) -> dict[str, dict]:
    """Return {str(path): {tag: value}}.  Empty dict when exiftool is unavailable."""
    exe = shutil.which("exiftool")
    out: dict[str, dict] = {}
    if not exe:
        return out
    paths = [str(p) for p in paths]
    for i in range(0, len(paths), chunk):
        part = paths[i:i + chunk]
        cmd = [exe, "-json", "-n", "-q", "-q"] + [f"-{t}" for t in TAGS] + ["-@", "-"]
        try:
            res = subprocess.run(cmd, input="\n".join(part) + "\n", capture_output=True,
                                 text=True, timeout=600)
            for row in json.loads(res.stdout or "[]"):
                out[row.get("SourceFile", "")] = row
        except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
            continue
    return out


def focus_point(exif: dict) -> tuple[float, float] | None:
    """Normalised (x, y) of the AF point in the sensor frame, or None."""
    for key in ("FocusLocation", "FocusLocation2"):
        val = exif.get(key)
        if val is None:
            continue
        try:
            nums = [float(v) for v in str(val).replace(",", " ").split()]
        except ValueError:
            continue
        if len(nums) < 4:
            continue
        # exiftool: "width height x y".  Some tools report 5 values ("width height ? x y").
        w, h = nums[0], nums[1]
        pairs = [(3, 4), (2, 3)] if len(nums) >= 5 else [(2, 3)]
        for ix, iy in pairs:
            x, y = nums[ix], nums[iy]
            if w > 0 and h > 0 and 0 <= x <= w and 0 <= y <= h and (x or y):
                return x / w, y / h
    return None


def capture_ts(exif: dict, path) -> float:
    """Capture time as a UNIX timestamp (whole seconds); falls back to file mtime."""
    raw = exif.get("DateTimeOriginal")
    if raw:
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return dt.datetime.strptime(str(raw)[:19], fmt).timestamp()
            except ValueError:
                pass
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def summary(exif: dict) -> str:
    """Short human-readable shooting info, e.g. 'ISO 800 · 1/500s · f/2.8 · 35mm'."""
    parts = []
    if exif.get("ISO"):
        parts.append(f"ISO {int(exif['ISO'])}")
    t = exif.get("ExposureTime")
    if t:
        parts.append(f"1/{round(1 / float(t))}s" if float(t) < 0.5 else f"{float(t):g}s")
    if exif.get("FNumber"):
        parts.append(f"f/{float(exif['FNumber']):g}")
    if exif.get("FocalLength"):
        parts.append(f"{float(exif['FocalLength']):g}mm")
    return " · ".join(parts)
