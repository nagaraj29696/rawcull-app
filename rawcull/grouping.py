"""Perceptual hashing and burst/duplicate grouping."""
from __future__ import annotations

import cv2
import numpy as np


def dhash_hex(gray: np.ndarray, size: int = 16) -> str:
    """Difference hash (size*size bits) as hex.  INTER_AREA averages heavily, so the hash is
    stable against slight blur/noise differences between frames of the same scene."""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return format(value, f"0{size * size // 4}x")


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def assign_groups(items: list[dict], time_gap: float, max_dist: int) -> None:
    """Chain consecutive shots (by capture time, then name) into groups.

    A new group starts when the time gap is larger than `time_gap` seconds or the scene
    changes (perceptual hash distance above `max_dist`).  Sets item["group"] in place.
    """
    ok = [it for it in items if not it.get("error")]
    ok.sort(key=lambda it: (it["capture_ts"], it["name"]))
    gid = -1
    prev = None
    for it in ok:
        if prev is None:
            new = True
        else:
            new = (it["capture_ts"] - prev["capture_ts"] > time_gap
                   or hamming_hex(it["dhash"], prev["dhash"]) > max_dist)
        if new:
            gid += 1
        it["group"] = gid
        prev = it
    for it in items:
        if it.get("error"):
            it["group"] = -1
