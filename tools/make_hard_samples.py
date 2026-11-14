"""Harder test shoot: sensor noise added AFTER the blur (like a real high-ISO camera) and
different kinds of scene.  Usage: python tools/make_hard_samples.py OUTDIR"""
import os
import sys

import cv2
import numpy as np


def edgy_scene(seed=3, W=1600, H=1067):
    r = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)
    for y in range(H):
        img[y] = (60 + y * 0.05, 90 + y * 0.04, 120 + y * 0.03)
    for _ in range(70):
        c = [float(v) for v in r.integers(20, 235, 3)]
        cv2.circle(img, (int(r.integers(0, W)), int(r.integers(0, H))), int(r.integers(15, 110)), c, -1)
    for _ in range(40):
        cv2.line(img, (int(r.integers(0, W)), int(r.integers(0, H))),
                 (int(r.integers(0, W)), int(r.integers(0, H))), (240, 240, 240), 3)
    tex = cv2.GaussianBlur(r.normal(0, 40, (H, W)).astype(np.float32), (0, 0), 1.2)
    return img + tex[..., None]


def smooth_scene(W=1600, H=1067):
    r = np.random.default_rng(9)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), np.float32)
    img[...] = (150 + 40 * np.sin(xx / 300.0) + 30 * np.cos(yy / 200.0))[..., None] * np.array([0.9, 1.0, 1.1])
    cv2.ellipse(img, (800, 520), (260, 340), 0, 0, 360, (170, 190, 235), -1)
    img[:] = cv2.GaussianBlur(img, (0, 0), 4) * 0.5 + img * 0.5
    for _ in range(6):
        cv2.line(img, (int(r.integers(0, W)), int(r.integers(0, H))),
                 (int(r.integers(0, W)), int(r.integers(0, H))), (30, 30, 30), 4)
    return img


def shoot(base, blur, noise, seed):
    img = cv2.GaussianBlur(base, (0, 0), blur) if blur else base.copy()
    img = img + np.random.default_rng(seed).normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


# (scene, [blur per frame]) -> one burst each.  Group 1 is a burst where EVERY frame is blurry.
PLAN = [
    ("edgy", 3, [0, 1.5, 3, 6]),     # one sharp frame among softer ones
    ("edgy", 4, [4, 5, 6, 8]),       # all blurry -> nothing should be picked
    ("smooth", 0, [0, 2, 4]),        # low-texture portrait-like scene, sharp frame must still win
]

if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    n = 0
    for gi, (kind, seed, blurs) in enumerate(PLAN):
        base = edgy_scene(seed) if kind == "edgy" else smooth_scene()
        for j, b in enumerate(blurs):
            n += 1
            p = os.path.join(out, f"DSC{n:05d}.jpg")
            cv2.imwrite(p, shoot(base, b, 14, n), [cv2.IMWRITE_JPEG_QUALITY, 90])
            t = 1_700_000_000 + gi * 120 + j
            os.utime(p, (t, t))
    print("wrote", n)
