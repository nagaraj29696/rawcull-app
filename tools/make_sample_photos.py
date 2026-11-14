import os, sys, cv2, numpy as np
out = sys.argv[1]
rng = np.random.default_rng(1)
def scene(seed):
    r = np.random.default_rng(seed)
    img = np.zeros((1080, 1620, 3), np.uint8)
    # gradient background + textured blobs
    for y in range(1080):
        img[y, :, :] = (40 + y * 0.12, 70 + y * 0.08, 110 + y * 0.05)
    for _ in range(60):
        c = tuple(int(v) for v in r.integers(30, 230, 3))
        cv2.circle(img, (int(r.integers(0,1620)), int(r.integers(0,1080))), int(r.integers(20,120)), c, -1)
    noise = r.normal(0, 25, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    for _ in range(80):
        cv2.line(img, (int(r.integers(0,1620)), int(r.integers(0,1080))), (int(r.integers(0,1620)), int(r.integers(0,1080))), (255,255,255), 2)
    return img
base_t = 1_700_000_000
plan = [  # (scene seed, blur sigma, brightness gain)
    [(1,0,1.0),(1,2.5,1.0),(1,6,1.0),(1,0.8,1.0)],
    [(2,4,1.0),(2,0,1.0),(2,0,2.6),(2,9,1.0)],
    [(3,0,1.0),(3,1.5,1.0),(3,3.5,1.0),(3,0.4,0.35)],
]
n = 0
for gi, grp in enumerate(plan):
    for j, (seed, sig, gain) in enumerate(grp):
        img = scene(seed)
        if sig: img = cv2.GaussianBlur(img, (0,0), sig)
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        n += 1
        p = os.path.join(out, f"DSC{n:05d}.jpg")
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        t = base_t + gi * 120 + j
        os.utime(p, (t, t))
print("wrote", n)
