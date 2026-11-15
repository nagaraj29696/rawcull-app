"""Subject detection (people, birds, animals) with YOLOX-S running on ONNX Runtime (CPU).

YOLOX is Apache-2.0 licensed.  The 36 MB model is downloaded once to ~/.cache/rawcull/models
and verified against a pinned SHA-256.  Set RAWCULL_MODEL=/path/to/yolox_s.onnx to use a file
you already have.

Small subjects (a bird that fills only a few percent of the frame) are the hard case for a
640x640 detector, so besides the full-frame pass we also run overlapping, up-scaled tiles
and merge the results.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_NAME = "yolox_s.onnx"
MODEL_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

PERSON, BIRD = 0, 14
ANIMALS = set(range(15, 24))          # cat dog horse sheep cow elephant bear zebra giraffe
SUBJECT_MODES = {
    "birds": {BIRD},
    "birds_animals": {BIRD} | ANIMALS,
    "animals": set(ANIMALS),
    "people": {PERSON},
    "any": {PERSON, BIRD} | ANIMALS,
}
SUBJECT_LABELS = {"birds": "Birds", "birds_animals": "Birds + animals", "animals": "Animals",
                  "people": "People", "any": "Any subject"}
DETECTABLE = SUBJECT_MODES["any"]


# ------------------------------------------------------------------------------ model ------
def model_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    d = base / "rawcull" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_path() -> Path:
    env = os.environ.get("RAWCULL_MODEL")
    return Path(env) if env else model_dir() / MODEL_NAME


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_ready() -> bool:
    p = model_path()
    return p.is_file() and (os.environ.get("RAWCULL_MODEL") is not None or p.stat().st_size > 30_000_000)


def ensure_model(progress=None) -> Path:
    """Return the model path, downloading + verifying it first if needed."""
    p = model_path()
    if os.environ.get("RAWCULL_MODEL"):
        if not p.is_file():
            raise FileNotFoundError(f"RAWCULL_MODEL points to a missing file: {p}")
        return p
    if p.is_file() and _sha256(p) == MODEL_SHA256:
        return p
    tmp = p.with_suffix(".part")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "rawcull"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 18)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(done, total)
    if _sha256(tmp) != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("Downloaded detector model failed its checksum; try again or place the file manually: "
                           f"{MODEL_URL} -> {p}")
    shutil.move(str(tmp), str(p))
    return p


# --------------------------------------------------------------------------- detector ------
def _iou_ios(a: np.ndarray, b: np.ndarray):
    """IoU and intersection-over-smaller between box a (4,) and boxes b (N,4)."""
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9), inter / (np.minimum(area_a, area_b) + 1e-9)


def _nms(boxes, scores, classes, iou_thr=0.5):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        iou, _ = _iou_ios(boxes[i], boxes[rest])
        same = classes[rest] == classes[i]
        order = rest[~(same & (iou > iou_thr))]
    return keep


class Detector:
    SIZE = 640

    def __init__(self, path: Path | None = None, threads: int = 1):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path or ensure_model()), opts,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        grids, strides = [], []
        for s in (8, 16, 32):
            n = self.SIZE // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids, 0).astype(np.float32)
        self.strides = np.concatenate(strides, 0).astype(np.float32)

    def _infer(self, bgr: np.ndarray, conf: float):
        """One 640x640 pass.  Returns (boxes xyxy in input pixels, scores, class ids)."""
        h, w = bgr.shape[:2]
        r = min(self.SIZE / h, self.SIZE / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        interp = cv2.INTER_LINEAR if r >= 1 else cv2.INTER_AREA
        canvas = np.full((self.SIZE, self.SIZE, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=interp)
        blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None].astype(np.float32))
        out = self.sess.run(None, {self.input_name: blob})[0][0].astype(np.float32)
        out[:, :2] = (out[:, :2] + self.grids) * self.strides
        out[:, 2:4] = np.exp(out[:, 2:4]) * self.strides
        cls_scores = out[:, 5:] * out[:, 4:5]
        cls = cls_scores.argmax(1)
        score = cls_scores[np.arange(len(cls)), cls]
        keep = (score >= conf) & np.isin(cls, list(DETECTABLE))
        if not keep.any():
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
        xywh, score, cls = out[keep, :4], score[keep], cls[keep]
        boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                          xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1) / r
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        k = _nms(boxes, score, cls)
        return boxes[k], score[k], cls[k]

    def detect(self, bgr: np.ndarray, conf: float = 0.30, small: bool = True,
               tile_up: float = 1.5, overlap: float = 0.25) -> list[dict]:
        """Detect people / birds / animals.  Boxes are returned normalised to 0..1.

        small=True adds overlapping tiles scaled up by `tile_up`, which finds birds that are only
        a few percent of the frame wide (at the cost of ~6-12 extra passes per photo)."""
        h, w = bgr.shape[:2]
        B, S, C = [], [], []
        b, s, c = self._infer(bgr, conf)
        B.append(b); S.append(s); C.append(c)

        if small:
            tile = int(round(self.SIZE / tile_up))
            if tile < min(h, w):
                step = max(1, int(tile * (1 - overlap)))
                xs = list(range(0, w - tile, step)) + [w - tile]
                ys = list(range(0, h - tile, step)) + [h - tile]
                for y0 in ys:
                    for x0 in xs:
                        crop = bgr[y0:y0 + tile, x0:x0 + tile]
                        b, s, c = self._infer(cv2.resize(crop, (self.SIZE, self.SIZE),
                                                         interpolation=cv2.INTER_CUBIC), conf)
                        if not len(s):
                            continue
                        b = b / tile_up + np.array([x0, y0, x0, y0])
                        # drop boxes cut by an inner tile border: a neighbouring tile sees them whole
                        m = 3.0
                        cut = ((b[:, 0] <= x0 + m) & (x0 > 0)) | ((b[:, 2] >= x0 + tile - m) & (x0 + tile < w)) \
                            | ((b[:, 1] <= y0 + m) & (y0 > 0)) | ((b[:, 3] >= y0 + tile - m) & (y0 + tile < h))
                        B.append(b[~cut]); S.append(s[~cut]); C.append(c[~cut])

        boxes = np.concatenate(B) if B else np.zeros((0, 4))
        scores = np.concatenate(S) if S else np.zeros(0)
        classes = np.concatenate(C) if C else np.zeros(0, int)
        if not len(scores):
            return []
        keep = _nms(boxes, scores, classes, 0.5)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        # drop fragments: a small box lying inside a larger one of the same class
        order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        final: list[int] = []
        for i in order:
            dup = False
            for j in final:
                if classes[i] == classes[j]:
                    _, ios = _iou_ios(boxes[i], boxes[j][None])
                    if ios[0] > 0.8 and scores[j] >= 0.5 * scores[i]:
                        dup = True
                        break
            if not dup:
                final.append(int(i))

        out = []
        for i in sorted(final, key=lambda i: -scores[i]):
            x1, y1, x2, y2 = boxes[i]
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue
            out.append({"cls": int(classes[i]), "name": COCO[int(classes[i])], "conf": round(float(scores[i]), 3),
                        "box": [float(x1 / w), float(y1 / h), float(x2 / w), float(y2 / h)]})
        return out
