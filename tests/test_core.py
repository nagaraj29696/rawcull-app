"""Run with:  python -m pytest tests   (or: python tests/test_core.py)"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rawcull import exif, xmp                      # noqa: E402
from rawcull.config import Settings                # noqa: E402
from rawcull.loader import map_sensor_point        # noqa: E402
from rawcull.pipeline import run_analysis          # noqa: E402
from rawcull.scoring import effective              # noqa: E402


def test_focus_point_and_orientation():
    assert exif.focus_point({"FocusLocation": "6000 4000 3000 2000"}) == (0.5, 0.5)
    assert exif.focus_point({"FocusLocation": "6000 4000 0 0"}) is None
    assert map_sensor_point(0.2, 0.3, 6) == (0.7, 0.2)   # 90 degrees clockwise
    assert map_sensor_point(0.2, 0.3, 5) == (0.3, 0.8)   # 90 degrees counter-clockwise


def test_xmp_preserves_existing_sidecar():
    with tempfile.TemporaryDirectory() as d:
        sc = Path(d) / "IMG_0001.ARW.xmp"
        sc.write_text('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="r">'
                      '<rdf:Description rdf:about="" xmp:Rating="1" darktable:history_end="5"/>'
                      '</rdf:RDF></x:xmpmeta>')
        xmp.write_sidecar(sc, 5, "reject")
        text = sc.read_text()
        assert 'darktable:history_end="5"' in text and 'xmp:Rating="-1"' in text
        assert (Path(d) / "IMG_0001.ARW.xmp.rawcull.bak").exists()


def test_best_of_each_burst_is_picked():
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_sample_photos.py"), d], check=True,
                       capture_output=True)
        res = {r["name"]: r for r in run_analysis(d, Settings(workers=2, subject_mode="off"))}
        picks = {n for n, r in res.items() if effective(r)[1] == "pick"}
        # One pick per burst, always one of the effectively sharp frames (sigma <= 0.8 px);
        # these frames are near-ties, so which of them wins is not asserted.
        assert len(picks) == 3
        assert any(p in picks for p in ("DSC00001.jpg", "DSC00004.jpg"))
        assert "DSC00006.jpg" in picks                       # DSC00007 is blown out and rejected
        assert any(p in picks for p in ("DSC00009.jpg", "DSC00012.jpg"))
        assert effective(res["DSC00007.jpg"])[1] == "reject"
        for name in ("DSC00003.jpg", "DSC00005.jpg", "DSC00008.jpg", "DSC00011.jpg"):   # heavily blurred
            assert effective(res[name])[1] == "reject", name


def test_noisy_blurry_shots_are_not_scored_good():
    """High-ISO noise added after the blur must not make soft photos look sharp, and a sharp
    low-texture scene must not be punished (both failed with a plain Laplacian measure)."""
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, str(ROOT / "tools" / "make_hard_samples.py"), d], check=True,
                       capture_output=True)
        res = {r["name"]: r for r in run_analysis(d, Settings(workers=2, subject_mode="off"))}
        picks = sorted(n for n, r in res.items() if effective(r)[1] == "pick")
        assert picks == ["DSC00001.jpg", "DSC00009.jpg"], picks      # nothing picked in the all-blurry burst
        for name in ("DSC00003.jpg", "DSC00004.jpg", "DSC00005.jpg", "DSC00006.jpg", "DSC00007.jpg",
                     "DSC00008.jpg", "DSC00011.jpg"):                 # blur sigma >= 3
            assert res[name]["score"] < 35 and effective(res[name])[1] != "pick", (name, res[name]["score"])
        assert res["DSC00009.jpg"]["score"] > 80                      # sharp, low-texture scene


def _fake_result(subject=None, **kw):
    r = {"name": "x.jpg", "error": None, "capture_ts": 0.0, "dhash": "0" * 64, "focus_var": 300.0,
         "edge_w": 3.2, "roi_source": "peak", "faces": [], "face_var": None, "clip_hi": 0.0, "clip_lo": 0.0,
         "mean_luma": 120.0, "subjects": [subject] if subject else [], "subjects_done": True, "subject_error": None}
    r.update(kw)
    return r


def _bird(edge_w, af_inside, native_w=120):
    return {"cls": 14, "name": "bird", "conf": 0.8, "box": [0.4, 0.4, 0.5, 0.5], "edge_w": edge_w,
            "edge_n": 400, "lap": 300.0, "native_w": native_w, "af_inside": af_inside, "src": "af_window"}


def test_subject_scoring():
    from rawcull.scoring import finalize
    s = Settings(subject_mode="birds")
    crisp, soft, miss = _fake_result(_bird(3.3, True)), _fake_result(_bird(6.5, True)), _fake_result(_bird(3.3, False))
    crisp["name"], soft["name"], miss["name"] = "crisp.jpg", "soft.jpg", "miss.jpg"
    finalize([crisp, soft, miss], s)
    assert crisp["score"] > 85 and crisp["focus_source"] == "subject"
    assert soft["score"] < 25 and soft["auto_flag"] == "reject"            # blurry bird, even if the frame is sharp
    assert miss["score"] < crisp["score"] and any("not on the bird" in t for t in miss["reasons"])
    # other subject types are ignored; with no subject the whole-frame measure takes over
    other = _fake_result(dict(_bird(3.3, True), cls=0, name="person"))
    finalize([other], s)
    assert other["focus_source"] == "generic" and other["subject"] is None
    finalize([other], Settings(subject_mode="people"))
    assert other["focus_source"] == "subject"


def test_results_are_json_serialisable():
    import json
    import numpy as np
    from rawcull.pipeline import _json_default
    json.dumps({"a": np.float32(1.5), "b": np.array([1, 2])}, default=_json_default)


def test_native_crops_match_across_camera_orientations():
    try:
        import pidng  # noqa: F401
    except ImportError:
        print("skip (pip install pidng to run this test)")
        return
    import cv2
    import numpy as np
    sys.path.insert(0, str(ROOT / "tools"))
    from make_synthetic_dng import write_dng
    from rawcull.native import NativeSource
    rng = np.random.default_rng(2)
    img = np.clip(cv2.GaussianBlur(rng.normal(120, 60, (400, 600, 3)).astype(np.float32), (0, 0), 2), 0, 255).astype(np.uint8)
    rot = {1: img, 3: cv2.rotate(img, cv2.ROTATE_180), 6: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
           8: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)}
    with tempfile.TemporaryDirectory() as d:
        crops = {}
        for o, im in rot.items():
            p = os.path.join(d, f"o{o}.dng")
            write_dng(im, p, orientation=o)
            with NativeSource(p) as src:
                assert (src.nw, src.nh) == (600, 400), (o, src.nw, src.nh)
                crops[o] = src.crop_px(100, 60, 300, 220).astype(np.float32)
        for o in (3, 6, 8):
            a, b = crops[1], crops[o]
            h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
            assert np.abs(a[:h, :w] - b[:h, :w]).mean() < 6, o       # same picture whatever the camera orientation


def test_detector_smoke():
    """Needs the model (set RAWCULL_MODEL or run `--fetch-model`) and a photo of birds in RAWCULL_TEST_BIRD."""
    from rawcull import detect
    img_path = os.environ.get("RAWCULL_TEST_BIRD")
    if not (detect.model_ready() and img_path and os.path.isfile(img_path)):
        print("skip (set RAWCULL_TEST_BIRD to a bird photo and have the model installed)")
        return
    import cv2
    d = detect.Detector(threads=2).detect(cv2.imread(img_path), small=False)
    assert any(x["name"] == "bird" for x in d), d


if __name__ == "__main__":
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp())
    for fn in (test_focus_point_and_orientation, test_xmp_preserves_existing_sidecar,
               test_best_of_each_burst_is_picked, test_noisy_blurry_shots_are_not_scored_good,
               test_subject_scoring, test_results_are_json_serialisable,
               test_native_crops_match_across_camera_orientations, test_detector_smoke):
        fn()
        print("ok  ", fn.__name__)
