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
        res = {r["name"]: r for r in run_analysis(d, Settings(workers=2))}
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
        res = {r["name"]: r for r in run_analysis(d, Settings(workers=2))}
        picks = sorted(n for n, r in res.items() if effective(r)[1] == "pick")
        assert picks == ["DSC00001.jpg", "DSC00009.jpg"], picks      # nothing picked in the all-blurry burst
        for name in ("DSC00003.jpg", "DSC00004.jpg", "DSC00005.jpg", "DSC00006.jpg", "DSC00007.jpg",
                     "DSC00008.jpg", "DSC00011.jpg"):                 # blur sigma >= 3
            assert res[name]["score"] < 35 and effective(res[name])[1] != "pick", (name, res[name]["score"])
        assert res["DSC00009.jpg"]["score"] > 80                      # sharp, low-texture scene


if __name__ == "__main__":
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp())
    for fn in (test_focus_point_and_orientation, test_xmp_preserves_existing_sidecar,
               test_best_of_each_burst_is_picked, test_noisy_blurry_shots_are_not_scored_good):
        fn()
        print("ok  ", fn.__name__)
