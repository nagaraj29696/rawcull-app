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
        res = run_analysis(d, Settings(workers=2))
        picks = sorted(r["name"] for r in res if effective(r)[1] == "pick")
        # the sharpest, well-exposed frame of each 4-shot burst
        assert picks == ["DSC00001.jpg", "DSC00006.jpg", "DSC00009.jpg"], picks


if __name__ == "__main__":
    os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp())
    for fn in (test_focus_point_and_orientation, test_xmp_preserves_existing_sidecar,
               test_best_of_each_burst_is_picked):
        fn()
        print("ok  ", fn.__name__)
