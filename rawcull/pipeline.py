"""Folder-level orchestration: parallel analysis, on-disk cache, decisions, exports."""
from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from . import exif as exif_mod
from . import xmp
from .analysis import ALGO_VERSION, analyze_file
from .config import Settings
from .loader import list_images
from .scoring import effective, finalize

ProgressCB = Callable[[int, int, str], None]


def cache_dir_for(folder) -> Path:
    key = hashlib.sha1(str(Path(folder).resolve()).encode()).hexdigest()[:12]
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rawcull" / key
    base.mkdir(parents=True, exist_ok=True)
    return base


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def load_decisions(folder) -> dict:
    return _read_json(cache_dir_for(folder) / "decisions.json", {})


def save_decisions(folder, results: list[dict]) -> None:
    data = {r["name"]: r["user"] for r in results if r.get("user")}
    _write_json(cache_dir_for(folder) / "decisions.json", data)


def run_analysis(folder, settings: Settings, progress: ProgressCB | None = None,
                 cancel: threading.Event | None = None, force: bool = False) -> list[dict]:
    """Analyse every image in `folder`; returns finalized results (scores, groups, picks)."""
    folder = Path(folder)
    cdir = cache_dir_for(folder)
    thumb_dir = cdir / "thumbs"
    files = list_images(folder)
    total = len(files)

    cache = {} if force else _read_json(cdir / "analysis.json", {})
    results: dict[str, dict] = {}
    todo: list[Path] = []
    for p in files:
        st = p.stat()
        fp = f"{st.st_size}:{int(st.st_mtime)}"
        hit = cache.get(p.name)
        if (hit and hit.get("fingerprint") == fp and hit.get("mode") == settings.mode
                and hit.get("algo") == ALGO_VERSION and not hit.get("error") and Path(hit.get("thumb", "")).exists()):
            results[p.name] = hit
        else:
            todo.append(p)

    done = len(results)
    if progress:
        progress(done, total, "")

    if todo:
        exif_map = exif_mod.read_exif_batch(todo)
        jobs = [{"path": str(p), "exif": exif_map.get(str(p), {}),
                 "thumb_dir": str(thumb_dir), "settings": settings.to_dict()} for p in todo]
        workers = settings.workers or os.cpu_count() or 2
        ctx = mp.get_context("spawn")  # fork is unsafe once Qt threads exist
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=ctx) as ex:
            futs = {ex.submit(analyze_file, j): j for j in jobs}
            try:
                for fut in as_completed(futs):
                    if cancel is not None and cancel.is_set():
                        ex.shutdown(wait=False, cancel_futures=True)
                        break
                    res = fut.result()
                    results[res["name"]] = res
                    done += 1
                    if progress:
                        progress(done, total, res["name"])
            finally:
                if cancel is not None and cancel.is_set():
                    ex.shutdown(wait=False, cancel_futures=True)

    _write_json(cdir / "analysis.json", {k: v for k, v in results.items() if not v.get("error")})

    ordered = [results[p.name] for p in files if p.name in results]
    finalize(ordered, settings)
    decisions = load_decisions(folder)
    for r in ordered:
        if r["name"] in decisions:
            r["user"] = decisions[r["name"]]
    return ordered


# --------------------------------------------------------------------------- exports ----
def write_xmp_all(results: list[dict], lightroom_style: bool = False) -> int:
    n = 0
    for r in results:
        if r.get("error"):
            continue
        rating, flag = effective(r)
        xmp.write_for(r["path"], rating, flag, lightroom_style)
        n += 1
    return n


def export_csv(results: list[dict], dest) -> None:
    cols = ["name", "group", "group_rank", "group_size", "score", "rating", "flag",
            "focus_score", "exposure_score", "face_score", "roi_source", "reasons"]
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in sorted(results, key=lambda r: -r.get("score", 0)):
            rating, flag = effective(r)
            w.writerow([r["name"], r["group"], r["group_rank"], r["group_size"], r["score"], rating, flag,
                        round(r["focus_s"], 3), round(r["exposure_s"], 3),
                        "" if r["face_s"] is None else round(r["face_s"], 3),
                        r.get("roi_source", ""), "; ".join(r["reasons"])])


def export_picks(results: list[dict], dest, copy: bool = False) -> int:
    """Link (or copy) every picked file into `dest`.  Links keep disk usage at zero."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in results:
        if r.get("error") or effective(r)[1] != "pick":
            continue
        target = dest / r["name"]
        if target.exists() or target.is_symlink():
            target.unlink()
        if copy:
            shutil.copy2(r["path"], target)
        else:
            try:
                os.symlink(r["path"], target)
            except OSError:
                shutil.copy2(r["path"], target)
        n += 1
    return n
