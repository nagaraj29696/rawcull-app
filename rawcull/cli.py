"""Command-line interface:  python -m rawcull --cli /path/to/photos [options]"""
from __future__ import annotations

import argparse
import sys

from .config import Settings
from .pipeline import export_csv, export_picks, run_analysis, write_xmp_all
from .scoring import effective


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rawcull", description="Rank and cull a folder of RAW photos.")
    p.add_argument("folder", nargs="?", help="folder containing ARW (or other RAW / JPEG) files")
    p.add_argument("--subject", choices=["off", "birds", "birds_animals", "animals", "people", "any"],
                   default="birds_animals",
                   help="judge focus on detected subjects of this type (default: birds_animals); off = whole frame")
    p.add_argument("--no-small-subjects", action="store_true",
                   help="skip the extra native-resolution search for small subjects (faster)")
    p.add_argument("--fetch-model", action="store_true",
                   help="download the subject-detection model (36 MB) and exit")
    p.add_argument("--mode", choices=["fast", "accurate"], default="fast",
                   help="fast = embedded JPEG previews, accurate = half-size RAW demosaic")
    p.add_argument("--workers", type=int, default=0, help="worker processes (default: all cores)")
    p.add_argument("--time-gap", type=float, default=8.0, help="seconds between shots of one group")
    p.add_argument("--picks-per-group", type=int, default=1)
    p.add_argument("--force", action="store_true", help="ignore the analysis cache")
    p.add_argument("--write-xmp", action="store_true", help="write rating/pick sidecar files")
    p.add_argument("--lightroom-xmp", action="store_true", help="also write IMG.xmp (Lightroom naming)")
    p.add_argument("--csv", metavar="FILE", help="write ranked results to a CSV file")
    p.add_argument("--export-picks", metavar="DIR", help="symlink picked files into DIR")
    p.add_argument("--copy", action="store_true", help="copy instead of symlink for --export-picks")
    p.add_argument("--top", type=int, default=20, help="rows to print (default 20)")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fetch_model:
        from . import detect

        def prog(done, total):
            sys.stderr.write(f"\r  downloading model {done / 1e6:.0f}/{total / 1e6:.0f} MB")
            sys.stderr.flush()

        path = detect.ensure_model(prog)
        sys.stderr.write("\n")
        print(f"Detector model ready: {path}")
        return 0
    if not args.folder:
        parser.error("folder is required")
    s = Settings(mode=args.mode, workers=args.workers, group_time_gap=args.time_gap,
                 picks_per_group=args.picks_per_group, write_lightroom_xmp=args.lightroom_xmp,
                 subject_mode=args.subject, small_subjects=not args.no_small_subjects)

    def progress(done, total, name):
        sys.stderr.write(f"\r  analysing {done}/{total} {name[:30]:<30}")
        sys.stderr.flush()

    notices: list[str] = []
    results = run_analysis(args.folder, s, progress, force=args.force, notices=notices,
                           model_progress=lambda d, t: sys.stderr.write(
                               f"\r  downloading detector model {d / 1e6:.0f}/{t / 1e6:.0f} MB"))
    sys.stderr.write("\n")
    for n in notices:
        print(f"! {n}")
    if not results:
        print("No images found.")
        return 1

    ok = [r for r in results if not r.get("error")]
    picks = [r for r in ok if effective(r)[1] == "pick"]
    rejects = [r for r in ok if effective(r)[1] == "reject"]
    groups = len({r["group"] for r in ok})
    if s.subject_mode != "off":
        print(f"{sum(1 for r in ok if r.get('subject'))}/{len(ok)} photos with a {s.subject_mode.replace('_', ' + ')} subject found")
    print(f"{len(ok)} images analysed, {groups} groups, {len(picks)} picks, {len(rejects)} auto-rejects, "
          f"{len(results) - len(ok)} unreadable")
    print(f"\n{'file':<22}{'score':>6}{'grp':>5}{'rank':>6}  flag    notes")
    for r in sorted(ok, key=lambda r: -r["score"])[:args.top]:
        rating, flag = effective(r)
        print(f"{r['name']:<22}{r['score']:>6.1f}{r['group']:>5}{r['group_rank']:>3}/{r['group_size']:<2}"
              f"  {flag:<7} {'; '.join(r['reasons'][:2])}")
    for r in results:
        if r.get("error"):
            print(f"! {r['name']}: {r['error']}")

    if args.csv:
        export_csv(ok, args.csv)
        print(f"\nCSV written: {args.csv}")
    if args.export_picks:
        n = export_picks(ok, args.export_picks, copy=args.copy)
        print(f"{n} picks exported to {args.export_picks}")
    if args.write_xmp:
        n = write_xmp_all(ok, lightroom_style=args.lightroom_xmp)
        print(f"XMP sidecars written for {n} files")
    return 0
