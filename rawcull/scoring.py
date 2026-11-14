"""Turn raw measurements into scores, ratings and pick/reject suggestions.

`finalize` is pure (no I/O) so it is cheap to re-run whenever settings change.
"""
from __future__ import annotations

import math

import numpy as np

from .config import Settings
from .grouping import assign_groups


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def focus_abs(var: float, s: Settings) -> float:
    """Map Laplacian variance to 0..1 on a log scale."""
    lo, hi = math.log10(1 + s.focus_lo), math.log10(1 + s.focus_hi)
    return _clamp((math.log10(1 + max(var, 0.0)) - lo) / (hi - lo))


def exposure_score(clip_hi: float, clip_lo: float, mean_luma: float) -> float:
    pen = clip_hi * 6 + clip_lo * 3
    if mean_luma < 45:
        pen += (45 - mean_luma) / 45 * 0.5
    elif mean_luma > 205:
        pen += (mean_luma - 205) / 50 * 0.5
    return _clamp(1.0 - pen)


def finalize(results: list[dict], s: Settings) -> list[dict]:
    """Add score / group / rank / auto rating / auto flag / reasons to each result in place."""
    ok = [r for r in results if not r.get("error")]
    assign_groups(results, s.group_time_gap, s.group_hash_dist)

    for r in ok:
        r["focus_s"] = focus_abs(r["focus_var"], s)
        r["exposure_s"] = exposure_score(r["clip_hi"], r["clip_lo"], r["mean_luma"])
        if r.get("face_var") is not None:
            eye_part = 0.3 if r.get("eyes_hint") == 0 else 1.0
            r["face_s"] = 0.75 * focus_abs(r["face_var"], s) + 0.25 * eye_part
        else:
            r["face_s"] = None

    groups: dict[int, list[dict]] = {}
    for r in ok:
        groups.setdefault(r["group"], []).append(r)

    for members in groups.values():
        best_var = max(m["focus_var"] for m in members) or 1.0
        for m in members:
            rel = m["focus_var"] / best_var
            focus = 0.65 * m["focus_s"] + 0.35 * rel if len(members) > 1 else m["focus_s"]
            if m["face_s"] is not None:
                score = 0.50 * focus + 0.20 * m["exposure_s"] + 0.30 * m["face_s"]
            else:
                score = 0.70 * focus + 0.30 * m["exposure_s"]
            m["score"] = round(100 * _clamp(score), 1)
        members.sort(key=lambda m: m["score"], reverse=True)
        for rank, m in enumerate(members, 1):
            m["group_rank"] = rank
            m["group_size"] = len(members)

    scores = np.array([r["score"] for r in ok]) if ok else np.array([0.0])
    p20, p40, p70, p90 = np.percentile(scores, [20, 40, 70, 90])

    for r in ok:
        sc = r["score"]
        r["auto_rating"] = 5 if sc >= p90 else 4 if sc >= p70 else 3 if sc >= p40 else 2 if sc >= p20 else 1
        reject = r["focus_s"] < s.reject_focus or r["exposure_s"] < s.reject_exposure
        if reject:
            r["auto_flag"] = "reject"
            r["auto_rating"] = 0
        elif r["group_rank"] <= s.picks_per_group:
            r["auto_flag"] = "pick"
        else:
            r["auto_flag"] = "none"
        r["reasons"] = _reasons(r, s)

    for r in results:
        if r.get("error"):
            r.update(score=0.0, group=-1, group_rank=0, group_size=0, auto_rating=0,
                     auto_flag="none", focus_s=0.0, exposure_s=0.0, face_s=None,
                     reasons=[r["error"]])
    return results


def _reasons(r: dict, s: Settings) -> list[str]:
    out = []
    src = {"af_point": "at the camera's AF point", "face": "on the main face",
           "peak": "in the sharpest area"}[r["roi_source"]]
    if r["focus_s"] < 0.25:
        out.append(f"Soft / out of focus {src}")
    elif r["focus_s"] > 0.75:
        out.append(f"Crisp focus {src}")
    if r["clip_hi"] > 0.03:
        out.append(f"Highlights clipped ({r['clip_hi'] * 100:.0f}% of frame)")
    if r["clip_lo"] > 0.10:
        out.append(f"Shadows crushed ({r['clip_lo'] * 100:.0f}% of frame)")
    if r["mean_luma"] < 45:
        out.append("Underexposed")
    elif r["mean_luma"] > 205:
        out.append("Overexposed")
    if r.get("faces") and r.get("eyes_hint") == 0:
        out.append("Eyes not detected on the main face (possible blink)")
    if r["group_size"] > 1:
        if r["group_rank"] == 1:
            out.append(f"Best of {r['group_size']} similar shots")
        else:
            out.append(f"Ranked {r['group_rank']} of {r['group_size']} similar shots")
    return out


def effective(r: dict) -> tuple[int, str]:
    """(rating, flag) after applying the user's manual overrides on top of the auto result."""
    user = r.get("user") or {}
    rating = user.get("rating")
    flag = user.get("flag")
    return (r.get("auto_rating", 0) if rating is None else rating,
            r.get("auto_flag", "none") if flag is None else flag)
