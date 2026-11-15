"""Turn raw measurements into scores, ratings and pick/reject suggestions.

`finalize` is pure (no I/O) so it is cheap to re-run whenever settings change.
Scores are *absolute*: a soft photo scores low even if it is the best of a soft group.
"""
from __future__ import annotations

import math

from .config import Settings
from .detect import SUBJECT_LABELS, SUBJECT_MODES
from .grouping import assign_groups


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def focus_abs(var: float, s: Settings) -> float:
    """Laplacian variance -> 0..1 on a log scale (secondary signal, sensitive to noise)."""
    lo, hi = math.log10(1 + s.focus_lo), math.log10(1 + s.focus_hi)
    return _clamp((math.log10(1 + max(var, 0.0)) - lo) / (hi - lo))


def edge_score(width_px: float, s: Settings) -> float:
    return _clamp((s.edge_blur_px - width_px) / (s.edge_blur_px - s.edge_sharp_px))


def sharp_score(edge_w: float | None, lap_var: float, s: Settings) -> float:
    """Sharpness 0..1.  Edge width carries 85%; Laplacian variance breaks ties between frames
    of the same scene.  If no clean edges exist at all in the focus area the shot is more
    likely blurred than sharp, so the weaker signal is discounted."""
    lap = focus_abs(lap_var, s)
    if edge_w is None:
        return 0.7 * lap
    return 0.85 * edge_score(edge_w, s) + 0.15 * lap


def subject_sharp_score(d: dict, s: Settings) -> float:
    """Sharpness 0..1 of a detected subject, measured on native-resolution pixels."""
    lap = focus_abs(d.get("lap", 0.0), s)
    ew = d.get("edge_w")
    if ew is None:
        return 0.7 * lap
    edge = _clamp((s.subject_blur_px - ew) / (s.subject_blur_px - s.subject_sharp_px))
    return 0.85 * edge + 0.15 * lap


def choose_subject(r: dict, s: Settings) -> dict | None:
    """The main subject among the stored detections for the selected subject type."""
    if s.subject_mode == "off" or not r.get("subjects"):
        return None
    classes = SUBJECT_MODES.get(s.subject_mode, set())
    cands = [d for d in r["subjects"] if d["cls"] in classes]
    if not cands:
        return None

    def prominence(d):
        x0, y0, x1, y1 = d["box"]
        return d["conf"] * math.sqrt(max((x1 - x0) * (y1 - y0), 1e-6)) * (1.6 if d.get("af_inside") else 1.0)

    return max(cands, key=prominence)


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
        subj = choose_subject(r, s)
        r["subject"] = subj
        if subj is not None:
            r["focus_s"] = subject_sharp_score(subj, s)
            r["focus_source"] = "subject"
        else:
            r["focus_s"] = sharp_score(r.get("edge_w"), r["focus_var"], s)
            r["focus_source"] = "generic"
        r["exposure_s"] = exposure_score(r["clip_hi"], r["clip_lo"], r["mean_luma"])
        if r.get("face_var") is not None:
            eye_part = 0.3 if r.get("eyes_hint") == 0 else 1.0
            r["face_s"] = 0.75 * sharp_score(r.get("face_edge_w"), r["face_var"], s) + 0.25 * eye_part
        else:
            r["face_s"] = None

        if r["face_s"] is not None:
            base = 0.55 * r["focus_s"] + 0.15 * r["exposure_s"] + 0.30 * r["face_s"]
        else:
            base = 0.80 * r["focus_s"] + 0.20 * r["exposure_s"]
        # Focus gates everything: a soft frame cannot be rescued by good exposure.
        gate = min(1.0, 0.4 + 1.2 * r["focus_s"])
        score = base * gate
        if subj is not None and subj.get("af_inside") is False:
            score *= s.subject_af_penalty          # the camera focused somewhere else than on the subject
        r["score"] = round(100 * _clamp(score), 1)

    groups: dict[int, list[dict]] = {}
    for r in ok:
        groups.setdefault(r["group"], []).append(r)
    for members in groups.values():
        members.sort(key=lambda m: m["score"], reverse=True)
        for rank, m in enumerate(members, 1):
            m["group_rank"] = rank
            m["group_size"] = len(members)

    for r in ok:
        sc = r["score"]
        stars = 5 if sc >= 88 else 4 if sc >= 75 else 3 if sc >= 60 else 2 if sc >= 40 else 1
        if r["group_rank"] > s.picks_per_group:
            stars = min(stars, 4)          # 5 stars only for the best of a group
        r["auto_rating"] = stars
        if r["focus_s"] < s.reject_focus or r["exposure_s"] < s.reject_exposure:
            r["auto_flag"] = "reject"
            r["auto_rating"] = 0
        elif r["group_rank"] <= s.picks_per_group and sc >= s.pick_min_score:
            r["auto_flag"] = "pick"
        else:
            r["auto_flag"] = "none"
        r["reasons"] = _reasons(r, s)

    for r in results:
        if r.get("error"):
            r.update(score=0.0, group=-1, group_rank=0, group_size=0, auto_rating=0,
                     auto_flag="none", focus_s=0.0, exposure_s=0.0, face_s=None, subject=None,
                     reasons=[r["error"]])
    return results


def _reasons(r: dict, s: Settings) -> list[str]:
    out = []
    fs = r["focus_s"]
    subj = r.get("subject")
    if subj is not None:
        name = subj["name"]
        if fs < 0.25:
            out.append(f"Soft / out of focus on the {name}")
        elif fs < 0.5:
            out.append(f"Slightly soft on the {name}")
        elif fs > 0.8:
            out.append(f"Crisp focus on the {name}")
        if subj.get("af_inside") is False:
            out.append(f"AF point is not on the {name}: focus may have landed on the background")
        elif subj.get("af_inside"):
            out.append(f"AF point is on the {name}")
        if subj.get("edge_w") is None:
            out.append(f"Too few clear edges on the {name} to judge its sharpness reliably")
        if subj.get("native_w", 999) < 80:
            out.append(f"Small subject ({subj['native_w']} px wide at native resolution)")
    else:
        src = {"af_point": "at the camera's AF point", "face": "on the main face",
               "peak": "in the sharpest area"}[r["roi_source"]]
        if fs < 0.25:
            out.append(f"Soft / out of focus {src}")
        elif fs < 0.5:
            out.append(f"Slightly soft {src}")
        elif fs > 0.8:
            out.append(f"Crisp focus {src}")
        if s.subject_mode != "off":
            label = SUBJECT_LABELS.get(s.subject_mode, "subject").lower()
            if r.get("subjects_done"):
                out.append(f"No {label} detected: focus judged {src}")
            elif r.get("subject_error"):
                out.append("Subject detection unavailable: " + r["subject_error"])
    if r["clip_hi"] > 0.03:
        out.append(f"Highlights clipped ({r['clip_hi'] * 100:.0f}% of frame)")
    if r["clip_lo"] > 0.10:
        out.append(f"Shadows crushed ({r['clip_lo'] * 100:.0f}% of frame)")
    if r["mean_luma"] < 45:
        out.append("Underexposed")
    elif r["mean_luma"] > 205:
        out.append("Overexposed")
    if r.get("faces") and r.get("eyes_hint") == 0 and (subj is None or subj["name"] == "person"):
        out.append("Eyes not detected on the main face (possible blink)")
    if r["group_size"] > 1:
        if r["group_rank"] == 1 and r["score"] < s.pick_min_score:
            out.append(f"Best of {r['group_size']} similar shots, but still not sharp enough to pick")
        elif r["group_rank"] == 1:
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
