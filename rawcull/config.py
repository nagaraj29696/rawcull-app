"""Tunable settings shared by the analysis engine, the CLI and the UI."""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields


@dataclass
class Settings:
    # "fast": use the JPEG preview embedded in each RAW (about 20x faster).
    # "accurate": demosaic the RAW at half resolution (slower, closer to the real pixels).
    mode: str = "fast"
    analysis_long_edge: int = 1600     # images are scaled to this before measuring
    thumb_size: int = 256              # cached thumbnail long edge
    af_box_frac: float = 0.14          # AF measurement box width as fraction of image width

    # burst / duplicate grouping
    group_time_gap: float = 8.0        # seconds between shots that still count as one group
    group_hash_dist: int = 70          # max perceptual-hash distance (of 256 bits)

    # sharpness mapping: Laplacian variance -> 0..1 on a log scale
    focus_lo: float = 8.0
    focus_hi: float = 800.0

    # automatic reject thresholds
    reject_focus: float = 0.12
    reject_exposure: float = 0.20

    picks_per_group: int = 1
    workers: int = 0                   # 0 = number of CPU cores
    write_lightroom_xmp: bool = False  # also write IMG_0001.xmp next to IMG_0001.ARW.xmp

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
