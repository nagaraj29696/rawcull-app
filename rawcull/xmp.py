"""Write ratings and pick/reject flags into XMP sidecar files.

darktable and digiKam use `IMG_0001.ARW.xmp`; Lightroom / Adobe Camera Raw use
`IMG_0001.xmp`.  Existing sidecars are *edited in place* (only the rating/label fields are
touched) so develop settings stored by darktable or Lightroom are preserved.  The first
time an existing sidecar is modified a `.rawcull.bak` copy is kept next to it.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

NS_XMP = "http://ns.adobe.com/xap/1.0/"
NS_DK = "http://www.digikam.org/ns/1.0/"

TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="RawCull">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="{ns_xmp}"
    xmlns:digiKam="{ns_dk}"
    xmp:Rating="{rating}"
    xmp:Label="{label}"
    digiKam:PickLabel="{pick}"/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""

LABEL = {"pick": "Green", "reject": "Red", "none": ""}
PICK_LABEL = {"pick": 3, "reject": 1, "none": 0}   # digiKam: 3 accepted, 1 rejected


def sidecar_paths(raw_path, lightroom_style: bool = False) -> list[Path]:
    p = Path(raw_path)
    paths = [p.with_name(p.name + ".xmp")]
    if lightroom_style:
        paths.append(p.with_suffix(".xmp"))
    return paths


def _set_value(text: str, qname: str, value: str, ns_decl: tuple[str, str] | None) -> str:
    """Set attribute `qname` on the first rdf:Description (or replace element form)."""
    elem = re.compile(rf"<{re.escape(qname)}>.*?</{re.escape(qname)}>", re.S)
    if elem.search(text):
        return elem.sub(f"<{qname}>{value}</{qname}>", text, count=1)
    attr = re.compile(rf'({re.escape(qname)}\s*=\s*)(".*?"|\'.*?\')', re.S)
    if attr.search(text):
        return attr.sub(lambda m: f'{m.group(1)}"{value}"', text, count=1)
    m = re.search(r"<rdf:Description\b", text)
    if not m:
        return text
    insert = f' {qname}="{value}"'
    if ns_decl and f"xmlns:{ns_decl[0]}=" not in text:
        insert = f' xmlns:{ns_decl[0]}="{ns_decl[1]}"' + insert
    return text[:m.end()] + insert + text[m.end():]


def write_sidecar(sidecar: Path, rating: int, flag: str) -> None:
    rating_val = -1 if flag == "reject" else max(0, min(5, int(rating)))
    if sidecar.exists():
        bak = sidecar.with_name(sidecar.name + ".rawcull.bak")
        if not bak.exists():
            shutil.copy2(sidecar, bak)
        text = sidecar.read_text(encoding="utf-8", errors="replace")
        text = _set_value(text, "xmp:Rating", str(rating_val), ("xmp", NS_XMP))
        text = _set_value(text, "xmp:Label", LABEL[flag], ("xmp", NS_XMP))
        text = _set_value(text, "digiKam:PickLabel", str(PICK_LABEL[flag]), ("digiKam", NS_DK))
    else:
        text = TEMPLATE.format(ns_xmp=NS_XMP, ns_dk=NS_DK, rating=rating_val,
                               label=LABEL[flag], pick=PICK_LABEL[flag])
    sidecar.write_text(text, encoding="utf-8")


def write_for(raw_path, rating: int, flag: str, lightroom_style: bool = False) -> list[Path]:
    written = []
    for sc in sidecar_paths(raw_path, lightroom_style):
        write_sidecar(sc, rating, flag)
        written.append(sc)
    return written
