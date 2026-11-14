# RawCull

Finds the best shots in a folder of RAW photos (built for Sony  `.ARW`, works with other RAW formats
and JPEGs too). It scores every photo, groups bursts and near-duplicates, picks the best of each group,
and shows everything in a preview window so you can confirm or override the picks quickly.

## Install (Ubuntu / Debian)

```bash
cd rawcull-app
./install.sh          # apt packages + a private Python venv + launcher
./run.sh ~/Pictures/my-shoot
```

Manual install: `sudo apt install libimage-exiftool-perl opencv-data libxcb-cursor0`, then
`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m rawcull`.

## Using the window

1. **Open folder** (Ctrl+O). Analysis runs on all CPU cores; results are cached, so re-opening is instant.
2. Photos appear as thumbnails: green border = suggested pick, red = suggested reject, with stars and score.
3. Click a photo to see it large. The **green box** is the area whose focus was measured (the camera's own AF
   point when exiftool is installed); amber boxes are detected faces.
4. Press **Z** to load the full-resolution RAW at 100% centred on the focus box, the real focus check.
5. **[** and **]** jump from group to group (landing on each group's best shot). **G** shows only the current
   group so you can compare burst frames side by side.
6. Override with **P** pick, **X** reject, **U** unflag, **0-5** stars. Manual choices are marked with an amber
   dot and are remembered between sessions.
7. **Write XMP ratings** saves ratings and pick/reject flags as sidecar files that darktable, digiKam and
   Lightroom read. **Export picks** links or copies only the picked files into another folder.

| Key | Action | Key | Action |
|---|---|---|---|
| arrows | move | Z | 100% zoom on focus box |
| P / X / U | pick / reject / unflag | F | fit to window |
| 0-5 | star rating | O | toggle overlays |
| [  ] | previous / next group | G | only this group |

## How the score works

| Measure | What it does |
|---|---|
| Focus | Sharpness (Laplacian variance, log-scaled) measured **only at the AF point** from the camera's EXIF data. Without exiftool it uses the main face, else the sharpest part of the frame. |
| Exposure | Penalises clipped highlights, crushed shadows and very dark / very bright frames. |
| Faces | If a face is found: sharpness on the face plus a blink hint (no eyes detected on a frontal face). |
| Groups | Shots within 8 s of each other that look alike (perceptual hash) form a group. Within a group the sharpest frame is compared against its siblings, so only the best one is suggested. |
| Rating | 1-5 stars by percentile across the whole shoot; clearly soft or badly exposed frames are suggested rejects. |

Modes: **Fast** reads the JPEG preview embedded in each ARW (about 1620 px, quick); **Accurate** demosaics the RAW
at half size (slower, closer to the real pixels).

## Command line

```bash
./run.sh --cli ~/Pictures/shoot --csv ranked.csv --export-picks ~/Pictures/shoot-picks --write-xmp
```

## Limits worth knowing

* It judges **technical quality** (focus, exposure, blinks), not composition or emotion; use it to cut 1000+
  frames down to a short list, then make the final call yourself.
* Written and tested with generated sample images; I could not test it against real A6400 `.ARW` files, so check the
  first results on a small folder. If the green focus box lands in the wrong place on portrait-orientation shots,
  tell me and I will adjust the orientation mapping.
* Face / blink detection uses OpenCV Haar cascades (needs the `opencv-data` package) and is a rough hint only.
* Thresholds live in `rawcull/config.py` (`Settings`) if you want to tune groups, reject levels or sharpness scale.
* The RAW files are never modified. The cache lives in `~/.cache/rawcull`.

## Layout

```
rawcull/analysis.py   per-photo measurements      rawcull/scoring.py   scores, ratings, picks
rawcull/loader.py     RAW preview / full decode   rawcull/grouping.py  burst grouping
rawcull/exif.py       exiftool wrapper            rawcull/xmp.py       sidecar writing
rawcull/pipeline.py   parallel run, cache, export rawcull/ui.py        Qt window
rawcull/cli.py        command-line mode           tests/               python tests/test_core.py
```
