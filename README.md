# RawCull

Finds the best shots in a folder of RAW photos (built for Sony A6400 `.ARW`, works with other RAW formats
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
| Focus | **Edge width**: how many pixels a boundary needs to go from dark to light (about 3 px = crisp, 6+ px = blurred). Unlike contrast measures it ignores scene texture and sensor noise, so a high-ISO blurry shot is not mistaken for a sharp one. Measured **at the camera's AF point** when exiftool provides it, otherwise on the main face, otherwise in the sharpest part of the frame. |
| Exposure | Penalises clipped highlights, crushed shadows and very dark / very bright frames. |
| Faces | If a face is found: sharpness on the face plus a blink hint (no eyes detected on a frontal face). |
| Groups | Shots within 8 s of each other that look alike (perceptual hash) form a group. Within a group the highest score is the suggested pick. |
| Score | Absolute, 0-100, and focus gates it: a soft frame cannot be rescued by good exposure, and a group where every frame is soft gets **no** pick. |
| Rating | 1-5 stars from the score (88+ = 5, 75+ = 4, 60+ = 3, 40+ = 2); 5 stars only for the best of a group. Soft or badly exposed frames are suggested rejects. |

If too many soft photos still look good (or good ones look soft), change **Blur limit** in the toolbar. It is the edge width
at which a photo counts as fully blurred; lower = stricter. Scores update instantly without re-analysing. The side panel
shows each photo's measured edge width so you can pick a sensible value from a few known-sharp and known-soft frames.
The status bar tells you whether the camera's AF point was found; if it says it was not, focus is measured in the sharpest area
of the frame, which cannot tell "subject blurry, background sharp" from a good shot.

Modes: **Fast** reads the JPEG preview embedded in each ARW (about 1620 px, quick); **Accurate** demosaics the RAW
at half size (slower, closer to the real pixels).

## Birds, animals and people (subject detection)

Pick **Subject** in the toolbar (default *Birds + animals*; *Off* gives the whole-frame behaviour). RawCull then:

1. **Finds the subject** with a YOLOX-S detector (COCO classes: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
   giraffe, person). The 36 MB model is downloaded once to `~/.cache/rawcull/models` (checksum verified); the app asks first.
2. **Looks harder for small birds.** If nothing was found where the camera focused, it searches 640 px and 320 px windows
   cut from the *native-resolution* RAW pixels around the AF point (or around the two sharpest regions when there is no AF
   point). A bird that is 40 px wide in the 1600 px preview is about 150 px wide in the RAW, which the detector can handle.
3. **Judges focus on the bird itself**, on native pixels (the RAW mosaic binned 2x2, so no slow demosaic), inside an ellipse
   in the subject box so background edges don't count. The side panel shows the measured edge width and the subject's size.
4. **Checks the AF point.** If the AF point is not on the subject the score is reduced and the reasons say
   *"AF point is not on the bird: focus may have landed on the background"*.

The preview draws the measured subject in green, other detected subjects in cyan and the AF point as a yellow cross.
Changing the Subject type re-scores instantly once subjects have been detected (a full pass only runs the first time).
If no subject of the chosen type is found, that photo falls back to the whole-frame focus measure and the reasons say so.
The **Blur limit** control also applies to subjects.

Honest limits: tiny or far-away birds (roughly under 60 px wide at native resolution), birds in dense foliage and
birds in flight that the detector misses will fall back to the whole-frame measure; the detector knows "bird", not
species; and the AF check needs the camera's AF point to be readable from the EXIF data.

## Command line

```bash
./run.sh --cli ~/Pictures/shoot --csv ranked.csv --export-picks ~/Pictures/shoot-picks --write-xmp
./run.sh --cli ~/Pictures/shoot --subject birds          # birds only (also: birds_animals, animals, people, any, off)
./run.sh --cli --fetch-model                              # download the detector model and exit
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
rawcull/detect.py     YOLOX detector, tiling      rawcull/subject.py   subject stage (small birds)
rawcull/native.py     native-resolution crops     tools/               synthetic test photos / DNGs
```
