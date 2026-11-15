"""Write a synthetic Bayer (RGGB) DNG from an RGB image so the RAW code paths can be tested without
a real camera file.  Needs `pip install pidng`.   Usage: make_synthetic_dng.py in.jpg out.dng"""
import sys

import cv2
import numpy as np


def write_dng(rgb_bgr: np.ndarray, out_path: str, orientation: int = 1) -> None:
    from pidng.core import RAW2DNG, DNGTags, Tag
    from pidng.defs import (CalibrationIlluminant, CFAPattern, DNGVersion,
                            PhotometricInterpretation, PreviewColorSpace)
    h2, w2 = rgb_bgr.shape[:2]
    big = cv2.resize(rgb_bgr, (w2 * 2, h2 * 2), interpolation=cv2.INTER_CUBIC)     # "sensor" resolution
    H, W = big.shape[:2]
    lin = (big.astype(np.float32) / 255.0) ** 2.2
    mosaic = np.zeros((H, W), np.float32)
    mosaic[0::2, 0::2] = lin[0::2, 0::2, 2]    # R
    mosaic[0::2, 1::2] = lin[0::2, 1::2, 1]    # G
    mosaic[1::2, 0::2] = lin[1::2, 0::2, 1]    # G
    mosaic[1::2, 1::2] = lin[1::2, 1::2, 0]    # B
    data = (256 + mosaic * (65000 - 256)).clip(0, 65535).astype(np.uint16)
    t = DNGTags()
    t.set(Tag.ImageWidth, W); t.set(Tag.ImageLength, H)
    t.set(Tag.TileWidth, W); t.set(Tag.TileLength, H)
    t.set(Tag.Orientation, int(orientation))    # TIFF values: 1 normal, 3 = 180, 6 = 90 CW, 8 = 90 CCW
    t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    t.set(Tag.SamplesPerPixel, 1); t.set(Tag.BitsPerSample, 16)
    t.set(Tag.CFARepeatPatternDim, [2, 2]); t.set(Tag.CFAPattern, CFAPattern.RGGB)
    t.set(Tag.BlackLevel, 256); t.set(Tag.WhiteLevel, 65535)
    t.set(Tag.ColorMatrix1, [[19549, 10000], [-7877, 10000], [-2582, 10000], [-5724, 10000], [10121, 10000],
                             [1917, 10000], [-1267, 10000], [-110, 10000], [6621, 10000]])
    t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)
    t.set(Tag.AsShotNeutral, [[1, 1], [1, 1], [1, 1]])
    t.set(Tag.Make, "Synthetic"); t.set(Tag.Model, "Test Bayer")
    t.set(Tag.DNGVersion, DNGVersion.V1_4); t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
    t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)
    r = RAW2DNG()
    r.options(t, path="", compress=False)
    name = out_path[:-4] if out_path.lower().endswith(".dng") else out_path
    r.convert(data, filename=name)


if __name__ == "__main__":
    write_dng(cv2.imread(sys.argv[1]), sys.argv[2])
