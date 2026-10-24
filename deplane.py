"""Star-trail stacker for DWARF raw FITS frames, with aircraft/satellite trail removal.

Idea: stars move only a pixel or two between consecutive 10 s frames, while a plane
crosses a large part of the field (or blinks). So any pixel in frame i that is much
brighter than the *dilated* max of frames i-1 and i+1 is a transient. Those pixels
are masked and replaced with the local background before max-stacking.
"""

import argparse
import glob
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from astropy.io import fits

import starlabels

# OpenCV Bayer codes are named after the 2nd row/col, so BGGR sensor -> BayerRG.
BAYER = {"BGGR": cv2.COLOR_BayerRG2BGR, "RGGB": cv2.COLOR_BayerBG2BGR,
         "GRBG": cv2.COLOR_BayerGB2BGR, "GBRG": cv2.COLOR_BayerGR2BGR}


def load_frames(folder, first, last):
    files = sorted(glob.glob(os.path.join(folder, "startrails_*.fits")))[first:last]
    hdr = fits.getheader(files[0])

    def read(f):
        return fits.getdata(f).astype(np.uint16)

    with ThreadPoolExecutor(8) as ex:
        frames = np.stack(list(ex.map(read, files)))
    return files, frames, hdr


def superpixel(raw):
    """2x2 Bayer cell sum -> half-res luminance (float32)."""
    r = raw.astype(np.float32)
    return r[0::2, 0::2] + r[0::2, 1::2] + r[1::2, 0::2] + r[1::2, 1::2]


def transient_masks(frames, black, sigma, star_motion, grow, min_area):
    """Return per-frame boolean masks (full res) of transient (plane) pixels."""
    n = len(frames)
    lum = np.stack([superpixel(f) for f in frames]) - 4 * black
    # robust noise estimate of the sky from frame-to-frame differences
    d = lum[1:min(n, 20)] - lum[0:min(n, 20) - 1]
    noise = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * star_motion + 1,) * 2)
    kg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
    dil = np.stack([cv2.dilate(x, k) for x in lum])
    masks = np.zeros((n,) + frames.shape[1:], bool)
    for i in range(n):
        nb = [j for j in (i - 1, i + 1) if 0 <= j < n]
        ref = np.max(dil[nb], axis=0)
        m = ((lum[i] - ref) > sigma * noise).astype(np.uint8)
        # drop isolated specks (twinkle, hot pixels, noise)
        cnt, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(cnt, np.uint8)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        m = cv2.dilate(keep[lab], kg)
        masks[i] = cv2.resize(m, frames.shape[2:0:-1], interpolation=cv2.INTER_NEAREST) > 0
    return masks, noise


def clean(frames, masks):
    """Replace masked pixels with the per-pixel median of a few nearby frames."""
    n = len(frames)
    out = frames.copy()
    for i in np.nonzero(masks.any(axis=(1, 2)))[0]:
        lo, hi = max(0, i - 3), min(n, i + 4)
        ys, xs = np.nonzero(masks[i])
        out[i, ys, xs] = np.median(frames[lo:hi, ys, xs], axis=0).astype(np.uint16)
    return out


class Stretch:
    """White balance on the sky, then a midtones transfer function (as in PixInsight/Siril)
    chosen so the sky lands on `sky_target`. Parameters are fixed from the final stack so
    every movie frame uses the same curve."""

    def __init__(self, stack_rgb, black, sky_target, white_pct, saturation):
        x = stack_rgb.astype(np.float32) - black
        sky = np.median(x.reshape(-1, 3), axis=0)
        self.wb = sky[1] / np.maximum(sky, 1)                   # neutral grey sky
        lum = (x * self.wb).mean(axis=2)
        self.white = np.percentile(lum, white_pct)
        s = sky[1] / self.white
        t = sky_target
        self.m = s * (1 - t) / (s - 2 * t * s + t)
        self.black, self.sat = black, saturation

    def __call__(self, rgb):
        x = np.clip((rgb.astype(np.float32) - self.black) * self.wb / self.white, 0, 1)
        m = self.m
        y = (m - 1) * x / ((2 * m - 1) * x - m)
        if self.sat != 1:
            g = y.mean(axis=2, keepdims=True)
            y = g + self.sat * (y - g)
        return (np.clip(y, 0, 1) * 255).astype(np.uint8)


def ffmpeg_writer(path, w, h, fps, *opts):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", *opts, path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder")
    p.add_argument("-o", "--out", default="out")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--last", type=int, default=None)
    p.add_argument("--sigma", type=float, default=4, help="transient threshold in noise sigmas")
    p.add_argument("--star-motion", type=int, default=3, help="max star drift/frame (half-res px)")
    p.add_argument("--grow", type=int, default=3, help="mask dilation (half-res px)")
    p.add_argument("--min-area", type=int, default=3, help="min transient blob size (half-res px)")
    p.add_argument("--no-clean", action="store_true", help="plain max stack (for comparison)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--sky", type=float, default=0.25, help="output sky brightness 0-1")
    p.add_argument("--white", type=float, default=99.95, help="white point percentile")
    p.add_argument("--saturation", type=float, default=1.3)
    p.add_argument("--fade", type=float, default=0,
                   help="movie trail half-life in frames (0 = trails never fade)")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--mask-video", action="store_true", help="also write a video of the masks")
    g = p.add_argument_group("star labels")
    g.add_argument("--labels", action="store_true",
                   help="identify stars and write startrails_labeled.mp4 with fading name tags")
    g.add_argument("--lat", type=float, default=39.05, help="site latitude in degrees (for refraction)")
    g.add_argument("--label-mag", type=float, default=4.75, help="faintest magnitude to label")
    g.add_argument("--label-hold", type=float, default=2.5, help="seconds a label stays fully visible")
    g.add_argument("--constellations", action="store_true", help="also show constellation names")
    g.add_argument("--font", help="font file to use instead of the bundled Inter")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    os.makedirs(a.out, exist_ok=True)
    files, frames, hdr = load_frames(a.folder, a.first, a.last)
    bayer = BAYER[hdr.get("BAYERPAT", "BGGR").strip()]
    black = float(np.percentile(frames[0], 0.1))
    print(f"{len(files)} frames, black level {black:.0f}")

    if not a.no_clean:
        masks, noise = transient_masks(frames, black, a.sigma, a.star_motion, a.grow, a.min_area)
        hit = masks.any(axis=(1, 2))
        print(f"noise {noise:.1f}; transients in {hit.sum()} frames, "
              f"{masks.sum() / masks[0].size:.2f} frame-equivalents of pixels masked")
        frames_c = clean(frames, masks)
    else:
        masks, frames_c = None, frames

    stack = frames_c.max(axis=0)
    rgb = cv2.cvtColor(stack, bayer)
    st = Stretch(rgb, black, a.sky, a.white, a.saturation)
    img = st(rgb)
    cv2.imwrite(os.path.join(a.out, "stack.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(a.out, "stack.png"), img)
    # linear 16-bit result for further processing elsewhere
    cv2.imwrite(os.path.join(a.out, "stack_linear16.tif"), (rgb.astype(np.uint32) * 64).clip(0, 65535).astype(np.uint16))
    print("wrote", os.path.join(a.out, "stack.jpg"))

    overlay = None
    if a.labels:
        overlay = starlabels.make_overlay(
            files, frames_c, black, superpixel, lambda f: st(cv2.cvtColor(f, bayer)), img, a.out,
            a.fps, lat=a.lat, label_mag=a.label_mag, hold=a.label_hold,
            constellations=a.constellations, font=a.font)

    h, w = stack.shape
    if not a.no_video:
        ff = ffmpeg_writer(os.path.join(a.out, "startrails.mp4"), w, h, a.fps, "-crf", "18", "-preset", "medium")
        if overlay:
            fl = ffmpeg_writer(os.path.join(a.out, "startrails_labeled.mp4"), w, h, a.fps,
                               "-crf", "18", "-preset", "medium")
        acc = np.zeros(stack.shape, np.float32)
        decay = 0.5 ** (1 / a.fade) if a.fade > 0 else 1.0
        for i, f in enumerate(frames_c):
            f = f.astype(np.float32)
            if decay < 1:
                # fade the excess over the current frame, so trails decay into
                # whatever the sky (or lit house) looks like now
                acc = f + np.maximum(acc - f, 0) * decay
            else:
                np.maximum(acc, f, out=acc)
            v = st(cv2.cvtColor(acc.astype(np.uint16), bayer))
            ff.stdin.write(v.tobytes())
            if overlay:
                fl.stdin.write(overlay.draw(v, i).tobytes())
        for proc in [ff] + ([fl] if overlay else []):
            proc.stdin.close()
            proc.wait()
            print("wrote", proc.args[-1])

    if a.mask_video and masks is not None:
        ff = ffmpeg_writer(os.path.join(a.out, "masks.mp4"), w, h, a.fps, "-crf", "23")
        for f, m in zip(frames, masks):
            v = st(cv2.cvtColor(f, bayer))
            v[m] = (0.5 * v[m] + (0, 0, 127)).astype(np.uint8)   # red overlay on detections
            ff.stdin.write(v.tobytes())
        ff.stdin.close()
        ff.wait()
        print("wrote", ff.args[-1])


if __name__ == "__main__":
    main()
