"""Star identification and fading name labels for deplane's star-trail movie.

Pipeline: detect stars in every raw frame -> plate-solve one frame by triangle
matching -> fit a fixed-camera / rotating-sky model to all frames -> work out when
each star is visible -> schedule non-overlapping labels -> draw them with Pillow.
"""

import csv
import itertools
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import maximum_filter1d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

log = logging.getLogger("deplane.labels")

HERE = os.path.dirname(os.path.abspath(__file__))
SIDEREAL = 360.98564736629 / 86400          # deg per second
PARAMS = ["theta0", "omega", "cam_az", "cam_alt", "roll", "f", "cx", "cy", "k1", "k2"]
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
# Orion's caption waits for the shoulders, Rigel and the belt (Saiph rises much later)
ORION_KEY = {27989, 24436, 25336, 26311, 26727, 25930}
MAX_LABELS = 15                             # most labels on screen at once


class SolveError(Exception):
    pass


# ---------------------------------------------------------------- catalog

class Catalog:
    def __init__(self, maxmag=6.0):
        with open(os.path.join(HERE, "data", "stars.csv"), encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if float(r["mag"]) <= maxmag]
        self.hip = np.array([int(r["hip"]) for r in rows])
        self.ra = np.array([float(r["ra"]) for r in rows])
        self.dec = np.array([float(r["dec"]) for r in rows])
        self.mag = np.array([float(r["mag"]) for r in rows])
        self.name = [r["name"] for r in rows]
        self.bayer = [r["bayer"] for r in rows]
        self.flam = [r["flam"] for r in rows]
        self.con = [r["con"] for r in rows]
        with open(os.path.join(HERE, "data", "constellations.csv"), encoding="utf-8") as f:
            self.con_name = {r["con"]: r["name"] for r in csv.DictReader(f)}

    def __len__(self):
        return len(self.mag)

    def label(self, i):
        """(main text, smaller designation) for star i, or None if it has neither."""
        con = self.con[i]
        if con == "Ori" and self.bayer[i] in ("θ1", "θ2"):
            return "Orion Nebula", "M42"
        b = self.bayer[i]
        if b:
            k = len(b.rstrip("0123456789"))
            des = f"{b[:k]}{b[k:].translate(SUP)} {con}"
        elif self.flam[i]:
            des = f"{self.flam[i]} {con}"
        else:
            des = ""
        if self.name[i]:                    # only IAU-WGSN names are in the catalog
            return self.name[i], des
        return (des, "") if des else None


def unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


# ---------------------------------------------------------------- detection

def detect(lum):
    """Point sources in a half-res luminance frame -> (N, 3) [x, y, flux] in full-res px."""
    # white top-hat: follows sharp sky/tree edges, so stars rising past them stay compact
    r = cv2.morphologyEx(lum, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    s = r[::4, ::4]
    sig = 1.4826 * np.median(np.abs(s - np.median(s)))
    n, lab, st, _ = cv2.connectedComponentsWithStats((r > 5 * sig).astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros((0, 3), np.float32)
    ys, xs = np.nonzero(lab)
    l, w = lab[ys, xs], r[ys, xs]
    fl = np.bincount(l, w, n)[1:]
    x = np.bincount(l, w * xs, n)[1:] / fl
    y = np.bincount(l, w * ys, n)[1:] / fl
    # stars are compact (Rigel ~22 px here); lit foreground makes big blobs
    st = st[1:]
    star = (st[:, cv2.CC_STAT_AREA] <= 80) & (np.maximum(st[:, 2], st[:, 3]) <= 14)
    return np.stack([2 * x + 0.5, 2 * y + 0.5, fl], 1)[star].astype(np.float32)


def detect_all(frames, black, superpixel):
    with ThreadPoolExecutor(8) as ex:
        dets = list(ex.map(lambda f: detect(superpixel(f) - 4 * black), frames))
    return keep_persistent(drop_static(dets, frames.shape[1:]))


def keep_persistent(dets, rad=3.0):
    """Keep sources seen within rad px in both neighbouring frames. Stars drift ~1-2 px
    per frame; flickering foliage, noise and plane strobes don't repeat."""
    trees = [cKDTree(d[:, :2]) if len(d) else None for d in dets]
    out = []
    for i, d in enumerate(dets):
        keep = np.ones(len(d), bool)
        for j in (i - 1, i + 1):
            if 0 <= j < len(dets) and len(d):
                keep &= trees[j].query(d[:, :2])[0] < rad if trees[j] else False
        out.append(d[keep])
    return out


def drop_static(dets, shape):
    """Remove sources that sit still for much of the session (hot pixels, ground lights).
    Stars, even Polaris, move across several pixels over the sampled frames."""
    H, W = shape
    sample = np.unique(np.linspace(0, len(dets) - 1, min(len(dets), 40)).astype(int))
    count = np.zeros((H // 2 + 1, W // 2 + 1), np.float32)
    for i in sample:
        hit = np.zeros_like(count)
        ij = (dets[i][:, 1::-1] / 2).astype(int)
        hit[ij[:, 0], ij[:, 1]] = 1
        count += cv2.dilate(hit, np.ones((3, 3), np.uint8))
    static = count >= 0.3 * len(sample)
    out = []
    for d in dets:
        ij = (d[:, 1::-1] / 2).astype(int)
        out.append(d[~static[ij[:, 0], ij[:, 1]]])
    return out


def frame_times(files):
    """Seconds since the first frame from DATE-OBS, or None if a header lacks it."""
    try:
        ts = [datetime.fromisoformat(fits.getheader(f)["DATE-OBS"]) for f in files]
    except (KeyError, ValueError):
        return None
    return np.array([(t - ts[0]).total_seconds() for t in ts])


# ---------------------------------------------------------------- model

def refract(alt):
    """True -> apparent altitude (radians), Bennett/Saemundsson."""
    a = np.degrees(alt)
    R = 1.02 / np.tan(np.radians(a + 10.3 / (a + 5.11))) / 60.0
    return alt + np.radians(np.where(a > -1.5, R, 0))


def enu(ra, dec, t, theta0, omega, lat):
    """Unit East-North-Up vectors (with refraction) at time t."""
    lat = np.radians(lat)
    ra, dec = np.radians(ra), np.radians(dec)
    H = np.radians(theta0) + np.radians(omega) * t - ra
    alt = np.arcsin(np.clip(np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(H), -1, 1))
    az = np.arctan2(-np.sin(H) * np.cos(dec), np.sin(dec) * np.cos(lat) - np.cos(dec) * np.sin(lat) * np.cos(H))
    alt = refract(alt)
    return np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], -1), np.degrees(alt)


def cam_axes(az0, alt0, roll):
    a0, e0, r0 = np.radians([az0, alt0, roll])
    fwd = np.array([np.cos(e0) * np.sin(a0), np.cos(e0) * np.cos(a0), np.sin(e0)])
    right = np.array([np.cos(a0), -np.sin(a0), 0.])
    up = np.cross(right, fwd)
    return np.cos(r0) * right + np.sin(r0) * up, -np.sin(r0) * right + np.cos(r0) * up, fwd


def project(p, ra, dec, t, lat):
    """Catalog position -> pixel. Returns px, py, Z (>0 in front of camera), apparent alt."""
    th0, om, az0, alt0, roll, f, cx, cy, k1, k2 = p
    V, alt = enu(ra, dec, t, th0, om, lat)
    r2, u2, fwd = cam_axes(az0, alt0, roll)
    X, Y, Z = V @ r2, V @ u2, V @ fwd
    x, y = X / Z, Y / Z
    rr = x * x + y * y
    d = 1 + k1 * rr + k2 * rr * rr
    return cx + f * x * d, cy - f * y * d, Z, alt


def kabsch(A, B):
    """Rotation R minimising sum |R a_i - b_i|^2 for unit-vector rows A, B."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def cam_vectors(xy, f, cx, cy):
    v = np.stack([(xy[:, 0] - cx) / f, -(xy[:, 1] - cy) / f, np.ones(len(xy))], 1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def cam_rotation(cv, V):
    """Matrix M with V ~ M @ cv, columns = camera (right, up, forward) axes in V's frame.
    Image right/up/forward is a left-handed basis, so fit a proper rotation on x-flipped
    camera vectors and flip back."""
    F = np.diag([-1.0, 1, 1])
    return kabsch(cv @ F, V) @ F


def fit(p, free, ra, dec, t, x, y, lat, f_scale=2.0):
    free = np.asarray(free)

    def res(q):
        pp = p.copy()
        pp[free] = q
        px, py, Z, _ = project(pp, ra, dec, t, lat)
        r = np.concatenate([px - x, py - y])
        r[np.concatenate([Z, Z]) <= 0.05] = 1e3
        return r

    s = least_squares(res, p[free], loss="soft_l1", f_scale=f_scale, x_scale="jac")
    out = p.copy()
    out[free] = s.x
    return out, np.hypot(*s.fun.reshape(2, -1))


# ---------------------------------------------------------------- solving

def triangle_solve(seeds, det, cat, W, H, f_guess):
    """Plate-solve one frame from its brightest moving sources (seeds): returns
    (R camera->equatorial, f, inlier pairs [(index into det, cat_i)])."""
    top = seeds[np.argsort(-seeds[:, 2])[:12], :2]
    bright = np.nonzero(cat.mag <= 3.5)[0]
    Vc = unit(cat.ra[bright], cat.dec[bright])
    ang = np.degrees(np.arccos(np.clip(Vc @ Vc.T, -1, 1)))
    probe = np.nonzero(cat.mag <= 4.5)[0]
    Vp = unit(cat.ra[probe], cat.dec[probe])
    # score against bright detections only: with hundreds of faint ones, chance matches swamp it
    tree = cKDTree(seeds[np.argsort(-seeds[:, 2])[:80], :2])
    full = cKDTree(det[:, :2])
    rad = 10
    s0 = f_guess * np.pi / 180
    best = (0, None)
    seen = set()
    for i, j, k in itertools.combinations(range(len(top)), 3):
        pt = top[[i, j, k]]
        dij, djk, dik = (np.hypot(*(pt[a] - pt[b])) for a, b in [(0, 1), (1, 2), (0, 2)])
        if min(dij, djk, dik) < 40:
            continue
        for s in s0 * np.arange(0.7, 1.36, 0.08):
            tol = 0.08
            for a, b in np.argwhere(np.abs(ang - dij / s) < tol * dij / s):
                c = np.nonzero((np.abs(ang[b] - djk / s) < tol * djk / s) &
                               (np.abs(ang[a] - dik / s) < tol * dik / s))[0]
                for c in c.tolist():
                    if c in (a, b) or (i, j, k, a, b, c) in seen:
                        continue
                    seen.add((i, j, k, a, b, c))
                    f = s * 180 / np.pi
                    R = cam_rotation(cam_vectors(pt, f, W / 2, H / 2), Vc[[a, b, c]])
                    v = Vp @ R                              # equatorial -> camera (rows)
                    ok = v[:, 2] > 0.3
                    px = W / 2 + f * v[ok, 0] / v[ok, 2]
                    py = H / 2 - f * v[ok, 1] / v[ok, 2]
                    inb = (px > 0) & (px < W) & (py > 0) & (py < H)
                    if inb.sum() < 5:
                        continue
                    d, _ = tree.query(np.stack([px[inb], py[inb]], 1), distance_upper_bound=rad)
                    n = int((d < rad).sum())
                    if n > best[0]:
                        best = (n, (R, f))
    if best[1] is None or best[0] < 8:
        raise SolveError(f"triangle match failed (best {best[0]} inliers)")
    R, f = best[1]
    v = Vp @ R
    ok = np.nonzero(v[:, 2] > 0.3)[0]
    px = W / 2 + f * v[ok, 0] / v[ok, 2]
    py = H / 2 - f * v[ok, 1] / v[ok, 2]
    d, di = full.query(np.stack([px, py], 1), distance_upper_bound=rad)
    pairs = {}
    for ci, dd, j in zip(probe[ok], d, di):
        if dd < rad and (j not in pairs or dd < pairs[j][1]):
            pairs[j] = (ci, dd)
    log.info(f"triangle match: {best[0]} inliers, scale {f * np.pi / 180:.1f} px/deg")
    return R, f, [(j, ci) for j, (ci, _) in pairs.items()]


def axes_to_params(R):
    """Camera axes (columns r2, u2, fwd in ENU) -> az, alt, roll in degrees."""
    r2, fwd = R[:, 0], R[:, 2]
    az = np.arctan2(fwd[0], fwd[1])
    alt = np.arcsin(np.clip(fwd[2], -1, 1))
    right = np.array([np.cos(az), -np.sin(az), 0.])
    up = np.cross(right, fwd)
    return np.degrees(az) % 360, np.degrees(alt), np.degrees(np.arctan2(r2 @ up, r2 @ right))


def associate(p, cat, idx, dets, t, frames_i, lat, W, H, tol_bright, tol_faint):
    """Match predicted catalog stars to detections in the given frames."""
    out = []
    for fi in frames_i:
        det = dets[fi]
        if len(det) < 2:
            continue
        px, py, Z, alt = project(p, cat.ra[idx], cat.dec[idx], t[fi], lat)
        ok = (Z > 0.2) & (px > -10) & (px < W + 10) & (py > -10) & (py < H + 10)
        if not ok.any():
            continue
        tree = cKDTree(det[:, :2])
        P = np.stack([px[ok], py[ok]], 1)
        d, j = tree.query(P, k=2)
        ci = idx[ok]
        tol = np.where(cat.mag[ci] < 4, tol_bright, tol_faint)
        good = (d[:, 0] < tol) & ~((d[:, 1] < 1.5 * d[:, 0]) & (d[:, 0] > 1.5))
        claim = {}
        for g in np.nonzero(good)[0]:
            score = d[g, 0] + 0.3 * cat.mag[ci[g]]
            if j[g, 0] not in claim or score < claim[j[g, 0]][0]:
                claim[j[g, 0]] = (score, g)
        for jj, (_, g) in claim.items():
            out.append((fi, ci[g], det[jj, 0], det[jj, 1]))
    return np.array(out) if out else np.zeros((0, 4))


def scan_theta0(p, m, cat, t, lat):
    """Choose theta0 by refraction: for each value, refit the camera and keep the lowest
    residual. If refraction can't decide (no stars near the horizon), prefer a level camera."""
    m = m[::max(1, len(m) // 6000)]
    fi, mc = m[:, 0].astype(int), m[:, 1].astype(int)
    args = cat.ra[mc], cat.dec[mc], t[fi], m[:, 2], m[:, 3]

    def trial(th0, free):
        q = p.copy()
        q[0] = th0 % 360
        V, _ = enu(args[0], args[1], args[2], q[0], q[1], lat)
        q[2:5] = axes_to_params(cam_rotation(cam_vectors(m[:, 2:4], q[5], q[6], q[7]), V))
        q, rr = fit(q, free, *args, lat)
        return np.median(rr), q                 # the median has a clean minimum; means are noisy

    free = [2, 3, 4, 5, 6, 7, 8, 9]
    coarse = [trial(th0, free) for th0 in np.arange(0, 360, 10.0)]
    cost = np.array([c for c, _ in coarse])
    if np.median(cost) < 1.05 * cost.min():
        log.warning("theta0 weakly determined by refraction; assuming a level camera")
        return min((r for r in coarse if r[0] < 1.05 * cost.min()), key=lambda r: abs(r[1][4]))[1]
    th = coarse[int(np.argmin(cost))][1][0]
    fine = [trial(th0, free) for th0 in th + np.arange(-10, 10.1, 1.0)]
    c, q = min(fine, key=lambda r: r[0])
    log.info(f"theta0 scan: {q[0]:.0f} deg (cost {c:.2f}, coarse median {np.median(cost):.2f})")
    return q


def solve(dets, times, cat, W, H, lat, f_guess):
    n = len(dets)
    known_t = times is not None
    t = times if known_t else np.arange(n, dtype=float) * 10.0
    # reference frame: most star-like sources in the middle half (avoids twilight / setup
    # frames, and frames where lit foreground dominates)
    trees = [cKDTree(d[:, :2]) if len(d) else None for d in dets]

    def tracked(i, k=15):
        """Sources that can be followed +-k frames along a steadily moving track."""
        d = dets[i]
        if not len(d):
            return d
        ends, hits = [], np.zeros(len(d))
        for step in (1, -1):
            pos = d[:, :2].copy()
            for j in range(i + step, i + step * (k + 1), step):
                if not (0 <= j < n) or trees[j] is None:
                    continue
                dd, jj = trees[j].query(pos)
                ok = dd < 3
                pos[ok] = dets[j][jj[ok], :2]
                hits += ok
            ends.append(pos)
        return d[(hits >= 1.6 * k) & (np.hypot(*(ends[0] - ends[1]).T) > 10)]

    mid = range(n // 4, max(n // 4 + 1, 3 * n // 4), max(1, n // 100))
    ref = max(mid, key=lambda i: len(tracked(i)))
    R, f, pairs = triangle_solve(tracked(ref), dets[ref], cat, W, H, f_guess)
    dj, ci = map(np.array, zip(*pairs))
    cv = cam_vectors(dets[ref][dj, :2], f, W / 2, H / 2)
    # theta0 is degenerate with the camera rotation (only refraction tells them apart), so
    # start from any theta0, fit with it held fixed, then pick it by a scan once the
    # distortion is known
    V, _ = enu(cat.ra[ci], cat.dec[ci], t[ref], 0.0, SIDEREAL, lat)
    p = np.array([0.0, SIDEREAL, *axes_to_params(cam_rotation(cv, V)), f, W / 2, H / 2, 0, 0])

    sample = np.unique(np.r_[np.arange(0, n, 5), n - 1])
    free_nodist = [2, 3, 4, 5, 6, 7] + ([] if known_t else [1])
    free_all = free_nodist + [8, 9]
    rounds = [(3.5, 15, 15, free_nodist), (4.5, 10, 6, free_nodist), (6.0, 10, 4.5, free_all)]
    prev = None
    for k in range(8):
        maxmag, tb, tf, free = rounds[min(k, len(rounds) - 1)]
        idx = np.nonzero(cat.mag <= maxmag)[0]
        m = associate(p, cat, idx, dets, t, sample, lat, W, H, tb, tf)
        if len(m) < 20:
            raise SolveError(f"only {len(m)} star matches")
        if k == len(rounds):
            p = scan_theta0(p, m, cat, t, lat)
            m = associate(p, cat, idx, dets, t, sample, lat, W, H, tb, tf)
        fi, mc = m[:, 0].astype(int), m[:, 1].astype(int)
        p, rr = fit(p, free, cat.ra[mc], cat.dec[mc], t[fi], m[:, 2], m[:, 3], lat)
        key = set(map(tuple, m[:, :2].astype(int)))
        log.info(f"fit round {k}: {len(m)} matches, {len(set(mc))} stars, "
                 f"median {np.median(rr):.2f} px, p90 {np.percentile(rr, 90):.2f} px")
        if k > len(rounds) and len(key ^ prev) < 0.01 * len(key):
            break
        prev = key
    per_frame = len(m) / len(sample)
    if np.median(rr) > 3 or per_frame < 5:
        raise SolveError(f"no convincing fit (median residual {np.median(rr):.1f} px, "
                         f"{per_frame:.1f} matches per frame)")
    # per-star constant offset (median residual), for well-matched stars
    px, py, _, _ = project(p, cat.ra[mc], cat.dec[mc], t[fi], lat)
    offsets = {}
    for c in np.unique(mc):
        s = mc == c
        if s.sum() >= 10:
            offsets[c] = np.median(np.stack([m[s, 2] - px[s], m[s, 3] - py[s]], 1), 0)
    return dict(p=p, t=t, known_t=known_t, ref=ref, matches=m, resid=rr, offsets=offsets)


# ---------------------------------------------------------------- visibility & selection

def visibility(sol, cat, idx, dets, lat, W, H):
    """Per-frame predicted positions (with offsets) and a visible flag, for stars idx."""
    p, t = sol["p"], sol["t"]
    n = len(dets)
    off = np.array([sol["offsets"].get(c, (0, 0)) for c in idx])
    px, py, Z, alt = project(p, cat.ra[idx][:, None], cat.dec[idx][:, None], t[None, :], lat)
    px, py = px + off[:, :1], py + off[:, 1:]
    inb = (Z > 0.2) & (alt > 0) & (px >= 4) & (px < W - 4) & (py >= 4) & (py < H - 4)
    tol = np.where(cat.mag[idx] < 4, 6.0, 4.0)        # model error grows toward the edges
    vis = np.zeros(px.shape, bool)
    for i in range(n):
        if len(dets[i]) == 0:
            continue
        d, _ = cKDTree(dets[i][:, :2]).query(np.stack([px[:, i], py[:, i]], 1))
        vis[:, i] = (d < tol) & inb[:, i]
    # stars low down flicker in and out (twinkle, branches): seen within +-5 frames counts
    vis = maximum_filter1d(vis, 11, axis=1) & inb
    return px, py, vis


def select(cat, idx, px, py, vis, fps, label_mag):
    win = max(1, int(fps))
    out = []
    for r, c in enumerate(idx):
        lab = cat.label(c)
        v = vis[r]
        if lab is None or v.sum() < 10:
            continue
        good = np.nonzero(np.convolve(v, np.ones(win) / win, mode="valid") >= 0.7)[0]
        if len(good) == 0:
            continue
        a, b = good[0], good[-1] + win - 1
        # judge steadiness on the longest stretch (gaps under 1 s allowed): a star glimpsed
        # through branches before it clears them shouldn't be rejected for the gap
        runs = np.split(good, np.nonzero(np.diff(good) > win)[0] + 1)
        run = max(runs, key=lambda r: r[-1] - r[0])
        frac, nvis = v[run[0]:run[-1] + win].mean(), int(v.sum())
        m = cat.mag[c]
        nebula = lab[0] == "Orion Nebula"
        if not ((m <= label_mag and frac >= 0.85 and nvis >= 80) or
                ((m < 2.5 or nebula) and frac >= 0.6)):
            continue
        out.append(dict(c=c, mag=m, label=lab[0], sub=lab[1], con=cat.con[c], hip=int(cat.hip[c]),
                        start=int(a), end=int(b), frac=float(frac), nvis=nvis, x=px[r], y=py[r]))
    out.sort(key=lambda o: o["mag"])
    kept = []
    for o in out:
        dup = False
        for k in kept:
            a, b = max(o["start"], k["start"]), min(o["end"], k["end"])
            if b > a and np.median(np.hypot(o["x"][a:b] - k["x"][a:b], o["y"][a:b] - k["y"][a:b])) < 10:
                dup = True
                break
        if not dup:
            kept.append(o)
    return kept


# ---------------------------------------------------------------- scheduling & drawing

class Fonts:
    def __init__(self, override=None):
        d = os.path.join(HERE, "fonts")
        self.paths = {k: override or os.path.join(d, f"Inter-{k}.ttf") for k in ("SemiBold", "Regular", "Medium")}
        self.cache = {}

    def __call__(self, kind, size):
        k = (kind, int(size))
        if k not in self.cache:
            self.cache[k] = ImageFont.truetype(self.paths[kind], int(size))
        return self.cache[k]


def text_size(m):
    return 30 if m < 1 else 26 if m < 2.3 else 22 if m < 3.6 else 18


def overlaps(A, B):
    return not (A[2] < B[0] or B[2] < A[0] or A[3] < B[1] or B[3] < A[1])


def label_box(o, f, side):
    """Bounding box of a label's text and its ring."""
    x, y, w, h = o["x"][f], o["y"][f], o["w"], o["h"]
    if side == 0:
        return (x - 12, y - 12, x + 14 + w, y + 4 + h)
    return (x - 14 - w, y - 4 - h, x + 12, y + 12)


def draw_label(d, fonts, o, x, y, a, side):
    s = o["fs"]
    fb, fs = fonts("SemiBold", s), fonts("Regular", s * 0.62)
    main = (255, 244, 225) if o["mag"] < 1 else (240, 240, 240)
    ca = int(255 * a * (1.0 if o["mag"] < 3.6 else 0.85))
    r = 7 if o["mag"] >= 2.3 else 10
    d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 220, 140, int(ca * 0.8)), width=2)
    w = fb.getlength(o["label"])
    ws = fs.getlength("  " + o["sub"]) if o["sub"] else 0
    tx, ty = (x + 14, y + 4) if side == 0 else (x - 14 - (w + ws + 4), y - 4 - s * 1.15)
    d.text((tx, ty), o["label"], font=fb, fill=main + (ca,), stroke_width=2,
           stroke_fill=(0, 0, 0, int(ca * 0.55)))
    if o["sub"]:
        d.text((tx + w, ty + s * 0.30), "  " + o["sub"], font=fs, fill=(190, 205, 225, int(ca * 0.9)),
               stroke_width=2, stroke_fill=(0, 0, 0, int(ca * 0.5)))


class Overlay:
    def __init__(self, stars, n, W, H, fps, hold, fonts, constellations, cat):
        self.W, self.H, self.n, self.fonts = W, H, n, fonts
        sec = lambda s: max(1, int(round(s * fps)))
        self.fin, self.fout = sec(0.4), sec(1.5)
        hold_f = lambda m: sec(hold * (1.2 if m < 2.3 else 0.93 if m < 3.6 else 0.73))
        for o in stars:
            s = text_size(o["mag"])
            w = fonts("SemiBold", s).getlength(o["label"])
            if o["sub"]:
                w += fonts("Regular", s * 0.62).getlength("  " + o["sub"])
            o["w"], o["h"], o["fs"] = w + 4, s * 1.15, s
            o["life"] = self.fin + hold_f(o["mag"]) + self.fout
        # stars already up at the start would all pop in together: stagger them
        early = [o for o in stars if o["start"] <= sec(0.2)]
        stars = [o for o in stars if not (o["start"] <= sec(0.2) and o["mag"] > 4.4)]
        for k, o in enumerate(sorted((o for o in early if o["mag"] <= 4.4), key=lambda o: o["mag"])):
            o["start"] = sec(0.33) + k * sec(0.67)
        # brightest first, so faint stars give way to bright ones rather than the reverse;
        # the Orion Nebula is faint as a "star" but worth showing
        prio = lambda o: min(o["mag"], 2.5) if o["label"] == "Orion Nebula" else o["mag"]
        self.labels = self._place(sorted(stars, key=prio), sec)
        self.caps = self._captions(cat, sec) if constellations else []

    def _place(self, order, sec):
        placed = []
        W, H, n = self.W, self.H, self.n
        for o in order:
            life, ok = o["life"], False
            for delay in range(0, sec(4) + 1, sec(0.33)):
                s0 = o["start"] + delay
                if s0 + life >= n:
                    if delay:
                        break
                    s0 = max(0, n - 1 - life)
                if delay and s0 > o["end"] - sec(0.67):
                    break
                for side in (0, 1):
                    bad = False
                    for f in range(s0, min(n, s0 + life), sec(0.2)):
                        b = label_box(o, f, side)
                        live = [q for q in placed if q["t0"] <= f < q["t0"] + q["life"]]
                        if b[0] < 4 or b[2] > W - 4 or b[1] < 4 or b[3] > H - 4 or len(live) >= MAX_LABELS or any(
                                overlaps(b, label_box(q, f, q["side"])) for q in live):
                            bad = True
                            break
                    if not bad:
                        o["t0"], o["side"], ok = s0, side, True
                        break
                if ok:
                    break
            if ok:
                placed.append(o)
            elif o["mag"] < 2.5:
                o["t0"], o["side"] = o["start"], 0
                placed.append(o)
        return placed

    def _captions(self, cat, sec):
        W, H, n = self.W, self.H, self.n
        font = self.fonts("Medium", 32)
        caps = []
        for con in sorted({o["con"] for o in self.labels}):
            mem = [o for o in self.labels if o["con"] == con]
            key = [o for o in mem if o["hip"] in ORION_KEY]
            if con == "Ori" and key:
                t0 = max(o["start"] for o in key) + sec(1.5)
            else:
                t0 = min(mem, key=lambda o: o["mag"])["t0"] + sec(1)
            name = " ".join(cat.con_name.get(con, con).upper())
            caps.append(dict(name=name, mem=mem, t0=t0, life=sec(1) + sec(4) + sec(2)))
        placed = []
        for cp in caps:
            f0 = min(cp["t0"], n - 1)
            ms = [o for o in cp["mem"] if o["start"] <= f0 <= o["end"]]
            if not ms:
                continue
            p0 = np.mean([(o["x"][f0], o["y"][f0]) for o in ms], 0)
            dx = np.mean([o["x"] - o["x"][f0] for o in ms], 0)
            dy = np.mean([o["y"] - o["y"][f0] for o in ms], 0)
            cw = self._tracked_w(font, cp["name"])
            best = None
            for ox in range(-400, 401, 40):
                for oy in range(-320, 321, 40):
                    sc, bad = np.hypot(ox, oy) * 0.02, False
                    for f in range(cp["t0"], min(n, cp["t0"] + cp["life"]), 8):
                        x, y = p0[0] + ox + dx[f], p0[1] + oy + dy[f]
                        B = (x - cw / 2, y - 20, x + cw / 2, y + 20)
                        if B[0] < 8 or B[2] > W - 8 or B[1] < 8 or B[3] > H * 2 / 3:
                            bad = True
                            break
                        for o in self.labels:
                            if o["t0"] <= f < o["t0"] + o["life"]:
                                sc += overlaps(B, label_box(o, f, o["side"]))
                        for q in placed:
                            if q["t0"] <= f < q["t0"] + q["life"]:
                                Q = (q["x"][f] - q["cw"] / 2, q["y"][f] - 20, q["x"][f] + q["cw"] / 2, q["y"][f] + 20)
                                sc += 5 * overlaps(B, Q)
                    if not bad and (best is None or sc < best[0]):
                        best = (sc, ox, oy)
            if best is None:
                continue
            cp.update(x=p0[0] + best[1] + dx, y=p0[1] + best[2] + dy, cw=cw)
            placed.append(cp)
        return placed

    @staticmethod
    def _tracked_w(font, txt, tr=7):
        return sum(font.getlength(ch) + (tr * 2.2 if ch == " " else tr) for ch in txt) - tr

    def _alpha(self, t0, life, fin, fout, f):
        t = f - t0
        if t < 0 or t >= life:
            return 0.0
        if t < fin:
            return t / fin
        if t > life - fout:
            return (life - t) / fout
        return 1.0

    def draw(self, bgr, f):
        """Composite the labels for movie frame f onto a BGR uint8 frame."""
        layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        font = self.fonts("Medium", 32)
        busy = False
        for cp in self.caps:
            a = self._alpha(cp["t0"], cp["life"], cp["life"] // 7, 2 * cp["life"] // 7, f)
            if a <= 0:
                continue
            busy = True
            x, y = cp["x"][f] - cp["cw"] / 2, cp["y"][f] - 20
            for ch in cp["name"]:
                d.text((x, y), ch, font=font, fill=(170, 200, 255, int(110 * a)))
                x += font.getlength(ch) + (7 * 2.2 if ch == " " else 7)
        for o in self.labels:
            a = self._alpha(o["t0"], o["life"], self.fin, self.fout, f)
            if a > 0:
                busy = True
                draw_label(d, self.fonts, o, o["x"][f], o["y"][f], a, o["side"])
        if not busy:
            return bgr
        img = Image.alpha_composite(Image.fromarray(bgr[..., ::-1]).convert("RGBA"), layer)
        return np.ascontiguousarray(np.asarray(img.convert("RGB"))[..., ::-1])

    def active(self, f):
        return sum(self._alpha(o["t0"], o["life"], self.fin, self.fout, f) > 0 for o in self.labels)


def label_stack(img, stars, fonts):
    """Label each identified star at the end of its trail on the still stack."""
    H, W = img.shape[:2]
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    boxes = []
    for o in sorted(stars, key=lambda o: o["mag"]):
        # the latest point of the trail well inside the frame, stepping back on collisions
        x, y = o["x"][:o["end"] + 1], o["y"][:o["end"] + 1]
        inside = np.nonzero((x > 30) & (x < W - 30) & (y > 30) & (y < H - 30))[0]
        done = False
        for f in inside[::-1][:5 * 30:30]:
            for side in (0, 1):
                b = label_box(o, f, side)
                if b[0] >= 4 and b[2] <= W - 4 and b[1] >= 4 and b[3] <= H - 4 and not any(overlaps(b, q) for q in boxes):
                    boxes.append(b)
                    draw_label(d, fonts, o, o["x"][f], o["y"][f], 1.0, side)
                    done = True
                    break
            if done:
                break
    out = Image.alpha_composite(Image.fromarray(img[..., ::-1]).convert("RGBA"), layer)
    return np.asarray(out.convert("RGB"))[..., ::-1]


# ---------------------------------------------------------------- entry point

def make_overlay(files, frames, black, superpixel, render, stack_img, out_dir, fps,
                 lat=39.05, label_mag=4.75, hold=2.5, constellations=False, font=None):
    """Solve the session and return an Overlay for the movie, or None if solving fails.
    render(raw_frame) -> stretched BGR image; also writes solution.json, solve_check.jpg
    and stack_labeled.jpg into out_dir."""
    n, (H, W) = len(frames), frames.shape[1:]
    hdr = fits.getheader(files[0])
    f_guess = hdr.get("FOCALLEN", 7.0) / (hdr.get("XPIXSZ", 2.9) / 1000)
    cat = Catalog(6.0)
    dets = detect_all(frames, black, superpixel)
    log.info(f"detected {np.median([len(d) for d in dets]):.0f} stars per frame (median)")
    times = frame_times(files)
    if times is None:
        log.warning("no DATE-OBS in headers: fitting the sky rotation rate")
    try:
        sol = solve(dets, times, cat, W, H, lat, f_guess)
    except SolveError as e:
        log.warning(f"star labels skipped: {e}")
        return None
    rr = sol["resid"]
    med = float(np.median(rr))

    fonts = Fonts(font)
    idx = np.nonzero(cat.mag <= max(label_mag, 5.0))[0]
    px, py, vis = visibility(sol, cat, idx, dets, lat, W, H)
    stars = select(cat, idx, px, py, vis, fps, label_mag)
    overlay = Overlay(stars, n, W, H, fps, hold, fonts, constellations, cat)
    log.info(f"{len(stars)} stars selected, {len(overlay.labels)} labels placed, "
             f"{len(overlay.caps)} constellation captions; "
             f"max {max(overlay.active(f) for f in range(0, n, 5))} labels on screen")

    _report_unmatched(sol, dets, cat, lat)
    m = sol["matches"]
    p = sol["p"]
    labelled = {id(o) for o in overlay.labels}
    json.dump(dict(
        params={k: float(v) for k, v in zip(PARAMS, p)},
        units=dict(theta0="deg (LST at first frame)", omega="deg/s" if sol["known_t"] else "deg/(10 s)",
                   cam_az="deg", cam_alt="deg", roll="deg", f="px", cx="px", cy="px"),
        lat=lat, ref_frame=int(sol["ref"]), frames=n,
        residual_px=dict(median=med, p90=float(np.percentile(rr, 90)), rms=float(np.sqrt(np.mean(rr ** 2)))),
        matched_stars=len(set(m[:, 1].astype(int))),
        stars=[dict(hip=o["hip"], label=o["label"], sub=o["sub"], mag=o["mag"], first=o["start"],
                    last=o["end"], labelled=id(o) in labelled) for o in sorted(stars, key=lambda o: o["start"])],
    ), open(os.path.join(out_dir, "solution.json"), "w"), indent=1, ensure_ascii=False)

    ref = sol["ref"]
    chk = render(frames[ref]).copy()
    for x, y, _ in dets[ref]:
        cv2.circle(chk, (int(round(x)), int(round(y))), 3, (0, 255, 0), 1)
    sx, sy, Z, alt = project(p, cat.ra, cat.dec, sol["t"][ref], lat)
    for x, y, z, al, mg in zip(sx, sy, Z, alt, cat.mag):
        if z > 0.2 and al > 0 and 0 <= x < W and 0 <= y < H and mg <= 5.5:
            cv2.circle(chk, (int(round(x)), int(round(y))), 9 if mg < 3 else 6, (0, 200, 255), 1)
    cv2.imwrite(os.path.join(out_dir, "solve_check.jpg"), chk, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(out_dir, "stack_labeled.jpg"), label_stack(stack_img, stars, fonts),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    return overlay


def _report_unmatched(sol, dets, cat, lat):
    """Log bright sources that move with the sky but match no catalog star (planets etc.)."""
    m, p, t = sol["matches"], sol["p"], sol["t"]
    if len(m) == 0:
        return
    mc = m[:, 1].astype(int)
    flux_cut = None
    hits = []
    for fi in np.unique(m[:, 0].astype(int))[::4]:
        det, mm, mci = dets[fi], m[m[:, 0] == fi], mc[m[:, 0] == fi]
        if len(det) == 0 or len(mm) < 5 or fi + 30 >= len(dets):
            continue
        tree = cKDTree(det[:, :2])
        used = set(tree.query(mm[:, 2:4])[1])
        if flux_cut is None:
            flux_cut = np.percentile(det[list(used), 2], 90)
        for j in np.nonzero(det[:, 2] >= flux_cut)[0]:
            if j in used:
                continue
            # follow it with the motion of the nearest matched star
            k = np.argmin(np.hypot(*(mm[:, 2:4] - det[j, :2]).T))
            x0, y0, _, _ = project(p, cat.ra[mci[k]], cat.dec[mci[k]], t[fi], lat)
            ok = 0
            for step in (10, 20, 30):
                x1, y1, _, _ = project(p, cat.ra[mci[k]], cat.dec[mci[k]], t[fi + step], lat)
                nxt = dets[fi + step]
                ok += len(nxt) > 0 and cKDTree(nxt[:, :2]).query(det[j, :2] + (x1 - x0, y1 - y0))[0] < 4
            if ok == 3:
                hits.append((fi, det[j, 0], det[j, 1]))
    if len(hits) >= 3:                      # one-offs are edge stars the fit skipped
        fi, x, y = hits[0]
        log.warning(f"bright source moving with the sky matches no catalog star in {len(hits)} sampled "
                    f"frames (planet?), first at frame {fi} ({x:.0f}, {y:.0f})")
