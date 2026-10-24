# Handoff: add star labels to deplane's star-trail movie

**Repo:** `rupello/deplane` (single script: `deplane.py`, ~190 lines; deps in `pyproject.toml`, managed with `uv`)
**Goal:** an optional `--labels` mode that identifies the stars in the session automatically and writes a movie where each star gets a name tag as it appears. The tag follows the star for a few seconds and then fades out, so the movie stays uncluttered. Optionally, constellation names fade in and out the same way.

**Status:** a working prototype exists, built on the DWARF's own exported MP4 of an Orion-rising session. It labelled 71 stars and 7 constellations, with a median position error of about 2 px. The prototype scripts are in `star_labels_prototype/` next to this file (`docs/star-labels/`). Port them into deplane. Don't reinvent them.

---

## 1. Feasibility: yes, and the raw FITS route is easier than the MP4 the prototype used

The prototype had to work on a compressed, *cumulative* trail video, so it recovered star positions indirectly from a per-pixel first-lit-time map (`timemap.py`, `heads.py`, `track.py`). deplane already has much better input:

- **Every raw frame is a single 10 s sub with point-like stars.** Per-frame star detection is just local peak finding. None of the time-map machinery needs to be ported.
- **The cleaned frames (`frames_c`) are already in memory** with planes removed, so there are no false detections from aircraft.
- **The movie loop already exists** (`main()`, the `for f in frames_c:` loop that pipes BGR frames to ffmpeg). Labels are an overlay drawn on each stretched 8-bit frame just before `ff.stdin.write(...)`.
- **scipy is already a dependency** (`least_squares` does the fit). astropy is already a dependency too, so planets and the Moon can be added cheaply (see §6).
- **FITS headers probably carry the timestamps** (`DATE-OBS`), and maybe site and optics information. Inspect a header first (`fits.getheader(...)`); don't assume keyword names. If `DATE-OBS` exists, the sky-rotation rate becomes a known constant instead of a fitted parameter.

**Cadence check from the prototype:** the fitted rotation was 0.041658° per output frame. At the sidereal rate of 0.0041781°/s, that is 9.97 s per frame. So the DWARF MP4 is one movie frame per raw 10 s sub, and that session is about 1440 raw frames. Note that the README's memory estimate (7 GB for 451 frames) implies roughly 22 GB for 1440 frames. That is an existing limitation, not part of this task, but don't make it worse: the labelling adds only small per-frame detection arrays.

---

## 2. Suggested user interface

```
uv run python deplane.py SESSION -o out --labels [--lat 39.05] [--label-mag 4.75] [--constellations]
```

New outputs, written only when `--labels` is given:

| File | Contents |
| --- | --- |
| `startrails_labeled.mp4` | same movie as `startrails.mp4`, plus fading labels. Keep the unlabelled movie too. |
| `stack_labeled.jpg` | still stack with each identified star labelled at the end of its trail (optional, cheap) |
| `solution.json` | fitted camera/sky model, RMS residual, matched star list with first/last visible frame |
| `solve_check.jpg` | reference frame with detections (green) and predicted catalog positions (rings), for debugging |

Suggested new options: `--labels`, `--lat` (degrees; used only for atmospheric refraction, see §4), `--label-mag` (faintest magnitude to label, default 4.75), `--label-hold` (seconds fully visible, default 2.5), `--constellations`, `--font` (path override).

---

## 3. Pipeline to implement

Put this in a new module (e.g. `starlabels.py`) and keep `deplane.py` thin: one call after cleaning to solve, one hook in the movie loop to draw.

### 3.1 Per-frame star detection
- Use the existing `superpixel()` half-res luminance on each cleaned frame, minus `4*black`. Subtract the local background (`cv2.medianBlur` or a morphological top-hat), threshold at about 5× the noise estimate (reuse `transient_masks`'s robust noise approach), take connected components, and keep the flux-weighted centroid and the total flux.
- Convert half-res to full-res coordinates with `x_full = 2*x_half + 0.5` (same for y). Movie frames are full-res debayered `stack.shape`.
- Store detections as a list of `(N_i, 3)` arrays `[x, y, flux]`, one per frame.
- Foreground lights (the house) will produce static detections. They are harmless because they never match the catalog, but you can drop sources that don't move across the session.

### 3.2 Star catalog (vendor it, no network at runtime)
The prototype used the **d3-celestial** npm package data, BSD-3-Clause (`npm pack d3-celestial@0.7.35`):
- `data/stars.6.json`: GeoJSON features with `id` = HIP number, `properties.mag`, `properties.bv`, and `geometry.coordinates = [ra_deg (−180..180), dec_deg]`. Normalise RA with `% 360`.
- `data/starnames.json`: keyed by HIP, with `name`, `bayer` (Greek letter, e.g. `π4`), `flam`, and `c` (constellation abbreviation, e.g. `Ori`).

Write a one-off script that extracts stars with mag ≤ 6 into a small vendored CSV (`data/stars.csv`: hip, ra, dec, mag, bv, name, bayer, flam, con), and include the BSD licence notice. `cat.py` in the prototype shows the parsing.

**Name hygiene:** this catalog contains non-IAU and wrong names, such as "Heka" on φ¹ Ori, "Saif al Jabbar", "Thabit", "Al Taj", "Becklin's Star" and "Trapezium". Only show a proper name if it is on an IAU-WGSN whitelist (the prototype hard-codes one for this field in `plan.py`). Otherwise show the Bayer designation with superscript digits (`π⁴ Ori`). Flamsteed numbers stay plain (`32 Ori`, not `³² Ori`). Label θ¹/θ² Ori as **"Orion Nebula · M42"**.

### 3.3 Camera + sky model (port `model.py` almost verbatim)
The camera is fixed on a tripod and the sky rotates. Each catalog star is mapped to a pixel at frame time `t`:

1. Hour angle `H = θ0 + ω·t − RA`. Compute alt/az for latitude `LAT`.
2. Apply refraction (Bennett/Saemundsson formula) to the altitude.
3. Convert to an ENU vector and rotate into the camera frame defined by `(cam_az, cam_alt, roll)`.
4. Apply a pinhole projection with radial distortion: `x = X/Z`, `y = Y/Z`, `d = 1 + k1·r² + k2·r⁴`, `px = cx + f·x·d`, `py = cy − f·y·d`.

There are 10 parameters: `θ0, ω, cam_az, cam_alt, roll, f, cx, cy, k1, k2`.

- `θ0` absorbs the unknown longitude and clock time, and the camera rotation absorbs latitude. So **latitude only matters for refraction**. Default `--lat` to a FITS site keyword if one exists, otherwise ask the user for it or default to 39.05 and log a warning.
- If `DATE-OBS` is present, use `t` in seconds since the first frame and fix `ω = 0.0041780746 °/s`. Otherwise use the frame index with `ω` free (that is what the prototype did).
- Refraction is about 0.5° (about 20 px) at the horizon. It matters for stars rising over the trees, so keep it.
- Distortion is needed: without `k1`/`k2`, stars near the frame edges are off by about 5 px.

### 3.4 Initial plate solve (`solve0.py`, triangle matching)
Run this on one reference frame with plenty of stars, for example the middle frame or the frame with the most detections.
- Take the ~10 brightest detections → 120 pixel triangles (skip any with a side < 40 px).
- Take catalog stars with mag ≤ 3 and their pairwise angular distances. For each candidate pixel scale `s` (px/deg), find catalog triples whose three angular sides match the pixel sides divided by `s` within 8%.
- For each candidate, fit a similarity transform on a gnomonic projection centred on the triple, project all nearby catalog stars, and count detections within 20 px. Keep the candidate with the most inliers. Try both reflection signs and log which one wins (it should be non-mirrored).
- **Gotcha that broke the first attempt:** the scale scan has to include the true scale. For the DWARF 3 wide camera at 1920×1080, `f ≈ 2312 px` (horizontal field of view ≈ 45°), which is **≈ 40 px/deg**. Scan about 26–57 px/deg. Better still, derive the scale from the FITS focal length and pixel size if they are present.

### 3.5 Global fit (`solve1.py` then `solve3.py`)
- Initialise from the triangle solution with a small multistart grid (prototype: `θ0` every 15°, `cam_az` ∈ {70, 90, 110, 130}, `cam_alt` ∈ {15, 30}). Fit distortion off first, using the 4 brightest identified stars across all frames.
- Then iterate: predict every catalog star (mag ≤ 6) in every sampled frame, associate it with a detection within a tolerance (tighter for faint stars: 10 px if mag < 4, otherwise 4.5 px), and reject ambiguous associations where the second-best match is within 1.5× the best. Refit all 10 parameters with `least_squares(loss='soft_l1', f_scale=2)`. Repeat until the matched set is stable (3–5 rounds).
- With raw frames you can associate detections directly per frame instead of building tracks. Subsample frames (every 5th) to keep the fit fast.
- Optional: a per-star constant offset (median residual) for matched stars tightens the label position by a pixel or two. The prototype does this in `plan.py`.

### 3.6 Visibility: when does each star "appear"?
For each catalog star with mag ≤ `--label-mag`, take its predicted position in every frame. Mark it visible in a frame if a detection lies within about 4 px.
- **Appearance** = the first frame of the first 1 s window (in *output* frames, i.e. `fps` frames) where at least 70% of frames are visible. Disappearance is the same rule from the end.
- This handles stars rising over trees, passing behind the tree on the right, and leaving the top edge.
- The prototype's `vis.py` does this with the time map. With raw frames, use detections instead.
- Selection filter used in the prototype: `mag ≤ 4.75 and visible fraction ≥ 0.85 and ≥ 80 visible frames`, OR `mag < 2.5 and fraction ≥ 0.6`. Sirius was only partly visible behind the tree and needed the second rule.
- **Dedupe:** drop a star if its path stays within 10 px of a brighter selected star (ι Ori and its companion, θ¹/θ² Ori, etc.).

### 3.7 Label scheduling (`plan.py`)
Timings are in output-movie frames at `fps`:
- **Life:** fade-in 12 frames, hold 90/70/55 frames for mag < 2.3 / < 3.6 / fainter, fade-out 45 frames.
- **Placement:** the label follows the star. The text sits right and slightly below the star, at `(x+14, y+4)`. That is perpendicular to the trail, which runs from the star toward the lower left. The fallback is above-left, right-aligned at `(x−14−w, y−4−h)`. Use the fallback near the right edge or when the primary position collides.
- **Collisions:** process labels greedily by appearance time. Sample every 6 frames over the label's life and reject a placement if its box overlaps any active label's box. Then try the other side, then delay the start in 10-frame steps up to 4 s. Drop the label if it still collides, unless mag < 2.5, in which case force it.
- **Start-of-video crowding:** stars already up at frame 0 would all appear at once. Stagger them brightest first, 20 frames apart, and skip those fainter than mag 4.4.

### 3.8 Constellation captions (optional, `--constellations`)
- Use letter-spaced caps (e.g. "O R I O N") at about 110/255 alpha, light blue, so they read as secondary to the star labels.
- Orion appears 1.5 s after its main stars (Betelgeuse, Rigel, Bellatrix, belt) are all up. Every other constellation appears 1 s after its brightest member's label.
- The anchor is the centroid of the visible member stars. It moves with the sky (the mean displacement of its members).
- Search offsets of ±400 px horizontally and ±320 px vertically (in 40 px steps) and keep the one that minimises overlap with active star labels and other captions over the caption's life (fade-in 1 s, hold 4 s, fade-out 2 s).

### 3.9 Rendering
- Draw with **Pillow** on a transparent RGBA layer, then `Image.alpha_composite` it onto the frame. `cv2.putText` has no Greek glyphs and looks poor.
- deplane's frames are **BGR**: convert to RGB before compositing and back to BGR afterwards (or composite on reversed channels).
- **Font:** vendor **Inter** (SIL OFL; SemiBold for names, Regular for the designation, Medium for captions) under `fonts/` with its licence. Inter covers Greek and superscript digits.
- **Each label consists of:**
  - a 2 px ring (radius 10 for mag < 2.3, otherwise 7) in warm white `(255,220,140)` at 80% of the label alpha
  - the name in the main font, sized 30/26/22/18 px by magnitude band (−∞,1), [1,2.3), [2.3,3.6), [3.6,∞)
  - for named stars, the Bayer designation in a smaller (0.62×) dimmer font after the name
  - a 2 px dark stroke for legibility
- Fainter-than-3.6 labels are drawn at 85% alpha.
- The prototype rendered 1440 frames at 1080p in about 80 s.

---

## 4. Expected results for the reference session (Orion rising)

Use these values to validate the port on the same session.

**Fitted model** (`star_labels_prototype/fitted_model_dwarf_mp4.json`, from the 1920×1080 DWARF MP4; the raw-frame fit should land very close if the raw frames are also 1920×1080):

| Parameter | Value |
| --- | --- |
| camera azimuth | 106.8° (ESE) |
| camera altitude | 8.2° |
| roll | 2.65° |
| f | 2312 px (HFOV ≈ 45°) |
| cx, cy | 939, 518 |
| k1, k2 | 0.194, −0.537 (normalised coordinates) |
| θ0 (LST at frame 0) | 343.6° (22.9 h) |
| ω | 0.04166°/frame (≈ 10 s/frame) |

**Matching quality:** 95 catalog stars matched across 151 track fragments. Median residual 1.85 px, 90th percentile 3.6 px.

**Timeline**, in output seconds at 30 fps:

| Time | Event |
| --- | --- |
| 0–4 s | Orion's shield (Tabit = π³ Ori, π⁴, π⁵), Zaurak, Rana, Ran and other Eridanus stars at the top |
| 7.4 s | Bellatrix rises over the left tree |
| 12.6–14.7 s | Rigel, Mintaka, Alnilam, Betelgeuse, Alnitak |
| ~19 s | Orion Nebula |
| ~20 s | Saiph |
| ~30 s | Arneb (Lepus) |
| ~34 s | Sirius rises just left of the right-hand tree, then goes behind it |
| ~35 s | Procyon at the left edge |
| ~39 s | Mirzam |
| ~43 s | Tarf (β Cnc) |

**No planets were in that field.** Any bright unmatched detection track should be reported in the log.

## 5. Acceptance criteria
1. `--labels` on the reference session solves without manual input and reports median residual ≤ 3 px.
2. `solve_check.jpg` shows rings centred on stars across the whole frame, including the corners.
3. In `startrails_labeled.mp4` every label sits on its star's trail head, follows it, and fades out. No two labels overlap. There are never more than about 15 labels on screen at once. The opening seconds are the busiest.
4. Without `--labels`, behaviour and outputs are unchanged.
5. If the solve fails (too few stars, cloudy session), log a warning and skip the labelled outputs. Don't crash the run.

## 6. Nice-to-haves (after the core works)
- **Planets/Moon:** if `DATE-OBS` and a location (`--lat/--lon` or header) are available, use `astropy.coordinates.get_body` (built-in ephemeris, no download) to get RA/Dec per frame and run them through the same projection and visibility test. Without date and location this isn't possible, because θ0 is fitted and doesn't give a calendar date.
- Draw Orion's stick figure briefly when it's fully up.
- A `--labels-only` mode that reuses a cached `solution.json` so re-renders don't re-solve.

## 7. Dependencies
- Add `pillow` to `pyproject.toml` at its latest major version (the owner's preference is latest majors), then `uv lock`.
- No other new runtime deps. Vendor the catalog CSV and the font.

## 8. Prototype file index (`star_labels_prototype/`)
These are scratch scripts written against the DWARF MP4, chained through pickles. Use them as reference implementations, not as code to import.

| File | What it does | Port? |
| --- | --- | --- |
| `model.py` | projection model + refraction | **yes, nearly verbatim** |
| `cat.py` | d3-celestial loader | adapt into the vendoring script |
| `solve0.py` | triangle-matching initial solve | yes |
| `solve1.py` | multistart fit from 4 identified stars | yes (generalise the hard-coded star names to "the triangle match") |
| `solve3.py` | iterative match + full fit | yes |
| `vis.py` | per-star visibility windows (time-map based) | logic yes, but use per-frame detections |
| `plan.py` | selection, dedupe, IAU names, scheduling | yes |
| `render.py` | Pillow overlay + constellation caption placement | yes, adapt to BGR and deplane's ffmpeg loop |
| `timemap.py`, `heads.py`, `track.py` | recovering star heads from a cumulative MP4 | **no**, raw frames make these unnecessary |
| `fitted_model_dwarf_mp4.json` | the fitted parameters above | test reference |
