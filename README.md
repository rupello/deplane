# deplane

Star-trail stacker for the raw FITS frames produced by the DWARF smart telescope's
*Star Trails* mode, with aircraft and satellite trails removed. It produces a stacked
image and a time-lapse movie of the trails building up, like the ones the DWARF makes
itself, but without the planes.

| DWARF-style max stack | deplane |
| --- | --- |
| ![before](docs/before.jpg) | ![after](docs/after.jpg) |

## Install

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on the `PATH`.

```sh
uv sync
```

## Usage

Point it at a DWARF star-trails folder (the one containing `startrails_*.fits`):

```sh
uv run python deplane.py /path/to/STARTRAILS_DWARF_RAW_WIDE_... -o out
```

This writes to `out/`:

| File | Contents |
| --- | --- |
| `stack.jpg`, `stack.png` | stretched star-trail stack |
| `stack_linear16.tif` | unstretched 16-bit RGB stack, for further processing elsewhere |
| `startrails.mp4` | time-lapse of the trails building up |
| `masks.mp4` | (with `--mask-video`) each raw frame with the removed pixels shown in red |

All frames are held in memory along with their masks and cleaned copies: a
451-frame 1920x1080 session needs roughly 7 GB of RAM.

### Examples

```sh
# comet-style movie: trails fade with a half-life of 20 frames
uv run python deplane.py SESSION -o out_fade --fade 20

# plain max stack without plane removal, for comparison
uv run python deplane.py SESSION -o out_raw --no-clean

# only frames 0-315, still image only
uv run python deplane.py SESSION --last 316 --no-video

# check what is being removed
uv run python deplane.py SESSION --mask-video
```

### Options

| Option | Default | Description |
| --- | --- | --- |
| `-o, --out` | `out` | output folder |
| `--first`, `--last` | all | frame range (Python slice indices into the sorted files) |
| `--sigma` | 4 | detection threshold in sky-noise sigmas; lower catches fainter planes but risks nibbling faint stars |
| `--star-motion` | 3 | max star drift between frames, in half-res pixels |
| `--grow` | 3 | how far to grow each detection, in half-res pixels |
| `--min-area` | 3 | smallest detection kept, in half-res pixels (filters twinkle and hot pixels) |
| `--no-clean` | | skip plane removal |
| `--sky` | 0.25 | output sky brightness, 0-1 |
| `--white` | 99.95 | white-point percentile |
| `--saturation` | 1.3 | colour saturation |
| `--fade` | 0 | movie trail half-life in frames; 0 means trails never fade |
| `--fps` | 30 | movie frame rate |
| `--no-video` | | skip the movie |
| `--mask-video` | | also write `masks.mp4` |

## How it works

1. **Load.** The raw Bayer frames (BGGR, 16-bit) are read with astropy.
2. **Detect transients.** Between consecutive 10 s frames a star moves only a pixel or
   two, but a plane crosses a large part of the field or blinks on and off. Each frame
   is compared with the max of its neighbours, dilated by `--star-motion`. Pixels much
   brighter than that are transients. Detection runs on a half-resolution luminance image
   (each 2x2 Bayer cell summed), and blobs smaller than `--min-area` are discarded.
3. **Clean.** Masked pixels are replaced with the per-pixel median of the nearby frames,
   i.e. the sky behind the plane.
4. **Stack.** The cleaned frames are max-stacked, then debayered.
5. **Stretch.** The sky is white-balanced to neutral grey, then a midtones transfer
   function (as in PixInsight and Siril) puts the sky at `--sky`. The curve is fixed
   from the final stack, so the movie doesn't flicker.
6. **Movie.** A running max of the cleaned frames is written to ffmpeg (H.264). With
   `--fade`, the brightness each pixel has above the current frame decays every frame,
   so trails fade into the current sky rather than to black.

## Limitations

- Meteors and satellites are removed along with planes.
- Sudden lighting changes on the ground (e.g. house lights switching on) are also
  flagged. This does no visible harm, because the replacement is the surrounding frames.
- Very faint plane strobes can leave a few dim dots; try a lower `--sigma`.
