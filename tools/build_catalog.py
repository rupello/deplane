"""One-off: build the vendored star catalog in data/ from upstream sources.

    npm pack d3-celestial@0.7.35 && tar xzf d3-celestial-0.7.35.tgz
    curl -LO https://www.pas.rochester.edu/~emamajek/WGSN/IAU-CSN.txt
    uv run python tools/build_catalog.py package/data IAU-CSN.txt

Positions, magnitudes and Bayer/Flamsteed designations come from d3-celestial
(BSD-3-Clause). Proper names come only from the IAU-WGSN catalog, matched by HIP
number, because d3-celestial's names include many non-IAU ones.
"""

import csv
import json
import os
import re
import sys

MAXMAG = 6.0
OUT = os.path.join(os.path.dirname(__file__), "..", "data")


def iau_names(path):
    """HIP -> IAU name (with diacritics)."""
    names = {}
    for line in open(path, encoding="utf-8"):
        if line[:1] in "#$" or not line.strip():
            continue
        name = line[18:36].strip()
        tok = line[36:].split()
        # the date column is always present; HIP sits four columns before it
        di = max(i for i, t in enumerate(tok) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t))
        hip = tok[di - 4]
        if hip != "_":
            names.setdefault(int(hip), name)
    return names


def main(d3c, iau):
    stars = json.load(open(os.path.join(d3c, "stars.6.json")))["features"]
    desig = json.load(open(os.path.join(d3c, "starnames.json")))
    cons = json.load(open(os.path.join(d3c, "constellations.json")))["features"]
    names = iau_names(iau)

    os.makedirs(OUT, exist_ok=True)
    n = 0
    with open(os.path.join(OUT, "stars.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hip", "ra", "dec", "mag", "bv", "name", "bayer", "flam", "con"])
        for s in stars:
            mag = s["properties"]["mag"]
            if mag > MAXMAG:
                continue
            hip = int(s["id"])
            ra, dec = s["geometry"]["coordinates"]
            d = desig.get(str(hip), {})
            w.writerow([hip, f"{ra % 360:.5f}", f"{dec:.5f}", mag, s["properties"].get("bv", ""),
                        names.get(hip, ""), d.get("bayer", ""), d.get("flam", ""), d.get("c", "")])
            n += 1
    with open(os.path.join(OUT, "constellations.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["con", "name"])
        for c in cons:
            w.writerow([c["id"], c["properties"]["name"]])
    print(f"{n} stars, {len(cons)} constellations, {len(names)} IAU names")


if __name__ == "__main__":
    main(*sys.argv[1:3])
