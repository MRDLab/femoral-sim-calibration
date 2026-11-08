#!/usr/bin/env python3
"""
Lumen inner-height and area measurement from B-mode screenshots of the cast phantom.

Method
------
1. For a chosen image column through the middle of a tube, take the vertical
   brightness profile (mean over a 25 px wide band), smooth it with a Gaussian
   (sigma = 1.5 px) and find the bright echo lines with scipy.signal.find_peaks.
2. The operator picks which peaks are the INNER front-wall echo and the INNER
   back-wall echo of each tube and records them in picks.csv.
   (Each wall gives two echoes, outer and inner surface; use the inner ones.
   If the inner front echo is too faint, place it one wall thickness below the
   outer front echo: ~54 px for the vein tube, ~34 px for the artery tube.)
3. Inner height (px) -> mm using the scanner's depth ruler, then corrected for
   sound travelling slower in water (1480 m/s) than the scanner assumes (1540 m/s).
4. Area as a fraction of the round lumen, assuming the wall keeps its length as
   the tube flattens into a rounded rectangle (stadium):
       x = h / D0,   A / A0 = x (2 - x)

Usage
-----
  # 1. Look at the echo peaks in one column to choose your picks
  python measure_lumen.py profile "Ultrasound Trials/Press_4-1.PNG" 430

  # 2. Compute areas, a per-frame CSV, a per-depth summary and a QC figure
  python measure_lumen.py measure --images "Ultrasound Trials" --picks picks.csv --out results

Requirements: numpy, scipy, pillow  (pip install numpy scipy pillow)
"""
import argparse
import csv
import os
import re
import statistics

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ---- Constants (edit if the scanner settings or tubes change) -----------------
PX_PER_MM = 22.24                 # scanner depth ruler: 10 mm ticks are 222.4 px apart in the 1179x2556 screenshot
C_WATER, C_SCANNER = 1480.0, 1540.0
MM_PER_PX = (1.0 / PX_PER_MM) * (C_WATER / C_SCANNER)
REST_DIAMETER_MM = {"artery": 9.0,  # McMaster 5041K761
                    "vein": 10.0}   # McMaster 5054K827
DEPTH_MM = {0: "0 (first visible image)", 1: 0, 2: 3, 3: 7, 4: 11, 5: 15, 6: 19, 7: 23, 8: 25.4}
TARGET_TCP_MM = {0: 195.0, 1: 195.0, 2: 192.0, 3: 188.0, 4: 184.0, 5: 180.0, 6: 176.0, 7: 172.0, 8: 169.6}
COLOURS = {"vein": (60, 200, 255), "artery": (255, 80, 80)}


def load_grey(path):
    return np.asarray(Image.open(path).convert("L"), dtype=float)


def echo_peaks(img, x, y_min=420, y_max=1150, half_width=12, sigma=1.5, prominence=15, min_level=55):
    """Return [(y, brightness)] of echo lines along column x."""
    profile = gaussian_filter1d(img[:, x - half_width:x + half_width + 1].mean(axis=1), sigma)
    peaks, _ = find_peaks(profile, prominence=prominence, distance=8)
    return [(int(y), int(profile[y])) for y in peaks if y_min < y < y_max and profile[y] > min_level]


def area_percent(height_px, vessel):
    x = height_px * MM_PER_PX / REST_DIAMETER_MM[vessel]
    return 100.0 * x * (2.0 - x)


def parse_frame(name):
    m = re.match(r"Press[_-](\d)[_-](\d)", name)
    return int(m.group(1)), int(m.group(2))


def find_image(folder, frame):
    for ext in (".PNG", ".png", ".jpg", ".jpeg"):
        p = os.path.join(folder, frame + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No image for {frame} in {folder}")


def cmd_profile(args):
    img = load_grey(args.image)
    for x in range(args.x - args.spread, args.x + args.spread + 1, args.step):
        print(f"x={x}: " + ", ".join(f"y={y} (I={i})" for y, i in echo_peaks(img, x)))


def cmd_measure(args):
    os.makedirs(args.out, exist_ok=True)
    picks = list(csv.DictReader(open(args.picks)))
    rows = []
    for p in picks:
        press, trial = parse_frame(p["frame"])
        v = p["vessel"]
        y1, y2 = int(p["inner_front_wall_y_px"]), int(p["inner_back_wall_y_px"])
        h = y2 - y1
        rows.append(dict(frame=p["frame"], press=press, trial=trial, depth_mm=DEPTH_MM[press],
                         target_tcp_mm=TARGET_TCP_MM[press], vessel=v, column_x_px=int(p["column_x_px"]),
                         inner_front_wall_y_px=y1, inner_back_wall_y_px=y2, inner_height_px=h,
                         inner_height_mm=round(h * MM_PER_PX, 3), rest_diameter_mm=REST_DIAMETER_MM[v],
                         area_pct_of_round=round(area_percent(h, v), 2)))
    rows.sort(key=lambda r: (r["press"], r["trial"], r["vessel"] != "vein"))

    with open(os.path.join(args.out, "phantom_lumen_measurements.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # per-depth summary
    with open(os.path.join(args.out, "phantom_lumen_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["press", "depth_mm", "vessel", "n", "area_mean_pct", "area_sd_pct", "inner_height_mean_mm"])
        for press in sorted({r["press"] for r in rows}):
            for v in ("vein", "artery"):
                sel = [r for r in rows if r["press"] == press and r["vessel"] == v]
                a = [r["area_pct_of_round"] for r in sel]
                sd = statistics.stdev(a) if len(a) > 1 else 0.0
                w.writerow([press, DEPTH_MM[press], v, len(a), round(statistics.mean(a), 1), round(sd, 1),
                            round(statistics.mean(r["inner_height_mm"] for r in sel), 2)])
                print(f"Press {press} ({DEPTH_MM[press]} mm) {v:6s}: {statistics.mean(a):5.1f} ± {sd:3.1f} %  (n={len(a)})")

    # QC figure: every frame with the measured heights drawn on
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    frames = sorted({r["frame"] for r in rows}, key=parse_frame)
    tiles = []
    for fr in frames:
        im = Image.open(find_image(args.images, fr)).convert("RGB")
        d = ImageDraw.Draw(im)
        for r in (r for r in rows if r["frame"] == fr):
            x, y1, y2, c = r["column_x_px"], r["inner_front_wall_y_px"], r["inner_back_wall_y_px"], COLOURS[r["vessel"]]
            d.line([(x - 60, y1), (x + 60, y1)], fill=c, width=4)
            d.line([(x - 60, y2), (x + 60, y2)], fill=c, width=4)
            d.line([(x, y1), (x, y2)], fill=c, width=3)
        tile = im.crop((160, 320, 1020, 1000))
        dd = ImageDraw.Draw(tile)
        press, _ = parse_frame(fr)
        label = ("contact" if press <= 1 else f"{DEPTH_MM[press]} mm") + "  ·  " + fr
        dd.rectangle([0, 0, 430, 38], fill=(0, 0, 0))
        dd.text((8, 4), label, fill=(255, 255, 0), font=font)
        tiles.append(tile.resize((430, 340)))
    cols = 6
    sheet = Image.new("RGB", (430 * cols, 340 * ((len(tiles) + cols - 1) // cols)), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 430, (i // cols) * 340))
    sheet.save(os.path.join(args.out, "phantom_lumen_segmentation.png"))
    print(f"Wrote {len(rows)} rows and a {len(frames)}-frame QC figure to {args.out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile", help="print echo peaks around a column")
    p.add_argument("image")
    p.add_argument("x", type=int)
    p.add_argument("--spread", type=int, default=30)
    p.add_argument("--step", type=int, default=15)
    p.set_defaults(func=cmd_profile)
    m = sub.add_parser("measure", help="compute areas from picks.csv")
    m.add_argument("--images", required=True)
    m.add_argument("--picks", required=True)
    m.add_argument("--out", default="results")
    m.set_defaults(func=cmd_measure)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
