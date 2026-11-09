"""Fig. 11 (appendix): every benchtop phantom image behind the cast validation, re-tiled from the originals. 2026-11-04.

Rows: first contact (press 0, the frames Fig. 10 uses as depth 0) and the seven depths 3 to 25.4 mm (presses 2 to 8).
Press 1, the second 0 mm set, is not plotted in Fig. 10 and is left out here too. Columns: presses 1 to 3.
Every tile is the same crop of a 1179 x 2556 scanner screenshot (columns 200..960, rows 420..1000 = 34.2 x 26.1 mm
at 22.24 px/mm) at the same scale, with the operator's inner-height picks (measurement/picks.csv) drawn as thin
caliper lines with end ticks: vein blue (left tube), artery red (right tube). One 10 mm scale bar. The dagger marks
the 25.4 mm press 3 frame, taken in a separate session (frames/Press_8-3_session2.PNG); its tubes sit about 4 mm
higher than in presses 1 and 2 (a different session zero).

Image provenance, verified here rather than assumed:
  * originals/IMG_5667..5689.PNG are presses 1-1 .. 8-2 in capture order (hash-matched to the operator's frame names
    on 2026-11-04; IMG_5687 is the duplicated 23 mm trial 3, see the trials README);
  * press 0 frames and the session-2 frame are in frames/ (gitignored); their SHA-256 prefixes are pinned below;
  * every pick (front and back wall row) must coincide with an echo peak on that image column, found by the same
    peak finder as the measurement (i4h-phantom/analysis/measure_lumen.py). A frame whose picks miss its echoes
    is the wrong frame, and the script stops.
Outputs: figures/fig_cast_segmentation_clean.{png,pdf} (3.5 in, 8 rows x 3 columns) and
         figures/fig_cast_segmentation_clean_wide.{png,pdf} (7.16 in, the same rows as two blocks of four);
         data/cast_segmentation_tiles.csv (which file is in which tile).
"""
import csv
import hashlib
import pathlib
import re
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "analysis"))
import print_size as PS  # noqa: E402
from measure_lumen import echo_peaks, PX_PER_MM, DEPTH_MM  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent.parent
import os  # noqa: E402
BENCHTOP = pathlib.Path(os.environ.get("BENCHTOP_DIR") or "BENCHTOP_DIR-not-set")   # the benchtop record; not in this repository
if not BENCHTOP.is_dir():
    raise SystemExit(f"!!! benchtop record not found at {BENCHTOP}: set BENCHTOP_DIR (README.md, \"What is not included\")")
TRIALS = BENCHTOP
FIG, DATA = HERE / "figures", HERE / "data"
X0, X1, Y0, Y1 = 200, 960, 420, 1000                        # the one crop, in screenshot pixels
ROWS = [0, 2, 3, 4, 5, 6, 7, 8]                             # press levels shown
ROW_LABEL = {0: "first\ncontact", 2: "3 mm", 3: "7 mm", 4: "11 mm", 5: "15 mm", 6: "19 mm", 7: "23 mm", 8: "25.4 mm"}
WALL_PX = {"vein": 54, "artery": 34}
n_rule = {"vein": 0, "artery": 0}
SHA = {"frames/Press_0-1.PNG": "eab138a1d701", "frames/Press_0-2.PNG": "3dcd76f4f07c", "frames/Press_0-3.PNG": "badc271778b9",
       "frames/Press_8-3_session2.PNG": "82a5e8ce3420", "originals/IMG_5687.PNG": "e666e49ebe2d"}


def image_for(press, trial):
    """Path of the screenshot for (press, trial), by the capture-order mapping verified on 2026-11-04."""
    if press == 0:
        return TRIALS / f"frames/Press_0-{trial}.PNG"
    if press == 8 and trial == 3:
        return TRIALS / "frames/Press_8-3_session2.PNG"
    return TRIALS / f"originals/IMG_{5667 + 3 * (press - 1) + (trial - 1)}.PNG"


def sha12(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


picks = {}
for r in csv.DictReader(open(TRIALS / "measurement/picks.csv")):
    m = re.match(r"Press[_-](\d)[_-](\d)", r["frame"]); key = (int(m.group(1)), int(m.group(2)))
    picks.setdefault(key, {})[r["vessel"]] = (int(r["column_x_px"]), int(r["inner_front_wall_y_px"]), int(r["inner_back_wall_y_px"]))

tiles = []
for press in ROWS:
    for trial in (1, 2, 3):
        p = image_for(press, trial); rel = str(p.relative_to(TRIALS))
        assert p.exists(), f"missing {p} (press {press}, trial {trial}); frames/ is gitignored, see the docstring"
        if rel in SHA:
            assert sha12(p) == SHA[rel], f"{rel}: sha {sha12(p)} != pinned {SHA[rel]}"
        pk = picks[(press, trial)]
        img = np.asarray(Image.open(p).convert("L"), dtype=float)
        assert img.shape == (2556, 1179), img.shape
        # the picks must land on echo peaks of this very image, and the vein must be the left tube
        for v, (x, y1, y2) in pk.items():
            peaks = [y for y, _ in echo_peaks(img, x)]
            # the back pick is an echo peak; the front pick is a peak, or (measure_lumen.py's rule for a faint inner
            # front echo) the outer front echo plus one wall thickness, 54 px for the vein tube and 34 px for the artery
            assert min(abs(y2 - q) for q in peaks) <= 3, f"{rel} ({press}-{trial}) {v}: back pick {y2} is not an echo peak at column {x} (peaks {peaks})"
            on_peak = min(abs(y1 - q) for q in peaks) <= 3
            by_rule = min(abs(y1 - (q + WALL_PX[v])) for q in peaks) <= 3
            assert on_peak or by_rule, f"{rel} ({press}-{trial}) {v}: front pick {y1} is neither an echo peak nor outer echo + wall at column {x} (peaks {peaks})"
            n_rule[v] += (not on_peak) and by_rule
            assert X0 + 25 <= x <= X1 - 25 and Y0 <= y1 - 40 and y2 + 40 <= Y1, f"{rel}: pick outside the crop"
        assert pk["vein"][0] < pk["artery"][0], f"{rel}: the vein must be the left tube"
        tiles.append(dict(press=press, trial=trial, depth=ROW_LABEL[press].replace("\n", " "), file=rel, sha256_12=sha12(p),
                          crop=img[Y0:Y1, X0:X1], picks=pk, dagger=(press == 8 and trial == 3)))
with open(DATA / "cast_segmentation_tiles.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["press", "trial", "depth", "file", "sha256_12", "separate_session"])
    for t in tiles:
        w.writerow([t["press"], t["trial"], t["depth"], t["file"], t["sha256_12"], t["dagger"]])

plt.rcParams.update(PS.RC)
TW, TH = X1 - X0, Y1 - Y0
TICK = 18                                                       # caliper end tick half-length, px


def draw(fig, ax, t, scale_bar):
    ax.imshow(t["crop"], cmap="gray", vmin=0, vmax=255, extent=[X0, X1, Y1, Y0], interpolation="bilinear")
    ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.set_axis_off()
    for v, col in (("vein", PS.VEIN), ("artery", PS.ARTERY)):
        x, y1, y2 = t["picks"][v]
        ax.plot([x, x], [y1, y2], color=col, lw=0.6, solid_capstyle="butt")
        for y in (y1, y2):
            ax.plot([x - TICK, x + TICK], [y, y], color=col, lw=0.6, solid_capstyle="butt")
    if t["dagger"]:
        ax.text(X1 - 12, Y0 + 10, "†", color="white", ha="right", va="top")
    if scale_bar:
        bar = 10 * PX_PER_MM
        ax.plot([X0 + 25, X0 + 25 + bar], [Y1 - 25] * 2, color="white", lw=1.5, solid_capstyle="butt")
        ax.text(X0 + 25 + bar / 2, Y1 - 35, "10 mm", color="white", ha="center", va="bottom")


def layout(width_in, blocks, out_stem):
    """`blocks`: list of row lists; each block is a label column plus three tile columns, blocks side by side."""
    LAB, GAP, HEAD, BGAP = 0.50, 0.03, 0.17, 0.06
    nb = len(blocks)
    tile_w = (width_in - nb * LAB - nb * 2 * GAP - (nb - 1) * BGAP) / (3 * nb)
    tile_h = tile_w * TH / TW
    nrows = max(len(b) for b in blocks)
    H = HEAD + nrows * tile_h + (nrows - 1) * GAP + 0.02
    fig = plt.figure(figsize=(width_in, H))
    for bi, rows in enumerate(blocks):
        bx = bi * (LAB + 3 * tile_w + 2 * GAP + BGAP)
        for ci in range(3):
            x = bx + LAB + ci * (tile_w + GAP)
            fig.text((x + tile_w / 2) / width_in, (H - HEAD + 0.05) / H, f"press {ci + 1}", ha="center", va="bottom")
        for ri, press in enumerate(rows):
            y = H - HEAD - (ri + 1) * tile_h - ri * GAP
            fig.text((bx + LAB - 0.05) / width_in, (y + tile_h / 2) / H, ROW_LABEL[press], ha="right", va="center")
            for ci in range(3):
                t = next(t for t in tiles if t["press"] == press and t["trial"] == ci + 1)
                ax = fig.add_axes([(bx + LAB + ci * (tile_w + GAP)) / width_in, y / H, tile_w / width_in, tile_h / H])
                draw(fig, ax, t, scale_bar=(bi == nb - 1 and ri == len(rows) - 1 and ci == 2))
    outs = [FIG / f"{out_stem}.png", FIG / f"{out_stem}.pdf"]
    PS.save(fig, width_in, *outs, dpi=400)
    print(f">>> wrote {outs[0]} ({width_in} x {H:.2f} in, tiles {tile_w:.2f} x {tile_h:.2f} in, {TW / PX_PER_MM:.1f} x {TH / PX_PER_MM:.1f} mm)")
    plt.close(fig)


layout(PS.COL_IN, [ROWS], "fig_cast_segmentation_clean")
layout(PS.TEXT_IN, [ROWS[:4], ROWS[4:]], "fig_cast_segmentation_clean_wide")
print("front picks placed by the outer-echo-plus-wall rule:", n_rule)
print(f"{len(tiles)} tiles; files:", ", ".join(sorted({t['file'].split('/')[0] for t in tiles})))
