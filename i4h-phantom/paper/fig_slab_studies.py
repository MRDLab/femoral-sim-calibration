"""Slab studies as vessel-collapse figures: element size (Fig. fig:mesh_convergence) and time step (fig:timestep_convergence).

    $PY_ANATOMY scripts/fig_slab_studies.py

Run selection is the tables' rule (make_tables.py). Since 2026-10-21: only the Femoral V5 queue's slab runs logged
"rc=0 GPU IDLE" (paper_meshes.usable), with NO fallback to a V4 original; a setting without runs is left out of the
figure and printed as missing. Every queue run had the GPU to itself, so each is used for speed.
Writes figures/fig_slab_element_size.{png,pdf} (one column: lumen area against depth, five element sizes),
figures/fig_slab_timestep_collapse.{png,pdf} (full width: gap from the 18.60 us reference, time step and iterations)
and data/slab_cost/per_pass.csv (wall clock per solver pass; no longer plotted, 2026-11-04).

2026-11-04 redraw (change order): encodings live in the captions, not the plot. Vein solid, artery dashed; settings are
a light-to-dark ramp of the vessel colour (lighter = nearer the reference / finer); the chosen value (8.33 ms step,
35 iterations) green; reference black. Open markers: the depth did not settle in at least one run. The time-step and
iteration panels plot the DIFFERENCE from the reference, with the +-3 point tolerance shaded.
"""
import csv
import glob
import json
import pathlib
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
FIG = ROOT / "figures"
OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
DEPTHS = [3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 25.4]
FRAME_MS = 1000.0 / 60.0
SLAB_DIGESTS = None   # each run's own provenance is checked below

import sys  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_meshes as PM  # noqa: E402  2026-10-21: V5 runs, one place
import print_size as PS  # noqa: E402  printed width, 9 pt minimum and the colour code, asserted on save
SLAB_SHAS = {s for _, s in PM.SLAB.values()}
plt.rcParams.update(PS.RC)
BODIES = (("vein", "-", PS.VEIN), ("artery", "--", PS.ARTERY))
XLIM = (0, 26)

def runs(tag, original=None):
    """The V5 queue's usable runs of `tag`. `original` is ignored: there is no fallback to a V4 run."""
    out = []
    for _, p in PM.usable("slab", tag, rows_key="rows"):
        d = json.load(open(p)); assert d["mesh_provenance"]["npz"]["sha256"][:12] in SLAB_SHAS, p
        out.append((d, {r["depth_mm"]: r for r in d["rows"]}))
    return out, "V5 runs"

solo_by_rep = {k: v / 60.0 for k, v in PM.queue_seconds("slab").items()}   # minutes, every queue run solo
solo_min = defaultdict(list)
for (_, t), v in solo_by_rep.items():
    solo_min[t].append(v)

def frames_of(d):
    """Frames in one press, from the run file: per depth, the ramp at its mm/frame (rounded up), the settle count, and the
    frame on which the settle test fires (as fig_initial_latency.py; gives the 325 of the 2026-09-07 record)."""
    prev, n = 0.0, 0
    for r in sorted(d["rows"], key=lambda r: r["depth_mm"]):
        n += int(np.ceil((r["depth_mm"] - prev) / d["ramp_mm_per_frame"] - 1e-9)) + r["settled_at"] + 1; prev = r["depth_mm"]
    return n

def series(rs, body):
    a = np.array([[r[x][f"{body}_pct"] for x in DEPTHS] for _, r in rs])
    return a.mean(0), (a.std(0, ddof=1) if len(a) > 1 else np.zeros(len(DEPTHS))), len(a)

def unsettled(rs):
    """Depths that did not settle, or were vetoed, in at least one run of the setting."""
    return sorted({x for _, r in rs for x in DEPTHS if r[x]["settled_at"] is None or r[x]["veto"] != "none"})

OPEN_DRAWN = []   # (setting label, depth) of every open marker actually drawn, for the caption

def draw(ax, rs, tint, label, ref=None, lw=1.0, zorder=2):
    """One setting: vein solid, artery dashed, in `tint` (a fraction of the vessel colour, or a colour to use for both).
    With `ref` ({body: mean}), the gap from the reference is drawn instead of the area. Open markers (same size) at depths
    that did not settle in at least one run. Returns the (vein, artery) legend handles."""
    if not rs:
        return None
    uns = unsettled(rs); handles = []
    for body, ls, base in BODIES:
        c = tint if isinstance(tint, str) else PS.ramp(base, 1, lo=tint)[0]
        m, sd, n = series(rs, body)
        y = m - ref[body] if ref is not None else m
        ax.plot(DEPTHS, y, ls, color=c, lw=lw, zorder=zorder)
        if n > 1:
            ax.errorbar(DEPTHS, y, yerr=sd, fmt="none", ecolor=c, elinewidth=0.6, capsize=0, zorder=zorder)
        for x, v in zip(DEPTHS, y):
            op = x in uns
            # open markers above everything: a hidden open marker is a settle flag that cannot be read (2026-11-04)
            ax.plot(x, v, "o", color=c, mfc="white" if op else c, mec=c, mew=0.8, zorder=8 if op else zorder + 1)
            if op: OPEN_DRAWN.append((label, body, x))
        handles.append(plt.Line2D([], [], ls=ls, color=c, marker="o", mfc=c, mec=c))
    return tuple(handles)

def legend(ax, entries, **kw):
    """Entries [(handles, label)]: each handle pair is drawn as a solid-vein, dashed-artery pair in that setting's tint."""
    ax.legend([h for h, _ in entries], [l for _, l in entries], handler_map={tuple: HandlerTuple(ndivide=2, pad=0.4)},
              handlelength=3.0, handletextpad=0.5, borderaxespad=0.2, labelspacing=0.3, columnspacing=1.2, **kw)

# ── element size (reference time step 18.60 us, 140 iterations): lumen area against depth, one column ─────────
SIZES = [("2p5", 2.5), ("2", 2.0), ("1p5", 1.5), ("1p25", 1.25), ("1p0", 1.0)]
TINTS = np.linspace(0.4, 1.0, len(SIZES))[::-1]       # 1 mm lightest ... 2.5 mm darkest (darker = coarser)
fig, ax = plt.subplots(figsize=(PS.COL_IN, 2.9), layout="constrained")
per_run, right = [], {}
for (k, h), t in zip(SIZES, TINTS):
    tag = f"dx{k}_s896_i140"
    rs, kind = runs(tag)
    assert all(d["substeps"] == 896 and d["iterations"] == 140 for d, _ in rs)
    if draw(ax, rs, t, f"{h:g} mm") is not None:
        right[h] = series(rs, "vein")[0][-1]
    # cost per solver pass (one iteration of one time step), kept in data/slab_cost/per_pass.csv; not plotted since
    # 2026-11-04 (the old panel (a)). A press has frames x 896 x 140 passes.
    for rep_, p in PM.usable("slab", tag, rows_key="rows"):
        d = json.load(open(p))
        if (rep_, tag) not in solo_by_rep: continue
        fr = frames_of(d); mins = solo_by_rep[(rep_, tag)]
        per_run.append(dict(element_mm=h, tets=d["n_tets"], repeat=rep_, frames=fr, wall_min=mins,
                            ms_per_pass=mins * 60e3 / (fr * 896 * 140)))
(ROOT / "data" / "slab_cost").mkdir(parents=True, exist_ok=True)
with open(ROOT / "data" / "slab_cost" / "per_pass.csv", "w", newline="") as f:
    _fields = ["element_mm", "tets", "repeat", "frames", "wall_min", "ms_per_pass"]
    w_ = csv.DictWriter(f, fieldnames=_fields); w_.writeheader(); w_.writerows(per_run)   # rewritten even when empty
pts = []
for (k, h) in SIZES:
    v = [r for r in per_run if r["element_mm"] == h]
    if not v: continue
    m, sd = np.mean([r["ms_per_pass"] for r in v]), (np.std([r["ms_per_pass"] for r in v], ddof=1) if len(v) > 1 else 0.0)
    pts.append((h, m, sd, len(v)))
missing = [h for _, h in SIZES if h not in right]
# the coarsest and finest vein curves named at their right ends; the three between are in order (caption)
for h, c in ((2.5, PS.VEIN), (1.0, PS.VEIN)):
    if h in right:
        ax.text(DEPTHS[-1] + 0.5, right[h], f"{h:g} mm", color=c, ha="left", va="center", clip_on=False)
if missing: ax.text(0.02, 0.03, "awaiting runs: " + ", ".join(f"{h:g} mm" for h in missing), transform=ax.transAxes, color="#b00")
ax.set_xlim(*XLIM); ax.set_ylim(0, 103); ax.set_yticks(range(0, 101, 20))
ax.set_xlabel("probe depth (mm)"); ax.set_ylabel("lumen area (% of rest)")
PS.save(fig, PS.COL_IN, FIG / "fig_slab_element_size.png", FIG / "fig_slab_element_size.pdf"); plt.close(fig)
print(f"element size: per-pass means (CSV only); "
      f"points {[(h, round(m, 3), round(sd, 3), n) for h, m, sd, n in pts]}; missing {missing}; "
      f"open markers drawn: {[o for o in OPEN_DRAWN]}")
OPEN_DRAWN.clear()

# ── time step (140 iterations) and iterations (2.08 ms): gap from the 18.60 us reference, full width ───────────
fig, (b0, b1) = plt.subplots(1, 2, figsize=(PS.TEXT_IN, 2.9), sharey=True, layout="constrained")
ref_rs, _ = runs("dx1p5_s896_i140")
assert ref_rs, "no reference runs (dx1p5_s896_i140): nothing to take a gap from"
REFM = {body: series(ref_rs, body)[0] for body, _, _ in BODIES}
for ax in (b0, b1):
    ax.axhspan(-PS.TOL_PTS, PS.TOL_PTS, color=PS.BAND, lw=0, zorder=0)
    ax.axhline(0, color=PS.REF, lw=1.0, zorder=1)
TS = [168, 67, 17, 8, 3, 2]            # substeps per 60 Hz frame; 2 = the chosen 8.33 ms step
CHOSEN_S, CHOSEN_I = 2, 35
ent = []
for s, t in zip(TS, np.linspace(0.4, 1.0, len(TS))):   # 0.099 ms lightest ... 8.33 ms darkest (darker = larger step)
    rs, kind = runs(f"ts_s{s}_i140")
    assert all(d["substeps"] == s and d["iterations"] == 140 for d, _ in rs)
    dt = FRAME_MS / s; lab = f"{dt:.3f} ms" if dt < 1 else f"{dt:.2f} ms"
    h = draw(b0, rs, PS.CHOSEN if s == CHOSEN_S else t, lab, ref=REFM, zorder=4 if s == CHOSEN_S else 2)
    if h: ent.append((h, lab))
legend(b0, ent, loc="upper left", ncols=2)
IT = [70, 35, 20, 10]
ent = []
for it, t in zip(IT, np.linspace(0.4, 1.0, len(IT))):   # 70 lightest ... 10 darkest (darker = fewer iterations)
    rs, kind = runs(f"ts_s8_i{it}")
    assert all(d["substeps"] == 8 and d["iterations"] == it for d, _ in rs)
    lab = f"{it} iterations"
    h = draw(b1, rs, PS.CHOSEN if it == CHOSEN_I else t, lab, ref=REFM, zorder=4 if it == CHOSEN_I else 2)
    if h: ent.append((h, lab))
legend(b1, ent, loc="upper left", ncols=2)
b0.set_title("(a)"); b1.set_title("(b)")
b0.set_ylabel("gap from reference (points)")
for ax in (b0, b1):
    ax.set_xlabel("probe depth (mm)"); ax.set_xlim(*XLIM); ax.set_ylim(-8, 8); ax.set_yticks(range(-8, 9, 4))
PS.save(fig, PS.TEXT_IN, FIG / "fig_slab_timestep_collapse.png", FIG / "fig_slab_timestep_collapse.pdf"); plt.close(fig)
print("time step: unsettled", {s: unsettled(runs(f"ts_s{s}_i140")[0]) for s in TS})
print("iterations: unsettled", {it: unsettled(runs(f"ts_s8_i{it}")[0]) for it in IT})
print("open markers drawn:", OPEN_DRAWN)
print("reference SD (points) vein", np.round(series(ref_rs, "vein")[1], 2).tolist(), "artery", np.round(series(ref_rs, "artery")[1], 2).tolist())
