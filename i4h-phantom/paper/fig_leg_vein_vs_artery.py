"""Leg model: vein against artery lumen area at each probe depth, for every solver setting and for the live run. 2026-10-01.

    $PY_ANATOMY scripts/fig_leg_vein_vs_artery.py

The paper's endpoint is a visible difference: the vein compresses while the artery holds. Full closure is not required.
Sources:
  solver settings  data/meshF_repeats/per_run.csv (scripts/analyze_meshF_repeats.py), 12 settings x 5 repeats, Mesh F
                   (c4e13e7d7574), CUDA Graphs on, solver only. The paired graph on/off runs are left out, as in the
                   timing figure: they repeat settings already shown.
  live run         i4h-phantom/out/live_bmode/legF_s2_i35.json, 2026-09-09: the 16.6 FPS run with ultrasound and room
                   render (the video), 8.33 ms x 35, one run. Its settled reading at each depth.
  flattened band   nominal reading of a fully flattened vein on this mesh, computed from it (13-19 % Mesh F, 14-20 % V5).
The phantom measurements can be overlaid on this figure when they exist.
"""
import csv
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType in PDFs: IEEE PDF eXpress rejects Type 3 fonts
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
FIG = ROOT / "figures"
RUNS = list(csv.DictReader(open(ROOT / "data" / "meshF_repeats" / "per_run.csv")))
DEPTHS = [3, 7, 11, 15, 19, 23, 25.4]
import sys  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_meshes as PM  # noqa: E402  2026-10-21: V5 runs, one place
REF, RT = PM.RUN_PREFIX + "s168_i140_ref", PM.RUN_PREFIX + "s2_i35"
tags = sorted({r["run"] for r in RUNS if not r["run"].endswith(("_graphon", "_graphoff"))})
assert len(tags) == 12 and REF in tags and RT in tags, tags

def series(tag, body):
    rs = [r for r in RUNS if r["run"] == tag]
    assert len(rs) == 5 and all(r["cuda_graph"] == "True" for r in rs), tag
    a = np.array([[float(r[f"{body}_pct_{x:g}mm"]) for x in DEPTHS] for r in rs])
    return a.mean(0), a.std(0, ddof=1)

live = json.load(open(PM.LIVE / f"{PM.LIVE_RUN}.json"))
assert live["mesh_sha256_12"] == PM.LEG_SHA and live["substeps"] == 2 and live["iterations"] == 35
lrows = {r["depth_mm"]: r for r in live["depth_rows"]}
assert sorted(lrows) == [float(x) for x in DEPTHS] and all(r["veto"] == "none" for r in lrows.values())

import print_size as PS  # noqa: E402  printed width and 9 pt minimum, asserted on save (2026-10-24)
COL = {"vein": "#1f5fbf", "artery": "#c0392b"}
with plt.rc_context(PS.RC):
    # 2026-10-25, by direction: one panel; vessel colours as before (vein blue, artery red) at full strength for the ten other
    # settings (the faint lines hid ten settings of five runs each); colours fixed across the paper: purple = initial (the
    # 0.099 ms x 140 reference), green = production (8.33 ms x 35), yellow reserved for the cast phantom. Finer axes:
    # y labelled every 10 % with 2 % minor gridlines, x every 5 mm with 1 mm minor gridlines.
    from matplotlib.ticker import MultipleLocator  # noqa: E402
    COL = {"vein": "#1f5fbf", "artery": "#c0392b"}; C_RT = "#2ca02c"; C_REF = "#7b3fa0"
    fig, ax = plt.subplots(figsize=(PS.COL_IN, 4.0), layout="constrained")
    # the nominal floor of THIS mesh: 0.8 dx x the vein's cut width 8.9-12.5 mm / its rest area (13-19 % Mesh F, 14-20 % V5)
    _m = np.load(PM.LEG); _mm = json.load(open(PM.LEG.with_suffix(".json")))
    FL = [100 * 0.8 * _mm["fine_dx_m"] * 1e3 * w / (float(_m["rest_area_vein"][0]) * 1e6) for w in (8.9, 12.5)]
    ax.axhspan(FL[0], FL[1], color="0.88", zorder=0)
    ax.text(0.8, sum(FL) / 2, f"flat vein: {FL[0]:.0f}-{FL[1]:.0f} %", va="center", color="0.3")
    for body in ("vein", "artery"):
        for t in tags:
            if t in (REF, RT): continue
            ax.plot(DEPTHS, series(t, body)[0], "-", color=COL[body], lw=0.9, alpha=0.8, marker="o", ms=2.2, zorder=2)
        ax.plot(DEPTHS, series(REF, body)[0], "--", color=C_REF, lw=1.3, zorder=3)
        m, sd = series(RT, body)
        ax.errorbar(DEPTHS, m, yerr=sd, fmt="*-", color=C_RT, lw=1.5, ms=6, capsize=2, zorder=4)
        ax.plot(DEPTHS, [lrows[float(x)][f"{body}_pct"] for x in DEPTHS], "D", mfc="white", mec=C_RT, mew=1.0, ms=4.5, zorder=5)
    ax.text(26.6, series(RT, "artery")[0][-1], "artery", color=COL["artery"], va="center")
    ax.text(26.6, series(RT, "vein")[0][-1], "vein", color=COL["vein"], va="center")
    h = [plt.Line2D([], [], color=COL["vein"], lw=0.9, marker="o", ms=2.2), plt.Line2D([], [], color=COL["artery"], lw=0.9, marker="o", ms=2.2),
         plt.Line2D([], [], color=C_REF, ls="--", lw=1.3), plt.Line2D([], [], color=C_RT, lw=1.5, marker="*", ms=6),
         plt.Line2D([], [], ls="", marker="D", mfc="white", mec=C_RT, mew=1.0, ms=4.5)]
    fig.legend(h, ["vein, other settings", "artery, other settings", "reference", "chosen", "full loop (1 run)"],
               loc="outside upper center", frameon=False, ncols=2, columnspacing=0.6,
               handlelength=1.4, handletextpad=0.3, labelspacing=0.2)
    ax.set_xlim(0, 34); ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(MultipleLocator(5)); ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(10)); ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.grid(which="major", alpha=0.35); ax.grid(which="minor", alpha=0.12)
    ax.set_xlabel("probe depth (mm)"); ax.set_ylabel("lumen area (% of rest)")
    PS.save(fig, PS.COL_IN, FIG / "fig_leg_vein_vs_artery.png", FIG / "fig_leg_vein_vs_artery.pdf"); plt.close(fig)
for body in ("vein", "artery"):
    print(body, "real-time", " ".join(f"{v:.1f}" for v in series(RT, body)[0]),
          "| live", " ".join(f"{lrows[float(x)][f'{body}_pct']:.1f}" for x in DEPTHS))
spread = {b: [max(series(t, b)[0][i] for t in tags) - min(series(t, b)[0][i] for t in tags) for i in range(7)] for b in ("vein", "artery")}
print("spread across settings (points):", {b: [round(v, 1) for v in s] for b, s in spread.items()})
