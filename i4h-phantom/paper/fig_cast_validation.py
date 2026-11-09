"""Cast validation figure and table: the cast leg's measured lumens against the blind simulation (2026-10-30).

Phantom: Phantoms/Femoral/print-ready/cast_2026-10-28/trials_2026-10-30/phantom_lumen_measurements_clean.csv
  inner height h (already x 1480/1540), area A/A0 = x(2 - x), x = h / D0 (rounded-rectangle flattening, wall length kept).
  D0 = the McMaster catalogue bores, artery 9.0 mm (5041K761), vein 10.0 mm (5054K827). The heights at first contact
  read 8-10 % short (a slight press already at contact), so they are reported as a sensitivity check: the CSV carries
  both versions. The duplicated 25.4 mm trial-3 frame is excluded (it is the 23 mm frame): n = 2 there.
  Depth 0 is the "first visible image" press (press 0).
Simulation: the accuracy reference (0.099 ms x 140, n = 1) on the cast mesh, digest 6ac6a3fe5006 -- the values the
  plan compares against, because production (8.33 ms x 35) missed the 3-point bar to its reference on the cast.
Agreement: within 5 percentage points at every depth (close: 3), set 2026-10-29 before any cast data.
"""
import csv
import gzip
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import print_size as PS  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent.parent
import os  # noqa: E402
BENCHTOP = pathlib.Path(os.environ.get("BENCHTOP_DIR") or "BENCHTOP_DIR-not-set")   # the benchtop record; not in this repository
if not BENCHTOP.is_dir():
    raise SystemExit(f"!!! benchtop record not found at {BENCHTOP}: set BENCHTOP_DIR (README.md, \"What is not included\")")
TRIALS = BENCHTOP / "phantom_lumen_measurements_clean.csv"
RUNS = pathlib.Path(__file__).resolve().parent.parent / "out" / "paper_cast" / "main"   # run_scripts/cast_main_20261029.sh
FIG, DATA = HERE / "figures", HERE / "data"
DEPTHS = [0.0, 3, 7, 11, 15, 19, 23, 25.4]
D0_CAT = {"vein": 10.0, "artery": 9.0}
COL = {"vein": "#1f5fbf", "artery": "#c0392b"}; C_CAST = "#f2c200"   # yellow = the real phantom, fixed across the paper
BAR, CLOSE = 5.0, 3.0

# ── phantom ──
rows = list(csv.DictReader(open(TRIALS)))
assert not any(r["frame"] in ("Press_8-3", "Press_8-3_shadowed") for r in rows), "the duplicated 23 mm frame must stay excluded"
h = {}
for r in rows:
    if r["press"] == "1":                      # the second 0 mm set ("contact"): not plotted, kept in the CSV
        continue
    d = 0.0 if r["press"] == "0" else float(r["depth_mm"])
    h.setdefault((d, r["vessel"]), []).append(float(r["inner_height_mm"]))
D0_CONTACT = {v: float(np.mean(h[(0.0, v)])) for v in ("vein", "artery")}
pct = lambda hh, d0: 100.0 * (hh / d0) * (2.0 - hh / d0)
cast = {(d, v): [pct(x, D0_CAT[v]) for x in h[(d, v)]] for d in DEPTHS for v in ("vein", "artery")}
cast_contact = {(d, v): [pct(x, D0_CONTACT[v]) for x in h[(d, v)]] for d in DEPTHS for v in ("vein", "artery")}

# ── simulation (reference) ──
_ref = RUNS / "cast_ref_s168_i140.json"
ref = json.load(open(_ref) if _ref.exists() else gzip.open(str(_ref) + ".gz"))
assert ref["mesh_sha256_12"] == "6ac6a3fe5006" and ref["substeps"] == 168 and ref["iterations"] == 140
sim = {0.0: {"vein": 100.0, "artery": 100.0}}
for r in ref["depth_rows"]:
    assert r["veto"] == "none" and r["skin_travel_mm"] / r["depth_mm"] >= 0.90, r
    sim[r["depth_mm"]] = {"vein": r["vein_pct"], "artery": r["artery_pct"]}
assert sorted(sim) == DEPTHS

# ── table ──
DATA.mkdir(exist_ok=True)
out = DATA / "cast_validation.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["depth_mm", "vessel", "n", "cast_mean_pct", "cast_sd_pct", "sim_ref_pct", "error_pts",
                "within_5", "within_3", "cast_mean_pct_contact_rest", "error_pts_contact_rest"])
    for d in DEPTHS[1:]:
        for v in ("vein", "artery"):
            c, cp = cast[(d, v)], cast_contact[(d, v)]
            e, ep = sim[d][v] - np.mean(c), sim[d][v] - np.mean(cp)
            w.writerow([d, v, len(c), round(np.mean(c), 2), round(np.std(c, ddof=1), 2), round(sim[d][v], 2), round(e, 2),
                        abs(e) <= BAR, abs(e) <= CLOSE, round(np.mean(cp), 2), round(ep, 2)])
print(f">>> wrote {out}; contact-frame rest diameters: vein {D0_CONTACT['vein']:.2f} mm, artery {D0_CONTACT['artery']:.2f} mm")
for row in csv.DictReader(open(out)):
    print("   ", row["depth_mm"], row["vessel"], row["n"], row["cast_mean_pct"], "±", row["cast_sd_pct"], "| sim", row["sim_ref_pct"],
          "| err", row["error_pts"], "| within 5:", row["within_5"], "| contact-rest err", row["error_pts_contact_rest"])

# per-trial values, so make_tables.py can register them as cells and verify_tables.py can check the printed means
with open(DATA / "cast_trials.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["depth_mm", "vessel", "trial_values_pct"])
    for d in DEPTHS[1:]:
        for v in ("vein", "artery"):
            w.writerow([d, v, ";".join(f"{x:.4f}" for x in cast[(d, v)])])

# ── LaTeX table, from the CSV just written (one source) ──
T = {(float(r["depth_mm"]), r["vessel"]): r for r in csv.DictReader(open(out))}
cellv = lambda r: (f"{float(r['cast_mean_pct']):.1f}$\\pm${float(r['cast_sd_pct']):.1f} & {float(r['sim_ref_pct']):.1f} & "
                   + (f"\\textbf{{{float(r['error_pts']):+.1f}}}" if r["within_5"] != "True" else f"{float(r['error_pts']):+.1f}"))
body = "\n".join(f"{d:g}~mm & {cellv(T[(d, 'vein')])} & {cellv(T[(d, 'artery')])} \\\\" for d in DEPTHS[1:])
ws = {v: sum(T[(d, v)]["within_5"] == "True" for d in DEPTHS[1:]) for v in ("vein", "artery")}
(HERE / "tables").mkdir(exist_ok=True)
(HERE / "tables" / "cast_validation.tex").write_text(f"""\\begin{{table}}[!tbp]
\\centering
\\caption{{Benchtop phantom against the blind simulated phantom: lumen area, \\% of rest. Benchtop: mean $\\pm$ SD of three
presses (the third 25.4~mm press from a separate session). Simulated: the simulated phantom, reference setting. Error:
simulated minus benchtop, in percentage points; bold: outside the 5-point tolerance set before the trials. Within 5 points: artery
{ws['artery']} of 7 depths, vein {ws['vein']} of 7.}}
\\label{{tab:cast_validation}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular*}}{{\\columnwidth}}{{@{{\\extracolsep{{\\fill}}}}lcccccc@{{}}}}
\\hline
 & \\multicolumn{{3}}{{c}}{{\\textbf{{Vein}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Artery}}}} \\\\
\\textbf{{Depth}} & bench. & sim. & error & bench. & sim. & error \\\\
\\hline
{body}
\\hline
\\end{{tabular*}}
\\end{{table}}
""")
print(">>> wrote tables/cast_validation.tex")

# ── figure ──
with plt.rc_context(PS.RC):
    fig, ax = plt.subplots(figsize=(PS.COL_IN, 3.6), layout="constrained")
    x = np.array(DEPTHS)
    for v in ("artery", "vein"):
        s = np.array([sim[d][v] for d in DEPTHS])
        ax.fill_between(x, s - BAR, s + BAR, color=COL[v], alpha=0.10, lw=0, zorder=1)
        ax.plot(x, s, "-", color=COL[v], lw=1.6, zorder=2)
        m = np.array([np.mean(cast[(d, v)]) for d in DEPTHS]); sd = np.array([np.std(cast[(d, v)], ddof=1) for d in DEPTHS])
        # 2026-10-30, by direction: benchtop means joined by a dashed line (simulated stays solid), for legibility
        ax.errorbar(x, m, yerr=sd, fmt="o", ls="--", lw=1.2, color=COL[v], mfc=C_CAST, mec=COL[v], mew=1.2, ms=5, ecolor=COL[v], elinewidth=1.0,
                    capsize=2, zorder=3)
    ax.text(25.9, sim[25.4]["artery"] + 5, "artery", color=COL["artery"], va="center")
    ax.text(25.9, sim[25.4]["vein"] - 4, "vein", color=COL["vein"], va="center")
    h_ = [plt.Line2D([], [], color="0.25", lw=1.6),
          plt.Line2D([], [], ls="--", lw=1.2, color="0.25", marker="o", mfc=C_CAST, mec="0.25", mew=1.2, ms=5),
          matplotlib.patches.Patch(color="0.25", alpha=0.15, lw=0)]
    fig.legend(h_, ["simulated", "benchtop, mean ± SD", "±5 points"], loc="outside upper center",
               ncols=2, frameon=False, handlelength=1.3, handletextpad=0.3, columnspacing=0.8)
    ax.set_xlim(-0.8, 30.5); ax.set_ylim(40, 103)
    ax.xaxis.set_major_locator(MultipleLocator(5)); ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(10)); ax.yaxis.set_minor_locator(MultipleLocator(2))
    ax.grid(which="major", alpha=0.35); ax.grid(which="minor", alpha=0.12)
    ax.set_xlabel("probe depth (mm)"); ax.set_ylabel("lumen area (% of rest)")
    PS.save(fig, PS.COL_IN, FIG / "fig_cast_validation.png", FIG / "fig_cast_validation.pdf"); plt.close(fig)
print(">>> wrote figures/fig_cast_validation.{png,pdf}")
