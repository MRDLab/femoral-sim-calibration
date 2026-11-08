"""Vein-wall and nerve stiffness sensitivity on the leg mesh (Mesh F), phase 2b of 2026-09-30. Open item 43.

    $PY_ANATOMY scripts/analyze_sensitivity.py

Reads i4h-phantom/out/paper_repeats_sens/ (launcher run_scripts/paper_repeats_sens_20260930.sh). The rule applied here is
the one declared in that launcher and in item 43 BEFORE any run; nothing below was chosen after seeing a result:
  A  every case's mean inside the control's range (min..max over its repeats) at every depth -> no detectable dependence
  B  a case's mean outside that range at some depth -> the shift per depth is reported
  C  a run of a case exits non-zero, vetoes, or does not settle all seven depths -> a limit of a one-element wall
The 600 kPa and 5.7 MPa vein cases are also read against the uniform 20 kPa case, which isolates the un-blending.
Only runs the launcher logged rc=0 with an idle GPU count. Each run's mesh digest is checked against the manifest.
Writes data/sensitivity/{per_run,per_case_summary,outcome,excluded_runs}.csv and figures/fig_leg_wall_sensitivity.png.
"""
import csv
import json
import pathlib
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
FIG, DATA = ROOT / "figures", ROOT / "data" / "sensitivity"
DATA.mkdir(parents=True, exist_ok=True)
PH = pathlib.Path(__file__).resolve().parent.parent / "out"
import sys  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_meshes as PM  # noqa: E402  2026-10-21: V5 meshes and runs, one place (V4 read out/paper_repeats_sens/)
_man = json.load(open(PM.SENS / "manifest.json"))
assert _man["source_sha256_12"] == PM.LEG_SHA, f"sensitivity meshes cut from {_man['source_sha256_12']}, not {PM.LEG_SHA}"
MAN = _man["cases"]
DIGEST = dict({c: v["sha256_12"] for c, v in MAN.items()}, control=PM.LEG_SHA)
LABEL = {"control": "control: vein wall 20 kPa, blended",
         "vein_uniform_20kpa": "vein wall 20 kPa, uniform",
         "vein_bend_034mm": "vein wall 66.4 kPa (bending match, 0.34 mm wall)",
         "vein_bend_050mm": "vein wall 211.1 kPa (bending match, 0.50 mm wall)",
         "vein_600kpa": "vein wall 600 kPa (the artery's value)",
         "vein_5p7mpa": "vein wall 5.7 MPa (measured in tension)",
         "nerve_20kpa": "nerve 20 kPa (half)", "nerve_80kpa": "nerve 80 kPa (double)"}
ORDER = list(LABEL)

# ── which runs count: the launcher's log decides ──────────────────────────────────────────────
status = PM.queue_status("sens")
runs, excluded, failures = {c: [] for c in ORDER}, [], {c: [] for c in ORDER}
for (rep, case), (rc, gpu) in sorted(status.items()):
    p = PM.run_json("sens", rep, case)
    if gpu != "IDLE":
        excluded.append([case, rep, f"GPU {gpu}"]); continue
    if rc != 0 or not p.exists():
        failures[case].append(f"r{rep}: rc={rc}, json {'present' if p.exists() else 'missing'}"); continue
    d = json.load(open(p))
    assert d["mesh_sha256_12"] == DIGEST[case], f"{p}: mesh {d['mesh_sha256_12']} is not {case} ({DIGEST[case]})"
    assert d["substeps"] == 2 and d["iterations"] == 35 and not d["leg_visual"], p   # None when --no-leg-visual
    rows = {r["depth_mm"]: r for r in d["depth_rows"]}
    bad = [f"{x:g} mm veto {r['veto']}" for x, r in rows.items() if r["veto"] != "none"]
    if len(rows) != 7:
        failures[case].append(f"r{rep}: {len(rows)} of 7 depths recorded"); continue
    if bad:   # the 250-frame settle budget ran out; that depth's value is its LAST frame, not a settled one. Kept, flagged C.
        failures[case].append(f"r{rep}: did not settle at " + "; ".join(bad))
    runs[case].append(dict(rep=rep, rows=rows, press_s=rows[max(rows)]["wall_at_settle_s"],
                           unsettled={x for x, r in rows.items() if r["veto"] != "none"}))
depths = [3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 25.4]   # fixed: with no control run yet, the CSV must still be rewritten, empty

with open(DATA / "per_run.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "repeat", "mesh_sha256_12"] + [f"vein_pct_{x:g}mm" for x in depths]
               + [f"artery_pct_{x:g}mm" for x in depths] + [f"settled_at_frame_{x:g}mm" for x in depths])
    for c in ORDER:
        for r in runs[c]:
            w.writerow([c, r["rep"], DIGEST[c]] + [round(r["rows"][x]["vein_pct"], 4) for x in depths]
                       + [round(r["rows"][x]["artery_pct"], 4) for x in depths] + [r["rows"][x]["settled_at"] for x in depths])
with open(DATA / "excluded_runs.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "repeat", "reason"]); w.writerows(excluded)
if not runs["control"]:
    # 2026-10-21: stop AFTER rewriting the CSVs. Crashing before them left the previous generation's numbers on disk,
    # and make_tables.py printed them (caught by verify_tables.py: "n 5 vs 0 usable raw runs").
    sys.exit(f"!!! no usable control run yet ({PM.GENERATION}): per_run.csv rewritten empty; figure and summary not drawn")

def stats(case, k):
    a = np.array([[r["rows"][x][k] for x in depths] for r in runs[case]])
    return dict(mean=a.mean(0), sd=a.std(0, ddof=1) if len(a) > 1 else np.zeros(len(depths)), lo=a.min(0), hi=a.max(0), n=len(a))
S = {c: {k: stats(c, k) for k in ("vein_pct", "artery_pct")} for c in ORDER if runs[c]}

with open(DATA / "per_case_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "description", "n_repeats", "probe_depth_mm", "vein_pct_mean", "vein_pct_sd", "vein_pct_min", "vein_pct_max",
                "artery_pct_mean", "artery_pct_sd", "artery_pct_min", "artery_pct_max"])
    for c in S:
        for i, x in enumerate(depths):
            v, a = S[c]["vein_pct"], S[c]["artery_pct"]
            w.writerow([c, LABEL[c], v["n"], x] + [round(float(s[i]), 3) for s in (v["mean"], v["sd"], v["lo"], v["hi"])]
                       + [round(float(s[i]), 3) for s in (a["mean"], a["sd"], a["lo"], a["hi"])])

# ── the declared rule ─────────────────────────────────────────────────────────────────────────
def against(case, base):
    out = []
    for k in ("vein_pct", "artery_pct"):
        m, lo, hi = S[case][k]["mean"], S[base][k]["lo"], S[base][k]["hi"]
        for i, x in enumerate(depths):
            shift = m[i] - S[base][k]["mean"][i]
            outside = m[i] < lo[i] or m[i] > hi[i]
            out.append(dict(quantity=k[:-4], depth=x, mean=m[i], base_mean=S[base][k]["mean"][i], base_lo=lo[i], base_hi=hi[i],
                            shift=shift, outside=outside))
    return out
with open(DATA / "outcome.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "read_against", "n_ok", "failed_runs", "outcome", "quantity", "probe_depth_mm", "case_mean_pct",
                "base_mean_pct", "base_min_pct", "base_max_pct", "shift_points", "outside_base_range"])
    for c in ORDER[1:]:
        bases = ["control"] + (["vein_uniform_20kpa"] if c in ("vein_600kpa", "vein_5p7mpa") else [])
        for b in bases:
            if c not in S or b not in S:
                w.writerow([c, b, len(runs[c]), " | ".join(failures[c]), "C" if failures[c] else "no data"]); continue
            rows = against(c, b)
            oc = "C" if failures[c] else ("B" if any(r["outside"] for r in rows) else "A")
            for r in rows:
                w.writerow([c, b, len(runs[c]), " | ".join(failures[c]), oc, r["quantity"], r["depth"], round(r["mean"], 3),
                            round(r["base_mean"], 3), round(r["base_lo"], 3), round(r["base_hi"], 3), round(r["shift"], 3),
                            r["outside"]])
            big = max(rows, key=lambda r: abs(r["shift"]))
            print(f"{c:20s} vs {b:20s} n={len(runs[c])} fail={len(failures[c])} outcome {oc}; largest shift "
                  f"{big['shift']:+.2f} points ({big['quantity']}, {big['depth']:g} mm); "
                  f"vein at deepest {S[c]['vein_pct']['mean'][-1]:.1f} %, artery {S[c]['artery_pct']['mean'][-1]:.1f} %")
for c in ORDER:
    if failures[c]:
        print(f"!!! {c}: " + " | ".join(failures[c]))

# ── figure: rows vein / artery, columns vein-wall cases / nerve cases ──────────────────────────
groups = (("vein-wall stiffness", [c for c in ORDER[1:] if c.startswith("vein")]),
          ("nerve stiffness", [c for c in ORDER if c.startswith("nerve")]))
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
for j, (gname, cases) in enumerate(groups):
    for i, k in enumerate(("vein_pct", "artery_pct")):
        ax = axes[i, j]; c0 = S["control"][k]
        ax.fill_between(depths, c0["lo"], c0["hi"], color="0.8", label="control: range over its repeats")
        ax.plot(depths, c0["mean"], "k-", lw=1.2, label=f"control mean (n = {c0['n']})")
        for n, c in enumerate(cases):
            if c not in S: continue
            s = S[c][k]; off = (n - (len(cases) - 1) / 2) * 0.12
            ax.errorbar(np.array(depths) + off, s["mean"], yerr=s["sd"], fmt="o-", ms=4, capsize=3, color=cols[n],
                        label=f"{LABEL[c]} (n = {s['n']})")
            uns = sorted(set().union(*(r["unsettled"] for r in runs[c])))
            for x in uns:   # hollow white marker: did not settle at this depth in at least one repeat
                ii = depths.index(x); ax.plot(x + off, s["mean"][ii], "o", ms=7, mfc="white", mec=cols[n], zorder=5)
        ax.set_ylabel(f"{'vein' if i == 0 else 'artery'} lumen area (% of resting area)")
        ax.set_title(f"{'Vein' if i == 0 else 'Artery'}: {gname} varied", fontsize=10)
        ax.grid(alpha=0.3)
        if i == 1: ax.set_xlabel("probe depth (mm)")
        if i == 0: ax.legend(fontsize=7.5, loc="lower left")
        if i == 1 and j == 0: ax.text(0.02, 0.03, "hollow marker: did not settle at that depth (value is the last frame)",
                                      transform=ax.transAxes, fontsize=7.5)
fig.suptitle("Leg model: does the result depend on the vein-wall or nerve stiffness? Mean ± SD over repeats,\n"
             "8.33 ms time step, 35 iterations per step; one material changed per case, everything else identical", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "fig_leg_wall_sensitivity.png", dpi=300); plt.close(fig)
print("wrote", DATA, "and", FIG / "fig_leg_wall_sensitivity.png")
