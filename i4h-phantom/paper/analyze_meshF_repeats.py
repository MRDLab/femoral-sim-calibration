"""Mesh F solver settings, five repeats each: per-run and per-setting tables (CSV), and the figures with error bars.

    $PY_ANATOMY scripts/analyze_meshF_repeats.py

Input since 2026-10-21: the Femoral V5 queue, i4h-phantom/out/paper_v5/leg/r1..r5 (run_scripts/paper_v5_rerun_20261021.sh),
chosen through paper_meshes.py. (V4 input was out/paper_repeats/, Mesh F; see git history.)
Only runs whose log line says "rc=0 GPU IDLE" are used; every other run is listed in excluded_runs.csv with its reason.
Definitions are those of the 2026-09-28 figures (i4h-phantom/analysis/plot_leg_wallclock_*.py), unchanged:
  wall clock per frame  median of successive differences of the timeline wall clock (ms)
  whole press           wall clock when the deepest depth settled (s)
  vein / artery %       lumen area as % of rest at each of the seven settled depths
Accuracy is new, because the reference now has repeats too. For repeat r of a setting:
  gap_r = max over the seven depths of |this run's % - mean of the reference repeats' %|
The reference's own gap_r (each reference repeat against the reference mean) is its scatter floor, reported beside it.
Error bars are one standard deviation over the repeats; n is printed for every point.
"""
import csv
import json
import os
import pathlib
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # TrueType in PDFs: IEEE PDF eXpress rejects Type 3 fonts
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
FIG, DATA = ROOT / "figures", ROOT / "data" / "meshF_repeats"
FIG.mkdir(exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
import sys  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_meshes as PM  # noqa: E402  2026-10-21: the paper's meshes and runs, one place
DIGEST = PM.LEG_SHA
REF_TAG = PM.RUN_PREFIX + "s168_i140_ref"
RT_TAG = PM.RUN_PREFIX + "s2_i35"          # the real-time setting
FRAME_MS = 1000.0 / 60.0

# ── which runs count: the launcher's own log decides ──────────────────────────────────────────
status = PM.queue_status("leg")
runs, excluded = [], []
for (rep, tag), (rc, gpu) in sorted(status.items()):
    path = PM.run_json("leg", rep, tag)
    if rc != 0 or gpu != "IDLE" or not path.exists():
        excluded.append([rep, tag, f"rc={rc}, GPU {gpu}, json {'present' if path.exists() else 'missing'}"]); continue
    d = json.load(open(path))
    assert d["mesh_sha256_12"] == DIGEST, f"{path}: mesh {d['mesh_sha256_12']} is not the paper's {PM.GENERATION} leg mesh {DIGEST}"
    tl = d["timeline"]; rows = {r["depth_mm"]: r for r in d["depth_rows"]}
    if len(rows) != 7:
        excluded.append([rep, tag, f"{len(rows)} of 7 depths"]); continue
    wall = np.array([f["wall"] for f in tl])
    runs.append(dict(rep=rep, tag=tag, substeps=d["substeps"], iterations=d["iterations"], graph=bool(d["cuda_graph"]),
                     dt_ms=FRAME_MS / d["substeps"], work=d["substeps"] * d["iterations"],
                     ms=float(np.median(np.diff(wall))) * 1e3, frames=len(tl),
                     press_s=rows[max(rows)]["wall_at_settle_s"], rows=rows))
depths = sorted(runs[0]["rows"]) if runs else []

# ── per-run CSV ───────────────────────────────────────────────────────────────────────────────
with open(DATA / "per_run.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["repeat", "run", "time_step_ms", "iterations_per_step", "cuda_graph", "solver_work_per_frame",
                "median_wall_clock_per_frame_ms", "frames_simulated", "whole_press_wall_clock_s"]
               + [f"vein_pct_{x:g}mm" for x in depths] + [f"artery_pct_{x:g}mm" for x in depths]
               + [f"settled_at_frame_{x:g}mm" for x in depths])
    for r in runs:
        w.writerow([r["rep"], r["tag"], round(r["dt_ms"], 4), r["iterations"], r["graph"], r["work"], round(r["ms"], 3),
                    r["frames"], round(r["press_s"], 3)] + [round(r["rows"][x]["vein_pct"], 4) for x in depths]
                   + [round(r["rows"][x]["artery_pct"], 4) for x in depths] + [r["rows"][x]["settled_at"] for x in depths])
with open(DATA / "excluded_runs.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["repeat", "run", "reason"]); w.writerows(excluded)

# ── per-setting summary ───────────────────────────────────────────────────────────────────────
groups = defaultdict(list)
for r in runs:
    groups[r["tag"]].append(r)
ref = groups.get(REF_TAG, [])
ref_mean = {x: {k: np.mean([r["rows"][x][k] for r in ref]) for k in ("vein_pct", "artery_pct")} for x in depths} if ref else None

def gap(r, key):
    return max(abs(r["rows"][x][key] - ref_mean[x][key]) for x in depths)

def msd(v):
    v = np.asarray(v, float); return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan")), len(v)

summary = []
for tag, rs in groups.items():
    s = dict(tag=tag, dt_ms=rs[0]["dt_ms"], iterations=rs[0]["iterations"], graph=rs[0]["graph"], work=rs[0]["work"])
    s["ms"], s["ms_sd"], s["n"] = msd([r["ms"] for r in rs])
    s["press"], s["press_sd"], _ = msd([r["press_s"] for r in rs])
    s["frames"], s["frames_sd"], _ = msd([r["frames"] for r in rs])
    for x in depths:
        for k in ("vein_pct", "artery_pct"):
            s[f"{k}_{x:g}"], s[f"{k}_{x:g}_sd"], _ = msd([r["rows"][x][k] for r in rs])
    if ref_mean:
        s["gap_vein"], s["gap_vein_sd"], _ = msd([gap(r, "vein_pct") for r in rs])
        s["gap_art"], s["gap_art_sd"], _ = msd([gap(r, "artery_pct") for r in rs])
        rp = np.mean([r["press_s"] for r in ref])
        s["speedup"], s["speedup_sd"], _ = msd([rp / r["press_s"] for r in rs])
    summary.append(s)
summary.sort(key=lambda s: (-s["work"], not s["graph"]))
cols = ["tag", "dt_ms", "iterations", "graph", "work", "n", "ms", "ms_sd", "frames", "frames_sd", "press", "press_sd"] + \
       [c for x in depths for k in ("vein_pct", "artery_pct") for c in (f"{k}_{x:g}", f"{k}_{x:g}_sd")] + \
       (["gap_vein", "gap_vein_sd", "gap_art", "gap_art_sd", "speedup", "speedup_sd"] if ref_mean else [])
names = {"tag": "run", "dt_ms": "time_step_ms", "iterations": "iterations_per_step", "graph": "cuda_graph", "work": "solver_work_per_frame",
         "n": "n_repeats", "ms": "wall_clock_per_frame_ms_mean", "ms_sd": "wall_clock_per_frame_ms_sd", "frames": "frames_mean",
         "frames_sd": "frames_sd", "press": "whole_press_s_mean", "press_sd": "whole_press_s_sd", "gap_vein": "vein_gap_to_ref_pts_mean",
         "gap_vein_sd": "vein_gap_to_ref_pts_sd", "gap_art": "artery_gap_to_ref_pts_mean", "gap_art_sd": "artery_gap_to_ref_pts_sd",
         "speedup": "speedup_mean", "speedup_sd": "speedup_sd"}
with open(DATA / "per_setting_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow([names.get(c, c + ("" if c.endswith("_sd") else "_mean")) for c in cols])
    for s in summary:
        w.writerow([round(s[c], 4) if isinstance(s[c], float) else s[c] for c in cols])

# ── the vein-closure table for the paper: the real-time setting, mean and SD over repeats ────
rt = groups.get(RT_TAG, [])
if rt:
    with open(DATA / "vein_closure_realtime_setting.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["probe_depth_mm", "vein_pct_mean", "vein_pct_sd", "artery_pct_mean", "artery_pct_sd", "n_repeats"])
        for x in depths:
            v, a = [r["rows"][x]["vein_pct"] for r in rt], [r["rows"][x]["artery_pct"] for r in rt]
            w.writerow([x, round(np.mean(v), 2), round(np.std(v, ddof=1), 2) if len(v) > 1 else "", round(np.mean(a), 2),
                        round(np.std(a, ddof=1), 2) if len(a) > 1 else "", len(v)])

# ── figure: wall clock per frame (the IDF proposal's appendix figure (a), now with error bars) ──
# Filled blue: graph on, physics only (the trend is fitted through these). Red: the 0.099 ms x 140 reference.
# Green star: the real-time setting. Hollow: graph off (phase 2). Black x: 0.019 ms x 140, killed 2026-09-09 at the
# 45 GB host-memory watchdog during CUDA graph capture; it marks a resource ceiling and carries NO timing.
PAIRED = ("_graphon",)          # phase-2 controls for the on/off comparison: same settings, not extra points here
on = [s for s in summary if s["graph"] and not s["tag"].endswith(PAIRED)]
off = [s for s in summary if not s["graph"]]
# ── cost model, fitted on every graph-on solver-only run (not on setting means), minimising RELATIVE error ──
# wall clock per frame = a * solver work + b * time steps per frame + c. Unweighted least squares is dominated by the
# slow settings (2 % of 3.8 s swamps a per-step cost of a few ms) and wrongly returned b < 0 on 2026-09-30; so the
# fit divides each row by its own time. Uncertainties: bootstrap over runs, 2000 resamples, fixed seed.
fit_runs = [r for r in runs if r["graph"] and not r["tag"].endswith(PAIRED)]
Wf = np.array([r["work"] for r in fit_runs], float); Sf = np.array([r["substeps"] for r in fit_runs], float)
Yf = np.array([r["ms"] for r in fit_runs], float); Xf = np.c_[Wf, Sf, np.ones_like(Wf)]
coef, *_ = np.linalg.lstsq(Xf / Yf[:, None], np.ones_like(Yf), rcond=None)
rng = np.random.default_rng(0); boot = []
for _ in range(2000):
    k = rng.integers(0, len(Yf), len(Yf)); cb, *_ = np.linalg.lstsq(Xf[k] / Yf[k, None], np.ones(len(k)), rcond=None); boot.append(cb)
coef_sd = np.array(boot).std(0)
resid = 100 * (Yf - Xf @ coef) / Yf
with open(DATA / "cost_model.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["term", "value_ms", "sd_ms", "meaning"])
    for t, v, e, m in zip(("a", "b", "c"), coef, coef_sd, ("per solver pass", "per time step", "per frame, fixed")):
        w.writerow([t, round(v, 4), round(e, 4), m])
    w.writerow(["n_runs", len(Yf), "", "graph-on solver-only runs, logged rc=0 GPU IDLE"])
    w.writerow(["mean_abs_residual_pct", round(float(np.mean(abs(resid))), 2), "", ""])
    w.writerow(["max_abs_residual_pct", round(float(np.max(abs(resid))), 2), "", ""])
if on:
    # Trend line for the figure: time = a * solver work + c (cost per solver pass + fixed cost per frame), fitted on the
    # same runs by relative error, bootstrap SDs. One line a reader can follow. The three-term model above (with its
    # per-time-step cost) stays in cost_model.csv for the text. Deviation convention: 100 * (measured - trend) / trend,
    # so positive means the run was SLOWER than the trend line.
    Xt = np.c_[Wf, np.ones_like(Wf)]
    ct, *_ = np.linalg.lstsq(Xt / Yf[:, None], np.ones_like(Yf), rcond=None)
    bt = []
    for _ in range(2000):
        k = rng.integers(0, len(Yf), len(Yf)); cb, *_ = np.linalg.lstsq(Xt[k] / Yf[k, None], np.ones(len(k)), rcond=None); bt.append(cb)
    ct_sd = np.array(bt).std(0)
    dev = 100 * (Yf - Xt @ ct) / (Xt @ ct)
    with open(DATA / "cost_model.csv", "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trend_a", round(ct[0], 4), round(ct_sd[0], 4), "figure trend line: per solver pass"])
        w.writerow(["trend_c", round(ct[1], 4), round(ct_sd[1], 4), "figure trend line: per frame, fixed"])
        w.writerow(["trend_mean_abs_dev_pct", round(float(np.mean(abs(dev))), 2), "", "100*(measured-trend)/trend"])
        w.writerow(["trend_max_abs_dev_pct", round(float(np.max(abs(dev))), 2), "", ""])
    # ── 2026-11-04 redesign, by direction (the author's change order: the figure "looks machine-made"; the reviewer had
    # already rejected the 2026-10 version as illegible). Paper Fig. 8 is ONE panel at the IEEEtran column width, 3.5 in,
    # so LaTeX never scales it; no visible text under 9 pt, asserted before saving. The residual panel is its own figure
    # (fig_meshF_wallclock_residual) for the supplement, with a +-5 % band. Every explanation lives in the LaTeX caption:
    # the plot labels only "reference" and "chosen"; the fit equation, the per-setting labels, the fold-improvement arrows,
    # the full-loop legend entry and the key floating above the axes are gone. Colours: reference black, chosen setting
    # green, other CUDA-Graph-on settings dark grey (filled), CUDA-Graph-off grey open circles. No purple anywhere.
    # Error bars (one SD over the five repeats, 0.4-2.3 % of the mean) are drawn but are smaller than the markers; the
    # legend therefore shows plain markers, not error-bar glyphs.
    COL_IN, MIN_PT = 3.5, 9.0
    STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
             "font.size": MIN_PT, "axes.labelsize": MIN_PT, "xtick.labelsize": MIN_PT, "ytick.labelsize": MIN_PT,
             "legend.fontsize": MIN_PT, "legend.frameon": False, "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.0,
             "lines.markersize": 3.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    REF_C, CHOSEN_C, ON_C, OFF_C, FIT_C = "#000000", "#2e8b57", "0.30", "0.45", "0.55"
    from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402
    _plain = FuncFormatter(lambda v, _: f"{v:,.0f}")

    def _fig_checks(fig, ax, width, name):
        """The checks that can fail: printed width, every visible text >= MIN_PT, nothing off the page, no label over
        another label or a tick label. FIG_DEBUG_DIR gets a picture of a failing layout; figures/ never does."""
        fig.canvas.draw(); _r = fig.canvas.get_renderer()
        small = [(t.get_text()[:30], t.get_fontsize()) for t in fig.findobj(matplotlib.text.Text)
                 if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-6]
        assert abs(fig.get_size_inches()[0] - width) < 1e-9, fig.get_size_inches()
        assert not small, f"text below {MIN_PT} pt: {small}"
        import print_size as _PS  # noqa: E402  the shared off-the-page check (2026-10-25)
        assert not _PS.clipped_text(fig), f"text runs off the figure: {_PS.clipped_text(fig)}"
        _axb = ax.get_window_extent(_r)
        _tx = [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
        _bb = [matplotlib.text.Text.get_window_extent(t, _r) for t in _tx]   # the text alone, not an Annotation's leader
        _tk = [l.get_window_extent(_r) for l in ax.get_yticklabels() + ax.get_xticklabels() if l.get_visible() and l.get_text()]
        _lg = [t.get_window_extent(_r) for t in (ax.get_legend().get_texts() if ax.get_legend() else [])]
        _out = [t.get_text()[:25] for t, b in zip(_tx, _bb) if b.x0 < _axb.x0 - 1 or b.x1 > _axb.x1 + 1 or b.y0 < _axb.y0 - 1 or b.y1 > _axb.y1 + 1]
        _ovl = [(_tx[i].get_text()[:20], _tx[j].get_text()[:20]) for i in range(len(_bb)) for j in range(i + 1, len(_bb)) if _bb[i].overlaps(_bb[j])]
        _ovl += [(t.get_text()[:20], "tick label") for t, b in zip(_tx, _bb) if any(b.overlaps(k) for k in _tk)]
        _ovl += [(t.get_text()[:20], "legend") for t, b in zip(_tx, _bb) if any(b.overlaps(k) for k in _lg)]
        if os.environ.get("FIG_DEBUG_DIR"):
            fig.savefig(pathlib.Path(os.environ["FIG_DEBUG_DIR"]) / f"{name}.png", dpi=150)
        assert not _out, f"label outside the axes: {_out}"
        assert not _ovl, f"labels overlap: {_ovl}"

    ref_s = next(s for s in on if s["tag"] == REF_TAG); rt_s = next(s for s in on if s["tag"] == RT_TAG)
    plain = sorted([s for s in on if s["tag"] not in (REF_TAG, RT_TAG)], key=lambda s: (s["work"], s["dt_ms"]))
    xs = [s["work"] for s in on]
    # the full loop at the chosen setting (solver + ray-traced ultrasound + 3-D view, one GPU), from the live run: one run,
    # one marker, described in the caption
    _L = json.load(open(PM.LIVE / f"{PM.LIVE_RUN}.json"))
    assert _L["mesh_sha256_12"] == DIGEST and _L["substeps"] * _L["iterations"] == rt_s["work"] and _L["iterations"] == rt_s["iterations"]
    _lms = float(np.median(np.diff([f["wall"] for f in _L["timeline"]]))) * 1e3

    def _pts(a, ss, **kw):
        a.errorbar([s["work"] for s in ss], [s["ms"] for s in ss], yerr=[np.nan_to_num(s["ms_sd"]) for s in ss],
                   capsize=0, elinewidth=0.8, linestyle="none", zorder=3, **kw)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(COL_IN, 3.3), layout="constrained")
        xx = np.logspace(np.log10(min(xs) * 0.6), np.log10(max(xs) * 1.6), 100)
        ax.plot(xx, ct[0] * xx + ct[1], "-", color=FIT_C, lw=1.0, zorder=1)
        ax.axhline(1000 / 60, color="0.35", ls="--", lw=0.8, zorder=1)
        _pts(ax, plain, fmt="o", color=ON_C, ms=3.5)
        if off:
            _pts(ax, off, fmt="o", mfc="white", color=OFF_C, mew=0.9, ms=3.5)
        _pts(ax, [ref_s], fmt="o", color=REF_C, ms=5)
        _pts(ax, [rt_s], fmt="*", color=CHOSEN_C, ms=9)
        ax.plot([rt_s["work"]], [_lms], "*", mfc="white", mec=CHOSEN_C, mew=0.9, ms=8.5, zorder=4)
        # the only two labels in the plot (by direction); everything else is in the caption
        ax.annotate("reference", (ref_s["work"], ref_s["ms"]), textcoords="offset points", xytext=(-6, 0),
                    ha="right", va="center", color=REF_C)
        ax.annotate("chosen", (rt_s["work"], rt_s["ms"]), (rt_s["work"] * 2.4, rt_s["ms"] * 0.37), ha="left", va="center",
                    color=CHOSEN_C, arrowprops=dict(arrowstyle="-", color=CHOSEN_C, lw=0.6, shrinkA=1, shrinkB=5))
        if PM.GENERATION == "V4":   # observed on Mesh F only (2026-09-09); never run on V5, so never drawn on a V5 figure
            ax.plot([125_440], [3e4], "kx", ms=7, mew=2, zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(min(xs) * 0.6, max(xs) * 2.2); ax.set_ylim(8, 1.2e5)
        for a_ in (ax.xaxis, ax.yaxis):   # after set_*scale, which resets the formatter; plain numbers, no minor labels
            a_.set_major_formatter(_plain); a_.set_minor_locator(LogLocator(subs=range(2, 10))); a_.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("solver work per frame (time steps × iterations)"); ax.set_ylabel("wall clock per frame (ms)")
        # small frameless key in the empty lower-right corner, between the real-time line and the fit line: plain markers,
        # no error-bar glyphs (the SDs are smaller than the markers)
        _h = [plt.Line2D([], [], color=FIT_C, lw=1.0),
              plt.Line2D([], [], ls="", marker="o", color=ON_C, ms=3.5),
              plt.Line2D([], [], ls="", marker="o", mfc="white", mec=OFF_C, mew=0.9, ms=3.5),
              plt.Line2D([], [], color="0.35", ls="--", lw=0.8)]
        _yf = np.log10(24 / 8) / np.log10(1.2e5 / 8)   # legend floor at 24 ms, clear of the 16.7 ms line
        ax.legend(_h, ["linear fit", "CUDA Graph on", "CUDA Graph off", "real time, 16.7 ms"], loc="lower right",
                  bbox_to_anchor=(1.0, _yf), handlelength=1.6, handletextpad=0.5, labelspacing=0.25, borderpad=0.2, borderaxespad=0.1)
        _fig_checks(fig, ax, COL_IN, "fig8_paper")
        fig.savefig(FIG / "fig_meshF_wallclock_per_frame.png", dpi=300); fig.savefig(FIG / "fig_meshF_wallclock_per_frame.pdf"); plt.close(fig)

        # ── supplement: each setting's departure from the fit line, its own 3.5 in figure. Point = mean of the repeats,
        # bar = fastest to slowest run; grey band = +-5 %. Same colours as the main figure, no purple.
        fig, ax = plt.subplots(figsize=(COL_IN, 2.1), layout="constrained")
        ax.axhspan(-5, 5, color="0.92", lw=0, zorder=0)
        ax.axhline(0, color="0.5", lw=0.8, zorder=1)
        for s_ in sorted(on, key=lambda q: (q["work"], q["dt_ms"])):
            dd = np.array([d_ for d_, fr in zip(dev, fit_runs) if fr["tag"] == s_["tag"]])
            if not len(dd): continue
            m = dd.mean()
            mk = dict(fmt="o", color=REF_C, ms=5) if s_["tag"] == REF_TAG else dict(fmt="*", color=CHOSEN_C, ms=9) if s_["tag"] == RT_TAG else dict(fmt="o", color=ON_C, ms=3.5)
            ax.errorbar([s_["work"]], [m], yerr=[[m - dd.min()], [dd.max() - m]], capsize=2, elinewidth=0.8, linestyle="none", zorder=3, **mk)
        ax.set_xscale("log"); ax.set_xlim(min(xs) * 0.6, max(xs) * 2.2); ax.set_ylim(-6.5, 6.5)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2.5)); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_major_formatter(_plain); ax.xaxis.set_minor_locator(LogLocator(subs=range(2, 10))); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("solver work per frame (time steps × iterations)"); ax.set_ylabel("departure from fit (%)")
        _fig_checks(fig, ax, COL_IN, "fig8_residual")
        fig.savefig(FIG / "fig_meshF_wallclock_residual.png", dpi=300); fig.savefig(FIG / "fig_meshF_wallclock_residual.pdf"); plt.close(fig)
    # ── IDF fellowship version of the top panel (2026-11-03, by direction). Separate from the paper figure: it is for a
    # non-specialist panel, so frame rates in FPS (the unit the proposal uses), sentence case, only three labelled points,
    # no formula, no per-setting labels. Same data and same live run as the paper figure; nothing here is refitted.
    def _draw_idf_panel():
        rt = next(s for s in on if s["tag"] == RT_TAG); ref = next(s for s in on if s["tag"] == REF_TAG)
        ref_off = next(s for s in off if s["tag"] == REF_TAG + "_graphoff")
        L = json.load(open(PM.LIVE / f"{PM.LIVE_RUN}.json"))
        assert L["mesh_sha256_12"] == DIGEST and L["substeps"] * L["iterations"] == rt["work"] and L["iterations"] == rt["iterations"]
        lms = float(np.median(np.diff([f["wall"] for f in L["timeline"]]))) * 1e3
        GREEN, PURPLE, BLUE, INK = "#2ca02c", "#7b3fa0", "#1f77b4", "0.25"
        fig, ax = plt.subplots(figsize=(5.0, 5.4), layout="constrained")
        xs = [s["work"] for s in on]
        xx = np.logspace(np.log10(min(xs) * 0.6), np.log10(max(xs) * 1.6), 100)
        ax.plot(xx, ct[0] * xx + ct[1], "-", color="0.65", lw=1.0, zorder=1, label="Linear fit")
        plain = [s for s in on if s["tag"] not in (REF_TAG, RT_TAG)]
        ax.plot([s["work"] for s in plain], [s["ms"] for s in plain], "o", color=BLUE, ms=4, zorder=3, label="Other settings tried")
        # the other CUDA Graph off settings, hollow blue (restored 2026-11-03: dropped in the IDF redesign, which hid data)
        # only the middle one (2026-11-03, by direction): the graph-off point at the final-run setting sits beside the green
        # stars and would be confused with them, so it is left out of this IDF version (it stays in the paper figure)
        off_other = [s for s in off if s["tag"] != REF_TAG + "_graphoff" and s["work"] != rt["work"]]
        assert len(off_other) == 1, [s["tag"] for s in off_other]
        ax.plot([s["work"] for s in off_other], [s["ms"] for s in off_other], "o", mfc="white", mec=BLUE, mew=1.2, ms=4.5, zorder=3)
        ax.plot([ref["work"]], [ref["ms"]], "o", color=PURPLE, ms=6, zorder=4)
        ax.plot([ref_off["work"]], [ref_off["ms"]], "o", mfc="white", mec=PURPLE, mew=1.2, ms=6, zorder=4)
        ax.plot([rt["work"]] * 2, [rt["ms"], lms], ":", color=GREEN, lw=1.0, zorder=2)
        ax.plot([rt["work"]], [rt["ms"]], "*", color=GREEN, ms=11, zorder=5)
        ax.plot([rt["work"]], [lms], "*", mfc="white", mec=GREEN, mew=1.0, ms=10, zorder=5)
        ax.annotate("Initial setting,\nCUDA Graph off", (ref_off["work"], ref_off["ms"]), textcoords="offset points",
                    xytext=(-9, 0), ha="right", va="center", color=PURPLE)
        ax.annotate("Initial setting,\nCUDA Graph on", (ref["work"], ref["ms"]), textcoords="offset points",
                    xytext=(-9, 0), ha="right", va="center", color=PURPLE)
        ax.annotate(f"With Isaac Sim viewer\nand ultrasound:\n{1000 / lms:.1f} FPS", (rt["work"], lms), (rt["work"] * 7, lms * 1.6),
                    ha="left", va="center", color=GREEN, zorder=6, bbox=dict(fc="white", ec="none", pad=1.0),
                    arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.6, shrinkA=1, shrinkB=6))
        ax.annotate(f"Final run\n{1000 / rt['ms']:.0f} FPS\n(physics only)", (rt["work"], rt["ms"]), (rt["work"] * 7, rt["ms"] * 0.95),
                    ha="left", va="center", color=GREEN, zorder=6, bbox=dict(fc="white", ec="none", pad=1.0),
                    arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.6, shrinkA=1, shrinkB=6))
        ax.axhline(1000 / 60, color="0.4", ls="--", lw=0.8, zorder=1)
        _xb = max(xs) * 1.6
        ax.text(_xb, 1000 / 60 * 0.9, "Real time (60 FPS)", ha="right", va="top", color="0.4")
        hi, lo = ref_off["ms"], rt["ms"]
        ax.annotate("", (_xb, lo), (_xb, hi), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9, shrinkA=0, shrinkB=0))
        for s_, y_ in ((rt, lo), (ref_off, hi)):
            ax.plot([s_["work"] * 1.12, _xb], [y_, y_], ":", color="0.6", lw=0.7, zorder=1)
        ax.text(_xb / 1.08, np.sqrt(hi * lo), f"{hi / lo:.0f}-fold faster per frame", rotation=90, ha="right", va="center", color=INK)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(min(xs) * 0.6, max(xs) * 2.2); ax.set_ylim(8, 100_000)   # ends on a labelled tick; closed box, so the top does not look cut off (2026-11-03)
        from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator  # noqa: E402
        _plain = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
        for a_ in (ax.xaxis, ax.yaxis):
            a_.set_major_formatter(_plain)
        ax.grid(True, which="major", color="0.9", lw=0.6); ax.set_axisbelow(True)
        inv = lambda v: 1000 / np.clip(v, 1e-9, None)
        sec = ax.secondary_yaxis("right", functions=(inv, inv))
        sec.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10, 100])); sec.yaxis.set_minor_locator(NullLocator())
        sec.yaxis.set_major_formatter(_plain); sec.set_ylabel("Frames per second")
        ax.set_xlabel("Computation per frame\n(time steps × solver iterations per time step)"); ax.set_ylabel("Time per frame (ms)")
        # the key above the plot (2026-11-03, by direction): only what has no label in the plot; filled = CUDA Graph on, hollow = off
        _h = [plt.Line2D([], [], color="0.65", lw=1.0),
              plt.Line2D([], [], ls="", marker="o", color=BLUE, ms=4),
              plt.Line2D([], [], ls="", marker="o", color="0.35", ms=5),
              plt.Line2D([], [], ls="", marker="o", mfc="white", mec="0.35", mew=1.2, ms=5)]
        fig.legend(_h, ["Linear fit", "Other settings tried", "CUDA Graph on (filled)", "CUDA Graph off (hollow)"],
                   loc="outside upper center", frameon=False, ncols=2, columnspacing=1.2, handletextpad=0.4, labelspacing=0.3)
        # the same checks as the paper figure: nothing under 9 pt, nothing off the page, no label over another or a tick
        fig.canvas.draw(); _r = fig.canvas.get_renderer()
        small = [(t_.get_text()[:30], t_.get_fontsize()) for t_ in fig.findobj(matplotlib.text.Text)
                 if t_.get_visible() and t_.get_text().strip() and t_.get_fontsize() < MIN_PT - 1e-6]
        assert not small, f"text below {MIN_PT} pt: {small}"
        import print_size as _PS  # noqa: E402
        assert not _PS.clipped_text(fig), f"text runs off the figure: {_PS.clipped_text(fig)}"
        _tx = [t_ for t_ in ax.texts if t_.get_visible() and t_.get_text().strip()]
        _bb = [matplotlib.text.Text.get_window_extent(t_, _r) for t_ in _tx]
        _tk = [l.get_window_extent(_r) for l in ax.get_yticklabels() + ax.get_xticklabels() if l.get_visible() and l.get_text()]
        _ovl = [(_tx[i].get_text()[:20], _tx[j].get_text()[:20]) for i in range(len(_bb)) for j in range(i + 1, len(_bb)) if _bb[i].overlaps(_bb[j])]
        _ovl += [(t_.get_text()[:20], "tick label") for t_, b_ in zip(_tx, _bb) if any(b_.overlaps(k) for k in _tk)]
        if os.environ.get("FIG_DEBUG_DIR"):
            fig.savefig(pathlib.Path(os.environ["FIG_DEBUG_DIR"]) / "fig8_idf.png", dpi=150)
        assert not _ovl, f"labels overlap: {_ovl}"
        fig.savefig(FIG / "fig_meshF_wallclock_per_frame_top_panel.png", dpi=300)
        fig.savefig(FIG / "fig_meshF_wallclock_per_frame_top_panel.pdf"); plt.close(fig)
    # 2026-11-04: this panel IS the paper's speed figure, printed at one column (3.5 in) from a 5.0 in drawing, so
    # every size is scaled up by 5.0/3.5 to land at MIN_PT (9 pt) on the page.
    _k = float(os.environ.get('TOP_PANEL_FONT_SCALE', '1.0'))   # 1.0 = the 2026-11-03 drawing; 5.0/3.5 would print at 9 pt but its labels collide
    with plt.rc_context({"font.size": MIN_PT * _k, "axes.labelsize": MIN_PT * _k, "xtick.labelsize": MIN_PT * _k,
                         "ytick.labelsize": MIN_PT * _k, "legend.fontsize": MIN_PT * _k}):
        _draw_idf_panel()
    _ntets = int(np.load(PM.LEG)["tets"].shape[0])
    _cv = [100 * np.std([r["ms"] for r in g], ddof=1) / np.mean([r["ms"] for r in g]) for g in groups.values() if len(g) > 1]
    with open(DATA / "wallclock_figure_caption_numbers.json", "w") as f:   # for the LaTeX caption: not typed by hand
        json.dump(dict(n_tets=_ntets, trend_ms_per_pass=round(float(ct[0]), 3), trend_ms_per_frame=round(float(ct[1]), 2),
                       repeat_spread_pct=[round(min(_cv), 1), round(max(_cv), 1)], figure_width_in=COL_IN, min_font_pt=MIN_PT), f, indent=1)

# ── figure: speed-up and accuracy, only once the reference has repeats ───────────────────────
if ref_mean and len(ref) >= 2:
    fam_dt = [s for s in summary if s["graph"] and s["iterations"] == 140]
    fam_it = [s for s in summary if s["graph"] and abs(s["dt_ms"] - FRAME_MS / 8) < 1e-6]
    rt_s = [s for s in summary if s["tag"] == RT_TAG]
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 6.0))
    for ax, key, sd, ylab in ((a0, "speedup", "speedup_sd", "speed-up over the reference (×), whole press"),
                              (a1, "gap_vein", "gap_vein_sd", "vein: largest gap to the reference mean\n(percentage points of resting area)"),
                              (a2, "gap_art", "gap_art_sd", "artery: largest gap to the reference mean\n(percentage points of resting area)")):
        for fam, col, lab in ((fam_dt, "#1f77b4", "time step varied, 140 iterations per step"),
                              (fam_it, "#ff7f0e", "iterations varied, 2.08 ms time step")):
            fam = sorted(fam, key=lambda s: s["work"])
            ax.errorbar([s["work"] for s in fam], [s[key] for s in fam], yerr=[np.nan_to_num(s[sd]) for s in fam],
                        fmt="o-", color=col, capsize=3, label=lab)
        if rt_s:
            ax.errorbar([rt_s[0]["work"]], [rt_s[0][key]], yerr=[np.nan_to_num(rt_s[0][sd])], fmt="*", ms=13, color="#2ca02c",
                        capsize=3, label="real-time setting: 8.33 ms time step, 35 iterations per step")
        ax.set_xscale("log"); ax.grid(True, which="both", alpha=0.3); ax.set_ylabel(ylab)
        ax.set_xlabel("solver work per frame\n(time steps per frame × iterations per step)")
    a0.set_yscale("log")
    rf = [s for s in summary if s["tag"] == REF_TAG][0]
    for ax, key in ((a1, "gap_vein"), (a2, "gap_art")):
        ax.axhspan(0, rf[key] + np.nan_to_num(rf[key + "_sd"]), color="0.85", zorder=0,
                   label="reference's own scatter about its mean")
        ax.axhline(3.0, ls=":", color="green", label="3-point acceptance bar (set 2026-09-10, before the runs)")
    h_, l_ = a1.get_legend_handles_labels(); fig.legend(h_, l_, fontsize=8, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Leg model: speed and accuracy of each solver setting, mean ± SD over repeats "
                 f"(reference: 0.099 ms time step, 140 iterations per step, n = {len(ref)})", fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95)); fig.savefig(FIG / "fig_meshF_speedup_and_accuracy.png", dpi=300); plt.close(fig)
else:
    print(f"!!! speed-up and accuracy figure NOT drawn: reference {REF_TAG} has {len(ref)} usable repeats (phase 2 pending)")

print(f"{len(runs)} usable runs, {len(excluded)} excluded; settings: " + ", ".join(f"{s['tag']} n={s['n']}" for s in summary))
