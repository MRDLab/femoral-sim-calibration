"""The paper's headline numbers, each from its run files with its definition beside it. 2026-10-22.

    $PY_ANATOMY scripts/headline_numbers.py

Writes data/headline_numbers.json. Every number the abstract, the results and the conclusion quote about speed and accuracy
comes from here, so an audit can trace each one to its runs. Definitions are the V4 paper's, unchanged:
  full loop        the live coupled run (solver + ray-traced ultrasound + rendered 3-D view, one GPU), ONE run as in V4:
                   frames per second = 1000 / median frame-to-frame wall clock (V4: 60.1 ms -> 16.6 FPS, legF_s2_i35)
  full-loop gap    largest |live run - reference mean| over the seven depths, vein and artery, percentage points
  solver only      the real-time setting (8.33 ms time step, 35 iterations per step, CUDA graph on), mean +- SD over its
                   repeats of the median frame-to-frame wall clock; frames per second = 1000 / that mean
  solver gap       per repeat, largest |run - reference mean| over the depths; mean +- SD over repeats
  reference        0.099 ms time step, 140 iterations per step: wall clock per frame with and without CUDA graphs
  speed-up         reference without CUDA graphs / real-time setting, per frame (V4: about 980x)
Two paths, and they must agree: values are computed from the raw run JSONs (chosen by the queue log), then compared with
data/meshF_repeats/per_run.csv (analyze_meshF_repeats.py). Any disagreement stops the script.
A number whose runs have not landed is written as null with "pending"; it is never filled from V4.
"""
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_meshes as PM  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
P = PM.RUN_PREFIX
RT, REF, REF_OFF = P + "s2_i35", P + "s168_i140_ref", P + "s168_i140_ref_graphoff"


def raw(tag):
    out = []
    for rep, p in PM.usable("leg", tag):
        d = json.load(open(p)); assert d["mesh_sha256_12"] == PM.LEG_SHA, p
        w = np.array([f["wall"] for f in d["timeline"]])
        out.append(dict(rep=rep, ms=float(np.median(np.diff(w))) * 1e3, rows={r["depth_mm"]: r for r in d["depth_rows"]},
                        graph=bool(d["cuda_graph"])))
    return out


def ms(xs):
    return dict(mean=round(float(np.mean(xs)), 3), sd=round(float(np.std(xs, ddof=1)), 3) if len(xs) > 1 else None, n=len(xs))


H = dict(generation=PM.GENERATION, leg_mesh=PM.LEG_SHA)
ref, rt, roff = raw(REF), raw(RT), raw(REF_OFF)
assert ref and rt, "the reference and the real-time setting must both have runs"
depths = sorted(ref[0]["rows"])
refm = {k: {x: np.mean([r["rows"][x][f"{k}_pct"] for r in ref]) for x in depths} for k in ("vein", "artery")}

# second path: per_run.csv must hold the same per-run values
csvr = {(int(r["repeat"]), r["run"]): r for r in csv.DictReader(open(ROOT / "data" / "meshF_repeats" / "per_run.csv"))}
for tag, rs in ((REF, ref), (RT, rt), (REF_OFF, roff)):
    for r in rs:
        c = csvr.get((r["rep"], tag))
        assert c is not None, f"{tag} r{r['rep']} missing from per_run.csv: rerun analyze_meshF_repeats.py"
        assert abs(float(c["median_wall_clock_per_frame_ms"]) - r["ms"]) < 1e-3, (tag, r["rep"], "ms")
        assert abs(float(c["vein_pct_25.4mm"]) - r["rows"][25.4]["vein_pct"]) < 1e-3, (tag, r["rep"], "vein")

# solver only, real-time setting
rt_ms = [r["ms"] for r in rt]
gv = [max(abs(r["rows"][x]["vein_pct"] - refm["vein"][x]) for x in depths) for r in rt]
ga = [max(abs(r["rows"][x]["artery_pct"] - refm["artery"][x]) for x in depths) for r in rt]
H["solver_only_realtime"] = dict(setting="8.33 ms time step, 35 iterations per step, CUDA graph on, solver only",
                                 ms_per_frame=ms(rt_ms), fps=round(1000 / float(np.mean(rt_ms)), 1),
                                 gap_vein_pts=ms(gv), gap_artery_pts=ms(ga), runs=[f"r{r['rep']}" for r in rt])
# the reference's own scatter about its mean (the floor the gaps are read against)
H["reference_self_scatter_pts"] = dict(vein=ms([max(abs(r["rows"][x]["vein_pct"] - refm["vein"][x]) for x in depths) for r in ref]),
                                       artery=ms([max(abs(r["rows"][x]["artery_pct"] - refm["artery"][x]) for x in depths) for r in ref]))
# reference cost, graph on and off; speed-up against graph off
H["reference_s_per_frame"] = dict(graph_on=ms([r["ms"] / 1e3 for r in ref]),
                                  graph_off=ms([r["ms"] / 1e3 for r in roff]) if roff else "pending: graph-off reference runs not landed")
H["speedup_realtime_vs_reference_graph_off"] = (round(float(np.mean([r["ms"] for r in roff]) / np.mean(rt_ms)), 0) if roff
                                                else "pending: graph-off reference runs not landed")
H["speedup_realtime_vs_reference_graph_on"] = round(float(np.mean([r["ms"] for r in ref]) / np.mean(rt_ms)), 0)

# full loop: the live coupled run, one run as in V4
lp = PM.LIVE / f"{PM.LIVE_RUN}.json"; L = json.load(open(lp))
assert L["mesh_sha256_12"] == PM.LEG_SHA and L["substeps"] == 2 and L["iterations"] == 35 and L["cuda_graph"], lp
lw = np.array([f["wall"] for f in L["timeline"]]); lms = float(np.median(np.diff(lw))) * 1e3
lrows = {r["depth_mm"]: r for r in L["depth_rows"]}
assert sorted(lrows) == depths and all(r["veto"] == "none" for r in lrows.values()), "live run did not settle at every depth"
H["full_loop"] = dict(run=lp.name, n_runs=1, setting="8.33 ms time step, 35 iterations per step, CUDA graph on",
                      includes="solver + ray-traced ultrasound (pipelined) + rendered 3-D view, one GPU",
                      median_ms_per_frame=round(lms, 1), fps=round(1000 / lms, 1),
                      mean_fps_whole_run=round((len(lw) - 1) / (lw[-1] - lw[0]), 1),
                      gap_vein_pts=round(max(abs(lrows[x]["vein_pct"] - refm["vein"][x]) for x in depths), 2),
                      gap_artery_pts=round(max(abs(lrows[x]["artery_pct"] - refm["artery"][x]) for x in depths), 2),
                      note="one run, as in the V4 paper; its rate is not a mean and carries no SD")
H["v4_for_comparison"] = dict(full_loop_fps=16.6, full_loop_ms=60.1, full_loop_gap_vein=2.27, full_loop_gap_artery=1.58,
                              solver_only_ms=30.1, solver_only_fps=33, speedup_vs_graph_off=980,
                              source="main_V1.tex as of 2026-10-21 (Mesh F, c4e13e7d7574); NOT V5 results")
json.dump(H, open(ROOT / "data" / "headline_numbers.json", "w"), indent=1)
print(json.dumps({k: v for k, v in H.items() if k != "v4_for_comparison"}, indent=1))
