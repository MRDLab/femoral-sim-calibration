"""Independent check of tables/*.tex: re-derive every number from the RAW run files, by a different path from
make_tables.py, and fail loudly on any mismatch.

    $PY_ANATOMY scripts/verify_tables.py

Different path on purpose (a check that shares the generator's code would share its bugs):
  - leg runs: read each run's JSON directly and decide usability from the V5 queue log (its own regex here), not from
    data/meshF_repeats/per_run.csv; wall clock recomputed from the timeline, not taken from the CSV.
  - slab floors: compared with the values the record published on 2026-09-01 (mesh_and_timestep_results.md §2). They still
    apply on V5: the V5 slabs' vein rings and rest areas are bit-identical to V4's (photos/2026-10-21/README.md).
  - slab time step: every cell re-derived from the raw V5 runs; the contact period recomputed from each run's own k_e.
  - mesh counts: np.ptp and array shapes on the npz, and the leg's own metadata JSON.
  - every table cell's printed mean is parsed back out of the .tex and compared with the re-derived mean.
  - 2026-10-21, new: no cell may cite a V4 source (Mesh F, out/paper_repeats*, out/mesh_study/, femoral_v4). A table
    that mixes generations fails here, however right each number is.
Since 2026-10-21 (Femoral V5) n < 5 (leg) or n < 3 (slab) is not a failure: such cells print red with their n. Counts must
still match the raw runs exactly.
"""
import json, pathlib, re, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_meshes as PM  # noqa: E402
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent; TAB = ROOT / "tables"
V = json.load(open(TAB / "values.json")); fails = []; checked = 0
P = PM.RUN_PREFIX; REF = P + "s168_i140_ref"; RT = P + "s2_i35"
LOG = (PM.RUNS / "queue.log").read_text() if (PM.RUNS / "queue.log").exists() else ""
def check(name, got, want, tol):
    global checked; checked += 1
    if abs(float(got) - float(want)) > tol: fails.append(f"{name}: table {got} vs re-derived {want}")
def lists(key, got, want, tol):
    got, want = sorted(got), sorted(want)
    if len(got) != len(want): fails.append(f"{key}: n {len(got)} vs {len(want)} usable raw runs"); return
    for g, w in zip(got, want): check(key, g, w, tol)

# no V4 source anywhere
V4_MARKS = ("c4e13e7d7574", "6f3c9f86ab61", "paper_repeats", "/mesh_study/", "femoral_v4", "legF_", "legE_", "/mesh_leg/")
for key, val in V.items():
    checked += 1
    if any(m in val["source"] for m in V4_MARKS):
        fails.append(f"{key}: cites a V4 source ({val['source'][:90]})")

# leg: raw JSON + the queue log, read with this script's own pattern
ok = {(int(m[1]), m[2]) for m in re.finditer(r"DONE r(\d+) (legV5_\S+) \(.*?\) rc=0 GPU IDLE", LOG)}
def raw(tag):
    out = []
    for rep, t in sorted(ok):
        if t != tag: continue
        p = PM.RUNS / "leg" / f"r{rep}" / f"{tag}.json"
        d = json.load(open(p))
        assert d["mesh_sha256_12"] == PM.LEG_SHA, p
        rows = {r["depth_mm"]: r for r in d["depth_rows"]}
        if len(rows) != 7: continue
        w = np.array([f["wall"] for f in d["timeline"]])
        out.append(dict(ms=np.median(np.diff(w)) * 1e3, rows=rows, graph=bool(d["cuda_graph"])))
    return out
for key, val in V.items():
    if key.startswith("vc."):
        depth, field = float(key[3:key.rindex(".")]), key[key.rindex(".") + 1:]   # "vc.25.4.vein": the depth has a dot
        lists(key, val["value"], [r["rows"][depth][f"{field}_pct"] for r in raw(RT) if r["graph"]], 0.01)
    elif key.startswith("app."):
        _, body, rest = key.split(".", 2); tag, depth = rest.split(".", 1)
        lists(key, val["value"], [r["rows"][float(depth)][f"{body}_pct"] for r in raw(tag) if r["graph"]], 0.01)

# slab floors against the record's published table (V5 rings and rest areas are V4's, bit for bit)
rec = {"2p5": 35.6, "2": 29.7, "1p5": 22.6, "1p25": 19.3, "1p0": 15.8}
for dx, f in rec.items(): check(f"mesh.{dx}.floor", V[f"mesh.{dx}.floor"]["value"], f, 0.05)
# leg floor: recomputed from the mesh file
_leg = np.load(PM.LEG); _lm = json.load(open(PM.LEG.with_suffix(".json")))
for g, w in zip(V["leg.floor_pct"]["value"], (8.9, 12.5)):
    check("leg.floor_pct", g, 100 * 0.8 * _lm["fine_dx_m"] * 1e3 * w / (float(_leg["rest_area_vein"][0]) * 1e6), 0.06)

# slab runs, from the queue log with this script's own pattern
slab_ok = {(int(m[1]), m[2]): float(m[3]) for m in re.finditer(r"DONE r(\d+) slab (\S+) \(.*?\) rc=0 GPU IDLE after (\d+) s", LOG)}
def slab_raw(tag):
    out = []
    for (rep, t), sec in sorted(slab_ok.items()):
        if t != tag: continue
        p = PM.RUNS / "slab" / f"r{rep}" / f"{tag}.json"
        if not p.exists(): continue
        d = json.load(open(p))
        if not d.get("complete") or len(d["rows"]) != 7: continue
        assert d["mesh_provenance"]["npz"]["sha256"][:12] in {s for _, s in PM.SLAB.values()}, p
        out.append(dict(rows={r["depth_mm"]: r for r in d["rows"]}, minutes=sec / 60.0, ke=float(d["soft_contact_ke"])))
    return out
_ref = slab_raw("dx1p5_s896_i140")
_refm = {x: np.mean([r["rows"][x]["vein_pct"] for r in _ref]) for x in _ref[0]["rows"]} if _ref else None
for s_ in (168, 67, 17, 8, 3, 2):
    rs = slab_raw(f"ts_s{s_}_i140")
    if f"ts.{s_}.ratio" in V:
        ke = (_ref or rs)[0]["ke"] if (_ref or rs) else None
        if ke: check(f"ts.{s_}.ratio", V[f"ts.{s_}.ratio"]["value"], (1000 / 60 / s_ * 1e3) / (2 * np.pi * np.sqrt(1000 * 1.5e-3 ** 3 / ke) * 1e6), 0.006)
    if f"ts.{s_}.gap" not in V:
        checked += 1
        if rs: fails.append(f"ts.{s_}: {len(rs)} usable runs but no table cell")
        continue
    if _refm: lists(f"ts.{s_}.gap", V[f"ts.{s_}.gap"]["value"], [max(abs(r["rows"][x]["vein_pct"] - _refm[x]) for x in r["rows"]) for r in rs], 0.005)
    lists(f"ts.{s_}.wall_min", V[f"ts.{s_}.wall_min"]["value"], [r["minutes"] for r in rs], 0.05)
# slab iterations at 2.08 ms (text only, no table): re-derived from the raw runs against the ts_s8_i140 control
_ctl = slab_raw("ts_s8_i140")
_ctlm = {x: np.mean([r["rows"][x]["vein_pct"] for r in _ctl]) for x in _ctl[0]["rows"]} if _ctl else None
for it in (70, 35, 20, 10):
    rs = slab_raw(f"ts_s8_i{it}")
    if f"it.{it}.gap" not in V:
        checked += 1; fails.append(f"it.{it}.gap: {len(rs)} usable runs but no cell"); continue
    if _ctlm: lists(f"it.{it}.gap", V[f"it.{it}.gap"]["value"], [max(abs(r["rows"][x]["vein_pct"] - _ctlm[x]) for x in r["rows"]) for r in rs], 0.005)
for dx in ("2p5", "2", "1p5", "1p25", "1p0"):
    rs = slab_raw(f"dx{dx}_s896_i140")
    lists(f"mesh.{dx}.v25", V[f"mesh.{dx}.v25"]["value"], [r["rows"][25.4]["vein_pct"] for r in rs], 0.005)
    lists(f"mesh.{dx}.wall_min", V[f"mesh.{dx}.wall_min"]["value"], [r["minutes"] for r in rs], 0.05)
    # headroom is a relationship: each run's vein at 25.4 mm minus that mesh's floor
    v25, hd, fl = sorted(V[f"mesh.{dx}.v25"]["value"]), sorted(V[f"mesh.{dx}.head"]["value"]), V[f"mesh.{dx}.floor"]["value"]
    if len(v25) != len(hd): fails.append(f"mesh.{dx}.head: n {len(hd)} vs {len(v25)}"); continue
    for v, h in zip(v25, hd): check(f"mesh.{dx}.head", h, v - fl, 0.06)

# graph on/off cells against the raw run files (no fallback to any other run)
for key in [k for k in V if k.startswith("graph.") and k.endswith((".on", ".off"))]:
    tag = {"graph.2x35.on": P + "s2_i35_graphon", "graph.2x35.off": P + "s2_i35_graphoff", "graph.8x35.on": P + "s8_i35_graphon",
           "graph.8x35.off": P + "s8_i35_graphoff", "graph.168x140.on": REF, "graph.168x140.off": REF + "_graphoff"}.get(key)
    if not tag: continue
    lists(key, V[key]["value"], [r["ms"] for r in raw(tag)], 0.01)

# mesh counts and settings
check("geo.leg.tets", V["geo.leg.tets"]["value"], _lm["n_tets"], 0); check("geo.leg.nodes", V["geo.leg.nodes"]["value"], _lm["n_nodes"], 0)
check("geo.leg.pinned", V["geo.leg.pinned"]["value"], _lm["n_pinned"], 0)
check("geo.leg.w", V["geo.leg.w"]["value"], np.ptp(_leg["rest"][:, 0]) * 1e3, 0.05)
check("geo.leg.l", V["geo.leg.l"]["value"], np.ptp(_leg["rest"][:, 1]) * 1e3, 0.05)
slab = np.load(PM.SLAB_15)
check("geo.slab.tets", V["geo.slab.tets"]["value"], slab["tets"].shape[0], 0)
check("geo.slab.d", V["geo.slab.d"]["value"], np.ptp(slab["rest"][:, 2]) * 1e3, 0.05)
run = json.load(open(PM.LIVE / f"{PM.LIVE_RUN}.json")); assert run["mesh_sha256_12"] == PM.LEG_SHA
check("num.leg.dt_ms", V["num.leg.dt_ms"]["value"], 1000 / 60 / run["substeps"], 0.006)
check("num.leg.iter", V["num.leg.iter"]["value"], run["iterations"], 0)
check("num.leg.gamma", V["num.leg.gamma"]["value"], float(np.unique(_leg["tet_materials"][:, 2])[0]), 0)

# speed-and-accuracy cells (tab:speed) re-derived from the raw leg run files, graph on, paired runs excluded by tag
_lr = [r for r in raw(REF) if r["graph"]]
_lrm = {k: {x: np.mean([r["rows"][x][f"{k}_pct"] for r in _lr]) for x in _lr[0]["rows"]} for k in ("vein", "artery")} if _lr else None
_lrp = np.mean([r["rows"][25.4]["wall_at_settle_s"] for r in _lr]) if _lr else None
for key, val in V.items():
    if not key.startswith("spd."): continue
    _, tag, q = key.split(".")
    rs = [r for r in raw(tag) if r["graph"]]
    if q != "ms" and not _lr:
        checked += 1
        if val["value"]: fails.append(f"{key}: values without any reference run")
        continue
    want = {"ms": lambda: [r["ms"] for r in rs], "speedupf": lambda: [np.mean([q["ms"] for q in _lr]) / r["ms"] for r in rs],
            "speedup": lambda: [_lrp / r["rows"][25.4]["wall_at_settle_s"] for r in rs],
            "gapv": lambda: [max(abs(r["rows"][x]["vein_pct"] - _lrm["vein"][x]) for x in r["rows"]) for r in rs],
            "gapa": lambda: [max(abs(r["rows"][x]["artery_pct"] - _lrm["artery"][x]) for x in r["rows"]) for r in rs]}[q]()
    lists(key, val["value"], want, 0.01 if q != "ms" else 0.01 * max(1.0, max(want, default=1.0)))

# vein-wall sensitivity cells against the raw run files; usable runs from the queue log, digests from the manifest
sok = {(int(m[1]), m[2]) for m in re.finditer(r"DONE r(\d+) sens (\S+) \(.*?\) rc=0 GPU IDLE", LOG)}
_man = json.load(open(PM.SENS / "manifest.json")); assert _man["source_sha256_12"] == PM.LEG_SHA
sdig = dict({c: v["sha256_12"] for c, v in _man["cases"].items()}, control=PM.LEG_SHA)
for key, val in V.items():
    if not key.startswith("sens.") or key == "sens.unsettled": continue
    _, case, body, depth = key.split(".", 3)
    want = []
    for rep, c in sorted(sok):
        if c != case: continue
        d = json.load(open(PM.RUNS / "leg_sens" / case / f"r{rep}" / f"{case}.json"))
        assert d["mesh_sha256_12"] == sdig[case], (case, rep)
        rows = {r["depth_mm"]: r for r in d["depth_rows"]}
        if len(rows) == 7: want.append(rows[float(depth)][f"{body}_pct"])
    lists(key, val["value"], want, 0.01)

# printed means in the .tex match the cell values
for tex in TAB.glob("*.tex"):
    for line in tex.read_text().splitlines():
        for m in re.finditer(r"(-?\d+\.\d+)\$\\pm\$", line):
            checked += 1  # a mean with an SD: it must equal the mean of some cell list
            mean = float(m[1]); tol = 10 ** -int(len(m[1].split('.')[1]))
            # a value printed in hours (followed by ~h) is stored in minutes: compare mean/60 for those
            hours = "~h" in line[m.end():].split("&")[0]
            scale = 60.0 if hours else 1.0
            if not any(isinstance(v["value"], list) and len(v["value"]) > 1 and abs(np.mean(v["value"]) / scale - mean) < tol for v in V.values()):
                fails.append(f"{tex.name}: printed mean {mean} matches no cell")
print(f"checked {checked}; {len(fails)} failures")
for f in fails: print("  FAIL", f)
sys.exit(1 if fails else 0)
