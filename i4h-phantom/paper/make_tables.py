"""Generate the paper's results tables from the run files, so no number is copied by hand.

    $PY_ANATOMY scripts/make_tables.py

Writes tables/<name>.tex (a whole table environment each; main_V1.tex \\input's them) and tables/values.json
(every cell with its source file), which scripts/verify_tables.py re-derives independently from the raw run files.

Since 2026-10-21 every source is Femoral V5, the cast leg (V4 without the pubic ramus, cut to 81 mm), named once in
scripts/paper_meshes.py and checked by content:
  leg     the V5 leg mesh, digest PM.LEG_SHA; repeats: data/meshF_repeats/per_run.csv (scripts/analyze_meshF_repeats.py)
  slab    the five V5 slab meshes, digests in PM.SLAB; runs: i4h-phantom/out/paper_v5/slab/r*/ through the queue log
  runs    only runs the V5 queue logged "rc=0 GPU IDLE". 🚨 There is NO fallback to a V4 run: a setting without V5 runs
          prints \\pending{awaiting runs}. (The V4 generator read the August originals when repeats were missing.)
Layout (2026-10-05): main_V1.tex loads tabularx. Column tables (simulation_geometry, phantom_parameters,
cost_and_graph) are set to \\columnwidth and wrap; wide ones (mesh_results, timestep_slab, speed_accuracy) are table*.
2026-10-30, from the author's main_V2 edits: float placement per table (FLOAT below), "benchtop phantom" for the physical
  phantom (never "cast"), "reference" / "chosen" for the two named leg settings, and one unit per quantity: time step and
  wall clock per frame in ms, whole-press wall clock in min, moduli in kPa. Tables are written in the paper's order (ORDER).
2026-11-04: "tolerance" everywhere a caption used "bar" (the text no longer says "acceptance bar").
Formatting rule: a value from fewer runs than planned (leg 5, slab 3) is wrapped in \\pending{} and shows its n.
Definitions (unchanged from the record):
  lumen %        settled lumen area / rest area x 100, at each commanded depth
  largest gap    max over the seven depths of |this run - reference|, percentage points (reference = mean of its repeats)
  floor          slab: half the rest ring perimeter x 0.8 x element size / rest area (femoral_fem.closure_floor_pct, item 27);
                 leg: 0.8 x element size x the vein's cut width 8.9-12.5 mm / rest area. Both nominal
  wall clock     leg: median frame-to-frame wall clock (ms); slab: whole seven-depth press, solo runs only (queue log)
"""
import csv
import json
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_meshes as PM  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TAB = ROOT / "tables"; TAB.mkdir(exist_ok=True)
FRAME_MS = 1000.0 / 60.0
DEPTHS = [3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 25.4]
VALUES = {}          # cell id -> {value, source}
AWAIT = "\\pending{awaiting runs}"
P = PM.RUN_PREFIX


# 2026-10-30: float placement and order as in the author's main_V2.tex; every table goes through emit(), written at the end.
# 2026-11-04: placement as in the author's Isaac_Sim_Calibration_Paper.zip (main_V4/V5): column tables float with [!tbp].
# The zip set the appendix tables as one-column `table`[H] for V4's \onecolumn appendix; the two-column supplement.tex
# needs them as `table*` (eight columns overflow a column), so ENV is empty and they float at the top of a page.
FLOAT = {"simulation_geometry": "!htbp", "phantom_parameters": "!tbp", "cost_and_graph": "!tbp", "vein_closure": "!tbp"}
ENV = {}   # name -> environment to switch to; empty = as emitted (table* for the wide ones)
ORDER = ["simulation_geometry", "phantom_parameters", "mesh_results", "timestep_slab", "cost_and_graph", "speed_accuracy",
         "vein_closure", "appendix_vein", "appendix_artery", "appendix_wall_sensitivity"]   # cast_validation: fig_cast_validation.py
OUT = {}


def emit(name, tex):
    env = re.match(r"\\begin\{(table\*?)\}\[t\]", tex)
    assert env, name
    tex = tex.replace(f"[t]", f"[{FLOAT.get(name, 't')}]", 1)
    if name in ENV:   # change the environment on both ends, never one of them
        tex = tex.replace(f"\\begin{{{env.group(1)}}}", f"\\begin{{{ENV[name]}}}", 1)
        assert tex.rstrip().endswith(f"\\end{{{env.group(1)}}}"), name
        tex = tex.rstrip()[: -len(f"\\end{{{env.group(1)}}}")] + f"\\end{{{ENV[name]}}}\n"
    OUT[name] = tex


def _public(source):
    """Source paths as published: relative to the simulation output folder, never a private absolute path (2026-11-05)."""
    s = str(source).replace(str(PM.PH) + "/", "simulation output: ")
    assert "/home/" not in s and "i4h-" not in s, f"private path left in a published source: {s}"
    return s


def cell(key, value, source):
    VALUES[key] = {"value": value, "source": _public(source)}
    return value


def fmt(vals, digits=1, unit="", target=5, sd=True):
    """mean (+- SD) with n; \\pending{} when n < target; AWAIT when there is nothing."""
    v = np.asarray(vals, float); n = len(v)
    if n == 0:
        return AWAIT
    m = f"{v.mean():.{digits}f}"
    s = f"{m}$\\pm${v.std(ddof=1):.{digits}f}" if (sd and n > 1) else m
    s += unit
    return s if n >= target else f"\\pending{{{s} (n={n})}}"


# ── meshes, by content ────────────────────────────────────────────────────────────────────────
LEG = PM.verified(PM.LEG, PM.LEG_SHA, "leg mesh")
leg_npz = np.load(LEG, allow_pickle=True); leg_meta = json.load(open(LEG.with_suffix(".json")))
SLAB = {dx: PM.verified(p, s, f"{dx} slab") for dx, (p, s) in PM.SLAB.items()}
DX_MM = {"2p5": 2.5, "2": 2.0, "1p5": 1.5, "1p25": 1.25, "1p0": 1.0}
HAS_RAMUS = bool((leg_npz["tet_body"] == "PubicRamus").any())
LEG_FLOOR = [100 * 0.8 * leg_meta["fine_dx_m"] * 1e3 * w / (float(leg_npz["rest_area_vein"][0]) * 1e6) for w in (8.9, 12.5)]
cell("leg.floor_pct", [round(x, 1) for x in LEG_FLOOR], f"{LEG.name}: 0.8 dx x cut width 8.9-12.5 mm / rest area")


def bbox(npz):
    X = npz["rest"] * 1e3; e = X.max(0) - X.min(0); return e


# ── slab runs: the V5 queue's only ────────────────────────────────────────────────────────────
def load_press(p):
    d = json.load(open(p)); rows = {r["depth_mm"]: r for r in d["rows"]}
    return d, rows


def slab_runs(tag):
    return [str(p) for _, p in PM.usable("slab", tag, rows_key="rows")]


SLAB_S = PM.queue_seconds("slab")


def slab_minutes(tag, files):
    """Whole-press wall clock (min) of exactly these runs, from the queue log. Every queue run is solo (idle-GPU rule)."""
    out = []
    for f in files:
        rep = int(pathlib.Path(f).parent.name[1:])
        if (rep, tag) in SLAB_S:
            out.append(SLAB_S[(rep, tag)] / 60.0)
    return out


def floor_pct(dx):
    npz = np.load(SLAB[dx], allow_pickle=True); X = npz["rest"] * 1e3
    ring = X[npz["ring_vein"]][:, [0, 2]]
    per = np.sum(np.linalg.norm(np.roll(ring, -1, 0) - ring, axis=1))
    return 100 * (0.5 * per * 0.8 * DX_MM[dx]) / float(npz["rest_area_vein"][0] * 1e6)


# ── table 1: simulation geometry and numerical parameters ─────────────────────────────────────
slab15 = np.load(SLAB["1p5"], allow_pickle=True)
leg_run_p = PM.LIVE / f"{PM.LIVE_RUN}.json"
leg_run = json.load(open(leg_run_p))
assert leg_run["mesh_sha256_12"] == PM.LEG_SHA, f"{leg_run_p} is not on the paper's leg mesh"
ref_files = slab_runs("dx1p5_s896_i140")
slab_ref = json.load(open(ref_files[0])) if ref_files else None
if slab_ref is not None:
    assert slab_ref["mesh_provenance"]["npz"]["sha256"][:12] == PM.SLAB_15_SHA
es, el = bbox(slab15), bbox(leg_npz)
g_slab, g_leg = np.unique(slab15["tet_materials"][:, 2]), np.unique(leg_npz["tet_materials"][:, 2])
assert len(g_slab) == 1 and len(g_leg) == 1, "damping is not uniform"
ke_leg = float(leg_run["ke"])
ke_slab = float(slab_ref["soft_contact_ke"]) if slab_ref else None
REF_SRC = pathlib.Path(ref_files[0]).name if ref_files else "awaiting runs"
size_leg = (f"{leg_meta['fine_dx_m']*1e3:g}~mm near the vessels; {leg_meta['bone_dx_m']*1e3:g}~mm femur; "
            + (f"{leg_meta['ramus_dx_m']*1e3:g}~mm pubic ramus; " if HAS_RAMUS else "")
            + f"{leg_meta['coarse_dx_m']*1e3:g}~mm elsewhere")
rows = [
    ("Domain", "block cut from the limb 20~mm below the inguinal ligament; side and deep faces held fixed",
     f"whole thigh segment, {el[1]:.0f}~mm long, held by a small fixed patch where the thigh rests on the table"),
    ("Lateral width", f"{cell('geo.slab.w', round(es[0],1), SLAB['1p5']):.1f}~mm", f"{cell('geo.leg.w', round(el[0],1), LEG):.1f}~mm"),
    ("Axial length", f"{cell('geo.slab.l', round(es[1],1), SLAB['1p5']):.1f}~mm", f"{cell('geo.leg.l', round(el[1],1), LEG):.1f}~mm"),
    ("Depth", f"{cell('geo.slab.d', round(es[2],1), SLAB['1p5']):.1f}~mm", f"{cell('geo.leg.d', round(el[2],1), LEG):.1f}~mm"),
    ("Nodes", f"{cell('geo.slab.nodes', len(slab15['rest']), SLAB['1p5']):,}", f"{cell('geo.leg.nodes', len(leg_npz['rest']), LEG):,}"),
    ("Tetrahedra", f"{cell('geo.slab.tets', len(slab15['tets']), SLAB['1p5']):,}", f"{cell('geo.leg.tets', len(leg_npz['tets']), LEG):,}"),
    ("Nodes held fixed", f"{cell('geo.slab.pinned', len(slab15['pinned']), SLAB['1p5']):,} (side and deep faces, bone)",
     f"{cell('geo.leg.pinned', len(leg_npz['pinned']), LEG):,} (bone {leg_meta['n_bone_pinned']}, table patch {leg_meta['n_table_pinned']})"),
    ("Element size", "1.5~mm, uniform", size_leg),
    ("Time step", (f"{cell('num.slab.dt_us', round(FRAME_MS/slab_ref['substeps']*1e3,2), REF_SRC) / 1e3:.4f}~ms "
                   f"({slab_ref['substeps']} per frame)") if slab_ref else AWAIT,
     f"{cell('num.leg.dt_ms', round(FRAME_MS/leg_run['substeps'],2), leg_run_p.name):.2f}~ms ({leg_run['substeps']} per frame)"),
    ("Iterations per time step", f"{cell('num.slab.iter', slab_ref['iterations'], REF_SRC)}" if slab_ref else AWAIT,
     f"{cell('num.leg.iter', leg_run['iterations'], leg_run_p.name)}"),
    ("Contact stiffness $k_e$", f"{cell('num.slab.ke', ke_slab, REF_SRC):,.0f}~N/m" if slab_ref else AWAIT,
     f"{cell('num.leg.ke', ke_leg, leg_run_p.name):,.0f}~N/m"),
    ("Damping $\\gamma$ (MC)", f"{cell('num.slab.gamma', float(g_slab[0]), SLAB['1p5']):.1f}~s", f"{cell('num.leg.gamma', float(g_leg[0]), LEG):.1f}~s"),
]
body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in rows)
emit("simulation_geometry", f"""\\begin{{table}}[t]
\\centering
\\caption{{Geometry and numerical settings of the two meshes, read from the mesh and run files. Sizes are the bounding box
of the resting mesh; lateral is across the limb, axial along it, depth from the skin inward. The leg column gives the
chosen setting; the reference setting is 0.099~ms (168 time steps per frame) with 140 iterations per step.
MC: modeling choice, not backed by a source. $\\gamma$ is in seconds in the damping form the solver uses.}}
\\label{{tab:simulation_geometry}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabularx}}{{\\columnwidth}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{2.35cm}}>{{\\raggedright\\arraybackslash}}X>{{\\raggedright\\arraybackslash}}X@{{}}}}
\\hline
\\textbf{{Parameter}} & \\textbf{{Slab (parameter study)}} & \\textbf{{Leg model}} \\\\
\\hline
{body}
\\hline
\\end{{tabularx}}
\\end{{table}}
""")

# phantom: the CAD design, as meshed, beside the benchtop phantom (molded from the same CAD; values measured later)
import trimesh  # noqa: E402
CADD = PM.ANATOMY; sec = json.load(open(PM.SECTION)); B = sec["bodies"]
R = B["SoftTissue"]["r_out"] * 1e3


def intima_depth(n):   # skin to the vessel's inner wall, perpendicular to the skin
    c = np.array(B[n]["c"]) * 1e3; return R - np.hypot(*c) - B[n]["r_in"] * 1e3


soft = trimesh.load(CADD / "SoftTissue.obj", force="mesh", process=False); ext = soft.bounds[1] - soft.bounds[0]
av = np.hypot(*(np.array(B["Vein"]["c"]) - np.array(B["Artery"]["c"]))) * 1e3
geo = [("Length along the limb", f"{ext[2]:.0f}~mm"), ("Diameter", f"{2 * R:.1f}~mm"),
       ("Skin and fat thickness", f"{R - B['SoftTissue']['r_in'] * 1e3:.1f}~mm"),
       ("Artery lumen diameter", f"{2 * B['BloodArtery']['r'] * 1e3:.2f}~mm"), ("Vein lumen diameter", f"{2 * B['BloodVein']['r'] * 1e3:.2f}~mm"),
       ("Artery wall thickness", f"{(B['Artery']['r_out'] - B['Artery']['r_in']) * 1e3:.2f}~mm"),
       ("Vein wall thickness", f"{(B['Vein']['r_out'] - B['Vein']['r_in']) * 1e3:.2f}~mm"),
       ("Skin to artery inner wall", f"{intima_depth('Artery'):.1f}~mm"), ("Skin to vein inner wall", f"{intima_depth('Vein'):.1f}~mm"),
       ("Artery to vein, center to center", f"{av:.1f}~mm")]
mat = [("Tissue model", "Neo-Hookean; moduli in Table~\\ref{tab:anatomy}"), ("Soft-tissue material", "--"),
       ("Young's modulus, muscle / fat", "--"), ("Vessel walls", "--")]   # moduli live in tab:anatomy only (blended walls)
for k_, v_ in geo + mat:
    cell(f"phantom.{k_}", v_, f"{PM.SECTION.name}, SoftTissue.obj ({PM.GENERATION})")

# 2026-10-30: the cast's column, from the cast's own section (out/femoral_cast/section_s020.json), computed by the SAME
# functions as the CAD column so the two cannot drift. Geometry is the mold design and catalogue tubes: not caliper-measured.
CSEC = PM.ANATOMY.parent / "femoral_cast" / "section_s020.json"
CB = json.load(open(CSEC))["bodies"]; CR = CB["SoftTissue"]["r_out"] * 1e3
c_intima = lambda n: CR - np.hypot(*(np.array(CB[n]["c"]) * 1e3)) - CB[n]["r_in"] * 1e3
c_av = np.hypot(*(np.array(CB["Vein"]["c"]) - np.array(CB["Artery"]["c"]))) * 1e3
cast_geo = [f"{ext[2]:.0f}~mm", f"{2 * CR:.1f}~mm", f"{CR - CB['SoftTissue']['r_in'] * 1e3:.1f}~mm",
            f"{2 * CB['BloodArtery']['r'] * 1e3:.2f}~mm", f"{2 * CB['BloodVein']['r'] * 1e3:.2f}~mm",
            f"{(CB['Artery']['r_out'] - CB['Artery']['r_in']) * 1e3:.2f}~mm", f"{(CB['Vein']['r_out'] - CB['Vein']['r_in']) * 1e3:.2f}~mm",
            f"{c_intima('Artery'):.1f}~mm", f"{c_intima('Vein'):.1f}~mm", f"{c_av:.1f}~mm"]
cast_mat = ["Neo-Hookean; values below", "SimuGel blends (muscle \\#0+\\#1, fat \\#2+\\#0)",
            "510 / 380~kPa (supplier, mixed by mass)", "silicone tubes, 70A artery / 35A vein"]
for (k_, _), v_ in zip(geo + mat, cast_geo + cast_mat):
    cell(f"phantom.cast.{k_}", v_, f"{CSEC.name} (cast section); materials.py Phantom* entries")
rows_ph = "\n".join(f"{k_} & {v_} & {c_} \\\\" for (k_, v_), c_ in zip(geo, cast_geo)) + "\n\\hline\n" + \
          "\n".join(f"{k_} & {v_} & {c_} \\\\" for (k_, v_), c_ in zip(mat, cast_mat))
emit("phantom_parameters", f"""\\begin{{table}}[t]
\\centering
\\caption{{The benchtop phantom against the anatomical model. Leg model (CAD): the cross-section 20~mm below the inguinal
ligament, as meshed, with the benchtop phantom's two simplifications (no pubic ramus; {ext[2]:.0f}~mm long); moduli in
Table~\\ref{{tab:anatomy}}. Benchtop: the mold's design dimensions and catalogue tube sizes, not caliper measurements of the
benchtop phantom; moduli are supplier values. It has no sheath, femoral canal or nerve.}}
\\label{{tab:phantom_parameters}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabularx}}{{\\columnwidth}}{{@{{}}>{{\\raggedright\\arraybackslash}}X>{{\\centering\\arraybackslash}}p{{2.0cm}}>{{\\centering\\arraybackslash}}p{{2.35cm}}@{{}}}}
\\hline
\\textbf{{Parameter}} & \\textbf{{Leg model (CAD)}} & \\textbf{{Benchtop}} \\\\
\\hline
{rows_ph}
\\hline
\\end{{tabularx}}
\\end{{table}}
""")

# 2026-10-30: the cast phantom's per-trial lumen areas, written by fig_cast_validation.py (which also writes the cast
# table), registered here so verify_tables.py checks every printed cast mean against its trials.
_ct = ROOT / "data" / "cast_trials.csv"
if _ct.exists():
    for r in csv.DictReader(open(_ct)):
        cell(f"cast.{float(r['depth_mm']):g}.{r['vessel']}", [float(x) for x in r["trial_values_pct"].split(";")],
             "data/cast_trials.csv (from Phantoms/.../trials_2026-10-30/phantom_lumen_measurements_clean.csv)")

# ── slab element size: cost and validity (what the convergence figure cannot show) ─────────────
er, n_mesh = [], 0
for dx in ("2p5", "2", "1p5", "1p25", "1p0"):
    tag = f"dx{dx}_s896_i140"
    files = slab_runs(tag)
    ds = [load_press(f) for f in files]
    assert all(d["substeps"] == 896 and d["iterations"] == 140 for d, _ in ds)
    npz = np.load(SLAB[dx], allow_pickle=True)
    v15 = [r[15.0]["vein_pct"] for _, r in ds]; v25 = [r[25.4]["vein_pct"] for _, r in ds]
    ninv = [r[25.4]["n_inverted"] for _, r in ds]; minj = [r[25.4]["minJ"] for _, r in ds]
    fl = floor_pct(dx); head = [v - fl for v in v25]
    wall = slab_minutes(tag, files)
    n_mesh = max(n_mesh, len(ds))
    cell(f"mesh.{dx}.v15", [round(x, 2) for x in v15], files); cell(f"mesh.{dx}.v25", [round(x, 2) for x in v25], files)
    cell(f"mesh.{dx}.floor", round(fl, 1), SLAB[dx]); cell(f"mesh.{dx}.ninv", ninv, files); cell(f"mesh.{dx}.minJ", [round(x, 2) for x in minj], files)
    cell(f"mesh.{dx}.wall_min", [round(w, 1) for w in wall], "queue.log")
    cell(f"mesh.{dx}.head", [round(x, 2) for x in head], "vein at 25.4 mm minus the floor, per run")
    er.append(f"{DX_MM[dx]:g}~mm & {len(npz['tets']):,} & {fmt(v25, 1, '\\%', 3)} & {fl:.1f}\\% & {fmt(head, 1, '', 3)} & "
              f"{fmt(ninv, 0, '', 3)} & {fmt(minj, 2, '', 3)} & {fmt(wall, 1, '~min', 3)} \\\\")
emit("mesh_results", f"""\\begin{{table*}}[t]
\\centering
\\caption{{Element-size study on the slab, cost and validity: 0.0186~ms time step, 140 iterations per step, no blood
pressure inside the vessels. Vein: settled lumen area at 25.4~mm, \\% of rest. Floor: the least area each mesh can
represent (half the resting lumen perimeter times the gap at which opposite walls stop, $0.8\\times$ element size, over
the resting area; nominal, untested). Headroom: vein minus floor, in points; near zero, the reading reflects the grid more
than the tissue. Inverted: elements turned inside out at 25.4~mm; worst ratio: the smallest element volume relative to
rest (below 0 = inverted). Wall clock: one seven-depth press, alone on the GPU. Red: fewer than the three planned runs; n shown.}}
\\label{{tab:mesh_results}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{6pt}}
\\begin{{tabular}}{{cccccccc}}
\\hline
\\textbf{{Element}} & \\textbf{{Tets}} & \\textbf{{Vein}} & \\textbf{{Floor}} & \\textbf{{Headroom}} &
\\textbf{{Inverted}} & \\textbf{{Worst ratio}} & \\textbf{{Wall clock}} \\\\
\\hline
{chr(10).join(er)}
\\hline
\\end{{tabular}}
\\end{{table*}}
""")

# ── slab time step (reference = the 18.60 us element-size run at 1.5 mm) ──────────────────────
ref_rows = [load_press(f)[1] for f in ref_files]
ref_mean = {x: np.mean([r[x]["vein_pct"] for r in ref_rows]) for x in DEPTHS} if ref_rows else None
# stability: time step against the contact spring's period. Period T = 2 pi sqrt(rho dx^3 / k_e), rho = 1000 kg/m^3,
# dx = 1.5 mm (the record's definition). k_e from the reference run when it exists, else from any V5 time-step run.
ke_s = ke_slab
if ke_s is None:
    for s in (2, 3, 8, 17, 67, 168):
        f_ = slab_runs(f"ts_s{s}_i140")
        if f_:
            ke_s = float(json.load(open(f_[0]))["soft_contact_ke"]); break
T_us = 2 * np.pi * np.sqrt(1000.0 * (1.5e-3) ** 3 / ke_s) * 1e6 if ke_s else None
if T_us:
    cell("ts.T_us", round(T_us, 1), "record's definition, rho 1000, dx 1.5 mm, ke from a V5 slab run")
tr = []
for s in (896, 168, 67, 17, 8, 3, 2):
    tag = "dx1p5_s896_i140" if s == 896 else f"ts_s{s}_i140"
    files = ref_files if s == 896 else slab_runs(tag)
    ds = [load_press(f) for f in files]
    assert all(d["substeps"] == s and d["iterations"] == 140 for d, _ in ds), s
    dt = FRAME_MS / s
    lab = f"{dt:.4f}~ms" if dt < 0.05 else (f"{dt:.3f}~ms" if dt < 1 else f"{dt:.2f}~ms")
    ratio = dt * 1e3 / T_us if T_us else None
    if ratio is not None:
        cell(f"ts.{s}.ratio", round(ratio, 2), "time step / T")
    if not ds:
        tr.append(f"{lab} & {s} & {f'{ratio:.2f}' if ratio is not None else '--'} & {AWAIT} & {AWAIT} & "
                  f"{'ref.' if s == 896 else AWAIT} & {AWAIT} \\\\")
        continue
    gaps = [max(abs(r[x]["vein_pct"] - ref_mean[x]) for x in DEPTHS) for _, r in ds] if ref_mean else []
    unsettled = max(sum(r[x]["settled_at"] is None for x in DEPTHS) for _, r in ds)   # depths that ran to the frame cap
    settled = unsettled == 0 and all(r[x]["veto"] == "none" for _, r in ds for x in DEPTHS)
    most = [max(r[x]["settled_at"] for x in DEPTHS if r[x]["settled_at"] is not None) for _, r in ds]
    cell(f"ts.{s}.gap", [round(x, 2) for x in gaps], files)
    cell(f"ts.{s}.most", most, files)
    wall = slab_minutes(tag, files); cell(f"ts.{s}.wall_min", [round(x, 1) for x in wall], "queue.log")
    tr.append(f"{lab} & {s} & {f'{ratio:.2f}' if ratio is not None else '--'} & "
              f"{'yes' if settled else f'\\textbf{{no}} ({unsettled} of 7)'} & {fmt(most, 0, '', 3)} & "
              f"{'ref.' if s == 896 else fmt(gaps, 2, '', 3)} & {fmt(wall, 1, '~min', 3)} \\\\")
emit("timestep_slab", f"""\\begin{{table*}}[t]
\\centering
\\caption{{Time-step study on the slab, stability: 1.5~mm elements, 140 iterations per step, no CUDA Graphs, no blood
pressure inside the vessels. The probe contact acts as a stiff spring on each surface node, with period
$T = 2\\pi\\sqrt{{\\rho\\,\\Delta x^3/k_e}} = {f'{T_us / 1e3:.3f}' if T_us else '--'}$~ms ($\\rho = 1000$~kg/m$^3$, $\\Delta x$ = 1.5~mm,
$k_e$ = {f'{ke_s:,.0f}' if ke_s else '--'}~N/m). Settled: every depth met the rule of 30 frames within 2\\%; no ($k$ of 7): $k$ depths ran to the frame cap without
settling. Most frames: the largest number of frames any settled depth needed. Largest gap: largest difference from the 0.0186~ms reference over the seven depths, in
points. Wall clock: one seven-depth press, alone on the GPU. Red: fewer than the three planned runs; n shown.}}
\\label{{tab:timestep_results}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{6pt}}
\\begin{{tabular}}{{ccccccc}}
\\hline
\\textbf{{Time step}} & \\textbf{{Per frame}} & \\textbf{{Step / $T$}} & \\textbf{{Settled}} & \\textbf{{Most frames}} &
\\textbf{{Largest gap}} & \\textbf{{Wall clock}} \\\\
\\hline
{chr(10).join(tr)}
\\hline
\\end{{tabular}}
\\end{{table*}}
""")

# slab iterations at 2.08 ms (no table; quoted in the text, Section "Time Step and Solver Convergence"). Added 2026-10-30
# so the four prose gaps are cells that verify_tables.py re-derives. Gap: largest |vein - control mean| over the seven
# depths, per run; control = the three 2.08 ms x 140 runs (ts_s8_i140), the same definition as the time-step table.
_ctl = [load_press(f)[1] for f in slab_runs("ts_s8_i140")]
_ctlm = {x: np.mean([r[x]["vein_pct"] for r in _ctl]) for x in DEPTHS} if _ctl else None
for it in (70, 35, 20, 10):
    files = slab_runs(f"ts_s8_i{it}")
    ds = [load_press(f) for f in files]
    assert all(d["substeps"] == 8 and d["iterations"] == it for d, _ in ds), it
    gaps = [max(abs(r[x]["vein_pct"] - _ctlm[x]) for x in DEPTHS) for _, r in ds] if _ctlm else []
    cell(f"it.{it}.gap", [round(x, 3) for x in gaps], files)
    print(f"iterations {it}: gap to the 140-iteration control {fmt(gaps, 2, '', 3)}")

# ── leg repeats (data/meshF_repeats/per_run.csv, written by analyze_meshF_repeats.py from the V5 queue) ──
per_run = list(csv.DictReader(open(ROOT / "data" / "meshF_repeats" / "per_run.csv")))
G = defaultdict(list)
for r in per_run:
    G[r["run"]].append(r)
# 🚨 A CSV is only as current as the last analysis run. Refuse one whose runs are not exactly the queue's usable runs:
# on 2026-10-21 a crashed analysis left V4 numbers in data/sensitivity/ and this script printed them.
_want = {t: len(PM.usable("leg", t)) for (_, t) in PM.queue_status("leg")}
_have = {t: len(v) for t, v in G.items()}
if _have != {t: n for t, n in _want.items() if n}:
    raise SystemExit(f"!!! data/meshF_repeats/per_run.csv does not match the V5 queue's usable leg runs; rerun "
                     f"scripts/analyze_meshF_repeats.py.\n    csv {_have}\n    queue {_want}")
REF = P + "s168_i140_ref"
# leg: cost model (from analyze_meshF_repeats.py) and the CUDA graph gain at equal settings
cmp_ = ROOT / "data" / "meshF_repeats" / "cost_model.csv"
cm = {r["term"]: r for r in csv.DictReader(open(cmp_))} if cmp_.exists() else None
if cm:
    for t in ("a", "b", "c", "trend_a", "trend_c"):
        cell(f"cost.{t}", float(cm[t]["value_ms"]), "data/meshF_repeats/cost_model.csv")


def ms_of(tag, graph):
    return [float(r["median_wall_clock_per_frame_ms"]) for r in G.get(tag, []) if r["cuda_graph"] == str(graph)]


gain_rows = []
for s_, it, on_tag, off_tag in ((2, 35, P + "s2_i35_graphon", P + "s2_i35_graphoff"), (8, 35, P + "s8_i35_graphon", P + "s8_i35_graphoff"),
                                (168, 140, REF, REF + "_graphoff")):
    on, off = ms_of(on_tag, True), ms_of(off_tag, False)
    cell(f"graph.{s_}x{it}.on", [round(x, 2) for x in on], "data/meshF_repeats/per_run.csv")
    cell(f"graph.{s_}x{it}.off", [round(x, 2) for x in off], "data/meshF_repeats/per_run.csv")
    dtl = FRAME_MS / s_; dts = f"{dtl:.3f}" if dtl < 1 else f"{dtl:.2f}"
    gains = [np.mean(off) / x for x in on] if (on and off) else []
    cell(f"graph.{s_}x{it}.gain", [round(g, 4) for g in gains], "graph off mean / each graph-on run")
    gain_rows.append(f"{dts}~ms, {it} & {fmt(off, 1)} & {fmt(on, 1)} & {fmt(gains, 2, '$\\times$')} \\\\")
if cm:
    cost_head = (f"\\multicolumn{{4}}{{@{{}}p{{\\columnwidth}}@{{}}}}{{\\textbf{{Cost per simulated frame}} ({cm['n_runs']['value_ms']} runs; "
                 f"residuals: mean {cm['mean_abs_residual_pct']['value_ms']}\\%, largest {cm['max_abs_residual_pct']['value_ms']}\\%)}} \\\\\n"
                 f"\\quad per solver pass & \\multicolumn{{3}}{{l}}{{{float(cm['a']['value_ms']):.3f} $\\pm$ {float(cm['a']['sd_ms']):.3f}~ms}} \\\\\n"
                 f"\\quad per time step & \\multicolumn{{3}}{{l}}{{{float(cm['b']['value_ms']):.2f} $\\pm$ {float(cm['b']['sd_ms']):.2f}~ms}} \\\\\n"
                 f"\\quad per frame, fixed & \\multicolumn{{3}}{{l}}{{{float(cm['c']['value_ms']):.2f} $\\pm$ {float(cm['c']['sd_ms']):.2f}~ms}} \\\\")
else:
    cost_head = f"\\multicolumn{{4}}{{@{{}}p{{\\columnwidth}}@{{}}}}{{\\textbf{{Cost per simulated frame}}: {AWAIT}}} \\\\"
emit("cost_and_graph", f"""\\begin{{table}}[t]
\\centering
\\caption{{Where the wall clock goes on the leg model, one RTX 5090, solver only. Top: the cost model fitted to every
repeated run, minimizing relative error; $\\pm$ is one standard deviation by bootstrap over runs. The per-time-step cost is
why two settings with equal solver work differ in speed. Bottom: median wall clock per frame with CUDA Graphs off and on
(the GPU work recorded once and replayed as one launch), at equal settings; gain = off / on. Red: fewer than the five
planned runs, or awaiting runs; n shown.}}
\\label{{tab:cost}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular*}}{{\\columnwidth}}{{@{{\\extracolsep{{\\fill}}}}lccc@{{}}}}
\\hline
{cost_head}
\\hline
\\textbf{{Time step, iterations}} & \\textbf{{Graph off (ms)}} & \\textbf{{Graph on (ms)}} & \\textbf{{Gain}} \\\\
\\hline
{chr(10).join(gain_rows)}
\\hline
\\end{{tabular*}}
\\end{{table}}
""")

# speed and accuracy per leg setting (replaces the four speed-up figures, 2026-10-01). Graph on; paired runs left out.
# every leg setting, also the appendix rows
APP = SPD = [(168, 140), (67, 140), (17, 140), (8, 140), (3, 140), (2, 140), (8, 70), (8, 35), (8, 20), (8, 10), (3, 35), (2, 35)]
refrs = [r for r in G.get(REF, []) if r["cuda_graph"] == "True"]
refv = {k: {x: np.mean([float(r[f"{k}_pct_{x:g}mm"]) for r in refrs]) for x in DEPTHS} for k in ("vein", "artery")} if refrs else None
ref_press = np.mean([float(r["whole_press_wall_clock_s"]) for r in refrs]) if refrs else None
# per frame, as Fig. 8 plots it (2026-10-27): the initial setting's mean ms/frame over each run's
ref_ms = np.mean([float(r["median_wall_clock_per_frame_ms"]) for r in refrs]) if refrs else None
rows_sp = []
for s_, it in SPD:
    tag = REF if (s_, it) == (168, 140) else f"{P}s{s_}_i{it}"
    rs = [r for r in G.get(tag, []) if r["cuda_graph"] == "True"]
    ms = [float(r["median_wall_clock_per_frame_ms"]) for r in rs]
    sp = [ref_press / float(r["whole_press_wall_clock_s"]) for r in rs] if ref_press else []
    spf = [ref_ms / m for m in ms] if ref_ms else []
    gv = [max(abs(float(r[f"vein_pct_{x:g}mm"]) - refv["vein"][x]) for x in DEPTHS) for r in rs] if refv else []
    ga = [max(abs(float(r[f"artery_pct_{x:g}mm"]) - refv["artery"][x]) for x in DEPTHS) for r in rs] if refv else []
    for k_, v_ in (("ms", ms), ("speedupf", spf), ("speedup", sp), ("gapv", gv), ("gapa", ga)):
        cell(f"spd.{tag}.{k_}", [round(y, 4) for y in v_], "data/meshF_repeats/per_run.csv")
    dtl = FRAME_MS / s_; dts = f"{dtl:.3f}" if dtl < 1 else f"{dtl:.2f}"
    over = lambda v: (f"\\textbf{{{fmt(v, 2)}}}" if v and round(float(np.mean(v)), 2) > 3.0 else fmt(v, 2))   # as printed
    name = f"{dts}~ms, {it}" + (" (reference)" if tag == REF else "") + (" (chosen)" if (s_, it) == (2, 35) else "")
    msf = fmt(ms, 0) if (ms and np.mean(ms) >= 100) else fmt(ms, 1)
    _spf = (fmt(spf, 1) if np.mean(spf) < 100 else fmt(spf, 0)) if spf else AWAIT
    rows_sp.append(f"{name} & {s_ * it:,} & {msf} & {(_spf if tag != REF else '1') if refrs else AWAIT} & {(fmt(sp, 1) if tag != REF else '1') if refrs else AWAIT} & {over(gv)} & {over(ga)} \\\\")
emit("speed_accuracy", f"""\\begin{{table*}}[t]
\\centering
\\caption{{Leg model: speed and accuracy of every solver setting, CUDA Graphs on, solver only. Every filled point of
Fig.~\\ref{{fig:speedup}} is a row here, labelled or not; the hollow points are in Table~\\ref{{tab:cost}}. Solver work
(the figure's x axis): time steps per frame $\\times$ iterations per step. Speed-up per frame: the reference setting's
ms per frame over this setting's, as the figure plots it. Speed-up per press: wall clock to settle all seven depths, against
the reference setting. Faster settings need more frames to settle, so the per-press gain is smaller. Gap: the largest
difference from the reference setting's mean over the seven depths, in percentage points of resting area. Bold: above the 3-point
tolerance set before the runs. Mean $\\pm$ SD over five runs; red: fewer, n shown.}}
\\label{{tab:speed}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{6pt}}
\\begin{{tabular}}{{lcccccc}}
\\hline
\\textbf{{Time step, iterations}} & \\textbf{{Work}} & \\textbf{{ms per frame}} & \\textbf{{Speed-up, frame}} & \\textbf{{Speed-up, press}} & \\textbf{{Vein gap}} & \\textbf{{Artery gap}} \\\\
\\hline
{chr(10).join(rows_sp)}
\\hline
\\end{{tabular}}
\\end{{table*}}
""")

# vein closure at the chosen setting
rt = [r for r in G.get(P + "s2_i35", []) if r["cuda_graph"] == "True"]
vr = []
for x in DEPTHS:
    v = [float(r[f"vein_pct_{x:g}mm"]) for r in rt]; a = [float(r[f"artery_pct_{x:g}mm"]) for r in rt]
    cell(f"vc.{x:g}.vein", [round(y, 2) for y in v], "data/meshF_repeats/per_run.csv")
    cell(f"vc.{x:g}.artery", [round(y, 2) for y in a], "data/meshF_repeats/per_run.csv")
    vr.append(f"{x:.1f}~mm & {fmt(v, 1, '\\%')} & {fmt(a, 1, '\\%')} \\\\")
emit("vein_closure", f"""\\begin{{table}}[t]
\\centering
\\caption{{Lumen area on the leg model as a percentage of rest, at the chosen setting: 8.33~ms time step, 35 iterations
per step, solver only, no blood pressure inside the vessels. Depth is the commanded probe travel past the resting skin.
Mean $\\pm$ SD over repeated runs; red: fewer than the five planned runs, n shown.}}
\\label{{tab:vein_closure}}
\\begin{{tabular}}{{ccc}}
\\hline
\\textbf{{Probe depth}} & \\textbf{{Vein}} & \\textbf{{Artery}} \\\\
\\hline
{chr(10).join(vr)}
\\hline
\\end{{tabular}}
\\end{{table}}
""")

# appendix: vein and artery at every depth for every leg setting (the main text shows the chosen setting only)
for body in ("vein", "artery"):
    rows_a = []
    for s_, it in APP:
        tag = REF if (s_, it) == (168, 140) else f"{P}s{s_}_i{it}"
        rs = [r for r in G.get(tag, []) if r["cuda_graph"] == "True"]
        dtl = FRAME_MS / s_; dts = f"{dtl:.3f}" if dtl < 1 else f"{dtl:.2f}"
        cells = []
        for x in DEPTHS:
            v = [float(r[f"{body}_pct_{x:g}mm"]) for r in rs]
            cell(f"app.{body}.{tag}.{x:g}", [round(y, 2) for y in v], "data/meshF_repeats/per_run.csv")
            cells.append(fmt(v, 1))
        name = f"{dts}~ms, {it}" + (" (reference)" if tag == REF else "") + (" (chosen)" if (s_, it) == (2, 35) else "")
        rows_a.append(f"{name} & " + " & ".join(cells) + " \\\\")
    emit(f"appendix_{body}", f"""\\begin{{table*}}[t]
\\centering
\\caption{{Leg model, {body} lumen area as a percentage of rest at each commanded probe depth, for every solver setting
tested: time step and iterations per step, CUDA Graphs on, solver only, no blood pressure inside the vessels. Mean
$\\pm$ SD over repeated runs; red: fewer than the five planned runs, n shown.}}
\\label{{tab:appendix_{body}}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{l{"c" * len(DEPTHS)}}}
\\hline
\\textbf{{Time step, iterations}} & {" & ".join(f"\\textbf{{{x:g}~mm}}" for x in DEPTHS)} \\\\
\\hline
{chr(10).join(rows_a)}
\\hline
\\end{{tabular}}
\\end{{table*}}
""")

# appendix: vein-wall and nerve stiffness sensitivity (open item 43; scripts/analyze_sensitivity.py writes the CSV)
SENS = [("control", "20~kPa, as in Table~\\ref{tab:anatomy} (control)"),
        ("vein_uniform_20kpa", "20~kPa, uniform"),
        ("vein_bend_034mm", "66.4~kPa: bending match, 0.34~mm wall"),
        ("vein_bend_050mm", "211.1~kPa: bending match, 0.50~mm wall"),
        ("vein_600kpa", "600~kPa (the artery wall's value)"),
        ("vein_5p7mpa", "5,700~kPa (tensile value, as a solid wall)"),
        ("nerve_20kpa", "nerve 20~kPa (half)"), ("nerve_80kpa", "nerve 80~kPa (double)")]
spc = ROOT / "data" / "sensitivity" / "per_run.csv"
sr = list(csv.DictReader(open(spc))) if spc.exists() else []
_sdig = dict({c: v["sha256_12"] for c, v in json.load(open(PM.SENS / "manifest.json"))["cases"].items()}, control=PM.LEG_SHA)
for r in sr:
    if r["mesh_sha256_12"] != _sdig[r["case"]]:
        raise SystemExit(f"!!! data/sensitivity/per_run.csv row {r['case']} r{r['repeat']} is mesh {r['mesh_sha256_12']}, not "
                         f"{PM.GENERATION}'s {_sdig[r['case']]}: stale; rerun scripts/analyze_sensitivity.py")
for case, _ in SENS:
    if sum(r["case"] == case for r in sr) != len(PM.usable("sens", case)):
        raise SystemExit(f"!!! data/sensitivity/per_run.csv has {sum(r['case'] == case for r in sr)} {case} runs, the queue "
                         f"{len(PM.usable('sens', case))}: rerun scripts/analyze_sensitivity.py")
# Not settled: read from the V5 runs themselves (V4 hard-coded the one case it saw). A cell is marked when ANY repeat of
# that case vetoed that depth, and the mark says how many of the repeats did.
unsettled = defaultdict(int)
for case, _ in SENS:
    for _, p in PM.usable("sens", case):
        for r in json.load(open(p))["depth_rows"]:
            if r["veto"] != "none":
                unsettled[(case, float(r["depth_mm"]))] += 1
blocks = []
for body in ("vein", "artery"):
    rows_s = [f"\\multicolumn{{{len(DEPTHS) + 1}}}{{l}}{{\\textit{{{body.capitalize()} lumen area, \\% of rest}}}} \\\\"]
    for case, name in SENS:
        rs = [r for r in sr if r["case"] == case]
        cells = []
        for x in DEPTHS:
            v = [float(r[f"{body}_pct_{x:g}mm"]) for r in rs]
            cell(f"sens.{case}.{body}.{x:g}", [round(y, 2) for y in v], "data/sensitivity/per_run.csv")
            k = unsettled.get((case, x), 0)
            cells.append(fmt(v, 1) + (f"$^{{\\dagger{k}}}$" if k else ""))
        rows_s.append(f"{name} & " + " & ".join(cells) + " \\\\")
    blocks.append("\n".join(rows_s))
cell("sens.unsettled", {f"{c}@{x:g}": k for (c, x), k in unsettled.items()}, "depth_rows veto != none, V5 sensitivity runs")
emit("appendix_wall_sensitivity", f"""\\begin{{table*}}[t]
\\centering
\\caption{{Leg model: sensitivity of the vessel response to the vein-wall and nerve stiffness. One material is changed per
case; everything else is identical (8.33~ms time step, 35 iterations per step, CUDA Graphs on, solver only). The control
is the paper's model. Bending match: the modulus at which a one-element (1.5~mm) wall bends like a real wall of that
thickness at 5,700~kPa, $E_\\mathrm{{eff}} = 5{{,}}700~\\mathrm{{kPa}} \\times (t/1.5~\\mathrm{{mm}})^3$~\\cite{{lebahn2025,alibazoner2021}}.
Mean $\\pm$ SD over five runs; red: fewer, n shown. $^{{\\dagger k}}$Not settled in $k$ of the repeats: the area was still
changing at the end of the settling allowance, and the value is the last frame. A fully flattened vein would read about
{LEG_FLOOR[0]:.0f}--{LEG_FLOOR[1]:.0f}\\% on this mesh (Section~\\ref{{sec:vascular}}).}}
\\label{{tab:appendix_wall}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{l{"c" * len(DEPTHS)}}}
\\hline
\\textbf{{Vein wall (or nerve) modulus}} & {" & ".join(f"\\textbf{{{x:g}~mm}}" for x in DEPTHS)} \\\\
\\hline
{(chr(10) + "\\hline" + chr(10)).join(blocks)}
\\hline
\\end{{tabular}}
\\end{{table*}}
""")

assert sorted(OUT) == sorted(ORDER), (sorted(OUT), sorted(ORDER))
for name in ORDER:
    (TAB / f"{name}.tex").write_text(OUT[name])
json.dump(VALUES, open(TAB / "values.json", "w"), indent=1, default=str)
print(f"wrote {len(list(TAB.glob('*.tex')))} tables and {len(VALUES)} cells ({PM.GENERATION}); slab element-size runs per row "
      f"(max): {n_mesh}; leg reference runs: {len(refrs)}")
