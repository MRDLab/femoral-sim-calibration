"""Export the femoral phantom's tetrahedral mesh so Isaac Lab can consume OUR mesh, not its own.

    $PY_PHYSICS anatomy_meshing/export_tetmesh.py --anatomy-dir femoral_v4 --station-s 30

WHY. Isaac Lab spawns deformables by tetrahedralising a procedural shape, and that
tetrahedralisation is NON-DETERMINISTIC -- measured 2026-08-20, identical configurations gave
152/538, 153/526, 157/537 and 273/1026 nodes/tets. Two consequences, both fatal to the
2026-08-18 re-run:

  1. Closure is read off NODE POSITIONS and the measurement floor scales with element size
     (0.8*dx), so a mesh that changes between runs changes the answer.
  2. The spurious force in that rig is PER-PARTICLE (7.09 mN per free node), so a calibration
     curve built across differently-sized meshes carries a different offset at every point --
     which is exactly why the 2026-08-20 low-load curve came out non-monotonic.

📌 So the mesh has to be OURS. `femoral_fem.build()` already produces it, with per-element
materials; this writes it to a file that the Isaac Lab side turns into a UsdGeom.TetMesh.

⚠️ THE HANDOVER IS A FILE ON PURPOSE. The physics venv has Newton 1.4.0 but no `pxr`; the Isaac
Lab venv has `pxr` but a different Newton. Geometry is built here and handed over, per the
standing rule that geometry work happens host-side and crosses as files.

🚨 WHAT MUST SURVIVE THE CROSSING, or the re-run is meaningless:
  - node positions, in metres, in the phantom's own frame
  - tet connectivity, index-aligned with the materials
  - **per-tetrahedron materials**. Isaac Lab's deformable asset is single-material; the artery
    wall is ~30x the vein wall and that contrast IS the 2026-08-18 result. Verified 2026-08-20
    that a post-spawn write to `Model.tet_materials` is honoured (split-stiffness test produced
    a 4.13x sag asymmetry against a 1.008 baseline), so these rows are what gets written.
  - the pinned node set, and the per-node body labels needed to find the lumens again.
"""

# ── repo path bootstrap ─────────────────────────────────────────────────────────
# This script moved out of the flat repository root on 2026-08-25 and imports other
# modules in this repository, so the root and the code directories go back on the path.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import repo_paths  # noqa: F401,E402
# ────────────────────────────────────────────────────────────────────────────────

import argparse
import pathlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--anatomy-dir", default="femoral_v4")
ap.add_argument("--station-s", type=float, default=30.0)
ap.add_argument("--dx", type=float, default=0.0015)
ap.add_argument("--substeps", type=int, default=512)
ap.add_argument("--probe-w", type=float, default=0.040)
ap.add_argument("--probe-len", type=float, default=0.012)
ap.add_argument("--ke", type=float, default=2.0e3)
ap.add_argument("--sheath-kpa", type=float, default=20.0)
ap.add_argument("--canal-kpa", type=float, default=None,
                help="Override FemoralCanal's Young's modulus, in kPa. 🎯 THE ONE-VARIABLE TEST "
                     "FOR THE 2026-08-25 CANAL HYPOTHESIS. The canal is 2 kPa loose areolar tissue "
                     "and it is the ONLY body present at s = 6 mm and absent at s = 20 mm (it spans "
                     "s = 0..12.5). Setting it to Perivascular's 3.0 kPa suppresses it as a "
                     "distinct soft inclusion while leaving the GEOMETRY untouched, which is what "
                     "makes this a single variable. ⚠️ The contrast being tested is only 2 -> 3 kPa "
                     "over a 2.6 mm-radius disc, so a NULL is a likely and informative outcome: it "
                     "would move the cause to the sheath's funnel cross-section or the ramus, which "
                     "also differ between the two stations.")
ap.add_argument("--perivascular", action="store_true",
                help="Grow a soft perivascular collar around each vessel wall. 🎯 THE LEADING "
                     "SUSPECT for the vein's closure-vs-depth SHAPE: with this off the sheath bore "
                     "is meshed SOLID at --sheath-kpa, so the vessels sit in bulk 20 kPa tissue and "
                     "the vein stops responding as the press deepens. V3/V4's Sheath body does NOT "
                     "enclose the vessels on the side the probe presses from, so softening the "
                     "envelope alone tests almost nothing -- the collar is what puts the compliance "
                     "where the sheath actually is, blending with the vessel adventitia.")
ap.add_argument("--out", default=None)
args = ap.parse_args()

import femoral_fem as F  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent  # repo root: this script now lives one level down
sec = (ROOT / "out" / args.anatomy_dir / f"section_s{int(round(args.station_s)):03d}.json").resolve()   # real out/, not a worktree symlink
if not sec.exists():
    raise SystemExit(f"!!! no section at {sec}")
print(f">>> section {sec}", flush=True)
# 🚨 Sheath has no sourced modulus and materials.py leaves it None. `None` means TWO different
# things in that file -- "lumen, treat as void" for blood, "nobody sourced this" for the sheath --
# and load_section refuses rather than let the second be silently read as the first, which would
# put a cavity where tissue should be. Set it explicitly, exactly as diag_press_target.py:203 does.
F.BODIES["Sheath"]["E_pa"] = args.sheath_kpa * 1e3
F.BODIES["Sheath"]["nu"] = F.NU
if args.canal_kpa is not None:
    _was = F.BODIES["FemoralCanal"]["E_pa"]
    F.BODIES["FemoralCanal"]["E_pa"] = args.canal_kpa * 1e3
    print(f">>> 🎯 CANAL OVERRIDE: FemoralCanal E {_was} -> {args.canal_kpa * 1e3} Pa "
          f"(geometry unchanged; this is the single-variable canal test)", flush=True)

F.load_section(str(sec), perivascular=args.perivascular)

ph = F.Phantom(args.dx, args.substeps, args.probe_w, args.probe_len, soft_contact_ke=args.ke)
m = ph.model

rest = np.asarray(ph.rest, dtype=np.float64)
tets = np.asarray(ph.tet_i, dtype=np.int64)
tmat = m.tet_materials.numpy().astype(np.float64)          # (n_tets, 3) = k_mu, k_lambda, k_damp
pmass = m.particle_mass.numpy().astype(np.float64)
pinv = m.particle_inv_mass.numpy().astype(np.float64)
pinned = np.flatnonzero(pinv <= 0.0).astype(np.int64)

print(f">>> nodes={len(rest):,}  tets={len(tets):,}  pinned={len(pinned):,}", flush=True)
print(f">>> unique material rows = {np.unique(tmat[:, :2], axis=0).shape[0]}", flush=True)
if len(tmat) != len(tets):
    raise SystemExit(f"!!! materials ({len(tmat)}) and tets ({len(tets)}) disagree -- "
                     f"every per-tet array downstream would be off by index")

# per-node body label, so the lumens can be found again on the far side
body_of = np.asarray(ph.body_of)
# 🪤 `body_of` is NOT per-node -- it is a 2D column structure. The first version of this export
# indexed it as if it were per-node, so the laterality check below reported "absent" for all three
# vessels and PASSED WITHOUT RUNNING. A check that cannot fire is worse than no check. Label each
# tetrahedron directly instead, by asking classify() what material sits at its centroid -- the same
# function build() used to assign the material in the first place.
cen = rest[tets].mean(axis=1)
tet_body = np.array([F.classify(float(cx), float(cz)) for cx, _, cz in cen], dtype="U32")
_u, _c = np.unique(tet_body, return_counts=True)
print(">>> tets per body: " + "  ".join(f"{a}={b}" for a, b in zip(_u, _c)), flush=True)

# 🎯 LATERALITY — ANCHORED ON A LANDMARK, NOT ON AN ASSUMED AXIS DIRECTION.
# A reflection preserves every diameter, depth and span, so only the ORDER of the bodies catches
# it. 🪤 But the first version of this check asserted "nerve < artery < vein in x" and that is a
# statement about which way +x points, which nothing here had established -- it would have
# condemned a perfectly good right leg. Establish the medial direction from a structure whose
# side is known (the pubic ramus is MEDIAL), then assert the vessel order along that direction.
# Measured 2026-08-20 on femoral_v4 s030: PubicRamus x = -10.90 mm, so NEGATIVE x is medial;
# Vein -11.40 (most medial), Artery -1.30. Right leg, correct.
def _cx(label):
    sel = tet_body == label
    return float(cen[sel, 0].mean()) if sel.any() else None

_ram, _mus = _cx("PubicRamus"), _cx("Muscle")
_anchor = "PubicRamus"
if _ram is None:
    # 2026-10-21, V5: the ramus was removed from the cast phantom. Anchor on the FEMORAL CANAL instead, the medial
    # compartment of the sheath. It ends at s = 12.5 mm, so it is usually absent from the block; read its centroid
    # from the anatomy's own OBJ. The section frame is asserted below to be X = LPS x, so the OBJ's x is the mesh's X.
    _sec_doc = __import__("json").load(open(sec))
    if "X = LPS x" not in _sec_doc["frame"]:
        raise SystemExit(f"!!! section frame {_sec_doc['frame']!r}: cannot carry an OBJ landmark into the mesh frame")
    _canal = ROOT / "out" / args.anatomy_dir / "FemoralCanal.obj"
    if not _canal.exists():
        raise SystemExit("!!! cannot anchor laterality: no PubicRamus in the mesh and no FemoralCanal.obj")
    _xs = [float(l.split()[1]) for l in open(_canal) if l.startswith("v ")]     # physics venv: no trimesh
    _ram = 0.5 * (min(_xs) + max(_xs)) * 1e-3                                    # bbox centre, mm -> m
    _anchor = "FemoralCanal (OBJ)"
if _ram is None or _mus is None:
    raise SystemExit("!!! cannot anchor laterality: medial landmark or Muscle absent")
medial_sign = 1.0 if _ram > _mus else -1.0      # +1 => medial is +x, -1 => medial is -x
print(f">>> frame anchor: {_anchor} x={_ram*1e3:+.2f} mm vs Muscle x={_mus*1e3:+.2f} mm  ->  "
      f"medial is {'+x' if medial_sign > 0 else '-x'}", flush=True)

_art, _vei, _ner = _cx("Artery"), _cx("Vein"), _cx("Nerve")
print(">>> vessels, centroid x (mm): "
      + "  ".join(f"{k}={(v*1e3):+.2f}" if v is not None else f"{k}=ABSENT"
                  for k, v in (("Nerve", _ner), ("Artery", _art), ("Vein", _vei))), flush=True)

# medial-ness = x * medial_sign, increasing toward medial. Order lateral -> medial must be
# nerve, artery, vein (the vein is the most medial of the three).
if _art is None or _vei is None:
    raise SystemExit("!!! LATERALITY: artery or vein missing -- cannot verify the leg side")
if not (_art * medial_sign < _vei * medial_sign):
    raise SystemExit(
        f"!!! LATERALITY: the VEIN must be MEDIAL to the artery on a right leg.\n"
        f"    medial is {'+x' if medial_sign > 0 else '-x'}; artery x={_art*1e3:+.2f} mm, "
        f"vein x={_vei*1e3:+.2f} mm.\n"
        "    This is the signature of a mesh mirrored along x -- a LEFT leg. Every diameter,\n"
        "    depth and span would still be correct, so nothing else here would catch it.")
if _ner is not None and not (_ner * medial_sign < _art * medial_sign):
    raise SystemExit("!!! LATERALITY: nerve must be LATERAL to the artery on a right leg")
print(">>> ✅ laterality: lateral -> medial reads "
      + ("nerve -> " if _ner is not None else "(nerve absent) -> ") + "artery -> vein — RIGHT leg", flush=True)
if _ner is None:
    print(">>> ⚠️ Nerve has ZERO tetrahedra in the FEM mesh. Nerve.obj exists for the ACOUSTIC "
          "model only, so the nerve appears in B-mode but has no mechanical presence. The "
          "lateral-most member of the ordering is therefore UNVERIFIED here.", flush=True)

# 🎯 RELATIONSHIP CHECKS, not just value checks. A wrong export passes every scalar test.
assert tets.min() >= 0 and tets.max() < len(rest), "tet indices out of range"
p = rest[tets]
vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0
print(f">>> rest volumes: min={vol.min():.6e}  max={vol.max():.6e}  n_negative={int((vol <= 0).sum())}", flush=True)
if (vol <= 0).any():
    raise SystemExit("!!! the exported mesh already contains inverted elements at REST")
print(f">>> total volume={vol.sum():.6e} m^3   total mass={pmass.sum():.6f} kg", flush=True)
print(f">>> bbox x[{rest[:,0].min()*1e3:+.2f},{rest[:,0].max()*1e3:+.2f}] "
      f"y[{rest[:,1].min()*1e3:+.2f},{rest[:,1].max()*1e3:+.2f}] "
      f"z[{rest[:,2].min()*1e3:+.2f},{rest[:,2].max()*1e3:+.2f}] mm", flush=True)

# 🎯 THE LUMEN RINGS AND THE PROBE FOOTPRINT MUST CROSS TOO.
# Closure is a polygon area through each lumen's ring of wall nodes (Phantom._lumen_rings), and
# it is read off NODE POSITIONS -- so it needs no force channel and survives the crossing intact.
# Export the ring node ids and the rest areas so the far side computes the SAME quantity rather
# than re-deriving rings from geometry, which is how two halves drift apart.
rings = {k: np.asarray(v, dtype=np.int64) for k, v in ph.rings.items()}
rest_areas = ph.areas(rest)
print(">>> lumen rings: " + "  ".join(
    f"{k}: {len(v)} nodes, rest area {rest_areas[k]*1e6:.2f} mm^2" for k, v in rings.items()), flush=True)

# 🎯 THE FULL LUMEN SURFACES, for lumen pressure.
# `rings` is one mid-length ring per lumen and is enough to MEASURE closure. Applying lumen
# pressure needs the whole wall: Phantom._refresh_lumen_pressure walks `surf[name]`, shaped
# (ny+1, nbin), and puts an outward, length-weighted follower force on every wall node. Without
# this the artery has nothing holding it open but its 600 kPa wall -- which is precisely the
# 2026-08-08 finding that pressurising the lumens is what makes the vein yield ~2.5x the artery.
surfs = {k: np.asarray(v, dtype=np.int64) for k, v in ph.surf.items()}
for k, v in surfs.items():
    print(f">>> lumen surface {k}: shape {v.shape} ({v.size} node refs)", flush=True)

# probe footprint: skin nodes the punch would press on, from the same geometry build() used
mmeta = ph.meta
x_probe = float(mmeta["x_probe"]); z_skin = float(mmeta["z_skin"])
half_w = args.probe_w / 2.0
half_l = args.probe_len / 2.0
top_of_col = {}
for nid, (X, Y, Z) in enumerate(rest):
    if abs(X - x_probe) <= half_w and abs(Y) <= half_l:
        key = (round(X, 9), round(Y, 9))
        if key not in top_of_col or Z > rest[top_of_col[key], 2]:
            top_of_col[key] = nid
footprint = np.array(sorted(top_of_col.values()), dtype=np.int64)
print(f">>> probe footprint: {len(footprint)} surface nodes over "
      f"x in [{(x_probe-half_w)*1e3:+.1f},{(x_probe+half_w)*1e3:+.1f}] mm, "
      f"y in [{-half_l*1e3:+.1f},{half_l*1e3:+.1f}] mm; skin z={z_skin*1e3:.2f} mm", flush=True)
if len(footprint) == 0:
    raise SystemExit("!!! the probe footprint selected ZERO nodes -- the press would do nothing")

out = pathlib.Path(args.out) if args.out else (ROOT / "out" / f"tetmesh_{args.anatomy_dir}_s{int(round(args.station_s)):03d}{'_periv' if args.perivascular else ''}.npz")
np.savez_compressed(
    out, rest=rest, tets=tets, tet_materials=tmat, particle_mass=pmass, pinned=pinned,
    body_of=body_of.astype("U32"), tet_body=tet_body,
    footprint=footprint,
    ring_names=np.array(sorted(rings.keys()), dtype="U16"),
    surf_names=np.array(sorted(surfs.keys()), dtype="U16"),
    **{f"surf_{k}": v for k, v in surfs.items()},
    ny=np.array([int(ph.meta["ny"])]),
    p_scale=np.array([float(getattr(ph, "_p_scale", 1.0))]),
    **{f"ring_{k}": v for k, v in rings.items()},
    **{f"rest_area_{k}": np.array([rest_areas[k]]) for k in rings},
    x_probe=np.array([x_probe]), z_skin=np.array([z_skin]),
    probe_w=np.array([args.probe_w]), probe_len=np.array([args.probe_len]), dx=np.array([args.dx]),
    meta=np.array([repr(ph.meta)], dtype="U100000"),
    anatomy_dir=np.array([args.anatomy_dir], dtype="U64"),
    station_s=np.array([args.station_s]),
)
print(f">>> wrote {out}  ({out.stat().st_size/1e6:.2f} MB)", flush=True)
print("EXPORT DONE", flush=True)
