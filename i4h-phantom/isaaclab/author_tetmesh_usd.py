"""Turn the exported femoral tet mesh into a UsdGeom.TetMesh that Isaac Lab can spawn.

    env/bin/python author_tetmesh_usd.py --npz i4h-phantom/out/tetmesh_femoral_v4_s030.npz

WHY A FILE AND NOT A PROCEDURAL SHAPE. Isaac Lab's own tetrahedralisation is non-deterministic
(measured 2026-08-20: identical configs gave 152/538, 153/526, 157/537, 273/1026 nodes/tets), and
both things we need depend on the mesh being fixed -- closure is read off node positions with a
floor that scales as 0.8*dx, and the rig's spurious force is per-particle. Authoring the mesh
ourselves removes that variable entirely.

⚠️ No pxr in the physics venv and a different Newton here, so the mesh crosses as a .npz and is
turned into USD on this side. Nothing is re-meshed in the process: points and tetVertexIndices are
written through unchanged, and the reload check below asserts that.
"""

import argparse
import pathlib
import numpy as np
from pxr import Usd, UsdGeom, Gf, Sdf, Vt

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True, help="Tet mesh to press. REQUIRED -- no default. Until 2026-08-25 this defaulted to tetmesh_femoral_v4_s030.npz while the work ran at s006/s020, so the default silently named a mesh no run used and the output could not be told apart from it (open items 18, 37). A wrong mesh must be a refusal to start, not a substitution.")
ap.add_argument("--out", default=None)
args = ap.parse_args()

d = np.load(args.npz, allow_pickle=True)
rest = d["rest"].astype(np.float64)
tets = d["tets"].astype(np.int64)
tmat = d["tet_materials"].astype(np.float64)
tet_body = d["tet_body"]
out = pathlib.Path(args.out) if args.out else pathlib.Path(args.npz).with_suffix(".usda")

print(f">>> nodes={len(rest):,}  tets={len(tets):,}  unique materials={np.unique(tmat[:,:2],axis=0).shape[0]}", flush=True)

if out.exists():
    out.unlink()
stage = Usd.Stage.CreateNew(str(out))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)          # our mesh is in METRES
root = UsdGeom.Xform.Define(stage, "/Phantom")
stage.SetDefaultPrim(root.GetPrim())

tm = UsdGeom.TetMesh.Define(stage, "/Phantom/sim_mesh")
# ⚠️ numpy scalars are rejected by Gf.Vec3f; .tolist() yields native Python floats.
tm.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in rest.astype(np.float32).tolist()]))
tm.CreateTetVertexIndicesAttr(Vt.Vec4iArray([Gf.Vec4i(*t) for t in tets.astype(np.int32).tolist()]))

# Per-tet body label and material rows ride along as custom primvars so the far side never has to
# recompute them -- recomputing is how the two halves drift apart.
pv = UsdGeom.PrimvarsAPI(tm.GetPrim())
pv.CreatePrimvar("tetBody", Sdf.ValueTypeNames.StringArray,
                 UsdGeom.Tokens.uniform).Set(Vt.StringArray([str(b) for b in tet_body.tolist()]))
pv.CreatePrimvar("tetKMu", Sdf.ValueTypeNames.FloatArray,
                 UsdGeom.Tokens.uniform).Set(Vt.FloatArray(tmat[:, 0].astype(np.float32).tolist()))
pv.CreatePrimvar("tetKLambda", Sdf.ValueTypeNames.FloatArray,
                 UsdGeom.Tokens.uniform).Set(Vt.FloatArray(tmat[:, 1].astype(np.float32).tolist()))
# 🎯 A VISUAL SURFACE, or there is nothing to render.
# The contrib loader takes UsdGeom.TetMesh as the SIMULATION mesh and a UsdGeom.Mesh as the
# VISUAL one. A TetMesh alone simulates fine and draws nothing, so the Isaac Sim viewport would
# show an empty stage and the deformation would be invisible -- which is most of the point of
# bringing Isaac Lab in. Build the boundary surface: a triangle belongs to the surface exactly
# when it is shared by ONE tetrahedron.
from collections import Counter  # noqa: E402
faces = []
for a, b, c, dd in tets.tolist():
    faces += [(a, c, b), (a, b, dd), (b, c, dd), (a, dd, c)]
cnt = Counter(tuple(sorted(f)) for f in faces)
surf = [f for f in faces if cnt[tuple(sorted(f))] == 1]
print(f">>> surface: {len(surf):,} boundary triangles from {len(tets):,} tets", flush=True)
vis = UsdGeom.Mesh.Define(stage, "/Phantom/vis_mesh")
vis.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in rest.astype(np.float32).tolist()]))
vis.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(surf)))
vis.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for f in surf for i in f]))
vis.CreateSubdivisionSchemeAttr("none")
vis.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.82, 0.55, 0.45)]))

stage.Save()
print(f">>> wrote {out}  ({out.stat().st_size/1e6:.2f} MB)", flush=True)

# 🎯 RELOAD AND COMPARE. Writing a file that cannot be read back identically is the whole risk of
# a cross-process handover, and it would show up downstream as a slightly different anatomy with
# every number still plausible.
s2 = Usd.Stage.Open(str(out))
t2 = UsdGeom.TetMesh(s2.GetPrimAtPath("/Phantom/sim_mesh"))
p2 = np.asarray(t2.GetPointsAttr().Get(), dtype=np.float64)
i2 = np.asarray(t2.GetTetVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 4)
dp = float(np.abs(p2 - rest).max())
print(f">>> reload: nodes={len(p2):,}  tets={len(i2):,}  max|dpoints|={dp:.3e} m  "
      f"indices identical={bool((i2 == tets).all())}", flush=True)
assert len(p2) == len(rest) and len(i2) == len(tets), "reload changed the mesh size"
assert (i2 == tets).all(), "reload changed the connectivity"
assert dp < 1e-6, f"reload moved points by {dp:.3e} m (float32 storage; must stay under 1 um)"
print(">>> ✅ round-trip clean", flush=True)
print("AUTHOR DONE", flush=True)
