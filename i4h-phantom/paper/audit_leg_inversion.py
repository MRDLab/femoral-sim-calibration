"""Element inversion on the leg mesh (Mesh F) at the chosen setting, every settled depth. 2026-10-09. Closes item 28's leg gap.

    $PY_ANATOMY scripts/audit_leg_inversion.py

Source: i4h-phantom/out/paper_repeats_inversion/r{1,2,3}/legF_s2_i35_pos.npz, written by isaaclab/live_press_bmode.py
--save-positions (8.33 ms time step, 35 iterations, CUDA Graphs on, solver only; the same flags as the paper's repeats).
Positions are saved in the mesh file's node order; the script checks the saved rest and tets equal Mesh F's.
Per element: volume ratio J = V / V_rest; J < 0 is inverted. Same definitions as audit_slab_inversion.py:
  vessel wall           tets labelled Vein or Artery (the leg mesh has no blended "void" cells)
  between the vessels   centroid within 3 mm of both lumens (16-node rings, rest positions, in the cross-section)
  depth below the skin  skin radius minus the centroid's distance from the limb axis
Writes data/leg_inversion/inversion_by_depth.csv.
"""
import csv
import glob
import hashlib
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
DATA = ROOT / "data" / "leg_inversion"; DATA.mkdir(parents=True, exist_ok=True)
import sys  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_meshes as PM  # noqa: E402  2026-10-21: V5 (V4 read out/paper_repeats_inversion/ on Mesh F)
MESH = PM.verified(PM.LEG, PM.LEG_SHA, "leg mesh")
M = np.load(MESH, allow_pickle=True)
body = M["tet_body"].astype(str); T = M["tets"].astype(np.int64); R0 = M["rest"].astype(np.float64)
r_skin = json.loads(str(M["meta"][0]))["r_skin_m"]

def vol(X):
    a, b, c, d = (X[T[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0

V0 = vol(R0); C0 = R0[T].mean(1)
assert (V0 > 0).all(), "inverted at rest"
rc = {k: R0[M[f"ring_{k}"]].mean(0) for k in ("vein", "artery")}
rr = {k: np.linalg.norm(R0[M[f"ring_{k}"]][:, [0, 2]] - rc[k][[0, 2]], axis=1).mean() for k in rc}
dwall = {k: (np.linalg.norm(C0[:, [0, 2]] - rc[k][[0, 2]], axis=1) - rr[k]) * 1e3 for k in rc}
between = (dwall["vein"] < 3.0) & (dwall["artery"] < 3.0)
wall = np.isin(body, ["Vein", "Artery"])
depth_below = (r_skin - np.hypot(C0[:, 0], C0[:, 2])) * 1e3

rows = []
for p in sorted(glob.glob(str(PM.RUNS / "leg_inversion" / "r*" / f"{PM.RUN_PREFIX}s2_i35_pos.npz"))):
    P = np.load(p); rep = int(pathlib.Path(p).parent.name[1:])
    assert str(P["mesh_sha256_12"]) == PM.LEG_SHA and int(P["substeps"]) == 2 and int(P["iterations"]) == 35
    assert np.array_equal(P["tets"], T) and np.abs(P["rest"] - R0).max() < 1e-9, "saved mesh is not the paper's leg mesh"
    for key in sorted((k for k in P.files if k.startswith("pos_")), key=lambda k: float(k[4:-2].replace("p", "."))):
        dep = float(key[4:-2].replace("p", ".")); J = vol(P[key]) / V0; inv = J < 0
        rows.append(dict(repeat=rep, depth_mm=dep, n_inverted=int(inv.sum()), pct_of_elements=round(100 * inv.mean(), 4),
                         min_volume_ratio=round(float(J.min()), 3),
                         max_depth_below_skin_mm=round(float(depth_below[inv].max()), 1) if inv.any() else "",
                         nearest_to_vessel_wall_mm=round(float(min(dwall["vein"][inv].min(), dwall["artery"][inv].min())), 1) if inv.any() else "",
                         n_between_artery_and_vein=int((inv & between).sum()), n_vessel_wall=int((inv & wall).sum()),
                         bodies=";".join(f"{b}:{int(((body == b) & inv).sum())}" for b in sorted(set(body[inv])))))
if not rows:   # 2026-10-21: rewrite, never leave the previous generation's file in place
    (DATA / "inversion_by_depth.csv").write_text(f"# no usable {PM.GENERATION} runs yet\n")
    raise SystemExit(f"!!! no usable {PM.GENERATION} position files yet: inversion_by_depth.csv rewritten empty")
with open(DATA / "inversion_by_depth.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
