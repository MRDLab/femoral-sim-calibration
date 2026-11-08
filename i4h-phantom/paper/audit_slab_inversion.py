"""Where the slab's inverted elements are, at every depth of the six time-step runs. 2026-10-01. Answers Table IV.

    $PY_ANATOMY scripts/audit_slab_inversion.py

Source since 2026-10-21: the Femoral V5 queue's first usable repeat of each time-step run, positions saved with
--save-positions as out/paper_v5/slab/r<k>/ts_s<S>_i140_pos.npz (1.5 mm V5 slab, paper_meshes.SLAB_15_SHA). A setting with
no V5 run is skipped and reported, never filled from V4. (V4 read the original runs of 2026-09-03, 6f3c9f86ab61.) The element-size runs of Table IV saved no positions that
survive (their default positions file was rewritten by the 2026-09-30 repeats), so other element sizes are not located.
Per element: volume ratio J = V / V_rest from the saved positions; J < 0 is inverted. Independent check: the count must
match the run's own logged n_inverted to within one element (float32 positions), or the script stops.
"Vessel wall": tets labelled Vein or Artery, plus blended boundary cells ("void") within 2 mm of either lumen.
Distances use rest positions, in the cross-section, from the 16-node lumen ring's mean circle.
Writes data/slab_inversion/inversion_by_depth.csv.
"""
import csv
import hashlib
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
DATA = ROOT / "data" / "slab_inversion"; DATA.mkdir(parents=True, exist_ok=True)
import sys  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_meshes as PM  # noqa: E402  2026-10-21: V5, one place
MESH = PM.verified(PM.SLAB_15, PM.SLAB_15_SHA, "1.5 mm slab")
M = np.load(MESH, allow_pickle=True)
body = M["tet_body"].astype(str); zskin = float(M["z_skin"][0]) * 1e3
xp, hw = float(M["x_probe"][0]) * 1e3, float(M["probe_w"][0]) * 1e3 / 2
FRAME_MS = 1000.0 / 60.0

def vol(X, T):
    a, b, c, d = (X[T[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0

def mm(a):
    return a * 1e3 if np.abs(a).max() < 1 else a

rows = []
for s in (168, 67, 17, 8, 3, 2):
    _u = PM.usable("slab", f"ts_s{s}_i140", rows_key="rows")
    if not _u:
        print(f"ts_s{s}_i140: no usable {PM.GENERATION} run yet -- skipped"); continue
    _jp = _u[0][1]
    P = np.load(_jp.with_name(_jp.stem + "_pos.npz")); js = json.load(open(_jp))
    assert js["mesh_provenance"]["npz"]["sha256"][:12] == PM.SLAB_15_SHA and js["substeps"] == s and js["iterations"] == 140
    T = P["tets"]; assert (T == M["tets"]).all()
    R = mm(P["rest"]); V0 = vol(R, T); C0 = R[T].mean(1)
    rc = {k: R[P[f"ring_{k}"]].mean(0) for k in ("vein", "artery")}
    rr = {k: np.linalg.norm(R[P[f"ring_{k}"]] - rc[k], axis=1).mean() for k in rc}
    dwall = {k: np.linalg.norm(C0[:, [0, 2]] - rc[k][[0, 2]], axis=1) - rr[k] for k in rc}
    wall = np.isin(body, ["Vein", "Artery"]) | ((body == "void") & (np.minimum(dwall["vein"], dwall["artery"]) < 2.0))
    between = (dwall["vein"] < 3.0) & (dwall["artery"] < 3.0)
    logged = {r["depth_mm"]: r["n_inverted"] for r in js["rows"]}
    settled_all = all(r["settled_at"] is not None and r["veto"] == "none" for r in js["rows"])
    # 2026-10-21: depth from the key's own name (pos_25p4mm -> 25.4), as audit_leg_inversion.py does. The V5 files carry no
    # depths_mm array, and pairing keys with an array by file order was fragile anyway.
    _keys = sorted((k for k in P.files if k.startswith("pos_")), key=lambda k: float(k[4:-2].replace("p", ".")))
    for key, dep in ((k, float(k[4:-2].replace("p", "."))) for k in _keys):
        X = mm(P[key]); J = vol(X, T) / V0; inv = J < 0
        assert abs(int(inv.sum()) - logged[float(dep)]) <= 1, (s, dep, int(inv.sum()), logged[float(dep)])
        c = C0[inv]
        rows.append(dict(time_step_ms=round(FRAME_MS / s, 3), substeps=s, settled_every_depth=settled_all, depth_mm=float(dep),
                         n_inverted=int(inv.sum()), pct_of_elements=round(100 * inv.mean(), 3),
                         max_depth_below_skin_mm=round(float((zskin - c[:, 2]).max()), 1) if inv.any() else "",
                         nearest_to_vessel_wall_mm=round(float(min(dwall["vein"][inv].min(), dwall["artery"][inv].min())), 1) if inv.any() else "",
                         n_between_artery_and_vein=int((inv & between).sum()), n_vessel_wall=int((inv & wall).sum()),
                         pct_within_3mm_of_probe_edge=round(100 * float(np.mean(np.abs(np.abs(c[:, 0] - xp) - hw) < 3)), 0) if inv.any() else ""))
if not rows:   # 2026-10-21: rewrite, never leave the previous generation's file in place
    (DATA / "inversion_by_depth.csv").write_text(f"# no usable {PM.GENERATION} runs yet\n")
    raise SystemExit(f"!!! no usable {PM.GENERATION} position files yet: inversion_by_depth.csv rewritten empty")
with open(DATA / "inversion_by_depth.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
