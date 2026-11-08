"""Mesh F variants for the vein-wall and nerve stiffness sensitivity test (2026-09-30). One variable per variant.

    $PY_ANATOMY anatomy_meshing/make_sensitivity_meshes.py

Why: Mesh F's vein wall is 20 kPa nominal (median 14.3 kPa as meshed, because the 0.5 mm wall is thinner than the 1.5 mm
elements and every wall element is blended with its neighbours). Lebahn et al. 2025 measured 5.7 MPa for femoral vein in
tension. The question: does the paper's result (vein flattens, artery holds) depend on this choice?

Each variant copies Mesh F (c4e13e7d7574) and changes ONLY the Young's modulus of every element labelled with one body,
set uniformly (Poisson ratio 0.45 and damping kept). Geometry, nodes, tets, pins, rings and USD are unchanged.
  vein_uniform_20kpa    Vein -> 20 kPa  : removes the blending at the nominal value (the isolating middle case)
  vein_600kpa           Vein -> 600 kPa : the artery wall's value
  vein_5p7mpa           Vein -> 5.7 MPa : Lebahn 2025, unfixed, longitudinal tension
  nerve_20kpa           Nerve -> 20 kPa : half its nominal 40 kPa
  nerve_80kpa           Nerve -> 80 kPa : double
Added 2026-09-30, before any sensitivity run, by direction: two BENDING-EQUIVALENT vein walls. A thin vein collapses by
bending, whose stiffness scales with E t^3. Mesh F's wall is one 1.5 mm element thick, so a band matching a real wall's
bending stiffness needs E_eff = E_real (t_real / 1.5 mm)^3, with E_real = 5.7 MPa (Lebahn 2025):
  vein_bend_034mm       Vein -> 66.4 kPa  : t_real = 0.34 mm, the measured CFV wall (Alibaz-Oner 2020)
  vein_bend_050mm       Vein -> 211.1 kPa : t_real = 0.50 mm, the CAD wall
The control is Mesh F itself, run the same way (see run_scripts/paper_repeats_sens_20260930.sh).
Each variant gets a new digest, recorded in its meta and in out/mesh_leg_sensitivity/manifest.json, so it can never be
taken for Mesh F.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import time

import numpy as np

OUT = (pathlib.Path(__file__).resolve().parent.parent / "out").resolve()
# 2026-10-21: source, its digest and the output directory are arguments, so the same cases can be cut from the V5 leg
# mesh. The defaults are the 2026-09-30 V4 run, unchanged: with no arguments this script still rebuilds exactly that set.
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=str(OUT / "mesh_leg" / "tetmesh_leg_s020_f1p5_c30_w30_bone"), help="mesh path without .npz")
ap.add_argument("--src-sha", default="c4e13e7d7574", help="the source's sha256 (12 hex); a mismatch refuses")
ap.add_argument("--dst", default=str(OUT / "mesh_leg_sensitivity"))
args = ap.parse_args()
SRC, SRC_SHA = pathlib.Path(args.src), args.src_sha
DST = pathlib.Path(args.dst); DST.mkdir(exist_ok=True)
NU = 0.45
CASES = {"vein_uniform_20kpa": ("Vein", 20e3), "vein_600kpa": ("Vein", 600e3), "vein_5p7mpa": ("Vein", 5.7e6),
         "nerve_20kpa": ("Nerve", 20e3), "nerve_80kpa": ("Nerve", 80e3),
         "vein_bend_034mm": ("Vein", 5.7e6 * (0.34 / 1.5) ** 3), "vein_bend_050mm": ("Vein", 5.7e6 * (0.50 / 1.5) ** 3)}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
assert sha(f"{SRC}.npz") == SRC_SHA, f"source digest {sha(f'{SRC}.npz')} is not the stated {SRC_SHA}"
src = dict(np.load(f"{SRC}.npz", allow_pickle=True))
meta0 = json.loads(str(src["meta"][0]))
manifest = {"source": f"{SRC}.npz", "source_sha256_12": SRC_SHA, "nu": NU, "cases": {}}
old = json.load(open(DST / "manifest.json"))["cases"] if (DST / "manifest.json").exists() else {}
for name, (body, E) in CASES.items():
    if (DST / f"{name}.npz").exists() and name in old:          # never rewrite: the launcher logged this digest
        manifest["cases"][name] = old[name]; print(f"{name:20s} exists, sha {old[name]['sha256_12']} (kept)"); continue
    d = {k: v.copy() for k, v in src.items()}
    k = d["tet_body"] == body
    assert k.sum() > 0, body
    mu = E / (2 * (1 + NU)); lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    tm = d["tet_materials"].astype(np.float64)
    before = (tm[k, 0] * (3 * tm[k, 1] + 2 * tm[k, 0]) / (tm[k, 1] + tm[k, 0]))
    tm[k, 0] = mu; tm[k, 1] = lam                       # column 2 (damping) untouched
    d["tet_materials"] = tm.astype(src["tet_materials"].dtype)
    changed = np.nonzero(np.any(d["tet_materials"] != src["tet_materials"], axis=1))[0]
    assert set(changed) <= set(np.nonzero(k)[0]), "an element outside the target body changed"
    meta = dict(meta0, sensitivity_variant=dict(case=name, body=body, E_pa=E, nu=NU, n_elements=int(k.sum()),
                E_before_pa=dict(min=float(before.min()), median=float(np.median(before)), max=float(before.max())),
                source_sha256_12=SRC_SHA, date=time.strftime("%Y-%m-%d")))
    d["meta"] = np.array([json.dumps(meta)])
    p = DST / f"{name}.npz"
    np.savez_compressed(p, **d)
    shutil.copyfile(f"{SRC}.usda", DST / f"{name}.usda")       # geometry is identical; the solver takes materials from the npz
    manifest["cases"][name] = dict(body=body, E_pa=E, n_elements=int(k.sum()), sha256_12=sha(p),
                                   E_before_median_pa=float(np.median(before)))
    print(f"{name:20s} {body:5s} {int(k.sum()):4d} elements  E {np.median(before)/1e3:7.1f} kPa (median before) -> {E/1e3:8.1f} kPa  sha {sha(p)}")
json.dump(manifest, open(DST / "manifest.json", "w"), indent=1)
