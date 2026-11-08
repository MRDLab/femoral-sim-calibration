"""The cast gel leg as a cross-section the leg mesher can read (2026-10-28).

Writes out/femoral_cast/section_s020.json in the same frame and format as V5's section, so graded_leg_mesh.py
builds the cast exactly as it built V5. The cast is an extrusion (two gel cylinders, two straight silicone tubes,
the V5 femur), so a section plus the leg length describes it completely. No OBJ is written: with none, every body is
present along the whole leg, which is the cast (all 81 mm).

Body ROLE names are V5's, so every tracker and check keyed on them works unchanged:
  SoftTissue = the cast's fat layer   Muscle = the cast's muscle core   Femur = V5's femur
  Artery / Vein = the silicone tube walls   BloodArtery / BloodVein = the water-filled bores (voids in the FEM)
The cast's MATERIALS are applied by graded_leg_mesh.py --materials cast (Phantom* entries in materials.py).

Sources (the cast record, not distributed):
  fat outer r 92.07 mm, muscle outer r 76.20 mm   -- acrylic tube and pipe bores, measured on their STLs
  artery tube 11.0 / 9.0 mm, vein tube 13.0 / 10.0  -- McMaster 5041K761, 5054K827
  centres 16.0 mm apart, vein 66.3 / artery 65.0 mm from the axis -- cap-plate holes, measured 2026-10-21
  femur: V5's section disc (confirmed the cast femur, 2026-10-28), on the axis like the plate's peg

The pair is placed symmetric about +Z (the probe axis, x = 0), artery lateral (x < 0) as on a right thigh.
🔴 Centring the probe on the pair is our choice. The B-mode frames show the pair about 3.9 mm off the image centre,
but which tube is which on screen is unconfirmed, so the offset cannot be signed. Stated, not hidden.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
from scipy.optimize import brentq

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--bending-equivalent-mm", type=float, default=None,
                help="2026-10-29: model each tube wall at this thickness (the element size) with the modulus that keeps the real "
                     "wall's bending stiffness, E_eff = E (t / t_model)^3, as the paper's appendix does for the vein wall. "
                     "Bore unchanged; the wall grows outward. Writes out/femoral_cast_bend<t>/")
args = ap.parse_args()

ROOT = pathlib.Path(__file__).resolve().parent.parent
V5 = ROOT / "out" / "femoral_v5" / "section_s020.json"
OUT = (ROOT / "out").resolve() / ("femoral_cast" if args.bending_equivalent_mm is None
                                  else f"femoral_cast_bend{args.bending_equivalent_mm:g}".replace(".", "p"))

MM = 1e-3
R_FAT, R_MUSCLE = 92.07, 76.20
ART_OD, ART_ID, VEIN_OD, VEIN_ID = 11.0, 9.0, 13.0, 10.0
WALL_E = {}   # per-wall modulus override, Pa; empty = the materials.py values
if args.bending_equivalent_mm is not None:
    import sys
    sys.path.insert(0, str(ROOT))
    import materials as MAT
    TM = args.bending_equivalent_mm
    for wall, od, idd, src in (("Artery", ART_OD, ART_ID, "PhantomCFA"), ("Vein", VEIN_OD, VEIN_ID, "PhantomCFV")):
        t = (od - idd) / 2
        assert t <= TM + 1e-9, f"{wall} wall {t} mm is already thicker than {TM} mm"
        WALL_E[wall] = MAT.BODIES[src]["E_pa"] * (t / TM) ** 3
        print(f">>> bending-equivalent {wall}: real wall {t:.2f} mm at {MAT.BODIES[src]['E_pa']/1e6:.2f} MPa "
              f"-> {TM:.2f} mm at {WALL_E[wall]/1e6:.3f} MPa")
    ART_OD, VEIN_OD = ART_ID + 2 * TM, VEIN_ID + 2 * TM
R_ART_C, R_VEIN_C, SPACING = 65.0, 66.3, 16.0

v5 = json.load(open(V5))
fem = v5["bodies"]["Femur"]

# artery at (-a, za), vein at (+a, zv): |A| = 65.0, |V| = 66.3, |A - V| = 16.0
f = lambda a: np.hypot(2 * a, np.sqrt(R_VEIN_C**2 - a**2) - np.sqrt(R_ART_C**2 - a**2)) - SPACING
a = brentq(f, 0.1, 30.0)
A = np.array([-a, np.sqrt(R_ART_C**2 - a**2)])
V = np.array([+a, np.sqrt(R_VEIN_C**2 - a**2)])

# ── checks that can fail ──────────────────────────────────────────────────────────────────────
checks = {}
checks["vessel spacing is the cap plates' 16.0 mm"] = abs(np.linalg.norm(A - V) - SPACING) < 1e-6
checks["artery 65.0 and vein 66.3 mm from the axis"] = (abs(np.linalg.norm(A) - R_ART_C) < 1e-6
                                                        and abs(np.linalg.norm(V) - R_VEIN_C) < 1e-6)
checks["lateral order: artery x < vein x (right thigh)"] = A[0] < V[0]
checks["both tubes lie inside the muscle core with a gap"] = (np.linalg.norm(A) + ART_OD / 2 < R_MUSCLE - 1.0
                                                              and np.linalg.norm(V) + VEIN_OD / 2 < R_MUSCLE - 1.0)
checks["the tubes do not touch each other"] = np.linalg.norm(A - V) > (ART_OD + VEIN_OD) / 2
fem_c = np.array(fem["c"]) / MM
checks["the femur is on the axis (the plate's peg), within 2 mm"] = np.linalg.norm(fem_c) < 2.0
checks["the femur clears both tubes"] = all(np.linalg.norm(P - fem_c) > fem["r"] / MM + od / 2
                                            for P, od in ((A, ART_OD), (V, VEIN_OD)))
for k, ok in checks.items():
    print(("  ✅ " if ok else "  🚨 ") + k)
assert all(checks.values()), [k for k, ok in checks.items() if not ok]

bodies = {
    "BloodArtery": dict(kind="disc", c=(A * MM).tolist(), r=ART_ID / 2 * MM),
    "Artery": dict(kind="annulus", c=(A * MM).tolist(), r_in=ART_ID / 2 * MM, r_out=ART_OD / 2 * MM),
    "BloodVein": dict(kind="disc", c=(V * MM).tolist(), r=VEIN_ID / 2 * MM),
    "Vein": dict(kind="annulus", c=(V * MM).tolist(), r_in=VEIN_ID / 2 * MM, r_out=VEIN_OD / 2 * MM),
    "Femur": dict(fem),
    "Muscle": dict(kind="disc", c=[0.0, 0.0], r=R_MUSCLE * MM),
    "SoftTissue": dict(kind="annulus", c=[0.0, 0.0], r_in=R_MUSCLE * MM, r_out=R_FAT * MM),
}
sec = {k: v5[k] for k in ("units", "frame", "station_s_mm", "ligament_lps_z_mm", "section_lps_z_mm", "fit_tol_mm")}
sec.update(
    mesh_dir=str(OUT),
    nest_order=["BloodArtery", "Artery", "BloodVein", "Vein", "Femur", "Muscle", "SoftTissue"],
    absent=["Nerve, Sheath, Perivascular, FemoralCanal, PubicRamus (not in the cast)"],
    bodies=bodies,
    generation="cast",
    roles={"SoftTissue": "PhantomFat", "Muscle": "PhantomMuscle", "Artery": "PhantomCFA", "Vein": "PhantomCFV",
           "BloodArtery": "PhantomCFALumen", "BloodVein": "PhantomCFVLumen", "Femur": "Femur (V5)"},
    derived_from=dict(femur_from=str(V5), femur_sha256_12=hashlib.sha256(open(V5, "rb").read()).hexdigest()[:12]),
    placement="pair symmetric about +Z (probe axis), artery lateral; centring is a modelling choice (see docstring)",
    wall_E_pa=WALL_E or None,
    wall_E_basis=(None if not WALL_E else
                  f"bending match E_eff = E (t / {args.bending_equivalent_mm} mm)^3, as the paper's appendix (Lebahn 2025, "
                  f"Alibaz-Oner 2021); E from materials.py Phantom* (Gent's relation, textbook)"),
    checks={k: bool(v) for k, v in checks.items()},
)
OUT.mkdir(parents=True, exist_ok=True)
p = OUT / "section_s020.json"
json.dump(sec, open(p, "w"), indent=1)
print(f">>> artery centre ({A[0]:+.3f}, {A[1]:.3f}) mm, vein centre ({V[0]:+.3f}, {V[1]:.3f}) mm")
print(f">>> vein lumen floor at depth {R_FAT - (V[1] - VEIN_ID / 2):.2f} mm, artery {R_FAT - (A[1] - ART_ID / 2):.2f} mm below the skin")
print(f">>> wrote {p}  sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()[:12]}")
