"""Reduce the V3 CAD to ONE cross-section artifact that both engines read.

Run on the host with the anatomy venv (needs trimesh; the physics venv has neither
trimesh nor shapely and this file exists partly so it never needs them):

    $PY_ANATOMY v3_section.py --station-s 30

WHY THIS FILE EXISTS. femoral_fem.py has carried the phantom cross-section as five
hard-coded module constants since V2:

    ARTERY_C, ARTERY_RO, ARTERY_RI = (0.0 * MM, 70.80 * MM), 5.50 * MM, 4.48 * MM

Those numbers were typed in from V2/README.md. Nothing ever checked them against the OBJs
raysim traces, and on 2026-08-07 exactly that gap bit: the leg USD had been built from V3
while the FEM was solving V2, so the two panels of the same animation were showing
different phantoms and every check passed. Chapter 5's standing lesson, third recurrence.

🎯 So the geometry is MEASURED off the CAD here, once, and written to a JSON that the FEM
reads. There is then one definition of where the vein is, and it is derived from the same
meshes the ray tracer images. A number that is wrong is now wrong in both engines
simultaneously -- which is strictly better than being wrong in one.

THE AXIAL AXIS, WHICH IS THE WHOLE POINT OF V3. V2 is a 445 mm extrusion: every
cross-section is identical, so it has no superior end and no place a sheath could taper.
V3 has `s` = distance distal (inferior) to the inguinal ligament, and the structures start
and stop along it:

    Femur / Muscle / SoftTissue   s = 0 .. 120
    BloodVein / Vein              s = 0 .. 80     (to the profunda vein junction)
    Artery / BloodArtery          s = 0 .. 37.5   (to the profunda artery origin)
    Sheath                        s = 0 .. 35     (funnel, tapering)
    Nerve                         s = 0 .. 25
    FemoralCanal                  s = 0 .. 12.5

⚠️ SO THE STATION IS A REAL CHOICE AND IT COSTS BODIES. At s = 30 -- the saphenofemoral
junction, where Zambetti measured every diameter and depth we use -- there is NO NERVE and
NO FEMORAL CANAL. That is not a bug in the slice; it is where those structures end. Pick
s <= 25 to keep the nerve, s <= 12.5 to keep the canal, and accept that the cross-section
dimensions are then off their measurement plane.

SHAPE FITTING, AND WHY IT IS NOT ALL CIRCLES. `trimesh.section` returns the OUTLINE of a
body, not a filled region, so a solid rod yields one loop and an annular wall yields two.
Each loop is fitted and the fit is VERIFIED against a tolerance -- a body that is not the
shape we claim fails the run rather than being silently approximated:

  * circle   -- Femur, Muscle, SoftTissue, Artery, Vein and the two lumens
  * ellipse  -- Nerve. V2 modelled it as a Ø10 rod; Lonchena 2016 measures 9.6 x 3.7 mm,
                a ~2.6:1 ribbon, and V3 builds it that way. The shape matters beyond looks:
                materials.py maps the nerve to `fat` because that is the only shipped
                raysim material bright enough to stand in for a hyperechoic nerve, so the
                nerve's SHAPE is what carries its sonographic identity. A round rod reads
                as another vessel.
  * polygon   -- Sheath. It is a funnel enclosing two vessels and a canal; it is not
                circular at any station and must not be forced into a circle.

FRAME. Output is the FEM's frame in METRES: X = LPS x, Z = -LPS y, which is what
femoral_fem.py's `classify()` has always taken. Skin is at Z = +0.0925 and the probe
presses down -Z from above. The conversion happens here and nowhere else.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import trimesh

MM = 1e-3

# Innermost-wins order, mirroring materials.py's nesting and the OBJ author order raysim
# resolves. A point inside the artery lumen is blood even though it is also inside the wall,
# the sheath and the muscle core -- so the first body in this list that contains the point
# is the answer. ⚠️ Edit this and materials.py together or the two engines disagree about
# what a point is made of, which is the failure this whole file exists to prevent.
# 🚨 PubicRamus ADDED 2026-08-11. V4's whole reason for existing is that the CFV's backstop at
# the groin is the superior pubic ramus, not the femoral shaft. It was NOT in this list, so the
# first V4 section silently dropped it and reported a perfectly clean-looking cross-section with
# no backstop in it -- a body list that omits a body is indistinguishable from a body that is
# absent by anatomy, which is the same class of failure as the catalogue-vs-mesh CFL check.
# It sits beside Femur: rigid, and outside every soft body in the nesting.
NEST_ORDER = ["BloodArtery", "Artery", "BloodVein", "Vein", "Nerve", "FemoralCanal",
              "Femur", "PubicRamus", "Sheath", "Muscle", "SoftTissue"]

# A loop is accepted as a circle only if every vertex sits within this of the fitted radius.
# 0.10 mm is roughly the CAD's own tessellation chord error at these radii; anything looser
# would quietly accept the Sheath as a circle.
CIRCLE_TOL_MM = 0.10
ELLIPSE_TOL_MM = 0.25


def fit_circle(P: np.ndarray):
    """Algebraic (Kasa) circle fit, then the WORST residual -- not the mean.

    A mean residual hides exactly the case that matters: an oval whose average radius is
    fine. The caller rejects on max, so a sheath cannot pass as a circle.
    """
    x, y = P[:, 0], P[:, 1]
    A = np.column_stack([x, y, np.ones(len(P))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    resid = float(np.abs(np.hypot(x - cx, y - cy) - r).max())
    return float(cx), float(cy), r, resid


def fit_ellipse_aa(P: np.ndarray):
    """Axis-aligned ellipse from the bounding box, with the worst residual reported.

    Axis-aligned is not an assumption about ellipses in general -- it is a claim about THIS
    body, and the residual test below is what makes it a claim rather than a hope. The
    femoral nerve in V3 is built as an axis-aligned ellipse; if a future CAD rotates it,
    this fit fails loudly instead of shrinking the ribbon to its bounding box.
    """
    cx = 0.5 * (P[:, 0].min() + P[:, 0].max())
    cy = 0.5 * (P[:, 1].min() + P[:, 1].max())
    a = 0.5 * (P[:, 0].max() - P[:, 0].min())
    b = 0.5 * (P[:, 1].max() - P[:, 1].min())
    # Residual as a radial distance, so the number is in millimetres and comparable with the
    # circle tolerance: scale the point onto the unit circle, then back out along its ray.
    t = np.hypot((P[:, 0] - cx) / a, (P[:, 1] - cy) / b)
    r_pt = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    resid = float(np.abs(r_pt - r_pt / np.maximum(t, 1e-12)).max())
    return float(cx), float(cy), float(a), float(b), resid


def loops_at(mesh: trimesh.Trimesh, z: float):
    """Closed 2D loops of a body's cross-section at LPS z, in FEM (X, Z) millimetres.

    Returns [] when the plane misses the body entirely -- which is the CORRECT answer for a
    structure that has ended, not an error. The caller decides whether a missing body
    matters at this station.
    """
    try:
        sec = mesh.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])
    except Exception:
        return []
    if sec is None:
        return []
    out = []
    for poly in sec.discrete:
        P = np.asarray(poly, dtype=np.float64)[:, :2]
        if len(P) < 3:
            continue
        # LPS (x, y) -> FEM (X, Z). This is the only place the flip happens.
        out.append(np.column_stack([P[:, 0], -P[:, 1]]))
    return out


def describe(name: str, loops):
    """Turn a body's loops into the smallest primitive that VERIFIABLY fits them."""
    fits = [fit_circle(P) for P in loops]
    worst = max(f[3] for f in fits)
    if worst <= CIRCLE_TOL_MM:
        rad = sorted(f[2] for f in fits)
        cx = float(np.mean([f[0] for f in fits]))
        cz = float(np.mean([f[1] for f in fits]))
        if len(rad) == 1:
            return dict(kind="disc", c=[cx, cz], r=rad[0], resid_mm=worst)
        if len(rad) == 2:
            return dict(kind="annulus", c=[cx, cz], r_in=rad[0], r_out=rad[1], resid_mm=worst)
        raise SystemExit(f"!!! {name}: {len(rad)} concentric loops -- no primitive for that")

    if len(loops) == 1:
        cx, cz, a, b, resid = fit_ellipse_aa(loops[0])
        if resid <= ELLIPSE_TOL_MM:
            return dict(kind="ellipse", c=[cx, cz], a=a, b=b, resid_mm=resid)

    # Polygon of last resort.
    # 🚨 THIS USED TO KEEP THE LARGEST LOOP ONLY, and that was right for exactly one body and
    # wrong for the one V4 added. The Sheath is a solid envelope, so its outer loop is the whole
    # story. The PUBIC RAMUS IS A RING -- superior limb, inferior limb, obturator foramen between
    # them -- so a transverse section cuts it TWICE, and the larger loop is the POSTERIOR limb.
    # Keeping only that one silently discarded the anterior surface, which is the entire reason
    # the body exists: measured 2026-08-11, the section reported the ramus top at 49.89 mm depth
    # while the real cross-section had it at 41.15 mm. A backstop 8.7 mm further away than the
    # geometry says, and nothing in the output looked wrong.
    # Every loop is kept now. `pts` stays the largest so old readers still work; `polys` is the
    # truth and is what femoral_fem uses when present.
    def _sorted(L):
        c = L.mean(axis=0)
        return L[np.argsort(np.arctan2(L[:, 1] - c[1], L[:, 0] - c[0]))]

    def _area(L):
        return abs(np.dot(L[:, 0], np.roll(L[:, 1], -1)) -
                   np.dot(L[:, 1], np.roll(L[:, 0], -1))) / 2.0

    ordered = sorted(loops, key=_area, reverse=True)
    polys = [_sorted(L).tolist() for L in ordered]
    return dict(kind="polygon", n_loops=len(loops), pts=polys[0],
                polys=polys, loop_areas_mm2=[float(_area(L)) for L in ordered],
                resid_mm=worst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "femoral_v3"))
    ap.add_argument("--station-s", type=float, default=30.0,
                    help="mm distal to the inguinal ligament. 30 = saphenofemoral junction, "
                         "Zambetti's measurement plane -- and past the nerve (ends at 25) "
                         "and the femoral canal (12.5). See the module docstring.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meshes = {}
    for f in sorted(os.listdir(args.mesh_dir)):
        if f.endswith(".obj"):
            meshes[os.path.splitext(f)[0]] = trimesh.load(
                os.path.join(args.mesh_dir, f), process=False)
    if not meshes:
        raise SystemExit(f"!!! no OBJs in {args.mesh_dir}")

    # s = 0 is the ligament plane and the spec puts it at the SUPERIOR end. Assert that
    # rather than assume it: if a future CAD export flips the axis, every station below
    # would silently mirror, and a mirrored femoral bundle is the kind of error that looks
    # entirely plausible in a render.
    shell = meshes.get("SoftTissue")
    if shell is None:
        raise SystemExit("!!! no SoftTissue body -- cannot locate the ligament plane")
    z_lig = float(shell.bounds[1][2])
    z_inf = float(shell.bounds[0][2])
    if not (z_lig > z_inf):
        raise SystemExit("!!! ligament plane is not the superior end of the shell")
    z = z_lig - args.station_s
    if not (z_inf <= z <= z_lig):
        raise SystemExit(f"!!! s = {args.station_s} mm is off the model "
                         f"(s spans 0 .. {z_lig - z_inf:.1f} mm)")

    print(f">>> {args.mesh_dir}")
    print(f">>> ligament plane (s=0) at LPS z = {z_lig:+.2f} mm; "
          f"station s = {args.station_s:.1f} mm -> LPS z = {z:+.2f} mm")
    print(f">>> frame out: FEM (X = LPS x, Z = -LPS y), METRES\n")

    bodies, absent = {}, []
    for name in NEST_ORDER:
        if name not in meshes:
            absent.append(name + " (no CAD body)")
            continue
        loops = loops_at(meshes[name], z)
        if not loops:
            s0 = z_lig - float(meshes[name].bounds[1][2])
            s1 = z_lig - float(meshes[name].bounds[0][2])
            absent.append(f"{name} (spans s = {s0:.1f}..{s1:.1f})")
            continue
        bodies[name] = describe(name, loops)

    print(f"{'body':14s} {'kind':9s} {'centre (X, Z) mm':>22s}  geometry")
    for n, d in bodies.items():
        c = f"({d['c'][0]:+7.2f},{d['c'][1]:+7.2f})" if "c" in d else " " * 17
        if d["kind"] == "disc":
            g = f"r = {d['r']:.2f}"
        elif d["kind"] == "annulus":
            g = f"r = {d['r_in']:.2f} .. {d['r_out']:.2f}"
        elif d["kind"] == "ellipse":
            g = f"semi-axes {d['a']:.2f} x {d['b']:.2f}  (ratio {d['a']/max(d['b'],1e-9):.2f}:1)"
        else:
            # report EVERY loop. Printing only the largest is how the ramus's anterior limb
            # stayed invisible for an afternoon even after the JSON started carrying it.
            LS = [np.array(q) for q in d.get("polys", [d["pts"]])]
            allp = np.vstack(LS)
            g = (f"{len(LS)} loop(s), {sum(len(q) for q in LS)} pts, "
                 f"X {allp[:,0].min():+.2f}..{allp[:,0].max():+.2f}  "
                 f"Z {allp[:,1].min():+.2f}..{allp[:,1].max():+.2f}")
        print(f"{n:14s} {d['kind']:9s} {c:>22s}  {g}   (fit resid {d['resid_mm']:.3f} mm)")

    if absent:
        print("\n❌ NOT PRESENT AT THIS STATION -- by anatomy, not by error:")
        for a in absent:
            print(f"   {a}")

    # Depth below skin is the number every clinical row in the spec is quoted against, so
    # print it here rather than leaving each reader to subtract from 92.5.
    skin = bodies.get("SoftTissue", {}).get("r_out")
    if skin:
        print(f"\nskin at Z = {skin:.2f} mm; depth below skin (near wall -> far wall):")
        for n, d in bodies.items():
            if n in ("SoftTissue", "Muscle"):
                continue
            if d["kind"] in ("disc", "annulus"):
                rr = d.get("r_out", d.get("r"))
                near, far = skin - (d["c"][1] + rr), skin - (d["c"][1] - rr)
            elif d["kind"] == "ellipse":
                near, far = skin - (d["c"][1] + d["b"]), skin - (d["c"][1] - d["b"])
            else:
                P = np.array(d["pts"])
                near, far = skin - P[:, 1].max(), skin - P[:, 1].min()
            print(f"   {n:14s} {near:6.2f} .. {far:6.2f} mm")

    def to_m(d):
        o = dict(d)
        for k in ("r", "r_in", "r_out", "a", "b"):
            if k in o:
                o[k] = o[k] * MM
        if "c" in o:
            o["c"] = [v * MM for v in o["c"]]
        if "pts" in o:
            o["pts"] = [[p[0] * MM, p[1] * MM] for p in o["pts"]]
        # 🪤 AND `polys` TOO. Added 2026-08-11 for the multi-loop pubic ramus and promptly
        # forgotten here, so the ramus shipped in MILLIMETRES while every other body was in
        # metres. Nothing errored: the FEM simply found no point inside a body 1000x too big
        # and reported "PubicRamus -- NOT IN THE ROI (0 tets)", which reads exactly like a
        # structure that has ended by anatomy. A new key in the dict is a new key here.
        if "polys" in o:
            o["polys"] = [[[q[0] * MM, q[1] * MM] for q in L] for L in o["polys"]]
        o.pop("resid_mm", None)
        return o

    out = args.out or os.path.join(args.mesh_dir, f"section_s{int(round(args.station_s)):03d}.json")
    doc = {
        "units": "m",
        "frame": "FEM: X = LPS x, Z = -LPS y. Skin centred on the origin, probe presses -Z.",
        "mesh_dir": args.mesh_dir,
        "station_s_mm": args.station_s,
        "ligament_lps_z_mm": z_lig,
        "section_lps_z_mm": z,
        "nest_order": [n for n in NEST_ORDER if n in bodies],
        "absent": absent,
        "fit_tol_mm": {"circle": CIRCLE_TOL_MM, "ellipse": ELLIPSE_TOL_MM},
        "bodies": {n: to_m(d) for n, d in bodies.items()},
    }
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
