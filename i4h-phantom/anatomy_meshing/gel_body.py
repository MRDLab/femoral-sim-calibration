"""Build the ultrasound GEL as a thin band around the limb, so the background can be air.

    $PY_ANATOMY anatomy_meshing/gel_body.py --obj-dir out/femoral_v3
    $PY_ANATOMY anatomy_meshing/gel_body.py --obj-dir out/deformed/travel_11.0mm

WHY THE BACKGROUND MUST BE AIR
------------------------------
Everything outside a limb is air. Until 2026-08-08 this project rendered it as WATER -- i.e.
it simulated a limb immersed in a water bath, which is not how anyone scans a groin.
Directive: **air is the background; water only when explicitly asked for.**

But raysim's background is a SINGLE material, so flipping it to air and stopping there turns
the exam into a non-exam. Measured on the `m_air` control already on disk, same pose as the
water reference:

    background   mean brightness   non-zero   mean below 25 mm (vessel depth)
    air                   14.99      19.5 %                        3.51
    water                 23.87      36.5 %                       16.30

4.6x less signal exactly where the vessels are, and the cause is not subtle -- raysim stores
air at Z = 0.000 MRayl EXACTLY:

    water | fat    0.12 % reflected,  99.88 % transmitted
    air   | fat   99.88 % reflected,   0.12 % transmitted

with air attenuating at 8.2 dB/cm/MHz against water's 0.0022. A 2 mm air gap between
transducer and skin is not a degraded scan, it is no scan. Which is exactly why gel exists.

WHAT THE GEL IS HERE
--------------------
🎯 A hollow cylinder: a band of `water` about `--thickness` mm thick wrapped round the whole
circumference of the limb, sitting directly on the skin. Nothing pose-dependent, nothing
regenerated per frame -- which is what makes it work for the full -90..+90 deg circumferential
sweep as a single static body. Physically it is a gel-coated limb, which is how you would prep
for a circumferential scan anyway.

📌 AND IT MAKES A FALSE STATE IMPOSSIBLE RATHER THAN MERELY DETECTABLE. With water everywhere,
a probe hovering 5 mm off the skin still returns a perfectly plausible B-mode: the water column
is anechoic, the anatomy images normally, and nothing in the picture says the transducer is not
touching. With air outside a finite gel band, a transducer that is not IN the gel has air in
front of it and images black. Chapter 8 was bitten four times in one day by instruments
returning correct-looking values for something they were not measuring; this is one class of
that closed by geometry instead of by an audit.

⚠️ SO THE STANDOFF NOW HAS TO FIT INSIDE THE BAND, AND TODAY IT DOES NOT. The renders stand the
probe 2.0 mm proud of the skin while the band is 1.0 mm thick, leaving ~1 mm of AIR between the
face and the gel -- every frame would go dark. That is a true statement about the pose, not a
bug in this file. Either the standoff comes down below the band thickness or the band is made
thick enough to swallow it, and that is a deliberate choice about what is being modelled: a
transducer pressed into a gel film, or a transducer standing off in a gel column.

COINCIDENT SURFACES
-------------------
🚨 The band's inner surface would otherwise be exactly coincident with the skin's outer
surface, and raysim's material stack is ONE DEEP -- it cannot order two hits at the same t.
This is the same failure the V2/V3 exports already work around with a 0.02 mm radial shrink on
nested bodies. So the band's inner radius is offset outward by `--eps`, and the gel therefore
starts a hair above the skin rather than exactly on it.
"""

from __future__ import annotations

# ── repo path bootstrap ─────────────────────────────────────────────────────────
# This script moved out of the flat repository root on 2026-08-25 and imports other
# modules in this repository, so the root and the code directories go back on the path.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import repo_paths  # noqa: F401,E402
# ────────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_geometry as PG  # noqa: E402  -- THE constants, OPEN_ITEMS 20


def measure_limb(path: str):
    """Outer radius and axial extent of the skin body, measured rather than assumed."""
    V = np.array([[float(t) for t in ln.split()[1:4]]
                  for ln in open(path) if ln.startswith("v ")])
    if len(V) < 8:
        raise SystemExit(f"!!! {path}: only {len(V)} vertices, that is not a limb")
    cx = 0.5 * (V[:, 0].min() + V[:, 0].max())
    cy = 0.5 * (V[:, 1].min() + V[:, 1].max())
    r = np.hypot(V[:, 0] - cx, V[:, 1] - cy)
    return float(cx), float(cy), float(r.max()), float(V[:, 2].min()), float(V[:, 2].max())


def annulus(cx, cy, r_in, r_out, z0, z1, nt=180):
    """Watertight hollow cylinder: two concentric walls plus two annular end caps.

    Wound so every normal points OUT of the solid -- outer wall outward, inner wall inward,
    caps away from the body. A flipped winding renders as an inside-out solid and raysim's
    material assignment follows the normals.
    """
    th = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    V, F = [], []

    def ring(rad, z):
        base = len(V)
        for t in th:
            V.append((cx + rad * np.cos(t), cy + rad * np.sin(t), z))
        return base

    o0, o1 = ring(r_out, z0), ring(r_out, z1)
    i0, i1 = ring(r_in, z0), ring(r_in, z1)
    for k in range(nt):
        k2 = (k + 1) % nt
        F += [(o0 + k, o1 + k2, o1 + k), (o0 + k, o0 + k2, o1 + k2)]      # outer wall
        F += [(i0 + k, i1 + k, i1 + k2), (i0 + k, i1 + k2, i0 + k2)]      # inner wall
        F += [(o0 + k, i0 + k2, o0 + k2), (o0 + k, i0 + k, i0 + k2)]      # z0 cap
        F += [(o1 + k, o1 + k2, i1 + k2), (o1 + k, i1 + k2, i1 + k)]      # z1 cap
    # 🚨 SIGNED VOLUME IS THE WITNESS, AND THE FIRST WINDING FAILED IT. A hollow cylinder can
    # be perfectly watertight and still be inside-out: the first version returned
    # watertight=True with volume -70.12 cm^3 against an ideal +70.14, i.e. right geometry,
    # inverted normals. raysim assigns material by normal direction, so an inverted gel band
    # would have made the OUTSIDE of the limb gel and the band itself air. `is_watertight`
    # alone would never have caught it -- the sign is the only thing that says which way the
    # solid faces.
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj-dir", required=True,
                    help="directory holding SoftTissue.obj and anatomy_config.json")
    ap.add_argument("--thickness", type=float, default=PG.GEL_THICKNESS_M * 1e3,
                    help="gel band thickness, mm. The default IS "
                         "probe_geometry.GEL_THICKNESS_M; overriding it without "
                         "moving the standoff is how OPEN_ITEMS 20 happened.")
    ap.add_argument("--eps", type=float, default=0.02,
                    help="radial offset above the skin so the two surfaces are not "
                         "coincident. Matches the export's own de-coincide epsilon.")
    ap.add_argument("--material", default="water",
                    help="gel is ~1520-1580 m/s and acoustically near-water; raysim's water "
                         "is anechoic and effectively lossless (mu0 = sigma = 0, "
                         "alpha 0.0022 dB/cm/MHz), which is how gel reads on a B-mode.")
    ap.add_argument("--background", default="air")
    ap.add_argument("--facets", type=int, default=180)
    ap.add_argument("--standoff-check", type=float, default=PG.PROBE_STANDOFF_M * 1e3,
                    help="the standoff the renders currently use, mm. Only used to warn when "
                         "the probe would sit outside the band and image black.")
    args = ap.parse_args()

    skin = os.path.join(args.obj_dir, "SoftTissue.obj")
    if not os.path.exists(skin):
        raise SystemExit(f"!!! no SoftTissue.obj in {args.obj_dir}")
    cx, cy, r_skin, z0, z1 = measure_limb(skin)
    r_in = r_skin + args.eps
    r_out = r_in + args.thickness

    V, F = annulus(cx, cy, r_in, r_out, z0, z1, args.facets)
    out = os.path.join(args.obj_dir, "Gel.obj")
    with open(out, "w") as fh:
        fh.write(f"# Gel: ultrasound coupling band, material={args.material}\n"
                 f"# hollow cylinder r {r_in:.4f} -> {r_out:.4f} mm, z {z0:.2f} .. {z1:.2f} mm\n"
                 f"# skin measured at r = {r_skin:.4f} mm; inner surface offset by "
                 f"eps={args.eps} mm so it is not coincident (raysim's stack is one deep)\n"
                 f"# SPACE=LPS, units mm\n")
        for v in V:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # 🚨 NORMALS ARE MANDATORY AND THIS FILE DID NOT WRITE THEM (found 2026-08-11).
        # raysim refuses the mesh outright -- `RuntimeError: Mesh: has no normals` -- so the
        # gel band could never have rendered, on top of the standoff/thickness contradiction
        # OPEN_ITEMS 20 already recorded. Two independent blockers in one body, and the second
        # was invisible until the first was fixed and a frame was actually attempted.
        # 📌 The normal is DERIVED FROM THE WINDING, not authored separately -- the winding is
        # what the signed-volume witness above validates, so there is exactly one source of
        # truth for which way the solid faces. Authoring normals independently would give the
        # inside-out failure a second place to hide.
        n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        if (ln < 1e-12).any():
            raise SystemExit(f"!!! {(ln < 1e-12).sum()} degenerate gel triangles -- "
                             f"cannot derive a normal; refusing to ship")
        n = n / ln
        for q in n:
            fh.write(f"vn {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}\n")
        for j, f in enumerate(F):
            fh.write(f"f {f[0]+1}//{j+1} {f[1]+1}//{j+1} {f[2]+1}//{j+1}\n")

    # Watertightness is checked if trimesh is available, and simply reported if not -- the
    # construction is closed by design, but "by design" is what this project keeps catching.
    wt = None
    try:
        import trimesh
        # 🪤 BUILD FROM THE ARRAYS, DO NOT RE-LOAD THE OBJ. Re-loading was how this check
        # worked until 2026-08-11, and the moment per-face normals were written it started
        # reporting watertight=False on geometry whose volume was still 99.98 % of ideal --
        # because `f v//vn` with a unique normal per face makes trimesh split the shared
        # vertices, so adjacency (and therefore watertightness) is destroyed by the ENCODING
        # rather than by the shape. The witness was measuring the file format, not the solid.
        # V and F are what the geometry IS; the OBJ is one serialisation of it.
        m = trimesh.Trimesh(vertices=V, faces=F, process=False)
        wt = bool(m.is_watertight)
        vol = float(m.volume)
        ideal = np.pi * (r_out ** 2 - r_in ** 2) * (z1 - z0)
        print(f">>> Gel.obj  {len(F)} tris  watertight={wt}  volume {vol/1000:.2f} cm^3 "
              f"(ideal {ideal/1000:.2f}, {100*vol/ideal:.2f}%)")
        if not wt:
            raise SystemExit("!!! gel band is not watertight -- refusing to ship it; raysim "
                             "needs closed bodies to resolve the material stack")
        if vol <= 0:
            raise SystemExit(f"!!! gel band is INSIDE OUT (signed volume {vol/1000:.2f} cm^3). "
                             f"Watertight but inverted: raysim would make the outside of the "
                             f"limb gel and the band air. Winding bug -- fix, do not ship.")
        if abs(vol - ideal) / ideal > 0.01:
            raise SystemExit(f"!!! gel volume {vol/1000:.2f} cm^3 is more than 1 % off the "
                             f"analytic {ideal/1000:.2f} cm^3 -- the band is not the shape "
                             f"it claims to be")
    except ImportError:
        print(f">>> Gel.obj  {len(F)} tris  (trimesh unavailable, watertightness unverified)")

    print(f"    skin r = {r_skin:.3f} mm, band {r_in:.3f} -> {r_out:.3f} mm "
          f"({args.thickness:.2f} mm thick), z {z0:.1f} .. {z1:.1f} mm")

    # HARD FAILURE, not the paragraph this used to print. The old version explained that a
    # probe outside the band images black "which is physically correct, not a bug" -- and
    # it was right, and the run shipped anyway, and photos/2026-08-08/bmode/ is an empty
    # directory because of it. A check that explains a problem and then proceeds is a
    # comment, not a check.
    clearance = PG.check_coupled(args.standoff_check * 1e-3, args.thickness * 1e-3)
    print(f"    coupled OK: the transducer face sits {clearance*1e3:.2f} mm INSIDE the "
          f"band at travel 0 -- {PG.summary()}")

    # 🚨 HARD FAILURE ON A MISSING CONFIG. This block used to be `if os.path.exists(cfg_path)`,
    # i.e. update-only, and that silence is what let the 2026-08-12 V4 campaign ship. V4 dirs
    # had no anatomy_config.json (v4_to_section.py never wrote one), so gel_body ran to
    # completion, printed a healthy `Gel.obj ... watertight=True`, and wrote NO `_provenance.gel`
    # block. `render_deformed.py:43` reads the standoff and the band thickness out of exactly
    # that block -- it runs inside the raysim container and cannot import probe_geometry.py --
    # so the renderer had nothing to read, and the gel body that IS on disk was invisible to
    # the config the renderer builds its world from. A successful-looking run that produced
    # half an artifact: defect class 2, an instrument reporting a true value (the mesh is fine)
    # about the wrong thing (whether the pipeline can use it).
    # 📌 The gel is not optional and neither is its provenance. Refuse.
    cfg_path = os.path.join(args.obj_dir, "anatomy_config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(
            f"!!! no anatomy_config.json in {args.obj_dir} -- Gel.obj was written but is NOT\n"
            f"    registered, and the `_provenance.gel` block render_deformed.py requires does\n"
            f"    not exist. A gel body no config names is a gel body raysim never adds.\n"
            f"    Write the config first:\n"
            f"      V2  stl_to_raysim.py · V3  build_v3_anatomy.py · V4  v4_to_section.py\n"
            f"      deformed states  deform_meshes.py (it copies the base config into each)\n"
            f"    REFUSING to report success on half an artifact.")
    cfg = json.load(open(cfg_path))
    prev = cfg["anatomy"].get("background_material")
    cfg["anatomy"]["background_material"] = args.background
    meshes = [x for x in cfg["anatomy"]["meshes"] if x["file"] != "Gel.obj"]
    # OUTERMOST in the nesting order: the gel is outside the skin and overlaps no tissue,
    # so every interior point still resolves exactly as it did before.
    meshes.append({"file": "Gel.obj", "material": args.material})
    cfg["anatomy"]["meshes"] = meshes
    cfg.setdefault("_provenance", {})["gel"] = dict(
        r_in_mm=r_in, r_out_mm=r_out, thickness_mm=args.thickness, eps_mm=args.eps,
        skin_r_mm=r_skin, material=args.material, watertight=wt,
        background_was=prev, background_now=args.background,
        # The renderer runs INSIDE the raysim container and cannot import
        # probe_geometry.py. Writing these here makes anatomy_config.json the one
        # artifact both sides read, instead of a constant typed on each side of a
        # container boundary -- which is how OPEN_ITEMS 20 happened.
        probe_standoff_mm=args.standoff_check,
        gel_clearance_mm=clearance * 1e3)
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"    config: background {prev!r} -> {args.background!r}, "
          f"Gel.obj added as {args.material}")
    if prev == "water":
        print("    🚨 every B-mode brightness metric measured before this change was taken "
              "against a WATER background and is NOT comparable with anything measured "
              "after it. Re-measure rather than restating.")


if __name__ == "__main__":
    main()
