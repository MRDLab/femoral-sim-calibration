"""Compression sonography on the femoral CAD phantom V2 — Newton FEM.

This is Chapter 6's Test C re-run on real anatomy instead of a synthetic two-lumen block:
the same question (does a VEIN collapse under probe pressure while an ARTERY does not?),
now on the V2 dimensions, with a real fat/muscle layering, a third structure (the nerve)
that must do neither, and a flat-faced probe rather than a sphere.

GEOMETRY. The phantom is a set of cylinders along one axis, so a transverse slab captures
all of it. We simulate a region of interest around the neurovascular bundle rather than the
whole 185 mm limb, in a frame chosen to match the existing Chapter 6 rigs:

    FEM +X = LPS +x   lateral, the axis the three structures are strung out along
    FEM +Y = LPS +z   along the vessels (the extrusion axis)
    FEM +Z = LPS -y   ANTERIOR, i.e. "up", toward the probe

so the limb axis sits at Z = 0 and the anterior skin pole at Z = +92.5, and every body is a
circle in the XZ plane exactly as in verify_bmode.py.

MATERIALS come from materials.py, which is the single table shared with the renderer. The
lumens are VOIDS -- there is no fluid in Newton, so the artery's resistance to collapse
comes entirely from its wall stiffness, which is standing in for wall + ~13 kPa transmural
blood pressure. Stated in materials.py, restated here: this answers "can the two be told
apart", not "is the collapse pressure quantitatively right".

STABILITY. Chapter 6's Test C produced clean geometry but an unusable force channel because
the stiff artery wall was 8.3x over its CFL limit. The fix that entry identified but never
applied -- more substeps -- is applied here and checked explicitly by `cfl_report()` before
the sweep runs, so the force numbers are either trustworthy or loudly flagged.
"""

from __future__ import annotations

import math
import argparse
import json
import pathlib
import sys

import numpy as np
import warp as wp

import newton

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from materials import BODIES  # noqa: E402
import probe_geometry as PG  # noqa: E402  -- THE standoff/gel constants, OPEN_ITEMS 20

OUT = pathlib.Path(__file__).parent / "out" / "fem"
OUT.mkdir(parents=True, exist_ok=True)

# ── phantom cross-section, LPS mm -> metres (see V2 README) ─────────────────────────────
MM = 1e-3
R_SKIN, R_MUSCLE = 92.5 * MM, 76.48 * MM
ARTERY_C, ARTERY_RO, ARTERY_RI = (0.0 * MM, 70.80 * MM), 5.50 * MM, 4.48 * MM
VEIN_C, VEIN_RO, VEIN_RI = (15.0 * MM, 67.66 * MM), 6.00 * MM, 5.48 * MM
NERVE_C, NERVE_R = (-12.0 * MM, 66.50 * MM), 5.00 * MM

# Region of interest: wide enough to hold nerve(-12-5) .. vein(15+6) plus free tissue
# either side of the probe footprint, and deep enough to put the pinned base well clear of
# the structures. Trimmed hard, because the artery wall is only 1.0 mm thick and has to be
# resolved -- which sets dx, which sets everything else.
X0, X1 = -0.024, 0.027          # lateral, 51 mm
LY = 0.016                      # along the vessels
Z0 = 0.056                      # bottom of the block (deep muscle), pinned

# ── the cross-section, as an ordered list of (body, containment test) ────────────────────
# Innermost first: the FIRST entry containing a point decides what that point is made of.
# `None` means the built-in V2 cross-section above; load_section() swaps in a V3 station.
#
# 🚨 WHY THIS INDIRECTION EXISTS AT ALL. Until 2026-08-08 the cross-section WAS the five
# constants above, typed in by hand from V2/README.md, and nothing ever compared them with
# the OBJs raysim traces. On 08-07 the leg USD was built from V3 while this file solved V2,
# so the two panels of one animation showed different phantoms and every assertion passed.
# v3_section.py now measures the geometry off the CAD and writes it to JSON; this reads it.
# The V2 path is kept unchanged so old runs still reproduce.
SECTION = None
SECTION_META = {}


def _disc(c, r):
    cx, cz = c
    return lambda x, z: (x - cx) ** 2 + (z - cz) ** 2 < r * r


def _ellipse(c, a, b):
    cx, cz = c
    return lambda x, z: ((x - cx) / a) ** 2 + ((z - cz) / b) ** 2 < 1.0


def _signed_area(P):
    """Shoelace area of a closed loop, in the section's own units (m²). Sign is orientation."""
    x, z = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


def _polygons(loops):
    """Union of several closed loops -- a body whose cross-section is more than one island.

    🚨 ADDED 2026-08-11 FOR THE PUBIC RAMUS. A ring sectioned transversely gives two loops, and
    treating only the larger as "the body" throws away the other limb. See v3_section.describe.
    """
    tests = [_polygon(P) for P in loops]

    def inside(x, z):
        return any(t(x, z) for t in tests)
    return inside


def _polygon(pts):
    """Crossing-number point-in-polygon. Written out rather than imported because the
    physics venv has neither shapely nor trimesh, and this is the only place a non-circular
    body appears -- the femoral sheath, which is a funnel and must not be forced to a circle.
    """
    P = np.asarray(pts, dtype=np.float64)
    xs, zs = P[:, 0], P[:, 1]
    xs2, zs2 = np.roll(xs, -1), np.roll(zs, -1)
    dz = np.where(zs2 == zs, 1e-30, zs2 - zs)

    def inside(x, z):
        crosses = ((zs > z) != (zs2 > z)) & (x < (xs2 - xs) * (z - zs) / dz + xs)
        return bool(crosses.sum() % 2)
    return inside


def _v2_section():
    """The historical cross-section as an explicit list. Same tests, same order and the
    same answers as the hand-written if-chain it replaced."""
    return [("void", _disc(ARTERY_C, ARTERY_RI)),
            ("Artery", _disc(ARTERY_C, ARTERY_RO)),
            ("void", _disc(VEIN_C, VEIN_RI)),
            ("Vein", _disc(VEIN_C, VEIN_RO)),
            ("Nerve", _disc(NERVE_C, NERVE_R)),
            ("Muscle", _disc((0.0, 0.0), R_MUSCLE)),
            ("SoftTissue", _disc((0.0, 0.0), R_SKIN))]


def load_section(path: str, loose_cuff: float = 0.0,
                 probe_half_w: float = 0.020, probe_margin: float = 0.006,
                 perivascular: bool = False):
    """Install a V3 station from v3_section.py's JSON, and resize the ROI to hold it.

    `loose_cuff` (metres) grows the sheath region by a perivascular collar of that thickness
    around each vessel wall, clipped to the muscle core.

    🚨 WHY A COLLAR IS NEEDED AT ALL -- MEASURED 2026-08-08, AND IT IS A CAD DEFECT.
    V3's `Sheath` body does NOT enclose the vessels on the side the probe presses from. Its
    anterior face sits at Z = 72.52 mm; the artery wall tops out at 73.99 and the vein at
    74.29, so both poke 1.5-1.8 mm clear of it. Only 55.3 % of the artery's perimeter and
    71.8 % of the vein's lie inside the envelope, and the uncovered arc is centred on the
    anterior direction. Trace the beam down from the skin over either vessel and the column
    reads SoftTissue -> Muscle -> vessel wall: there is no sheath in the load path.
    📌 So softening the envelope ALONE tests almost nothing -- it puts the compliance behind
    and beside the vessels, where the press does not act. That run would come back showing
    little change and would read as "loose tissue is not the mechanism", which would be a
    wrong conclusion with a plausible mechanism already attached. The collar is what puts
    the compliance where Moore's sheath actually is: blending with the vessel adventitia.
    ⚠️ Clipped to the muscle core deliberately. The spec sets the 16 mm fat shell so the
    fat/muscle boundary lands one artery wall above the intima, so any collar over ~2 mm
    would otherwise punch into subcutaneous fat and quietly thin the shell.

    Rebinds the module globals the rest of this file reads -- including the lumen centres
    and radii the area trackers key off, so `areas()` measures the vessel that is actually
    meshed rather than the V2 vessel that used to be.

    ⚠️ THE ROI FLOOR MOVES, AND THAT IS THE POINT OF THE V3 RUN, NOT A SIDE EFFECT.
    Z0 = 56.0 mm put a FULLY RIGID, FULLY CLAMPED plane 5.66 mm behind the vein's far wall.
    That is a stiffer and far closer backstop than any bone, so every V2 collapse number was
    measured against an artificial one. V3's premise (spec section 4) is that the FEMUR is
    the backstop -- and the femur sits ~51.6 mm behind the vein. Dropping the floor into the
    bone replaces the artificial clamp with the anatomical one.
    📌 EXPECT LESS CLOSURE, NOT MORE, FROM THIS CHANGE ALONE. Removing a clamp 5.66 mm behind
    the vein and putting bone 51.6 mm behind it takes support AWAY. If the vein closes better
    in V3 it is because of the compliance body, and the A/B is what separates the two.
    """
    global SECTION, SECTION_META, X0, X1, Z0, R_SKIN, R_MUSCLE
    global ARTERY_C, ARTERY_RO, ARTERY_RI, VEIN_C, VEIN_RO, VEIN_RI, NERVE_C, NERVE_R

    doc = json.load(open(path))
    if doc.get("units") != "m":
        raise SystemExit(f"!!! {path}: expected metres, got {doc.get('units')!r}")
    B = doc["bodies"]
    SECTION_META = {k: doc[k] for k in
                    ("station_s_mm", "section_lps_z_mm", "ligament_lps_z_mm", "absent",
                     "mesh_dir")}

    # 🚨 A MISSING MODULUS MUST NOT BECOME A HOLE. `E_pa is None` means two completely
    # different things in materials.py: "this is a lumen, there is no fluid in Newton, treat
    # it as void" (BloodArtery, BloodVein) and "nobody has sourced this yet" (Sheath). Mapping
    # both to void turns the entire femoral sheath envelope into a 32 x 13 mm cavity -- and
    # the vein would then collapse spectacularly into it and the run would look like a
    # success. Every zero needs a witness; so does every void.
    # 🎯 LUMEN_FILL PUTS BLOOD IN THE LUMEN AS SOLID ELEMENTS, 2026-08-13.
    # The default stays void, and the reasoning is at _refresh_lumen_pressure: a real vein
    # collapses because its blood is free to leave AXIALLY, so a trapped incompressible filler
    # models a SEALED BAG and would hold the lumen near 100 % of rest.
    # ⚠️ But that is an ARGUMENT, and this project measures arguments instead of trusting them.
    # LUMEN_FILL turns it into an experiment: fill the lumen with a solid of stated stiffness
    # and sweep it. A high bulk modulus should hold the vein open -- the sealed-bag failure the
    # comment predicts -- and a low one should approach the void. Where the tube law actually
    # lands is a measurement, and it settles the question instead of re-arguing it.
    # 📌 ACOUSTICALLY THE LUMEN HAS ALWAYS BEEN BLOOD: materials.py maps BloodVein to raysim's
    # `blood` and always has. This flag is about the FEM only; the two must not be conflated.
    _LUMEN_BODY = {"vein": "BloodVein", "artery": "BloodArtery"}
    LEGITIMATE_VOIDS = {"BloodArtery", "BloodVein"} - {_LUMEN_BODY[k] for k in LUMEN_FILL}
    for _k, _spec in LUMEN_FILL.items():
        _b = _LUMEN_BODY[_k]
        BODIES[_b]["E_pa"], BODIES[_b]["nu"] = float(_spec["E_pa"]), float(_spec["nu"])
        _K = _spec["E_pa"] / (3.0 * (1.0 - 2.0 * _spec["nu"]))
        print(f">>> 🩸 LUMEN FILL: {_b} is a SOLID, E = {_spec['E_pa']:.4g} Pa, "
              f"nu = {_spec['nu']:.4f}  ->  bulk K = {_K:.4g} Pa   (NOT a void)")
    orphans = [n for n in doc["nest_order"]
               if n not in LEGITIMATE_VOIDS and (BODIES[n]["E_pa"] is None
                                                 or BODIES[n]["nu"] is None)]
    if orphans:
        raise SystemExit(
            f"!!! {', '.join(orphans)} reach the mesh with no modulus.\n"
            f"    materials.py leaves these None because nothing on disk sources them.\n"
            f"    Set them before calling load_section -- silently voiding a solid body\n"
            f"    would put a cavity where tissue should be and the collapse curve would\n"
            f"    still come out monotonic and plausible.")

    # 🎯 THE PERIVASCULAR COMPARTMENT, SYNTHESISED FROM THE SHEATH'S OWN INNER LOOP.
    # MEASURED 2026-08-14, diag_funnel_section.py, section_s030 of femoral_v4_gap12:
    #   the CAD authors `Sheath` as TWO NESTED LOOPS -- outer 334.27 mm2, inner 291.61 mm2,
    #   all 222 inner vertices inside the outer -- i.e. a fascial SHELL of wall area 42.66 mm2.
    #   `_polygons` UNIONS them, so the FEM has always meshed the funnel as a SOLID: 177.51 mm2
    #   of the envelope (53.1 %) is fill, and 100 % of that fill carries the `--sheath-kpa`
    #   default of 20 kPa -- muscle. Trace a press column straight down onto the vein and it
    #   reads: fat 16.00 mm -> 2.30 mm of that 20 kPa fill -> vein wall. The vein has been
    #   embedded in muscle on the side the probe presses from, for every press this project
    #   has ever run, and item 29's "no collapse mechanism" is measured against that phantom.
    # 📌 So this is a SPLIT, not an addition -- no CAD rebuild, no new geometry, no new OBJ.
    #   The bore is already in the file; it was being thrown away by a union.
    #   `Perivascular` is inserted immediately BEFORE `Sheath` in nest order, so the vessels
    #   (which come first) still win inside it and `Sheath` keeps only the wall between the
    #   loops. Bore composition, measured: 142.32 mm2 fill + 32.48 vessel wall + 116.86 lumen.
    # 🎯 WHY IT MUST BE THE SOFTEST THING IN THE LOAD PATH: so probe force yields HERE first
    #   and the vein collapses into the space that gives, instead of the whole column
    #   displacing in bulk -- which is what 2026-08-07 measured (skin -6.29 mm, vein -0.66 mm).
    # ⚠️ ALLOWED TO FAIL. If closure does not improve with a 3 kPa bore, the perivascular
    #   hypothesis is refuted and item 29 needs a different lever; that is a reportable result
    #   and this flag must not be walked outside 1-5 kPa to chase a nicer number.
    if perivascular:
        sh = B.get("Sheath")
        loops = (sh or {}).get("polys")
        if sh is None or not loops or len(loops) < 2:
            raise SystemExit(
                "!!! --perivascular asked for, but this section's Sheath has no inner loop.\n"
                f"    Sheath present: {sh is not None}; loops: {0 if not loops else len(loops)}.\n"
                "    The compartment is DERIVED from that loop -- without it there is nothing\n"
                "    to derive it from, and inventing a boundary here would put a soft body of\n"
                "    our own construction exactly where the answer is supposed to come from.\n"
                "    Run against a V3/V4 section, or run without --perivascular.")
        _areas = [abs(_signed_area(np.asarray(P, dtype=np.float64))) for P in loops]
        _outer = int(np.argmax(_areas))
        _bore = [i for i in range(len(loops)) if i != _outer]
        # 🚨 NESTING IS VERIFIED, NOT ASSUMED. A funnel sectioned off-axis can give two
        # DISJOINT islands rather than a shell (the pubic ramus does exactly that two bodies
        # up), and unioning those as "the bore" would hand a soft modulus to a solid limb of
        # the sheath. Same defect class as the V4-detected-as-V3 USD: plausible and wrong.
        _out_test = _polygon(loops[_outer])
        for i in _bore:
            P = np.asarray(loops[i], dtype=np.float64)
            n_in = sum(1 for x, z in P if _out_test(float(x), float(z)))
            if n_in < len(P):
                raise SystemExit(
                    f"!!! Sheath loop {i} is NOT nested inside loop {_outer} "
                    f"({n_in}/{len(P)} vertices inside).\n"
                    "    These are disjoint islands, not a shell, so there is no 'bore' to\n"
                    "    soften and --perivascular would put loose areolar tissue inside a\n"
                    "    solid limb of the fascia. Refusing rather than guessing.")
        BODIES["Perivascular"]["nu"] = (NU if BODIES["Perivascular"].get("nu") is None
                                        else BODIES["Perivascular"]["nu"])
        _peri_test = _polygons([loops[i] for i in _bore])
        print(f">>> 🎯 PERIVASCULAR COMPARTMENT ON. Sheath split at its own inner loop:")
        print(f"    fascial wall (outer - bore)  {(_areas[_outer] - sum(_areas[i] for i in _bore)) * 1e6:8.2f} mm²"
              f"   E = {BODIES['Sheath']['E_pa'] / 1e3:g} kPa   (--sheath-kpa)")
        print(f"    bore, before the vessels     {sum(_areas[i] for i in _bore) * 1e6:8.2f} mm²"
              f"   E = {BODIES['Perivascular']['E_pa'] / 1e3:g} kPa   "
              f"({BODIES['Perivascular']['E_src']})")
        print(f"    ⚠️ FEM ONLY. No OBJ is written for this body, so the B-mode still traces a "
              f"uniform sheath —")
        print(f"       the mechanics change and the image does not. Say so wherever this run "
              f"is quoted.")
    else:
        _peri_test = None
        print(">>> perivascular compartment OFF — the sheath bore is meshed SOLID at "
              f"{BODIES['Sheath']['E_pa'] / 1e3:g} kPa (as-built, leg A).")

    tests = []
    for name in doc["nest_order"]:
        d = B[name]
        if name == "Sheath" and _peri_test is not None:
            # BEFORE the envelope, so the bore wins inside it and `Sheath` keeps the wall.
            tests.append(("Perivascular", _peri_test))
        # A lumen is a VOID in the FEM -- there is no fluid in Newton. Transmural pressure,
        # if enabled, is a follower load on the wall ring instead (see _refresh_lumen_pressure).
        out = "void" if name in LEGITIMATE_VOIDS else name
        if d["kind"] == "disc":
            tests.append((out, _disc(d["c"], d["r"])))
        elif d["kind"] == "annulus":
            tests.append((out, _disc(d["c"], d["r_out"])))
        elif d["kind"] == "ellipse":
            tests.append((out, _ellipse(d["c"], d["a"], d["b"])))
        elif d["kind"] == "polygon":
            if name == "Sheath" and loose_cuff > 0.0:
                # Sheath envelope UNION a collar about each vessel, INTERSECTED with the
                # muscle core. Union rather than a polygon offset because the shortfall is
                # local to the anterior arc and an offset of a non-convex funnel is a
                # numerical exercise with no anatomical meaning; the collar is exactly the
                # "blends with the adventitia" the source describes.
                env = (_polygons(d["polys"]) if d.get("polys") else _polygon(d["pts"]))
                r_mu = B["Muscle"].get("r", B["Muscle"].get("r_out"))
                cuffs = [(tuple(B[w]["c"]), B[w]["r_out"] + loose_cuff)
                         for w in ("Artery", "Vein") if w in B]

                def _sheath_plus_collar(x, z, _env=env, _cuffs=cuffs, _rmu=r_mu):
                    if x * x + z * z >= _rmu * _rmu:
                        return False          # never eat into the subcutaneous fat shell
                    if _env(x, z):
                        return True
                    return any((x - cx) ** 2 + (z - cz) ** 2 < rr * rr
                               for (cx, cz), rr in _cuffs)
                tests.append((out, _sheath_plus_collar))
            else:
                tests.append((out, _polygons(d["polys"]) if d.get("polys")
                              else _polygon(d["pts"])))
        else:
            raise SystemExit(f"!!! {name}: unknown section kind {d['kind']!r}")
    SECTION = tests

    R_SKIN = B["SoftTissue"]["r_out"]
    R_MUSCLE = B["Muscle"].get("r", B["Muscle"].get("r_out"))
    ARTERY_C = tuple(B["Artery"]["c"])
    ARTERY_RO, ARTERY_RI = B["Artery"]["r_out"], B["BloodArtery"]["r"]
    VEIN_C = tuple(B["Vein"]["c"])
    VEIN_RO, VEIN_RI = B["Vein"]["r_out"], B["BloodVein"]["r"]
    if "Nerve" in B:
        NERVE_C = tuple(B["Nerve"]["c"])
        # The nerve is a RIBBON here, not a rod (Lonchena 9.6 x 3.7 mm). The pair tracker
        # measures its HEIGHT, so the semi-MINOR axis is the radius to hand it -- the
        # semi-major would hunt for surface nodes 4.8 mm above a structure 1.85 mm tall.
        NERVE_R = B["Nerve"]["b"]
    else:
        NERVE_C, NERVE_R = None, None

    # ROI: hold every structure present at this station plus free tissue either side of the
    # probe, and floor the block INSIDE the femur so the bone is what the press works against.
    xs, zs = [], []
    for name, d in B.items():
        if name in ("Muscle", "SoftTissue", "Femur"):
            continue
        if d["kind"] == "polygon":
            P = np.asarray(d["pts"])
            xs += [P[:, 0].min(), P[:, 0].max()]
            zs += [P[:, 1].min(), P[:, 1].max()]
        elif d["kind"] == "ellipse":
            xs += [d["c"][0] - d["a"], d["c"][0] + d["a"]]
            zs += [d["c"][1] - d["b"], d["c"][1] + d["b"]]
        else:
            rr = d.get("r_out", d.get("r"))
            xs += [d["c"][0] - rr, d["c"][0] + rr]
            zs += [d["c"][1] - rr, d["c"][1] + rr]
    # 🚨 THE ROI MUST CONTAIN THE PROBE, NOT JUST THE ANATOMY. This sized itself from the
    # structures alone on the first attempt and cost a whole diagnostic run: the footprint of
    # a 40 mm probe centred at x = 0 spans -20..+20 mm, while the anatomy-only ROI came out
    # at -12.35..+35.75, so the face OVERHUNG THE PINNED LATERAL FACE BY 7.65 mm and was
    # pressing straight onto clamped nodes. The measured plant was ~830 N/mm where E*A/h
    # predicts ~0.09 N/mm -- about 9000x too stiff -- and the probe could not get past
    # 2.10 mm of depth at 181 N. V2 never showed it because its hand-picked -24..+27 mm block
    # happened to clear the same probe by 4 mm. Nothing raised; the run simply reported a
    # rigid phantom, which is indistinguishable from a real stiffness result.
    pad = 0.008
    lo = min(list(xs) + [-probe_half_w - probe_margin])
    hi = max(list(xs) + [probe_half_w + probe_margin])
    X0, X1 = float(lo - pad), float(hi + pad)
    # 🪤 THE FEMUR IS NOT ALWAYS A DISC. This read `fem["c"][1] + fem["r"]` unconditionally,
    # which is only true where the bone sections as a circle. At s = 6 mm -- the first station
    # that has the femoral canal in it -- the femur sections as a POLYGON (131 points,
    # X -23.32..+11.70) because the shaft is flaring into the neck, and the sweep died with
    # KeyError: 'c' on all four trials. The quantity actually wanted is the TOP of the bone,
    # whatever shape it is: the ROI floor goes 2 mm INSIDE the femur so the bone is what the
    # press works against rather than a clamped plane in front of it.
    fem = B.get("Femur")
    if fem is None:
        Z0 = float(min(zs) - pad)
    else:
        k = fem["kind"]
        if k == "disc":
            femur_top = fem["c"][1] + fem["r"]
        elif k == "annulus":
            femur_top = fem["c"][1] + fem["r_out"]
        elif k == "ellipse":
            femur_top = fem["c"][1] + fem["b"]
        elif k == "polygon":
            _pts = (np.vstack([np.asarray(P) for P in fem["polys"]])
                    if fem.get("polys") else np.asarray(fem["pts"]))
            femur_top = float(_pts[:, 1].max())
        else:
            raise SystemExit(f"!!! Femur has section kind {k!r} and the ROI floor does not "
                             f"know how to find its top. Refusing rather than guessing a "
                             f"floor -- the floor is the backstop the whole press works "
                             f"against.")
        Z0 = float(femur_top - 0.002)
    return doc


RHO_DEFAULT = 1050.0
NU = 0.45


def lame(E: float, nu: float):
    return E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


# ── probe angle around the limb ──────────────────────────────────────────────────────────
# THETA is the contact angle: 0 = the ANTERIOR POLE, positive toward the patient's left
# (FEM +X), which is the side the vein is on.
#
# 🔗 THIS IS THE SAME CONVENTION sweep_circumference.py USES, deliberately. One angle, one
# meaning, on both halves of the simulator -- so a swept FEM solve and a swept B-mode render
# at "theta = +12.2 deg" are the same pose, and the deformed meshes can be handed straight
# from one to the other. Chapter 5's standing lesson was two artifacts with nothing keeping
# them consistent; this is the parameter that would have caused it again.
#
# HOW THE ROTATION IS APPLIED. The FEM block does NOT move. It stays a window of tissue with
# the probe pressing down its +Z, its sides pinned, and its base pinned -- so the probe, the
# driven joint, the force estimator and the pinning are all untouched by theta. What rotates
# is the ANATOMY within the window. The skin is a circle about the limb axis, so it maps to
# itself under this rotation and the block's top surface is unchanged; only the internal
# structures move through the window as theta sweeps.
#
# The consequence to expect, and it is physically right: past roughly +/-20 deg the vessels
# leave the window entirely, because they sit ~70 mm out from the axis and the window is
# ~50 mm wide. At those angles the probe is pressing on plain muscle and there is nothing to
# collapse. The lumen trackers below therefore SKIP structures that are out of the window
# rather than failing -- an empty result at 45 deg is the correct answer, not an error.
THETA = 0.0

# Whether VBD's particle self-contact is enabled, and at what radius (in units of dx).
#
# 📌 WHY THIS IS A KNOB AND NOT A CONSTANT, 2026-08-08. Self-contact exists here for exactly
# one reason -- opposing lumen walls must be able to STOP each other, or a "collapsed" vessel
# is two walls passing through one another (see Phantom.__init__ and closure_floor_pct). It
# is a lumen feature. But it is switched on for the WHOLE mesh, and newton's VBD implements
# it as vertex-triangle / edge-edge contact on the SURFACE mesh with a topological filter of
# 2 rings -- so it can also fire between surface primitives that are merely close, at a
# penalty stiffness of soft_contact_ke = 2e5 N/m EACH. On a stair-stepped voxel skin with
# 1.5 mm steps and a 0.6 mm contact radius that is a parasitic scaffold with no physics in it,
# and it would read as tissue stiffness. It has never been isolated. A homogeneous block has
# no lumens, so it is the configuration where self-contact has NO legitimate job at all and
# turning it off is a clean single-variable test -- which is what --self-contact 0 is for.
# ⚠️ Do NOT turn it off for a lumen run to make closure look better; that would be trading a
# measurement for an artefact in the other direction.
SELF_CONTACT = True
SELF_CONTACT_R_SCALE = 0.4

# VBD solver iterations per substep.
#
# 🚨 WHY THIS IS A KNOB AS OF 2026-08-08 (evening). It was hard-coded at 10 and never varied.
# VBD is a block-coordinate-descent (Gauss-Seidel-like) solve, so information propagates roughly
# ONE NODE PER ITERATION. That makes 10 a statement about the MESH, not about accuracy: halving
# dx doubles the node count a boundary condition has to cross, at an unchanged iteration budget,
# so the same physical problem is solved LESS converged on the finer grid -- and an
# under-converged implicit solve reads STIFF.
# 📌 That is the shape of the measurement that put this here: the identical homogeneous 20 kPa
# block, identical probe, identical BCs, read 0.478 N/mm at dx = 3.0 mm and ~150 N/mm at
# dx = 1.5 mm, against a closed form of 0.62. A converging FEM does not do that, and the
# direction (finer = stiffer) is the opposite of volumetric locking, which was the ranked
# leading suspect. See OPEN_ITEMS 17 and diag_block_control.py.
# ⚠️ Raising it is NOT a free fix. Cost is linear in iterations, and it must be shown to
# CONVERGE -- k falling and then flattening as iterations rise -- not merely to fall to a value
# that happens to match the closed form. A number tuned until it agrees is the one move this
# project does not make.
VBD_ITERATIONS = 10

# Whether the ROI's lateral faces are clamped. See the note in build().
#
# ⚠️ THE CLAIM THAT USED TO SIT HERE ("this was True and is the reason the vein could not
# collapse") IS REFUTED AND HAS BEEN REMOVED. force_ladder.py ran the experiment: freeing the
# sides moved the vein from 87.8 % to 92.0 % of rest area, i.e. it made collapse WORSE, not
# better -- lateral confinement concentrates load rather than resisting it. Left as a knob
# (--pin-sides) because it is a legitimate modelling choice, not because it is a suspect.
PIN_SIDES = True

# ── rigid inclusions: bodies too stiff to integrate, and never meant to deform ───────────
# 🚨 THE FEMUR MAKES THIS PHANTOM UNINTEGRABLE, AND NOTHING NOTICED FOR TWO DAYS.
# V3 added a cortical-bone rod at E = 17 GPa. Against dx = 1.5 mm that is c_p = 3470 m/s and
# dt_max = 0.4 us, so at 384 substeps (dt = 43.4 us) the run sits **100.42x OVER** the
# stability limit -- cfl_report prints "force numbers are NOT trustworthy" and asks for 38562
# substeps, which is 100x the cost and flatly infeasible. Every FEM force number taken since
# the femur landed was computed in that state.
#
# The fix is not more substeps, it is noticing that a 17 GPa body IS NOT A MATERIAL HERE.
# materials.py says so itself: "six orders above the soft tissue around it, so in practice it
# is a rigid backstop rather than a material -- which is the entire point of adding it".
# Simulating a rigid backstop as a deformable solid buys nothing physical (bone does not
# measurably indent under a 10 N probe) and costs the entire timestep budget.
#
# Pinning its nodes makes it what it was always supposed to be: an infinitely stiff, immovable
# boundary. That is MORE faithful than a 17 GPa FEM body the solver cannot converge, and it
# removes the femur from the stiffness spectrum entirely, so the limit falls back to the
# artery. ⚠️ The artery is still 1.32x over at 384 substeps -- see cfl_report's docstring --
# so this fix alone does not make the run legal; it makes it fixable by raising substeps to a
# number that is affordable (>= 507) instead of one that is not.
RIGID_E_PA = 1.0e9

# How a cell that is PARTLY a rigid inclusion is converted to a material.
#
# 🚨 "exclude" IS A DEFECT FIX, MEASURED 2026-08-11, AND "legacy" REPRODUCES THE BUG.
# RIGID_E_PA above pins the nodes of every cell whose DOMINANT body is the femur. It says
# nothing about a cell that is 40 % bone and 60 % muscle -- that cell's dominant body is
# Muscle, so it is never pinned, and cell_material's arithmetic Lame average hands it a
# modulus of order (bone fraction) x 17 GPa. Measured on the V3 leg at s = 30 mm:
#
#     dx 1.5 mm   924 of 159,786 tets at E >= 1 MPa, 462 of them >= 10 GPa, max 17 GPa
#     dx 3.0 mm   150 of  18,180 tets at E >= 1 MPa,  30 of them >= 10 GPa, max 10.2 GPa
#
# Those are DEFORMABLE elements with free nodes, and at the production 512 substeps the worst
# of them sits **101x over its own CFL limit** (dt_max 0.321 us against dt 32.6 us). Delete
# them and the worst element in the mesh is 1.01x -- which is the number cfl_report believed
# it was reporting. It could not see them because they are LABELLED "Muscle": the report reads
# the material CATALOGUE for the bodies present, never the per-tet moduli the homogenisation
# actually produced. Same defect shape as the 2026-08-06 "cfl_report checked the catalogue,
# not the mesh" item, one layer deeper -- it learned to check WHICH bodies are meshed and
# still never checked WHAT modulus they were meshed at.
#
# 📌 The count and the peak modulus BOTH change with dx, so the two resolutions do not contain
# the same femur. Any dx comparison taken with "legacy" is confounded by that.
#
# "exclude" makes the femur what materials.py already says it is -- "a rigid backstop rather
# than a material". A cell more than half bone becomes a pinned bone cell; a cell less than
# half bone drops the bone sub-samples from the average and is meshed as the soft tissue that
# fills the rest of it. The bone boundary is then a staircase of pinned nodes, at the same
# ">half" convention cell_material already uses to decide void against solid.
# ⚠️ This is a MODELLING change as well as a numerical one: it quantises the bone surface to
# the grid. That is the intended behaviour of a pinned backstop and it is stated here so the
# A/B against "legacy" is one variable and is on the record.
RIGID_CELL_POLICY = "exclude"

# ── transmural (blood) pressure on the lumen walls ───────────────────────────────────────
# Pa, per lumen. 0.0 = the pre-2026-08-06 behaviour: bare voids, no fluid, so the artery's
# only defence is its wall modulus. See Phantom._refresh_lumen_pressure for the mechanics
# and main()'s --artery-kpa / --vein-kpa for the switch.
#
# 🔴 PROVENANCE: MODELLING CHOICE, NOTHING ON DISK. These are textbook mean pressures
# (artery ~90-100 mmHg -> 12-13 kPa; vein ~5-10 mmHg -> 0.7-1.3 kPa). No paper in
# i4h-references reports either. They must be recorded in sources_needed.md §1 alongside the
# E_pa values and must NOT be quoted as sourced. Defaults below are therefore OFF.
LUMEN_P_DEFAULT = {"artery": 0.0, "vein": 0.0}


def to_limb(x: float, z: float):
    """Block coordinates -> limb coordinates. Rotation by THETA about the limb axis."""
    c, s = np.cos(THETA), np.sin(THETA)
    return x * c + z * s, -x * s + z * c


def to_block(X: float, Z: float):
    """Limb coordinates -> block coordinates. The inverse of to_limb."""
    c, s = np.cos(THETA), np.sin(THETA)
    return X * c - Z * s, X * s + Z * c


def in_window(X: float, Z: float, margin: float = 0.0) -> bool:
    """Is a limb-frame structure inside the simulated block at the current THETA?"""
    x, z = to_block(X, Z)
    return (X0 + margin) <= x <= (X1 - margin) and (Z0 + margin) <= z <= R_SKIN


def classify(x: float, z: float) -> str:
    """Innermost-wins point classification -- the same rule the mesh nesting encodes.

    Takes BLOCK coordinates and rotates them into the limb frame, so every circle test below
    is written against the anatomy's own coordinates exactly as before.

    Returns a body name from materials.py, or 'void' for a lumen or outside the limb.
    """
    x, z = to_limb(x, z)
    global _V2_CACHE
    sec = SECTION
    if sec is None:
        if _V2_CACHE is None:
            _V2_CACHE = _v2_section()
        sec = _V2_CACHE
    for name, contains in sec:
        if contains(x, z):
            return name
    return "void"


_V2_CACHE = None


def cell_material(x0: float, z0: float, dx: float, nsub: int = 5):
    """Volume-fraction material for one cell, by sub-sampling its XZ footprint.

    Cell-centre sampling is not good enough here. The artery wall is 1.02 mm thick, so at
    any practical dx a wall cell is only partly wall -- centre sampling either drops the
    stiff wall entirely (the artery then behaves like a plain hole and collapses MORE than
    the vein, which is what a first coarse run actually did) or over-weights it. Averaging
    the Lame parameters over the cell's sub-samples is standard homogenisation and lets the
    wall survive at a mesh size we can afford.

    Returns (mu, lam, rho, dominant_body) or None when the cell is more than half void.
    """
    xs = x0 + (np.arange(nsub) + 0.5) * dx / nsub
    zs = z0 + (np.arange(nsub) + 0.5) * dx / nsub
    names = [classify(x, z) for x in xs for z in zs]
    solid = [n for n in names if n != "void"]
    if len(solid) * 2 <= len(names):
        return None

    # 🚨 A RIGID INCLUSION MUST NOT BE AVERAGED INTO A SOFT ONE. See RIGID_CELL_POLICY for the
    # measurement that put this here: without it, one bone sub-sample in twenty-five hands the
    # cell a modulus ~34,000x muscle's, the cell is labelled by its soft dominant body, it is
    # never pinned, and no instrument in this file can see it.
    dom_pool = solid
    if RIGID_CELL_POLICY == "exclude":
        rigid = [n for n in solid
                 if BODIES[n].get("E_pa") is not None and BODIES[n]["E_pa"] >= RIGID_E_PA]
        soft = [n for n in solid if n not in set(rigid)]
        if rigid and (len(rigid) * 2 > len(solid) or not soft):
            # Mostly bone: name it after the bone so build() pins every one of its nodes, and
            # give it a SOFT modulus. Every node of this cell is pinned, so its material never
            # acts -- but it would otherwise sit in model.tet_materials and set the stability
            # limit for a wave that cannot propagate through immovable nodes.
            dom = max(set(rigid), key=rigid.count)
            fill = max(set(soft), key=soft.count) if soft else "Muscle"
            b = BODIES[fill]
            mu, lam = lame(b["E_pa"], b["nu"])
            return float(mu), float(lam), float(b["rho"] or RHO_DEFAULT), dom
        if soft:
            solid, dom_pool = soft, soft      # drop the bone sub-samples from the average

    mus, lams, rhos = [], [], []
    for n in solid:
        b = BODIES[n]
        mu, lam = lame(b["E_pa"], b["nu"])
        mus.append(mu)
        lams.append(lam)
        rhos.append(b["rho"] or RHO_DEFAULT)
    dom = max(set(dom_pool), key=dom_pool.count)
    return float(np.mean(mus)), float(np.mean(lams)), float(np.mean(rhos)), dom


TETS = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7),
        (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]
CORNER = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
          (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]


def tet_volumes(v: np.ndarray, ti: np.ndarray) -> np.ndarray:
    """SIGNED volume of every tet, m^3. `v` is (n_node, 3) positions, `ti` is (n_tet, 4).

    📌 ONE OPERATOR FOR REST AND FOR DEFORMED, DELIBERATELY. build()'s rest-mesh assert and
    Phantom.mesh_quality()'s per-frame witness are the same three lines of arithmetic, and the
    moment there are two copies of "is this element inverted" they will disagree at exactly the
    rung where it matters. That is defect class 1 (two artifacts and nothing reconciles them),
    which is the failure mode this project loses the most time to.

    Signed, not absolute: the sign IS the measurement. `abs()` here would turn an inverted
    element into a healthy-looking one, which is precisely the blindness this function exists
    to remove.
    """
    return np.einsum("ij,ij->i",
                     np.cross(v[ti[:, 1]] - v[ti[:, 0]], v[ti[:, 2]] - v[ti[:, 0]]),
                     v[ti[:, 3]] - v[ti[:, 0]]) / 6.0


# ── vessel-wall shell support (2026-08-13) ───────────────────────────────────────────────
# Populated by the CALLER, never here, and empty means "no shell" -- the same pattern as
# BODIES["Sheath"]["E_pa"]. A default would put an unowned modelling choice underneath every
# lumen in the project.
#   SHELL = {"vein": dict(E_pa=..., t_m=..., nu=...)}
SHELL: dict = {}

# 🩸 Blood in the lumen as SOLID elements. Empty = void (the default). Set by the caller:
#   LUMEN_FILL = {"vein": dict(E_pa=..., nu=...)}
# See load_section for what this is testing and why the default is what it is.
LUMEN_FILL: dict = {}


def ring_buckling_pa(E_pa: float, t_m: float, R_m: float, nu: float) -> float:
    """Critical external pressure for a long thin cylinder to buckle (Levy).

        p_cr = E t^3 / (4 R^3 (1 - nu^2))

    🎯 THIS IS THE CALIBRATION TARGET FOR THE SHELL, and it is a closed form rather than a
    fitted constant -- the same footing as the patch test that verifies the F_tissue operator.
    An isolated ring (no surrounding tissue) driven with `set_lumen_pressure` must lose
    circularity at this pressure. If it does not, `edge_ke` is wrong and no tube law computed
    with it may be quoted.

    📌 It also puts the whole problem in scale. At E = 20 kPa, t = 0.50 mm, R = 5.24 mm this is
    ~5.4 Pa = 0.056 cmH2O, while Gero 1968 measures the vein as elliptical anywhere below
    16 cmH2O = 1569 Pa. So a real vein sits 300-500x PAST its buckling threshold across the
    entire physiological range: its compliance there is shape change, not stretch. A model with
    no buckling is stuck in the pre-buckled regime everywhere, which is exactly the flat
    Dmax/Dmin the 2026-08-13 tube law measured.
    """
    return E_pa * t_m ** 3 / (4.0 * R_m ** 3 * (1.0 - nu * nu))


def shell_stiffness(E_pa: float, t_m: float, nu: float, L_edge: float):
    """Map a physical (E, t, nu) wall onto newton's triangle and bending parameters.

    Returns (tri_ke, tri_ka, edge_ke).

    MEMBRANE. tri_ke/tri_ka carry in-plane stretch and area change; the plate values are
    E*t/(1 - nu^2) and its nu-weighted partner. These matter far less than bending here: a
    buckling vein barely stretches, it folds.

    BENDING, and this is the number that decides everything. newton's dihedral element stores
        E_edge = 0.5 * edge_ke * L_rest * (theta - theta_0)^2
    (particle_vbd_kernels.py:665-724, `k = stiffness * edge_rest_length`). Matching that to the
    Kirchhoff-Love plate energy 0.5 * D * kappa^2 per unit area, with D = E t^3/(12(1-nu^2)),
    on a triangulated surface of edge length L: theta ~= kappa * L, and an equilateral mesh
    carries ~6/(sqrt(3) L^2) edges per unit area, so

        0.5 * (6/(sqrt(3) L^2)) * edge_ke * L * (kappa L)^2  ==  0.5 * D * kappa^2
        =>  edge_ke = D * sqrt(3) / (6 * L)

    🚨 THAT DERIVATION ASSUMES AN EQUILATERAL MESH AND A SMALL-ANGLE DIHEDRAL, AND OUR RING IS
    NEITHER -- 16 circumferential bins on a voxel grid gives a quad mesh split into right
    triangles with two edge lengths, and a collapsing vein leaves small angles immediately. So
    this is an ESTIMATE, not a calibration, and it is tagged as one everywhere it is printed.
    🎯 The isolated-ring run against ring_buckling_pa() is what turns it into a measurement, and
    until that run exists no shell result may be quoted. Do NOT close the gap by scaling
    edge_ke until the tube law matches Gero -- that is fitting the answer, and this project's
    standing rule names it as the most expensive error available.
    """
    D = E_pa * t_m ** 3 / (12.0 * (1.0 - nu * nu))
    edge_ke = D * math.sqrt(3.0) / (6.0 * max(L_edge, 1e-12))
    membrane = E_pa * t_m / (1.0 - nu * nu)
    return membrane, membrane * nu, edge_ke


def _ring_ik(idx, dx, nx, nz, j, cx, cz, ri, nbin: int = 16):
    """The ordered (i, k) lumen-boundary cells at one y-plane, or None if unresolved.

    📌 Factored out of Phantom._lumen_rings on 2026-08-13 so the SHELL built at build time and
    the rings measured at run time come from ONE implementation. Two copies of this walk would
    be defect class 1 -- a shell hinged on one set of nodes while areas() measured another, and
    both would look entirely plausible.
    """
    best = {}
    for i in range(nx + 1):
        for k in range(nz + 1):
            p = idx[i, j, k]
            if p < 0:
                continue
            x, z = X0 + i * dx, Z0 + k * dx
            r = np.hypot(x - cx, z - cz)
            if r < ri or r > ri + 2.2 * dx:
                continue
            b = int(((np.arctan2(z - cz, x - cx) + np.pi) / (2 * np.pi)) * nbin) % nbin
            if b not in best or r < best[b][0]:
                best[b] = (r, i, k)
    if len(best) < 6:
        return None
    return [(i, k) for _, i, k in
            sorted(best.values(),
                   key=lambda t: np.arctan2(Z0 + t[2] * dx - cz, X0 + t[1] * dx - cx))]


def build(dx: float, probe_w: float, probe_len: float):
    # 🚨 THE PROBE MUST NOT REACH THE PINNED LATERAL FACES. Refuse rather than simulate.
    # A probe overhanging a clamped boundary presses on immovable nodes, and the run then
    # reports a plant three to four orders of magnitude too stiff while looking entirely
    # normal: monotonic force, converging settles, lumens politely at 100 %. That is this
    # project's signature failure -- an instrument returning a plausible value for something
    # it is not measuring -- and on 2026-08-08 it burned a full engagement ladder before the
    # arithmetic (830 N/mm measured vs 0.09 N/mm predicted) gave it away. One assert is
    # cheaper than that diagnosis, every time.
    if PIN_SIDES:
        half = probe_w / 2.0
        clear_lo = (0.0 - half) - X0        # x_probe is 0.0; see build()'s x_probe below
        clear_hi = X1 - (0.0 + half)
        need = 2.0 * dx
        if clear_lo < need or clear_hi < need:
            raise SystemExit(
                f"!!! the probe footprint reaches the PINNED lateral faces.\n"
                f"    ROI   X [{X0*1e3:+.2f}, {X1*1e3:+.2f}] mm, sides clamped\n"
                f"    probe {probe_w*1e3:.1f} mm wide at x = 0 -> [{-half*1e3:+.2f}, "
                f"{half*1e3:+.2f}] mm\n"
                f"    clearance: {clear_lo*1e3:+.2f} mm left, {clear_hi*1e3:+.2f} mm right; "
                f"need >= {need*1e3:.2f} mm (2*dx) on both\n"
                f"    Pressing on clamped nodes yields a plant orders of magnitude too stiff\n"
                f"    and a probe that cannot indent -- and NOTHING about the output looks\n"
                f"    wrong. Widen the ROI, narrow the probe, or run with --pin-sides 0.")
    nx = int(round((X1 - X0) / dx))
    ny = int(round(LY / dx))
    # top of the grid must clear the skin apex
    nz = int(round((R_SKIN + 2 * dx - Z0) / dx))

    def xc(i):
        return X0 + (i + 0.5) * dx

    def zc(k):
        return Z0 + (k + 0.5) * dx

    body_of = np.empty((nx, nz), dtype=object)
    cell_mat = {}
    for i in range(nx):
        for k in range(nz):
            cm = cell_material(X0 + i * dx, Z0 + k * dx, dx)
            if cm is None:
                body_of[i, k] = "void"
            else:
                cell_mat[(i, k)] = cm
                body_of[i, k] = cm[3]
    solid2d = body_of != "void"

    used = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    for i in range(nx):
        for k in range(nz):
            if solid2d[i, k]:
                used[i:i + 2, :, k:k + 2] = True

    idx = -np.ones((nx + 1, ny + 1, nz + 1), dtype=np.int64)
    verts = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                if not used[i, j, k]:
                    continue
                idx[i, j, k] = len(verts)
                verts.append((X0 + i * dx, -LY / 2 + j * dx, Z0 + k * dx))

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    tets, k_mu, k_lam, dens, tet_body = [], [], [], [], []
    for i in range(nx):
        for k in range(nz):
            if not solid2d[i, k]:
                continue
            mu, lam, rho, name = cell_mat[(i, k)]
            for j in range(ny):
                c = [idx[i + a, j + bb, k + cc] for a, bb, cc in CORNER]
                assert min(c) >= 0
                for t in TETS:
                    tets.extend((c[t[0]], c[t[1]], c[t[2]], c[t[3]]))
                    k_mu.append(mu)
                    k_lam.append(lam)
                    dens.append(rho)
                    tet_body.append(name)

    v = np.asarray(verts, dtype=np.float64)
    ti = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
    # ⚠️ THIS IS A REST-MESH CHECK AND IT SAYS NOTHING ABOUT THE RUN. It fires once, on the
    # undeformed grid, where every tet is dx^3/6 by construction and cannot possibly be
    # inverted. Five press ladders were driven to 24 mm of indentation behind this assert with
    # negative-volume elements sitting in the SETTLED state from 9 mm down and nothing printing
    # a thing -- see Phantom.mesh_quality(), which is the deformed-mesh counterpart and the one
    # that actually watches the run.
    vol = tet_volumes(v, ti)
    assert (vol > 0).all(), f"{(vol <= 0).sum()} inverted tets"

    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(*p) for p in verts], indices=tets,
        density=float(np.mean(dens)),
        k_mu=np.array(k_mu, dtype=np.float32),
        k_lambda=np.array(k_lam, dtype=np.float32),
        k_damp=1.0,
        particle_radius=0.25 * dx,
        validate_mesh=True,
        label="femoral_roi",
    )

    # Pin the deep face (Z0), and OPTIONALLY the two lateral faces (x = X0, x = X1).
    #
    # ⚠️ A REFUTED HYPOTHESIS USED TO BE ARGUED HERE AT LENGTH -- that pinning the sides made
    # this a confined compression test and was why the vein would not close. force_ladder.py
    # tested it: freeing the sides moved the vein 87.8 % -> 92.0 %, i.e. the WRONG WAY.
    # Confinement concentrates load into the bundle; removing it lets the block relieve
    # itself sideways and the vein closes LESS. The boundary condition is not the blocker.
    #
    # The thin-shell buckling estimate that motivated it (4.1 Pa critical pressure at
    # t/R = 0.087, E = 20 kPa) is also the wrong model: it assumes a tube in a FLUID under
    # uniform external pressure. This lumen is a void in a SOLID, so closing it means
    # displacing its area into a nu = 0.45 medium. The governing pressure is then of order
    # E (~20 kPa), not E(t/R)^3. Probe pressure at 10 N over 25x12 mm is ~33 kPa -- the same
    # order as E -- so PARTIAL collapse is the correct answer for a 20 kPa perivascular
    # medium, and no boundary condition or solver setting will turn it into closure.
    #
    # ⚠️ NOTE WHAT Z0 ALREADY IS. Z0 = 56.0 mm and the vein's far wall is at 61.66 mm, so
    # this model already has a FULLY RIGID, FULLY CLAMPED backstop 5.66 mm behind the vein.
    # That is a far better backstop than V3's femur, whose near surface is 52.5 mm behind the
    # vein's far wall. The vein still stops at ~80 %. Adding the femur therefore cannot be
    # the fix that V3_PARAMETRIC_SPEC.md §4 hopes it is -- that experiment has effectively
    # already been run, by accident, and it failed.
    pinned = set()
    for i in range(nx + 1):
        for j in range(ny + 1):
            if idx[i, j, 0] >= 0:
                pinned.add(int(idx[i, j, 0]))
    if PIN_SIDES:
        for j in range(ny + 1):
            for k in range(nz + 1):
                for i in (0, nx):
                    if idx[i, j, k] >= 0:
                        pinned.add(int(idx[i, j, k]))

    # Rigid inclusions (see RIGID_E_PA): pin every node of any body stiffer than the
    # threshold, so it acts as an immovable backstop instead of a 17 GPa body that the
    # explicit substep cannot integrate and VBD's 10 iterations cannot converge.
    _rigid_names = {n for n, b in BODIES.items()
                    if b.get("E_pa") is not None and b["E_pa"] >= RIGID_E_PA}
    n_rigid_nodes = 0
    if _rigid_names:
        _t = np.asarray(tets, dtype=np.int64).reshape(-1, 4)
        for _ti, _name in enumerate(tet_body):
            if _name in _rigid_names:
                for _v in _t[_ti]:
                    if int(_v) not in pinned:
                        pinned.add(int(_v))
                        n_rigid_nodes += 1
        print(f">>> rigid inclusions {sorted(_rigid_names)}: pinned {n_rigid_nodes} extra "
              f"node(s). They are backstops, not materials -- see RIGID_E_PA.", flush=True)

    # Probe: a flat-faced rigid box on a driven prismatic joint, long axis across the
    # vessels -- how a linear probe is held for transverse compression.
    # The probe always sits at the block's centre line, pressing radially inward -- that is
    # what makes THETA free: the pose is baked into the anatomy's rotation, not into the
    # probe's placement, so the joint, the pinning and the force estimator never change.
    #
    # ⚠️ This moves the theta = 0 pose. The previous hard-coded probe sat at the artery/vein
    # MIDPOINT (x = 7.5 mm), which in the shared convention is theta = +6.05 deg, not 0. Any
    # number carried over from a run before 2026-08-04 was measured at that angle -- pass
    # --probe-theta 6.05 to reproduce it.
    x_probe = 0.0
    z_skin = R_SKIN
    # BOTH OF THESE COME FROM probe_geometry.py AND NOTHING ELSE. The standoff is the
    # datum `past_skin` is measured from, and it also has to sit inside the gel band or
    # every B-mode renders black (OPEN_ITEMS 20). Four independent copies existed before
    # 2026-08-11, one of them in the reporting path.
    half = (probe_w / 2, probe_len / 2, PG.PROBE_HALF_THICKNESS_M)
    start_z = z_skin + half[2] + PG.PROBE_STANDOFF_M
    link = builder.add_link(xform=wp.transform(wp.vec3(x_probe, 0.0, start_z), wp.quat_identity()),
                            mass=0.2)
    builder.add_shape_box(body=link, hx=half[0], hy=half[1], hz=half[2])
    builder.add_joint_prismatic(
        parent=-1, child=link, axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(x_probe, 0.0, start_z), wp.quat_identity()),
        target_pos=0.0, target_ke=2.0e5, target_kd=2.0e3,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    # ── 🎯 THE VESSEL WALL AS A SHELL, added 2026-08-13 ──────────────────────────────────
    # WHY THIS EXISTS. A vein coapts by BUCKLING -- circular -> elliptical -> slit -- and
    # buckling is resisted by BENDING stiffness. In a tet continuum bending is carried by
    # resolving the thickness, and this wall is 0.50 mm against dx 1.50 mm, i.e. 0.33 of one
    # element. Measured 2026-08-13 (diag_tube_law.py, no probe, no contact): over 2700 -> 850
    # Pa the lumen's Dmax/Dmin moved 1.018 -> 1.028, while Gero 1968's dog femoral vein moves
    # 1.03 -> 1.18 over the same band. The section changes AREA and not SHAPE, which is a void
    # being squeezed rather than a tube collapsing.
    #
    # 🎯 AND THE SAMPLING ARGUMENT IS THE WRONG FRAME, which is the point of doing it this way.
    # "Two elements through the wall" is a Nyquist-style requirement and it only binds while
    # the thickness is a SAMPLED dimension. In a shell it is a PARAMETER -- membrane stiffness
    # goes as E*t, bending as E*t^3/(12(1-nu^2)) -- so a surface element carries the wall's
    # bending correctly at ANY dx. There is nothing left to sample, so the requirement dissolves.
    #
    # 📌 NO NEW DEGREES OF FREEDOM AND NO ATTACHMENT CONSTRAINT. newton's add_triangle/add_edge
    # take PARTICLE INDICES, so the shell is built on the particles that already bound the
    # lumen and the wall and the tissue share nodes by construction. That removes the one thing
    # that usually goes wrong when a shell is coupled to a continuum.
    #
    # ⚠️ NO DOUBLE COUNT -- AND THAT IS AN ACCIDENT OF THE CURRENT MATERIALS, NOT A DESIGN.
    # BODIES["Vein"]["E_pa"] is 20 kPa, IDENTICAL to Muscle and Sheath, so there is no distinct
    # wall in the tets for the shell to duplicate. 🚨 IF THE VEIN TET BAND IS EVER GIVEN ITS OWN
    # MODULUS THIS BECOMES A DOUBLE COUNT, and the band must be set to the surrounding value.
    if SHELL:
        for name, spec in SHELL.items():
            C, ri = {"vein": (VEIN_C, VEIN_RI), "artery": (ARTERY_C, ARTERY_RI)}[name]
            if not in_window(*C, margin=ri + 2.2 * dx):
                continue
            ik = _ring_ik(idx, dx, nx, nz, ny // 2, *to_block(*C), ri)
            if ik is None:
                raise SystemExit(f"!!! shell requested for {name} but its ring is unresolved at "
                                 f"dx = {dx*1e3:.2f} mm -- refusing to build a PARTIAL shell, "
                                 f"which would bend in some directions and not others")
            V = np.array([[idx[i, jj, k] for i, k in ik] for jj in range(ny + 1)])
            nj, nb = V.shape
            L = float(np.mean([np.linalg.norm(np.asarray(verts[V[ny // 2, (b + 1) % nb]])
                                              - np.asarray(verts[V[ny // 2, b]]))
                               for b in range(nb)]))
            tri_ke, tri_ka, edge_ke = shell_stiffness(spec["E_pa"], spec["t_m"], spec["nu"], L)
            ti, tj, tk, ei, ej, ek, el = [], [], [], [], [], [], []
            for jj in range(nj - 1):
                for b in range(nb):
                    a, bb = V[jj, b], V[jj, (b + 1) % nb]
                    c, d = V[jj + 1, (b + 1) % nb], V[jj + 1, b]
                    ti += [a, a]; tj += [bb, c]; tk += [c, d]
                    # Bending across the shared diagonal a-c; opposites are bb and d. This
                    # hinge measures CIRCUMFERENTIAL curvature, which is the mode a collapsing
                    # vein buckles in, so it is the one that has to be right.
                    ei.append(bb); ej.append(d); ek.append(a); el.append(c)
            builder.add_triangles(ti, tj, tk, tri_ke=[tri_ke] * len(ti),
                                  tri_ka=[tri_ka] * len(ti),
                                  tri_kd=[spec.get("tri_kd", 1.0e-4)] * len(ti))
            builder.add_edges(ei, ej, ek, el, edge_ke=[edge_ke] * len(ei),
                              edge_kd=[spec.get("edge_kd", 1.0e-6)] * len(ei))
            p_cr = ring_buckling_pa(spec["E_pa"], spec["t_m"], ri, spec["nu"])
            print(f">>> SHELL on {name}: {len(ti)} triangles + {len(ei)} bending edges on "
                  f"{V.size} EXISTING nodes (no new DOFs, no attachment)")
            print(f"    E {spec['E_pa']/1e3:.1f} kPa · t {spec['t_m']*1e3:.2f} mm · mean edge "
                  f"{L*1e3:.2f} mm  ->  tri_ke {tri_ke:.4g}  edge_ke {edge_ke:.4g}")
            print(f"    🔬 closed-form ring buckling p_cr = E t³/(4 R³(1-ν²)) = {p_cr:.2f} Pa "
                  f"({p_cr/98.0665:.3f} cmH₂O)")
            print("    ⚠️ edge_ke is DERIVED, NOT CALIBRATED. Until an isolated-ring run "
                  "reproduces that\n       p_cr, every shell number is UNVERIFIED — see "
                  "shell_stiffness().")

    builder.color()
    model = builder.finalize()

    m_arr = model.particle_mass.numpy()
    inv = model.particle_inv_mass.numpy()
    for p in pinned:
        m_arr[p] = 0.0
        inv[p] = 0.0          # BOTH -- zeroing only inv_mass does not pin under VBD
    model.particle_mass.assign(m_arr)
    model.particle_inv_mass.assign(inv)

    model.soft_contact_ke = 2.0e5
    model.soft_contact_kd = 1.0e2
    model.soft_contact_mu = 0.5

    meta = dict(nx=nx, ny=ny, nz=nz, dx=dx, X0=X0, LY=LY, Z0=Z0,
                rho_mesh=float(np.mean(dens)), rigid_cell_policy=RIGID_CELL_POLICY,
                n_nodes=len(verts), n_tets=len(k_mu), n_pinned=len(pinned),
                # Which materials actually carry elements. cfl_report needs this to avoid
                # setting the stability limit from a body that is in the catalogue but not in
                # the mesh -- see the note at the top of that function.
                bodies_present=sorted(set(tet_body)),
                n_rigid_pinned=int(n_rigid_nodes),
                x_probe=float(x_probe), start_z=float(start_z), z_skin=float(z_skin),
                probe_standoff_m=PG.PROBE_STANDOFF_M,
                probe_half_thickness_m=PG.PROBE_HALF_THICKNESS_M)
    return model, idx, link, meta, np.asarray(verts), body_of, solid2d


def cfl_report(dx: float, sub: int, frame_dt: float, present: set | None = None,
               tet_materials=None, rho_mesh: float | None = None) -> None:
    """Chapter 6 Test C's force channel was unusable because the stiff wall was 8.3x over
    its stability limit. Check it BEFORE the sweep rather than discovering it after.

    🚨 FIXED 2026-08-06 -- THIS CHECK WAS OPTIMISTIC BY A FACTOR OF 1.95 AND IS WHY
    CHAPTER 8 BELIEVED IT WAS "COMFORTABLY INSIDE" THE LIMIT.

    It used c = sqrt(E/rho). That is the bar (thin-rod) wave speed. The fastest wave in a
    bulk continuum is the DILATATIONAL (P) wave, c_p = sqrt((lambda + 2 mu)/rho), and it is
    the one the timestep has to resolve. At nu = 0.45,

        lambda + 2 mu = E (1 - nu) / ((1 + nu)(1 - 2 nu)) = 3.79 E

    so c_p = 1.95 * sqrt(E/rho). Every ratio this function printed was HALF what it should
    have been -- and the near-incompressibility that makes the gap large is exactly the
    property this phantom depends on, so the error is worst where it matters most.

    Consequence for the numbers already published: at dx = 1.5 mm and 384 substeps
    (dt = 43.4 us) the 600 kPa artery wall has c_p = 45.6 m/s, dt_max = 32.9 us, i.e.
    **1.32x OVER the limit** -- not the 0.68x Chapter 8 recorded. "More substeps" was still
    not the cause of the force-channel failure (that was open-loop position driving), but the
    claim that the run sat inside the CFL limit was wrong.
    """
    dt = frame_dt / sub
    print(f"\nCFL check (dt = {dt*1e6:.1f} us, dx = {dx*1e3:.2f} mm) -- DILATATIONAL wave:")
    # 🚨 CHECK THE BODIES THAT ARE ACTUALLY MESHED, NOT THE MATERIAL TABLE.
    # This loop used to run over all of BODIES, which is the CATALOGUE of materials, not the
    # contents of the ROI. V3 added a 17 GPa femur to the catalogue; the ROI block does not
    # reach it (0 tets, confirmed by the rigid-inclusion pass finding 0 nodes to pin), and the
    # report still declared "worst case 100.42x over -- force numbers are NOT trustworthy;
    # raise --substeps to >= 38562". That is a 100x cost increase demanded on account of a
    # body that is not in the simulation, and it would have been paid.
    #
    # `present` is the set of body names carrying at least one tet. Absent bodies are still
    # PRINTED -- silently dropping rows is how the opposite error gets made -- but they cannot
    # set the limit, because a body with no elements contributes no wave to integrate.
    worst = 0.0
    for name, b in BODIES.items():
        if b["E_pa"] is None:
            continue
        rho = b["rho"] or RHO_DEFAULT
        nu = b["nu"] if b["nu"] is not None else NU
        mu, lam = lame(b["E_pa"], nu)
        c_p = np.sqrt((lam + 2.0 * mu) / rho)        # P-wave: the fastest wave present
        c_s = np.sqrt(mu / rho)                      # shear, for reference only
        dt_max = dx / c_p
        ratio = dt / dt_max
        # A body pinned as a rigid inclusion contributes NO elastic wave -- its nodes do not
        # move, so its modulus is not in the integrated system and must not set the limit.
        # Still printed, because silently dropping the worst row is how a 100x violation
        # would come back disguised as a clean report.
        if present is not None and name not in present:
            print(f"  {name:<12} E={b['E_pa']/1e3:9.1f} kPa  -- NOT IN THE ROI (0 tets), "
                  f"excluded from the limit (would be {ratio:.1f}x)")
            continue
        if b["E_pa"] >= RIGID_E_PA:
            print(f"  {name:<12} E={b['E_pa']/1e3:9.1f} kPa  -- PINNED RIGID INCLUSION, "
                  f"excluded from the limit (would be {ratio:.1f}x)")
            continue
        worst = max(worst, ratio)
        flag = "  <-- OVER LIMIT" if ratio > 1 else ""
        print(f"  {name:<12} E={b['E_pa']/1e3:7.1f} kPa  nu={nu:.2f}  "
              f"c_p={c_p:6.1f} m/s (c_s={c_s:5.1f})  "
              f"dt_max={dt_max*1e6:7.1f} us  dt/dt_max={ratio:5.2f}{flag}")
    if worst > 1.0:
        print(f"  🚨 catalogue worst case {worst:.2f}x over -- force numbers are NOT trustworthy; "
              f"raise --substeps to >= {int(np.ceil(sub*worst))}")
    else:
        print(f"  catalogue: worst case {worst:.2f}x of the limit")

    # ── THE AUTHORITATIVE ROW: the array the solver actually integrates ──────────────────
    # 🚨 EVERYTHING ABOVE IS THE CATALOGUE, AND THE CATALOGUE CANNOT SEE A HOMOGENISED CELL.
    # cell_material averages Lame parameters over a cell's sub-samples, so a cell straddling
    # two bodies carries a modulus that belongs to NEITHER of them and is filed under whichever
    # body dominates. On 2026-08-11 that put 924 tets of the V3 leg at E >= 1 MPa and 462 of
    # them above 10 GPa -- all of them labelled "Muscle", all of them with free nodes, and the
    # loop above reported "OK: worst case 0.74x" for a mesh whose worst element was 101x over.
    # This block reads model.tet_materials, which is the only artifact that cannot lie about
    # what is being integrated.
    if tet_materials is not None:
        tm = np.asarray(tet_materials)
        mu_t, lam_t = tm[:, 0].astype(np.float64), tm[:, 1].astype(np.float64)
        rho = float(rho_mesh) if rho_mesh else RHO_DEFAULT
        c_p = np.sqrt((lam_t + 2.0 * mu_t) / rho)
        ratio_t = dt / (dx / c_p)
        e_t = mu_t * (3.0 * lam_t + 2.0 * mu_t) / (lam_t + mu_t)
        i = int(np.argmax(ratio_t))
        n_over = int((ratio_t > 1.0).sum())
        print(f"  MESH (model.tet_materials, {len(tm):,} tets, rho {rho:.0f}): worst element "
              f"E = {e_t[i]:.4g} Pa -> dt/dt_max = {ratio_t[i]:.2f}x"
              + (f", {n_over:,} tet(s) over the limit" if n_over else ", none over the limit"))
        if ratio_t[i] > 1.0:
            # ⚠️ Two different failures, and they must not print the same banner. If the mesh is
            # only a little worse than the catalogue, the catalogue DID see the offending body
            # and the message is "raise substeps". If the mesh is far worse, the catalogue was
            # blind -- a homogenised cell whose modulus belongs to no body in the table -- and
            # that is a defect, not a setting. Saying "the catalogue does not show it" when it
            # plainly does is how a real banner gets learned as noise and stops being read.
            hidden = ratio_t[i] > 1.5 * max(worst, 1e-30)
            if hidden:
                print(f"  🚨 THE MESH IS OVER THE LIMIT AND THE CATALOGUE ROWS ABOVE DO NOT SHOW "
                      f"IT ({ratio_t[i]:.1f}x against the catalogue's {worst:.2f}x).\n"
                      f"     Some cell carries a HOMOGENISED modulus belonging to no body in the "
                      f"table -- see RIGID_CELL_POLICY.")
            else:
                print(f"  🚨 the mesh is over the limit, consistent with the catalogue's worst "
                      f"row ({worst:.2f}x).")
            print(f"     {n_over:,} of {len(tm):,} tets are unintegrable at {sub} substeps; the "
                  f"worst needs >= {int(np.ceil(sub*ratio_t[i])):,}.\n"
                  f"     Force numbers from this mesh are NOT trustworthy.")


def contact_cfl_report(dx: float, sub: int, frame_dt: float, ke: float, rho: float,
                       n_contact: int | None = None, k_tissue_n_per_mm: float | None = None):
    """The CONTACT oscillator's own resolution check. Returns dt/T.

    🚨 WHY THIS EXISTS — 2026-08-11, AND IT IS THE FIX FOR THE FORCE CHANNEL.
    `cfl_report` above checks whether the substep can resolve the fastest ELASTIC wave in the
    tissue. Nothing ever checked the other oscillator in the system: a soft-contact particle on
    a penalty spring. Its period is T = 2*pi*sqrt(m/ke) with m = rho*dx^3, and at the production
    settings that is **26.4 us against a 32.55 us substep** — dt/T = 1.23, i.e. the contact
    spring completes more than a full cycle inside one substep and is not resolved at all.

    An unresolved stiff spring under a 10-iteration Gauss-Seidel solve does not behave like a
    spring. It behaves like a constraint the solver cannot relax, and the probe then fights it
    instead of pressing the tissue. Measured consequence, same mesh, same substeps, same rung:

        ke = 2e5  dt/T = 1.23   F_drive 281 N vs F_tissue 0.97 N  — x289, NEVER SETTLES
        ke = 2e3  dt/T = 0.12   F_drive 0.788 N vs F_tissue 0.728 N — x1.08, settles at frame 110

    That is item 17's entire "the plant is 85x too stiff", and item 19's "the rungs never
    settle", in one number that nothing was printing.

    ⚠️ THE POINT IS THE CRITERION, NOT THE VALUE. ke is NOT free to be tuned until the force
    curve looks right — that is the one move this project does not make. It is bounded on BOTH
    sides, and the run must state both:

        UPPER bound (this function): dt/T must be small, or the contact is unintegrable.
                                     dt/T <= 0.1  ->  ke <= 1320 N/m at dx 1.5 mm, 512 substeps.
        LOWER bound (the caller):    the penalty must still be RIGID relative to the tissue, or
                                     the probe sinks into a soft mattress and the contact stops
                                     being a contact. n_contact * ke >> k_tissue.

    At dx = 1.5 mm those bounds are ~3 decades apart (216 nodes x 2e3 N/m = 432 N/mm of penalty
    against a 0.49 N/mm tissue — still 880x rigid), so there is a wide legitimate window and no
    tuning is required to sit in it. If the window ever closes, the answer is more substeps or a
    different contact formulation, NOT a value picked to make the answer agree.

    📌 dt/T scales as dt / (dx^1.5 * sqrt(rho/ke)), so it moves with BOTH dx and substeps. That
    is why the plant looked resolution-dependent: halving dx cut the particle mass 8x and made
    the contact oscillator 2.8x faster at an unchanged substep.
    """
    dt = frame_dt / sub
    m = rho * dx ** 3
    T = 2.0 * np.pi * np.sqrt(m / ke) if ke > 0 else float("inf")
    r = dt / T
    verdict = "RESOLVED" if r <= 0.1 else ("MARGINAL" if r <= 0.3 else "🚨 UNRESOLVED")
    print(f"\nCONTACT check (dt = {dt*1e6:.1f} us, particle mass = rho*dx^3 = {m:.3e} kg):")
    print(f"  soft_contact_ke = {ke:.3g} N/m  ->  contact period T = {T*1e6:.1f} us   "
          f"dt/T = {r:.3f}   {verdict}")
    if r > 0.1:
        ke_ok = m * (2.0 * np.pi / (10.0 * dt)) ** 2
        print(f"  🚨 THE CONTACT OSCILLATOR IS NOT RESOLVED. The drive-lag force channel will "
              f"read the\n     contact model rather than the tissue, and rungs will not settle "
              f"(item 24).\n     For dt/T <= 0.1 at these settings: soft_contact_ke <= "
              f"{ke_ok:.0f} N/m, or substeps >= {int(np.ceil(sub*r/0.1)):,}.")
    if n_contact and k_tissue_n_per_mm:
        pen = n_contact * ke / 1e3            # N/mm, in parallel under the aperture
        print(f"  lower bound: {n_contact} nodes x {ke:.3g} N/m = {pen:.0f} N/mm of penalty "
              f"against a {k_tissue_n_per_mm:.3f} N/mm tissue -> x{pen/k_tissue_n_per_mm:.0f} "
              f"{'OK, still effectively rigid' if pen/k_tissue_n_per_mm > 50 else '🚨 TOO SOFT — the contact is becoming a mattress'}")
    return r


class Admittance:
    """Closed-loop admittance control, transplanted from Chapter 6's slab benchmark.

    WHY THIS EXISTS. The femoral rig drives the probe OPEN-LOOP to a position: set a target
    depth, let a stiff prismatic joint (ke = 2e5 N/m) chase it. The 2026-08-04 diagnostic
    showed that never settles -- with the target held fixed for 100 frames the probe was
    still swinging +/-100 mm/s and the reported force between 66 and 561 N, and the probe sat
    ~2.3 mm short of target indefinitely. The "force" was just ke x lag, and the 27 % scatter
    across geometrically identical sweep angles was that oscillation sampled at different
    phases.

    Chapter 6 already solved this on the slab and the femoral rig never inherited it. Commanding
    a FORCE and letting depth be whatever the tissue gives cannot ring in the same way: there is
    no stiff spring being held in tension against the contact.

    THE LAW (physics_calibration_and_impedance_methods.md eq. 16-19). Target dynamics

        m_a * v_dot + b_a * v = f* - f_contact                                        (16)

    with NO stiffness term -- a virtual spring would define an equilibrium depth, and depth is
    exactly what we want the tissue to decide.

    🚨 INTEGRATE IT EXACTLY, NOT WITH FORWARD EULER. Chapter 6 tried Euler first; it is only
    conditionally stable (dt < 2*m_a/b_a) and the first closed-loop run limit-cycled. That was
    a controller bug that looked like physics, and it is worth not repeating while we are in
    the middle of diagnosing a different oscillation. The zero-order-hold exact solution:

        v_{k+1} = v_k * exp(-dt/tau) + (f* - f_k)/b_a * (1 - exp(-dt/tau)),  tau = m_a/b_a
        q*_{k+1} = q*_k + v_{k+1} * dt,   |v| <= v_max                            (17b, 18, 19)

    Unconditionally stable for any dt > 0 and positive (m_a, b_a): the amplification factor
    exp(-dt/tau) lies in (0, 1) and can never alternate sign.

    Gains are Chapter 6's measured values, which held 4.998 +/- 0.025 N on the slab and
    reproduced depth to 1 um on a return-to-setpoint. They are a starting point here, not a
    guarantee -- the plant is a 185 mm limb, not a slab, and Chapter 6 itself measured the
    plant varying 20-40 % with scene randomisation alone.
    """

    def __init__(self, m_a=4.0, b_a=400.0, v_max=0.030, tau_f=0.020, dt=1.0 / 60.0):
        self.m_a, self.b_a, self.v_max, self.dt = m_a, b_a, v_max, dt
        self.tau = m_a / b_a
        self.tau_f = tau_f
        self.v = 0.0
        self.q = 0.0
        self.f_filt = None
        self._decay = np.exp(-dt / self.tau)
        # estimator low-pass, tau_f = 20 ms: slower than the estimator's own noise, faster
        # than anything the loop needs to track (tau_f = 2 * tau)
        self._alpha = 1.0 - np.exp(-dt / tau_f)

    def step(self, f_setpoint: float, f_measured: float) -> float:
        """One control period. Returns the new position target (metres, negative = deeper)."""
        self.f_filt = f_measured if self.f_filt is None else \
            self.f_filt + self._alpha * (f_measured - self.f_filt)
        err = f_setpoint - self.f_filt
        self.v = self.v * self._decay + (err / self.b_a) * (1.0 - self._decay)
        self.v = float(np.clip(self.v, -self.v_max, self.v_max))
        # sign: the probe presses along -Z, so a POSITIVE force error must drive it deeper
        self.q -= self.v * self.dt
        return self.q


class Phantom:
    def __init__(self, dx, substeps, probe_w, probe_len, soft_contact_ke=None):
        (self.model, self.idx, self.body, self.meta,
         self.rest, self.body_of, self.solid2d) = build(dx, probe_w, probe_len)
        self.dx = dx
        # 🪤 RECONCILE THE MESH build() WROTE WITH THE MESH THE MODEL ACTUALLY HOLDS, ONCE.
        # newton's ModelBuilder.add_tetrahedron SILENTLY DROPS any element whose rest volume is
        # <= 0 -- it prints "inverted tetrahedral element" to stdout and simply does not append
        # it (builder.py:8275-8283). So a model can legitimately carry FEWER elements than the
        # tet list build() thinks it handed over, and every downstream per-tet array (materials,
        # the F_tissue operator's k_mu/k_lambda, mesh_quality's rest volumes) would then be
        # off-by-index with nothing raising. build()'s assert makes that impossible today; this
        # makes it impossible to reintroduce quietly.
        self.tet_i = self.model.tet_indices.numpy().reshape(-1, 4).astype(np.int64)
        if len(self.tet_i) != self.meta["n_tets"]:
            raise SystemExit(
                f"!!! the model and the mesh disagree about how many elements exist: "
                f"model.tet_indices has {len(self.tet_i):,}, build() wrote "
                f"{self.meta['n_tets']:,}.\n"
                f"    newton drops rest-inverted tets without raising, so every per-tet array "
                f"downstream is now\n    misaligned. Do not run: the force operator would be "
                f"reading the wrong material for every element\n    past the first drop.")
        # rest volumes for mesh_quality(). Computed ONCE from the same operator as build()'s
        # assert; on this uniform grid they are all dx^3/6, but they are measured, not assumed.
        self.tet_vol0 = tet_volumes(self.rest, self.tet_i)
        # Re-run the stability check against what was ACTUALLY meshed. The diag scripts call
        # cfl_report before this point, when the body list is not yet known, so that call can
        # only check the catalogue; this one is the authoritative verdict.
        cfl_report(dx, substeps, 1.0 / 60.0,
                   present=set(self.meta.get("bodies_present", [])),
                   tet_materials=self.model.tet_materials.numpy(),
                   rho_mesh=self.meta.get("rho_mesh"))
        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=VBD_ITERATIONS,
            rigid_body_particle_contact_buffer_size=4 * self.model.particle_count,
            # opposing lumen walls must be able to STOP each other, or a "collapsed"
            # vessel is really just two walls passing through one another.
            # ⚠️ See SELF_CONTACT: this is a lumen feature applied to the whole mesh, and it
            # has a penalty stiffness attached. --self-contact 0 is the isolating leg.
            particle_enable_self_contact=SELF_CONTACT,
            particle_self_contact_radius=SELF_CONTACT_R_SCALE * dx,
            particle_self_contact_margin=(SELF_CONTACT_R_SCALE + 0.1) * dx,
        )
        # The contact oscillator's own resolution check -- see contact_cfl_report. It is printed
        # for EVERY Phantom, not only the diagnostics, because the thing it catches is invisible
        # in the output and cost this project four days.
        # 🪤 the check must see the ke the RUN uses, not build()'s default. Callers that set
        # soft_contact_ke after construction were getting a gate that fired on 2e5 while the run
        # used 2e3 -- a loud, correct-looking warning about a value nobody was using.
        if soft_contact_ke is not None:
            self.model.soft_contact_ke = float(soft_contact_ke)
        self.contact_dt_over_T = contact_cfl_report(
            dx, substeps, 1.0 / 60.0, float(self.model.soft_contact_ke),
            float(self.meta.get("rho_mesh") or RHO_DEFAULT))
        self.self_contact = bool(SELF_CONTACT)
        self.cp = newton.CollisionPipeline(self.model, soft_contact_margin=0.001)
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.contacts = self.cp.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.s0)
        wp.copy(self.s1.body_q, self.s0.body_q)
        self.frame_dt, self.sub = 1.0 / 60.0, substeps
        self._target = 0.0
        self._lumen_pairs()
        self._lumen_rings()
        self.lumen_p = dict(LUMEN_P_DEFAULT)
        self._p_scale = 1.0
        self._pf = wp.zeros(self.model.particle_count, dtype=wp.vec3)
        self._static_load = False

    def _lumen_rings(self):
        """Nodes forming the wall of each lumen, at mid-length.

        A single node pair measures only the vertical diameter, and at any affordable dx it
        is badly quantised (a first coarse run reported 9.00 mm rest height for the artery,
        the vein AND the solid nerve -- all three were just three cells apart). The lumen's
        cross-sectional AREA, taken as the polygon through its ring of wall nodes, is what
        actually says whether a vessel has closed, and it degrades gracefully with dx.
        """
        j = self.meta["ny"] // 2
        nx, ny, nz = self.meta["nx"], self.meta["ny"], self.meta["nz"]
        self.rings, self.surf, self.ring_ri = {}, {}, {}
        for name, C, ri in (("vein", VEIN_C, VEIN_RI), ("artery", ARTERY_C, ARTERY_RI)):
            if not in_window(*C, margin=ri + 2.2 * self.dx):
                continue          # swept out of the block at this THETA -- see the note there
            cx, cz = to_block(*C)
            # One node per angular bin: the INNERMOST surviving node in that direction.
            # Taking every node within a radial band instead would sweep in wall nodes up
            # to sqrt(2)*dx out, which barely move, and would damp the measured collapse.
            nbin = 16
            best = {}
            for i in range(nx + 1):
                for k in range(nz + 1):
                    p = self.idx[i, j, k]
                    if p < 0:
                        continue
                    x, z = X0 + i * self.dx, Z0 + k * self.dx
                    r = np.hypot(x - cx, z - cz)
                    if r < ri or r > ri + 2.2 * self.dx:
                        continue
                    b = int(((np.arctan2(z - cz, x - cx) + np.pi) / (2 * np.pi)) * nbin) % nbin
                    if b not in best or r < best[b][0]:
                        best[b] = (r, i, k)
            if len(best) < 6:
                # Under-resolved is still an error at angles where the lumen IS in the
                # window -- it means dx is too coarse, not that the vessel has swept away.
                raise SystemExit(f"{name}: only {len(best)} ring nodes at dx="
                                 f"{self.dx*1e3:.2f} mm, theta={np.degrees(THETA):+.2f} deg "
                                 f"-- lumen in the window but unresolved")
            # Order by REST angle once, and keep that order. areas() re-sorts per call (it
            # only needs a polygon), but the pressure load needs a consistent traversal or
            # its outward normals flip sign; and the grid is a prism, so the same (i, k)
            # ring exists in every y-plane.
            ik = [(i, k) for _, i, k in
                  sorted(best.values(),
                         key=lambda t: np.arctan2(Z0 + t[2] * self.dx - cz,
                                                  X0 + t[1] * self.dx - cx))]
            self.rings[name] = np.array([self.idx[i, j, k] for i, k in ik])
            self.surf[name] = np.array([[self.idx[i, jj, k] for i, k in ik]
                                        for jj in range(ny + 1)])
            self.ring_ri[name] = ri

    @staticmethod
    def _poly_area(pts_xz: np.ndarray) -> float:
        """Shoelace area of the ring, ordered by angle about its own centroid."""
        c = pts_xz.mean(axis=0)
        d = pts_xz - c
        order = np.argsort(np.arctan2(d[:, 1], d[:, 0]))
        p = pts_xz[order]
        return float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                         np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2.0)

    def areas(self, pq=None):
        pq = self.s0.particle_q.numpy() if pq is None else pq
        return {n: self._poly_area(pq[r][:, [0, 2]]) for n, r in self.rings.items()}

    def _lumen_pairs(self):
        """The node pair whose vertical separation IS each lumen's height: the closest
        surviving node below and above the lumen centre, on its own centreline."""
        self.pairs = {}
        nx, ny, nz = self.meta["nx"], self.meta["ny"], self.meta["nz"]
        j = ny // 2
        for name, C, ri in (("vein", VEIN_C, VEIN_RI), ("artery", ARTERY_C, ARTERY_RI),
                            ("nerve", NERVE_C, NERVE_R)):
            if C is None:
                # The structure does not exist at this axial station. At s = 30 -- the
                # saphenofemoral junction, where every diameter we use was measured -- the
                # nerve has already ended (it spans s = 0..25). Absent is the correct
                # answer, not a resolution failure; see v3_section.py's docstring.
                continue
            if not in_window(*C, margin=ri + 2.2 * self.dx):
                continue          # swept out of the block at this THETA
            cx, cz = to_block(*C)
            i = int(round((cx - X0) / self.dx))
            col = [(k, self.idx[i, j, k]) for k in range(nz + 1) if self.idx[i, j, k] >= 0]
            if name == "nerve":
                # solid, not a lumen: track its top and bottom SURFACE nodes, to show it
                # neither collapses nor resists like the artery
                kb = min(range(len(col)), key=lambda n: abs(Z0 + col[n][0] * self.dx - (cz - ri)))
                kt = min(range(len(col)), key=lambda n: abs(Z0 + col[n][0] * self.dx - (cz + ri)))
                self.pairs["nerve"] = (col[kb][1], col[kt][1])
                continue
            below = [p for k, p in col if Z0 + k * self.dx < cz]
            above = [p for k, p in col if Z0 + k * self.dx > cz]
            if not below or not above:
                raise SystemExit(f"{name}: in the window at theta={np.degrees(THETA):+.2f} deg "
                                 f"but not resolved at dx={self.dx*1e3:.1f} mm")
            self.pairs[name] = (below[-1], above[0])

    # ── transmural pressure: a FOLLOWER LOAD, not a filler material ──────────────────────
    #
    # 🚨 THE INTUITIVE IMPLEMENTATION IS WRONG FOR THE VEIN AND MUST NOT BE USED. Filling the
    # lumen with an incompressible fluid or a near-incompressible solid models a SEALED BAG.
    # A real vein collapses precisely because its blood is free to leave AXIALLY -- it is
    # incompressible but not trapped. A sealed filler would hold the vein near 100 % of rest
    # area indefinitely, and would land near the existing ~80 % floor for an entirely
    # different reason, which is worse than the current state because the number would look
    # familiar and be caused by something new.
    #
    # What is physically right, and what this implements, is a transmural PRESSURE boundary
    # condition on the lumen wall: a constant normal traction that follows the deforming
    # surface. It carries no volume constraint, so the lumen is free to close; what resists
    # closure is the pressure difference, which is the actual clinical mechanism and is the
    # ~10 kPa that separates an artery from a vein.
    #
    # DISCRETISATION. The lumen boundary in each y-plane is a closed polygon of ring nodes
    # ordered by rest angle. For a polygon traversed counter-clockwise in (x, z), the
    # tributary outward normal at vertex i is 0.5 * R90(p_{i+1} - p_{i-1}) with
    # R90(tx, tz) = (tz, -tx); its magnitude is half the sum of the two adjacent edge
    # lengths, i.e. the correct nodal length weight. The nodal force is p * dy * that vector,
    # recomputed from the DEFORMED positions -- so it is a genuine follower load, and it
    # weakens automatically as the wall flattens. Sum over a closed polygon is identically
    # zero, so no net force is injected into the block.
    #
    # 🔴 VALIDITY LIMIT, STATED SO IT CANNOT BE MISREAD. This is validated only against the
    # ANALYTIC cross-section this file builds, which does have graded wall bands (Artery
    # 600 kPa, Vein 20 kPa). It does NOT transfer to V3 as built: V3's CFA/CFV are bare
    # lumens with no wall bodies AND their walls are exactly tangent (centre-to-centre
    # 10.02 mm against the spec's 15.0 mm), so an artery pressure load there would press
    # directly on the vein's near wall through zero intervening tissue. Fix the V3 geometry
    # before reading anything into a pressurised V3 run.
    def set_lumen_pressure(self, artery_pa=0.0, vein_pa=0.0):
        self.lumen_p = {"artery": float(artery_pa), "vein": float(vein_pa)}

    def set_static_load(self, node_ids, total_n):
        """Apply a KNOWN downward external load of `total_n` newtons, spread evenly over
        `node_ids`. Nothing in the project's physics needs this -- it exists so the force
        READOUT can be anchored against a load whose value is known independently of the
        tissue model.

        2026-08-12, after a reviewer pointed out that the patch test VERIFIES the operator and does
        not VALIDATE the channel: the only external anchors we have are boundary behaviours
        (no contact -> 0 N, and coaptation at <= 10 N). This adds the one anchor that is
        traceable AND material-independent: at static equilibrium the sum over free nodes of
        (K u) equals the applied external load for ANY constitutive model, so a known load
        pins the readout's span without assuming the tissue is right.

        ⚠️ Mutually exclusive with lumen pressure -- both write self._pf, and the pressure
        path recomputes it every frame. Do not set both.
        """
        f = np.zeros((self.model.particle_count, 3), dtype=np.float32)
        ids = np.asarray(node_ids, dtype=np.int64)
        if len(ids) == 0:
            raise SystemExit("!!! set_static_load with no nodes -- UNINFORMATIVE")
        f[ids, 2] = -float(total_n) / len(ids)          # -z is into the tissue
        self._pf.assign(f)
        self._static_load = bool(total_n)
        return float(total_n)

    def _refresh_lumen_pressure(self):
        """Recompute the follower forces from the CURRENT lumen shape, once per FRAME.

        Not per substep: the wall moves by well under a micrometre in one 43 us substep, so
        refreshing at frame rate is a negligible lag, whereas a device->host readback 384
        times a frame is not affordable. The result is held in a device-side buffer and
        copied back into particle_f after each clear_forces().
        """
        active = {n: p for n, p in self.lumen_p.items() if p and n in self.surf}
        if not active:
            return False
        pq = self.s0.particle_q.numpy()
        f = np.zeros((self.model.particle_count, 3), dtype=np.float32)
        nyp = self.meta["ny"] + 1
        for name, p in active.items():
            ring = self.surf[name]                       # (ny+1, nbin) node ids
            for jj in range(nyp):
                ids = ring[jj]
                P = pq[ids][:, [0, 2]]
                e = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
                n = 0.5 * np.column_stack([e[:, 1], -e[:, 0]])     # outward, length-weighted
                # half weight on the two free end planes
                w = self.dx * (0.5 if jj in (0, nyp - 1) else 1.0)
                fn = (p * self._p_scale * w) * n
                np.add.at(f, ids, np.column_stack([fn[:, 0], np.zeros(len(ids)), fn[:, 1]]))
        self._pf.assign(f)
        return True

    def pressure_summary(self):
        """Total outward force each pressurised lumen is currently applying, N.

        Printed rather than assumed: a sign error here is silent (the lumen would be sucked
        shut and the artery would read as more compressible than the vein), and the total is
        also the cheapest sanity check that the area weights are right --
        p * perimeter * LY should be recovered to a few percent.
        """
        out = {}
        pq = self.s0.particle_q.numpy()
        for name, p in self.lumen_p.items():
            if not p or name not in self.surf:
                continue
            P = pq[self.surf[name][self.meta["ny"] // 2]][:, [0, 2]]
            perim = float(np.sum(np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)))
            out[name] = dict(pa=p * self._p_scale, perimeter_mm=perim * 1e3,
                             total_N=p * self._p_scale * perim * LY)
        return out

    def ramp_pressure(self, frames):
        """Bring the lumen pressure up linearly over `frames` frames, then hold.

        Applying 12 kPa in one step against a lumped nodal mass of ~3.5e-6 kg is a large
        explicit impulse; the elastic response is implicit and will not explode, but the
        transient rings for hundreds of frames and every reading taken during it is a phase
        sample -- the exact failure Chapter 6 spent a chapter removing from the force
        channel. Ramp, then settle, then measure.
        """
        for i in range(max(1, frames)):
            self._p_scale = (i + 1) / max(1, frames)
            self.step()
        self._p_scale = 1.0

    # ── probe state, read from the BODY rather than from the controller ──────────────────
    def probe_z(self):
        return float(self.s0.body_q.numpy()[self.body][2])

    def probe_depth(self):
        """ACHIEVED indentation of the probe, m, positive = pressed in.

        ⚠️ Not the same as the commanded target, and the difference is the whole subject of
        Chapter 6: the probe sat 2.1 mm short of target indefinitely while the rig reported
        the target as its depth. Anything published as "depth" must come from here.
        """
        return self.meta["start_z"] - self.probe_z()

    def probe_speed(self):
        """|v_z| of the probe, m/s. Newton's body_qd is (linear, angular), so [2] is v_z --
        verified against newton 1.4.0's State docstring, not assumed."""
        return abs(float(self.s0.body_qd.numpy()[self.body][2]))

    def force_settled(self, max_frames=600, v_tol=1e-4, window=30, tare=0.0):
        """Hold the current target until the probe is AT REST, then average the force.

        🚨 THIS IS CHAPTER 8'S OWN "NEXT #1", AND IT IS WHY EVERY FORCE NUMBER IN THAT
        CHAPTER SHOULD BE IGNORED. force() is an actuator-side balance,
        F = ke*(q - target) + kd*qd, which is exact only at rest. Sampled while the probe is
        still moving it reports the drive damper: at the 1 mm-per-frame drive rate the rig
        used, qd = 0.06 m/s and kd*qd alone is 120 N. Chapter 8 read one instantaneous sample
        after a fixed frame count and published -0.23, 5.69, 31.54, 7.72, 148.90 N across
        smoothly increasing indentation.

        Returns (mean, sd, speed, at_rest). `at_rest` False means the number is NOT a contact
        force and must be reported as such rather than quietly averaged into a table.
        """
        fs = []
        for _ in range(max_frames):
            self.step()
            fs.append(self.force() - tare)
            if len(fs) >= window and self.probe_speed() < v_tol:
                break
        fs = np.asarray(fs[-window:])
        v = self.probe_speed()
        return float(fs.mean()), float(fs.std()), float(v), bool(v < v_tol)

    def mesh_quality(self, near_degenerate_J: float, pq=None) -> dict:
        """Element quality of the CURRENT (deformed) mesh. The witness the press ladders lacked.

        Returns dict(min_jacobian, n_inverted, n_near_degenerate, worst_tet_id, worst_tet_xyz,
        near_degenerate_J, n_tets) with J = V/V0 per tet, V from `tet_volumes` on the live
        particle positions and V0 the same operator on `self.rest`.

        `near_degenerate_J` is REQUIRED and deliberately has no default -- it decides which
        rungs a run is allowed to quote, so it is a 🔴 MODELLING CHOICE and the caller has to
        state it. The J <= 0 crossing is NOT a choice and needs no tag: an element whose signed
        volume has changed sign has passed through zero volume and turned inside out, and no
        continuum quantity computed over it means anything.

        🚨 WHY THIS EXISTS -- 2026-08-13. IT VOIDS THE DEEP HALF OF EVERY EXISTING LADDER.
        The 2026-08-12 press campaign reported rungs at 8.99 / 10.99 / 12.98 mm past skin as
        "clean and monotone". Reconstructing the topology from the saved node states and
        computing J per tet says otherwise -- v3, settled state, minJ by rung:

            1 mm 0.903 · 3 mm 0.757 · 5 mm 0.628 · 7 mm 0.447 · 9 mm 0.094
            11 mm -0.083 (4 inverted) · 13 mm -0.164 (2 inverted) · 16.1 mm -65.8 (13,273)

        i.e. NEGATIVE-VOLUME ELEMENTS IN THE SETTLED EQUILIBRIUM STATE, two rungs inside what
        was called clean, and the visible 16 mm explosion is the CASCADE, not the onset. All
        four V4 ladders do the same thing (first negative at 8.99-10.99 mm).

        ⚠️ AND THE CLOSURE CURVE GIVES NO SIGN AT ALL. Newton's VBD integrates STABLE
        Neo-Hookean (particle_vbd_kernels.py:189, Smith 2018), which has NO INVERSION BARRIER BY
        CONSTRUCTION, and its per-vertex Hessian drops the s*d2J/dF2 term (kernels 218-222),
        leaving H = mu*I9 + lambda_nh*cof(F) (x) cof(F). As an element flattens cof(F) -> 0 and
        the only curvature resisting the flattening mode is mu, while the force still carries
        s = lambda_nh*(J - alpha). At nu = 0.45 lambda_nh/mu = 10, and there is no line search
        anywhere in solver_vbd.py. So the solver SITS HAPPILY on inverted tets: forces stay
        finite, rungs settle, lumen areas decrease politely, the table looks monotone. This is
        defect class 2 in its purest form -- an instrument returning a true value about the
        wrong thing -- and nothing in the rig could see it, because femoral_fem's only inversion
        check ran on the REST mesh and diag_mesh_stiffness's ran once at construction.

        ⚠️ DO NOT ANSWER THIS WITH MORE SUBSTEPS. It is not a CFL failure. Across the rung where
        the mesh inverts, the elastic CFL sits at 1.02x and the contact oscillator at
        dt/T = 0.124, both UNCHANGED. Substeps buy nothing against a solver with no inversion
        barrier. The existing ke A/B says the same thing from the other side: gap20 (ke 2e3) and
        gap20_ke2e4 (ke 2e4) are the same mesh, and inversion onset moved 9.53 -> 9.61 mm (i.e.
        not at all) while the cascade moved 12.97 -> 14.37 mm. Contact stiffness sets WHEN an
        already-inverted seed tips over, not WHETHER it seeds.

        📌 The degeneracy is not bulk and not the anatomy. At 13 mm total mesh volume is
        conserved to 2.7 % and the median J of the top layer under the probe is 0.80; the bad
        elements are a handful at the RIM OF THE CONTACT PATCH, at depth 0, at the two geometric
        singularities of a rigid FLAT punch on a VOXELISED curved skin (the risers of the top
        staircase plateau, and the punch edge itself). That is why `worst_tet_xyz` is returned
        and logged: the coordinate is the evidence for the seat, and legs B (flat skin) and D
        (narrow probe) are the controlled test of it.

        COST. One host copy of particle_q plus one einsum over n_tets. ~15 ms per call at
        159,786 tets, against a ~2 h ladder. There is no version of this that is too expensive
        to run every frame, and a strided version would be able to step over the frame where
        the minimum happens.
        """
        pq = self.s0.particle_q.numpy() if pq is None else pq
        v = np.asarray(pq, dtype=np.float64)
        vol = tet_volumes(v, self.tet_i)
        J = vol / self.tet_vol0
        w = int(np.argmin(J))
        c = v[self.tet_i[w]].mean(axis=0)
        return dict(
            min_jacobian=float(J[w]),
            # J <= 0 is the element turning inside out. Not a threshold, not a choice.
            n_inverted=int((J <= 0.0).sum()),
            # 🔴 near-degenerate INCLUDES the inverted ones, so n_near_degenerate >= n_inverted
            # always. Stated because a reader who assumes the two sets are disjoint will read
            # "4 inverted, 4 near-degenerate" as eight bad elements.
            n_near_degenerate=int((J < float(near_degenerate_J)).sum()),
            near_degenerate_J=float(near_degenerate_J),
            worst_tet_id=w,
            worst_tet_xyz=[float(c[0]), float(c[1]), float(c[2])],
            n_tets=int(len(J)),
        )

    def closure_floor_pct(self):
        """The smallest lumen area this DISCRETISATION can report, as % of rest.

        🚨 THE ACCEPTANCE CRITERION IN V3_PARAMETRIC_SPEC.md §3 IS "~0 % OF REST LUMEN AREA".
        THIS RIG CANNOT REPORT THAT NUMBER, FOR REASONS THAT HAVE NOTHING TO DO WITH THE
        TISSUE. Opposing walls are stopped by particle self-contact at roughly
        2 * particle_self_contact_radius = 0.8 * dx = 1.2 mm at dx = 1.5 mm. A perfectly
        flattened lumen is then a slit of length ~half the rest perimeter and width 1.2 mm:

            vein, rest inner radius 5.48 mm -> perimeter 34.4 mm
            floor area ~ 17.2 mm * 1.2 mm  = 20.6 mm^2  = 22 % of the 94.3 mm^2 rest area

        So a vein that closed PERFECTLY would still be reported at ~22 %, and the measured
        79.9 % is nowhere near that floor -- the floor is not what is limiting the current
        result, but it does mean "~0 %" is unreachable as a MEASUREMENT and the criterion has
        to be restated against this number or dx has to come down (and with only ~7 nodes
        across an 11 mm lumen at dx = 1.5 mm, a flattened tube is not representable anyway).
        Printed with every run so it cannot be forgotten.

        🚨 CORRECTED 2026-08-25 -- OPEN ITEM 27, step 1. THE OPERATOR USED TO DISAGREE WITH THE
        ONE EVERY CLOSURE PERCENTAGE IS MEASURED WITH.

        The version above computed the floor analytically from the NOMINAL lumen radius `ri`, in
        both the perimeter and the rest area:

            floor % = 100 * (pi*ri * gap) / (pi*ri**2) = 100 * gap / ri

        But no run divides by pi*ri**2. Every reported closure % is `100 * areas()[n] /
        rest_area[n]`, where BOTH sides are the shoelace area of the 16-node ring -- and that ring
        is built from the innermost surviving grid node in each of 16 angular bins within
        `[ri, ri + 2.2*dx]` (see `_rings` above), so on a 1.5 mm grid it sits at an effective
        radius of ~5.52 mm for the vein, not at 4.971. The two rest areas are 23 % apart:

            vein   analytic pi*ri**2 = 77.6 mm^2   measured ring = 95.6 mm^2
            artery analytic          = 38.0 mm^2   measured ring = 51.7 mm^2   (36 % apart)

        🚨 AND THE OLD FLOOR WAS TOO HIGH, WHICH IS THE DANGEROUS DIRECTION. A vein reading 22 %
        would be AT the true floor while the record said it was still above it. The quoted 24.1 %
        (vein) / 34.5 % (artery) become ~21.8 % / ~29.6 % under one consistent operator.

        This version measures the ring's OWN rest perimeter and OWN rest area, so the floor and
        the closure percentages are the same operator on the same nodes.

        ⚠️ STILL DERIVED, NOT MEASURED. This is step 1 of item 27. It removes the operator
        mismatch; it does NOT test the two remaining assumptions -- that opposing walls actually
        settle at 0.8*dx under load, and that a collapsed lumen is a straight slit of unchanged
        perimeter on a ring of 16 nodes across a lumen ~7 nodes wide. Step 2 (drive one lumen's
        ring together kinematically with no probe and read the area off `areas()`) is the direct
        measurement and has NOT been run.

        ⚠️ AND IT MOVES AGAIN WHEN ITEM 26 LANDS. `ri` comes from the V3 CAD, whose vessel OUTER
        diameters were set to Zambetti's LUMEN figures, so the vein lumen is ~9 % small and the
        artery ~22 %. Item 26 is deliberately deferred (trigger: the vein reaching below ~50 % of
        rest; it is at 87.3 %). Restate the floor in the criterion ONCE, after 26 -- not twice.
        """
        # ⚠️ Must track the solver's ACTUAL radius, not a copy of the default. If
        # --self-contact 0 the walls are not stopped at all and there is no floor.
        gap = (2.0 * SELF_CONTACT_R_SCALE * self.dx) if self.self_contact else 0.0
        out = {}
        for name, r in self.rings.items():
            ring_xz = self.rest[r][:, [0, 2]]
            rest_area = self._poly_area(ring_xz)
            if rest_area <= 0.0:
                raise SystemExit(f"{name}: rest ring area is {rest_area} -- the floor cannot be "
                                 f"computed and must not be reported as a number")
            # A fully flattened lumen is a slit: length ~half the ring's OWN rest perimeter,
            # width `gap`. Perimeter and area come from the same ring in the same traversal
            # order, so the ratio carries no operator mismatch.
            out[name] = 100.0 * (0.5 * self._poly_perimeter(ring_xz) * gap) / rest_area
        return out

    @staticmethod
    def _poly_perimeter(pts_xz: np.ndarray) -> float:
        """Perimeter of the ring, ordered by angle about its own centroid.

        📌 The ordering is duplicated from `_poly_area` ON PURPOSE rather than factored out:
        these two must traverse the ring identically or the floor stops being a ratio of like to
        like, which is the mismatch item 27 was opened for. If one is changed, change both.
        """
        c = pts_xz.mean(axis=0)
        d = pts_xz - c
        p = pts_xz[np.argsort(np.arctan2(d[:, 1], d[:, 0]))]
        return float(np.hypot(*(np.roll(p, -1, axis=0) - p).T).sum())

    def press_to_force(self, f_setpoint, max_frames=900, tol_n=0.05, settle_frames=30,
                       adm=None, log=None, v_tol=5.0e-4):
        """Close the admittance loop until the contact force holds `f_setpoint`.

        Returns (converged, frames_used, controller). Convergence requires the FILTERED force
        to stay within tol_n of setpoint for settle_frames consecutive control periods -- a
        single sample landing on target is not convergence, which is precisely the error the
        open-loop rig made when it read force at a fixed frame count.

        ⚠️ AND IT REQUIRES THE PROBE TO BE AT REST AS WELL. Testing the FILTERED force alone
        was not enough: at tau_f = 0.15 s the low-pass smooths over ~9 frames, so the filter
        can sit on setpoint for the full 30-period window while the raw force still swings
        +/-100 N and the probe still moves at tens of mm/s. That is the same non-settled
        state Chapter 6 diagnosed, hidden one layer further down. `v_tol` makes it a
        condition rather than something to notice later.
        """
        adm = adm or Admittance(dt=self.frame_dt)
        adm.q = self._target
        hits = 0
        for k in range(max_frames):
            self.set_target(adm.step(f_setpoint, self.force()))
            self.step()
            f = adm.f_filt
            at_rest = self.probe_speed() < v_tol
            hits = hits + 1 if (abs(f - f_setpoint) <= tol_n and at_rest) else 0
            if log is not None and k % 25 == 0:
                log(k, f, self.force(), adm.q, adm.v)
            if hits >= settle_frames:
                return True, k + 1, adm
        return False, max_frames, adm

    def set_target(self, dz):
        self._target = float(dz)
        t = self.control.joint_target_pos.numpy()
        t[0] = self._target
        self.control.joint_target_pos.assign(t)

    def step(self):
        # clear_forces() zeroes particle_f every substep, so the pressure load has to be
        # re-applied inside the loop -- but it is COMPUTED once per frame (see
        # _refresh_lumen_pressure) and only copied device-to-device here.
        # a static known load (set_static_load) uses the same buffer and the same copy, but
        # is NOT recomputed per frame -- it is a fixed external force by construction.
        pressurised = self._refresh_lumen_pressure() or getattr(self, "_static_load", False)
        for _ in range(self.sub):
            self.s0.clear_forces()
            if pressurised:
                wp.copy(self.s0.particle_f, self._pf)
            self.cp.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, self.frame_dt / self.sub)
            self.s0, self.s1 = self.s1, self.s0

    def force(self):
        """Force balance on the driven joint -- the same actuator-side estimator Chapter 6
        calibrated to 21 mN, used because SensorContact.total_force reads 0 under VBD."""
        z = float(self.s0.body_q.numpy()[self.body][2])
        vz = float(self.s0.body_qd.numpy()[self.body][2])
        q = z - self.meta["start_z"]
        return -(2.0e5 * (self._target - q) - 2.0e3 * vz)

    def force_terms(self):
        """The same estimate, split into its two terms. Returns (f, ke_term, kd_term).

        🚨 WHY THIS EXISTS. force() is ke*lag + kd*velocity, and those two are diagnostically
        completely different things. The ke term is a POSITION measurement scaled by 2e5 N/m;
        the kd term is a VELOCITY measurement scaled by 2e3 N.s/m. At the 2026-08-04 open-loop
        scale, kd*qd alone was 120 N at the drive rate used. At the 2026-08-06 closed-loop
        scale the probe still swings +/-2 mm/s during the "hold", and 2e3 * 0.002 = 4 N --
        which is essentially the whole +/-5.23 N scatter that made the channel unquotable.
        Reporting only the sum makes a motion artefact indistinguishable from a contact force,
        which is exactly how 400 N of drive lag got published as a tissue reaction.

        Splitting them turns "is the noise physical?" into something the log answers directly:
        if kd_term carries the variance, the channel is reading the probe's own limit cycle
        and the fix is in the CONTROLLER (m_a, b_a, tau_f), not in the tissue or the solver.
        """
        z = float(self.s0.body_q.numpy()[self.body][2])
        vz = float(self.s0.body_qd.numpy()[self.body][2])
        q = z - self.meta["start_z"]
        ke_term = -2.0e5 * (self._target - q)
        kd_term = 2.0e3 * vz
        return ke_term + kd_term, ke_term, kd_term

    def heights(self):
        pq = self.s0.particle_q.numpy()
        return {n: float(pq[hi][2] - pq[lo][2]) for n, (lo, hi) in self.pairs.items()}

    def rest_heights(self):
        return {n: float(self.rest[hi][2] - self.rest[lo][2]) for n, (lo, hi) in self.pairs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.0015, help="node spacing, m")
    ap.add_argument("--substeps", type=int, default=96)
    ap.add_argument("--probe-w", type=float, default=0.040, help="probe footprint across vessels, m")
    ap.add_argument("--probe-len", type=float, default=0.012, help="probe footprint along vessels, m")
    ap.add_argument("--max-mm", type=float, default=16.0)
    ap.add_argument("--settle", type=int, default=60)
    ap.add_argument("--save-states", default="0,4,8,12,16",
                    help="mm of probe travel at which to dump node positions for mesh warping")
    ap.add_argument("--probe-theta", type=float, default=0.0,
                    help="probe contact angle around the limb, deg. 0 = anterior pole, "
                         "positive toward the vein. SAME CONVENTION as "
                         "sweep_circumference.py. Note the pre-2026-08-04 hard-coded pose "
                         "was +6.05 deg, not 0.")
    ap.add_argument("--pin-sides", type=int, choices=(0, 1), default=1,
                    help="clamp the ROI's lateral faces. Was a module constant with a "
                         "refuted rationale attached; freeing them makes the vein close "
                         "LESS (87.8%% -> 92.0%%), so this is a modelling choice, not a fix.")
    ap.add_argument("--anatomy", choices=("v2", "v3"), default="v2",
                    help="v2 = the built-in hand-typed cross-section, kept so every run "
                         "before 2026-08-08 reproduces. v3 = measured off the CAD by "
                         "v3_section.py, which adds the femur, the sheath envelope and a "
                         "real axial station.")
    ap.add_argument("--station-s", type=float, default=30.0,
                    help="v3 only: mm distal to the inguinal ligament. 30 = saphenofemoral "
                         "junction, Zambetti's measurement plane. ⚠️ AT 30 THERE IS NO NERVE "
                         "(ends at s = 25) AND NO FEMORAL CANAL (12.5) -- that is anatomy, "
                         "not a slicing bug. Use <= 25 to keep the nerve.")
    ap.add_argument("--section", default=None,
                    help="v3 only: explicit path to a v3_section.py JSON, overriding the "
                         "one --station-s would pick.")
    ap.add_argument("--sheath-kpa", type=float, default=None,
                    help="v3 only, REQUIRED, and it is the single number that separates the "
                         "two legs of the 2026-08-08 A/B.\n"
                         "🚨 materials.py leaves Sheath E_pa = None ON PURPOSE -- the wall "
                         "thickness and modulus of the femoral sheath are unsourced (spec "
                         "section 3: Gray's is pre-metric, Moore gives shape and length but "
                         "no thickness, Fenech gives compartments but no dimension). So a "
                         "value MUST be chosen here rather than defaulted, and it must be "
                         "reported as a modelling choice.\n"
                         "📌 What the CAD body actually is: a SOLID ENVELOPE enclosing both "
                         "vessels, ~32 x 13 mm, not a thin fascial shell. So this modulus is "
                         "the stiffness of what fills the sheath, and the honest A/B is:\n"
                         "   20.0 = muscle, i.e. exactly what V2 had abutting the vessels "
                         "(leg A, 'as-built' -- materials.py already maps Sheath to the "
                         "raysim `muscle` material, so this invents nothing)\n"
                         "    2.0 = loose areolar / adipose, the spec's LooseTissue and the "
                         "same value materials.py already carries for FemoralCanal (leg B)\n"
                         "Neither number is new. What is new is that they are now the only "
                         "difference between two otherwise identical runs.")
    ap.add_argument("--loose-cuff-mm", type=float, default=0.0,
                    help="v3 only: thickness of a perivascular collar added to the sheath "
                         "region, mm. 0 = the CAD envelope exactly as authored.\n"
                         "🚨 WHY THIS EXISTS: V3's sheath does NOT cover the probe-facing "
                         "side of the vessels. Measured at s = 30, its anterior face is at "
                         "Z = 72.52 mm against wall tops of 73.99 (artery) and 74.29 (vein), "
                         "so only 55.3%% / 71.8%% of their perimeters are inside it and the "
                         "gap is centred on the anterior arc. The column from skin to vessel "
                         "reads SoftTissue -> Muscle -> wall, with no sheath in it at all. "
                         "Softening the envelope alone therefore puts compliance where the "
                         "press does not act. Clipped to the muscle core so it cannot thin "
                         "the 16 mm fat shell. 2.0 is the largest collar that stays inside "
                         "the muscle at this station.")
    ap.add_argument("--force-n", type=float, default=None,
                    help="press to this CONTACT FORCE with the admittance controller instead "
                         "of driving open-loop to a fixed travel. 🚨 THE OPEN-LOOP LADDER "
                         "BELOW IS THE PATH THAT PRODUCED EVERY UNTRUSTWORTHY FORCE NUMBER "
                         "IN CHAPTER 8 -- it drives a 2e5 N/m spring to a position and reads "
                         "ke*lag as if it were a contact force. Use this for anything "
                         "quotable; the ladder is kept only to reproduce old runs.")
    ap.add_argument("--b-a", type=float, default=900.0, help="admittance damping, N.s/m")
    ap.add_argument("--tau-f", type=float, default=0.15, help="force low-pass, s")
    ap.add_argument("--artery-kpa", type=float, default=0.0,
                    help="transmural pressure on the ARTERY lumen wall, kPa. 🔴 MODELLING "
                         "CHOICE -- textbook mean arterial pressure is ~12-13 kPa "
                         "(90-100 mmHg) but NOTHING in i4h-references reports it. Default 0 "
                         "keeps the pre-2026-08-06 bare-void behaviour.")
    ap.add_argument("--vein-kpa", type=float, default=0.0,
                    help="transmural pressure on the VEIN lumen wall, kPa. 🔴 MODELLING "
                         "CHOICE -- ~0.7-1.3 kPa (5-10 mmHg), unsourced as above.")
    ap.add_argument("--pressure-ramp", type=int, default=60,
                    help="frames over which to bring the lumen pressure up before the rest "
                         "areas are re-baselined")
    args = ap.parse_args()

    global THETA, PIN_SIDES
    THETA = np.radians(args.probe_theta)
    PIN_SIDES = bool(args.pin_sides)

    if args.anatomy == "v3":
        if args.sheath_kpa is None:
            raise SystemExit(
                "!!! --anatomy v3 requires --sheath-kpa, and that is deliberate.\n"
                "    materials.py leaves Sheath E_pa = None because nothing on disk gives a\n"
                "    femoral sheath modulus. Defaulting it here would launder a modelling\n"
                "    choice into a result. Pass 20 (muscle, leg A) or 2 (loose areolar, leg B).")
        sec = args.section or str(pathlib.Path(__file__).parent / "out" / "femoral_v3" /
                                  f"section_s{int(round(args.station_s)):03d}.json")
        if not pathlib.Path(sec).exists():
            raise SystemExit(f"!!! no section at {sec}\n"
                             f"    build it first:  $PY_ANATOMY "
                             f"v3_section.py --station-s {args.station_s:g}")
        BODIES["Sheath"]["E_pa"] = args.sheath_kpa * 1e3
        # nu is None alongside E_pa for the same reason -- nobody sourced it either. 0.45 is
        # what every other soft body here uses (near-incompressible), so the sheath is not
        # made special by a second silent choice.
        BODIES["Sheath"]["nu"] = NU
        BODIES["Sheath"]["E_src"] = "choice"
        BODIES["Sheath"]["E_note"] = (f"🔴 SET AT RUN TIME to {args.sheath_kpa:g} kPa via "
                                      f"--sheath-kpa. Unsourced; see that flag's help.")
        doc = load_section(sec, loose_cuff=args.loose_cuff_mm * MM)
        print(f">>> anatomy V3, station s = {doc['station_s_mm']:.1f} mm "
              f"(LPS z = {doc['section_lps_z_mm']:+.1f}), from {doc['mesh_dir']}")
        print(f">>> sheath envelope E = {args.sheath_kpa:g} kPa  🔴 modelling choice"
              + (f", + {args.loose_cuff_mm:g} mm perivascular collar"
                 if args.loose_cuff_mm > 0 else ", CAD envelope exactly as authored"))
        print(f">>> ROI  X [{X0*1e3:+.1f}, {X1*1e3:+.1f}]  Z [{Z0*1e3:+.1f}, {R_SKIN*1e3:+.1f}] mm "
              f"-- floor is inside the FEMUR, not an artificial clamp behind the vein")
        if doc["absent"]:
            print(">>> not present at this station: " + "; ".join(doc["absent"]))

    wp.init()
    ph = Phantom(args.dx, args.substeps, args.probe_w, args.probe_len)
    m = ph.meta
    print(f"nodes={m['n_nodes']:,}  tets={m['n_tets']:,}  pinned={m['n_pinned']:,}  "
          f"grid {m['nx']}x{m['ny']}x{m['nz']}  dx={args.dx*1e3:.2f} mm")
    print(f"probe: {args.probe_w*1e3:.0f} x {args.probe_len*1e3:.0f} mm flat face, "
          f"theta={args.probe_theta:+.2f} deg, skin contact z={m['z_skin']*1e3:.2f} mm")

    # At angles where the bundle has swept out of the window there is nothing to measure,
    # and the collapse columns below index the trackers directly. Say so and stop, rather
    # than dying in a KeyError -- the swept driver (sweep_fem.py) is what handles the
    # nothing-in-view case properly.
    missing = [n for n in ("vein", "artery", "nerve") if n not in ph.pairs]
    if missing:
        raise SystemExit(
            f"\nat theta={args.probe_theta:+.2f} deg the following are outside the simulated "
            f"window: {', '.join(missing)}.\nThis is expected past roughly +/-20 deg -- the "
            f"structures sit ~70 mm out from the limb axis and the window is ~50 mm wide.\n"
            f"For a swept study use sweep_fem.py, which records 'nothing in view' as a "
            f"result instead of an error.")
    cfl_report(args.dx, args.substeps, 1.0 / 60.0)

    rest = ph.rest_heights()
    rest_a = ph.areas(ph.rest)
    print(f"\nrest heights: vein {rest['vein']*1e3:.2f} mm  artery {rest['artery']*1e3:.2f} mm  "
          f"nerve {rest['nerve']*1e3:.2f} mm")
    print(f"rest lumen area: vein {rest_a['vein']*1e6:.2f} mm^2 "
          f"(ideal {np.pi*(VEIN_RI*1e3)**2:.2f})   artery {rest_a['artery']*1e6:.2f} mm^2 "
          f"(ideal {np.pi*(ARTERY_RI*1e3)**2:.2f})   ring nodes "
          f"vein={len(ph.rings['vein'])} artery={len(ph.rings['artery'])}")
    floor = ph.closure_floor_pct()
    print("MEASUREMENT FLOOR (self-contact + dx, NOT tissue): "
          + "  ".join(f"{k} >= {v:.0f}% of rest" for k, v in floor.items())
          + "   <- the spec's '~0%' acceptance criterion is below this floor")

    # ── transmural pressure, if asked for ────────────────────────────────────────────────
    # Ramp first, then RE-BASELINE the rest areas: a pressurised artery is larger than its
    # CAD lumen, and comparing a pressurised deformed state against an unpressurised rest
    # would report the artery at >100% and silently flatter it.
    if args.artery_kpa or args.vein_kpa:
        ph.set_lumen_pressure(artery_pa=args.artery_kpa * 1e3, vein_pa=args.vein_kpa * 1e3)
        print(f"\ntransmural pressure 🔴 (modelling choice, unsourced): "
              f"artery {args.artery_kpa:.2f} kPa, vein {args.vein_kpa:.2f} kPa "
              f"-- ramping over {args.pressure_ramp} frames")
        ph.ramp_pressure(args.pressure_ramp)
        for _ in range(args.settle * 2):
            ph.step()
        for name, s in ph.pressure_summary().items():
            print(f"  {name:<7} p={s['pa']/1e3:5.2f} kPa  perimeter {s['perimeter_mm']:5.2f} mm"
                  f"  total outward load {s['total_N']*1e3:6.1f} mN")
        rest_a = ph.areas()
        rest = ph.heights()
        print(f"  re-baselined rest lumen area: vein {rest_a['vein']*1e6:.2f} mm^2  "
              f"artery {rest_a['artery']*1e6:.2f} mm^2")

    ph.step()
    tare = ph.force()

    if args.force_n is not None:
        # ── admittance mode: depth is an OUTPUT ─────────────────────────────────────────
        ok, frames, adm = ph.press_to_force(
            args.force_n, max_frames=900, tol_n=0.4,
            adm=Admittance(b_a=args.b_a, tau_f=args.tau_f, dt=ph.frame_dt))
        f_mean, f_sd, v, at_rest = ph.force_settled(tare=tare)
        h, a = ph.heights(), ph.areas()
        print(f"\nsetpoint {args.force_n:.1f} N: "
              f"{'CONVERGED' if ok else 'DID NOT CONVERGE'} in {frames} frames\n"
              f"  achieved depth {ph.probe_depth()*1e3:6.3f} mm "
              f"(controller target {-adm.q*1e3:6.3f} mm, lag "
              f"{(ph.probe_depth()+adm.q)*1e3:+.3f} mm)\n"
              f"  settled force {f_mean:.2f} +/- {f_sd:.2f} N at |v| = {v*1e3:.3f} mm/s "
              f"{'(AT REST)' if at_rest else '🚨 (STILL MOVING -- not a contact force)'}\n"
              f"  vein {100*a['vein']/rest_a['vein']:.1f}%  "
              f"artery {100*a['artery']/rest_a['artery']:.1f}%  "
              f"nerve {100*h['nerve']/rest['nerve']:.1f}%  "
              f"(floor {floor.get('vein', 0):.0f}%)")
        json.dump(dict(mode="force", setpoint_N=args.force_n, converged=bool(ok),
                       frames=int(frames), depth_mm=ph.probe_depth() * 1e3,
                       force_mean_N=f_mean, force_sd_N=f_sd, probe_speed_mm_s=v * 1e3,
                       at_rest=at_rest,
                       artery_kpa=args.artery_kpa, vein_kpa=args.vein_kpa,
                       area_pct={k: 100.0 * a[k] / rest_a[k] for k in a},
                       closure_floor_pct=floor, meta=m),
                  open(OUT / "femoral_force.json", "w"), indent=2)
        print(f"\nwrote {OUT/'femoral_force.json'}")
        return

    saves = {float(s) for s in args.save_states.split(",") if s != ""}
    states, rows = {}, []
    print("\n🚨 OPEN-LOOP POSITION LADDER -- THE FORCE COLUMN BELOW IS NOT A CONTACT FORCE.\n"
          "   It is ke*(target - q) + kd*qd on a 2e5 N/m drive that never settles; Chapter 8\n"
          "   published -0.23, 5.69, 31.54, 7.72, 148.90 N from this path across smoothly\n"
          "   increasing indentation. Use --force-n for anything quotable. The geometry\n"
          "   columns are read from node positions and are unaffected.")
    print(f"\n{'travel_mm':>10}{'F (N)':>9}{'veinA%':>9}{'artA%':>8}{'nerve%':>8}"
          f"{'vein_mm':>9}{'artery_mm':>10}   <- area %, then vertical diameter")
    if 0.0 in saves:
        states["0.0"] = ph.s0.particle_q.numpy().copy()
    for tgt in np.arange(1.0, args.max_mm + 1e-9, 1.0):
        ph.set_target(-tgt * MM)
        for _ in range(args.settle):
            ph.step()
        h = ph.heights()
        a = ph.areas()
        F = ph.force() - tare
        # 🚨 Self-flag the reading instead of letting it pass as a contact force. force() is
        # exact only at rest; open-loop position driving never settles (Chapter 6, 2026-08-04:
        # 2.9 s of hold with the target FIXED and the probe still swinging +/-40 mm/s). The
        # probe speed at the moment of the read is now recorded next to the force, so a
        # non-settled number is visible in the results file rather than in a GIF a day later.
        v_read = ph.probe_speed()
        row = dict(travel_mm=float(tgt), force_N=float(F),
                   probe_speed_mm_s=float(v_read * 1e3),
                   force_at_rest=bool(v_read < 1e-4),
                   probe_depth_mm=float(ph.probe_depth() * 1e3),
                   **{f"{k}_mm": h[k] * 1e3 for k in h},
                   **{f"{k}_pct": 100.0 * h[k] / rest[k] for k in h},
                   **{f"{k}_area_mm2": a[k] * 1e6 for k in a},
                   **{f"{k}_area_pct": 100.0 * a[k] / rest_a[k] for k in a})
        # Divergence guard. A first run at 144 substeps tracked cleanly to 11 mm of travel
        # and then blew up at 13 mm -- reporting a lumen at 1642% of its rest area and a
        # NEGATIVE artery diameter, i.e. inverted elements. Without this guard those values
        # land in the results file looking like data. Stop at the first sign and keep
        # everything measured up to that point.
        pq = ph.s0.particle_q.numpy()
        diverged = (not np.isfinite(pq).all()
                    or row["vein_area_pct"] > 120.0 or row["artery_area_pct"] > 120.0
                    or h["vein"] < 0 or h["artery"] < 0)
        if diverged:
            print(f"{tgt:10.1f}   🚨 DIVERGED (inverted elements) -- sweep stopped here; "
                  f"last trustworthy travel = {tgt-1:.1f} mm")
            rows.append(dict(row, diverged=True))
            break
        rows.append(row)
        print(f"{tgt:10.1f}{F:9.2f}{row['vein_area_pct']:9.1f}{row['artery_area_pct']:8.1f}"
              f"{row['nerve_pct']:8.1f}{h['vein']*1e3:9.3f}{h['artery']*1e3:10.3f}")
        if float(tgt) in saves:
            states[f"{tgt:.1f}"] = ph.s0.particle_q.numpy().copy()

    ph.set_target(0.0)
    for _ in range(200):
        ph.step()
    rel, rel_a = ph.heights(), ph.areas()
    print(f"\nafter release (elastic recovery): "
          f"vein area {100*rel_a['vein']/rest_a['vein']:.1f}%  "
          f"artery area {100*rel_a['artery']/rest_a['artery']:.1f}%  "
          f"nerve {100*rel['nerve']/rest['nerve']:.1f}%")

    np.savez_compressed(OUT / "fem_states.npz", rest=ph.rest,
                        **{f"s{k}": v for k, v in states.items()})
    json.dump(dict(rows=rows, rest={k: v * 1e3 for k, v in rest.items()},
                   rest_area_mm2={k: v * 1e6 for k, v in rest_a.items()},
                   released={k: v * 1e3 for k, v in rel.items()},
                   released_area_pct={k: 100.0 * rel_a[k] / rest_a[k] for k in rel_a},
                   meta=m, dx=args.dx, substeps=args.substeps,
                   E={k: (b["E_pa"] or 0) for k, b in BODIES.items()}),
              open(OUT / "femoral_collapse.json", "w"), indent=2)
    print(f"\nwrote {OUT/'femoral_collapse.json'} and fem_states.npz "
          f"({len(states)} saved states: {sorted(states)})")


if __name__ == "__main__":
    main()
