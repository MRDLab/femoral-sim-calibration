"""Convert the V4 SolidWorks STL export into the OBJ set `v3_section.py` reads — and reposition
the pelvic ramus while doing it.

    $PY_ANATOMY anatomy_meshing/v4_to_section.py --ramus-gap-mm 12

🚨 THIS IS A FRAME CONVERSION, NOT A FILE COPY, AND THAT IS THE DANGEROUS PART.
The V4 export and the V3 pipeline do not share an axial axis:

    V4 STL export     x = medio-lateral   y = AXIAL (s increases distally)   z = anterior (+ = skin)
    v3_section.py     expects LPS, with the AXIAL axis on z and s = z_ligament - z,
                      and hands the FEM  X = LPS x,  Z = -LPS y

so the mapping is  **LPS = (-V4.x, -V4.z, -V4.y)**. A wrong sign here mirrors the bundle or
flips proximal for distal, and *both* look entirely plausible in a render — which is exactly
the failure that cost 2026-08-07 (the leg USD was V3 while the FEM solved V2 and every
assertion passed). So every geometric quantity that can be checked against a measurement taken
BEFORE the transform is checked after it, and the run refuses to write if any disagrees.

## What is repositioned, and why each move is or is not a judgement call

Measured on the as-exported V4 (s = 0 at the shell's proximal face):

    CFA    s   0.0 ..  37.5     ✅ matches the spec
    CFV    s   0.0 ..  80.0     ✅
    Sheath s   0.0 ..  35.0     ✅
    Femur  s -58.1 .. 120.0     ✅ head/neck proudly proximal of the thigh segment
    Ramus  s -36.4 .. -22.6     🚨 ENTIRELY OUTSIDE THE LIMB

and laterally the ramus sits **57.9 mm medial** of the vein with only a 6.5 mm x-extent, so it
does not reach under the vessel it is supposed to back up.

    move in y   NOT a judgement call. A backstop outside the meshed volume is not a backstop.
    move in x   NOT a judgement call. A backstop the vein does not sit over is not a backstop.
    move in z   🔴 A JUDGEMENT CALL, AND THE ONE NOBODY HAS SOURCED. `--ramus-gap-mm` is
                therefore a PARAMETER with no defensible default. Sweep it. The experiment
                "vein closure vs backstop gap" turns the missing number into a result, and says
                how close the bone must be for the clinical manoeuvre to work at all.

⚠️ The ramus is also only 0.31 cm³ (bbox 6.5 x 13.7 x 13.8 mm), which is a stub rather than a
ramus. `--ramus-scale` stretches it along the limb axis so it spans the access window; that is a
**modelling choice** pending Jarragh's dimensions and is logged as one.

## 🚨 THE AXIAL ORIGIN — ONE CONVENTION, STATED, ASSERTED, AND SHARED WITH V3

Until 2026-08-13 the recentre below moved x and y and left z alone, under a trailing comment
that claimed "axial (z) origin is the ligament plane". It was not. Every V4 OBJ sat at LPS
z ≈ −1131 … −954 mm (the SolidWorks assembly origin) where every V3 OBJ sits at z ∈ [−100, +20],
and one un-applied translation broke every consumer that maps an ABSOLUTE z:

    Isaac room render   the leg authored at world Y ≈ −1.05 m; the placement gate's Y-overhang
                        refusal was a TRUE POSITIVE and was right to refuse
    circumferential sweep  `probe_pose()` pins the probe to z = 0, which on V4 was s = −1011.7 mm
                        -- a metre proximal of the phantom, in open background
    B-mode deform chain `deform_meshes.slab_and_refine` rebuilt every body over
                        `zs = linspace(±half_len)`, discarding input z, and would have silently
                        TELEPORTED the V4 anatomy 1042 mm rather than errored

📌 THE CONVENTION, and it is `build_v3_anatomy.py`'s, adopted verbatim rather than improved on:

    **LPS z = 0 IS the imaging plane, at s = `--image-plane-s`; the inguinal ligament plane
    (s = 0) therefore sits at LPS z = +`--image-plane-s`.**

With the shared default of 20.0 that puts the ligament at z = +20.0, exactly where V3 puts it
(`build_v3_anatomy.py:377`, `z_shift = -(lig_lps_z - image_plane_s)`), so V3-vs-V4 comparisons
are in one frame. ⚠️ "make z = 0 mean s = 0" is arguably cleaner and is NOT the same convention;
adopting it would silently move every V3-vs-V4 comparison already on disk, so it is not adopted.
The shift is ASSERTED (raise, not warn) after every body is built, and printed.

✅ THE PRESS LADDERS ARE NOT AFFECTED AND MUST NOT BECOME AFFECTED. `v3_section.py:222` derives
the station from `SoftTissue.bounds[1][2]` -- the mesh's own superior end -- not from an absolute
origin, so the section's `bodies` block (the only part `femoral_fem.load_section` reads for
physics) is invariant under this shift. `ligament_lps_z_mm` and `section_lps_z_mm` are metadata
that femoral_fem only prints, and they DO change: that is the shift becoming visible, not the
physics moving. Re-generate and diff the `bodies` block before believing any of this.
"""

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
import pathlib
import sys

import numpy as np

# Gap between a synthesised lumen and the wall bore it sits in, mm. NOT a tuning knob: raysim
# resolves overlapping bodies one level deep and coincident surfaces are undefined behaviour, so
# this only has to exceed the discretisation mismatch between a 64-gon lumen and the CAD wall's
# own facets (measured worst case 0.0075 mm). Chapter 8 used 0.02 mm on V2 for the same reason.
LUMEN_DECOINCIDE_MM = 0.025
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import materials as MAT  # noqa: E402  -- THE body table and THE anatomy_config writer

SRC = pathlib.Path(os.environ.get("V4_STL_DIR", "V4_STL_DIR-not-set"))   # FemoralV4/ until 2026-10-22, renamed V4/
# 🚨 ONE DIRECTORY PER GAP, WRITTEN DIRECTLY. `out/femoral_v4` was a scratch directory that the
# campaign wrote three times and copied out of afterwards (run_v4_campaign.sh:63-65), so what
# survived on disk was the LAST gap's OBJs described by whichever section had been written last:
# gap20 meshes under a gap12 section, and nothing in either artifact said so. That is defect
# class 1 -- two artifacts and nothing reconciling them -- manufactured by the build script.
OUT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "out"

# V4 part name -> the body name v3_section.py's NEST_ORDER expects
NAME = {"Leg_Shell": "SoftTissue", "CFA": "Artery", "CFV": "Vein", "Nerve": "Nerve",
        "Sheathe": "Sheath", "Canal": "FemoralCanal", "Femur": "Femur",
        "Pelvic_Ramus": "PubicRamus"}


def to_lps(m):
    """V4 export -> LPS. LPS = (-x, -z, -y).

    🚨 THE SIGN ON x WAS MISSING UNTIL 2026-08-18 AND IT MIRRORED THE PATIENT.
    The previous mapping was `(x, -z, -y)`, whose matrix

        [ 1  0  0 ]
        [ 0  0 -1 ]      det = -1
        [ 0 -1  0 ]

    has determinant **-1**: a reflection, not a rotation. The old code noticed -- its comment
    read "the axis swap flips handedness; restore outward normals" -- and called `.invert()`,
    which fixes the FACE WINDING so normals still point outward. It does not undo the
    geometric reflection. The normals were repaired and the anatomy was left mirrored.

    Measured consequence: the neurovascular bundle came out lateral-to-medial as
    **vein, artery, nerve**, i.e. a LEFT thigh, where V2 and V3 both read nerve, artery, vein
    -- a RIGHT thigh -- and where the CAD was authored as a right leg. The femoral canal
    confirmed it independently: it is the MEDIAL compartment and it sat on the lateral side.

    🪤 WHY NOTHING CAUGHT IT. A reflection is exactly right in every single-body check --
    each vessel is at the correct depth, the correct distance from the axis, the correct
    diameter. It is only wrong in the ORDER, and nothing asserted the order. `render_deformed`
    even recorded "V4 is the exact MIRROR of V3 in x" and treated it as a probe-placement bug.

    Negating x restores determinant +1 and puts the bundle back at V3's positions to within a
    few tenths of a millimetre. `.invert()` is no longer needed and would now BREAK the normals.
    """
    v = m.vertices.copy()
    out = m.copy()
    out.vertices = np.column_stack([-v[:, 0], -v[:, 2], -v[:, 1]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ramus-gap-mm", type=float, default=None,
                    help="gap from the CFV FAR wall to the anterior surface of the ramus. "
                         "🔴 UNSOURCED -- there is no default and there should not be one. "
                         "Sweep it.")
    ap.add_argument("--ramus-scale-s", type=float, default=4.0,
                    help="stretch the ramus along the limb axis so it spans the access window. "
                         "🔴 modelling choice pending Jarragh's dimensions.")
    ap.add_argument("--station-s", type=float, default=30.0)
    # 🪤 THESE TWO DEFAULTS ARE NOT FREE CHOICES -- THEY ARE build_v3_anatomy.py's, COPIED.
    # They fix where LPS z = 0 lands, and every renderer puts the probe at z = 0. Changing
    # either here without changing it there puts V3 and V4 in different frames, which is the
    # failure this whole file's witness block exists to prevent. See the module docstring.
    ap.add_argument("--image-plane-s", type=float, default=20.0,
                    help="the station, mm distal to the inguinal ligament, that sits at "
                         "LPS z = 0. MUST match build_v3_anatomy.py --image-plane-s.")
    ap.add_argument("--target-plane-s", type=float, default=20.0,
                    help="station at which the annotation targets are measured. MUST match "
                         "build_v3_anatomy.py --target-plane-s.")
    ap.add_argument("--background", default="air",
                    help="raysim background. `air` is the directive since 2026-08-11; the "
                         "coupling is the explicit gel band gel_body.py builds, not a bath.")
    ap.add_argument("--out-root", default=str(OUT_ROOT),
                    help="parent of the per-gap output directory femoral_v4_gap<G>")
    args = ap.parse_args()
    if args.ramus_gap_mm is None:
        raise SystemExit(
            "!!! --ramus-gap-mm is required and deliberately has no default.\n"
            "    It is the CFV-far-wall -> anterior-ramus distance, the one number nobody has\n"
            "    sourced, and it decides whether the vein can close. Pick it explicitly and it\n"
            "    goes in the record as a stated modelling choice; sweep it and it becomes a\n"
            "    result. See Phantoms/Femoral/V4/V4_PARAMETRIC_SPEC.md section 4.")
    OUT = pathlib.Path(args.out_root) / f"femoral_v4_gap{args.ramus_gap_mm:g}"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f">>> output directory: {OUT}  (one per gap -- see the OUT_ROOT comment)")

    raw = {}
    for f in sorted(SRC.glob("*.STL")):
        stem = f.stem.replace("PhantomV4 - ", "").replace("-1", "")
        if stem in NAME:
            raw[stem] = trimesh.load(f, process=False)
    missing = set(NAME) - set(raw)
    if missing:
        raise SystemExit(f"!!! missing V4 parts: {sorted(missing)}")

    # ── measurements taken BEFORE the transform, to check it afterwards ──────────────────
    shell, cfv, cfa, ram = raw["Leg_Shell"], raw["CFV"], raw["CFA"], raw["Pelvic_Ramus"]
    skin_v4 = float(shell.bounds[1][2])
    pre = dict(cfv_dia=float(cfv.bounds[1][0] - cfv.bounds[0][0]),
               cfa_dia=float(cfa.bounds[1][0] - cfa.bounds[0][0]),
               cfv_near_depth=skin_v4 - float(cfv.bounds[1][2]),
               cfv_far_depth=skin_v4 - float(cfv.bounds[0][2]),
               # 🪤 MAGNITUDE, NOT SIGNED DIFFERENCE. A rigid transform preserves DISTANCES;
               # it does not preserve a signed coordinate difference along an axis it negates.
               # This witness used to compare the signed value, which is preserved only by a
               # transform that leaves x alone -- i.e. it passed for the MIRRORING map and
               # blocked the fix for it. See to_lps.
               cfa_cfv_separation=abs(float(cfa.bounds.mean(axis=0)[0]
                                            - cfv.bounds.mean(axis=0)[0])),
               s_len=float(shell.bounds[1][1] - shell.bounds[0][1]))
    s0_y = float(shell.bounds[0][1])
    print(">>> as exported (V4 frame): CFV Ø %.2f, CFA Ø %.2f, CFV near-wall %.2f mm deep, "
          "CFA-CFV separation %.1f mm" % (pre["cfv_dia"], pre["cfa_dia"], pre["cfv_near_depth"],
                                          pre["cfa_cfv_separation"]))

    # ── reposition the ramus, in the V4 frame where the measurements above live ─────────
    R = ram.copy()
    c_before = R.bounds.mean(axis=0)
    # stretch along the limb axis (y) about its own centre so it spans the access window
    V = R.vertices.copy()
    V[:, 1] = c_before[1] + (V[:, 1] - c_before[1]) * args.ramus_scale_s
    R.vertices = V
    # x: centre it under the CFV.  y: centre it on the access window s = 0..40.
    # z: anterior surface placed --ramus-gap-mm below the CFV far wall.
    dx = float(cfv.bounds.mean(axis=0)[0]) - float(R.bounds.mean(axis=0)[0])
    dy = (s0_y + 20.0) - float(R.bounds.mean(axis=0)[1])
    # 🪤 ORDER MATTERS. The z placement is measured on the ramus's cross-section AT THE STATION,
    # so the y move has to happen FIRST -- otherwise the station band is sampled where the ramus
    # is not, and the gate fires on a ramus that is about to be perfectly well placed.
    R.apply_translation([dx, dy, 0.0])
    y_st = s0_y + args.station_s

    def top_at(mesh):
        """Most anterior point of the ACTUAL cross-section at the station.

        🪤 A +/-3 mm band of VERTICES was the first attempt and it was wrong by 9 mm. The ramus
        is an oblique bar with a coarse tessellation, so the highest vertex within 3 mm of the
        plane is not the surface height AT the plane -- and the converter then reported a 12 mm
        gap while v3_section, which takes a real section, measured 21 mm. Two artifacts
        disagreeing about the same distance, which is this project's oldest bug. Slice it."""
        sec = mesh.section(plane_origin=[0.0, y_st, 0.0], plane_normal=[0.0, 1.0, 0.0])
        if sec is None:
            raise SystemExit(f"!!! the ramus has no cross-section at s = {args.station_s:g} mm "
                             f"(spans s {mesh.bounds[0][1]-s0_y:.1f}.."
                             f"{mesh.bounds[1][1]-s0_y:.1f})")
        return float(np.asarray(sec.vertices)[:, 2].max())

    dz = (float(cfv.bounds[0][2]) - args.ramus_gap_mm) - top_at(R)
    R.apply_translation([0.0, 0.0, dz])
    print(">>> ramus moved  dx %+.1f  dy %+.1f  dz %+.1f mm  (stretched x%.1f along s)"
          % (dx, dy, dz, args.ramus_scale_s))
    print("    now spans s %.1f .. %.1f, x %.1f .. %.1f; AT s=%g the anterior surface is "
          "%.1f mm below the CFV far wall"
          % (R.bounds[0][1] - s0_y, R.bounds[1][1] - s0_y, R.bounds[0][0], R.bounds[1][0],
             args.station_s, float(cfv.bounds[0][2]) - top_at(R)))
    raw["Pelvic_Ramus"] = R

    # ── transform, and CHECK the transform ───────────────────────────────────────────────
    lps = {NAME[k]: to_lps(v) for k, v in raw.items()}
    sh, vn, ar = lps["SoftTissue"], lps["Vein"], lps["Artery"]
    z_lig = float(sh.bounds[1][2])
    post = dict(cfv_dia=float(vn.bounds[1][0] - vn.bounds[0][0]),
                cfa_dia=float(ar.bounds[1][0] - ar.bounds[0][0]),
                # FEM Z = -LPS y, and the skin apex is FEM Z max
                cfv_near_depth=float(-sh.bounds[0][1]) - float(-vn.bounds[0][1]),
                cfa_cfv_separation=abs(float(ar.bounds.mean(axis=0)[0]
                                             - vn.bounds.mean(axis=0)[0])),
                s_len=float(sh.bounds[1][2] - sh.bounds[0][2]))
    print("\n>>> TRANSFORM WITNESSES (before -> after; a mismatch means the frame map is wrong)")
    bad = []
    for k in ("cfv_dia", "cfa_dia", "cfv_near_depth", "cfa_cfv_separation", "s_len"):
        ok = abs(pre[k] - post[k]) < 0.05
        print(f"    {k:<20} {pre[k]:8.3f} -> {post[k]:8.3f}   {'OK' if ok else '🚨 MISMATCH'}")
        if not ok:
            bad.append(k)
    # ── 🚨 WHICH LEG IS THIS? The check that was missing until 2026-08-18 ────────────────
    # A reflection is correct in every single-body test: each vessel keeps its depth, its
    # distance from the axis and its diameter. It is wrong only in the ORDER, and nothing
    # asserted the order -- so `LPS = (x, -z, -y)`, determinant -1, shipped a LEFT thigh from a
    # right-leg CAD and every render looked plausible. V2 and V3 were right thighs; the mirror
    # was even noticed in render_deformed.py ("V4 is the exact MIRROR of V3 in x") and filed as
    # a probe-placement bug.
    #
    # In LPS +x is patient-LEFT, so on a RIGHT thigh lateral is -x and medial is +x, and the
    # bundle runs nerve -> artery -> vein going -x to +x (the NAV mnemonic, lateral to medial).
    nv = lps.get("Nerve")
    if nv is not None:
        x_n = float(nv.bounds.mean(axis=0)[0])
        x_a = float(ar.bounds.mean(axis=0)[0])
        x_v = float(vn.bounds.mean(axis=0)[0])
        order_ok = x_n < x_a < x_v
        print(f"\n>>> LATERALITY: nerve {x_n:+.2f} | artery {x_a:+.2f} | vein {x_v:+.2f} mm "
              f"(LPS x, +x = patient left)")
        print(f"    lateral-to-medial reads "
              f"{'nerve -> artery -> vein = RIGHT thigh ✅' if order_ok else 'vein -> artery -> nerve = LEFT thigh 🚨'}")
        if not order_ok:
            raise SystemExit(
                "!!! this is a LEFT thigh. The CAD is authored as a RIGHT leg and V2/V3 are "
                "right thighs.\n    A determinant -1 frame map mirrors the patient while "
                "leaving every single-body\n    measurement correct -- check to_lps.")
    if bad:
        raise SystemExit(f"!!! the frame conversion changed {bad} -- LPS = (-x, -z, -y) is wrong "
                         f"for this export. REFUSING to write; a mirrored bundle looks correct "
                         f"in every render.")
    # s = 0 must be the SUPERIOR end, which is what v3_section.py asserts too
    s_cfa = (z_lig - float(ar.bounds[1][2]), z_lig - float(ar.bounds[0][2]))
    s_cfv = (z_lig - float(vn.bounds[1][2]), z_lig - float(vn.bounds[0][2]))
    print(f"    CFA spans s {s_cfa[0]:.1f}..{s_cfa[1]:.1f} (spec 0..37.5)")
    print(f"    CFV spans s {s_cfv[0]:.1f}..{s_cfv[1]:.1f} (spec 0..80)")
    pr = lps["PubicRamus"]
    s_ram = (z_lig - float(pr.bounds[1][2]), z_lig - float(pr.bounds[0][2]))
    print(f"    RAMUS spans s {s_ram[0]:.1f}..{s_ram[1]:.1f}   present at s={args.station_s:g}? "
          f"{s_ram[0] <= args.station_s <= s_ram[1]}")
    if not (s_ram[0] <= args.station_s <= s_ram[1]):
        raise SystemExit("!!! the ramus is still absent at the press station -- it cannot act as "
                         "a backstop. Raise --ramus-scale-s.")

    # ── the lumens. V4 exports vessel WALLS as thin tubes and has no Blood* bodies ───────
    # v3_section.py's nesting needs a lumen body inside each wall. Deriving it from the wall's
    # own inner surface keeps ONE definition of where the lumen is.
    for wall, blood in (("Artery", "BloodArtery"), ("Vein", "BloodVein")):
        w = lps[wall]
        c = w.bounds.mean(axis=0)
        r_out = float(max(w.bounds[1][0] - c[0], w.bounds[1][1] - c[1]))
        # wall thickness from the tube's own volume: V = pi(ro^2-ri^2)L
        L = float(w.bounds[1][2] - w.bounds[0][2])
        ri = float(np.sqrt(max(r_out ** 2 - w.volume / (np.pi * L), 1e-9)))
        # 🚨 DE-COINCIDE, ADDED 2026-08-18 AFTER IT COST US THE WHOLE B-MODE.
        # Built at exactly `ri` the lumen's outer surface IS the wall's inner surface, and the
        # two are discretised differently -- this is a 64-gon, the CAD wall is its own polygon --
        # so lumen vertices land OUTSIDE the bore. Measured overlap: 0.0039 mm (vein), 0.0075 mm
        # (artery). raysim resolves overlapping bodies ONE LEVEL DEEP, and with coincident
        # surfaces it picked the WALL: the lumens rendered at -5.1 dB and -2.9 dB against muscle
        # where V2 measured -52.4 dB, i.e. the vessels were invisible and no compression could
        # be seen in any image. With this epsilon: -30.7 dB and -34.7 dB, walls still in place.
        # 📌 Chapter 8 already recorded that V2 needed a 0.02 mm de-coincide for exactly this
        # reason. Nobody applied it to V4's synthesised lumens, and nothing checked.
        ri = max(ri - LUMEN_DECOINCIDE_MM, 1e-6)
        inner = trimesh.creation.cylinder(radius=ri, height=L, sections=64)
        inner.apply_translation([c[0], c[1], c[2]])
        lps[blood] = inner
        print(f"    derived {blood}: r {ri:.4f} mm inside the {wall} tube "
              f"(r_out {r_out:.2f}), de-coincided by {LUMEN_DECOINCIDE_MM:.3f} mm")

    # ── RECENTRE, IN ALL THREE AXES. V4 exports around a SolidWorks assembly origin ~1075,
    # 1098 mm from zero; V3's OBJs are centred on the limb axis and femoral_fem treats R_SKIN as
    # an ABSOLUTE Z with the limb at (0, 0). Left uncorrected the first V4 section reported the
    # vessels at Z = -1080 mm against a skin at +92.45, i.e. the limb and its contents in
    # different places.
    # 🚨 AND THE AXIAL AXIS WAS LEFT OUT UNTIL 2026-08-13, under a trailing comment that CLAIMED
    # it had been handled ("axial (z) origin is the ligament plane"). It had not: the V4 OBJs
    # shipped at LPS z ≈ -1131..-954 while V3's sit at -100..+20. The comment was the only thing
    # asserting the convention, and a comment is not an assertion. See the module docstring for
    # the three consumers that broke and the one that did not.
    c = lps["SoftTissue"].bounds.mean(axis=0)
    z_lig_pre = float(lps["SoftTissue"].bounds[1][2])
    dz_axial = args.image_plane_s - z_lig_pre      # ligament plane -> LPS z = +image_plane_s
    for m in lps.values():
        m.apply_translation([-c[0], -c[1], dz_axial])
    print(f"\n>>> recentred on the limb axis: moved x {-c[0]:+.1f}, y {-c[1]:+.1f} mm")
    print(f">>> AXIAL SHIFT APPLIED: z {dz_axial:+.3f} mm. The ligament plane (s = 0) was at "
          f"LPS z = {z_lig_pre:+.2f} and is now at {args.image_plane_s:+.2f}, so s = "
          f"{args.image_plane_s:g} mm sits at LPS z = 0 -- build_v3_anatomy.py's convention, "
          f"which is where every renderer puts the probe.")

    # ── SYNTHESISE THE MUSCLE CORE. V4 has Leg_Shell (-> SoftTissue, the 16 mm fat shell) and
    # no Muscle body at all. cell_material nests innermost-first, so with Muscle missing every
    # point inside r = 76.45 mm that is not a vessel resolves to VOID -- a hollow limb. That
    # would not error; it would quietly produce a phantom with a 15 cm cavity in it.
    sh2 = lps["SoftTissue"]
    r_out = float(np.hypot(sh2.vertices[:, 0], sh2.vertices[:, 1]).max())
    r_in = float(np.hypot(sh2.vertices[:, 0], sh2.vertices[:, 1]).min())
    L = float(sh2.bounds[1][2] - sh2.bounds[0][2])
    muscle = trimesh.creation.cylinder(radius=r_in, height=L, sections=180)
    muscle.apply_translation([0.0, 0.0, float(sh2.bounds.mean(axis=0)[2])])
    lps["Muscle"] = muscle
    print(f">>> synthesised Muscle as the shell's inner core: r {r_in:.2f} mm "
          f"(fat shell {r_out - r_in:.2f} mm thick) -- 🔴 a MODELLING CHOICE, V4 has no Muscle body")

    # ── ASSERT THE AXIAL CONVENTION, over EVERY body, AFTER the last one is built ──────────
    # 🪤 Muscle is synthesised above, i.e. after the shift, so an assertion placed at the shift
    # would have certified a frame two bodies short of the export. Assert what is written.
    z_lig_post = float(lps["SoftTissue"].bounds[1][2])
    if abs(z_lig_post - args.image_plane_s) > 1e-6:
        raise SystemExit(
            f"!!! AXIAL CONVENTION VIOLATED. The ligament plane (the shell's superior end) is "
            f"at LPS z = {z_lig_post:+.6f} mm, not the required {args.image_plane_s:+.6f}.\n"
            f"    Every renderer puts the probe at LPS z = 0 and reads s = 0 as the ligament, "
            f"so this is not cosmetic:\n"
            f"    off-convention OBJs image open background and author the leg a metre out of "
            f"place. REFUSING to write.")
    print(f"\n>>> axial convention CHECKED: ligament plane (s = 0) at LPS z = "
          f"{z_lig_post:+.3f} mm, s = {args.image_plane_s:g} mm at LPS z = 0.000")
    print(f"{'body':<14}{'s span (mm)':>18}   at the imaged plane?")
    for name in sorted(lps):
        b = lps[name].bounds
        s0, s1 = z_lig_post - float(b[1][2]), z_lig_post - float(b[0][2])
        reach = float(b[0][2]) < 0.0 < float(b[1][2])
        print(f"{name:<14}{s0:8.1f} ..{s1:8.1f}   "
              f"{'imaged' if reach else '❌ ABSENT at s = %g' % args.image_plane_s}")

    # 🚨 WRITE VERTEX NORMALS HERE. raysim refuses a mesh without them --
    # "RuntimeError: Mesh: has no normals" -- and trimesh's default OBJ export omits them.
    # `fix_obj_normals.py` existed to patch that afterwards, which meant EVERY regeneration
    # of this anatomy silently produced a mesh set the renderer would reject, and the failure
    # surfaced minutes later inside a container. Writing them at source removes the manual step.
    for name, m in lps.items():
        trimesh.repair.fix_normals(m)
        with open(OUT / f"{name}.obj", "w") as fh:
            fh.write(f"# {name}, from the V4 export. SPACE=LPS, units mm.\n")
            np.savetxt(fh, m.vertices, fmt="v %.6f %.6f %.6f")
            np.savetxt(fh, m.vertex_normals, fmt="vn %.6f %.6f %.6f")
            for t in m.faces + 1:
                fh.write("f %d//%d %d//%d %d//%d\n" % (t[0], t[0], t[1], t[1], t[2], t[2]))

    # ── the annotation targets, MEASURED from the geometry that was just exported ──────────
    # Same derivation as build_v3_anatomy.py: centre and width of the lumen/nerve outline at
    # the target plane, in LPS mm. Derived here rather than copied so a V4 frame can never be
    # annotated with V3's numbers -- the drift `build_v3_anatomy.py` already closed for V3.
    z_t = args.image_plane_s - args.target_plane_s
    targets, no_target = {}, []
    for body, label in (("BloodArtery", "artery"), ("BloodVein", "vein"), ("Nerve", "nerve")):
        sec = lps[body].section(plane_origin=[0.0, 0.0, z_t], plane_normal=[0.0, 0.0, 1.0])
        if sec is None:
            no_target.append(f"{label} ({body} does not reach s = {args.target_plane_s:g} mm)")
            continue
        P = np.vstack([np.asarray(q, np.float64)[:, :2] for q in sec.discrete])
        targets[label] = [float(P[:, 0].mean()), float(P[:, 1].mean()), float(np.ptp(P[:, 0]))]
    print(f"\n--- annotation targets at s = {args.target_plane_s:g} mm (LPS z = {z_t:+.2f}) ---")
    for k, (x, y, w) in targets.items():
        print(f"   {k:<7} centre ({x:+7.3f}, {y:+7.3f})  width {w:5.2f} mm")
    for a in no_target:
        print(f"   ❌ {a}")

    meta = dict(source=str(SRC), frame="LPS = (V4.x, -V4.z, -V4.y)",
                ramus_gap_mm=args.ramus_gap_mm, ramus_scale_s=args.ramus_scale_s,
                ramus_move_mm=[dx, dy, dz], witnesses_pre=pre, witnesses_post=post,
                s_cfa=s_cfa, s_cfv=s_cfv, s_ramus=s_ram,
                axial_shift_mm=dz_axial, ligament_lps_z_mm=z_lig_post,
                image_plane_s_mm=args.image_plane_s, station_s_mm=args.station_s,
                lateral_recentre_mm=[-float(c[0]), -float(c[1])])
    json.dump(meta, open(OUT / "v4_conversion.json", "w"), indent=1, default=float)
    print(f"\nwrote {len(lps)} OBJs to {OUT}")

    # ── anatomy_config.json, via THE one writer ────────────────────────────────────────────
    # Every field below is derived from an artifact this run measured: the shell's own radii,
    # the asserted ligament plane, and the outlines sectioned out of the exported OBJs. The
    # generation is passed EXPLICITLY -- this converter knows it built V4, and an auto-detect
    # that is wrong writes a config that looks right (materials.detect_generation is the
    # backstop for readers who do not know, not for the writer who does).
    MAT.emit_anatomy_config(
        str(OUT), generation="v4", background_material=args.background,
        geometry={
            "r_skin_mm": r_out,
            "r_shell_inner_mm": r_in,
            "ligament_lps_z_mm": z_lig_post,
            "image_plane_s_mm": args.image_plane_s,
            "target_plane_s_mm": args.target_plane_s,
            "targets": targets,
            "station_s_mm": args.station_s,
        },
        provenance={
            "source": str(SRC),
            "frame": "LPS mm; LPS = (V4.x, -V4.z, -V4.y), recentred on the Leg_Shell bbox "
                     "centre in x/y and shifted axially so the inguinal ligament plane sits "
                     "at LPS z = +image_plane_s_mm (build_v3_anatomy.py's convention). See "
                     "v4_to_section.py's docstring for the five witnesses that check the "
                     "frame map and the assertion that checks the axial origin.",
            "axial_shift_mm": dz_axial,
            "synthesised_bodies": ["Muscle", "BloodArtery", "BloodVein"],
            "missing_bodies": ["LooseTissue", "Septum"],
            "missing_reason": "no CAD body exists for either; same as V3, logged in "
                              "CAD_CHANGELOG.md.",
            "note_lumen_diameters": "🚨 The CAD treats CFA_dia/CFV_dia as the tube's OUTER "
                                    "diameter, so the LUMENS are smaller than the spec's "
                                    "intima-to-intima figures -- the same V3 defect, and the "
                                    "reason deform_meshes.EXPECTED_RADII must be derived from "
                                    "the section rather than typed from the spec.",
            "ramus_gap_mm": args.ramus_gap_mm,
            "ramus_gap_status": "🔴 MODELLING CHOICE, UNSOURCED. --ramus-gap-mm has no "
                                "default; this directory is one rung of a sweep, not a "
                                "measurement. See BODIES['PubicRamus'].",
            "ramus_scale_s": args.ramus_scale_s,
            "ramus_scale_status": "🔴 MODELLING CHOICE. The CAD ramus is a 0.31 cm^3 stub; it "
                                  "is stretched along s to span the access window, pending "
                                  "real dimensions.",
            "muscle_status": "🔴 MODELLING CHOICE. V4 ships no Muscle body; the core is "
                             "synthesised as the shell's inner cylinder.",
        })

    print(f"\nnext: $PY_ANATOMY v3_section.py --mesh-dir {OUT} "
          f"--station-s {args.station_s:g}")
    print(f"      $PY_ANATOMY anatomy_meshing/gel_body.py --obj-dir {OUT}")


if __name__ == "__main__":
    main()
