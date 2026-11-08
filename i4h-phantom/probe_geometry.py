"""THE probe/gel geometry contract. One definition, imported by everything that needs it.

    import probe_geometry as PG      # pure python, no numpy, no warp, no trimesh

📌 IT IS IMPORTABLE INSIDE THE raysim CONTAINER, and believing it was not is what produced the
fifth and sixth copies of the standoff. `render_deformed.py` said "This file runs INSIDE the
raysim container and cannot import probe_geometry.py"; the wrapper mounts the repo at /w and
runs `python /w/raysim_ultrasound/render_deformed.py`, so /w IS sys.path[0] and the import works. What genuinely
cannot cross that boundary is the *anatomy*, which is why the numbers still travel in
anatomy_config.json -- but the CODE that reads and validates that block now lives here once
(`gel_from_config`) instead of once per renderer, and it CHECKS the config against the
constants below rather than trusting it. It is NOT importable inside the Isaac Sim container:
render_leg_scene.py is copied into a scratch dir on its own (render_leg_scene.sh), so the USD
`<stem>_report.json` is the contract that crosses THAT boundary.

🚨 WHY THIS FILE EXISTS — OPEN_ITEMS 20, and it was worse than that item said.
Item 20 records that the gel band (1.0 mm) and the probe standoff (2.0 mm) contradicted each
other, so the transducer face sat 1 mm outside the gel in AIR and every B-mode rendered black.
The fix it asked for was to derive both from one value. Auditing for that on 2026-08-11 turned up
**four** independent copies of the standoff, not two:

    femoral_fem.py:680        start_z = z_skin + half[2] + 0.002
    diag_engagement.py:198    past_skin_mm = achieved - 2.0        <- a LITERAL, in the reporting
    diag_deformable_leg.py    --standoff default 0.002
    gel_body.py               --standoff-check default 2.0

The reporting copy is the dangerous one: change the standoff anywhere else and
`past_skin_mm` keeps subtracting 2.0 and silently mislabels every depth in the ladder — while
`past_skin` is precisely the column that SURVIVED the 2026-08-11 force-channel finding, because
it is a displacement rather than a force. The one trustworthy column in the record was one edit
away from being quietly wrong.

⚠️ THE CHOICE MADE, AND WHY IT IS THE CONSERVATIVE ONE.
Two reconciliations were available: bring the standoff inside the band, or thicken the band to
swallow the standoff. **The band is thickened and the standoff is left at 2.0 mm**, because:

  · the standoff is the datum for `past_skin`, and every depth-based number in the record —
    which is to say every number that survived 2026-08-11 — is measured from it. Moving it would
    invalidate the one half of the record that is still good, to fix a rendering problem.
  · 2.0 mm of clearance also keeps travel 0 genuinely OUT of contact: the collision pipeline's
    soft_contact_margin is 1.0 mm and the particle radius is 0.25*dx = 0.375 mm at dx 1.5 mm, so
    a standoff much below ~1.5 mm would put the probe inside the contact margin before the ladder
    starts, and the force tare would no longer be a tare.
  · the band is a MODELLING choice with no such coupling. A 3.5 mm coupling layer is a gel
    standoff column rather than a pressed film -- which is a real clinical configuration, and it
    is stated here rather than implied.

📌 So the physical claim this file makes is: **the transducer images through a gel column and is
never in air.** At travel 0 the face sits GEL_CLEARANCE_M inside the band; it only ever moves
deeper. That is what makes "a hovering probe images black" usable as a control -- a black frame
now means the POSE is wrong, not that the coupling was never there.
"""

# ── the single definitions ───────────────────────────────────────────────────────────────
PROBE_STANDOFF_M = 0.002        # probe face above the skin apex at travel 0
PROBE_HALF_THICKNESS_M = 0.006  # half the probe box's thickness in z
GEL_THICKNESS_M = 0.0035        # ultrasound gel band around the limb

# how far inside the gel the face sits at travel 0. Derived, never typed.
GEL_CLEARANCE_M = GEL_THICKNESS_M - PROBE_STANDOFF_M

# minimum clearance we are willing to ship. Below this the face is close enough to the band's
# outer surface that mesh tolerance, the 0.02 mm de-coincide offset and any deformation of the
# limb could put it back in air.
MIN_GEL_CLEARANCE_M = 0.001


def check_coupled(standoff_m: float | None = None, thickness_m: float | None = None) -> float:
    """Raise unless the probe face is inside the gel band with margin. Returns the clearance.

    🚨 A HARD FAILURE, NOT A WARNING. gel_body.py used to print a paragraph explaining that a
    probe outside the band images black "which is physically correct, not a bug" -- and it was
    right, and the run shipped anyway, and `photos/2026-08-08/bmode/` is an empty directory
    because of it. A check that explains the problem and then proceeds is a comment.
    """
    s = PROBE_STANDOFF_M if standoff_m is None else standoff_m
    t = GEL_THICKNESS_M if thickness_m is None else thickness_m
    clearance = t - s
    if clearance < MIN_GEL_CLEARANCE_M:
        raise SystemExit(
            f"!!! THE TRANSDUCER IS NOT COUPLED. standoff {s*1e3:.2f} mm against a "
            f"{t*1e3:.2f} mm gel band\n"
            f"    leaves {clearance*1e3:+.2f} mm of clearance; at least "
            f"{MIN_GEL_CLEARANCE_M*1e3:.2f} mm is required.\n"
            f"    raysim stores air at Z = 0.000 MRayl exactly, so air|fat reflects 99.88 %% and\n"
            f"    EVERY FRAME WILL RENDER BLACK. See probe_geometry.py and OPEN_ITEMS 20.\n"
            f"    Fix the constants in probe_geometry.py -- both halves move together or neither.")
    return clearance


def summary() -> str:
    return (f"probe standoff {PROBE_STANDOFF_M*1e3:.2f} mm · gel band "
            f"{GEL_THICKNESS_M*1e3:.2f} mm · face sits {GEL_CLEARANCE_M*1e3:.2f} mm INSIDE the "
            f"gel at travel 0")


# ── the anatomy-side half of the contract ────────────────────────────────────────────────
# Everything below reads anatomy_config.json, which gel_body.py / build_v3_anatomy.py /
# v4_to_section.py write next to the OBJs. These are the ONLY readers of that block that the
# renderers are allowed to have -- render_deformed.py and sweep_circumference.py had two
# subtly different inline copies, and sweep_circumference.py's copy did not exist at all: it
# carried `R_SKIN = 92.5` and `STANDOFF = 2.0` as literals that happened to agree with this
# file and were enforced by nothing.

_GEL_KEYS = ("skin_r_mm", "probe_standoff_mm", "thickness_mm", "gel_clearance_mm",
             "r_in_mm", "r_out_mm", "material")


def gel_from_config(cfg: dict, where: str = "anatomy_config.json") -> dict:
    """The validated `_provenance.gel` block of an anatomy config. Raises, never warns.

    🚨 AND IT CROSS-CHECKS THE CONFIG AGAINST THE CONSTANTS ABOVE. A config is a FILE on disk,
    written by a run that may predate the current constants -- exactly the two-artifacts-and-
    nothing-reconciles-them failure this project pays for most often. Reading the standoff out
    of a stale config is not one source of truth, it is a second one wearing the first one's
    name. So the numbers are read from the config (they must match the gel mesh that was
    actually built) AND asserted equal to this module's (they must match the FEM's travel
    datum). If they disagree, the gel mesh and the press ladder are describing different
    geometry and neither the film nor `past_skin` means what it says.
    """
    gel = (cfg.get("_provenance") or {}).get("gel")
    if not gel:
        raise SystemExit(
            f"!!! {where} has no _provenance.gel block.\n"
            f"    Run anatomy_meshing/gel_body.py on that mesh dir first -- it is what reconciles the standoff\n"
            f"    with the band, and without it a renderer would be GUESSING the coupling.\n"
            f"    See OPEN_ITEMS 20.")
    missing = [k for k in _GEL_KEYS if k not in gel]
    if missing:
        raise SystemExit(f"!!! {where} _provenance.gel is missing {missing} -- it was written "
                         f"by a gel_body.py older than 2026-08-11. Rebuild the gel.")
    if gel["gel_clearance_mm"] <= 0:
        raise SystemExit(
            f"!!! the transducer face is {-gel['gel_clearance_mm']:.2f} mm OUTSIDE the gel band "
            f"({where}).\n"
            f"    Every frame would render black. Fix probe_geometry.py, rebuild the gel, "
            f"re-run.")
    # The config is in mm, this module is in m. 1e-6 mm is float noise, not a tolerance to tune.
    for key, want_m in (("probe_standoff_mm", PROBE_STANDOFF_M),
                        ("thickness_mm", GEL_THICKNESS_M),
                        ("gel_clearance_mm", GEL_CLEARANCE_M)):
        if abs(float(gel[key]) - want_m * 1e3) > 1e-6:
            raise SystemExit(
                f"!!! {where} was built against a DIFFERENT probe geometry: {key} = "
                f"{float(gel[key]):.6f} mm there, {want_m*1e3:.6f} mm in probe_geometry.py.\n"
                f"    The gel mesh on disk and the FEM's travel datum are two artifacts and "
                f"they disagree.\n"
                f"    Re-run anatomy_meshing/gel_body.py on that mesh dir; do NOT edit the number in either "
                f"place to match.")
    check_coupled(float(gel["probe_standoff_mm"]) * 1e-3, float(gel["thickness_mm"]) * 1e-3)
    return gel


def station_lps_z_mm(cfg: dict, station_s_mm: float, where: str = "anatomy_config.json") -> float:
    """CAD/LPS z of the axial station `s`, in mm. ONE definition of the s -> z mapping.

    🚨 s IS NOT z, AND NOTHING IN EITHER ARTIFACT USED TO SAY SO. s = 0 is the inguinal
    ligament plane and s runs INFERIORLY (distally); LPS z runs superiorly, so
    z = ligament_lps_z_mm - s. On V3 the ligament plane sits at z = +20 mm, so the CAD z = 0
    that every renderer pinned the probe to is s = +20 mm -- while the 2026-08-12 press
    ladders were solved at s = 30. Two stations, 10 mm apart, and neither artifact declared
    which one it was. On the un-recentred V4 OBJs z = 0 was s = -1011.67 mm, a metre proximal
    of the phantom, in open background.

    The measured axial spans (identical in V3 and V4_gap12, both re-measured 2026-08-13):
        nerve  s 0..25 · CFA 0..37.5 · CFV 0..80 · PubicRamus s -7.58..47.58 · limb 0..120
    so all five bodies coexist for s <= 25. "All three in the imaging window" is satisfiable,
    at s = 20 among others; it is NOT satisfiable at s = 30, where the nerve does not exist.
    """
    geom = cfg.get("geometry") or {}
    if "ligament_lps_z_mm" not in geom:
        raise SystemExit(
            f"!!! {where} has no geometry.ligament_lps_z_mm, so the station s cannot be mapped "
            f"to a CAD z.\n"
            f"    That number is written by build_v3_anatomy.py / v4_to_section.py. Without it "
            f"a probe placed\n"
            f"    at 'station 20' is placed at an undeclared depth along the limb.")
    return float(geom["ligament_lps_z_mm"]) - float(station_s_mm)


# ── 🚩 OPEN, RAISED 2026-08-18: THE GEL BAND IS THE WHOLE LIMB, AND IT SHOULD NOT BE ────────
# `gel_body.py` builds GEL_THICKNESS_M as an ANNULUS around the entire circumference — 192 mm
# outer diameter on a 92.5 mm-radius limb. Real coupling gel is a local blob under the probe
# footprint, of order 40–60 mm wide.
#
# Why it is not merely cosmetic:
#   * it wraps the limb in a water shell at EVERY angle, so a circumferential sweep images a
#     limb that is coupled everywhere — which is exactly the control the air background exists
#     to provide (a probe off the skin must image black, OPEN_ITEMS 20 closed on that).
#   * it changes the outer acoustic boundary the beam crosses at angles the probe is not at.
#   * it is one more body whose extent nobody stated, in a project whose most expensive defect
#     class is two artifacts that nobody reconciles.
#
# ⚠️ NOT CHANGED YET, DELIBERATELY. Narrowing it is a geometry change that would invalidate the
# comparability of every B-mode on record with every other one, and the canal sweep is mid-
# flight against the current geometry. Change it as its own single-variable step, with a
# before/after on one identical frame, not folded into a deliverable.
GEL_BAND_IS_FULL_CIRCUMFERENCE = True   # 🔴 known-wrong, tracked, not yet fixed
