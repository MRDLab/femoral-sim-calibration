"""Canonical per-body property table for the femoral CAD phantom.

Because we authored this anatomy, every body's identity is known by construction, so every
body gets an explicit acoustic AND mechanical assignment here rather than being inferred
downstream. One file, imported by both the raysim render and the Newton FEM run, so the two
halves of the simulator cannot drift apart the way the VOXEL-MAN USD and OBJ artifacts did.

PROVENANCE IS TRACKED PER VALUE. Three tiers, and they are not interchangeable:

  "nvidia"  -- the value NVIDIA ships in raysim's material table
               (csrc/core/material.cpp, `Materials::Materials()`). Authoritative for what
               the ray tracer will actually do.
  "ours"    -- a value this project added and documented (the `air` material).
  "repr"    -- REPRESENTATIVE literature figure, chosen by us because no NVIDIA value
               exists. Real, sourced to a range, but not measured for this phantom and NOT
               to be quoted as if it were. Every one of these is a modelling choice.

🔴 BEFORE ADDING OR CHANGING ANY VALUE HERE, CHECK THE CITED SOURCE FIRST. What is
on disk and what it can back is indexed in `References/README.md`, which records per paper
what it *cannot* support as well as what it can. As of 2026-07-31 the position is:

  * GEOMETRY is well sourced -- Zambetti 2025 (CFA/CFV diameter and depth with dispersion),
    Sandgren 1999 (diameter vs body size), Prună 2019 (diameter along the course), Fenech,
    Roche & Boyle 2024 (N-A-V ordering, femoral sheath compartments, the nerve lying outside
    and lateral to the sheath), and Gray's 16th ed. Parts 1-6, of which
    Part2_Scarpas_Triangle_Femoral_Sheath is directly the structure V3 needs to add.
  * MECHANICAL properties are sourced by NOTHING on disk. Every `E_pa` below is `repr`. The
    femoral PDFs are imaging/anatomy papers and report no elastic moduli. Tracked as an open
    gap in `research-planning/sources_needed.md` §1.
  * ACOUSTIC values are NVIDIA's own, transcribed from raysim's C++ table -- authoritative
    for what the renderer does, and still unverified against any published tissue-property
    reference.

The acoustic column is split deliberately: `raysim_material` is what the renderer is told,
`acoustic_true` is what the tissue actually is. Where they differ the difference is stated
in `acoustic_gap` -- that is the honest record of what the current renderer cannot express
(no vessel-wall material, no nerve material), and it is exactly the list to revisit if a
material is ever added to raysim.
"""

from __future__ import annotations

# ── raysim's shipped material table, transcribed from csrc/core/material.cpp ─────────────
# Material(impedance MRayl, attenuation dB/(cm*MHz), speed m/s, mu0, mu1, sigma, specularity)
RAYSIM_MATERIALS = {
    "water": dict(impedance=1.48, attenuation=0.0022, speed=1480.0, mu0=0.0, mu1=0.0, sigma=0.0, spec=0.0, src="nvidia"),
    "blood": dict(impedance=1.61, attenuation=0.18, speed=1570.0, mu0=0.1, mu1=0.1, sigma=0.1, spec=0.0, src="nvidia"),
    "fat":   dict(impedance=1.38, attenuation=0.63, speed=1450.0, mu0=1.0, mu1=0.0, sigma=1.0, spec=0.0, src="nvidia"),
    "liver": dict(impedance=1.65, attenuation=0.70, speed=1550.0, mu0=0.7, mu1=0.0, sigma=0.3, spec=1e-5, src="nvidia"),
    "muscle": dict(impedance=1.70, attenuation=1.09, speed=1580.0, mu0=0.5, mu1=0.8, sigma=0.4, spec=0.0, src="nvidia"),
    "bone":  dict(impedance=7.80, attenuation=5.00, speed=4080.0, mu0=0.8, mu1=0.9, sigma=0.5, spec=0.0, src="nvidia"),
    "air":   dict(impedance=4.13e-4, attenuation=8.2, speed=343.0, mu0=0.0, mu1=0.0, sigma=0.0, spec=1.0, src="ours"),
}

# ── per-body assignment ──────────────────────────────────────────────────────────────────
# raysim_material : the name handed to rs.Materials().get_index() -- must be a key above.
# E_pa, nu, rho   : Newton FEM parameters (Pa, -, kg/m^3).
# *_src           : provenance tier for that value.
BODIES = {
    "SoftTissue": dict(
        role="skin + subcutaneous fat shell, 0.00 -> 16.00 mm depth",
        raysim_material="fat", raysim_src="nvidia",
        acoustic_true="skin (Z 1.6, atten 2-4) over subcutaneous fat (Z 1.38)",
        acoustic_gap="skin and fat are ONE body, so the skin/fat interface echo cannot be "
                     "produced. Not a shortcut -- no source gives an anterior-thigh fat "
                     "thickness, so the split point would have to be invented.",
        E_pa=3.0e3, E_src="repr",
        E_note="subcutaneous adipose. Corrected 2026-10-28: NOT shear-wave elastography. Backed only as a "
               "quasi-static ex vivo value (Alkhouli 2013, 1.6 +/- 0.8 kPa; Samani 2007, 3.25 kPa). In vivo "
               "thigh MRE gives SHEAR 3.04 kPa, i.e. E ~9 kPa (Chakouch 2015): 3 kPa may be a shear value "
               "read as E. Ledger: research-planning/sources_needed.md, 2026-10-28.",
        nu=0.45, nu_src="repr", rho=950.0, rho_src="repr",
    ),
    "Muscle": dict(
        role="muscle core, 16.00 mm depth -> axis",
        raysim_material="muscle", raysim_src="nvidia",
        acoustic_true="relaxed skeletal muscle, Z ~1.70 MRayl",
        acoustic_gap="none -- raysim's `muscle` is the right material for this body.",
        E_pa=20.0e3, E_src="repr",
        E_note="relaxed thigh muscle, ISOTROPIC. Corrected 2026-10-28: not 'transverse'; measured across the "
               "fibres E ~5-8 kPa (Gennisson 2010; Knight 2022). Backed as isotropic by thigh MRE at rest, "
               "E ~11-23 kPa (Bensamoun 2006; Chakouch 2015).",
        nu=0.45, nu_src="repr", rho=1050.0, rho_src="repr",
    ),
    "Artery": dict(
        role="common femoral artery WALL, OD 11.0 / ID 9.0 mm (1.00 mm wall)",
        raysim_material="muscle", raysim_src="nvidia",
        acoustic_true="arterial wall, elastin/collagen, Z ~1.7-1.8 MRayl, hyperechoic "
                      "relative to the surrounding muscle",
        acoustic_gap="🚨 raysim has NO vessel-wall material. Mapping the wall to `muscle` "
                     "makes it acoustically IDENTICAL to the surrounding muscle, i.e. the "
                     "wall is invisible. The lumen still reads anechoic against muscle, so "
                     "the vessel is visible -- but as a hole, not as a walled tube.",
        E_pa=600.0e3, E_src="repr",
        E_note="Chapter 6 Test C value; stands in for wall stiffness AND the ~13 kPa "
               "transmural blood pressure that is not modelled. Femoral artery incremental "
               "modulus is 0.5-1.5 MPa in vivo.",
        nu=0.45, nu_src="repr", rho=1100.0, rho_src="repr",
    ),
    "BloodArtery": dict(
        role="common femoral artery LUMEN, Ø 9.0 mm, intimal surface at 17.20 mm depth",
        raysim_material="blood", raysim_src="nvidia",
        acoustic_true="flowing arterial blood; anechoic at B-mode frequencies",
        acoustic_gap="static -- raysim has no flow, so this is stationary blood. That is "
                     "correct for B-mode and is exactly the gap Doppler would have to fill.",
        E_pa=None, E_src="n/a", E_note="modelled as a VOID in the FEM -- no fluid in Newton; "
                                       "the artery's collapse resistance comes from its wall E.",
        nu=None, nu_src="n/a", rho=1060.0, rho_src="repr",
    ),
    "Vein": dict(
        role="common femoral vein WALL, OD 12.0 / ID 11.0 mm (0.50 mm wall)",
        raysim_material="muscle", raysim_src="nvidia",
        acoustic_true="venous wall, thin, less elastin than artery",
        acoustic_gap="same as Artery -- no vessel-wall material, so the wall is invisible.",
        E_pa=20.0e3, E_src="repr",
        E_note="Chapter 6 Test C 'compliant wall' value. A vein's collapsibility is "
               "structural (thin wall, low transmural pressure) more than material; 20 kPa "
               "is a proxy for both and is the number Test C already validated the "
               "discrimination against. 2026-10-28, full texts read: BACKED as a low-pressure "
               "circumferential modulus of human saphenous vein: Wesly 1975, 27 +/- 12 kPa at "
               "10 cmH2O (SE, n 4); Vesely 2015, E ~16-35 kPa over 0-0.8 kPa (3*mu, converted). "
               "Not a high-strain tensile modulus: excised femoral vein is 5.6-5.7 MPa (Lebahn 2025). "
               "No femoral-vein low-pressure value found.",
        nu=0.45, nu_src="repr", rho=1050.0, rho_src="repr",
    ),
    "BloodVein": dict(
        role="common femoral vein LUMEN, Ø 11.0 mm, intimal surface at 17.70 mm depth",
        raysim_material="blood", raysim_src="nvidia",
        acoustic_true="venous blood; anechoic, slow flow can produce rouleaux echoes",
        acoustic_gap="static blood, as above.",
        E_pa=None, E_src="n/a", E_note="VOID in the FEM -- this is the lumen that has to close.",
        nu=None, nu_src="n/a", rho=1060.0, rho_src="repr",
    ),
    "Nerve": dict(
        role="femoral nerve, Ø 10.0 mm, lateral to the artery",
        raysim_material="fat", raysim_src="nvidia",
        acoustic_true="peripheral nerve: fascicles in epineurium, honeycomb texture, "
                      "HYPERECHOIC relative to muscle in the short axis",
        acoustic_gap="🚨 raysim has NO nerve material. `fat` is chosen as the proxy because "
                     "it is the only shipped material with maximum scattering (mu0=1.0, "
                     "sigma=1.0) plus an impedance step against muscle -- it reproduces "
                     "'brighter and more textured than the surrounding muscle', which is "
                     "the sonographic signature that matters. It is NOT a measured nerve.",
        E_pa=40.0e3, E_src="repr",
        E_note="peripheral nerve shear-wave elastography, 20-60 kPa. Stiffer than muscle, "
               "far softer than artery -- so it should neither collapse nor resist like the artery.",
        nu=0.45, nu_src="repr", rho=1040.0, rho_src="repr",
    ),
    # ── V3 additions ─────────────────────────────────────────────────────────────────────
    # Three bodies V2 does not have. Appended here rather than kept in a separate table so
    # there is exactly ONE property file for the phantom -- the whole reason this module
    # exists is that Chapter 5 lost time to anatomy living in two artifacts that nothing
    # reconciled. BODY_ORDER_V2 / BODY_ORDER_V3 below say which build uses which subset.
    "PubicRamus": dict(
        role="superior pubic ramus -- the CFV's backstop at the access window (V4)",
        raysim_material="bone", raysim_src="nvidia",
        acoustic_true="cortical bone, Z 7.80 MRayl against muscle's 1.70",
        acoustic_gap="none acoustically. 🚨 Its POSITION is the open question: the CFV-far-wall "
                     "to anterior-ramus gap is 🔴 UNSOURCED (three searches, 2026-08-11) and is "
                     "the number that decides whether the vein can coapt. Swept, not assumed -- "
                     "see v4_to_section.py --ramus-gap-mm.",
        E_pa=17.0e9, E_src="repr",
        E_note="cortical bone, same as Femur. Above RIGID_E_PA, so it is PINNED as a backstop "
               "rather than meshed as a material -- which is the whole point of it.",
        nu=0.30, nu_src="repr", rho=1900.0, rho_src="repr",
    ),
    "Femur": dict(
        role="femoral shaft, Ø 23.4 mm rod (V3 only)",
        raysim_material="bone", raysim_src="nvidia",
        acoustic_true="cortical bone, Z 7.80 MRayl against muscle's 1.70",
        acoustic_gap="none acoustically -- `bone` is the right material, and it is the one "
                     "V3 body whose material is not a compromise. 🚨 But every DIMENSION and "
                     "its position are 🔴 modelling choices: V3_PARAMETRIC_SPEC section 4 "
                     "records that nothing on disk measures the femur. And at s = 0..40 the "
                     "real bone is femoral head/neck/ramus, not shaft, so a straight rod is "
                     "a stated approximation.",
        E_pa=17.0e9, E_src="repr",
        E_note="cortical bone longitudinal modulus, 15-20 GPa. Six orders above the soft "
               "tissue around it, so in practice it is a rigid backstop rather than a "
               "material -- which is the entire point of adding it (spec section 4).",
        nu=0.30, nu_src="repr", rho=1900.0, rho_src="repr",
    ),
    "Sheath": dict(
        role="femoral sheath, funnel-shaped fascial tube, s = 0..35 mm (V3 only)",
        raysim_material="muscle", raysim_src="nvidia",
        acoustic_true="dense collagenous fascia; a thin bright line in vivo, Z ~1.75",
        acoustic_gap="🚨 raysim has NO fascia material. `muscle` is the nearest impedance "
                     "and is therefore INVISIBLE against the muscle it sits in -- the same "
                     "gap already logged for the vessel walls and the nerve. This body earns "
                     "its place in the FEM, not in the B-mode. Spec section 5 says it "
                     "outright: adding bodies the renderer cannot distinguish makes the "
                     "mechanics better and the image no better.",
        E_pa=None, E_src="none",
        E_note="🔴 UNBACKED, AND LEFT None ON PURPOSE. Spec section 3 records that Gray's is "
               "pre-metric, Moore gives shape and length but no thickness, and Fenech gives "
               "compartments but no dimension -- neither wall thickness nor stiffness has a "
               "source. A plausible number invented here would silently become the thing "
               "that decides whether the vein closes, which is the exact measurement V3 "
               "exists to make. Log it in sources_needed.md section 1 before picking one.",
        nu=None, nu_src="none", rho=1100.0, rho_src="repr",
    ),
    "Perivascular": dict(
        role="the CONTENTS of the femoral sheath -- loose areolar tissue filling the bore "
             "between the CFA, the CFV and the inner face of the funnel. Synthesised at "
             "load time from the Sheath's own inner loop; NOT a separate CAD body.",
        raysim_material="fat", raysim_src="nvidia",
        acoustic_true="loose areolar / fibrofatty tissue, indistinguishable from fat in vivo",
        acoustic_gap="`fat` is the same proxy FemoralCanal uses and for the same tissue, so "
                     "the two are acoustically identical to each other -- which is correct "
                     "here, they ARE the same tissue in different compartments. This body "
                     "earns its place in the FEM, not in the B-mode.",
        # 📌 NOT A NEW NUMBER. The default is BODIES["SoftTissue"]["E_pa"] by reference, set
        # just below this table, so the fill IS soft tissue rather than something new that
        # resembles it. Nothing unbacked is introduced by adding this body -- the 3 kPa is the
        # subcutaneous-adipose value already in use, carrying its existing provenance. The
        # bracket 1-5 kPa covers FemoralCanal's 2 kPa too; both are the same tissue class.
        E_pa=None, E_src="ref:SoftTissue",
        E_note="loose areolar / fibrofatty, 1-5 kPa. 🎯 THE MECHANISM THIS BODY EXISTS FOR: "
               "it must be the MOST COMPLIANT thing in the load path, so that probe force "
               "yields HERE first and the vein -- whose lumen is a void -- collapses into "
               "the space that gives, instead of the whole column displacing in bulk. That "
               "is the 2026-08-07 finding ('the probe does NOT travel to the vein') stated "
               "as a design requirement. ⚠️ IT IS THEREFORE A SWEPT PARAMETER, NEVER A TUNED "
               "ONE: if closure only appears at a stiffness outside 1-5 kPa, that is a "
               "reportable failure of the hypothesis, not a licence to move the number. "
               "Measured 2026-08-14 (diag_funnel_section.py): without this body the funnel "
               "bore is 142.32 mm2 of `Sheath` at the --sheath-kpa default of 20 kPa, i.e. "
               "muscle, and the column straight down onto the vein reads "
               "fat 16.0 mm -> 2.3 mm of that 20 kPa fill -> vein wall.",
        nu=0.45, nu_src="repr", rho=950.0, rho_src="repr",
    ),
    "FemoralCanal": dict(
        role="femoral canal, medial compartment of the sheath, s = 0..12.5 mm (V3 only)",
        raysim_material="fat", raysim_src="nvidia",
        acoustic_true="loose areolar tissue, lymphatics and Cloquet's node",
        acoustic_gap="`fat` is a defensible proxy for loose areolar tissue -- both are "
                     "low-impedance and highly scattering -- but it is a proxy, and it is "
                     "the same material the nerve borrows, so the two are acoustically "
                     "indistinguishable from each other.",
        E_pa=2.0e3, E_src="repr",
        E_note="loose areolar / adipose, 1-5 kPa. Softer than muscle deliberately: this "
               "compartment is the medial compliance the vein needs somewhere to displace "
               "into. 🔴 No femoral-specific source.",
        nu=0.45, nu_src="repr", rho=950.0, rho_src="repr",
    ),
}

# Which bodies each CAD generation ships, in OUTERMOST-FIRST author order. raysim resolves
# overlapping bodies innermost-wins and build_leg_usd.py authors USD prims in this order, so
# the acoustic and the physics nesting stay consistent by construction.
BODY_ORDER_V2 = ["SoftTissue", "Muscle", "Artery", "BloodArtery", "Vein", "BloodVein", "Nerve"]
# 🚨 V3 SHIPS THE VESSEL WALLS AND NOT THE LUMENS -- THE EXACT OPPOSITE OF THE FIRST READING.
# The CAD's CFA and CFV are 1.0 mm / 0.5 mm WALL TUBES (measured: vertex radii at exactly
# 3.5/4.5 and 5.0/5.5, volumes matching tubes and not rods). So `Artery` and `Vein` come
# straight from CAD, and it is `BloodArtery` / `BloodVein` that have to be synthesised into
# their bores -- the same edit CAD_CHANGELOG section 2 item 4 has owed since V2. `Muscle` is
# synthesised too, because nothing in the export fills r < 76.5 mm.
# ⚠️ The bores are Ø7.0 and Ø10.0, not the Ø9.0 / Ø11.0 the spec gives as intima-to-intima:
# the CAD read CFA_dia/CFV_dia as the tube's OUTER diameter. Logged in CAD_CHANGELOG section 5.
BODY_ORDER_V3 = ["SoftTissue", "Muscle", "Femur", "Sheath", "FemoralCanal",
                 "Vein", "Artery", "BloodVein", "BloodArtery", "Nerve"]
# 🚨 V4 ADDS THE BACKSTOP, AND NOT HAVING THIS LIST COST US A SILENTLY WRONG USD.
# On 2026-08-12 `build_leg_usd.py` auto-detected V4 as "v3" -- its test is "is Femur present?",
# and V4 ships a Femur too -- so it authored the V4 leg from BODY_ORDER_V3: ten bodies with
# `PubicRamus` ABSENT, while the FEM in the same campaign pinned it ("rigid inclusions
# ['Femur', 'PubicRamus']", out/campaign/2026-08-12/press_v4_gap12.log). The stage and the
# solver disagreed about whether the phantom has a backstop at all, and nothing raised. That
# is the one body the whole ramus-gap sweep exists to test, and the render would have shown
# no acoustic shadow where the bone is -- a plausible-looking picture of the wrong phantom.
# 🪤 So detect V4 on `PubicRamus`, NEVER on `Femur`.
# Nesting: the ramus is OUTSIDE the sheath (it is the floor the sheath's posterior fascia lies
# on -- see BODIES["PubicRamus"]), so it authors before the sheath and innermost-wins hands the
# vessels back to the vessels.
# `Perivascular` sits INSIDE the sheath and OUTSIDE the vessels, so it authors between them
# and innermost-wins hands the lumens back to the lumens. Added to this list 2026-08-18, when
# build_perivascular_obj.py finally gave it an OBJ -- before that the FEM had a soft bore and
# the renderer did not, and every figure was a picture of a phantom the solver was not solving.
BODY_ORDER_V4 = ["SoftTissue", "Muscle", "Femur", "PubicRamus", "Sheath", "Perivascular",
                 "FemoralCanal", "Vein", "Artery", "BloodVein", "BloodArtery", "Nerve"]

# The perivascular fill IS soft tissue, by reference rather than by a second copy of the
# number. Bound here instead of in the literal above so there is exactly one 3 kPa in this
# file and `E_src="ref:SoftTissue"` can be taken at face value.
BODIES["Perivascular"]["E_pa"] = BODIES["SoftTissue"]["E_pa"]

# 📌 `Perivascular` NOW HAS AN OBJ, as of 2026-08-18 (build_perivascular_obj.py), built from
# the sheath's OWN INNER SURFACE so the render and the FEM cannot drift apart. Before that it
# was FEM-only and the asymmetry had to be stated in every caption: the solver saw a soft bore
# and the beam saw uniform sheath. That is defect class 1 -- two artifacts nobody reconciles --
# and stating it was better than hiding it but was never a fix.
# ⚠️ Rebuild the OBJ whenever Sheath.obj changes, or the two diverge again silently.
# V5 (2026-10-21) is the cast gel leg: V4 with the pubic ramus removed and the limb cut at s = 81 mm
# (anatomy_meshing/v5_from_v4.py). Same bodies otherwise, same order, same materials.
BODY_ORDER_V5 = [b for b in BODY_ORDER_V4 if b != "PubicRamus"]

# ── THE CAST GEL LEG (2026-10-28) ────────────────────────────────────────────────────────
# The physical phantom, for the validation runs. Record and sources:
#   the cast record (not distributed)
# Geometry is V5's outline (cast fat/muscle radii 92.07 / 76.20 mm against V5's 92.50 / 76.50 mm), with
# commercial silicone tubes for the vessels and no sheath, canal, perivascular bore or nerve.
# Two more provenance tiers, used only here:
#   "supplier" -- a manufacturer's published value (SimuGel datasheet v2.3, McMaster product pages).
#   "textbook" -- a standard published relation or handbook value, not measured on this part.
# 🟡 The gels are MIXES of SimuGel formulations. Moduli and densities are mass-weighted means of the
# datasheet values: the supplier's guidance, relayed 2026-10-28, not printed in the datasheet. The
# +/- in E_note is the datasheet batch spread combined by mass, assuming independent batches.
# ⚠️ Acoustic: raysim has no SimuGel and no silicone. Each body is mapped to the nearest BUILT-IN
# material until those are added to the container (the route `air` took, 2026-07-16).
BODIES.update({
    "PhantomFat": dict(
        role="cast fat layer, radius 76.20 -> 92.07 mm (acrylic tube bore)",
        raysim_material="fat", raysim_src="nvidia",
        acoustic_true="SimuGel #2/#0 mix: c 1449-1457 m/s, Z 1.37-1.41 MRayl, 2.5-3.4 dB/cm at 5 MHz",
        acoustic_gap="raysim `fat` (1450 m/s, Z 1.38, 0.63 dB/cm/MHz) is close on speed and impedance. "
                     "Speckle parameters for SimuGel are published nowhere; `fat`'s are a stand-in.",
        E_pa=380.0e3, E_src="supplier",
        E_note="SimuGel #2 454 g (0.26 MPa) + #0 272 g (0.58 MPa), mixed by mass: 0.380 +/- 0.039 MPa",
        nu=0.49, nu_src="textbook", rho=959.2, rho_src="supplier",
    ),
    "PhantomMuscle": dict(
        role="cast muscle core, radius 0 -> 76.20 mm (pipe bore)",
        raysim_material="fat", raysim_src="nvidia",
        acoustic_true="SimuGel #0/#1 mix: c 1449-1465 m/s, Z 1.37 MRayl, 3.0-3.4 dB/cm at 5 MHz",
        acoustic_gap="the same gel family as the fat, so raysim `fat` again. Mapping it to `muscle` "
                     "(1580 m/s, Z 1.70) would invent a fat/muscle echo the cast barely has.",
        E_pa=513.3e3, E_src="supplier",
        E_note="SimuGel #0 454 g (0.58 MPa) + #1 227 g (0.38 MPa), mixed by mass, no extra cream: "
               "0.513 +/- 0.055 MPa",
        nu=0.49, nu_src="textbook", rho=941.9, rho_src="supplier",
    ),
    "PhantomCFA": dict(
        role="artery tube WALL, McMaster 5041K761 silicone 70A, OD 11.0 / ID 9.0 mm (1.0 mm wall)",
        raysim_material="muscle", raysim_src="nvidia",
        acoustic_true="silicone rubber, c ~1000 m/s, Z ~1.0-1.1 MRayl, strongly attenuating",
        acoustic_gap="🚨 no silicone in raysim. `muscle` gives the wall an impedance step against "
                     "the gel, so it shows as an edge, but its speed and attenuation are wrong.",
        E_pa=5.5e6, E_src="textbook",
        E_note="Durometer 70A (supplier) converted by Gent's relation: about 5.5 MPa. Not measured "
               "on this tube; the plan brackets it x0.5 and x2.",
        nu=0.49, nu_src="textbook", rho=1150.0, rho_src="textbook",
    ),
    "PhantomCFV": dict(
        role="vein tube WALL, McMaster 5054K827 silicone 35A, OD 13.0 / ID 10.0 mm (1.5 mm wall)",
        raysim_material="muscle", raysim_src="nvidia",
        acoustic_true="silicone rubber, c ~1000 m/s, Z ~1.0-1.1 MRayl, strongly attenuating",
        acoustic_gap="🚨 no silicone in raysim; as PhantomCFA.",
        E_pa=1.4e6, E_src="textbook",
        E_note="Durometer 35A (supplier) converted by Gent's relation: about 1.4 MPa. Not measured "
               "on this tube; the plan brackets it x0.5 and x2.",
        nu=0.49, nu_src="textbook", rho=1150.0, rho_src="textbook",
    ),
    "PhantomCFALumen": dict(
        role="artery tube bore, 9.0 mm, water-filled, open to an extended tube",
        raysim_material="water", raysim_src="nvidia",
        acoustic_true="water", acoustic_gap="none -- raysim's `water` is what is in the tube.",
        E_pa=None, E_src="n/a",
        E_note="VOID in the FEM. Open tubing extension, so no trapped volume; any lumen pressure is "
               "the hydrostatic head, applied as a surface load in its own case.",
        nu=None, nu_src="n/a", rho=998.0, rho_src="textbook",
    ),
    "PhantomCFVLumen": dict(
        role="vein tube bore, 10.0 mm, water-filled, open to an extended tube",
        raysim_material="water", raysim_src="nvidia",
        acoustic_true="water", acoustic_gap="none -- raysim's `water` is what is in the tube.",
        E_pa=None, E_src="n/a",
        E_note="VOID in the FEM; as PhantomCFALumen. This is the lumen whose area is compared.",
        nu=None, nu_src="n/a", rho=998.0, rho_src="textbook",
    ),
})
# The femur is V5's anatomical femur (confirmed 2026-10-28), so it keeps the `Femur` entry and is
# pinned by RIGID_E_PA. The FREE-femur case (the cast's femur is held only by gel) needs its own
# handling in the FEM builder; it is not a material-table change.
# nu = 0.49 for the four phantom bodies: "nearly incompressible, nu -> 1/2", Mott, Dorgan & Roland 2008, J Sound Vib
# 312:572-575 (on disk, i4h-references/General). The paper gives no number; 0.49 is our choice inside that range,
# below 0.499 because linear tets lock (stiffen) as nu -> 1/2. Checked against 0.495 in its own case. B/G: 0.49 -> ~50.
# The V5 tissue keeps nu = 0.45, an unsourced choice; changing it would change the anatomy results.
BODY_ORDER_CAST = ["PhantomFat", "PhantomMuscle", "Femur",
                   "PhantomCFV", "PhantomCFA", "PhantomCFVLumen", "PhantomCFALumen"]
BODY_ORDERS = {"v2": BODY_ORDER_V2, "v3": BODY_ORDER_V3, "v4": BODY_ORDER_V4, "v5": BODY_ORDER_V5,
               "cast": BODY_ORDER_CAST}


def detect_generation(have):
    """Which CAD generation a set of body names is -- by a body only that generation ships.

    🪤 `Femur` is not that body: V3 and V4 both ship one, which is exactly how the V4 leg USD
    came out without its backstop. Test most-specific generation first.
    """
    have = set(have)
    if "PhantomMuscle" in have:   # only the cast gel leg ships it (2026-10-28)
        return "cast"
    if "PubicRamus" in have:
        return "v4"
    # V5 is V4 without the ramus. Of V3/V4/V5 only V4 and V5 ship the perivascular bore, so its
    # presence without a ramus is V5. Writers pass the generation explicitly; this is the fallback.
    if "Perivascular" in have:
        return "v5"
    if "Femur" in have or "Sheath" in have:
        return "v3"
    return "v2"


# 🚨 Geometry that is itself a modelling choice, carried here so it travels with the
# properties (see Phantoms/Femoral/V2/README.md "Still unresolved"):
UNSOURCED_GEOMETRY = [
    "Femoral nerve diameter (10.0 mm) -- Fenech, Roche & Boyle 2024 state normative "
    "sonographic measurements for this nerve do not exist.",
    "Artery<->vein centre-to-centre separation (15.0 mm) -- no source on disk reports it.",
    "Wall thicknesses (artery 1.00 / vein 0.50 mm) -- marked 'Estimated' in the Biometric table.",
    "Groin-level layout only: the vein is medial at Poupart's ligament, posterior higher up, "
    "lateral distally. Parallel rods are valid at the groin and nowhere else.",
]


def raysim_mesh_map(nerve_material: str | None = None):
    """[(body, material)] in the order the world should be built."""
    out = []
    for name, b in BODIES.items():
        mat = nerve_material if (name == "Nerve" and nerve_material) else b["raysim_material"]
        out.append((name, mat))
    return out


# ── THE ONE WRITER OF anatomy_config.json ────────────────────────────────────────────────
# 🚨 THERE USED TO BE ONE WRITER PER SPECIMEN AND THE THIRD ONE WAS NEVER WRITTEN.
# `stl_to_raysim.py` emits it for V2, `build_v3_anatomy.py` emits it for V3, and
# `v4_to_section.py` emitted NOTHING -- so every V4 directory on disk (2026-08-12 campaign)
# had ten OBJs and no config. The consequences were not "a missing file":
#   · `build_leg_usd.py:147` SystemExits on the missing config BEFORE its --cad auto-detect
#     runs, so the campaign passed `--cad v3` to get past it and authored the V4 leg from
#     BODY_ORDER_V3 -- ten prims with PubicRamus absent, while the FEM in the same campaign
#     pinned it. The stage and the solver disagreed about whether the phantom has a backstop.
#   · `gel_body.py` skipped its `_provenance.gel` block (it was update-only), so
#     `render_deformed.py` -- which reads the standoff and the gel thickness out of exactly
#     that block, because it runs inside a container that cannot import probe_geometry.py --
#     had nothing to read.
#   · `sweep_circumference.py` / `render_bmode_femoral.py` cannot build a world without it.
# 📌 So the fix is not "add a third copy". It is ONE writer, here, next to the body table it
# has to agree with, called by every converter. A field this function cannot derive is a
# REQUIRED ARGUMENT -- a default here would be a modelling choice arriving silently in a file
# that everything downstream trusts as a measurement.
_GEOMETRY_REQUIRED = ("r_skin_mm", "r_shell_inner_mm", "ligament_lps_z_mm",
                      "image_plane_s_mm", "target_plane_s_mm", "targets")


def emit_anatomy_config(out_dir, *, geometry, provenance, background_material,
                        generation=None, nerve_material=None):
    """Write `<out_dir>/anatomy_config.json` from the OBJs that are actually in `out_dir`.

    The mesh list is taken from the DIRECTORY, cross-checked against BODY_ORDERS[generation],
    and written in that generation's outermost-first author order -- so the config, the USD
    author order and raysim's innermost-wins nesting come from one list rather than three.

    `generation` defaults to `detect_generation()` on the files present. ⚠️ Pass it explicitly
    when the caller knows which generation it just built: an auto-detect that is wrong writes
    a config that LOOKS right.

    Every key in `_GEOMETRY_REQUIRED` must be present in `geometry` -- a KeyError beats a
    plausible default. `provenance` is copied verbatim; `gel_body.py` adds `_provenance.gel`
    afterwards and is the only other thing allowed to touch this file.
    """
    import json as _json
    import os as _os

    have = sorted(f[:-4] for f in _os.listdir(out_dir) if f.endswith(".obj"))
    have_set = set(have) - {"Gel"}          # the gel is not anatomy; gel_body.py appends it
    gen = generation or detect_generation(have_set)
    if gen not in BODY_ORDERS:
        raise SystemExit(f"!!! unknown CAD generation {gen!r}; known: {sorted(BODY_ORDERS)}")
    order = BODY_ORDERS[gen]

    missing = [b for b in order if b not in have_set]
    extra = sorted(have_set - set(order))
    if missing:
        raise SystemExit(
            f"!!! {out_dir} is missing {missing} for CAD generation {gen!r}.\n"
            f"    Refusing to write a config that names a body list the meshes do not have --\n"
            f"    that is the same class of failure as the V4 leg USD without its backstop.")
    if extra:
        # Not fatal, but it must never be silent: an OBJ nothing references is either a body
        # the order forgot (the PubicRamus case) or scratch that will confuse the next reader.
        print(f"    ⚠️ {extra} are in {out_dir} but not in BODY_ORDER_{gen.upper()} -- they "
              f"will NOT be rendered, imaged or authored into the USD.")

    for k in _GEOMETRY_REQUIRED:
        if k not in geometry:
            raise SystemExit(
                f"!!! emit_anatomy_config: geometry[{k!r}] is required and has no default.\n"
                f"    Derive it from an artifact the caller measured; a number invented here\n"
                f"    would arrive in anatomy_config.json wearing the authority of a measurement.")

    meshes = []
    for name in order:
        mat = (nerve_material if (name == "Nerve" and nerve_material)
               else BODIES[name]["raysim_material"])
        meshes.append({"file": f"{name}.obj", "material": mat})

    cfg = {
        "anatomy": {
            "mesh_dir": _os.path.abspath(out_dir),
            "background_material": background_material,
            "meshes": meshes,
        },
        "geometry": dict(geometry),
        "_provenance": dict(provenance, cad_generation=gen,
                            body_order=f"BODY_ORDER_{gen.upper()}",
                            written_by="materials.emit_anatomy_config"),
    }
    path = _os.path.join(out_dir, "anatomy_config.json")
    _json.dump(cfg, open(path, "w"), indent=2)
    print(f">>> wrote {path}  ({gen}, {len(meshes)} bodies, background {background_material!r})")
    return cfg


def summary() -> str:
    rows = ["body          raysim_mat  src      E (kPa)   src    rho    role",
            "-" * 100]
    for n, b in BODIES.items():
        e = "void" if b["E_pa"] is None else f"{b['E_pa']/1e3:8.1f}"
        rows.append(f"{n:<13} {b['raysim_material']:<11} {b['raysim_src']:<8} {e:>8}  "
                    f"{b['E_src']:<6} {b['rho']:>6.0f}  {b['role'][:40]}")
    return "\n".join(rows)


if __name__ == "__main__":
    print(summary())
    print("\nAcoustic gaps (what the renderer cannot currently express):")
    for n, b in BODIES.items():
        if b["acoustic_gap"].startswith(("none", "static")):
            continue
        print(f"\n  {n}: {b['acoustic_gap']}")
    print("\nUnsourced geometry carried from V2:")
    for u in UNSOURCED_GEOMETRY:
        print(f"  - {u}")
