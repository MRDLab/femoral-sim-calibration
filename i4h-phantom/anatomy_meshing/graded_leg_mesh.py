"""A graded tetrahedral mesh of the WHOLE V4 limb segment, so the press acts on the leg and not on a block.

    $PY_PHYSICS anatomy_meshing/graded_leg_mesh.py \\
        --anatomy-dir femoral_v4_gap12 --station-s 20 --fine-dx-mm 1.5 --coarse-dx-mm 10 --grade-mm 25 \\
        --arc-deg 45 --s-window 0 40 --fine-depth-mm auto --perivascular --out out/mesh_leg/tetmesh_leg_s020_f1p5_c10.npz

Directed 2026-09-09 evening (photos/2026-09-10/README.md §1-2). Element size:
  * FINE  (--fine-dx-mm, 1.5 mm) inside the reachable window: s in --s-window (0..40 mm), |angle| <= --arc-deg (45 deg)
    about the anterior pole, depth from the skin down to the deepest lumen wall (+ one element) when --fine-depth-mm auto
  * COARSE (--coarse-dx-mm, 10 mm) everywhere else, with a linear ramp over --grade-mm (25 mm) so there is no jump
    ("1.5 -> 4 -> 7 -> 10", the author). The ramp distance is measured from the fine region's boundary.

Geometry: a cylinder of the CAD skin radius over the leg USD's axial span (s = 0..120 mm; FEM Y = LPS z - z_station,
so y = +0.020 .. -0.100 m). Meshed with gmsh (HXT) under a size callback.

Materials: the SAME classifier every study used (femoral_fem.load_section + classify, innermost-wins on the s = 20 mm
section, with the perivascular collar and the sheath modulus set as export_tetmesh.py sets them), sub-sampled at 11
points per tet and homogenised with femoral_fem's own rules (RIGID_CELL_POLICY "exclude": bone-dominant -> pinned
backstop; > half void -> void; else arithmetic Lame mean). Axial variation is carried by each body's s-range read from
its CAD OBJ: a body is only present where its OBJ exists (artery 0..37.5, vein 0..80, sheath 0..35, nerve 0..25,
canal 0..12.5, ramus -7.6..47.6). 🔴 STATED APPROXIMATION: the cross-section SHAPE is the s = 20 mm section extruded
along s; only body presence varies with s. The profunda/superficial femoral beyond s = 37.5 are not modelled.

Lumens are voids (tets removed). Their boundary triangles are kept, oriented toward the void, so closure can be read
as the plane-slice area at the station (mesh-independent, no node ring). Bone: nodes pinned. Table patch: nodes in the
bottom --table-cap-mm of the leg, pinned. Ends: free.

Checks that can fail (pre-registered 2026-09-10 §3.1): skin radius; lumen void centroids vs the section discs at the
station within 1 mm; artery x < vein x; per-body volume fractions inside the fine sector at the station within 5 % of
the classifier's own 2-D areas (Monte Carlo); no inverted tets at rest.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--anatomy-dir", default="femoral_v4_gap12")
ap.add_argument("--station-s", type=float, default=20.0)
ap.add_argument("--fine-dx-mm", type=float, default=1.5)
ap.add_argument("--coarse-dx-mm", type=float, default=10.0)
ap.add_argument("--grade-mm", type=float, default=25.0, help="ramp distance fine -> coarse")
ap.add_argument("--arc-deg", type=float, default=45.0)
ap.add_argument("--s-window", type=float, nargs=2, default=(0.0, 40.0))
ap.add_argument("--fine-depth-mm", default="auto", help="'auto' = deepest lumen wall + one fine element")
ap.add_argument("--leg-s-range", type=float, nargs=2, default=(0.0, 120.0), help="axial span of the mesh, mm from the ligament")
ap.add_argument("--table-cap-mm", type=float, default=6.0, help="the bottom sliver of the leg pinned as the table patch")
ap.add_argument("--probe-w", type=float, default=0.040)
ap.add_argument("--probe-len", type=float, default=0.012)
ap.add_argument("--sheath-kpa", type=float, default=20.0)
ap.add_argument("--perivascular", action="store_true")
ap.add_argument("--nsub", type=int, default=11)
ap.add_argument("--bone-dx-mm", type=float, default=5.0,
                help="element size inside and within --bone-margin-mm of the bone bodies (Femur, PubicRamus) over their s-range, so the bony "
                     "backstop survives the coarse region. 2026-09-09 evening: at coarse 30 mm the femur had 4 tets and the ramus none.")
ap.add_argument("--bone-margin-mm", type=float, default=4.0)
ap.add_argument("--ramus-dx-mm", type=float, default=2.5, help="element size for the PubicRamus (a thin ring; 5 mm elements miss it)")
ap.add_argument("--materials", choices=("anatomy", "cast"), default="anatomy",
                help="cast (2026-10-28): the role bodies take the cast's Phantom* values from materials.py")
ap.add_argument("--wall-dx-mm", type=float, default=None,
                help="element size inside each vessel wall within the fine s-window (default: none, as V5)")
ap.add_argument("--wall-margin-mm", type=float, default=1.0, help="refine this far either side of the wall")
ap.add_argument("--gel-sd", type=float, default=0.0,
                help="cast only: shift both gels by this many supplier SDs (fat 0.039, muscle 0.055 MPa), e.g. -1 or +1")
ap.add_argument("--wall-scale", type=float, default=1.0,
                help="cast only: multiply both tube-wall moduli (after any bending equivalence), e.g. 0.5 or 2")
ap.add_argument("--femur-free-mpa", type=float, default=None,
                help="2026-10-29, cast only: the cast's femur is a printed bone held only by the gel. Mesh it as a free elastic "
                     "body of this modulus (MPa, below the 1 GPa rigid-pin threshold) instead of a pinned backstop")
ap.add_argument("--out", required=True)
args = ap.parse_args()

import femoral_fem as F  # noqa: E402  (physics venv)
import gmsh  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OBJ = (ROOT / "out" / args.anatomy_dir).resolve()    # the real out/, not a worktree symlink
sec_path = OBJ / f"section_s{int(round(args.station_s)):03d}.json"
if not sec_path.exists():
    raise SystemExit(f"!!! no section at {sec_path}")
sec = json.load(open(sec_path))
assert sec["units"] == "m" and "X = LPS x, Z = -LPS y" in sec["frame"], sec["frame"]
z_station = float(sec["section_lps_z_mm"])
F.BODIES["Sheath"]["E_pa"] = args.sheath_kpa * 1e3
F.BODIES["Sheath"]["nu"] = F.NU
# 2026-10-28, the cast gel leg: the section keeps V5's ROLE names so every tracker works unchanged; the cast's own
# materials go onto those roles here, from the Phantom* entries in materials.py (sources recorded there).
CAST_ROLES = {"SoftTissue": "PhantomFat", "Muscle": "PhantomMuscle", "Artery": "PhantomCFA", "Vein": "PhantomCFV"}
if args.materials == "cast":
    assert sec.get("generation") == "cast", f"--materials cast needs the cast section, got generation {sec.get('generation')!r}"
    for role, src in CAST_ROLES.items():
        for k in ("E_pa", "nu", "rho"):
            F.BODIES[role][k] = F.BODIES[src][k]
        print(f">>> cast material: {role:<10} <- {src:<13} E {F.BODIES[role]['E_pa']/1e3:8.1f} kPa  nu {F.BODIES[role]['nu']}  "
              f"rho {F.BODIES[role]['rho']}", flush=True)
    if args.femur_free_mpa is not None:
        assert args.femur_free_mpa * 1e6 < F.RIGID_E_PA, "a free femur must sit below the rigid-pin threshold"
        F.BODIES["Femur"]["E_pa"] = args.femur_free_mpa * 1e6
        print(f">>> FREE femur: E {args.femur_free_mpa:g} MPa, not pinned (the cast's printed bone is held only by the gel)", flush=True)
    for wall, e in (sec.get("wall_E_pa") or {}).items():   # bending-equivalent walls, declared by the section itself
        F.BODIES[wall]["E_pa"] = float(e)
        print(f">>> section wall override: {wall} E {e/1e6:.3f} MPa ({sec['wall_E_basis']})", flush=True)
    # 2026-10-29: the plan's sensitivity cases. SDs are the supplier batch spread combined by mass (materials.py notes).
    GEL_SD_PA = {"SoftTissue": 0.039e6, "Muscle": 0.055e6}
    for role, sd in GEL_SD_PA.items():
        F.BODIES[role]["E_pa"] += args.gel_sd * sd
    for wall in ("Artery", "Vein"):
        F.BODIES[wall]["E_pa"] *= args.wall_scale
    if args.gel_sd or args.wall_scale != 1.0:
        print(f">>> sensitivity: gels {args.gel_sd:+g} SD (fat {F.BODIES['SoftTissue']['E_pa']/1e3:.1f}, muscle "
              f"{F.BODIES['Muscle']['E_pa']/1e3:.1f} kPa); walls x{args.wall_scale:g} (artery {F.BODIES['Artery']['E_pa']/1e6:.3f}, "
              f"vein {F.BODIES['Vein']['E_pa']/1e6:.3f} MPa)", flush=True)
elif sec.get("generation") == "cast":
    raise SystemExit("!!! the cast section needs --materials cast; meshing it with tissue values would be neither phantom")
F.load_section(str(sec_path), perivascular=args.perivascular)
B = sec["bodies"]
R_SKIN = float(B["SoftTissue"]["r_out"])
lumen_disc = {"artery": (np.array(B["BloodArtery"]["c"]), float(B["BloodArtery"]["r"])),
              "vein": (np.array(B["BloodVein"]["c"]), float(B["BloodVein"]["r"]))}
deepest = min(c[1] - r for c, r in lumen_disc.values())            # lowest lumen wall (FEM z), m
fine_depth = (R_SKIN - deepest + args.fine_dx_mm * 1e-3) if args.fine_depth_mm == "auto" else float(args.fine_depth_mm) * 1e-3
FINE, COARSE, GRADE = args.fine_dx_mm * 1e-3, args.coarse_dx_mm * 1e-3, args.grade_mm * 1e-3
ARC = np.radians(args.arc_deg)
y_of_s = lambda s_mm: (z_station - (s_mm - args.station_s) - 0.0) * 1e-3 if False else ((args.station_s - s_mm) * 1e-3)  # FEM y from s: y = (station - s)/1000
Y_LO, Y_HI = y_of_s(args.leg_s_range[1]), y_of_s(args.leg_s_range[0])       # s=120 -> y=-0.100, s=0 -> y=+0.020
WIN_Y = (y_of_s(args.s_window[1]), y_of_s(args.s_window[0]))
R_IN = R_SKIN - fine_depth
print(f">>> skin r {R_SKIN*1e3:.3f} mm; deepest lumen wall at depth {(R_SKIN-deepest)*1e3:.2f} mm -> fine depth {fine_depth*1e3:.2f} mm "
      f"(r >= {R_IN*1e3:.2f} mm); fine window y in [{WIN_Y[0]*1e3:+.1f}, {WIN_Y[1]*1e3:+.1f}] mm (s {args.s_window[0]:g}..{args.s_window[1]:g}), "
      f"arc ±{args.arc_deg:g} deg; leg y in [{Y_LO*1e3:+.0f}, {Y_HI*1e3:+.0f}] mm", flush=True)

# each body's axial presence, from its own CAD OBJ (LPS z range -> s range)
def obj_s_range(name):
    p = OBJ / f"{name}.obj"
    if not p.exists():
        return None
    zs = [float(l.split()[3]) for l in open(p) if l.startswith("v ")]
    lo, hi = min(zs), max(zs)                                      # LPS z, mm
    return (z_station + args.station_s - hi, z_station + args.station_s - lo)   # s = z_station + station - LPS z  (LPS z decreases distally)
S_RANGE = {n: obj_s_range(n) for n in ("Artery", "BloodArtery", "Vein", "BloodVein", "Nerve", "Sheath", "Perivascular", "FemoralCanal", "PubicRamus", "Femur", "Muscle", "SoftTissue")}
print(">>> body s-ranges from the CAD OBJs (mm): " + "  ".join(f"{n} {r[0]:.1f}..{r[1]:.1f}" for n, r in S_RANGE.items() if r), flush=True)


# bone footprint (section discs / polygons) for the bone refinement region
_bone_shapes = []
for _n in ("Femur", "PubicRamus"):
    _b = B.get(_n)
    if not _b: continue
    if _b.get("kind") == "disc":
        _bone_shapes.append((_n, "disc", np.array(_b["c"]), float(_b["r"])))
    elif _b.get("kind") == "polygon":
        _bone_shapes.append((_n, "pts", np.array(_b["pts"], dtype=float), 0.0))
BONE_DX, BONE_MARGIN = args.bone_dx_mm * 1e-3, args.bone_margin_mm * 1e-3


BONE_DX_OF = {"Femur": BONE_DX, "PubicRamus": args.ramus_dx_mm * 1e-3}


def bone_size(x, y, z, coarse_here):
    """Size cap from the bone bodies: each body's own dx inside its margin, ramping back to coarse_here over GRADE."""
    s_here = args.station_s - y * 1e3
    cap = np.inf
    for name, kind, geo, r in _bone_shapes:
        rng = S_RANGE.get(name)
        if rng and not (rng[0] <= s_here <= rng[1]):
            continue
        if kind == "disc":
            db = max(np.hypot(x - geo[0], z - geo[1]) - r, 0.0)
        else:
            db = float(np.min(np.hypot(geo[:, 0] - x, geo[:, 1] - z)))
        bdx = BONE_DX_OF[name]
        if db < BONE_MARGIN:
            cap = min(cap, bdx)
        elif db < BONE_MARGIN + GRADE:
            cap = min(cap, bdx + (coarse_here - bdx) * (db - BONE_MARGIN) / GRADE)
    return cap


def size_at(x, y, z):
    """Element size, m: fine inside the sector, linear ramp to coarse over GRADE outside it; never coarser than BONE_DX
    within BONE_MARGIN of a bone body (the backstop must be meshed, whatever the coarse size).
    The ramp distance is the EUCLIDEAN distance to the fine region (radial in every direction, rounded isolines), not
    the largest single component -- by direction 2026-09-09 evening: "the expansion is radial and not just directional"."""
    r = np.hypot(x, z)
    d_r = max(R_IN - r, 0.0)                                                        # below the fine region's floor
    d_y = max(abs(y - 0.5 * (WIN_Y[0] + WIN_Y[1])) - 0.5 * (WIN_Y[1] - WIN_Y[0]), 0.0)   # beyond the s window
    d_a = max((abs(x) - z * np.tan(ARC)) * np.cos(ARC), 0.0)                        # outside the ±arc wedge
    d = np.sqrt(d_r * d_r + d_y * d_y + d_a * d_a)
    size = FINE + (COARSE - FINE) * min(1.0, d / GRADE)
    return min(size, bone_size(x, y, z, size), wall_size(x, y, z))


# 2026-10-28: the cast's silicone walls are 1.0 / 1.5 mm, thinner than a 1.5 mm element. Within the fine s-window,
# cap the size at WALL_DX inside each wall band (+/- WALL_MARGIN), ramping back to FINE over 2 * WALL_MARGIN.
WALLS = [(np.array(B[w]["c"]), float(B[w]["r_in"]), float(B[w]["r_out"])) for w in ("Artery", "Vein") if B.get(w, {}).get("kind") == "annulus"]
WALL_DX = None if args.wall_dx_mm is None else args.wall_dx_mm * 1e-3
WALL_MARGIN = args.wall_margin_mm * 1e-3


def wall_size(x, y, z):
    if WALL_DX is None or not (WIN_Y[0] <= y <= WIN_Y[1]):
        return np.inf
    cap = np.inf
    for c, ri, ro in WALLS:
        dw = max(abs(np.hypot(x - c[0], z - c[1]) - 0.5 * (ri + ro)) - 0.5 * (ro - ri), 0.0)
        if dw <= WALL_MARGIN:
            cap = min(cap, WALL_DX)
        elif dw <= 3 * WALL_MARGIN:
            cap = min(cap, WALL_DX + (FINE - WALL_DX) * (dw - WALL_MARGIN) / (2 * WALL_MARGIN))
    return cap


# ---------------------------------------------------------------- gmsh
t0 = time.perf_counter()
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("leg")
gmsh.model.occ.addCylinder(0.0, Y_LO, 0.0, 0.0, Y_HI - Y_LO, 0.0, R_SKIN)
gmsh.model.occ.synchronize()
for opt, val in (("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.MeshSizeExtendFromBoundary", 0),
                 ("Mesh.Algorithm", 6), ("Mesh.Algorithm3D", 10), ("Mesh.Optimize", 1), ("Mesh.OptimizeNetgen", 0),
                 ("Mesh.MeshSizeMin", (FINE if WALL_DX is None else min(FINE, WALL_DX)) * 0.8), ("Mesh.MeshSizeMax", COARSE)):
    gmsh.option.setNumber(opt, val)
gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: size_at(x, y, z))
gmsh.model.mesh.generate(3)
node_tags, coords, _ = gmsh.model.mesh.getNodes()
etypes, etags, enodes = gmsh.model.mesh.getElements(3)
assert len(etypes) == 1 and etypes[0] == 4, f"expected linear tets only, got types {etypes}"
tag2idx = {int(t): i for i, t in enumerate(node_tags)}
X = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
T = np.array([tag2idx[int(t)] for t in enodes[0]], dtype=np.int64).reshape(-1, 4)
gmsh.finalize()
print(f">>> gmsh: {len(X):,} nodes, {len(T):,} tets in {time.perf_counter()-t0:.1f} s", flush=True)

# positive orientation
p = X[T]
vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0
flip = vol < 0
T[flip, 2], T[flip, 3] = T[flip, 3].copy(), T[flip, 2].copy()
vol = np.abs(vol)
assert (vol > 0).all(), "degenerate tets"

# ---------------------------------------------------------------- materials by sub-sampled classification
t0 = time.perf_counter()
# 11 barycentric sample points: centroid, 4 toward the vertices (0.7), 6 edge midpoints
bary = [np.full(4, 0.25)]
for i in range(4):
    b = np.full(4, 0.1); b[i] = 0.7; bary.append(b)
for i in range(4):
    for j in range(i + 1, 4):
        b = np.zeros(4); b[i] = b[j] = 0.5; bary.append(b)
bary = np.array(bary)[: args.nsub]
S_of = lambda y: args.station_s - y * 1e3       # s (mm) of a FEM y


def present(name, s):
    r = S_RANGE.get(name)
    return True if r is None else (r[0] <= s <= r[1])


def classify_at(x, y, z):
    """Innermost body present at this s; lumens named; outside-limb void -> SoftTissue (cylinder radius vs section)."""
    s = S_of(y)
    for name, contains in F.SECTION:
        if contains(x, z):
            if name == "void":
                for lum, (c, r) in lumen_disc.items():
                    if np.hypot(x - c[0], z - c[1]) <= r * 1.02:
                        return "BloodArtery" if lum == "artery" else "BloodVein"
                return "void_outside"
            if present(name, s):
                return name
    return "void_outside"


mu_of, lam_of, rho_of = {}, {}, {}
for n, b in F.BODIES.items():
    if b.get("E_pa") is not None:
        mu_of[n], lam_of[n] = F.lame(b["E_pa"], b["nu"]); rho_of[n] = b["rho"] or F.RHO_DEFAULT
mu_of["void_outside"], lam_of["void_outside"], rho_of["void_outside"] = mu_of["SoftTissue"], lam_of["SoftTissue"], rho_of["SoftTissue"]
tmat = np.zeros((len(T), 3), dtype=np.float64); tet_body = np.empty(len(T), dtype="U32"); trho = np.zeros(len(T))
keep = np.ones(len(T), dtype=bool); pin_bone = np.zeros(len(X), dtype=bool)
lumen_of_removed = np.empty(len(T), dtype="U16")
# 🪤 2026-10-21: every tie below is broken by `max(sorted(set(...)))`. A bare `max(set(...))` follows the per-process
# string-hash order, so two builds of identical input gave 234 of Mesh F's 134,375 tet LABELS differently (never a
# material, mass or node), 58 of them Perivascular <-> Vein -- the label the vein-wall sensitivity cases select by.
for ti in range(len(T)):
    P = bary @ X[T[ti]]
    names = [classify_at(*q) for q in P]
    lum = [n for n in names if n in ("BloodArtery", "BloodVein")]
    solid = [("SoftTissue" if n == "void_outside" else n) for n in names if n not in ("BloodArtery", "BloodVein")]
    if len(lum) * 2 > len(names):
        keep[ti] = False; lumen_of_removed[ti] = max(sorted(set(lum)), key=lum.count); tet_body[ti] = lumen_of_removed[ti]; continue
    rigid = [n for n in solid if F.BODIES.get(n, {}).get("E_pa") and F.BODIES[n]["E_pa"] >= F.RIGID_E_PA]
    soft = [n for n in solid if n not in set(rigid)]
    if rigid and (len(rigid) * 2 > len(solid) or not soft):
        dom = max(sorted(set(rigid)), key=rigid.count); fill = max(sorted(set(soft)), key=soft.count) if soft else "Muscle"
        tmat[ti] = (mu_of[fill], lam_of[fill], 1.0); trho[ti] = rho_of[fill]; tet_body[ti] = dom; pin_bone[T[ti]] = True; continue
    pool = soft if soft else solid
    tmat[ti] = (np.mean([mu_of[n] for n in pool]), np.mean([lam_of[n] for n in pool]), 1.0)
    trho[ti] = np.mean([rho_of[n] for n in pool]); tet_body[ti] = max(sorted(set(pool)), key=pool.count)
print(f">>> materials in {time.perf_counter()-t0:.1f} s; removed lumen tets: artery {(lumen_of_removed=='BloodArtery').sum():,}, vein {(lumen_of_removed=='BloodVein').sum():,}; "
      f"bone-pinned nodes {int(pin_bone.sum()):,}", flush=True)

# ---------------------------------------------------------------- lumen boundary faces (oriented toward the void), then drop void tets
FACES = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]     # face opposite vertex i, outward for a positive tet
def face_key(f):
    return tuple(sorted(int(v) for v in f))
void_faces = {}
for ti in np.flatnonzero(~keep):
    for fi in FACES:
        void_faces.setdefault(face_key(T[ti][list(fi)]), []).append(lumen_of_removed[ti])
lumen_tris = {"artery": [], "vein": []}
for ti in np.flatnonzero(keep):
    for fi in FACES:
        f = T[ti][list(fi)]
        k = face_key(f)
        if k in void_faces:
            lum = max(sorted(set(void_faces[k])), key=void_faces[k].count)
            lumen_tris["artery" if lum == "BloodArtery" else "vein"].append(f)     # outward from the kept tet = toward the void
T_keep = T[keep]; tmat = tmat[keep]; trho = trho[keep]; tet_body = tet_body[keep]; vol = vol[keep]
used = np.zeros(len(X), dtype=bool); used[T_keep.ravel()] = True
newidx = -np.ones(len(X), dtype=np.int64); newidx[used] = np.arange(int(used.sum()))
rest = X[used]; tets = newidx[T_keep]; pin_bone = pin_bone[used]
lumen_tris = {k: newidx[np.array(v, dtype=np.int64)] for k, v in lumen_tris.items()}
assert (tets >= 0).all() and all((v >= 0).all() for v in lumen_tris.values())
print(f">>> kept {len(rest):,} nodes, {len(tets):,} tets; lumen boundary triangles: artery {len(lumen_tris['artery']):,}, vein {len(lumen_tris['vein']):,}", flush=True)

# ---------------------------------------------------------------- pinned (bone + table patch), footprint, mass
r_xz = np.hypot(rest[:, 0], rest[:, 2])
table = rest[:, 2] <= -(R_SKIN - args.table_cap_mm * 1e-3)
pinned = np.flatnonzero(pin_bone | table).astype(np.int64)
surface = r_xz >= R_SKIN - 0.6 * FINE
footprint = np.flatnonzero(surface & (np.abs(rest[:, 0]) <= args.probe_w / 2) & (np.abs(rest[:, 1]) <= args.probe_len / 2)).astype(np.int64)
assert len(footprint) > 0, "probe footprint selected zero skin nodes"
pmass = np.zeros(len(rest)); np.add.at(pmass, tets.ravel(), np.repeat(vol * trho / 4.0, 4))
print(f">>> pinned {len(pinned):,} nodes (bone {int(pin_bone.sum()):,}, table patch {int(table.sum()):,}); footprint {len(footprint)} skin nodes; mass {pmass.sum():.3f} kg", flush=True)


# ---------------------------------------------------------------- closure operator: shared with the live loop (closure_ops.py)
from closure_ops import slice_area  # noqa: E402


rest_area = {k: slice_area(rest, v, 0.0) for k, v in lumen_tris.items()}
print(">>> rest lumen areas at the station (plane slice): " + "  ".join(f"{k} {a*1e6:.2f} mm^2 (disc {np.pi*lumen_disc[k][1]**2*1e6:.2f})" for k, a in rest_area.items()), flush=True)

# ---------------------------------------------------------------- checks that can fail
checks = {}
r_nodes = float(r_xz.max())
checks["skin: max node radius within one fine element of the section skin radius"] = (abs(r_nodes - R_SKIN) <= FINE, dict(r_nodes_mm=r_nodes*1e3, r_skin_mm=R_SKIN*1e3))
cen_all = X[T].mean(1)
for lum, (c, r) in lumen_disc.items():
    name = "BloodArtery" if lum == "artery" else "BloodVein"
    sel = (~keep) & (lumen_of_removed == name) & (np.abs(cen_all[:, 1]) <= 3e-3)
    cc = cen_all[sel][:, [0, 2]].mean(0) if sel.any() else np.array([np.nan, np.nan])
    dev = float(np.linalg.norm(cc - c)) * 1e3
    checks[f"{lum} void centroid at the station within 1 mm of the section disc centre"] = (dev <= 1.0, dict(void_mm=(cc*1e3).round(2).tolist(), disc_mm=(c*1e3).round(2).tolist(), dev_mm=round(dev, 3)))
ax = cen_all[(~keep) & (lumen_of_removed == "BloodArtery")][:, 0].mean(); vx = cen_all[(~keep) & (lumen_of_removed == "BloodVein")][:, 0].mean()
checks["lateral order: artery x < vein x (right thigh)"] = (ax < vx, dict(artery_x_mm=ax*1e3, vein_x_mm=vx*1e3))
# volume fractions inside the fine sector at the station slab vs the classifier's own 2-D areas (Monte Carlo)
rng = np.random.default_rng(0)
def in_sector(x, z):
    return (np.hypot(x, z) >= R_IN) & (np.hypot(x, z) <= R_SKIN) & (np.abs(x) <= z * np.tan(ARC))
n_mc = 300000; xs = rng.uniform(-R_SKIN, R_SKIN, n_mc); zs = rng.uniform(R_IN * np.cos(ARC) * 0.9, R_SKIN, n_mc)
m = in_sector(xs, zs); xs, zs = xs[m], zs[m]
mc = np.array([classify_at(x, 0.0, z) for x, z in zip(xs, zs)])
slab = np.abs(rest[tets].mean(1)[:, 1]) <= 2.5e-3
cen_k = rest[tets].mean(1); sec_k = slab & in_sector(cen_k[:, 0], cen_k[:, 2])
vol_sector = vol[sec_k].sum()
# add the removed lumen volume back for the fraction denominator (voids are part of the sector)
cen_rm = cen_all[~keep]; rm_slab = (np.abs(cen_rm[:, 1]) <= 2.5e-3) & in_sector(cen_rm[:, 0], cen_rm[:, 2])
p_rm = X[T[~keep]]; vol_rm = np.abs(np.einsum("ij,ij->i", np.cross(p_rm[:, 1] - p_rm[:, 0], p_rm[:, 2] - p_rm[:, 0]), p_rm[:, 3] - p_rm[:, 0]) / 6.0)
denom = vol_sector + vol_rm[rm_slab].sum()
frac_rows = []
for body in ("BloodArtery", "BloodVein", "Artery", "Vein", "Nerve", "Perivascular", "Sheath", "Muscle", "SoftTissue"):
    f_mc = float((mc == body).mean())
    if body in ("BloodArtery", "BloodVein"):
        f_mesh = float(vol_rm[rm_slab & (lumen_of_removed[~keep] == body)].sum() / denom)
    else:
        f_mesh = float(vol[sec_k & (tet_body == body)].sum() / denom)
    frac_rows.append((body, f_mc, f_mesh))
worst = max(abs(a - b) for _, a, b in frac_rows)
checks["body fractions in the fine sector at the station within 5 points of the classifier's 2-D areas"] = (worst <= 0.05, {b: dict(area_2d=round(a, 4), mesh=round(c, 4)) for b, a, c in frac_rows})
for _bn in ("Femur", "PubicRamus"):
    _sel = tet_body == _bn; _nt = int(_sel.sum())
    _nodes = np.unique(tets[_sel].ravel()) if _nt else np.array([], dtype=np.int64)
    if _bn == "Femur" and _bn in B and args.femur_free_mpa is not None:
        checks[f"Femur is meshed FREE (>= 50 tets, none of its own nodes pinned, E {args.femur_free_mpa:g} MPa)"] = (
            bool(_nt >= 50 and not pin_bone[_nodes].any()), dict(tets=_nt, pinned_nodes=int(pin_bone[_nodes].sum()), nodes=int(len(_nodes))))
    elif _bn in B:
        checks[f"{_bn} is meshed as a pinned backstop (>= 50 tets, all its nodes pinned)"] = (
            bool(_nt >= 50 and pin_bone[_nodes].all()), dict(tets=_nt, pinned_nodes=int(pin_bone[_nodes].sum()), nodes=int(len(_nodes))))
    else:
        # 2026-10-21, V5: the ramus was removed from the cast phantom. Absent from the section must mean absent from the mesh.
        checks[f"{_bn} is absent from the section, so it has no tets"] = (_nt == 0, dict(tets=_nt))
checks["no inverted tets at rest"] = (bool((vol > 0).all()), dict(min_vol_mm3=float(vol.min()*1e9), max_vol_mm3=float(vol.max()*1e9)))
if WALL_DX is not None:
    # the wall must be resolved where it is measured: at the station, at least 2 elements across each wall
    _cs = rest[tets].mean(1)
    for _w, (_c, _ri, _ro) in zip(("Artery", "Vein"), WALLS):
        _sel = (tet_body == _w) & (np.abs(_cs[:, 1]) <= 3e-3)
        _ed = np.concatenate([np.linalg.norm(rest[tets[_sel][:, i]] - rest[tets[_sel][:, j]], axis=1)
                              for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]) if _sel.any() else np.array([np.inf])
        _across = (_ro - _ri) / float(np.median(_ed))
        checks[f"{_w} wall resolved at the station: >= 2 elements across its {(_ro-_ri)*1e3:.2f} mm"] = (
            bool(_across >= 2.0), dict(wall_tets=int(_sel.sum()), median_edge_mm=round(float(np.median(_ed)) * 1e3, 3), across=round(_across, 2)))
bad = []
for k, (ok, ev) in checks.items():
    print(("  ✅ " if ok else "  🚨 ") + k + f"  {ev}", flush=True)
    if not ok:
        bad.append(k)

# element-size audit: edge lengths by region
e = np.concatenate([np.linalg.norm(rest[tets[:, i]] - rest[tets[:, j]], axis=1) for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
cen = rest[tets].mean(1); fine_mask = np.array([size_at(*c) <= FINE * 1.05 for c in cen])
print(f">>> tets in the fine region: {int(fine_mask.sum()):,} ({100*fine_mask.mean():.1f} %); fine-region tet volume median {np.median(vol[fine_mask])*1e9:.3f} mm^3; "
      f"edge length median all {np.median(e)*1e3:.2f} mm", flush=True)
uniq, cnt = np.unique(tet_body, return_counts=True)
print(">>> tets per body: " + "  ".join(f"{a}={b:,}" for a, b in zip(uniq, cnt)), flush=True)

out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
body_of = np.full(len(rest), "SoftTissue", dtype="U32")
meta = dict(kind="graded_leg", date="2026-09-09", built=time.strftime("%Y-%m-%d %H:%M"), fine_dx_m=FINE, coarse_dx_m=COARSE, grade_m=GRADE, arc_deg=args.arc_deg, s_window_mm=list(args.s_window),
            fine_depth_m=fine_depth, r_in_m=R_IN, leg_s_range_mm=list(args.leg_s_range), leg_y_m=[Y_LO, Y_HI], r_skin_m=R_SKIN, z_skin=R_SKIN, x_probe=0.0,
            station_s=args.station_s, z_station_lps_mm=z_station, section=str(sec_path), sheath_kpa=args.sheath_kpa, perivascular=args.perivascular,
            body_s_ranges_mm=S_RANGE, table_cap_m=args.table_cap_mm * 1e-3, bone_dx_m=BONE_DX, ramus_dx_m=args.ramus_dx_mm * 1e-3, bone_margin_m=BONE_MARGIN, n_nodes=len(rest), n_tets=len(tets), n_pinned=len(pinned),
            n_bone_pinned=int(pin_bone.sum()), n_table_pinned=int(table.sum()), checks={k: [ok, ev] for k, (ok, ev) in checks.items()},
            approximation="cross-section shape = s=20 mm section extruded along s; body presence by CAD s-range; lumens void; bone pinned; ends free")
if args.materials == "cast" or WALL_DX is not None:   # only for builds that use them, so V5's metadata is unchanged
    meta.update(materials=args.materials, cast_roles=CAST_ROLES if args.materials == "cast" else None, femur_free_mpa=args.femur_free_mpa, gel_sd=args.gel_sd, wall_scale=args.wall_scale,
                wall_dx_m=WALL_DX, wall_margin_m=WALL_MARGIN, generation=sec.get("generation"))
np.savez_compressed(out, rest=rest, tets=tets, tet_materials=tmat.astype(np.float32), particle_mass=pmass, pinned=pinned,
                    body_of=body_of, tet_body=tet_body, footprint=footprint,
                    ring_names=np.array(["artery", "vein"], dtype="U16"),
                    **{f"ring_{k}": np.unique(v.ravel()).astype(np.int64) for k, v in lumen_tris.items()},
                    **{f"lumen_tris_{k}": v.astype(np.int64) for k, v in lumen_tris.items()},
                    **{f"rest_area_{k}": np.array([rest_area[k]]) for k in lumen_tris},
                    closure_mode=np.array(["slice"], dtype="U8"), surf_names=np.array([], dtype="U16"), ny=np.array([0]), p_scale=np.array([1.0]),
                    x_probe=np.array([0.0]), z_skin=np.array([R_SKIN]), probe_w=np.array([args.probe_w]), probe_len=np.array([args.probe_len]),
                    dx=np.array([FINE]), meta=np.array([json.dumps(meta, default=str)], dtype="U200000"),
                    anatomy_dir=np.array([args.anatomy_dir], dtype="U64"), station_s=np.array([args.station_s]))
json.dump(meta, open(out.with_suffix(".json"), "w"), indent=1, default=str)
print(f">>> wrote {out} ({out.stat().st_size/1e6:.1f} MB), sha256 {hashlib.sha256(open(out,'rb').read()).hexdigest()[:12]}" + ("" if not bad else f"\n🚨 {len(bad)} CHECK(S) FAILED: {bad}"), flush=True)
if bad:
    raise SystemExit(2)
print("MESH DONE", flush=True)
