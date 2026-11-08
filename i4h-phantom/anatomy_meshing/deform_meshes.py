"""Warp the femoral CAD meshes by the FEM displacement field, so raysim can image them.

This is the link that has never existed in this project. Chapter 6 ends with force control
working and the image not responding to it -- "force and image are causally disconnected",
because the pose bridge anchors to a rigid body and the anatomy the ray tracer sees never
changes shape. Here the FEM's node displacements are carried onto the CAD surfaces and
written back out as OBJs, so the B-mode rendered from them is the B-mode OF the deformed
tissue. Press harder, the vein closes, and the image shows it.

METHOD. The FEM is a structured grid over a region of interest; the CAD meshes are smooth
full-length cylinders. Rather than image the FEM's own staircase surface -- which at 1.5 mm
spacing would put visible steps on a 9 mm lumen -- each CAD vertex takes a smoothly
weighted displacement from the FEM nodes near it:

    u(p) = sum_i w_i u_i / sum_i w_i        w_i = exp(-(d_i / (1.5 dx))^2),  k = 8 nearest

scaled by a confidence factor that decays to zero where there are no nearby nodes. That
handles the region-of-interest boundary for free: vertices far from the simulated block --
most of a 445 mm cylinder -- simply do not move, and the transition is smooth rather than a
step. It also respects the discontinuity across a lumen, because the far wall of an open
vessel is many node spacings away and gets negligible weight.

Frames: the FEM works in metres with (X, Y, Z) = (LPS x, LPS z, -LPS y); the OBJs are LPS
millimetres. Both conversions happen here and nowhere else.

🚨 THIS FILE HAD NEVER RUN ON ANYTHING BUT V2 UNTIL 2026-08-13, AND FAILED THREE DIFFERENT WAYS
-------------------------------------------------------------------------------------------
(a) `EXPECTED_RADII` was a hard-coded transcription of `Phantoms/Femoral/V2/README.md` and
    SystemExited on V3 (Muscle 76.547 measured vs 76.480 expected) *and* on V4 (Artery 3.500 vs
    4.500 -- V3/V4 CAD treats CFA_dia/CFV_dia as the tube's OUTER diameter). Two artifacts and
    nothing reconciling them, again. It is now DERIVED from `section_s*.json`, the same
    measured artifact `femoral_fem.load_section` builds the physics from, so both sides of the
    assertion come from one place. ⚠️ And it is only applied where it can honestly apply: the
    section is one plane, and if that plane is outside the imaged slab it certifies nothing --
    the run says so instead of quietly comparing across stations.
(b) `BODY_ORDER` was seven names and omitted Femur, Sheath, FemoralCanal **and PubicRamus**,
    while `render_deformed.py:86` silently `continue`s past a mesh it cannot find -- so a V4
    B-mode would have rendered with no acoustic shadow where the bone is, i.e. WITHOUT the
    backstop the entire ramus-gap sweep exists to test, and the picture would have looked fine.
    The order now comes from the directory's own `anatomy_config.json`, which
    `materials.emit_anatomy_config` writes from `BODY_ORDERS[generation]`.
(c) `slab_and_refine` rebuilt EVERY body as a solid of revolution over `zs = linspace(±half_len)`
    -- discarding the input z entirely. On V4 (LPS z ≈ −1131 … −954 before the axial fix) that
    would have silently TELEPORTED the anatomy ~1042 mm onto the FEM rather than erroring, and
    it extended every body to the full slab regardless of its true extent (the Nerve is 25 mm of
    CAD and would have come back 80 mm long). The slab is now centred on the FEM's OWN node
    cloud and each body is clipped to its own axial extent.
    🪤 AND THE REBUILD ITSELF IS NOT UNIVERSALLY VALID, which is the part (b) drags in behind it.
    A solid of revolution is what V2's seven bodies are; it is NOT what the Sheath (a funnel),
    the PubicRamus (a RING -- two loops in a transverse cut) or the V4 Femur (head + neck + shaft,
    81 mm of x-extent) are. Rebuilding those as cylinders about their own bbox centre would
    produce a confidently wrong anatomy. So the revolution assumption is now CERTIFIED by
    measurement over the whole slab -- several real cross-sections, all circular, all the same
    radii -- and anything that fails certification keeps its real triangles and is simply
    subdivided. A body is never approximated silently.
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
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_section import CIRCLE_TOL_MM, fit_circle  # noqa: E402 -- ONE circle fit, not two

# Radii agreeing with the section within this are "the same body". 0.05 mm was the tolerance
# the old hard-coded assertion used and is kept, so this fix does not also loosen anything.
RADIUS_TOL_MM = 0.05
# Refusal threshold for the real-triangle path. subdivide_to_size is adaptive, but a 445 mm
# cylinder at max_edge 2.5 mm is still ~10^6 triangles and the original docstring records the
# process hanging at ~22 M. Estimate first, refuse loudly, never hang.
MAX_FACES = 1_500_000


def bodies_from_config(obj_dir: str):
    """Author order, straight from the directory's own anatomy_config.json.

    🎯 NOT a module constant. The constant that used to live here was V2's seven bodies and it
    is what would have imaged V4 without its backstop. The config is written by
    `materials.emit_anatomy_config` from `BODY_ORDERS[generation]`, so the deformed anatomy,
    the USD prims and raysim's nesting all come from one list.
    """
    cfg = json.load(open(os.path.join(obj_dir, "anatomy_config.json")))
    order = [m["file"][:-4] for m in cfg["anatomy"]["meshes"] if m["file"] != "Gel.obj"]
    return cfg, order


def load_section(obj_dir: str, explicit: str | None = None):
    """The section JSON in `obj_dir`, or None. Radii come back in MILLIMETRES.

    The file is written in metres by `v3_section.py` (`units: m`) and is the same artifact
    `femoral_fem.load_section` meshes -- which is the entire point of reading it here.

    🪤 AND A DIRECTORY CAN HOLD MORE THAN ONE. `out/femoral_v3` currently has BOTH
    `section_s020.json` and `section_s030.json` -- the press station and the sweep station,
    which is refutation (b)'s open item sitting on disk as two files. Picking `sorted(...)[0]`
    would silently certify this run's radii against whichever station sorts first. Refuse and
    make the caller name one.
    """
    cands = sorted(f for f in os.listdir(obj_dir)
                   if f.startswith("section_s") and f.endswith(".json"))
    if explicit:
        cands = [os.path.basename(explicit)]
    elif len(cands) > 1:
        raise SystemExit(
            f"!!! {obj_dir} holds {len(cands)} sections: {cands}.\n"
            f"    They are different STATIONS along the limb, and the radii this file checks "
            f"against depend on which.\n"
            f"    Name one with --section; an alphabetical guess is how two artifacts stop "
            f"reconciling.")
    if not cands:
        return None
    doc = json.load(open(os.path.join(obj_dir, cands[0])))
    if doc.get("units") != "m":
        raise SystemExit(f"!!! {cands[0]}: expected units 'm', got {doc.get('units')!r}")
    out = {}
    for name, d in doc["bodies"].items():
        if d["kind"] == "disc":
            out[name] = (None, d["r"] * 1e3)
        elif d["kind"] == "annulus":
            out[name] = (d["r_in"] * 1e3, d["r_out"] * 1e3)
        # ellipse / polygon carry no radius pair -- deliberately absent rather than faked
    return dict(doc=doc, radii=out, path=os.path.abspath(os.path.join(obj_dir, cands[0])),
                lps_z_mm=doc["section_lps_z_mm"], station_s_mm=doc["station_s_mm"])


def certify_revolution(m: trimesh.Trimesh, z_lo: float, z_hi: float, n_planes: int,
                       tol: float = CIRCLE_TOL_MM):
    """Is this body a solid of revolution about a fixed axis, OVER THE WHOLE SLAB?

    Returns (cx, cy, r_in|None, r_out) or (None, reason). Measured, not assumed: several real
    cross-sections are taken across the slab and every one must be one or two concentric
    circles with the same centre and the same radii.

    🚨 THE ONE-PLANE VERSION OF THIS TEST IS NOT SAFE and it is worth saying why, because
    `section_s030.json` reports the V4 Femur as `disc r = 11.69` and that is TRUE at s = 30 and
    useless as a certificate: the same body carries the femoral head and neck and spans 81 mm
    in x. A single station is an instrument returning a true value about the wrong thing.
    """
    L = z_hi - z_lo
    if L <= 0:
        return None, "no overlap with the slab"
    inset = max(0.01, 0.01 * L)
    zs = np.linspace(z_lo + inset, z_hi - inset, max(n_planes, 2))
    seen = []
    for z in zs:
        try:
            sec = m.section(plane_origin=[0.0, 0.0, float(z)], plane_normal=[0.0, 0.0, 1.0])
        except Exception as e:                       # noqa: BLE001
            return None, f"section at z = {z:+.2f} raised {type(e).__name__}"
        if sec is None:
            return None, f"no cross-section at z = {z:+.2f} mm (body does not span the slab)"
        loops = [np.asarray(p, np.float64)[:, :2] for p in sec.discrete if len(p) >= 3]
        if len(loops) not in (1, 2):
            return None, f"{len(loops)} loops at z = {z:+.2f} mm -- not a disc or an annulus"
        fits = [fit_circle(P) for P in loops]
        worst = max(f[3] for f in fits)
        if worst > tol:
            return None, (f"cross-section at z = {z:+.2f} mm is not circular "
                          f"(worst radial residual {worst:.3f} mm > {tol:.2f})")
        rr = sorted(f[2] for f in fits)
        seen.append((float(np.mean([f[0] for f in fits])),
                     float(np.mean([f[1] for f in fits])),
                     rr[0] if len(rr) == 2 else None, rr[-1]))
    if len({(s[2] is None) for s in seen}) != 1:
        return None, "solid at some stations and annular at others"
    cx = float(np.mean([s[0] for s in seen]))
    cy = float(np.mean([s[1] for s in seen]))
    # 🪤 THE ACROSS-PLANE TOLERANCE IS THE *TESSELLATION* TOLERANCE, NOT THE SECTION ONE.
    # Both numbers are radii in mm and it is tempting to use one. They measure different things:
    # `tol` (= v3_section.CIRCLE_TOL_MM, 0.10 mm) is the CAD's own chord error, so it is what
    # bounds how much a fitted radius may wander between two planes of the SAME faceted prism;
    # RADIUS_TOL_MM (0.05) is how closely two independent ARTIFACTS must agree. Using 0.05 here
    # refused the V4 skin shell -- radii 92.432..92.499 over 80 mm, pure faceting on a ~110-gon --
    # and dropped a straight extrusion onto the real-triangle path at 931 k triangles. It cost
    # nothing but memory, and that is exactly the kind of tolerance that gets "adjusted" later
    # for the wrong reason. The bodies this test must refuse fail it by 0.7 to 6 mm.
    for k, lab in ((0, "centre x"), (1, "centre y"), (3, "outer radius")):
        v = [s[k] for s in seen]
        if max(v) - min(v) > tol:
            return None, (f"{lab} varies by {max(v)-min(v):.3f} mm across the slab "
                          f"({min(v):.3f} .. {max(v):.3f}) -- the body tapers or bends")
    r_in = None
    if seen[0][2] is not None:
        v = [s[2] for s in seen]
        if max(v) - min(v) > tol:
            return None, f"inner radius varies by {max(v)-min(v):.3f} mm across the slab"
        r_in = float(np.mean(v))
    return (cx, cy, r_in, float(np.mean([s[3] for s in seen]))), "solid of revolution"


def subdivide_real(m: trimesh.Trimesh, name: str, max_edge: float) -> trimesh.Trimesh:
    """Keep the body's REAL triangles and refine them. For everything the revolution test refuses.

    🚨 Refining is not optional: without it the warp silently does NOTHING. These bodies are
    CAD extrusions, so a long cylinder carries vertices ONLY on its two end caps and the side
    wall is one long quad end to end. The FEM region of interest is a 16 mm slab in the middle,
    so not a single vertex falls inside it, every vertex gets negligible weight, and the
    "deformed" mesh comes back identical to the rest pose. The first attempt reported
    `max move 0.000 mm` on all seven bodies while the FEM was displacing nodes by 9.15 mm.

    ⚠️ NO SLAB CLIPPING HERE, ON PURPOSE. `slice_plane(cap=True)` cannot close an annular
    section (the PubicRamus is a ring), and a body clipped by a cap that failed silently is
    worse than a body that is a few thousand triangles larger than it needs to be. The bodies
    that reach this path are the short ones -- Sheath 35 mm, Nerve 25 mm, ramus ~55 mm -- so
    the cost is small; the guard below refuses rather than hangs if that ever stops being true.
    """
    V0 = np.asarray(m.vertices, np.float64)
    F0 = np.asarray(m.faces, np.int64)

    def longest_edge(V, F):
        E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        return float(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).max())

    # 🚨 UNIFORM (CONFORMING) FIRST, ADAPTIVE ONLY AS A DECLARED FALLBACK — and the difference
    # is watertightness, which raysim needs to resolve its material stack.
    # `subdivide_to_size` is adaptive: it splits only the faces that are too long, so a split
    # face and its unsplit neighbour meet at a HANGING VERTEX. The result is geometrically the
    # same solid (volume identical to 4 dp, measured) and topologically OPEN. Merging the
    # vertices afterwards does not fix it -- measured 2026-08-13: Sheath watertight True in the
    # source, False after `subdivide_to_size`, still False after merge_vertices, volume 1.5404
    # cm^3 either way. That is a T-junction, not a crack, and it is exactly the kind of thing
    # that renders fine 99 % of the time and drops a ray through a seam the other 1 %.
    # Uniform `subdivide` splits every face 1 -> 4, so neighbours stay conforming, but it costs
    # 4^rounds and rounds is set by the LONGEST edge -- the V4 femur's 120 mm shaft quads need 6
    # rounds, i.e. 12.7 M triangles. So: take uniform when it fits under the guard, say so
    # loudly when it does not, and never hang (the original code did, at ~22 M).
    e0 = longest_edge(V0, F0)
    rounds = 0
    while e0 / (2 ** rounds) > max_edge and rounds < 12:
        rounds += 1
    est_uniform = len(F0) * (4 ** rounds)
    if est_uniform <= MAX_FACES:
        V, F = V0, F0
        for _ in range(rounds):
            V, F = trimesh.remesh.subdivide(V, F)
        out = trimesh.Trimesh(vertices=V, faces=F, process=True)
        print(f"      REAL TRIANGLES kept (not a solid of revolution), {rounds} UNIFORM rounds, "
              f"{len(F0)} -> {len(out.faces)} tris, edge {e0:.1f} -> "
              f"{longest_edge(np.asarray(out.vertices), np.asarray(out.faces)):.2f} mm, "
              f"watertight={out.is_watertight}")
        return out

    V, F = trimesh.remesh.subdivide_to_size(V0, F0, max_edge=max_edge)
    out = trimesh.Trimesh(vertices=V, faces=F, process=True)
    print(f"      REAL TRIANGLES kept, but 🚨 ADAPTIVELY (uniform would be {est_uniform:,} tris, "
          f"over the {MAX_FACES:,} guard):\n"
          f"      {len(F0)} -> {len(out.faces)} tris with T-JUNCTIONS, watertight="
          f"{out.is_watertight}. raysim has never been run on a non-conforming\n"
          f"      body here, so this one carries an untested caveat. Fixes, in order of "
          f"preference: clip it to the slab, or raise --max-edge for it.")
    return out


def revolve(cx: float, cy: float, r_in, r_out: float, z_lo: float, z_hi: float,
            max_edge: float) -> trimesh.Trimesh:
    """Regenerate a certified solid of revolution over [z_lo, z_hi], finely tessellated.

    trimesh's `slice_plane(cap=True)` fails on annular sections -- its cap triangulation cannot
    close a ring -- so a certified body is REGENERATED rather than cut. Every number here was
    measured by `certify_revolution` off the CAD, so the 0.02 mm de-coincide shrinks that
    stl_to_raysim.py / build_v3_anatomy.py apply to nested bodies survive.
    """
    nz = max(int(round((z_hi - z_lo) / max_edge)) + 1, 3)
    nt = max(int(round(2 * np.pi * r_out / max_edge)), 24)
    zs = np.linspace(z_lo, z_hi, nz)
    th = np.linspace(0, 2 * np.pi, nt, endpoint=False)

    def ring(rad):
        return np.column_stack([cx + rad * np.cos(th), cy + rad * np.sin(th)])

    verts, faces = [], []

    def wall(rad, outward):
        base = len(verts)
        for z in zs:
            xy = ring(rad)
            verts.extend([(x, y, z) for x, y in xy])
        for k in range(nz - 1):
            for i in range(nt):
                j = (i + 1) % nt
                a, b = base + k * nt + i, base + k * nt + j
                c, d = base + (k + 1) * nt + i, base + (k + 1) * nt + j
                faces.extend([(a, c, d), (a, d, b)] if outward else [(a, d, c), (a, b, d)])
        return base

    o = wall(r_out, True)
    if r_in is not None:
        n = wall(r_in, False)                      # inner wall faces inward
        for k, sgn in ((0, -1), (nz - 1, 1)):      # annular end caps
            for i in range(nt):
                j = (i + 1) % nt
                a, b = o + k * nt + i, o + k * nt + j
                c, d = n + k * nt + i, n + k * nt + j
                faces.extend([(a, b, d), (a, d, c)] if sgn > 0 else [(a, d, b), (a, c, d)])
    else:
        for k, sgn in ((0, -1), (nz - 1, 1)):      # disc caps via a centre fan
            ctr = len(verts)
            verts.append((cx, cy, zs[k]))
            for i in range(nt):
                j = (i + 1) % nt
                a, b = o + k * nt + i, o + k * nt + j
                faces.append((ctr, a, b) if sgn > 0 else (ctr, b, a))

    out = trimesh.Trimesh(vertices=np.asarray(verts, np.float64),
                          faces=np.asarray(faces, np.int64), process=False)
    trimesh.repair.fix_normals(out)
    return out


def check_against_section(name, r_in, r_out, section, z_lo, z_hi):
    """Reconcile a regenerated body against the section JSON the FEM was built from.

    🎯 THIS IS THE ACTUAL FIX FOR THE OLD `EXPECTED_RADII`. Both sides of the assertion now come
    from one measured artifact -- the OBJ this run just sectioned, and the section JSON that
    `femoral_fem.load_section` meshes -- instead of one side coming from a README transcribed by
    hand in the V2 era. That transcription SystemExited on V3 (Muscle 76.547 vs 76.480) and on V4
    (Artery 3.500 vs 4.500, because V3/V4 CAD reads CFA_dia/CFV_dia as OUTER diameter), which is
    how this whole file came to have never run on anything but V2.

    ⚠️ AND IT ONLY APPLIES WHERE IT CAN. The section is ONE plane. If that plane lies outside the
    imaged slab, comparing radii across it is comparing two stations, so the check is declined and
    said out loud rather than silently widened. That gap is real and is not this file's to close:
    the press station is s = 30 while LPS z = 0 -- where the FEM slab sits -- is s = 20.
    """
    if section is None:
        return "no section JSON in the OBJ dir -- radii UNCHECKED"
    zs = section["lps_z_mm"]
    if not (z_lo <= zs <= z_hi):
        return (f"section plane s = {section['station_s_mm']:.0f} mm (LPS z = {zs:+.1f}) is "
                f"OUTSIDE the imaged slab [{z_lo:+.1f}, {z_hi:+.1f}] -- radii UNCHECKED")
    exp = section["radii"].get(name)
    if exp is None:
        return f"not a disc/annulus in the section (or absent at s = {section['station_s_mm']:.0f})"
    for got, want, lab in ((r_in, exp[0], "inner"), (r_out, exp[1], "outer")):
        if (got is None) != (want is None):
            raise SystemExit(
                f"!!! {name}: {lab} radius solid/annulus mismatch -- the mesh says "
                f"{got!r} mm, {section['path']} says {want!r} mm.\n"
                f"    The renderer and the FEM would be imaging different bodies.")
        if got is not None and abs(got - want) > RADIUS_TOL_MM:
            raise SystemExit(
                f"!!! {name}: {lab} radius {got:.3f} mm differs from {section['path']}'s "
                f"{want:.3f} mm by {abs(got-want):.3f} mm (> {RADIUS_TOL_MM:.2f}).\n"
                f"    The B-mode would show a different vessel from the one the force curve "
                f"was measured on.")
    return f"matches {os.path.basename(section['path'])} within {RADIUS_TOL_MM:.2f} mm"


def build_slab_bodies(obj_dir, order, section, z_centre, half_len, max_edge, n_planes):
    """The rest-pose imaging meshes, built ONCE and reused for every FEM state.

    Returns {name: trimesh} plus a printed table. Bodies that do not reach the slab are
    reported and dropped -- never silently, because `render_deformed.py:86` `continue`s past a
    mesh it cannot find and a missing backstop looks exactly like a phantom that never had one.
    """
    z_lo_req, z_hi_req = z_centre - half_len, z_centre + half_len
    built, dropped, open_bodies = {}, [], []
    print(f"\n--- rest-pose imaging meshes, slab LPS z [{z_lo_req:+.1f}, {z_hi_req:+.1f}] mm, "
          f"max edge {max_edge:.2f} mm ---")
    for name in order:
        p = os.path.join(obj_dir, f"{name}.obj")
        if not os.path.exists(p):
            raise SystemExit(f"!!! {p} is named by anatomy_config.json and is not on disk")
        # 🚨 `process=True` — MERGE THE VERTICES, AND IT IS NOT COSMETIC HERE.
        # These OBJs are written per-face (`f v//vn`, or trimesh's own exporter), so a
        # `process=False` load gives every triangle its own three vertices: SoftTissue comes in
        # as 1332 vertices instead of 222 and reports `is_watertight=False` on a solid whose
        # volume is 99.98 % of ideal. gel_body.py:184 already documents that as a WITNESS bug --
        # the check measuring the file format rather than the solid. On the real-triangle path
        # below it is worse than a bad witness: `subdivide_to_size` splits each triangle
        # independently, so an unmerged mesh cracks along every shared edge and the OBJ handed
        # to raysim is genuinely open. raysim needs closed bodies to resolve its material stack.
        m = trimesh.load(p, process=True)
        zb0, zb1 = float(m.bounds[0][2]), float(m.bounds[1][2])
        # CLIP TO THE BODY'S OWN AXIAL EXTENT. The old code ignored input z entirely and
        # regenerated every body over the full +/-half_len, so the 25 mm Nerve came back 80 mm
        # long and the V4 anatomy would have been teleported 1042 mm onto the FEM.
        z_lo, z_hi = max(z_lo_req, zb0), min(z_hi_req, zb1)
        if z_hi - z_lo <= 0:
            dropped.append(f"{name}: spans LPS z {zb0:+.1f}..{zb1:+.1f}, no overlap with the slab")
            print(f"  {name:<13} ❌ NOT IN THE IMAGED SLAB (z {zb0:+.1f}..{zb1:+.1f})")
            continue
        # ⚠️ SOURCE WATERTIGHTNESS IS A WITNESS AND IT CAUGHT SOMETHING. Measured 2026-08-13 on
        # the merged geometry: `PhantomV4 - Femur-1.STL` is NOT closed -- 4 broken faces, straight
        # out of SolidWorks -- while every other V4 part is. An open body has no defined interior,
        # so raysim's material stack for it is undefined. It is reported here rather than raised
        # because it is a CAD defect upstream of this file, and blocking the imaging chain on it
        # would hide it behind a stack trace instead of putting it in the run log.
        if not m.is_watertight:
            open_bodies.append(name)
            print(f"  {name:<13} 🚨 SOURCE MESH IS NOT WATERTIGHT ({len(m.faces)} faces, merged "
                  f"to {len(m.vertices)} verts) -- raysim's material stack is undefined inside it")
        cert, why = certify_revolution(m, z_lo, z_hi, n_planes)
        print(f"  {name:<13} z {z_lo:+7.1f}..{z_hi:+7.1f}  {why}")
        if cert is None:
            built[name] = subdivide_real(m, name, max_edge)
        else:
            cx, cy, r_in, r_out = cert
            built[name] = revolve(cx, cy, r_in, r_out, z_lo, z_hi, max_edge)
            kind = (f"annulus r {r_in:.2f}..{r_out:.2f}" if r_in is not None
                    else f"rod r {r_out:.2f}")
            note = check_against_section(name, r_in, r_out, section, z_lo, z_hi)
            print(f"      regenerated: {kind} mm about ({cx:+.2f}, {cy:+.2f}), "
                  f"{len(m.faces)} -> {len(built[name].faces)} tris, "
                  f"watertight={built[name].is_watertight}")
            print(f"      vs section: {note}")
    if dropped:
        print("\n  ⚠️ dropped from the imaged anatomy (reported, not silent):")
        for d in dropped:
            print(f"     {d}")
    if open_bodies:
        print(f"\n  🚨 NOT WATERTIGHT IN THE SOURCE CAD: {open_bodies}. Every B-mode from this "
              f"run carries that caveat.")
    return built, dropped, open_bodies


def lps_mm_to_fem_m(V: np.ndarray) -> np.ndarray:
    """(x, y, z)_LPS_mm -> (x, z, -y)_FEM_m"""
    return np.column_stack([V[:, 0], V[:, 2], -V[:, 1]]) * 1e-3


def fem_m_to_lps_mm(U: np.ndarray) -> np.ndarray:
    """Inverse of the above, applied to a DISPLACEMENT vector."""
    return np.column_stack([U[:, 0], -U[:, 2], U[:, 1]]) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "fem/fem_states.npz"))
    ap.add_argument("--obj-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "femoral_v2"))
    ap.add_argument("--out-root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "deformed"))
    ap.add_argument("--dx", type=float, default=0.0015)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--blend-sigma-mm", type=float, default=None,
                    help="RENDER-ONLY. Width of the Gaussian falloff used to carry the FEM "
                         "displacement into the part of the limb the FEM never solved. "
                         "Default 1.5*dx = 2.25 mm, which fades from full amplitude to zero "
                         "in about two millimetres and DRAWS A HARD STEP in the skin -- "
                         "visible as the stepped shoulder in the 2026-08-12 frames. That step "
                         "is not anatomy and not a solved result; it is where the ROI ends. "
                         "⚠️ THIS CHANGES NO MEASURED NUMBER. Closure, force and mesh quality "
                         "all come from the FEM nodes; this only decides how the render "
                         "extrapolates beyond them. Recorded in _provenance either way.")
    ap.add_argument("--max-edge", type=float, default=2.5, help="subdivide until no edge exceeds this, mm")
    ap.add_argument("--half-len", type=float, default=40.0, help="keep +/- this many mm about the imaging plane")
    # 🔴 MODELLING CHOICE, and it flags rather than gates. A deformed anatomy whose peak node
    # displacement grows FASTER than the probe travel that caused it is reporting more motion
    # than was imposed -- physically possible under lateral flow, but far more often the sign
    # of a state that has come apart. Flagged so a B-mode built on it carries the warning.
    # (Added 2026-08-13: referenced at line 539 but never declared, so every run crashed here.)
    ap.add_argument("--disp-slope-flag", type=float, default=3.0,
                    help="flag a state whose d|u|max/d(travel) exceeds this. 🔴 modelling "
                         "choice; it annotates the report, it does not drop the state.")
    ap.add_argument("--section", default=None,
                    help="which section_s*.json in --obj-dir to reconcile the rebuilt radii "
                         "against. Required when the directory holds more than one -- they are "
                         "different stations and an alphabetical guess is a silent choice.")
    ap.add_argument("--revolution-planes", type=int, default=5,
                    help="how many real cross-sections certify a body as a solid of "
                         "revolution before it is regenerated analytically. One plane is not "
                         "enough -- see certify_revolution's docstring and the V4 femur.")
    args = ap.parse_args()

    z = np.load(args.states)
    rest = z["rest"]
    tree = cKDTree(rest)
    states = sorted([f for f in z.files if f.startswith("s")],
                    key=lambda s: float(s[1:]))
    print(f"FEM nodes: {len(rest):,}   states: {[s[1:] for s in states]}")

    base_cfg, order = bodies_from_config(args.obj_dir)
    section = load_section(args.obj_dir, args.section)
    gen = base_cfg.get("_provenance", {}).get("cad_generation", "?")
    print(f">>> anatomy {args.obj_dir}  (generation {gen}, {len(order)} bodies from "
          f"anatomy_config.json)")
    print(f"    {order}")

    # 🚨 THE SLAB IS CENTRED ON THE FEM'S OWN NODES, NOT ON A CONSTANT. FEM Y is LPS z, so the
    # rest cloud says exactly where in the limb the solve happened -- and it is the artifact
    # that produced the displacement field being applied. The old code centred on 0 by
    # construction (`zs = linspace(-half_len, half_len)`), which is only right when the OBJs
    # already sit at z ≈ 0; on the pre-2026-08-13 V4 export they sat at z ≈ -1042 mm and the
    # anatomy would have been teleported onto the FEM without a word.
    fem_z_lo, fem_z_hi = float(rest[:, 1].min()) * 1e3, float(rest[:, 1].max()) * 1e3
    z_centre = 0.5 * (fem_z_lo + fem_z_hi)
    print(f">>> FEM region of interest spans LPS z {fem_z_lo:+.2f} .. {fem_z_hi:+.2f} mm "
          f"(centre {z_centre:+.2f}); imaging slab is that centre +/- {args.half_len:.0f} mm")

    # ⚠️ AND THE ONE RECONCILIATION THIS FILE CANNOT MAKE, PRINTED RATHER THAN BURIED.
    # The FEM extrudes ONE cross-section, measured at `station_s_mm`, uniformly along Y. The
    # CAD bodies are NOT uniform along s -- the nerve ends at 25, the canal at 12.5, the sheath
    # tapers -- so the warped anatomy is the CAD at the slab's station displaced by a field
    # solved from the section's station. When those two differ, say so in the caption.
    # 🪤 THE CRITERION IS "IS THE SECTION PLANE INSIDE THE SOLVED SLAB", NOT A THRESHOLD IN mm.
    # The FEM grid rounds LY to a whole number of cells, so its centre is offset from the
    # nominal plane by up to dx/2 -- 0.25 mm on the current ladders. A bare `!= station` fires
    # on that and buries the V4 case, where the section is 10.25 mm away and genuinely outside
    # the block that was solved. Ask the question the data can answer: does the plane the
    # cross-section was measured at lie within the node cloud?
    lig = base_cfg.get("geometry", {}).get("ligament_lps_z_mm")
    if section is not None and lig is not None:
        s_slab = lig - z_centre
        inside = fem_z_lo <= section["lps_z_mm"] <= fem_z_hi
        print(f"    section station s = {section['station_s_mm']:.1f} mm (LPS z = "
              f"{section['lps_z_mm']:+.2f}); FEM slab centre s = {s_slab:.1f} mm -- section "
              f"plane is {'INSIDE' if inside else 'OUTSIDE'} the solved block")
        if not inside:
            print(f"    🚨 STATION MISMATCH. The deformed bodies are the s = {s_slab:.1f} mm CAD "
                  f"displaced by a field solved from the s = {section['station_s_mm']:.1f} mm "
                  f"cross-section,\n"
                  f"       and the two stations do not even overlap. In V3/V4 the anatomy is NOT "
                  f"uniform along s -- the nerve ends at 25, the canal at 12.5, the sheath "
                  f"tapers -- so this\n"
                  f"       is a real discrepancy, not a rounding one. Every figure from this run "
                  f"must name BOTH stations. Reconciling them costs one FEM ladder per specimen\n"
                  f"       (refutation (b) of the 2026-08-13 harvest); it is a cost, not an "
                  f"argument for editing the acceptance criterion.")

    slab, dropped, open_bodies = build_slab_bodies(
        args.obj_dir, order, section, z_centre, args.half_len, args.max_edge,
        args.revolution_planes)
    if not slab:
        raise SystemExit("!!! no body reaches the imaging slab -- nothing to deform")

    # ── THE DISPLACEMENT WITNESS, COMPUTED OVER THE WHOLE LADDER BEFORE ANYTHING IS WARPED ──
    # 🚨 WHY THIS EXISTS. On `fem_states_deep_v4_gap20.npz` the max node displacement goes
    # 11.10 mm at 11.0 mm past skin -> 268.31 mm at 12.6 mm. That is the inversion cascade the
    # 2026-08-13 harvest documents (Newton's VBD integrates stable Neo-Hookean, which has no
    # inversion barrier by construction, and `solver_vbd.py` has no line search). Without a
    # witness this file carries those node positions onto the CAD surfaces and raysim renders a
    # confident-looking B-mode of a blown solve.
    # 📌 TWO NUMBERS, BECAUSE THE OBVIOUS ONE ALONE CRIES WOLF. |u|max / travel sits at
    # 1.01-1.12 on every HEALTHY rung -- a node at the punch edge is dragged slightly further
    # than the indentation by lateral flow at nu = 0.45, which is real. So the ratio is reported
    # and never flagged on. The flag is on d|u|max / d(travel) between consecutive rungs: the
    # peak displacement cannot grow faster than the thing pushing it, beyond that same bulge.
    # Measured on gap20: 1.00-1.02 across every good rung, 160.8 across the cascade. The
    # threshold is 🔴 a modelling choice, but the data separates by a factor of ~100, so
    # ANY value in [1.5, 100] gives the identical verdict -- it is not load-bearing, and
    # --disp-slope-flag is there to prove that rather than to tune it.
    u_max = {st: float(np.abs(z[st] - rest).max() * 1e3) for st in states}
    trav = {st: float(st[1:]) for st in states}
    slope, suspect = {}, {}
    prev = None
    for st in states:
        if prev is None or abs(trav[st] - trav[prev]) < 1e-9:
            slope[st] = float("nan")
        else:
            slope[st] = (u_max[st] - u_max[prev]) / (trav[st] - trav[prev])
        suspect[st] = bool(np.isfinite(slope[st]) and slope[st] > args.disp_slope_flag)
        prev = st
    print(f"\n--- displacement witness (travel is mm PAST SKIN; flag at "
          f"d|u|/d(travel) > {args.disp_slope_flag:g} 🔴) ---")
    print(f"{'travel':>8} {'|u|max mm':>11} {'|u|max/travel':>15} {'d|u|/d(travel)':>16}   verdict")
    for st in states:
        s = f"{slope[st]:16.2f}" if np.isfinite(slope[st]) else " " * 16
        print(f"{trav[st]:8.1f} {u_max[st]:11.2f} "
              f"{u_max[st]/max(abs(trav[st]),1e-9):15.2f} {s}   "
              f"{'🚨 SUSPECT -- treat as a blown solve' if suspect[st] else 'ok'}")
    if any(suspect.values()):
        n = sum(suspect.values())
        print(f"  🚨 {n} of {len(states)} states flagged. They are WRITTEN ANYWAY and labelled, "
              f"not dropped:\n"
              f"     the real exclusion criterion is `settled AND agree < 0.25 AND "
              f"n_inverted == 0 AND minJ > 0.1`,\n"
              f"     which needs `Phantom.mesh_quality()` (femoral_fem.py -- does not exist "
              f"yet). Dropping rungs on a\n"
              f"     ratio invented here would be the same sin in the other direction. Every "
              f"state carries its verdict in\n"
              f"     its own anatomy_config.json `_provenance.deformed_from`.")

    # ⚠️ RENDER-ONLY EXTRAPOLATION WIDTH -- see --blend-sigma-mm.
    # Outside the FEM ROI the displacement is UNKNOWN, so the render has to choose something.
    # The old fixed 1.5*dx chose a 2.25 mm falloff, which is not a neutral choice: it puts a
    # sharp step into the skin surface that a reader interprets as tissue geometry. Widening
    # it removes an artefact rather than hiding a result -- the FEM's own numbers are
    # untouched, because they are read from nodes and never from this mesh.
    sigma = (args.blend_sigma_mm * 1e-3) if args.blend_sigma_mm else (1.5 * args.dx)
    print(f">>> render blend sigma {sigma*1e3:.2f} mm "
          f"({'--blend-sigma-mm' if args.blend_sigma_mm else 'default 1.5*dx'})"
          f"   ⚠️ render-only; changes no measured number")
    summary = []

    for st in states:
        pos = z[st]
        u_nodes = pos - rest
        travel = st[1:]
        out = os.path.join(args.out_root, f"travel_{travel}mm")
        os.makedirs(out, exist_ok=True)
        print(f"\n--- travel {travel} mm   max node displacement {u_max[st]:.2f} mm "
              f"({u_max[st]/max(abs(trav[st]),1e-9):.2f}x travel)"
              f"{'   🚨 SUSPECT, see the witness table above' if suspect[st] else ''} ---")

        rows = []
        for name, m in slab.items():
            V = np.asarray(m.vertices, np.float64)
            P = lps_mm_to_fem_m(V)

            d, idx = tree.query(P, k=args.k)
            w = np.exp(-(d / sigma) ** 2)
            wsum = w.sum(axis=1)
            # confidence: a vertex needs a few genuinely close nodes before it is allowed
            # to move at full amplitude. Far from the ROI this goes to 0 smoothly.
            conf = np.clip(wsum / 4.0, 0.0, 1.0)
            safe = np.where(wsum > 1e-12, wsum, 1.0)
            u_fem = (w[:, :, None] * u_nodes[idx]).sum(axis=1) / safe[:, None]
            u_fem *= conf[:, None]

            V_new = V + fem_m_to_lps_mm(u_fem)
            mm = trimesh.Trimesh(vertices=V_new, faces=np.asarray(m.faces), process=False)
            write_obj(os.path.join(out, f"{name}.obj"), V_new, np.asarray(m.faces),
                      f"# {name}, deformed by FEM at {travel} mm probe travel\n"
                      f"# SPACE=LPS, units mm\n")
            moved = np.linalg.norm(fem_m_to_lps_mm(u_fem), axis=1)
            rows.append(dict(body=name, watertight=bool(mm.is_watertight),
                             max_disp_mm=float(moved.max()),
                             n_moved=int((moved > 0.01).sum()), n_verts=int(len(V))))
            print(f"  {name:<12} max move {moved.max():6.3f} mm  "
                  f"({(moved > 0.01).sum():>4}/{len(V)} verts)  "
                  f"watertight={mm.is_watertight}")

        cfg = json.loads(json.dumps(base_cfg))
        cfg["anatomy"]["mesh_dir"] = out
        # 🪤 THE MESH LIST HAS TO FOLLOW THE BODIES THAT WERE ACTUALLY WRITTEN. Copying the base
        # config verbatim would name bodies this slab dropped, and `render_deformed.py:86`
        # silently `continue`s past a mesh it cannot find -- so a dropped backstop would leave
        # no trace anywhere in the imaging chain. Keep Gel.obj's entry: gel_body.py runs on
        # each travel dir afterwards and re-appends it, but the ordering must survive if it does not.
        cfg["anatomy"]["meshes"] = [x for x in base_cfg["anatomy"]["meshes"]
                                    if x["file"][:-4] in slab or x["file"] == "Gel.obj"]
        cfg["_provenance"]["deformed"] = f"FEM state at {travel} mm probe travel"
        cfg["_provenance"]["deformed_from"] = dict(
            states=os.path.abspath(args.states), obj_dir=os.path.abspath(args.obj_dir),
            slab_centre_lps_z_mm=z_centre, slab_half_len_mm=args.half_len,
            blend_sigma_mm=float(sigma * 1e3),
            blend_sigma_is_default=bool(args.blend_sigma_mm is None),
            fem_roi_lps_z_mm=[fem_z_lo, fem_z_hi],
            section=None if section is None else section["path"],
            section_station_s_mm=None if section is None else section["station_s_mm"],
            slab_station_s_mm=None if lig is None else lig - z_centre,
            bodies_dropped=dropped,
            bodies_not_watertight_in_source=open_bodies,
            max_node_disp_mm=u_max[st],
            max_node_disp_over_travel=u_max[st] / max(abs(trav[st]), 1e-9),
            d_max_disp_d_travel=None if not np.isfinite(slope[st]) else slope[st],
            disp_slope_flag=args.disp_slope_flag,
            displacement_witness=(
                "ok" if not suspect[st] else
                "🚨 SUSPECT: peak node displacement grows faster than the probe that drives it "
                "(d|u|/d(travel) = %.1f). Treat as a blown solve -- do not publish a B-mode from "
                "this state -- until Phantom.mesh_quality() reports n_inverted == 0 and "
                "minJ > 0.1 for it." % slope[st]))
        json.dump(cfg, open(os.path.join(out, "anatomy_config.json"), "w"), indent=2)
        summary.append(dict(travel_mm=float(travel), dir=out, bodies=rows,
                            max_node_disp_mm=u_max[st],
                            d_max_disp_d_travel=None if not np.isfinite(slope[st]) else slope[st],
                            suspect=suspect[st]))

    json.dump(summary, open(os.path.join(args.out_root, "deform_report.json"), "w"), indent=2)
    print(f"\nwrote {len(states)} deformed anatomies under {args.out_root}")
    bad = [s for s in summary if s["suspect"]]
    if bad:
        print(f"🚨 {len(bad)} of {len(summary)} states are flagged SUSPECT and are NOT "
              f"publishable without Phantom.mesh_quality():")
        for s in bad:
            print(f"     travel {s['travel_mm']:6.1f} mm past skin -> max node disp "
                  f"{s['max_node_disp_mm']:8.2f} mm, d|u|/d(travel) = "
                  f"{s['d_max_disp_d_travel']:.1f}")


def write_obj(path, V, F, header):
    v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    N = np.divide(N, np.where(ln == 0, 1.0, ln))
    with open(path, "w") as fh:
        fh.write(header)
        np.savetxt(fh, V, fmt="v %.5f %.5f %.5f")
        np.savetxt(fh, N, fmt="vn %.5f %.5f %.5f")
        idx = F + 1
        np.savetxt(fh, np.column_stack([idx, idx]).reshape(-1, 6)[:, [0, 3, 1, 4, 2, 5]],
                   fmt="f %d//%d %d//%d %d//%d")


if __name__ == "__main__":
    main()
