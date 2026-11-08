"""Derive the Femoral V5 OBJ set from the V4 set every study used, with exactly two changes.

    $PY_ANATOMY anatomy_meshing/v5_from_v4.py

Directed 2026-10-21, after the gel leg was cast from the V4 CAD with two simplifications:

  1. the pubic ramus is removed (too complex to mould);
  2. the limb is cut from the INFERIOR end until skin and muscle are 81 mm long (s = 0 .. 81 mm).
     The inferior end is cut because the cross-section is uniform there.

Nothing else changes: same bodies, same positions, same materials. So V5 is built FROM
`out/femoral_v4_gap12` (the anatomy behind Mesh F and the slab), not by re-running the CAD
converter. Every body that ends above s = 81 mm is copied byte for byte; the ones that run past
it are clipped at that plane and capped.

Checks that can fail (each refuses to write):
  * the source is the V4 set the paper used: ramus gap 12 mm, ligament at LPS z = +20 mm
  * every copied body is byte-identical to its V4 file
  * every clipped body gains no open edge from the cut (V4's Femur keeps its 4), ends at s = 81 mm, and is unchanged above the cut
    (vertex-for-vertex: every V4 vertex above the plane is present in the clipped body)
  * the prismatic bodies lose volume in proportion to length: Muscle and SoftTissue keep 81/120
  * no PubicRamus anywhere in the output
  * the bundle still reads nerve < artery < vein in LPS x (a right thigh)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import repo_paths  # noqa: F401,E402
import argparse
import hashlib
import json
import pathlib
import shutil
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import materials as MAT  # noqa: E402

OUT_ROOT = (pathlib.Path(repo_paths.ROOT) / "out").resolve()   # the real out/, not a worktree symlink
LEG_LEN_MM = 81.0           # skin and muscle, superior to inferior, as cast
REMOVED = ("PubicRamus",)
CLIPPED = ("SoftTissue", "Muscle", "Femur", "Gel")
SECTION_STATIONS = (6.0, 20.0)   # the stations V4 carries current sections for (its s = 30 file predates the mirror fix)
SECTION_TOL_MM = 0.02            # fresh V5 section vs V4's, per body: the fit effect measured 0.013 mm at most


def sha12(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]


def write_obj(path, m, name):
    """Same format as v4_to_section.py: vertices, vertex normals, faces (raysim needs the normals)."""
    with open(path, "w") as fh:
        fh.write(f"# {name}, Femoral V5 (V4 clipped at s = {LEG_LEN_MM:g} mm). SPACE=LPS, units mm.\n")
        np.savetxt(fh, m.vertices, fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, m.vertex_normals, fmt="vn %.6f %.6f %.6f")
        for t in m.faces + 1:
            fh.write("f %d//%d %d//%d %d//%d\n" % (t[0], t[0], t[1], t[1], t[2], t[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(OUT_ROOT / "femoral_v4_gap12"))
    ap.add_argument("--out", default=str(OUT_ROOT / "femoral_v5"))
    args = ap.parse_args()
    SRC, OUT = pathlib.Path(args.src), pathlib.Path(args.out)

    conv = json.load(open(SRC / "v4_conversion.json"))
    cfg4 = json.load(open(SRC / "anatomy_config.json"))
    if conv["ramus_gap_mm"] != 12.0 or abs(conv["ligament_lps_z_mm"] - 20.0) > 1e-9:
        raise SystemExit(f"!!! {SRC} is not the V4 set the paper used (gap {conv['ramus_gap_mm']}, "
                         f"ligament z {conv['ligament_lps_z_mm']})")
    if OUT.exists():
        raise SystemExit(f"!!! {OUT} exists. Archive it (rename) before rebuilding; never overwrite a set runs may cite.")
    OUT.mkdir(parents=True)

    z_lig = float(conv["ligament_lps_z_mm"])
    z_cut = z_lig - LEG_LEN_MM                 # LPS z decreases distally; keep z >= z_cut
    objs = sorted(p.stem for p in SRC.glob("*.obj"))
    print(f">>> source {SRC}: {objs}")
    print(f">>> ligament (s = 0) at LPS z = {z_lig:+.3f}; cut at s = {LEG_LEN_MM:g} mm -> LPS z = {z_cut:+.3f}")

    record = dict(source=str(SRC), source_conversion=conv, leg_length_mm=LEG_LEN_MM, cut_lps_z_mm=z_cut,
                  removed=list(REMOVED), clipped={}, copied={}, date="2026-10-21",
                  changes=["pubic ramus removed", f"inferior end cut so skin and muscle span s = 0..{LEG_LEN_MM:g} mm"])
    bad = []
    for name in objs:
        src = SRC / f"{name}.obj"
        if name in REMOVED:
            print(f"    {name:13s} REMOVED")
            continue
        m4 = trimesh.load(src, process=False)
        m4.merge_vertices(merge_tex=True, merge_norm=True)   # the OBJs carry per-vertex normals; merge on position only
        s_lo, s_hi = z_lig - m4.bounds[1][2], z_lig - m4.bounds[0][2]
        if s_hi <= LEG_LEN_MM + 1e-9:
            if name in CLIPPED:
                bad.append(f"{name} was expected to cross the cut and does not (s {s_lo:.1f}..{s_hi:.1f})")
            shutil.copy2(src, OUT / src.name)
            same = sha12(src) == sha12(OUT / src.name)
            record["copied"][name] = dict(sha256_12=sha12(src), s_span_mm=[s_lo, s_hi])
            print(f"    {name:13s} copied, s {s_lo:6.1f}..{s_hi:6.1f}, byte-identical {same}")
            if not same:
                bad.append(f"{name} copy differs")
            continue
        if name not in CLIPPED:
            bad.append(f"{name} crosses the cut (s up to {s_hi:.1f}) but is not in the expected clip list {CLIPPED}")
            continue
        # V4's Femur keeps the 4 open edges of its CAD export (at s ~ 14.5 mm, the hole print-ready filled). It is
        # NOT repaired here: V5 changes nothing but the two directed edits. The test is that the cut adds no open edge.
        open4 = len(trimesh.grouping.group_rows(m4.edges_sorted, require_count=1))
        zlev = np.unique(np.round(m4.vertices[:, 2], 6))
        if len(zlev) == 2:
            # A pure extrusion (skin shell, muscle core, gel band): vertices only at the two end planes. Moving the
            # inferior end ring to the cut is exact and keeps V4's topology; a planar slice of an annulus is not
            # (its cap left 9 and 24 open edges on the shell and the gel, 2026-10-21).
            m5 = m4.copy()
            V = m5.vertices.copy(); V[np.isclose(V[:, 2], zlev[0]), 2] = z_cut; m5.vertices = V
            how = "end ring moved"
        else:
            m5 = m4.slice_plane(plane_origin=[0.0, 0.0, z_cut], plane_normal=[0.0, 0.0, 1.0], cap=True)
            trimesh.repair.fix_normals(m5)
            how = "sliced and capped"
        s5 = (z_lig - m5.bounds[1][2], z_lig - m5.bounds[0][2])
        # unchanged above the cut: every V4 vertex strictly above the plane must survive exactly
        above = m4.vertices[m4.vertices[:, 2] > z_cut + 1e-6]
        dist, _ = cKDTree(m5.vertices).query(above)
        open5 = len(trimesh.grouping.group_rows(m5.edges_sorted, require_count=1))
        ok_wt = open5 == open4 and m5.volume > 0
        ok_end = abs(s5[1] - LEG_LEN_MM) < 1e-6
        ok_keep = float(dist.max()) < 1e-6
        frac = m5.volume / m4.volume
        info = dict(method=how, v4_sha256_12=sha12(src), v4_volume_mm3=float(m4.volume), v5_volume_mm3=float(m5.volume),
                    volume_ratio=frac, s_span_mm=list(map(float, s5)), watertight=bool(m5.is_watertight), open_edges_v4=open4, open_edges_v5=open5,
                    max_dev_of_kept_v4_vertices_mm=float(dist.max()), v4_vertices_above_cut=int(len(above)))
        if name in ("Muscle", "SoftTissue", "Gel"):
            # prismatic over s = 0..120: volume goes exactly with length
            expect = LEG_LEN_MM / (s_hi - s_lo)
            info["expected_ratio_prismatic"] = expect
            if abs(frac - expect) > 1e-3:
                bad.append(f"{name}: volume ratio {frac:.5f} vs prismatic {expect:.5f} (cross-section not uniform below the cut?)")
        record["clipped"][name] = info
        print(f"    {name:13s} CLIPPED s {s_lo:6.1f}..{s_hi:6.1f} -> {s5[0]:6.1f}..{s5[1]:6.1f}; volume x{frac:.4f}; "
              f"watertight {ok_wt}; kept V4 vertices max dev {dist.max():.1e} mm")
        for ok, what in ((ok_wt, f"open edges {open4} -> {open5} (the cut must add none)"), (ok_end, f"ends at s {s5[1]:.3f}"), (ok_keep, "moved a V4 vertex above the cut")):
            if not ok:
                bad.append(f"{name}: {what}")
        write_obj(OUT / f"{name}.obj", m5, name)

    # relationship checks on what was WRITTEN
    have = sorted(p.stem for p in OUT.glob("*.obj"))
    if any(r in have for r in REMOVED):
        bad.append("a removed body is in the output")
    for name in have:
        b = trimesh.load(OUT / f"{name}.obj", process=False).bounds
        if z_lig - b[0][2] > LEG_LEN_MM + 1e-6:
            bad.append(f"{name} reaches s = {z_lig - b[0][2]:.2f} mm, past the cut")
    cx = {n: float(trimesh.load(OUT / f"{n}.obj", process=False).bounds.mean(axis=0)[0]) for n in ("Nerve", "Artery", "Vein")}
    order_ok = cx["Nerve"] < cx["Artery"] < cx["Vein"]
    print(f">>> laterality (LPS x): nerve {cx['Nerve']:+.2f} | artery {cx['Artery']:+.2f} | vein {cx['Vein']:+.2f} "
          f"-> {'RIGHT thigh' if order_ok else 'NOT a right thigh'}")
    if not order_ok:
        bad.append("bundle order is not nerve < artery < vein")
    if bad:
        raise SystemExit("!!! REFUSING: " + "; ".join(bad) + f"\n    (partial output left in {OUT}; archive it before retrying)")

    # anatomy_config: V4's geometry block, with V5's provenance. Materials are V4's, unchanged.
    geo = dict(cfg4["geometry"])
    prov = dict(cfg4.get("_provenance", {}))
    for k in ("ramus_gap_mm", "ramus_gap_status", "ramus_scale_s", "ramus_scale_status"):
        prov.pop(k, None)
    prov.update(source=str(SRC), derived_by="anatomy_meshing/v5_from_v4.py",
                v5_changes=record["changes"], leg_length_mm=LEG_LEN_MM, cut_lps_z_mm=z_cut)
    prov.pop("cad_generation", None); prov.pop("gel", None)
    MAT.emit_anatomy_config(str(OUT), generation="v5", background_material=cfg4.get("anatomy", {}).get("background_material", "air"),
                            geometry=geo, provenance=prov)
    # carry V4's gel provenance (the gel band is V4's, clipped), so render_deformed.py reads the same standoff and thickness
    cfg5 = json.load(open(OUT / "anatomy_config.json"))
    if "gel" in cfg4.get("_provenance", {}):
        cfg5["_provenance"]["gel"] = dict(cfg4["_provenance"]["gel"], clipped_at_s_mm=LEG_LEN_MM)
    for m in cfg4["anatomy"]["meshes"]:
        if m["file"] == "Gel.obj":
            cfg5["anatomy"]["meshes"].append(dict(m))
    for k, v in cfg4["anatomy"].items():
        if k not in cfg5["anatomy"]:
            cfg5["anatomy"][k] = v
    # the config must be V4's with only the ramus gone: same bodies, same order, same materials
    want = [m for m in cfg4["anatomy"]["meshes"] if m["file"] not in {f"{r}.obj" for r in REMOVED}]
    if cfg5["anatomy"]["meshes"] != want:
        raise SystemExit(f"!!! V5 config mesh list differs from V4's minus the ramus:\n    {cfg5['anatomy']['meshes']}\n    {want}")
    if cfg5["geometry"] != cfg4["geometry"]:
        raise SystemExit("!!! V5 geometry block differs from V4's")
    json.dump(cfg5, open(OUT / "anatomy_config.json", "w"), indent=2)

    # ── the sections: V4's, with only the ramus entry removed ──────────────────────────────────────────────
    # A section carries no axial length, so of the two V5 changes only the ramus reaches it. Re-sectioning the
    # clipped OBJs is NOT used: every side wall above the cut is the same plane as in V4, but the walls are split
    # into triangles with diagonals, and a plane crosses a diagonal at a point that depends on the body's length.
    # The section polygon traces the same chords with its sample points slid along them, and v3_section's
    # point-weighted circle fit then moves skin, muscle and femur by up to 13 um (measured 2026-10-21). That is a
    # fitting effect, not anatomy, so V4's own section is kept and the fresh V5 section is only an independent check.
    import subprocess
    sec_py = pathlib.Path(repo_paths.ROOT) / "v3_section.py"
    py = sys.executable
    record["sections"] = {}
    for st in SECTION_STATIONS:
        tag = f"section_s{int(st):03d}.json"
        sec4 = json.load(open(SRC / tag))
        # 1) the V4 file must still be what today's code makes from the V4 OBJs (s = 30 was not, 2026-10-21: stale 08-13)
        chk4, chk5 = OUT / f".check_v4_{tag}", OUT / f".check_v5_{tag}"
        subprocess.run([py, str(sec_py), "--mesh-dir", str(SRC), "--station-s", f"{st:g}", "--out", str(chk4)],
                       check=True, capture_output=True)
        if json.load(open(chk4))["bodies"] != sec4["bodies"]:
            raise SystemExit(f"!!! {SRC / tag} is not reproduced by today's v3_section.py -- stale, refusing to carry it into V5")
        # 2) V5 = V4 minus the ramus
        sec5 = dict(sec4, mesh_dir=str(OUT))
        sec5["bodies"] = {k: v for k, v in sec4["bodies"].items() if k not in REMOVED}
        sec5["nest_order"] = [k for k in sec4["nest_order"] if k not in REMOVED]
        sec5["absent"] = list(sec4["absent"]) + [f"{r} (removed in V5: not in the cast phantom)" for r in REMOVED]
        sec5["derived_from"] = dict(file=str(SRC / tag), sha256_12=sha12(SRC / tag), edit=f"removed {list(REMOVED)}",
                                    reason="a section has no axial length; only the ramus removal reaches it")
        # 3) independent check: a fresh section of the V5 OBJs agrees body by body
        subprocess.run([py, str(sec_py), "--mesh-dir", str(OUT), "--station-s", f"{st:g}", "--out", str(chk5)],
                       check=True, capture_output=True)
        fresh = json.load(open(chk5))["bodies"]
        if set(fresh) != set(sec5["bodies"]):
            raise SystemExit(f"!!! s = {st:g}: fresh V5 section has bodies {sorted(fresh)}, expected {sorted(sec5['bodies'])}")
        dev = {}
        for b, d in sec5["bodies"].items():
            worst = 0.0
            for k in ("c", "r", "r_in", "r_out", "a", "b"):
                if k in d:
                    worst = max(worst, float(np.max(np.abs(np.asarray(d[k], float) - np.asarray(fresh[b][k], float)))) * 1e3)
            if "loop_areas_mm2" in d:
                la, lb = np.asarray(d["loop_areas_mm2"], float), np.asarray(fresh[b]["loop_areas_mm2"], float)
                if la.shape != lb.shape or np.max(np.abs(la - lb) / np.abs(la)) > 1e-3:
                    raise SystemExit(f"!!! s = {st:g} {b}: loop areas {la} vs fresh {lb}")
            dev[b] = worst
        if max(dev.values()) > SECTION_TOL_MM:
            raise SystemExit(f"!!! s = {st:g}: fresh V5 section deviates {max(dev.values()):.4f} mm > {SECTION_TOL_MM} mm: {dev}")
        json.dump(sec5, open(OUT / tag, "w"), indent=1)
        chk4.unlink(); chk5.unlink()
        record["sections"][tag] = dict(v4_sha256_12=sha12(SRC / tag), bodies=sorted(sec5["bodies"]),
                                       fresh_v5_section_max_dev_mm=dev)
        print(f"    {tag}: V4 minus {list(REMOVED)}; fresh V5 section agrees, worst {max(dev.values())*1e3:.1f} um "
              f"({max(dev, key=dev.get)})")
    json.dump(record, open(OUT / "v5_derivation.json", "w"), indent=1, default=float)
    print(f">>> wrote {len(have)} OBJs, anatomy_config.json, {len(SECTION_STATIONS)} sections and v5_derivation.json to {OUT}")


if __name__ == "__main__":
    main()
