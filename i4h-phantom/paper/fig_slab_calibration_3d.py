"""Slab calibration geometry in 3-D: the same content as fig_slab_calibration.png, seen at a modest oblique angle.

    $PY_ANATOMY scripts/fig_slab_calibration_3d.py

(a) the leg mesh, cut open at y = 0 (20 mm below the inguinal ligament) with the tissues on the cut face, and the
    slab's box drawn where it sits. Only the 20 mm between the cut and the inguinal ligament is drawn; the 100 mm
    below the cut would stand between the camera and the cut face.
(b) the slab mesh, outer surface coloured by tissue, with the probe face and the visible nodes held fixed.

Same meshes, digests and assertions as fig_slab_calibration.py: the two panels are a claim that they are one
subject, so artery, vein and nerve centroids must agree within 1 mm and both must read nerve < artery < vein in x.
Writes figures/fig_slab_calibration_3d.png only. Plain matplotlib.
"""
import hashlib
import pathlib
import sys
import ast

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection           # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
FIG = ROOT / "figures"
PH = pathlib.Path(__file__).resolve().parent.parent / "out"
import sys  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_meshes as PM                                         # noqa: E402  2026-10-21: the paper's meshes, one place
MESHES = {"slab": (PM.SLAB_15, PM.SLAB_15_SHA), "leg": (PM.LEG, PM.LEG_SHA)}

# colours and tissue names: identical to fig_slab_calibration.py
COL = {"SoftTissue": "#f1e2bc", "Muscle": "#d68f7e", "Femur": "#ebe8da", "PubicRamus": "#d3cdb9", "Sheath": "#6f9bbd",
       "Perivascular": "#cfe0a6", "Artery": "#c8382d", "Vein": "#3a5fb0", "Nerve": "#e9b93a", "void": "#b5b5b5"}
ROLE = {"SoftTissue": "skin and subcutaneous fat", "Muscle": "muscle", "Femur": "femur (held fixed)",
        "PubicRamus": "superior pubic ramus (held fixed)", "Sheath": "femoral sheath wall",
        "Perivascular": "sheath contents (areolar tissue)", "Artery": "common femoral artery wall",
        "Vein": "common femoral vein wall", "Nerve": "femoral nerve",
        "void": "boundary cells with blended tissue properties"}
ORDER = list(COL)
EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
ELEV, AZIM = 22, -55         # one camera for both panels: from the distal side, as the 2-D section is viewed (inner thigh to the right),
                             # raised above the front of the thigh and turned toward the inner thigh
LIGHT = np.array([0.35, 0.8, 0.6]); LIGHT = LIGHT / np.linalg.norm(LIGHT)


def load(name):
    path, want = MESHES[name]
    got = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    if got != want:
        sys.exit(f"!!! {path} has digest {got}, the record names {want}: refusing")
    d = np.load(path, allow_pickle=True)
    return dict(X=d["rest"] * 1e3, T=d["tets"], body=d["tet_body"].astype(str), pinned=d["pinned"],
                digest=got, path=path, d=d)


def slice_y(m, y):
    """Exact intersection of every tetrahedron with the plane y = const: polygons in (x, z), and their tet index."""
    P = m["X"][m["T"]]; dd = P[:, :, 1] - (y + 1e-3)
    idx = np.nonzero((dd.min(1) < 0) & (dd.max(1) > 0))[0]; polys = []
    for ti in idx:
        p, dv, pts = P[ti], dd[ti], []
        for a, b in EDGES:
            if dv[a] * dv[b] < 0:
                t = dv[a] / (dv[a] - dv[b]); pts.append((p[a] + t * (p[b] - p[a]))[[0, 2]])
        pts = np.array(pts); c = pts.mean(0)
        polys.append(pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))])
    return idx, polys


def centroid_xz(m, body, ylim):
    c = m["X"][m["T"]].mean(1); k = (m["body"] == body) & (np.abs(c[:, 1]) < ylim)
    return c[k][:, [0, 2]].mean(0)


def boundary(m):
    """Boundary faces (vertex indices, outward winding) and the tetrahedron that owns each."""
    T = m["T"]
    F4 = T[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3); own = np.repeat(np.arange(len(T)), 4)
    _, inv, cnt = np.unique(np.sort(F4, axis=1), axis=0, return_inverse=True, return_counts=True)
    bnd = cnt[inv.ravel()] == 1
    return F4[bnd], own[bnd]


def clip_y(tri, keep_below, y0=0.0):
    """Clip one polygon to the half-space y <= y0 (keep_below) or y >= y0. Sutherland-Hodgman, one plane."""
    s = (tri[:, 1] - y0) * (1 if keep_below else -1)          # <= 0 is kept
    out = []
    for i in range(len(tri)):
        a, b, sa, sb = tri[i], tri[(i + 1) % len(tri)], s[i], s[(i + 1) % len(tri)]
        if sa <= 0: out.append(a)
        if sa * sb < 0: out.append(a + sa / (sa - sb) * (b - a))
    return np.array(out) if len(out) >= 3 else None


def shade(rgb, normal):
    lam = 0.55 + 0.45 * abs(float(normal @ LIGHT))
    return tuple(np.clip(np.array(matplotlib.colors.to_rgb(rgb)) * lam, 0, 1))


def box_edges(lo, hi):
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return [(c[i], c[j]) for i in range(8) for j in range(i + 1, 8) if np.sum(c[i] != c[j]) == 1]


def draw_box(ax, lo, hi, **kw):
    for a, b in box_edges(lo, hi):
        ax.plot(*np.array([a, b]).T, **kw)


def probe_faces(xp, zs, pw, pl, thick=4.0):
    lo, hi = np.array([xp - pw / 2, -pl / 2, zs + 2.0]), np.array([xp + pw / 2, pl / 2, zs + 2.0 + thick])
    c = lambda i, j, k: np.array([(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]])
    quads = []
    for ax_ in range(3):
        for side in (0, 1):
            pts = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                idx = [None] * 3; idx[ax_] = side; o = [a for a in range(3) if a != ax_]; idx[o[0]] = u; idx[o[1]] = v
                pts.append(c(*idx))
            quads.append(np.array(pts))
    return quads


def main():
    slab, leg = load("slab"), load("leg")
    smeta = ast.literal_eval(str(slab["d"]["meta"][0]))
    bb_lo, bb_hi = slab["X"].min(0), slab["X"].max(0)
    half_y = 0.5 * (bb_hi[1] - bb_lo[1])

    # ── the relation the two panels claim: one frame, one anatomy (as in fig_slab_calibration.py) ─────────
    rows = []
    for b in ("Nerve", "Artery", "Vein"):
        cs, cl = centroid_xz(slab, b, half_y), centroid_xz(leg, b, half_y)
        dist = float(np.hypot(*(cs - cl)))
        rows.append((b, cs.round(2), cl.round(2), round(dist, 2)))
        assert dist < 1.0, f"{b} centroid differs by {dist:.2f} mm between slab and leg"
    for m, name in ((slab, "slab"), (leg, "leg")):
        xs = [centroid_xz(m, b, half_y)[0] for b in ("Nerve", "Artery", "Vein")]
        assert xs[0] < xs[1] < xs[2], f"{name}: nerve < artery < vein in x fails ({xs}): mirrored anatomy?"
        print(f"{name}: nerve, artery, vein x = {np.round(xs, 2)} mm (ordered)")
    for r in rows:
        print(f"{r[0]:6s} centroid (x, z) slab {r[1]}  leg {r[2]}  distance {r[3]} mm")

    zs, pw = float(slab["d"]["z_skin"][0]) * 1e3, float(slab["d"]["probe_w"][0]) * 1e3
    pl, xp = float(slab["d"]["probe_len"][0]) * 1e3, float(slab["d"]["x_probe"][0]) * 1e3
    zl = float(leg["X"][:, 2].max())

    fig = plt.figure(figsize=(12, 7.2))
    axA = fig.add_axes([0.00, 0.30, 0.52, 0.65], projection="3d")
    axB = fig.add_axes([0.49, 0.30, 0.48, 0.65], projection="3d")

    # ── (a) the leg: the part between the cut (y = 0) and the inguinal ligament, cut face coloured, slab box ─────────
    BF, BO = boundary(leg); X = leg["X"]
    polys, fcs = [], []
    for f in BF:
        tri = X[f]; n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); n = n / (np.linalg.norm(n) + 1e-12)
        part = clip_y(tri, False)                                 # keep y >= 0
        if part is not None and abs(n[1]) < 0.99:                 # the far end face is hidden from this camera
            polys.append(part); fcs.append(shade("#efe7d6", n) + (1.0,))
    idx, sl = slice_y(leg, 0.0)
    for p, b in zip(sl, leg["body"][idx]):
        polys.append(np.column_stack([p[:, 0], np.zeros(len(p)), p[:, 1]])); fcs.append(matplotlib.colors.to_rgba(COL[b]))
    axA.computed_zorder = False
    axA.add_collection3d(Poly3DCollection(polys, facecolors=fcs, edgecolors=(0, 0, 0, 0.35), linewidths=0.08, zorder=2))
    # slab box: the part standing out of the cut face solid, the part inside the leg dotted
    for a, b in box_edges(bb_lo, bb_hi):
        seg = np.array([a, b])
        if seg[0, 1] != seg[1, 1]:                                # an edge along the limb crosses the cut face
            lo_, hi_ = seg[np.argsort(seg[:, 1])]; mid = lo_.copy(); mid[1] = 0.0
            axA.plot(*np.array([lo_, mid]).T, color="k", lw=1.5, zorder=5)
            axA.plot(*np.array([mid, hi_]).T, color="k", lw=0.9, ls=":", zorder=5)
        elif seg[0, 1] < 0:
            axA.plot(*seg.T, color="k", lw=1.5, zorder=5)
        else:
            axA.plot(*seg.T, color="k", lw=0.9, ls=":", zorder=5)
    draw_box(axA, np.array([bb_lo[0], 0.0, bb_lo[2]]), np.array([bb_hi[0], 0.0, bb_hi[2]]), color="k", lw=1.2, ls="--", zorder=5)
    axA.add_collection3d(Poly3DCollection(probe_faces(xp, zl, pw, pl), facecolors="#444", edgecolors="k", linewidths=0.4, zorder=6))
    axA.text(bb_lo[0], bb_lo[1], bb_lo[2] - 14, "slab", fontsize=9, ha="center", zorder=7)
    R = float(np.abs(X[:, [0, 2]]).max())
    axA.set_xlim(-R, R); axA.set_ylim(-10, 20); axA.set_zlim(-R, R + 8)
    axA.set_box_aspect((2 * R, 30, 2 * R + 8), zoom=1.05)
    axA.set_yticks([0, 20])
    axA.set_title("(a) Leg mesh, cut 20 mm below the inguinal ligament,\nwith the slab's box where it sits", fontsize=10)

    # ── (b) the slab: outer surface coloured by tissue, probe face, visible nodes held fixed ────────────────
    BF, BO = boundary(slab); X = slab["X"]
    tris = X[BF]; nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
    fc = [shade(COL[b], n) for b, n in zip(slab["body"][BO], nrm)]
    axB.computed_zorder = False
    axB.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors=(0, 0, 0, 0.35), linewidths=0.08, zorder=2))
    axB.add_collection3d(Poly3DCollection(probe_faces(xp, zs, pw, pl), facecolors="#444", edgecolors="k", linewidths=0.4, zorder=6))
    axB.text(xp, -pl / 2, zs + 13, f"probe face, {pw:g} mm wide", ha="center", fontsize=8.5, zorder=7)
    P = X[slab["pinned"]]
    vis = (np.abs(P[:, 0] - bb_hi[0]) < 1e-6) | (np.abs(P[:, 1] - bb_lo[1]) < 1e-6)   # on the two side faces the camera sees
    axB.scatter(P[vis, 0], P[vis, 1], P[vis, 2], s=0.5, c="k", depthshade=False, zorder=5)
    axB.set_xlim(bb_lo[0], bb_hi[0]); axB.set_ylim(bb_lo[1], bb_hi[1]); axB.set_zlim(bb_lo[2], zs + 16)
    axB.set_box_aspect((bb_hi[0] - bb_lo[0], bb_hi[1] - bb_lo[1], zs + 16 - bb_lo[2]), zoom=1.0)
    axB.set_yticks([-8, 0, 8])
    axB.set_title("(b) Slab mesh, coloured by tissue, with the probe face\nand the nodes held fixed", fontsize=10)

    for ax in (axA, axB):
        ax.view_init(elev=ELEV, azim=AZIM)
        ax.set_xlabel("across the thigh (mm)", fontsize=8, labelpad=4)
        ax.set_ylabel("along the limb (mm)", fontsize=8, labelpad=4)
        ax.set_zlabel("front to back (mm)", fontsize=8, labelpad=4)
        ax.tick_params(labelsize=7, pad=1)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False; ax.grid(False)

    hs = [Patch(fc=COL[b], ec="k", lw=0.4, label=ROLE[b]) for b in ORDER if (slab["body"] == b).any() or (leg["body"] == b).any()]
    hs.append(Patch(fc="#444", ec="k", lw=0.4, label="probe face, 2 mm above the skin"))
    hs.append(Line2D([], [], color="k", marker=".", ls="none", ms=4, label=f"nodes held fixed ({len(slab['pinned']):,} in the slab; visible faces)"))
    hs.append(Line2D([], [], color="k", lw=1.5, label="the slab's box, drawn in the leg (dotted inside the leg)"))
    fig.legend(handles=hs, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.075))
    fig.text(0.5, 0.008,
             "(a) The leg mesh, cut where the slab sits; the tissues show on the cut face. Only the 20 mm from the cut up to the inguinal ligament is drawn.\n"
             "(b) The slab, at the same angle. The slab is a rectangular cut of the leg's cross-section on a regular grid, so its skin is stepped.\n"
             f"Slab: {len(slab['T']):,} tetrahedra; leg: {len(leg['T']):,} tetrahedra. "
             f"Artery, vein and nerve sit at the same positions in both, to within {max(r[3] for r in rows):.2f} mm.",
             ha="center", va="bottom", fontsize=7.5, color="#333")
    out = FIG / "fig_slab_calibration_3d.png"
    fig.savefig(out, dpi=250)
    # panel (a) alone, for the paper's slab-and-envelope figure (panel (b) there is fig_leg_sweep_envelope.png)
    fig.canvas.draw(); bb = axA.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(FIG / "fig_slab_calibration_3d_panel_a.png", dpi=300, bbox_inches=bb.padded(0.05)); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
