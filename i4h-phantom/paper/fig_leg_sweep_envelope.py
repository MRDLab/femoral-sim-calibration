"""Leg mesh cross-section with the probe's sweep envelope: +-45 degrees about the front of the thigh. 2026-10-01.

    $PY_ANATOMY scripts/fig_leg_sweep_envelope.py

Panel (b) of the slab-and-envelope figure (fig:slab_calibration). Panel (a) is the leg-with-slab-box view, written by
fig_slab_calibration_3d.py as fig_slab_calibration_3d_panel_a.png. Same tissue colours as that figure.
Everything drawn comes from the mesh file (Mesh F, c4e13e7d7574) and its own metadata:
  cut face      exact intersection of every tetrahedron with the plane 20 mm below the inguinal ligament (y = 0)
  fine window   meta arc_deg (45), r_in_m (inner radius of the fine region) and r_skin_m: the 1.5 mm region
  probe         probe_w (40 mm face), drawn 2 mm off the skin (the run's standoff) at -45, 0 and +45 degrees
The sweep is a rotation about the limb axis. Only the drawn probe sweeps; the press is at 0 degrees (fig_probe_motion.py).
"""
import hashlib
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.collections import PolyCollection                 # noqa: E402
from matplotlib.patches import Patch, Polygon, FancyArrowPatch     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paper_meshes as PM                                         # noqa: E402  2026-10-21: the paper's meshes, one place
MESH = PM.verified(PM.LEG, PM.LEG_SHA, "leg mesh")
COL = {"SoftTissue": "#f1e2bc", "Muscle": "#d68f7e", "Femur": "#ebe8da", "PubicRamus": "#d3cdb9", "Sheath": "#6f9bbd",
       "Perivascular": "#cfe0a6", "Artery": "#c8382d", "Vein": "#3a5fb0", "Nerve": "#e9b93a"}
ROLE = {"SoftTissue": "skin and subcutaneous fat", "Muscle": "muscle", "Femur": "femur (held fixed)",
        "PubicRamus": "superior pubic ramus (held fixed)", "Sheath": "femoral sheath wall",
        "Perivascular": "sheath contents (areolar tissue)", "Artery": "common femoral artery wall",
        "Vein": "common femoral vein wall", "Nerve": "femoral nerve"}
EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])

d = np.load(MESH, allow_pickle=True)
meta = json.loads(str(d["meta"][0]))
X, T, body = d["rest"] * 1e3, d["tets"], d["tet_body"].astype(str)
ARC, R_IN, R_SKIN = meta["arc_deg"], meta["r_in_m"] * 1e3, meta["r_skin_m"] * 1e3
PW, STANDOFF = float(d["probe_w"][0]) * 1e3, 2.0
assert abs(meta["fine_depth_m"] * 1e3 - (R_SKIN - R_IN)) < 0.01

# exact cut of every tetrahedron by the plane y = 0
P = X[T]; dd = P[:, :, 1] - 1e-3
idx = np.nonzero((dd.min(1) < 0) & (dd.max(1) > 0))[0]; polys = []
for ti in idx:
    p, dv, pts = P[ti], dd[ti], []
    for a, b in EDGES:
        if dv[a] * dv[b] < 0:
            t = dv[a] / (dv[a] - dv[b]); pts.append((p[a] + t * (p[b] - p[a]))[[0, 2]])
    pts = np.array(pts); c = pts.mean(0)
    polys.append(pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))])

def rot(xz, deg):   # rotate about the limb axis; 0 degrees = straight up, the front of the thigh; + toward the inner thigh (+x)
    a = np.radians(deg); R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    return xz @ R.T

def draw(ax, rasterize=False, title="(b) Leg mesh, cut 20 mm below the inguinal ligament,\nwith the probe's sweep and press"):
    """Draw the cut, fine window, probe, sweep and press on ax; returns the tissue legend handles."""
    ax.add_collection(PolyCollection(polys, facecolors=[COL[b] for b in body[idx]], edgecolors=(0, 0, 0, 0.35), linewidths=0.15,
                                      rasterized=rasterize))
    # fine window: the annular sector r_in..r_skin, +-ARC
    th = np.radians(np.linspace(-ARC, ARC, 100))
    sector = np.vstack([np.column_stack([R_SKIN * np.sin(th), R_SKIN * np.cos(th)]),
                        np.column_stack([R_IN * np.sin(th[::-1]), R_IN * np.cos(th[::-1])])])
    ax.add_patch(Polygon(sector, closed=True, fill=False, ec="k", lw=1.3, ls="--", zorder=4))
    for s in (-ARC, ARC):   # rays from the axis to the window edges
        ax.plot(*np.array([[0, 0], rot(np.array([[0, R_SKIN + 30]]), s)[0]]).T, color="0.35", lw=0.8, ls=":", zorder=3)
    ax.plot([0, 0], [0, R_SKIN + 30], color="0.35", lw=0.8, ls=":", zorder=3)
    # probe faces 2 mm off the skin
    face = np.array([[-PW / 2, R_SKIN + STANDOFF], [PW / 2, R_SKIN + STANDOFF], [PW / 2, R_SKIN + STANDOFF + 5],
                     [-PW / 2, R_SKIN + STANDOFF + 5]])
    for s, a in ((-ARC, 0.35), (ARC, 0.35), (0, 1.0)):
        ax.add_patch(Polygon(rot(face, s), closed=True, fc="#444", ec="k", lw=0.6, alpha=a, zorder=5))
    # sweep arc and press arrow
    r_arc = R_SKIN + 16
    ta = np.radians(np.linspace(-ARC, ARC, 60))
    ax.plot(r_arc * np.sin(ta), r_arc * np.cos(ta), color="#1f77b4", lw=1.6, zorder=6)
    for s in (-ARC, ARC):
        q = rot(np.array([[0, r_arc]]), s)[0]; q2 = rot(np.array([[0, r_arc]]), s * 0.93)[0]
        ax.add_patch(FancyArrowPatch(q2, q, arrowstyle="-|>", mutation_scale=12, color="#1f77b4", zorder=6))
    # 2026-10-25, print size: notes shortened and left at the rc size (9 pt in the paper's figures, print_size.py)
    ax.text(0, r_arc + 4, "sweep $\\pm45^\\circ$", ha="center", va="bottom", color="#1f77b4")
    ax.add_patch(FancyArrowPatch((PW / 2 + 6, R_SKIN + 12), (PW / 2 + 6, R_SKIN - 14), arrowstyle="-|>", mutation_scale=12,
                                 color="#d62728", lw=1.6, zorder=6))
    ax.text(118, 136, "press\n25.4 mm", color="#d62728", ha="right", va="top")
    ax.annotate("1.5 mm\nelements", xy=rot(np.array([[0, R_IN + 4]]), -30)[0],
                xytext=(-118, 22), arrowprops=dict(arrowstyle="-", lw=0.6), ha="left")
    ax.text(4, 8, "limb axis", color="0.3")
    ax.plot(0, 0, "k+", ms=6)
    ax.set_xlim(-122, 122); ax.set_ylim(-100, 140); ax.set_aspect("equal")
    ax.set_xlabel("across the thigh (mm), + medial")
    ax.set_ylabel("front to back (mm), + anterior")
    ax.set_title(title)
    hs = [Patch(fc=COL[b], ec="k", lw=0.4, label=ROLE[b]) for b in COL if (body == b).any()]
    return hs


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    ax.legend(handles=draw(ax), fontsize=6.8, loc="lower left", ncol=2, framealpha=0.9)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "fig_leg_sweep_envelope.png", dpi=300); plt.close(fig)
    print(f"fine window r {R_IN:.1f}..{R_SKIN:.1f} mm, +-{ARC:g} deg; {len(polys)} cut polygons")
