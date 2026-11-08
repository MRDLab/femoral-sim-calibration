"""The paper's mesh figure (fig:slab_calibration), one panel, one column. 2026-10-01; redrawn 2026-11-04 for ISMR 2027.

    $PY_ANATOMY scripts/fig_slab_and_sweep.py

The leg mesh cut at s = 20 mm (20 mm below the inguinal ligament), seen from the feet: medial to the right, anterior up.
On that one cut: the slab's rectangular footprint (solid black), the 1.5 mm region (dashed), the probe at -45, 0 and
+45 degrees with the sweep arc and the press arrow, and an inset zoomed on the femoral sheath (artery, vein, nerve,
sheath wall and contents) with a connector box. Mesh edges are drawn at 0.3 pt inside the 1.5 mm region and at 0.2 pt
elsewhere. The 3-D panel of the 2026-10 version was dropped by the author's change order of 2026-11-04.

Geometry: the cut polygons, fine-region radii, arc and probe width come from fig_leg_sweep_envelope.py (the leg mesh,
digest checked); the slab footprint is the slab mesh's bounding box in x and z (digest checked). Both scripts' own
outputs are unchanged. Same subject checks as fig_slab_calibration_3d.py: vessel and nerve centroids agree within 1 mm
between slab and leg, and both read nerve < artery < vein in x (a right thigh).
Mesh polygons are rasterised at 300 dpi inside a vector PDF (text and lines stay vector) so the file stays small.
Writes figures/fig_slab_and_sweep.pdf (the paper) and .png (for viewing).
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl                                          # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.collections import PolyCollection                 # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Polygon, Rectangle   # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fig_slab_calibration_3d as m3                               # noqa: E402  meshes, digests, centroid checks
import fig_leg_sweep_envelope as env                               # noqa: E402  the cut polygons and the fine region
import print_size as PS                                           # noqa: E402  printed width and 9 pt minimum, asserted on save

FIG = pathlib.Path(__file__).resolve().parent.parent.parent / "figures"
slab, leg = m3.load("slab"), m3.load("leg")
bb_lo, bb_hi = slab["X"].min(0), slab["X"].max(0)
half_y = 0.5 * (bb_hi[1] - bb_lo[1])
worst = 0.0
for b in ("Nerve", "Artery", "Vein"):
    dist = float(np.hypot(*(m3.centroid_xz(slab, b, half_y) - m3.centroid_xz(leg, b, half_y))))
    assert dist < 1.0, f"{b} centroid differs by {dist:.2f} mm between slab and leg"
    worst = max(worst, dist)
for m, name in ((slab, "slab"), (leg, "leg")):
    xs = [m3.centroid_xz(m, b, half_y)[0] for b in ("Nerve", "Artery", "Vein")]
    assert xs[0] < xs[1] < xs[2], f"{name}: nerve < artery < vein in x fails ({xs}): mirrored anatomy?"

# ── style: the paper's serif block, on top of print_size's 9 pt minimum ───────────────────────────────────────
mpl.rcParams.update(PS.RC)
mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "legend.frameon": False, "axes.grid": False, "lines.linewidth": 1.0, "pdf.fonttype": 42})
COL = dict(env.COL); COL["Artery"], COL["Vein"] = "#c0392b", "#1f5fbf"   # the paper's vessel colours (fig_leg_vein_vs_artery.py)
C_SWEEP, C_PRESS = "#1f77b4", "#c0392b"

# ── the cut, from fig_leg_sweep_envelope: polygons, bodies, fine region, probe ────────────────────────────────
polys, bodies = env.polys, env.body[env.idx]
ARC, R_IN, R_SKIN, PW, STANDOFF = env.ARC, env.R_IN, env.R_SKIN, env.PW, env.STANDOFF
cen = np.array([p.mean(0) for p in polys])
r_c, th_c = np.hypot(cen[:, 0], cen[:, 1]), np.degrees(np.arctan2(cen[:, 0], cen[:, 1]))
fine = (r_c >= R_IN - 1e-6) & (r_c <= R_SKIN + 1e-6) & (np.abs(th_c) <= ARC)
assert 0.2 < fine.mean() < 0.95, f"fine-region share of cut polygons {fine.mean():.2f}: the sector test is wrong"
sx0, sx1, sz0, sz1 = float(bb_lo[0]), float(bb_hi[0]), float(bb_lo[2]), float(bb_hi[2])
DX_SLAB = float(slab["d"]["dx"][0]) * 1e3            # the slab's skin is stepped on its grid, so its top may sit under R_SKIN by < dx
assert sx0 < -PW / 2 < PW / 2 < sx1 and sz0 < R_IN < R_SKIN <= sz1 + DX_SLAB, \
    f"slab footprint x {sx0}..{sx1}, z {sz0}..{sz1} does not enclose the probe ({PW} mm) and the fine region ({R_IN}..{R_SKIN})"


def tissue(ax, rasterize=True):
    """Fill the cut by tissue; edges 0.3 pt inside the 1.5 mm region, 0.2 pt elsewhere."""
    for k, lw, a in ((~fine, 0.2, 0.3), (fine, 0.3, 0.45)):
        ax.add_collection(PolyCollection([polys[i] for i in np.nonzero(k)[0]], facecolors=[COL[b] for b in bodies[k]],
                                         edgecolors=(0, 0, 0, a), linewidths=lw, rasterized=rasterize, zorder=1))


def fine_outline(ax, lw=1.1):
    th = np.radians(np.linspace(-ARC, ARC, 100))
    sector = np.vstack([np.column_stack([R_SKIN * np.sin(th), R_SKIN * np.cos(th)]),
                        np.column_stack([R_IN * np.sin(th[::-1]), R_IN * np.cos(th[::-1])])])
    ax.add_patch(Polygon(sector, closed=True, fill=False, ec="k", lw=lw, ls="--", zorder=4))


# ── the figure: one column, one panel ─────────────────────────────────────────────────────────────────────────
XL, ZL = (-100, 100), (-97, 123)
fig = plt.figure(figsize=(PS.COL_IN, 5.0))
ax_w = 0.80 * PS.COL_IN; ax_h = ax_w * (ZL[1] - ZL[0]) / (XL[1] - XL[0])
ax = fig.add_axes([0.17, 1 - (0.06 + ax_h / 5.0), 0.80, ax_h / 5.0])
tissue(ax)
ax.add_patch(Rectangle((sx0, sz0), sx1 - sx0, sz1 - sz0, fill=False, ec="k", lw=1.2, zorder=5))   # the slab's footprint
fine_outline(ax)
for s in (-ARC, 0, ARC):   # rays from the limb axis to the sweep limits and the press
    ax.plot(*np.array([[0, 0], env.rot(np.array([[0, R_SKIN + 24]]), s)[0]]).T, color="0.35", lw=0.7, ls=":", zorder=3)
ax.plot(0, 0, "k+", ms=6, mew=0.9, zorder=6)
face = np.array([[-PW / 2, R_SKIN + STANDOFF], [PW / 2, R_SKIN + STANDOFF], [PW / 2, R_SKIN + STANDOFF + 5],
                 [-PW / 2, R_SKIN + STANDOFF + 5]])
for s, a in ((-ARC, 0.35), (ARC, 0.35), (0, 1.0)):
    ax.add_patch(Polygon(env.rot(face, s), closed=True, fc="#444", ec="k", lw=0.6, alpha=a, zorder=5))
r_arc = R_SKIN + 15
ta = np.radians(np.linspace(-ARC, ARC, 60))
ax.plot(r_arc * np.sin(ta), r_arc * np.cos(ta), color=C_SWEEP, lw=1.4, zorder=6)
for s in (-ARC, ARC):
    q = env.rot(np.array([[0, r_arc]]), s)[0]; q2 = env.rot(np.array([[0, r_arc]]), s * 0.93)[0]
    ax.add_patch(FancyArrowPatch(q2, q, arrowstyle="-|>", mutation_scale=11, color=C_SWEEP, zorder=6))
ax.text(0, r_arc + 4, "sweep ±45°", ha="center", va="bottom", color=C_SWEEP)
xa = PW / 2 + 7
ax.add_patch(FancyArrowPatch((xa, r_arc - 3), (xa, R_SKIN - 14), arrowstyle="-|>", mutation_scale=11, color=C_PRESS, lw=1.4, zorder=6))
ax.text(xa + 4, r_arc + 1, "press", color=C_PRESS, ha="left", va="bottom")
ax.set_xlim(*XL); ax.set_ylim(*ZL); ax.set_aspect("equal")
ax.set_xticks([-100, -50, 0, 50, 100]); ax.set_yticks([-100, -50, 0, 50, 100])
ax.set_xlabel("across the thigh (mm), + medial")
ax.set_ylabel("front to back (mm), + anterior")

# ── inset on the femoral sheath, over the back of the thigh (muscle and fat only there) ───────────────────────
IX, IZ = (-20.0, 32.0), (60.0, 81.0)               # the window: nerve, artery, vein, sheath wall and contents
ins_w = 164.0; ins_h = ins_w * (IZ[1] - IZ[0]) / (IX[1] - IX[0])
ins_x0, ins_z0 = 94.0 - ins_w, -96.0
axins = ax.inset_axes([ins_x0, ins_z0, ins_w, ins_h], transform=ax.transData)
tissue(axins)
fine_outline(axins, lw=0.9)
axins.set_xlim(*IX); axins.set_ylim(*IZ); axins.set_aspect("equal")
axins.set_xticks([]); axins.set_yticks([])
for sp in axins.spines.values():
    sp.set_linewidth(0.9)
axins.plot([IX[1] - 12.0, IX[1] - 2.0], [IZ[1] - 2.0, IZ[1] - 2.0], color="k", lw=1.6, solid_capstyle="butt", zorder=7)
axins.text(IX[1] - 13.0, IZ[1] - 2.0, "10 mm", ha="right", va="center", zorder=7,
           bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.0))
# the inset's window and its two connectors: dash-dot grey, so they read apart from the solid slab footprint and the
# dashed 1.5 mm region (matplotlib >= 3.10 returns one InsetIndicator)
ind = ax.indicate_inset_zoom(axins, edgecolor="0.25", lw=0.8, ls="-.", alpha=1.0, zorder=7)
for art in [ind.rectangle, *ind.connectors]:
    art.set_linestyle("-."); art.set_linewidth(0.8); art.set_edgecolor("0.25")

# ── legend: only what is drawn ─────────────────────────────────────────────────────────────────────────────────
SHORT = {"SoftTissue": "skin and fat", "Muscle": "muscle", "Femur": "femur (fixed)", "PubicRamus": "pubic ramus (fixed)",
         "Sheath": "sheath wall", "Perivascular": "sheath contents", "Artery": "artery wall", "Vein": "vein wall",
         "Nerve": "femoral nerve"}
present = [b for b in m3.ORDER if b in SHORT and (bodies == b).any()]
hs = [Patch(fc=COL[b], ec="k", lw=0.4, label=SHORT[b]) for b in present]
hs += [Patch(fc="#444", ec="k", lw=0.4, label="probe face"),
       Line2D([], [], color="k", lw=1.2, label="slab footprint"),
       Line2D([], [], color="k", lw=1.1, ls="--", label="1.5 mm region")]
fig.legend(handles=hs, loc="lower center", ncol=2, bbox_to_anchor=(0.55, 0.0), handlelength=1.3, columnspacing=1.6,
           labelspacing=0.3, handletextpad=0.5)
PS.save(fig, PS.COL_IN, FIG / "fig_slab_and_sweep.pdf", FIG / "fig_slab_and_sweep.png", dpi=300); plt.close(fig)
print(f"wrote {FIG / 'fig_slab_and_sweep.pdf'} {(FIG / 'fig_slab_and_sweep.pdf').stat().st_size / 1e6:.2f} MB; "
      f"slab {len(slab['T']):,} tets, leg {len(leg['T']):,} tets, worst centroid gap {worst:.2f} mm, "
      f"fine-region polygons {fine.sum()}/{len(polys)}; slab footprint x {sx0:g}..{sx1:g}, z {sz0:.1f}..{sz1:.1f} mm")
