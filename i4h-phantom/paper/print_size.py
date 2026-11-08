"""Print-size rules for the paper's plots (graphs with labelled axes). 2026-10-24, by direction after review.

IEEEtran 10 pt conference, two columns (2026-11-04): one column 3.5 in, full width 7.16 in. A plot is drawn AT its
printed width (figure* -> TEXT_IN, figure -> COL_IN) and no visible text may be smaller than MIN_PT. `save` asserts
both before writing, so a figure cannot drift back to unreadable. Not for the mesh pictures, video frames or drawings.

RC is the paper's one style (serif at 9 pt, no legend frames, no grid, open spines). The colour code is the paper's
one colour code: vein blue, artery red, reference black, the chosen setting green. Other solver settings are a
light-to-dark ramp of the vessel's colour (`ramp`), never a colour of their own. Encodings (solid/dashed, open
markers, n) are explained in captions, not drawn in the plot.
"""
import matplotlib
import matplotlib.colors
import matplotlib.text
import numpy as np

TEXT_IN, COL_IN, MIN_PT = 7.16, 3.5, 9.0
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
      "font.size": MIN_PT, "axes.labelsize": MIN_PT, "axes.titlesize": MIN_PT, "xtick.labelsize": MIN_PT,
      "ytick.labelsize": MIN_PT, "legend.fontsize": MIN_PT, "legend.title_fontsize": MIN_PT, "figure.titlesize": MIN_PT,
      "legend.frameon": False, "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.0, "lines.markersize": 3.5,
      "axes.spines.top": True, "axes.spines.right": True,
      "pdf.fonttype": 42}   # TrueType in PDFs: IEEE PDF eXpress rejects Type 3 fonts
VEIN, ARTERY, REF, CHOSEN = "#1f5fbf", "#c0392b", "#000000", "#2e8b57"
BAND = "0.90"              # tolerance band and closure-floor band: light grey, behind everything
TOL_PTS = 3.0              # the +-3 point tolerance on "gap from reference"


def ramp(colour, n, lo=0.4):
    """`n` tints of `colour`, light to dark: from a `lo` fraction of the way from white up to the full colour."""
    c = np.array(matplotlib.colors.to_rgb(colour))
    return [tuple(1.0 - (1.0 - c) * t) for t in np.linspace(lo, 1.0, n)]


def small_text(fig):
    fig.canvas.draw()
    return [(t.get_text()[:40], round(t.get_fontsize(), 1)) for t in fig.findobj(matplotlib.text.Text)
            if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-6]


def clipped_text(fig, tol_px=1.0):
    """Visible text whose box runs past the figure's edge (a legend pushed off the page passes the size check)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height
    # tick labels whose tick lies outside the axis's view range are never drawn: leave them out
    undrawn = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval()); eps = 1e-9 * max(1.0, abs(hi - lo))
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - eps <= tk.get_loc() <= hi + eps):
                    undrawn.update((id(tk.label1), id(tk.label2)))
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        if not (t.get_visible() and t.get_text().strip()) or id(t) in undrawn:
            continue
        if type(t).__name__ == "Text3D":
            continue   # 3-D text reports no usable window extent (measured ~70,000 px off); check those labels by eye
        b = t.get_window_extent(r)
        if b.x0 < -tol_px or b.y0 < -tol_px or b.x1 > W + tol_px or b.y1 > H + tol_px:
            out.append(t.get_text()[:40])   # partly cut off by the figure edge
    return out


def save(fig, width_in, *paths, dpi=300):
    """Assert the printed width, the smallest text, and that no text runs off the figure; then save to every path."""
    w = fig.get_size_inches()[0]
    assert abs(w - width_in) < 1e-9, f"figure is {w} in wide, its printed width is {width_in} in"
    small = small_text(fig)
    assert not small, f"text below {MIN_PT} pt: {small}"
    off = clipped_text(fig)
    assert not off, f"text runs off the figure: {off}"
    for p in paths:
        fig.savefig(p, dpi=dpi)
