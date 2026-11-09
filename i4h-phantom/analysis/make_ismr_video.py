"""The ISMR 2027 submission video: the paper's story in the paper's order, 1280 x 720, under 180 s and 10 MB.

    $PY_ANATOMY analysis/make_ismr_video.py OUT.mp4 [--stills DIR]

Sections, in the order of Main.pdf:
  title -> problem (Introduction) -> how the simulation works (Methods) -> benchtop phantom and its simulated twin
  (Fig. 1, Fig. 2) -> solver calibration on the slab -> speed and accuracy on the leg (Fig. 3, live comparison) ->
  vein against artery on the leg (live cut-away run, Fig. 4) -> conclusion and acknowledgment.

Inputs:
  analysis/video_assets/      the paper's figures as printed (extracted from Main.pdf and Supplement.pdf), the logos
  out/live_bmode/             the live runs: legV5_s2_i35 (plain) and legV5_s2_i35_cuttex (cut-away), frames and timelines
  ../clips/submission/        the chosen-vs-intermediate comparison clip, reused for the speed section
  ../figures/fig_slab_element_size.png
Every number on screen is quoted from Main.pdf. Live clips carry a playback-speed badge: 1x means as fast as it was
computed on our RTX 5090.
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                       # i4h-phantom/
REPO = ROOT.parent                       # the repository
A = HERE / "video_assets"
LB = ROOT / "out" / "live_bmode"
W, H, FPS = 1280, 720, 30

BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
REG = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
_fc = {}
def font(size, bold=True):
    k = (size, bold)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(BOLD if bold else REG, size)
    return _fc[k]

WHITE, BLACK = (255, 255, 255), (12, 12, 12)
MAROON, GOLD = (122, 0, 25), (255, 204, 51)
GREEN_ON_DARK = (140, 225, 110)
BOX_LIGHT, BOX_DARK = (150, 150, 150), (55, 55, 55)

# Logos: mrd-logo.png (the lab's file, trimmed of its white margin) and umn-wordmark-light.svg (the university's
# wordmark for light backgrounds), rendered to PNG; the -on-dark copy only turns the wordmark's black text white.
LOGO_MRD = Image.open(A / "mrd-logo-trimmed.png").convert("RGB")
LOGO_UMN_LIGHT = Image.open(A / "umn-wordmark-on-light.png").convert("RGBA")
LOGO_UMN_DARK = Image.open(A / "umn-wordmark-on-dark.png").convert("RGBA")


def scaled(img, w=None, h=None):
    s = (w / img.width) if w else (h / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)


# ------------------------------------------------------------------ drawing helpers
def wrap(d, text, f, maxw):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split():
            t = (cur + " " + w).strip()
            if d.textlength(t, font=f) <= maxw or not cur:
                cur = t
            else:
                lines.append(cur); cur = w
        lines.append(cur)
    return lines


def text_block(d, xy, text, f, maxw, fill, align="left", spacing=6):
    """Draw wrapped text with its top-left (or top-centre) at xy; return its height."""
    x, y = xy
    lh = f.size + spacing
    for i, ln in enumerate(wrap(d, text, f, maxw)):
        tw = d.textlength(ln, font=f)
        xx = x - tw / 2 if align == "center" else x
        d.text((xx, y + i * lh), ln, font=f, fill=fill)
    return len(wrap(d, text, f, maxw)) * lh - spacing


def box(im, rect, text, size=24, dark=False, fill_text=None, radius=16, alpha=235, align="center"):
    """A rounded call-out box with wrapped, vertically centred text."""
    d0 = ImageDraw.Draw(im)
    x0, y0, x1, y1 = rect
    f = font(size)
    lines = wrap(d0, text, f, x1 - x0 - 28)
    lh = size + 6
    th = len(lines) * lh - 6
    if y1 is None:
        y1 = y0 + th + 26
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    col = BOX_DARK if dark else BOX_LIGHT
    od.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=col + (alpha,))
    im.paste(Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB"))
    d = ImageDraw.Draw(im)
    tc = fill_text or (WHITE if dark else BLACK)
    ty = y0 + (y1 - y0 - th) / 2 - 2
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=f)
        tx = (x0 + x1 - tw) / 2 if align == "center" else x0 + 14
        d.text((tx, ty + i * lh), ln, font=f, fill=tc)
    return (x0, y0, x1, y1)


def arrow(im, p0, p1, col=BLACK, w=3):
    d = ImageDraw.Draw(im)
    d.line([p0, p1], fill=col, width=w)
    v = np.array(p1, float) - np.array(p0, float); n = np.linalg.norm(v) or 1; v /= n
    q = np.array([-v[1], v[0]])
    tip = np.array(p1, float)
    d.polygon([tuple(tip), tuple(tip - 14 * v + 6 * q), tuple(tip - 14 * v - 6 * q)], fill=col)


def caption(im, text, dark=True, colour=None, y=None, size=28):
    """Bottom caption, at most two lines. On dark frames: white bold text with a shadow, as in the lab's style.
    On white slides: a grey bar."""
    d = ImageDraw.Draw(im)
    f = font(size)
    lines = wrap(d, text, f, W - 120)
    assert len(lines) <= 2, f"caption over two lines: {text}"
    if len(lines) == 2:                      # balance the two lines: no single dangling word
        words = text.split()
        best = min(range(1, len(words)), key=lambda i: max(d.textlength(" ".join(words[:i]), font=f),
                                                           d.textlength(" ".join(words[i:]), font=f)))
        lines = [" ".join(words[:best]), " ".join(words[best:])]
    lh = size + 8
    y0 = (H - 14 - len(lines) * lh) if y is None else y
    if dark:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0)); sd = ImageDraw.Draw(sh)
        for i, ln in enumerate(lines):
            tw = d.textlength(ln, font=f)
            sd.text(((W - tw) / 2, y0 + i * lh), ln, font=f, fill=(0, 0, 0, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))
        sh = sh.filter(ImageFilter.GaussianBlur(2))
        im.paste(Image.alpha_composite(im.convert("RGBA"), sh).convert("RGB"))
        d = ImageDraw.Draw(im)
        for i, ln in enumerate(lines):
            tw = d.textlength(ln, font=f)
            d.text(((W - tw) / 2, y0 + i * lh), ln, font=f, fill=colour or WHITE)
    else:
        tw = max(d.textlength(ln, font=f) for ln in lines)
        box(im, ((W - tw) / 2 - 22, y0 - 8, (W + tw) / 2 + 22, y0 + len(lines) * lh), text, size=size,
            fill_text=colour or BLACK)


_LG = {}
def logos(im, dark):
    """MRD top left, the UMN wordmark top right. On dark frames the MRD logo sits on a small white card."""
    if not _LG:
        _LG["mrd"] = scaled(LOGO_MRD, h=50); _LG["mrd_s"] = scaled(LOGO_MRD, h=36)
        _LG["umn_l"] = scaled(LOGO_UMN_LIGHT, w=300); _LG["umn_d"] = scaled(LOGO_UMN_DARK, w=300)
    if dark:
        m = _LG["mrd_s"]
        ImageDraw.Draw(im).rounded_rectangle((10, 6, 10 + m.width + 12, 6 + m.height + 8), radius=8, fill=WHITE)
        im.paste(m, (16, 10))
        u = _LG["umn_d"]; im.paste(u, (W - u.width - 14, 14), u)
    else:
        im.paste(_LG["mrd"], (14, 8))
        u = _LG["umn_l"]; im.paste(u, (W - u.width - 14, 16), u)


def badge(im, text):
    d = ImageDraw.Draw(im); f = font(20)
    tw = d.textlength(text, font=f)
    box(im, (98, 12, 98 + tw + 36, 46), text, size=20, dark=True, radius=10)


def fit(img, maxw, maxh):
    s = min(maxw / img.width, maxh / img.height)
    return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS), s


def fade(fr, t, dur, tin=0.4, tout=0.4, to=(0, 0, 0)):
    a = min(1.0, t / tin if tin else 1.0, (dur - t) / tout if tout else 1.0)
    if a >= 1:
        return fr
    return Image.blend(Image.new("RGB", fr.size, to), fr, max(0.0, a))


# ------------------------------------------------------------------ live runs
class Run:
    """A live run's frames and timeline (out/live_bmode/<tag>.{json,frames.npz})."""
    PX = 500 / 80.0                      # B-mode: 500 px over 80 mm, both directions
    CROPW, CROPH = int(40 * 500 / 80), int(48 * 500 / 80)   # 40 mm wide (the probe) x 48 mm deep
    CX0 = 250 - int(40 * 500 / 80) // 2

    def __init__(self, tag):
        self.r = json.load(open(LB / f"{tag}.json"))
        self.tl = self.r["timeline"]
        self.z = np.load(LB / f"{tag}.frames.npz")
        self.wall = np.array([f["wall"] for f in self.tl])
        self.lastview = None

    def at_wall(self, w):
        return int(np.clip(np.searchsorted(self.wall, w, side="right") - 1, 0, len(self.tl) - 1))

    def first(self, phase_not="sweep"):
        return next(i for i, f in enumerate(self.tl) if f["phase"] != phase_not)

    def view(self, k):
        key = f"view_f{k:04d}"
        if key in self.z.files:
            self.lastview = Image.fromarray(self.z[key])
        return self.lastview

    def bmode(self, k):
        return self.z[f"f{k:04d}"]

    def label_px(self, k, name, img):
        """Darkest 7x7 patch inside the FEM-predicted lumen box (as analysis/make_live_video_plain.py)."""
        f = self.tl[k]
        x = 250 + f[f"{name}_cx_mm"] * self.PX
        y = (f["r_face_mm"] - f[f"{name}_cz_mm"]) * self.PX
        hh = max(4, int(f[f"{name}_cut_h_mm"] * self.PX / 2)); hw = max(4, int(f[f"{name}_cut_w_mm"] * self.PX / 6))
        a = img.astype(float); c = []
        for yy in range(int(y) - hh, int(y) + hh + 1, 2):
            for xx in range(int(x) - hw, int(x) + hw + 1, 2):
                c.append((xx, yy, a[yy - 3:yy + 4, xx - 3:xx + 4].mean()))
        dark = [q for q in c if q[2] < 8]
        return min(dark, key=lambda q: (q[0] - x) ** 2 + (q[1] - y) ** 2) if dark else min(c, key=lambda q: q[2])


def live_frame(run, k, labels=True, pct=False):
    """Isaac Sim view (left) and the ultrasound image (right) on black, panels 540 px tall."""
    im = Image.new("RGB", (W, H), BLACK)
    PH = 556; top = 92
    v = run.view(k)
    vw = int(v.width * PH / v.height)
    vimg = v.resize((vw, PH), Image.LANCZOS)
    b = run.bmode(k)
    crop = Image.fromarray(b[0:run.CROPH, run.CX0:run.CX0 + run.CROPW]).convert("RGB")
    bw = int(PH * run.CROPW / run.CROPH)
    bimg = crop.resize((bw, PH), Image.LANCZOS)
    gap = 16
    x0 = (W - vw - gap - bw) // 2
    im.paste(vimg, (x0, top)); im.paste(bimg, (x0 + vw + gap, top))
    d = ImageDraw.Draw(im)
    d.text((x0, top - 34), "Isaac Sim: probe and leg", font=font(22), fill=WHITE)
    d.text((x0 + vw + gap, top - 34), "Simulated ultrasound", font=font(22), fill=WHITE)
    if labels and run.tl[k]["phase"] != "sweep":
        sx = bw / run.CROPW; sy = PH / run.CROPH
        for name, col in (("artery", (240, 80, 80)), ("vein", (90, 160, 255))):
            px, py, _ = run.label_px(k, name, b)
            X = x0 + vw + gap + (px - run.CX0) * sx; Y = top + py * sy
            d.ellipse((X - 5, Y - 5, X + 5, Y + 5), fill=col)
            txt = f"{name} {run.tl[k][name + '_pct']:.0f} %" if pct else name
            tw = d.textlength(txt, font=font(22))
            tx = X - tw - 14 if name == "artery" else min(X + 14, x0 + vw + gap + bw - tw - 6)
            d.text((tx, Y - 52), txt, font=font(22), fill=col, stroke_width=3, stroke_fill=BLACK)
            arrow(im, (tx + (tw if name == "artery" else 0), Y - 22), (X + (-6 if name == "artery" else 6), Y - 6), col=col, w=2)
            d = ImageDraw.Draw(im)
    return im, (x0, top, vw, bw, PH, gap)


def live_segment(run, w0, w1, speed, dur, cap_fn, labels=True, pct=False, extra=None):
    """Play the run from wall-clock w0 to w1 (seconds into the run) at `speed` x as computed, for `dur` seconds."""
    def frame(t):
        k = run.at_wall(min(w1, w0 + t * speed))
        im, geom = live_frame(run, k, labels=labels, pct=pct)
        logos(im, dark=True)
        badge(im, "Playback speed: as computed" if abs(speed - 1) < 1e-6 else f"Playback speed: {speed:g}x")
        if extra:
            extra(im, t, geom, run.tl[k])
        cap_fn(im, t)
        return im
    return frame


# ------------------------------------------------------------------ the sections
def title(dur):
    def frame(t):
        im = Image.new("RGB", (W, H), WHITE)
        mrd = scaled(LOGO_MRD, h=118); umn = scaled(LOGO_UMN_LIGHT, w=520)
        im.paste(mrd, (40, 28)); im.paste(umn, (W - umn.width - 40, 28 + (mrd.height - umn.height) // 2), umn)
        d = ImageDraw.Draw(im)
        y = 190
        y += text_block(d, (W / 2, y), "Simulating Vein Compressibility for Ultrasound-Guided Femoral Access:",
                        font(40), 1080, BLACK, align="center", spacing=10) + 10
        y += text_block(d, (W / 2, y), "Speed, Accuracy, and Phantom Validation",
                        font(40), 1080, BLACK, align="center", spacing=10) + 44
        d.line((W / 2 - 220, y - 22, W / 2 + 220, y - 22), fill=MAROON, width=3)
        y += text_block(d, (W / 2, y), "Amit Manicka¹, Laurent Coulon², Darwin Ang², Michael Loushin³ "
                        "and Timothy M. Kowalewski¹", font(24, False), 1220, BLACK, align="center") + 26
        for ln in ("¹ Department of Mechanical Engineering, University of Minnesota, Minneapolis, MN, USA",
                   "² Lightside Surgical, Seattle, WA, USA",
                   "³ Visible Heart Laboratory, Department of Anesthesiology, University of Minnesota Medical School, "
                   "Minneapolis, MN, USA"):
            y += text_block(d, (W / 2, y), ln, font(19, False), 1150, (70, 70, 70), align="center") + 8
        text_block(d, (W / 2, H - 70), "ISMR 2027  ·  International Symposium on Medical Robotics", font(20), 1000,
                   MAROON, align="center")
        return fade(im, t, dur, tin=0.0, tout=0.5, to=WHITE)
    return frame


def figure_slide(img_path, dur, callouts, cap_text, cap_colour=None, fig_box=(30, 70, 760, 610), steps=None):
    """A paper figure on white, with grey call-out boxes and arrows. callouts: [(t_on, (x0,y0,x1,y1|None), text,
    [(fx, fy) arrow targets in figure pixels])]. cap_text: str or [(t_on, text, colour)]."""
    fig = Image.open(img_path).convert("RGB")
    fimg, s = fit(fig, fig_box[2] - fig_box[0], fig_box[3] - fig_box[1])
    fx0 = fig_box[0] + (fig_box[2] - fig_box[0] - fimg.width) // 2
    fy0 = fig_box[1] + (fig_box[3] - fig_box[1] - fimg.height) // 2

    def frame(t):
        im = Image.new("RGB", (W, H), WHITE)
        im.paste(fimg, (fx0, fy0))
        logos(im, dark=False)
        for t_on, rect, text, targets in callouts:
            if t < t_on:
                continue
            r = box(im, rect, text, size=23)
            for (px, py) in targets:
                tx, ty = fx0 + px * s, fy0 + py * s
                arrow(im, (r[0], (r[1] + r[3]) / 2), (tx, ty))      # boxes sit right of the figure
        caps = cap_text if isinstance(cap_text, list) else [(0, cap_text, cap_colour)]
        cur = [c for c in caps if t >= c[0]]
        if cur:
            caption(im, cur[-1][1], dark=False, colour=cur[-1][2], size=26)
        return fade(im, t, dur, tin=0.35, tout=0.35, to=WHITE)
    return frame


def comparison(dur, t0):
    """The chosen setting against the intermediate one, from clips/submission (panels only, header and footer cut)."""
    src = REPO / "clips" / "submission" / "submission_cutaway_smooth_colours.mp4"
    n = int(dur * FPS)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t0), "-i", str(src), "-frames:v", str(n), "-vf",
                          "crop=1260:450:0:100", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, 450, 1260, 3)

    def frame(t):
        k = min(int(t * FPS), len(frames) - 1)
        im = Image.new("RGB", (W, H), BLACK)
        f = Image.fromarray(frames[k]).resize((1280, 457), Image.LANCZOS)
        im.paste(f, (0, 118))
        d = ImageDraw.Draw(im)
        for x, ttl in ((10, "3-D view, chosen setting"), (571, "Ultrasound: chosen setting"),
                       (927, "Ultrasound: finer setting")):
            d.text((x, 80), ttl, font=font(22), fill=WHITE)
        logos(im, dark=True)
        badge(im, "Playback speed: as computed")
        caption(im, "The chosen setting's image stays within one frame of the probe. A slower, finer setting "
                    "falls behind it." if t < dur / 2 else
                    "Green: within one frame of the probe. Amber: further behind.", dark=True)
        return fade(im, t, dur)
    return frame


def closing(dur):
    def frame(t):
        im = Image.new("RGB", (W, H), WHITE)
        logos(im, dark=False)
        d = ImageDraw.Draw(im)
        d.text((80, 80), "Conclusion", font=font(34), fill=MAROON)
        y = 136
        for b in ("Isaac Sim predicted a low-fidelity benchtop phantom's artery within 5 points at every depth, and its "
                  "vein within 3 points through 11 mm, then collapsed the vein more than the phantom did.",
                  "The solver's time per frame fell from 29,700 ms to 30.4 ms, while staying within 1.03 points of the "
                  "reference setting.",
                  "With literature-based tissue properties, the leg model kept the vein-versus-artery difference that "
                  "ultrasound needs, at every solver setting.",
                  "Next: blood pressure and flow, validated force and acoustics, and a physical robot in the loop."):
            d.ellipse((86, y + 12, 96, y + 22), fill=MAROON)
            y += text_block(d, (112, y), b, font(25, False), 1080, BLACK) + 18
        d.line((80, y + 10, W - 80, y + 10), fill=(200, 200, 200), width=2)
        y += 34
        d.text((80, y), "Acknowledgment", font=font(28), fill=MAROON)
        y += 44
        text_block(d, (80, y), "The authors thank Professor Karthik Desingh and the Robotics: Perception and Manipulation "
                   "Laboratory for computing resources.", font(23, False), 1120, BLACK)
        text_block(d, (W / 2, H - 56), "Code, data and videos: github.com/MRDLab/femoral-sim-calibration", font(21),
                   1100, MAROON, align="center")
        return fade(im, t, dur, tin=0.35, tout=0.6, to=WHITE)
    return frame


# ------------------------------------------------------------------ the storyboard
def storyboard():
    plain = Run("legV5_s2_i35")
    cut = Run("legV5_s2_i35_cuttex")
    seg = []

    seg.append(("title", 5.0, title(5.0)))

    # problem: the live plain run's sweep, as computed
    def cap_problem(im, t):
        if t < 5.0:
            caption(im, "In ultrasound-guided femoral access, probe pressure deforms the tissue and changes the "
                        "anatomy the robot sees")
        elif t < 9.5:
            caption(im, "A simulator must reproduce these mechanics fast enough for a robot to interact with")
        else:
            caption(im, "How much computational effort does it take to reproduce a clinical signal faithfully "
                        "enough for a robot?", colour=GOLD)
    seg.append(("problem", 14.0, live_segment(plain, plain.wall[0], plain.wall[-1], 1.0, 14.0, cap_problem, labels=False)))

    seg.append(("causal chain", 6.0, figure_slide(
        A / "figS1_causal_chain.png", 6.0, [],
        "Every link from robot action to robot observation passes through tissue mechanics, so the physics engine "
        "decides what the robot sees", fig_box=(140, 80, 1140, 600))))

    # method: the cut-away run, from the end of the sweep into the first presses, as computed
    k0 = cut.first()
    def cap_method(im, t):
        caption(im, "Tissue: Stable Neo-Hookean finite elements, solved on one GPU with Vertex Block Descent" if t < 6
                else "Each frame, the deformed tissue is passed to NVIDIA's ultrasound ray tracer")
    seg.append(("method", 12.0, live_segment(cut, cut.wall[max(0, k0 - 40)], cut.wall[-1], 1.0, 12.0, cap_method)))

    # benchtop phantom
    seg.append(("fig1", 9.0, figure_slide(
        A / "fig1_setup.png", 9.0,
        [(0.6, (800, 110, 1240, None), "Simulated phantom: same geometry and supplier-specified materials, no input "
          "adjusted", [(1000, 600)]),
         (2.4, (800, 330, 1240, None), "Benchtop phantom: gel and silicone tubes, pressed by a UR5 robot to seven "
          "depths, three times each", [(1000, 1900)])],
        "Step 1: can the simulation predict a simple phantom of known materials?")))
    seg.append(("fig2", 10.0, figure_slide(
        A / "fig2_benchtop.png", 10.0,
        [(0.5, (800, 150, 1240, None), "Artery: within 5 points of the benchtop phantom at every depth", [(946, 735)]),
         (3.0, (800, 330, 1240, None), "Vein: within 3 points through 11 mm, then collapsed more than the benchtop "
          "phantom", [(903, 905)])],
        [(0, "Lines: simulated phantom. Markers: benchtop phantom, mean ± SD of three robot presses", None),
         (6.0, "It reproduced the artery at every depth and the vein through 11 mm, without tuning", (0, 110, 0))])))

    # solver calibration on the slab
    seg.append(("slab", 5.0, figure_slide(
        REPO / "figures" / "fig_slab_element_size.png", 5.0,
        [(0.4, (800, 200, 1240, None), "Element size, time step and iterations were first set on a slab cut from "
          "the leg (Supplement, Figs. S4 and S5)", [])],
        "1.5 mm elements and 35 iterations per step were carried to the leg model")))

    # speed and accuracy
    seg.append(("fig3", 12.0, figure_slide(
        A / "fig3_speed.png", 12.0,
        [(0.5, (800, 100, 1240, None), "Initial (reference) setting: 29,700 ms per frame", [(1199, 486)]),
         (2.5, (800, 260, 1240, None), "Chosen setting: 30.4 ms per frame, 977 times faster", [(327, 1251)]),
         (5.0, (800, 420, 1240, None), "With the 3-D view and ultrasound in the loop: 56.4 ms per frame",
          [(327, 1170)])],
        [(0, "Wall clock per frame against solver work, for twelve solver settings on one RTX 5090", None),
         (7.5, "Accuracy did not improve with more work: the chosen setting stayed within 1.03 points of the "
               "reference", (0, 110, 0))])))
    seg.append(("comparison", 10.0, comparison(10.0, 24.0)))

    # the clinical question on the leg: cut-away press at 3x, with live percentages
    kp = cut.first()
    def cap_leg(im, t):
        if t < 5:
            caption(im, "The leg model uses literature-based tissue properties. Does it tell the vein from the artery?")
        else:
            caption(im, "The vein collapses under the probe while the artery holds its shape", colour=GREEN_ON_DARK)
    seg.append(("leg", 12.0, live_segment(cut, cut.wall[kp], cut.wall[-1], 3.0, 12.0, cap_leg, labels=True, pct=True)))
    seg.append(("fig4", 10.0, figure_slide(
        A / "fig4_leg.png", 10.0,
        [(0.5, (800, 130, 1240, None), "Every one of the twelve solver settings: the vein compressed far more than "
          "the artery", [(811, 372), (811, 830)]),
         (3.0, (800, 330, 1240, None), "At 25.4 mm: vein 27.7 %, artery 83.7 % of rest (mean of five runs)",
          [(811, 830)])],
        [(0, "Vein (blue) and artery (red) lumen area against probe depth, leg model", None),
         (5.5, "This reproduces the compressibility criterion clinicians use to tell them apart", (0, 110, 0))])))

    seg.append(("closing", 12.0, closing(12.0)))
    return seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--stills", default=None, help="write one still per section here instead of encoding")
    ap.add_argument("--target-mb", type=float, default=9.4)
    a = ap.parse_args()
    seg = storyboard()
    total = sum(d for _, d, _ in seg)
    print(f">>> {len(seg)} sections, {total:.1f} s")
    if a.stills:
        p = pathlib.Path(a.stills); p.mkdir(parents=True, exist_ok=True)
        for i, (name, d, fn) in enumerate(seg):
            for frac in (0.3, 0.85):
                fn(d * frac).save(p / f"{i:02d}_{name.replace(' ', '_')}_{int(frac*100)}.png")
        return
    out = pathlib.Path(a.out)
    master = out.with_suffix(".master.mkv")
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
                            "-pix_fmt", "yuv420p", str(master)], stdin=subprocess.PIPE)
    for name, d, fn in seg:
        for i in range(int(round(d * FPS))):
            enc.stdin.write(fn(i / FPS).tobytes())
        print(f"    {name:<12} {d:5.1f} s", flush=True)
    enc.stdin.close(); assert enc.wait() == 0
    # two-pass to fit the conference's 10 MB limit
    kbps = int(a.target_mb * 8 * 1024 / total)
    common = ["-c:v", "libx264", "-preset", "slow", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(master), *common, "-pass", "1", "-passlogfile",
                    str(out.with_suffix("")), "-an", "-f", "mp4", "/dev/null"], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(master), *common, "-pass", "2", "-passlogfile",
                    str(out.with_suffix("")), "-an", str(out)], check=True)
    mb = out.stat().st_size / 1e6
    print(f">>> {out}: {total:.1f} s, {mb:.2f} MB, {W}x{H}, {FPS} fps (ISMR: <= 180 s, <= 10 MB, >= 480 px, >= 20 fps)")
    assert total <= 180 and mb <= 10.0


if __name__ == "__main__":
    main()
