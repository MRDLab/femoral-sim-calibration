"""Re-encode a coupled live-loop run with a PLAIN-ENGLISH side panel, for a grant proposal / paper. 2026-09-28.

  $PY_ANATOMY analysis/make_live_video_plain.py out/live_bmode/legF_s2_i35.json OUT_DIR

Same frames as analysis/make_live_video.py (nothing re-simulated, nothing smoothed); only the labelling changes:
  * the ultrasound image is cropped to the central 48 x 48 mm (the full image is 80 x 80 mm; the vessels sit in the
    top-left quarter of it) and the artery and vein are labelled from the FEM lumen centroids the loop logged every frame;
  * the side panel says what is on screen in words a reviewer outside the project can read: no solver notation,
    no item numbers, no in-house shorthand.
Checks that can fail: the lateral-mirror check of the original encoder; the item-39 near-uniform-view guard; and a new
one, that the FEM-to-image mapping puts BOTH labels on a dark lumen in every settled frame (a wrong scale or a wrong
origin lands on speckle and the encode refuses).

Outputs: <tag>_plain_realtime_60fps.mp4, <tag>_plain_asrun_NNfps.mp4, <tag>_plain_preview.gif, and full-resolution
PNGs of the first press frame and of every settled frame (frame_<depth>mm.png).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

src = sys.argv[1]
OUTDIR = sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
tag = os.path.basename(src)[:-5]
r = json.load(open(src))
fr = np.load(src[:-5] + ".frames.npz")
mesh = np.load(r["mesh"], allow_pickle=True)
# 2026-10-21: the V5 leg mesh (the cast leg: no pubic ramus, 81 mm long) joins Mesh F. Anything else is refused.
LEG_MESHES = {"c4e13e7d7574": "Mesh F (V4)", "d14afb0ae307": "V5 leg mesh"}
assert r["mesh_sha256_12"] in LEG_MESHES, f"not a paper leg mesh: {r['mesh_sha256_12']}"
assert '"kind": "graded_leg"' in str(mesh["meta"][0]), "not a whole-leg mesh; the block is retired"
_mmeta = json.loads(str(mesh["meta"][0]))
LEG_LEN = float(_mmeta["leg_s_range_mm"][1] - _mmeta["leg_s_range_mm"][0])
HAS_RAMUS = bool((mesh["tet_body"] == "PubicRamus").any())
# Nominal closure floor (photos/2026-09-10 README §8): self-contact stops the walls at 0.8 dx, times the lumen's cut width
# 8.9-12.5 mm (CAD, the same in V4 and V5), over this mesh's own rest area. 13-19 % on Mesh F; 14-20 % on V5.
FLOOR = tuple(100 * 0.8 * float(_mmeta["fine_dx_m"]) * 1e3 * w / (float(mesh["rest_area_vein"][0]) * 1e6) for w in (8.9, 12.5))
tl = r["timeline"]
rows = {f["frame"]: f for f in tl}
keys = sorted([k for k in fr.files if k.startswith("f")], key=lambda k: int(k[1:]))
assert len(keys) == len(tl), f"{len(keys)} saved frames vs {len(tl)} timeline rows"
wall = np.array([f["wall"] for f in tl]); interval = np.diff(wall, prepend=wall[0]) * 1e3
med_ms = float(np.median(interval[1:]))
asrun_fps = max(1, int(round(1e3 / med_ms)))
mean_fps = (len(tl) - 1) / (wall[-1] - wall[0])

# ---- image geometry: 500 px over 80 mm in BOTH directions (6.25 px/mm); depth 0 is the probe face; lateral 0 is
# the probe centre, +x to the right (verify_bmode.py's column convention). The FEM centroids are in mm about the limb
# axis, so depth below the face = r_face_mm - z_mm.
PX = 500 / 80.0
CROP_MM = 48.0                     # 48 mm deep
CROP = int(CROP_MM * PX)           # 300 px
CROPW_MM = 40.0                    # 40 mm across: the probe's width, so no padding either side (2026-09-30; runbook crop rule)
CROPW = int(CROPW_MM * PX)         # 250 px
CX0 = 250 - CROPW // 2             # crop columns 125..375, rows 0..300
def to_px(f, name):
    return 250 + f[f"{name}_cx_mm"] * PX, (f["r_face_mm"] - f[f"{name}_cz_mm"]) * PX
def lumen_px(f, name, img):
    """Label point: the darkest 7 x 7 patch inside the FEM-predicted lumen box (centroid +/- half the cut height
    vertically, +/- a third of the cut width laterally). The FEM centroid places the box; the image places the dot within
    it. Returns (x, y, mean intensity of that patch). For a flattened vein the near wall's echo fills the top of the slit,
    so the dark lumen sits 1-2 mm below the centroid; the box absorbs that, a wrong mapping does not fit in it."""
    x, y = to_px(f, name)
    hh = max(4, int(f[f"{name}_cut_h_mm"] * PX / 2)); hw = max(4, int(f[f"{name}_cut_w_mm"] * PX / 6))
    img = np.asarray(img, dtype=float)
    cands = []
    for yy in range(int(y) - hh, int(y) + hh + 1, 2):
        for xx in range(int(x) - hw, int(x) + hw + 1, 2):
            cands.append((xx, yy, img[yy - 3:yy + 4, xx - 3:xx + 4].mean()))
    dark = [c for c in cands if c[2] < 8]                       # the dark patch nearest the FEM centroid, else the darkest
    return min(dark, key=lambda c: (c[0] - x) ** 2 + (c[1] - y) ** 2) if dark else min(cands, key=lambda c: c[2])
def to_crop(x, y):
    return (x - CX0) * W / CROPW, y * Hh / CROP

# ---- check 1 (from the original encoder): lateral order artery-left-of-vein in the first press frame.
cent = {str(n): float((mesh["rest"][mesh[f"ring_{n}"]][:, 0] * 1e3).mean()) for n in mesh["ring_names"]}
first_press = next(f for f in tl if f["phase"] != "sweep")
b = fr[f"f{first_press['frame']:04d}"].astype(float)
band = b[int(15 / 80 * 500):int(40 / 80 * 500)]
dcols = np.where(b[int(5 / 80 * 500):int(10 / 80 * 500)].mean(axis=0) > 10)[0]
mask = band < 8; mask[:, :dcols.min() + 4] = False; mask[:, dcols.max() - 3:] = False
lab, nlab = ndimage.label(mask)
blobs = sorted([(np.where(lab == i)[1].mean(), (lab == i).sum()) for i in range(1, nlab + 1) if (lab == i).sum() > 150], key=lambda t: t[0])
assert len(blobs) >= 2 and (blobs[0][1] < blobs[-1][1]) == (cent["artery"] < cent["vein"]), "LATERAL MIRROR"
print(f"ordering check OK: artery x={cent['artery']:+.2f} < vein x={cent['vein']:+.2f} mm, smaller lumen on the left")

# ---- check 2: the label mapping lands on a dark lumen in the first press frame and every settled frame.
settled = {}
for f in tl:
    if f["phase"] == "settle": settled[f["depth_mm"]] = f["frame"]
check_frames = [first_press["frame"]] + [settled[d] for d in sorted(settled)]
for fi in check_frames:
    f = rows[fi]; img = fr[f"f{fi:04d}"].astype(float)
    msg = []
    for name in ("artery", "vein"):
        x0, y0 = to_px(f, name); x, y, m = lumen_px(f, name, img)
        assert m < 15, f"no dark lumen inside the predicted {name} box at frame {fi} (darkest patch mean {m:.0f} at px {x},{y})"
        msg.append(f"{name} centroid px ({x0:.0f},{y0:.0f}) -> dot ({x},{y}) mean {m:.1f}")
    print(f"label check OK frame {fi:4d} depth {f['depth_mm']:5.1f} mm: " + "; ".join(msg))

# ---- check 3 (item 39): the room view is a picture, not a near-uniform field.
view_keys = sorted([k for k in fr.files if k.startswith("view_f")], key=lambda k: int(k[6:]))
assert view_keys, "no room-view frames in this run"
view_by_frame = {int(k[6:]): k for k in view_keys}
def nonmodal_fraction(a):
    a = np.asarray(a)[..., :3].astype(int) // 32
    vals, cnt = np.unique(a.reshape(-1, 3), axis=0, return_counts=True)
    return float((np.abs(a - vals[cnt.argmax()]).sum(-1) != 0).mean())
worst = min(((int(k[6:]), nonmodal_fraction(fr[k])) for k in view_keys[::max(1, len(view_keys) // 40)]), key=lambda t: t[1])
assert worst[1] >= 0.03, f"near-uniform view at frame {worst[0]}"
print(f"view check OK: non-modal fraction min {worst[1]:.3f}")

Hh = 500
W = int(round(Hh * CROPW_MM / CROP_MM)) // 2 * 2   # true aspect for the 40 x 48 mm crop
vh0, vw0 = fr[view_keys[0]].shape[:2]
VIEW_W = int(round(vw0 * Hh / vh0)) // 2 * 2
PANEL = 470
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21)
font_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
font_l = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
RUN_DATE = r.get("date", "2026-09-09")             # runs before 2026-10-12 did not record a date; the only one encoded is 2026-09-09
LABEL_DATE = "2026-09-30" if RUN_DATE == "2026-09-09" else RUN_DATE
if r.get("viewer_cut_y") is not None:
    _side = {"below": "hip end", "above": "knee end"}[r["viewer_cut_keep"]]
    VIEW_TITLE = (f"3-D view: thigh cut open at the ultrasound plane, from the {_side}",
                  "Vein is left of the artery here, right of it in the ultrasound image"
                  if r["viewer_cut_keep"] == "below" else "")
    # check: the caption's left/right claim, from the recorded camera. Project the lumen centroids of the first press frame
    # onto the camera's right vector (look-at with +z up, as the loop poses it); refuse if the order disagrees.
    _eye = np.array([float(v) for v in r["viewer_eye"].split(",")]); _tgt = np.array([float(v) for v in r["viewer_target"].split(",")])
    _fwd = (_tgt - _eye) / np.linalg.norm(_tgt - _eye); _right = np.cross(_fwd, [0.0, 0.0, 1.0])
    _fp = next(f for f in tl if f["phase"] != "sweep")
    _sx = {n: float(np.dot(np.array([_fp[f"{n}_cx_mm"], _fp[f"{n}_cy_mm"], _fp[f"{n}_cz_mm"]]) * 1e-3 - _eye, _right)) for n in ("artery", "vein")}
    assert (_sx["vein"] < _sx["artery"]) == (r["viewer_cut_keep"] == "below"), f"3-D view vessel order disagrees with the caption: {_sx}"
    print(f"3-D view order check OK: screen-right coordinate vein {_sx['vein']*1e3:+.1f} mm, artery {_sx['artery']*1e3:+.1f} mm")
else:
    VIEW_TITLE = ("3-D view: the simulated thigh (finite-element model) and the ultrasound probe", "")
N_TETS = int(mesh["tets"].shape[0]) if "tets" in mesh.files else None

def compose(i, k):
    f = rows[int(k[1:])]
    fi = int(k[1:])
    canvas = Image.new("RGB", (VIEW_W + W + PANEL, Hh), (12, 12, 12))
    # left: the 3-D room view
    vk = view_by_frame.get(fi) or view_by_frame[max(j for j in view_by_frame if j <= fi)]
    canvas.paste(Image.fromarray(fr[vk]).convert("RGB").resize((VIEW_W, Hh), Image.LANCZOS), (0, 0))
    dv = ImageDraw.Draw(canvas)
    dv.text((10, 8), VIEW_TITLE[0], fill=(40, 40, 40), font=font_s)
    if VIEW_TITLE[1]:
        dv.text((10, 24), VIEW_TITLE[1], fill=(40, 40, 40), font=font_s)
    # middle: the ultrasound image, 40 mm across (the probe width) by 48 mm deep, enlarged
    crop = Image.fromarray(fr[k][0:CROP, CX0:CX0 + CROPW]).convert("RGB").resize((W, Hh), Image.LANCZOS)
    canvas.paste(crop, (VIEW_W, 0))
    d = ImageDraw.Draw(canvas)
    d.text((VIEW_W + 10, 8), "Ultrasound image (40 mm wide, 48 mm deep)", fill=(255, 255, 255), font=font_s)
    for mm in range(0, 49, 10):
        y = int(mm / CROP_MM * (Hh - 1))
        d.line([(VIEW_W + W - 8, y), (VIEW_W + W - 1, y)], fill=(200, 200, 60))
        d.text((VIEW_W + W - 52, min(Hh - 16, max(0, y - 7))), f"{mm} mm", fill=(200, 200, 60), font=font_s)
    d.text((VIEW_W + W - 130, Hh - 34), "depth below probe", fill=(200, 200, 60), font=font_s)
    if f["phase"] != "sweep":
        for name, label, col in (("artery", "artery", (255, 120, 120)), ("vein", "vein", (120, 170, 255))):
            px, py, _ = lumen_px(f, name, fr[k])
            x, y = to_crop(px, py)
            x += VIEW_W
            dx = -1 if name == "artery" else 1
            tx, ty = (x + dx * 70, y - 60) if y > 110 else (x + dx * 60, y + 62)   # keep the text off the title line
            tx = min(max(tx, VIEW_W + 40), VIEW_W + W - 75)                          # and inside the image, clear of the depth ticks
            d.line([(tx, ty + 8), (x, y)], fill=col, width=2)
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
            tw = d.textlength(label, font=font_l)
            d.text((tx - tw / 2, ty - 10), label, fill=col, font=font_l)
    # right: the panel, in plain words
    x = VIEW_W + W + 16; y = 14
    phase = f["phase"]
    title = {"sweep": "Scanning round the thigh to find the vessels",
             "ramp": "Pressing: probe moving into the skin",
             "settle": "Pressing: probe held, tissue settling"}[phase]
    d.text((x, y), title, fill=(255, 255, 255), font=font_b); y += 36
    depth_line = f"Probe pushed into the skin   {f.get('probe_depth_mm', 0.0):5.1f} mm" if phase != "sweep" else f"Probe angle round the thigh  {f.get('theta_deg', 0.0):+5.0f} deg"
    d.text((x, y), depth_line, fill=(220, 220, 220), font=font); y += 30
    vein_pct, art_pct = f.get("vein_pct", float("nan")), f.get("artery_pct", float("nan"))
    d.text((x, y), "Vessel cross-section, as % of its resting size:", fill=(220, 220, 220), font=font); y += 26
    d.text((x + 16, y), f"vein     {vein_pct:5.0f} %", fill=(120, 170, 255), font=font_b); y += 28
    d.text((x + 16, y), f"artery   {art_pct:5.0f} %", fill=(255, 120, 120), font=font_b); y += 34
    d.text((x, y), f"Simulated time elapsed        {f['sim_t']:5.2f} s", fill=(220, 220, 220), font=font); y += 24
    d.text((x, y), f"Real time elapsed             {f['wall']:5.1f} s", fill=(220, 220, 220), font=font); y += 24
    d.text((x, y), f"This frame took               {interval[i]:5.0f} ms", fill=(220, 220, 220), font=font); y += 24
    d.text((x, y), f"Image behind the tissue by    {f.get('total_s', 0) * 1e3:5.0f} ms", fill=(220, 220, 220), font=font); y += 34
    foot = [f"Finite-element model of a {LEG_LEN:g} mm thigh segment: femur, {'pubic bone, ' if HAS_RAMUS else ''}",
            f"muscle, fat, femoral artery, vein and nerve ({N_TETS:,} elements)" if N_TETS else "muscle, fat, femoral artery, vein and nerve",
            f"Time step {16.667 / r['substeps']:.1f} ms, {r['iterations']} solver iterations per step",
            "Ultrasound image ray-traced from the deformed tissue",
            (f"Everything runs together on one GPU at {1e3 / med_ms:.1f} frames per second" if r.get("viewer_cut_y") is None
             else "Everything runs together on one GPU"),    # cut-away runs are illustrations: their speed is not quoted
            *(["3-D outlines: CAD cross-section, moved with the simulation"] if r.get("viewer_cut_texture") else []),
            f"Vein area cannot be read below about {FLOOR[0]:.0f}-{FLOOR[1]:.0f} % of rest at",
            "this mesh resolution; blood pressure inside the vessels is off"]
    for ln in foot:
        d.text((x, y), ln, fill=(150, 150, 150), font=font_s); y += 17
    d.text((x, Hh - 22), f"frame {f['frame']} of {len(tl)}   run of {RUN_DATE}, labels {LABEL_DATE}", fill=(120, 120, 120), font=font_s)
    return canvas

tmp = tempfile.mkdtemp(prefix="live_video_plain_", dir=os.environ.get("CLAUDE_JOB_DIR", "/tmp"))
stills = {first_press["frame"]: "frame_rest.png"} | {settled[d]: f"frame_press_{d:g}mm.png" for d in settled}
for i, k in enumerate(keys):
    canvas = compose(i, k)
    canvas.save(os.path.join(tmp, f"{i:05d}.png"))
    if int(k[1:]) in stills:
        canvas.save(os.path.join(OUTDIR, stills[int(k[1:])]))

outs = []
for name, fps in [("realtime_60fps", 60), (f"asrun_{asrun_fps}fps", asrun_fps)]:
    out = f"{OUTDIR}/{tag}_plain_{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{tmp}/%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out], check=True)
    outs.append(out)
gif = f"{OUTDIR}/{tag}_plain_preview.gif"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "60", "-i", f"{tmp}/%05d.png",
                "-vf", "select='not(mod(n\\,4))',setpts=N/15/TB,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer",
                "-r", "15", gif], check=True)
outs.append(gif)
shutil.rmtree(tmp)
for o in outs:
    print(f"{os.path.getsize(o)/1e6:6.1f} MB  {o}")
print(f"{len(keys)} frames; median interval {med_ms:.1f} ms -> as-run {asrun_fps} fps; mean {mean_fps:.1f} fps; simulated {tl[-1]['sim_t']:.2f} s; wall {tl[-1]['wall']:.1f} s")
print("stills:", stills)
