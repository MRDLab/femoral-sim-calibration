"""ISMR 2027 submission video: one probe, two ultrasound images, one clock.

  $PY_ANATOMY analysis/make_submission_video.py SRC.mp4 CHOSEN.json INTERMEDIATE.json OUT.mp4 \
      --view "cut-away view, element colours" [--cap-s 180 --cap-mb 10]

2026-11-05. A RE-LAYOUT of an existing comparison clip (make_comparison_video.py --clock compute). Nothing is re-simulated,
re-timed, resampled or interpolated: output frame i is source frame i, so output time t is the source's computer time t.
The source frames were deleted after review, so the panels are cropped from the encoded comparison clip.

  source layout (910 x 998):  row 1 = chosen setting   [3-D view | B-mode]
                              row 2 = intermediate     [3-D view | B-mode]
  output layout:              [chosen 3-D view | chosen B-mode | intermediate B-mode]

The intermediate run's own 3-D view is dropped: the chosen run's probe is the one reference for both images. The chosen
B-mode keeps up with that probe BY CONSTRUCTION (same run), and the video says so on screen.

Labels come from the per-frame records (`timeline` in each run JSON), with the composer's own rules:
  3-D view at t = newest frame whose `wall` <= t;  B-mode at t = newest image whose `image_ready_wall` <= t.

Checks that can fail (each refuses to write the video):
  * same subject: mesh digest, B-mode scene digest, sweep, depths, camera identical in both runs;
  * the source clip has exactly the frame count the records imply (ceil((last event + 2 s hold) x 30));
  * ALIGNMENT: for each B-mode panel, pixels must change when the records say a new image arrived and stay put otherwise.
    A crop of the wrong row or a clock offset fails this. Threshold and agreement are printed;
  * with caps: ffprobe-measured duration, size, height >= 480 px, >= 20 frames/s.
"""
import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("chosen"); ap.add_argument("inter"); ap.add_argument("out")
ap.add_argument("--view", required=True)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--hold-s", type=float, default=2.0, help="must equal the source composer's hold")
ap.add_argument("--cap-s", type=float, default=None)
ap.add_argument("--cap-mb", type=float, default=None)
a = ap.parse_args()

# source geometry, from make_comparison_video.py (PH 420, HDR 56, TOP 46, VW 560 for this camera, BW 350)
SW, SH = 910, 998
PH, HDR, TOP, VW, BW = 420, 56, 46, 560, 350
ROW_Y = [TOP + HDR, TOP + (HDR + PH) + HDR]          # panel tops: 102, 578


def load(path):
    r = json.load(open(path)); tl = r["timeline"]
    g = lambda k: np.array([f[k] for f in tl])
    return dict(r=r, wall=g("wall"), ready=g("image_ready_wall"), sim=g("sim_t"),
                phase=[f["phase"] for f in tl], theta=g("theta_deg"), target=g("depth_mm"), pdepth=g("probe_depth_mm"),
                med_ms=float(np.median(np.diff(g("wall"))) * 1e3), rt=float(g("sim_t")[-1] / g("image_ready_wall")[-1]),
                name=os.path.basename(path)[:-5])


C, I = load(a.chosen), load(a.inter)
SAME = ("mesh_sha256_12", "sweep_deg", "sweep_step_deg", "viewer_eye", "viewer_target", "viewer_cut_y", "leg_visual")
for k in SAME:
    assert C["r"].get(k) == I["r"].get(k), f"not the same subject: {k}"
assert C["r"]["worker"]["scene_sha"] == I["r"]["worker"]["scene_sha"], "different B-mode scenes"
assert [d["depth_mm"] for d in C["r"]["depth_rows"]] == [d["depth_mm"] for d in I["r"]["depth_rows"]], "different depths"
SUBJ = f"mesh {C['r']['mesh_sha256_12']}, B-mode scene {C['r']['worker']['scene_sha']}"
print("same subject OK:", SUBJ)

end_s = max(max(x["ready"][-1], x["wall"][-1]) for x in (C, I)) + a.hold_s
n_expect = int(np.ceil(end_s * a.fps))
pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height,nb_read_frames", "-of", "json", a.src],
                               capture_output=True, text=True, check=True).stdout)["streams"][0]
assert (pr["width"], pr["height"]) == (SW, SH), f"source is {pr['width']}x{pr['height']}, expected {SW}x{SH}"
n_src = int(pr["nb_read_frames"])
assert n_src == n_expect, f"source has {n_src} frames, records imply {n_expect}: clock or source mismatch"
print(f"frame count OK: {n_src} frames = ceil(({end_s - a.hold_s:.2f} + {a.hold_s}) s x {a.fps})")


def pose(x, k):
    if x["phase"][k] == "sweep":
        return f"sweep, angle {x['theta'][k]:+.0f}°"
    return f"press, {max(x['pdepth'][k], 0):.1f} of {x['target'][k]:g} mm"


# ---- output layout
OW = VW + 2 * BW
BAR, PHDR, FOOT = 64, 66, 80
OH = BAR + PHDR + PH + FOOT
OH += OH % 2
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
fb = ImageFont.truetype(F + "-Bold.ttf", 17); fm = ImageFont.truetype(F + "-Bold.ttf", 14)
fr = ImageFont.truetype(F + ".ttf", 14); fs = ImageFont.truetype(F + ".ttf", 12)
WHITE, GREY, AMBER, GREEN = (255, 255, 255), (170, 170, 170), (255, 210, 90), (140, 220, 140)
ch_set = f"{16.667 / C['r']['substeps']:.2f} ms step, {C['r']['iterations']} iterations"
in_set = f"{16.667 / I['r']['substeps']:.2f} ms step, {I['r']['iterations']} iterations"


def compose(src, t):
    img = Image.new("RGB", (OW, OH), (12, 12, 12)); d = ImageDraw.Draw(img)
    d.text((10, 6), f"ISMR 2027 submission video  |  {a.view}", fill=WHITE, font=fb)
    d.text((10, 30), f"Computer time since start  {t:6.1f} s   (not real time).  One clock for all three panels; "
                     "nothing is slowed down or sped up.", fill=WHITE, font=fr)
    d.text((10, 48), "Each panel shows the newest frame its setting had finished at this moment.", fill=GREY, font=fs)
    y = BAR + PHDR
    for x0, (sx, sy, w) in zip((0, VW, VW + BW), ((0, ROW_Y[0], VW), (VW, ROW_Y[0], BW), (VW, ROW_Y[1], BW))):
        img.paste(Image.fromarray(src[sy:sy + PH, sx:sx + w]), (x0, y))
    kl = int(np.searchsorted(C["wall"], t, side="right")) - 1
    kc = int(np.searchsorted(C["ready"], t, side="right")) - 1
    ki = int(np.searchsorted(I["ready"], t, side="right")) - 1
    probe = pose(C, kl) if kl >= 0 else None
    # green = within ONE frame of the probe shown: every image is computed just after its own 3-D view, so even the
    # chosen image trails by exactly one frame (measured 2026-11-05: 99.9-100 % of output frames, all four runs)
    near = {pose(C, j) for j in (kl, kl - 1) if j >= 0}
    d.text((10, BAR + 4), "3-D view: chosen setting", fill=WHITE, font=fm)
    d.text((10, BAR + 22), ch_set, fill=GREY, font=fs)
    d.text((10, BAR + 42), f"Probe now: {probe}" if probe else "Starting", fill=WHITE, font=fr)
    for x0, x, k, title, sub in ((VW, C, kc, "Ultrasound: chosen setting", ch_set),
                                 (VW + BW, I, ki, "Ultrasound: intermediate setting", in_set)):
        d.text((x0 + 8, BAR + 4), title, fill=WHITE, font=fm)
        d.text((x0 + 8, BAR + 22), sub, fill=GREY, font=fs)
        if k < 0:
            msg, col = "No image yet", GREY
        else:
            p = pose(x, k); last = k == len(x["ready"]) - 1
            msg = "Finished: all images computed" if last else f"Image of probe at: {p}"
            col = GREEN if (last or p in near) else AMBER
        d.text((x0 + 8, BAR + 42), msg, fill=col, font=fs)
        if x0 > 0:
            d.line([(x0, BAR), (x0, BAR + PHDR + PH)], fill=(60, 60, 60), width=2)
    fy = BAR + PHDR + PH + 6
    d.text((10, fy), f"Chosen setting: {C['med_ms']:.0f} ms per frame, {C['rt']:.2f}x real time.   "
                     f"Intermediate: {I['med_ms']:.0f} ms per frame, {I['rt']:.2f}x real time.", fill=WHITE, font=fr)
    d.text((10, fy + 20), "The probe moves at the chosen setting's own pace. Every image is computed just after its own 3-D view,",
           fill=GREY, font=fs)
    d.text((10, fy + 37), f"so even the chosen image is one frame ({C['med_ms']:.0f} ms) behind.  Green = within one frame of the probe; "
                          "amber = further behind.", fill=GREY, font=fs)
    d.text((10, fy + 54), f"One run per setting, 2026-11-05, same leg model ({SUBJ}). Illustration only; "
                          "quantitative results are from the five-run repeats.", fill=GREY, font=fs)
    return img, (kl, kc, ki)


raw = os.path.splitext(a.out)[0] + ".lossless.mkv"
dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", a.src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       stdout=subprocess.PIPE)
enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                        "-r", str(a.fps), "-i", "-", "-c:v", "libx264rgb", "-qp", "0", "-preset", "ultrafast", raw],
                       stdin=subprocess.PIPE)
prev = None; diffs = []; idx = []
for i in range(n_src):
    buf = dec.stdout.read(SW * SH * 3)
    assert len(buf) == SW * SH * 3, f"short read at frame {i}"
    src = np.frombuffer(buf, np.uint8).reshape(SH, SW, 3)
    out, ks = compose(src, i / a.fps)
    enc.stdin.write(out.tobytes())
    cur = [src[ROW_Y[r]:ROW_Y[r] + PH, VW:VW + BW].astype(np.int16) for r in (0, 1)]
    if prev is not None:
        diffs.append([float(np.abs(c - p).mean()) for c, p in zip(cur, prev)])
    prev = cur; idx.append(ks)
assert not dec.stdout.read(1), "source has more frames than counted"
dec.wait(); enc.stdin.close(); assert enc.wait() == 0, "lossless intermediate failed"

# ---- alignment check: a new image (records) <=> the panel's pixels change (source)
# A clock offset or the wrong row must FAIL this, so the records are shifted by -3..+3 output frames and the true
# alignment (shift 0) must be the unique best. An arrival "changes" the panel if mean |diff| > 0.5 grey levels.
# Arrivals that do not change are allowed only while the probe holds still ("settle"): consecutive images are then
# nearly identical. First version (geometric-mean threshold) refused a correctly aligned clip; see the README.
diffs = np.array(diffs); idx = np.array(idx); tt = np.arange(n_src) / a.fps
ok = True
for col, (nm, x) in enumerate((("chosen B-mode", C), ("intermediate B-mode", I))):
    frac = {}
    for off in range(-3, 4):
        k = np.searchsorted(x["ready"], tt + off / a.fps, side="right") - 1
        ev = np.diff(k) != 0
        frac[off] = float(np.mean(diffs[ev, col] > 0.5))
    k = idx[:, 1 + col]; ev = np.diff(k) != 0
    miss_ph = sorted({x["phase"][j] for j in k[1:][ev & (diffs[:, col] <= 0.5)]})
    rival = max(v for o, v in frac.items() if o != 0)
    print(f"alignment {nm}: {ev.sum()} new images; changed on arrival at shift 0: {frac[0]:.1%}; best other shift: "
          f"{rival:.1%}; misses only in phases {miss_ph}")
    ok &= frac[0] >= 0.90 and frac[0] > rival and set(miss_ph) <= {"settle"}
if not ok:
    raise SystemExit(f"REFUSED: panels do not change when the records say they should (lossless kept as {raw})")

tmp = a.out + ".partial.mp4"
encargs = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
if a.cap_mb:
    kbps = int(a.cap_mb * 0.92 * 8e3 / (n_src / a.fps))
    encargs += ["-b:v", f"{kbps}k"]
    for p in (1, 2):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, *encargs, "-pass", str(p), "-passlogfile", tmp + ".log",
                        *(["-f", "mp4", os.devnull] if p == 1 else [tmp])], check=True)
    for f_ in os.listdir(os.path.dirname(os.path.abspath(tmp))):
        if f_.startswith(os.path.basename(tmp) + ".log"):
            os.remove(os.path.join(os.path.dirname(os.path.abspath(tmp)), f_))
else:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, *encargs, "-crf", "20", tmp], check=True)
os.remove(raw)

pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height,r_frame_rate:format=duration,size", "-of", "json", tmp],
                               capture_output=True, text=True, check=True).stdout)
dur = float(pr["format"]["duration"]); mb = int(pr["format"]["size"]) / 1e6; h = pr["streams"][0]["height"]
num, den = map(int, pr["streams"][0]["r_frame_rate"].split("/")); fps = num / den
print(f"measured: {dur:.1f} s, {mb:.2f} MB, {pr['streams'][0]['width']}x{h} px, {fps:g} frames/s")
bad = [m for m, good in [(f"duration {dur:.1f} s > {a.cap_s}", a.cap_s is None or dur <= a.cap_s),
                         (f"size {mb:.2f} MB > {a.cap_mb}", a.cap_mb is None or mb <= a.cap_mb),
                         (f"height {h} < 480", h >= 480), (f"{fps:g} fps < 20", fps >= 20)] if not good]
if bad:
    raise SystemExit("REFUSED (file kept as " + tmp + "): " + "; ".join(bad))
os.replace(tmp, a.out)
for x in (C, I):
    print(f"{x['name']}: {len(x['sim'])} frames, median {x['med_ms']:.1f} ms per frame, {x['rt']:.3f}x real time, "
          f"last image at {x['ready'][-1]:.1f} s")
print("wrote", a.out)
