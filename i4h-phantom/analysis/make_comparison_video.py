"""Two live runs, one above the other, on one shared clock: shows how far the ultrasound image falls behind the probe.

  $PY_ANATOMY analysis/make_comparison_video.py TOP.json BOTTOM.json OUT.mp4 [--cap-s 180 --cap-mb 10]

2026-11-05. Nothing is re-simulated, interpolated or smoothed. At video time t (seconds since the run's loop started):
  * left panel  = that run's own 3-D view frame for SIMULATED time t: where the probe would be if it moved at real speed;
  * right panel = the newest ultrasound image whose recorded `image_ready_wall` is <= t: what the computer had finished.
The gap between the two is the image's age, printed per row. Before a run's first image exists the panel is black.

--clock compute (added 2026-11-05): both panels on COMPUTER time, played at 1x. The left panel is then the newest 3-D view
whose recorded `wall` (stamped right after that frame's FEM step and render, same zero as `image_ready_wall`) is <= t:
what the computer had shown by then. The video is NOT real time for either run: each row prints its own measured ratio of
simulated time to computer time (simulated motion / time of its last image), so no row can be read as real time.
  * extra check: no 3-D view is ever shown before its recorded `wall` time (asserted per output frame).

Checks that can fail (each refuses to write the video):
  * same subject: mesh digest, B-mode scene digest, sweep, depths, camera and sweep step identical in both runs;
  * every frame saved, and `image_ready_wall` non-decreasing (the "newest image" rule needs it);
  * no image is ever shown before its recorded ready time (asserted per output frame, not trusted to the construction);
  * with --cap-s / --cap-mb: the video is not truncated to fit; a run too long for the cap is refused, and the encoded file
    is measured (ffprobe) for duration, size, height >= 480 px and >= 20 frames/s.
"""
import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("top"); ap.add_argument("bottom"); ap.add_argument("out")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--hold-s", type=float, default=2.0, help="seconds held after the last image arrives")
ap.add_argument("--names", default="final run,earlier setting", help="row names, top,bottom")
ap.add_argument("--cap-s", type=float, default=None)
ap.add_argument("--cap-mb", type=float, default=None)
ap.add_argument("--clock", choices=("sim", "compute"), default="sim",
                help="left panel on simulated time (default) or on computer time (see docstring)")
a = ap.parse_args()

FRAME_S = 1 / 60.0                       # one simulated frame
PX = 500 / 80.0                          # B-mode: 500 px over 80 mm
CROP, CROPW = int(48 * PX), int(40 * PX) # 48 mm deep, 40 mm across (the probe width), as make_live_video_plain.py
CX0 = 250 - CROPW // 2


def load(path):
    r = json.load(open(path)); fr = np.load(path[:-5] + ".frames.npz"); tl = r["timeline"]
    keys = {int(k[1:]): k for k in fr.files if k.startswith("f")}
    views = {int(k[6:]): k for k in fr.files if k.startswith("view_f")}
    assert len(keys) == len(tl), f"{path}: {len(keys)} saved frames vs {len(tl)} timeline rows"
    ready = np.array([f["image_ready_wall"] for f in tl]); sim = np.array([f["sim_t"] for f in tl])
    assert np.all(np.diff(ready) >= 0), f"{path}: image_ready_wall decreases"
    wall = np.array([f["wall"] for f in tl])
    assert np.all(np.diff(wall) >= 0), f"{path}: wall decreases"
    assert np.all(ready >= wall), f"{path}: an image is recorded ready before its frame was simulated"
    return dict(r=r, fr=fr, tl=tl, keys=keys, views=views, ready=ready, sim=sim, wall=wall,
                rt=float(sim[-1] / ready[-1]),
                med_ms=float(np.median(np.diff(wall)) * 1e3), name=os.path.basename(path)[:-5])


runs = [load(a.top), load(a.bottom)]
for x, nm in zip(runs, a.names.split(",")): x["row"] = nm.strip()
SAME = ("mesh_sha256_12", "sweep_deg", "sweep_step_deg", "viewer_eye", "viewer_target", "viewer_cut_y", "leg_visual")
for k in SAME:
    assert runs[0]["r"].get(k) == runs[1]["r"].get(k), f"not the same subject: {k} {runs[0]['r'].get(k)} vs {runs[1]['r'].get(k)}"
assert runs[0]["r"]["worker"]["scene_sha"] == runs[1]["r"]["worker"]["scene_sha"], "different B-mode scenes"
d0 = [(row["depth_mm"]) for row in runs[0]["r"]["depth_rows"]]; d1 = [(row["depth_mm"]) for row in runs[1]["r"]["depth_rows"]]
assert d0 == d1, f"different press depths: {d0} vs {d1}"
print(f"same subject OK: mesh {runs[0]['r']['mesh_sha256_12']}, scene {runs[0]['r']['worker']['scene_sha']}, depths {d0}")

COMPUTE = a.clock == "compute"
end_s = max(max(x["ready"][-1], x["wall"][-1] if COMPUTE else x["sim"][-1]) for x in runs) + a.hold_s
if a.cap_s is not None and end_s > a.cap_s:
    raise SystemExit(f"REFUSED: the slower run needs {end_s:.1f} s, over the {a.cap_s:g} s cap. Not truncating.")
n_out = int(np.ceil(end_s * a.fps))

PH = 420                                 # panel height
v0 = runs[0]["fr"][next(iter(runs[0]["views"].values()))]
VW = int(round(v0.shape[1] * PH / v0.shape[0])) // 2 * 2
BW = int(round(PH * 40 / 48)) // 2 * 2
HDR, TOP = 56, 46
W = VW + BW; H = TOP + 2 * (HDR + PH)
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
font = ImageFont.truetype(F + ".ttf", 15); font_b = ImageFont.truetype(F + "-Bold.ttf", 17)
font_s = ImageFont.truetype(F + ".ttf", 12)


def label(x):
    r = x["r"]
    s = f"{x['row'].capitalize()}: time step {16.667 / r['substeps']:.2f} ms, {r['iterations']} iterations  ({x['med_ms']:.0f} ms per frame"
    return s + (f", {x['rt']:.2f}x real time)" if COMPUTE else ")")


_cache = {}
def panel(x, key, size, box=None):
    """Resized panel, cached: most output frames repeat the previous image."""
    ck = (x["name"], key, size)
    if ck not in _cache:
        if len(_cache) > 64: _cache.clear()
        a_ = x["fr"][key]
        if box: a_ = a_[box]
        _cache[ck] = Image.fromarray(a_).convert("RGB").resize(size, Image.LANCZOS)
    return _cache[ck]


def frame(t):
    img = Image.new("RGB", (W, H), (12, 12, 12)); d = ImageDraw.Draw(img)
    if COMPUTE:
        d.text((10, 6), f"Computer time since start  {t:6.1f} s   (not real time)", fill=(255, 255, 255), font=font_b)
        d.text((10, 27), "Plays at the computer's own pace.  Left: newest 3-D view finished.  Right: newest ultrasound image finished.",
               fill=(170, 170, 170), font=font_s)
    else:
        d.text((10, 6), f"Time since start  {t:6.1f} s", fill=(255, 255, 255), font=font_b)
        d.text((10, 27), "Left: probe moving at real speed (simulated time).  Right: the newest ultrasound image the computer had finished.",
               fill=(170, 170, 170), font=font_s)
    for row, x in enumerate(runs):
        y0 = TOP + row * (HDR + PH)
        if COMPUTE:
            kl = int(np.searchsorted(x["wall"], t, side="right")) - 1             # newest frame simulated and rendered by t
        else:
            kl = int(np.searchsorted(x["sim"], t + 1e-9, side="right")) - 1      # newest simulated frame with sim_t <= t
        if kl >= 0:
            if COMPUTE:
                assert x["wall"][kl] <= t, "a 3-D view is shown before it existed"
            vj = kl if kl in x["views"] else max(j for j in x["views"] if j <= kl)
            img.paste(panel(x, x["views"][vj], (VW, PH)), (0, y0 + HDR))
        elif not COMPUTE:
            img.paste(panel(x, x["views"][min(x["views"])], (VW, PH)), (0, y0 + HDR))
        kr = int(np.searchsorted(x["ready"], t, side="right")) - 1               # newest image with ready <= t
        d.text((10, y0 + 6), label(x), fill=(255, 255, 255), font=font_b)
        if kr >= 0:
            assert x["ready"][kr] <= t, "an image is shown before it existed"
            img.paste(panel(x, x["keys"][kr], (BW, PH), (slice(0, CROP), slice(CX0, CX0 + CROPW))), (VW, y0 + HDR))
            age = min(t, x["sim"][-1]) - x["sim"][kr]       # after the motion ends the probe stays at its last pose
            done = kr == len(x["sim"]) - 1
            if COMPUTE:
                msg = ("Finished: all images computed." if done else
                       f"Motion simulated so far {x['sim'][kr]:5.2f} of {x['sim'][-1]:.2f} s   (image {kr + 1} of {len(x['sim'])})")
            else:
                msg = ("Finished: all images computed." if done else
                       f"Image shows the probe as it was {age:5.1f} s ago   (image {kr + 1} of {len(x['sim'])})")
        else:
            d.text((VW + 20, y0 + HDR + PH // 2), "waiting for the first image", fill=(200, 200, 200), font=font)
            msg = "No image yet"
        d.text((10, y0 + 31), msg, fill=(255, 210, 90) if kr < len(x["sim"]) - 1 else (140, 220, 140), font=font)
    return img


tmp = a.out + ".partial.mp4"
enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
if a.cap_mb:
    kbps = int(a.cap_mb * 0.92 * 8e3 / end_s)           # two-pass to a size target, 8 % margin for the container
    enc += ["-b:v", f"{kbps}k"]
else:
    enc += ["-crf", "20"]
# frames go once into a lossless intermediate (a raw dump of the 5-minute clip would be ~27 GB), then are encoded from it
raw = os.path.splitext(a.out)[0] + ".lossless.mkv"
pipe = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(a.fps), "-i", "-", "-c:v", "libx264rgb", "-qp", "0", "-preset", "ultrafast", raw],
                        stdin=subprocess.PIPE)
for i in range(n_out):
    pipe.stdin.write(frame(i / a.fps).tobytes())
pipe.stdin.close(); assert pipe.wait() == 0, "lossless intermediate failed"
src = ["-i", raw]
if a.cap_mb:
    for p in (1, 2):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, *enc, "-pass", str(p), "-passlogfile", tmp + ".log",
                        *(["-f", "mp4", os.devnull] if p == 1 else [tmp])], check=True)
    for f_ in os.listdir(os.path.dirname(os.path.abspath(tmp))):
        if f_.startswith(os.path.basename(tmp) + ".log"):
            os.remove(os.path.join(os.path.dirname(os.path.abspath(tmp)), f_))
else:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, *enc, tmp], check=True)
os.remove(raw)

pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=height,r_frame_rate:format=duration,size", "-of", "json", tmp],
                               capture_output=True, text=True, check=True).stdout)
dur = float(pr["format"]["duration"]); mb = int(pr["format"]["size"]) / 1e6; h = pr["streams"][0]["height"]
num, den = map(int, pr["streams"][0]["r_frame_rate"].split("/")); fps = num / den
print(f"measured: {dur:.1f} s, {mb:.2f} MB, {W}x{h} px, {fps:g} frames/s")
if a.cap_s is not None or a.cap_mb is not None:
    bad = [m for m, ok in [(f"duration {dur:.1f} s > {a.cap_s}", a.cap_s is None or dur <= a.cap_s),
                           (f"size {mb:.2f} MB > {a.cap_mb}", a.cap_mb is None or mb <= a.cap_mb),
                           (f"height {h} < 480", h >= 480), (f"{fps:g} fps < 20", fps >= 20)] if not ok]
    if bad:
        raise SystemExit("REFUSED (file kept as " + tmp + "): " + "; ".join(bad))
os.replace(tmp, a.out)
for x in runs:
    print(f"{x['name']}: {len(x['sim'])} frames, simulated {x['sim'][-1]:.2f} s, last image at {x['ready'][-1]:.1f} s, "
          f"median {x['med_ms']:.1f} ms per frame")
print("wrote", a.out)
