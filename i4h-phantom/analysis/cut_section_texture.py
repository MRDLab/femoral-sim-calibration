"""Texture of the CAD section for the cut-away viewer: smooth tissue outlines in place of element boundaries. 2026-10-12.
The section is the one the mesh was built from (V4 for Mesh F, V5 since 2026-10-21), read from the mesh's own meta.

  $PY_PHYSICS analysis/cut_section_texture.py MESH.npz OUT_DIR [N_PX]

ILLUSTRATION ONLY (paper and grant pictures). The colours come from the CAD section the mesh was built from, classified
by the mesher's own function (femoral_fem.SECTION, loaded as graded_leg_mesh.py loads it). The viewer maps each cut point
to this texture by its REST (x, z), so the outlines move with the simulated deformation. ⚠️ The solver's material
assignment is per element and differs from these outlines by up to an element (30 mm in the coarse region).

Writes cut_texture.png (row 0 = largest z, so USD's v = 0 at the bottom is the smallest z) and cut_texture.json
(extent, digest of the mesh it was made for, and the per-element agreement check).
"""
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import femoral_fem as F  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "isaaclab"))
from cut_view import COLOR  # noqa: E402

npz, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
os.makedirs(out, exist_ok=True)
d = np.load(npz, allow_pickle=True)
sha = hashlib.sha256(open(npz, "rb").read()).hexdigest()[:12]
meta = json.loads(str(d["meta"][0]))
assert meta["kind"] == "graded_leg", meta["kind"]
F.BODIES["Sheath"]["E_pa"] = meta["sheath_kpa"] * 1e3; F.BODIES["Sheath"]["nu"] = F.NU      # as graded_leg_mesh.py
F.load_section(meta["section"], perivascular=bool(meta["perivascular"]))
names = [n for n, _ in F.SECTION]
R = float(meta["r_skin_m"]) * 1.01
x0, W = -R, 2 * R

zc = R - (np.arange(N) + 0.5) * W / N                 # row 0 = top = largest z
xc = x0 + (np.arange(N) + 0.5) * W / N
X, Z = np.meshgrid(xc, zc)
cls = np.full(X.size, -1, dtype=np.int16); Xf, Zf = X.ravel(), Z.ravel()
# polygon bodies (Sheath, its Perivascular bore, PubicRamus) have a scalar test; run it only inside their loops' bounding box
_secB = json.load(open(meta["section"]))["bodies"]
# 2026-10-21: only the polygon bodies this section has (Femoral V5 has no PubicRamus)
_pts = np.concatenate([np.asarray(P) for n in ("Sheath", "PubicRamus") if n in _secB for P in _secB[n]["polys"]])
_box = np.nonzero((Xf >= _pts[:, 0].min()) & (Xf <= _pts[:, 0].max()) & (Zf >= _pts[:, 1].min()) & (Zf <= _pts[:, 1].max()))[0]
for i, (_, contains) in enumerate(F.SECTION):
    try:
        hit = np.asarray(contains(Xf, Zf), dtype=bool)
    except ValueError:
        hit = np.zeros(Xf.size, dtype=bool)
        hit[_box] = [contains(float(x), float(z)) for x, z in zip(Xf[_box], Zf[_box])]
    cls[(cls < 0) & hit] = i
cls = cls.reshape(X.shape)
# a lumen is a hole in the mesh, so any cut face drawn over a lumen pixel is wall elements (the mesher keeps a tet unless
# most of it is lumen): paint each lumen in its own wall's colour. The voids precede their walls in nest order.
pal = np.array([COLOR[names[i + 1]] if n == "void" else COLOR[n] for i, n in enumerate(names)] + [COLOR["SoftTissue"]])
assert [names[i + 1] for i, n in enumerate(names) if n == "void"] == ["Artery", "Vein"], names
img = (np.clip(pal[cls], 0, 1) * 255).astype(np.uint8)          # cls -1 (outside skin) -> last row: skin colour
Image.fromarray(img).save(os.path.join(out, "cut_texture.png"))

# check that can fail: the section the texture shows is the section the mesh was built from. Every fine element
# (all four nodes within 1.6 dx of the y = 0 plane, in the fine window) should carry the texture's class at its centroid.
rest, tets, body = d["rest"], d["tets"], d["tet_body"].astype(str)
c = rest[tets].mean(1)
near = np.abs(c[:, 1]) < 1.6 * float(d["dx"][0])
fine = near & (np.linalg.norm(c[:, [0, 2]], axis=1) > meta["r_in_m"])
col = np.clip(((c[:, 0] - x0) / W * N).astype(int), 0, N - 1); row = np.clip(((R - c[:, 2]) / W * N).astype(int), 0, N - 1)
tex_name = np.array(names + ["SoftTissue"])[cls[row, col]]
agree = float((tex_name[fine] == body[fine]).mean())
agree_all = float((tex_name[near] == body[near]).mean())
print(f"texture {N} px over {W*1e3:.1f} mm ({W/N*1e6:.0f} um/px); classes {names}")
print(f"agreement with the mesh's element tissue at y = 0: fine window {agree*100:.1f} % of {fine.sum()} elements; whole cut {agree_all*100:.1f} % of {near.sum()}")
assert agree > 0.85, f"texture disagrees with the fine-window elements ({agree*100:.1f} %): wrong section, frame or flip"
# the whole-cut figure cannot see a left-right mirror (the bundle is ~2 % of the elements), so test the bundle alone, and
# show the test can fail: the same comparison against the texture mirrored in x must disagree.
BUNDLE = {"Artery", "Vein", "Nerve", "Sheath", "Perivascular"}
bun = near & np.isin(body, list(BUNDLE))
agree_b = float((tex_name[bun] == body[bun]).mean())
col_m = np.clip(((-c[:, 0] - x0) / W * N).astype(int), 0, N - 1)
agree_bm = float((np.array(names + ["SoftTissue"])[cls[row, col_m]][bun] == body[bun]).mean())
print(f"bundle elements at y = 0 ({bun.sum()}): agreement {agree_b*100:.1f} %; against the x-mirrored texture {agree_bm*100:.1f} % (must be low)")
assert agree_b > 0.85 and agree_bm < 0.5, "bundle agreement does not discriminate the mirror"
json.dump(dict(date="2026-10-12", mesh=npz, mesh_sha256_12=sha, section=meta["section"], n_px=N, x0_m=x0, z_top_m=R, width_m=W,
               uv="u = (x_rest - x0) / width, v = (z_rest - (z_top - width)) / width", classes=names,
               agreement_fine=agree, agreement_bundle=agree_b, agreement_bundle_mirrored=agree_bm, agreement_whole_cut=agree_all, purpose="illustration only"),
          open(os.path.join(out, "cut_texture.json"), "w"), indent=1)
