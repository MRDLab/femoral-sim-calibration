"""Render-only cut-away of the deformed tet mesh for the Isaac Sim viewer. 2026-10-12.

The deformable's own render surface is closed, so the camera sees skin only. This builds, from the CURRENT node
positions, the part of the mesh on one side of a plane y = y_cut, opened at that plane:
  * the cut face: every tet crossing the plane sliced into a convex polygon, coloured by the tet's tissue;
  * the boundary surface (skin, far end cap, lumen walls) on the kept side, triangles clipped at the plane.
Nothing here touches the solver. Pure numpy, so it is testable outside Isaac Sim.

With y_cut = 0 the cut face IS the B-mode imaging plane, the plane the closure operator slices (closure_ops.py).
"""
import numpy as np

# the leg USD's own material colours (out/usd/femoral_leg_v4.usda), so this view matches the project's other renders
COLOR = {"SoftTissue": (0.97, 0.83, 0.55), "Muscle": (0.72, 0.16, 0.15), "Femur": (0.95, 0.93, 0.86),
         "PubicRamus": (0.90, 0.90, 0.93), "Sheath": (0.88, 0.86, 0.70), "Perivascular": (0.86, 0.84, 0.76),
         "FemoralCanal": (0.72, 0.62, 0.85), "Vein": (0.11, 0.25, 0.74), "Artery": (0.83, 0.06, 0.06),
         "Nerve": (0.16, 0.72, 0.28)}
_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])     # face i is opposite vertex i


def boundary_faces(tets):
    """Faces used by exactly one tet, and the index of that tet."""
    f = tets[:, _FACES].reshape(-1, 3)
    owner = np.repeat(np.arange(len(tets)), 4)
    key = np.sort(f, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    keep = cnt[inv.ravel()] == 1
    return f[keep], owner[keep]


def _fan(poly, n):
    """poly (m, k, 3) with n (m,) valid vertices each (3 or 4), already in cyclic order -> triangles (t, 3, 3)."""
    tri = [poly[:, [0, 1, 2]]]
    quad = n == 4
    if quad.any():
        tri.append(poly[quad][:, [0, 2, 3]])
    return np.concatenate(tri), np.concatenate([np.arange(len(poly)), np.nonzero(quad)[0]])


def slice_tets(pos, tets, y_cut, rest=None):
    """Convex polygon of every tet crossing y = y_cut, fanned to triangles. Returns (tris (t,3,3), tet index (t,)), and
    with `rest` also the same points at their REST positions (same edge, same parameter): the material point each is."""
    d = pos[tets][:, :, 1] - y_cut
    hit = np.nonzero((d.min(1) < 0) & (d.max(1) > 0))[0]
    P = pos[tets[hit]]; D = d[hit]
    a, b = _EDGES[:, 0], _EDGES[:, 1]
    da, db = D[:, a], D[:, b]
    cross = (da < 0) != (db < 0)                                       # (m, 6)
    t = np.where(cross, da / np.where(cross, da - db, 1.0), 0.0)
    pts = P[:, a] + t[..., None] * (P[:, b] - P[:, a])                 # (m, 6, 3)
    if rest is not None:
        R0 = rest[tets[hit]]
        pts = np.concatenate([pts, R0[:, a] + t[..., None] * (R0[:, b] - R0[:, a])], axis=2)    # (m, 6, 6): deformed | rest
    n = cross.sum(1)
    assert set(np.unique(n)) <= {3, 4}, f"slice polygon sizes {np.unique(n)}"
    # order the crossing points round their centroid in the plane (x, z)
    c = (pts * cross[..., None]).sum(1) / n[:, None]
    ang = np.arctan2(pts[..., 2] - c[:, None, 2], pts[..., 0] - c[:, None, 0])                 # ordered by the DEFORMED point
    ang = np.where(cross, ang, np.inf)
    order = np.argsort(ang, axis=1)[:, :4]
    poly = np.take_along_axis(pts, order[..., None], axis=1)
    tris, src = _fan(poly, n)
    if rest is not None:
        return tris[..., :3], hit[src], tris[..., 3:]
    return tris, hit[src]


def clip_tris(pos, faces, y_cut, keep_below=True):
    """Triangles of `faces` clipped to the kept half-space. Returns (tris (t,3,3), face index (t,))."""
    s = 1.0 if keep_below else -1.0
    d = s * (pos[faces][:, :, 1] - y_cut)                              # kept where d <= 0
    P = pos[faces]
    inside = d <= 0
    k = inside.sum(1)
    out_t, out_i = [P[k == 3]], [np.nonzero(k == 3)[0]]
    for nin in (1, 2):
        idx = np.nonzero(k == nin)[0]
        if not len(idx):
            continue
        Pi, Di, In = P[idx], d[idx], inside[idx]
        # rotate each triangle so the lone vertex (inside if nin == 1, outside if nin == 2) comes first; keeps winding
        lone = np.argmax(In if nin == 1 else ~In, axis=1)
        rot = (lone[:, None] + np.arange(3)) % 3
        Pi = np.take_along_axis(Pi, rot[..., None], axis=1); Di = np.take_along_axis(Di, rot, axis=1)
        def at(i, j):
            t = Di[:, i] / (Di[:, i] - Di[:, j])
            return Pi[:, i] + t[:, None] * (Pi[:, j] - Pi[:, i])
        p01, p02 = at(0, 1), at(0, 2)
        if nin == 1:
            out_t.append(np.stack([Pi[:, 0], p01, p02], 1)); out_i.append(idx)
        else:
            out_t += [np.stack([p01, Pi[:, 1], Pi[:, 2]], 1), np.stack([p01, Pi[:, 2], p02], 1)]; out_i += [idx, idx]
    return np.concatenate(out_t), np.concatenate(out_i)


class CutView:
    """Precomputes the boundary once; build(pos) returns {tissue: (points (n,3) float32, indices (n,) int)} per frame."""

    def __init__(self, tets, tet_body, y_cut=0.0, keep_below=True, rest=None, texture=None):
        """texture: dict(x0_m, z_top_m, width_m) from analysis/cut_section_texture.py. With it (and `rest`), the cut face is
        returned as one textured part, '_cutface' -> (points, uv), coloured by the V4 section at each point's rest (x, z)
        instead of by element. ILLUSTRATION ONLY: the solver's tissue assignment is per element."""
        self.rest = None if rest is None else np.asarray(rest, dtype=np.float64); self.texture = texture
        assert (rest is None) == (texture is None), "texture mode needs both rest positions and the texture extent"
        self.tets = np.asarray(tets, dtype=np.int64); self.body = np.asarray(tet_body).astype(str)
        self.y_cut = float(y_cut); self.keep_below = keep_below
        self.faces, self.owner = boundary_faces(self.tets)
        self.names = sorted(set(self.body))
        unknown = set(self.names) - set(COLOR)
        assert not unknown, f"no colour for tissue(s) {unknown}"

    def build(self, pos):
        bt, bi = clip_tris(pos, self.faces, self.y_cut, self.keep_below)
        if self.texture is not None:
            ct, ci, cr = slice_tets(pos, self.tets, self.y_cut, rest=self.rest)
            tx = self.texture
            uv = np.stack([(cr[..., 0] - tx["x0_m"]) / tx["width_m"], (cr[..., 2] - (tx["z_top_m"] - tx["width_m"])) / tx["width_m"]], -1)
            out = {"_cutface": (ct.reshape(-1, 3).astype(np.float32), uv.reshape(-1, 2).astype(np.float32))}
            body = self.body[self.owner[bi]]
            for name in self.names:
                m = body == name
                if m.any():
                    out[name] = bt[m].reshape(-1, 3).astype(np.float32)
            return out
        ct, ci = slice_tets(pos, self.tets, self.y_cut)
        tris = np.concatenate([ct, bt]); body = np.concatenate([self.body[ci], self.body[self.owner[bi]]])
        out = {}
        for name in self.names:
            m = body == name
            if m.any():
                out[name] = tris[m].reshape(-1, 3).astype(np.float32)
        return out
