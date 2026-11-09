"""Closure read as the plane-slice area of a lumen void, from its oriented boundary triangles. numpy only.

Used by anatomy_meshing/graded_leg_mesh.py (physics venv) at rest and by isaaclab/live_press_bmode.py (Isaac Lab venv)
every frame, so both sides run ONE operator (defect class 1: two artifacts and nothing reconciling them).

The boundary triangles come from the tet mesh: faces of kept tets that were shared with removed (lumen) tets, wound
outward from the kept tet, i.e. their normal points INTO the void. Cutting them with the plane y = y0 gives one segment
per crossing triangle. With the segment direction fixed as d = n x y_hat (n the triangle normal) every segment of a
closed loop runs the same way round, so the shoelace sum over unordered segments is the loop's signed area; the
absolute value is returned. A jagged (staircase) boundary may cut into several loops; their areas add.

⚠️ Floor: opposing lumen walls are stopped by particle self-contact at 0.8 * dx, so the smallest area this can read
for a flattened lumen is about (0.8 * dx) * (major axis). Report it next to the mesh's dx.
"""
import numpy as np


def slice_area(pos: np.ndarray, tris: np.ndarray, y0: float = 0.0) -> float:
    """Area (m^2) of the lumen cross-section at plane y = y0. `pos` (n, 3) node positions, `tris` (m, 3) oriented."""
    P = pos[tris]                                             # (m, 3, 3)
    d = P[:, :, 1] - y0                                       # signed distance of each vertex to the plane
    s = np.sign(d)
    cross = ~((s > 0).all(1) | (s < 0).all(1))
    if not cross.any():
        return 0.0
    P, d = P[cross], d[cross]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])        # triangle normals (toward the void)
    dirv = np.cross(n, np.array([0.0, 1.0, 0.0]))             # in-plane direction consistent with the winding
    pts = np.full((len(P), 2, 3), np.nan)
    cnt = np.zeros(len(P), dtype=int)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        di, dj = d[:, i], d[:, j]
        hit = (di * dj < 0) | ((di == 0) & (dj != 0)) | ((dj == 0) & (di != 0) & (cnt == 0))
        t = np.where(di != dj, di / np.where(di != dj, di - dj, 1.0), 0.0)
        q = P[:, i] + t[:, None] * (P[:, j] - P[:, i])
        for k in range(2):
            put = hit & (cnt == k)
            pts[put, k] = q[put]
        cnt = cnt + hit.astype(int)
    ok = cnt >= 2
    a, b = pts[ok, 0], pts[ok, 1]
    swap = np.einsum("ij,ij->i", b - a, dirv[ok]) < 0
    a2 = np.where(swap[:, None], b, a); b2 = np.where(swap[:, None], a, b)
    signed = a2[:, 0] * b2[:, 2] - b2[:, 0] * a2[:, 2]        # shoelace in (x, z)
    return 0.5 * abs(float(signed.sum()))


def slice_extent(pos: np.ndarray, tris: np.ndarray, y0: float = 0.0):
    """(x_min, x_max, z_min, z_max) of the cut, m, for the aspect-ratio reading. NaNs if no crossing."""
    P = pos[tris]; d = P[:, :, 1] - y0; s = np.sign(d)
    cross = ~((s > 0).all(1) | (s < 0).all(1))
    if not cross.any():
        return (np.nan,) * 4
    P, d = P[cross], d[cross]
    pts = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        di, dj = d[:, i], d[:, j]
        hit = di * dj < 0
        t = di[hit] / (di[hit] - dj[hit])
        pts.append(P[hit, i] + t[:, None] * (P[hit, j] - P[hit, i]))
    q = np.concatenate(pts)
    return float(q[:, 0].min()), float(q[:, 0].max()), float(q[:, 2].min()), float(q[:, 2].max())


if __name__ == "__main__":
    # self-test that can fail: a unit cylinder's boundary (normals inward = toward the void), sliced at mid-height
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False); r = 0.005
    ring0 = np.column_stack([r * np.cos(th), np.full_like(th, -0.01), r * np.sin(th)])
    ring1 = np.column_stack([r * np.cos(th), np.full_like(th, +0.01), r * np.sin(th)])
    pos = np.vstack([ring0, ring1]); m = len(th)
    tris = []
    for i in range(m):
        j = (i + 1) % m
        tris.append((i, j, m + i)); tris.append((j, m + j, m + i))
    tris = np.array(tris)
    # orient toward the axis (into the void)
    P = pos[tris]; n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); c = P.mean(1)
    flip = np.einsum("ij,ij->i", n, -c * np.array([1, 0, 1])) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    A = slice_area(pos, tris, 0.0)
    print(f"cylinder r={r*1e3:.1f} mm: slice area {A*1e6:.3f} mm^2 vs pi r^2 = {np.pi*r*r*1e6:.3f}")
    assert abs(A - np.pi * r * r) / (np.pi * r * r) < 2e-3
    # and with the opposite (outward) orientation the magnitude must be identical
    A2 = slice_area(pos, tris[:, [0, 2, 1]], 0.0); assert abs(A2 - A) < 1e-12
    print("closure_ops self-test OK")
