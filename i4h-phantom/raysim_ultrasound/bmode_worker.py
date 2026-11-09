"""Live B-mode worker: raysim in the `i4h_raysim:live` container, fed FEM node positions per frame.

    docker run --rm --gpus all --network host -e PYTHONUNBUFFERED=1 \
        -e PROBE_THETA_DEG=0 -e STATION_S_MM=20 \
        -v $PWD:/w:ro -v $PWD/out/live_render_rest/travel_0.0mm:/rest:ro -v $PWD/out/live_bmode:/out \
        --entrypoint bash i4h_raysim:live -lc "python /w/raysim_ultrasound/bmode_worker.py --port 5557"

What it does, once:
  * builds the raysim World from the REST render meshes (the 2026-09-08 chain's own tooling at zero
    displacement: deform_meshes.py + gel_body.py -> out/live_render_rest/travel_0.0mm), creates the
    simulator, reads back every body's Assimp-ordered vertices/faces via the 2026-09-09 patch
  * on the client's handshake (FEM rest nodes, FEM frame, metres) computes per-vertex interpolation
    weights with the SAME formula as deform_meshes.py: k nearest FEM nodes, w = exp(-(d/sigma)^2),
    confidence clip(sum w / 4, 0, 1). Brute-force k-NN in cupy, cached to /out by content hash.
Per frame (request = FEM node positions + probe depth):
  * u = pos - rest (FEM m) -> per body u_v = conf * sum(w u[idx]) / sum(w) -> LPS mm -> set_vertices
  * gel film under the aperture measured on the deformed SoftTissue, initial_material chosen
    CONDITIONALLY exactly as render_deformed.py does (coupled -> gel material, else background)
  * rebuild_geometry(), simulate(), 500x500 uint8 back to the client with per-stage timings

Protocol (stdlib TCP, one connection): [u32 header_len][json header][raw bytes].
  handshake: {"type":"hello", n, sigma_mm, k, ...} + rest float32 (n,3)
  frame:     {"type":"frame", frame, depth_mm, sim_t} + pos float32 (n,3)
  reply:     {"type":"image", timings..., film_mm, coupled, r_face_mm, ...} + uint8 (H,W)

🔴 Modelling choices carried from the chain, unchanged: blend sigma (render-only), GEL_MIN 0.3 mm,
the aperture half-width 8 mm for the film measurement, and that the Gel body is warped with the
tissue field like every other body (the chain rebuilt it per state instead).
"""
import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import time

import numpy as np
import cupy as cp
import cupyx
import raysim.cuda as rs

sys.path.insert(0, "/w")
import probe_geometry as PG  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rest-dir", default="/rest")
ap.add_argument("--out", default="/out")
ap.add_argument("--port", type=int, default=5557)
ap.add_argument("--update-normals", action="store_true", help="recompute vertex normals from the deformed faces each frame")
ap.add_argument("--knn-chunk", type=int, default=2048)
ap.add_argument("--rigid", default="Femur,PubicRamus",
                help="bodies NOT re-uploaded per frame (bone, treated as rigid in the RENDER only). 🔴 render approximation; "
                     "the worker reports the displacement it discards as rigid_max_mm so the choice is measured, not assumed. Empty = none.")
args = ap.parse_args()

PROBE_THETA_DEG = float(os.environ["PROBE_THETA_DEG"])
STATION_S_MM = float(os.environ["STATION_S_MM"])
GEL_MIN = 0.3
APERTURE_HALF = 8.0
D2R = np.pi / 180.0


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ scene, built once
cfg_path = os.path.join(args.rest_dir, "anatomy_config.json")
cfg_all = json.load(open(cfg_path))
cfg = cfg_all["anatomy"]
gel = PG.gel_from_config(cfg_all, where=cfg_path)
R_SKIN = float(gel["skin_r_mm"]); STANDOFF = float(gel["probe_standoff_mm"])
GEL_MAT = gel.get("material", "water")
say(f"[gel] skin r {R_SKIN:.2f} mm, standoff {STANDOFF:.2f}, band {gel['thickness_mm']:.2f} mm, gel material {GEL_MAT!r}")

missing = [m["file"] for m in cfg["meshes"] if not os.path.exists(os.path.join(args.rest_dir, m["file"]))]
if missing:
    raise SystemExit(f"!!! rest dir lists meshes that are not on disk: {missing}. Refusing a partial anatomy.")

t0 = time.perf_counter()
mats = rs.Materials()
world = rs.World(cfg["background_material"])
meshes = {}
for m in cfg["meshes"]:
    mesh = rs.Mesh(os.path.join(args.rest_dir, m["file"]), mats.get_index(m["material"]))
    world.add(mesh)
    meshes[m["file"][:-4]] = mesh
sim = rs.RaytracingUltrasoundSimulator(world, mats)
names = list(meshes)
V0 = {n: meshes[n].get_vertices().astype(np.float32) for n in names}          # LPS mm, Assimp order
F = {n: meshes[n].get_indices().astype(np.int64) for n in names}
say(f"[scene] {len(names)} bodies, {sum(len(v) for v in V0.values()):,} vertices, "
    f"{sum(len(f) for f in F.values()):,} triangles, built in {time.perf_counter()-t0:.1f} s; background {cfg['background_material']!r}")
V0_gpu = {n: cp.asarray(V0[n]) for n in names}
F_gpu = {n: cp.asarray(F[n]) for n in names}
rest_sha_scene = hashlib.sha256(b"".join(V0[n].tobytes() for n in names)).hexdigest()[:12]

sp = rs.SimParams()
sp.conv_psf = True; sp.buffer_size = 4096; sp.t_far = 80.0; sp.b_mode_size = (500, 500)
ang = PROBE_THETA_DEG * D2R
ROT = np.array([-90 * D2R, 0.0, ang], np.float32)
n_in = np.array([np.sin(ang), -np.cos(ang)])
n_lat = np.array([np.cos(ang), np.sin(ang)])
skin_lat = V0["SoftTissue"][:, :2] @ n_lat
skin_under = (np.abs(skin_lat) <= APERTURE_HALF) & (np.abs(V0["SoftTissue"][:, 2]) <= 2.5)
say(f"[pose] theta {PROBE_THETA_DEG:+.2f} deg, station s = {STATION_S_MM:g} mm; {int(skin_under.sum())} skin vertices under the aperture")


def lps_mm_to_fem_m(V):
    return cp.stack([V[:, 0], V[:, 2], -V[:, 1]], axis=1) * 1e-3


def fem_m_to_lps_mm(U):
    return cp.stack([U[:, 0], -U[:, 2], U[:, 1]], axis=1) * 1e3


# ------------------------------------------------------------------ weights (per client handshake)
W = None


def build_weights(rest_fem_m: np.ndarray, sigma_m: float, k: int):
    """k nearest FEM rest nodes per render vertex; same formula as deform_meshes.py."""
    key = hashlib.sha256(rest_fem_m.tobytes() + f"{sigma_m:.9g}|{k}|{rest_sha_scene}".encode()).hexdigest()[:16]
    cache = os.path.join(args.out, f"weights_{key}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        say(f"[weights] cache hit {cache}")
        return {n: (cp.asarray(z[f"{n}_idx"]), cp.asarray(z[f"{n}_w"]), cp.asarray(z[f"{n}_conf"])) for n in names}
    t0 = time.perf_counter()
    R = cp.asarray(rest_fem_m, dtype=cp.float32)                  # (N, 3)
    R2 = (R * R).sum(1)
    out, save = {}, {}
    for n in names:
        P = lps_mm_to_fem_m(V0_gpu[n]).astype(cp.float32)           # (M, 3)
        M = P.shape[0]
        idx_all = cp.empty((M, k), dtype=cp.int64); d_all = cp.empty((M, k), dtype=cp.float32)
        for s in range(0, M, args.knn_chunk):
            Pc = P[s:s + args.knn_chunk]
            d2 = (Pc * Pc).sum(1)[:, None] + R2[None, :] - 2.0 * Pc @ R.T      # (c, N)
            part = cp.argpartition(d2, k, axis=1)[:, :k]
            dk = cp.take_along_axis(d2, part, axis=1)
            order = cp.argsort(dk, axis=1)
            idx_all[s:s + args.knn_chunk] = cp.take_along_axis(part, order, axis=1)
            d_all[s:s + args.knn_chunk] = cp.sqrt(cp.maximum(cp.take_along_axis(dk, order, axis=1), 0.0))
        w = cp.exp(-(d_all / sigma_m) ** 2)
        wsum = w.sum(1)
        conf = cp.clip(wsum / 4.0, 0.0, 1.0)
        safe = cp.where(wsum > 1e-12, wsum, 1.0)
        w = w / safe[:, None]                                        # normalised weights
        out[n] = (idx_all, w.astype(cp.float32), conf.astype(cp.float32))
        save[f"{n}_idx"] = cp.asnumpy(idx_all); save[f"{n}_w"] = cp.asnumpy(w); save[f"{n}_conf"] = cp.asnumpy(conf)
        say(f"[weights] {n:<13} {M:>9,} verts  moving (conf>0.01): {int((conf > 0.01).sum()):>9,}")
    cp.cuda.Stream.null.synchronize()
    np.savez(cache, **save)
    say(f"[weights] built in {time.perf_counter()-t0:.1f} s -> {cache}")
    return out


def vertex_normals(V, Fi):
    e1 = V[Fi[:, 1]] - V[Fi[:, 0]]; e2 = V[Fi[:, 2]] - V[Fi[:, 0]]
    fn = cp.cross(e1, e2)
    N = cp.zeros_like(V)
    for c in range(3):
        cupyx.scatter_add(N, Fi[:, c], fn)
    nrm = cp.linalg.norm(N, axis=1, keepdims=True)
    return N / cp.maximum(nrm, 1e-12)


# ------------------------------------------------------------------ one frame
def pose_vectors(theta_deg: float):
    """Probe angle round the limb -> rotation, beam and lateral unit vectors, skin vertices under the aperture."""
    a = theta_deg * D2R
    rot = np.array([-90 * D2R, 0.0, a], np.float32)
    nin = np.array([np.sin(a), -np.cos(a)]); nlat = np.array([np.cos(a), np.sin(a)])
    lat = V0["SoftTissue"][:, :2] @ nlat
    under = np.where((np.abs(lat) <= APERTURE_HALF) & (np.abs(V0["SoftTissue"][:, 2]) <= 2.5))[0]
    return a, rot, nin, nlat, cp.asarray(under)


_pose_cache = {}


def render(pos_fem_m: np.ndarray, rest_fem_m_gpu, depth_mm: float, theta_deg: float = PROBE_THETA_DEG):
    t = {}
    pc = time.perf_counter
    key = round(float(theta_deg), 4)
    if key not in _pose_cache:
        _pose_cache[key] = pose_vectors(key)
    ang, ROT, n_in, n_lat, skin_under_gpu = _pose_cache[key]
    t0 = pc()
    U = cp.asarray(pos_fem_m, dtype=cp.float32) - rest_fem_m_gpu     # (N, 3) FEM m
    V_new = {}
    for n in names:
        idx, w, conf = W[n]
        u_v = (w[:, :, None] * U[idx]).sum(1) * conf[:, None]          # (M, 3) FEM m
        V_new[n] = V0_gpu[n] + fem_m_to_lps_mm(u_v)
    cp.cuda.Stream.null.synchronize(); t["warp"] = pc() - t0
    t0 = pc()
    rigid_max = 0.0
    for n in names:
        if n in RIGID:
            rigid_max = max(rigid_max, float(cp.linalg.norm(V_new[n] - V0_gpu[n], axis=1).max()))
            continue
        if DEVICE_PATH:
            Vc = cp.ascontiguousarray(V_new[n], dtype=cp.float32)
            meshes[n].set_vertices_device(int(Vc.data.ptr), int(Vc.shape[0]))          # device->device, no host copy
            if args.update_normals:
                Nc = cp.ascontiguousarray(vertex_normals(V_new[n], F_gpu[n]), dtype=cp.float32)
                meshes[n].set_normals_device(int(Nc.data.ptr), int(Nc.shape[0]))
        else:
            meshes[n].set_vertices(cp.asnumpy(V_new[n]).astype(np.float32))
            if args.update_normals:
                meshes[n].set_normals(cp.asnumpy(vertex_normals(V_new[n], F_gpu[n])).astype(np.float32))
    t["upload"] = pc() - t0
    t0 = pc(); sim.rebuild_geometry(); t["rebuild"] = pc() - t0
    # probe pose and gel film, as render_deformed.py
    t0 = pc()
    drop = max(depth_mm - STANDOFF, 0.0)
    r_face = R_SKIN + STANDOFF - drop
    skin = cp.asnumpy(V_new["SoftTissue"][skin_under_gpu])
    proj = skin[:, :2] @ n_in
    film = None
    if len(proj):
        r_skin_def = float(proj.max()); film = r_face - r_skin_def
        if film < GEL_MIN:
            r_face = r_skin_def + GEL_MIN; film = GEL_MIN
    coupled = film is not None and film > 0.0
    sp.initial_material = GEL_MAT if coupled else cfg["background_material"]
    pos = np.array([r_face * np.sin(ang), -r_face * np.cos(ang), 0.0], np.float32)
    probe = rs.LinearArrayProbe(rs.Pose(position=pos, rotation=ROT), num_elements_x=256, width=40.0, frequency=7.0,
                                elevational_height=5.0, num_el_samples=1, f_num=1.0, pulse_duration=2.0)
    t["pose"] = pc() - t0
    t0 = pc()
    b = sim.simulate(probe, sp); b = np.asarray(b.get() if hasattr(b, "get") else b)
    img = (np.clip((b + 60.0) / 60.0, 0, 1) * 255).astype(np.uint8)
    t["simulate"] = pc() - t0
    t["total"] = sum(t.values())
    return img, dict(film_mm=film, coupled=bool(coupled), r_face_mm=float(r_face), initial_material=sp.initial_material, rigid_max_mm=rigid_max, theta_deg=float(theta_deg),
                     probe_pos_mm=[float(v) for v in pos], fill=float((img > 10).mean()), **{f"t_{k}": v for k, v in t.items()})


skin_under_gpu = cp.asarray(np.where(skin_under)[0])
RIGID = set(x for x in args.rigid.split(",") if x)
DEVICE_PATH = hasattr(rs.Mesh, "set_vertices_device")
say(f"[render] vertex upload path: {'device pointer (D2D)' if DEVICE_PATH else 'host array (D2H+H2D)'}")
say(f"[render] rigid in the render (not re-uploaded): {sorted(RIGID) or None}")


# ------------------------------------------------------------------ protocol
def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("client closed")
        buf += chunk
    return bytes(buf)


def recv_msg(conn):
    hl = struct.unpack("<I", recv_exact(conn, 4))[0]
    hdr = json.loads(recv_exact(conn, hl))
    payload = recv_exact(conn, int(hdr.get("nbytes", 0)))
    return hdr, payload


def send_msg(conn, hdr: dict, payload: bytes = b""):
    hdr = dict(hdr, nbytes=len(payload))
    h = json.dumps(hdr).encode()
    conn.sendall(struct.pack("<I", len(h)) + h + payload)


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", args.port)); srv.listen(1)
say(f"[worker] READY on 127.0.0.1:{args.port}  (rest set {args.rest_dir}, scene sha {rest_sha_scene})")
while True:
    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    say(f"[worker] client {addr}")
    rest_gpu = None
    try:
        while True:
            hdr, payload = recv_msg(conn)
            if hdr["type"] == "hello":
                rest = np.frombuffer(payload, dtype=np.float32).reshape(-1, 3)
                rest_gpu = cp.asarray(rest)
                W = build_weights(rest, float(hdr["sigma_mm"]) * 1e-3, int(hdr["k"]))
                # warm-up: cupy compiles its kernels on first use (~1 s); do it here, not on the client's first frame
                t_w = time.perf_counter(); render(rest, rest_gpu, 0.0); say(f"[worker] warm-up render {time.perf_counter()-t_w:.2f} s")
                send_msg(conn, dict(type="ready", n_bodies=len(names), n_vertices=int(sum(len(v) for v in V0.values())),
                                    n_triangles=int(sum(len(f) for f in F.values())), scene_sha=rest_sha_scene,
                                    r_skin_mm=R_SKIN, standoff_mm=STANDOFF, gel_material=GEL_MAT,
                                    background=cfg["background_material"], b_mode_size=[500, 500], t_far_mm=80.0))
            elif hdr["type"] == "frame":
                t_in = time.perf_counter()
                pos = np.frombuffer(payload, dtype=np.float32).reshape(-1, 3)
                img, info = render(pos, rest_gpu, float(hdr["depth_mm"]), float(hdr.get("theta_deg", PROBE_THETA_DEG)))
                info.update(type="image", frame=hdr["frame"], depth_mm=hdr["depth_mm"], sim_t=hdr.get("sim_t"),
                            t_worker=time.perf_counter() - t_in, shape=list(img.shape))
                send_msg(conn, info, img.tobytes())
            elif hdr["type"] == "bye":
                send_msg(conn, dict(type="bye")); break
    except ConnectionError as e:
        say(f"[worker] {e}")
    finally:
        conn.close()
        say("[worker] client gone; waiting for the next one")
