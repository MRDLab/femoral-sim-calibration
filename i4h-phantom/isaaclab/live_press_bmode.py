"""The coupled live loop: Isaac Lab + Newton VBD press at 3 x 35 (CUDA graph on) with a live raysim B-mode.

Every simulated frame: step the FEM, read the node positions (0.2 ms), send them to bmode_worker.py
(container `i4h_raysim:live`, TCP on localhost), receive the 500x500 B-mode, record where the wall
clock went (FEM step / transfer / worker warp+upload+rebuild+simulate). The press follows the
2026-09-07 protocol (ramp 0.4 mm/frame to each depth, settle test 30 frames within 2 %) so the closure
numbers can be compared with the 3 x 35 timeline run of the same day.

Setup mirrors press_femoral_punch.py / profile_press_frame.py. Frames are saved as uint8 npz:
every --save-every-th frame plus the settled frame at each depth.

Usage (Isaac Lab venv), with the worker already listening:
  python -u live_press_bmode.py --headless --npz M.npz --usd M.usda --substeps 3 --iterations 35 --cuda-graph \
      --worker-port 5557 --out out/live_bmode/live_s3_i35.json
"""
import argparse
import hashlib
import json
import os
import pathlib
import socket
import struct
import sys
import time
from collections import deque

import numpy as np
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--npz", required=True)
ap.add_argument("--substeps", type=int, default=3)
ap.add_argument("--iterations", type=int, default=35)
ap.add_argument("--density", type=float, default=1000.0)
ap.add_argument("--ke", type=float, default=2.0e3)
ap.add_argument("--self-contact-dx-mm", type=float, default=None,
                help="length the self-contact radius (0.4x) and margin (0.5x) scale from; default the mesh's fine dx")
ap.add_argument("--no-self-contact", action="store_true",
                help="2026-10-29, cast diagnostic: self-contact radius is 0.4 x the mesh's fine dx (0.6 mm at 1.5 mm); a mesh "
                     "refined below that (0.35 mm tube walls) is everywhere 'in contact' with itself. Only for lumens that stay open.")
ap.add_argument("--ramp-mm-per-frame", type=float, default=0.4)
ap.add_argument("--depths", default="3,7,11,15,19,23,25.4")
ap.add_argument("--cuda-graph", action="store_true")
ap.add_argument("--no-worker", action="store_true", help="FEM only: no B-mode worker, no images. For reference/production runs where only the numbers matter.")
ap.add_argument("--worker-host", default="127.0.0.1")
ap.add_argument("--worker-port", type=int, default=5557)
ap.add_argument("--blend-sigma-mm", type=float, default=12.0, help="render-only, as the 2026-09-08 chain")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--save-every", type=int, default=5)
ap.add_argument("--sweep-deg", type=float, default=0.0, help="sweep the probe angle -X..+X..0 round the limb BEFORE the press (1 deg/frame). "
                "Open-loop over rest tissue, like the 2026-09-08 chain: the punch hovers 2 mm off the skin, the FEM keeps stepping.")
ap.add_argument("--sweep-step-deg", type=float, default=1.0)
ap.add_argument("--pipeline", action="store_true",
                help="overlap: the B-mode of frame i is received AFTER frame i+1's FEM step has been issued, so the image "
                     "lags the tissue by one simulated frame (16.7 ms) and the loop runs at max(FEM, worker) instead of the sum.")
ap.add_argument("--viewer", action="store_true", help="render the Isaac Sim view (RTX camera) from the same loop and save it per frame")
ap.add_argument("--viewer-every", type=int, default=1, help="render the Isaac Sim view every N simulated frames (the FEM and B-mode run every frame)")
ap.add_argument("--viewer-size", default="640x480")
ap.add_argument("--leg-visual", default="out/femoral_v4_gap12/leg_visual_s020",
                help="render-only V4 leg with the slot for the block, from anatomy_meshing/leg_visual_slot.py (2026-09-09). "
                     "Missing directory = refusal, not a slab-only render.")
ap.add_argument("--leg-usd", default="out/usd/femoral_leg_v4.usda", help="the V4 leg USD the 2026-08-25 room render used; its interior bodies and materials are referenced as-is")
ap.add_argument("--no-leg-visual", action="store_true", help="reproduce the 2026-09-09 §11h slab-only view")
ap.add_argument("--allow-retired-block", action="store_true", help="press a retired block/slab mesh (historical reproduction ONLY; the leg is the domain since 2026-09-09)")
ap.add_argument("--out", required=True)
ap.add_argument("--viewer-eye", default=None, help="camera position x,y,z in m (FEM frame: +y superior, +z anterior); overrides the default framing. Render only.")
ap.add_argument("--viewer-target", default=None, help="camera look-at point x,y,z in m; with --viewer-eye. Render only.")
ap.add_argument("--viewer-cut-y", type=float, default=None,
                help="2026-10-12: draw the leg cut open at the plane y = this (m) instead of the closed deformable surface: the cut face "
                     "coloured by tissue, the boundary (skin, far cap, lumen walls) on the kept side. Rebuilt from the node positions every "
                     "viewer frame (isaaclab/cut_view.py). Render only; adds its build time to viewer_s. y = 0 is the B-mode plane.")
ap.add_argument("--viewer-cut-texture", default=None,
                help="2026-10-12: directory from analysis/cut_section_texture.py. Colours the cut face from the V4 section at each point's rest "
                     "position (smooth outlines) instead of by element. ILLUSTRATION ONLY, for paper and grant figures.")
ap.add_argument("--viewer-cut-keep", choices=("below", "above"), default="below", help="which side of the cut plane is drawn: below = inferior (-y)")
ap.add_argument("--save-positions", default=None,
                help="2026-10-09: save node positions at each settled depth, in the MESH FILE's node order (npz), for the "
                     "element-inversion audit (paper/audit_leg_inversion.py). Off by default.")
AppLauncher.add_app_launcher_args(ap)
args_cli = ap.parse_args()
if args_cli.viewer:
    args_cli.enable_cameras = True          # render products need this

T_WALL0 = time.perf_counter()
app = AppLauncher(args_cli).app

import torch  # noqa: E402
import warp as wp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_newton.physics import NewtonCfg, NewtonManager  # noqa: E402
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg  # noqa: E402
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg  # noqa: E402


@configclass
class VBDSolverCfgBigContact(VBDSolverCfg):
    rigid_body_particle_contact_buffer_size: int = 8192


def say(*a):
    print(*a, flush=True)


def poly_area(pts_xz):
    x, z = pts_xz[:, 0], pts_xz[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))


# ------------------------------------------------------------------ worker connection
def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("worker closed")
        buf += chunk
    return bytes(buf)


def recv_msg(conn):
    hl = struct.unpack("<I", recv_exact(conn, 4))[0]
    hdr = json.loads(recv_exact(conn, hl))
    return hdr, recv_exact(conn, int(hdr.get("nbytes", 0)))


def send_msg(conn, hdr, payload=b""):
    h = json.dumps(dict(hdr, nbytes=len(payload))).encode()
    conn.sendall(struct.pack("<I", len(h)) + h + payload)


conn = None
if not args_cli.no_worker:
    conn = socket.create_connection((args_cli.worker_host, args_cli.worker_port), timeout=600)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

# ------------------------------------------------------------------ setup (as press_femoral_punch.py)
d = np.load(args_cli.npz, allow_pickle=True)
mesh_sha = hashlib.sha256(open(args_cli.npz, "rb").read()).hexdigest()[:12]
our_rest = d["rest"].astype(np.float64); our_tets = d["tets"].astype(np.int64); our_mat = d["tet_materials"].astype(np.float64)
pinned_ids = d["pinned"].astype(np.int64); footprint = d["footprint"].astype(np.int64)
ring_names = [str(n) for n in d["ring_names"]]; rings = {n: d[f"ring_{n}"].astype(np.int64) for n in ring_names}
dx = float(d["dx"][0]); frame_dt = 1.0 / 60.0

# 🚫 THE BLOCK IS RETIRED (2026-09-09 evening): "Discontinue the slab from any future work. We do our future work only
# on the leg." Every mesh pressed from now on must be a graded whole-leg mesh (meta kind "graded_leg"). A block mesh is
# refused unless --allow-retired-block is passed, which exists only to reproduce a historical result; say why in the log.
_meta_txt = str(d["meta"][0]) if "meta" in d.files else ""
if '"kind": "graded_leg"' not in _meta_txt and not args_cli.allow_retired_block:
    raise SystemExit(f"!!! {args_cli.npz} is not a graded whole-leg mesh. The block/slab was RETIRED on 2026-09-09; future work is on the "
                     f"leg only (photos/2026-09-10). Pass --allow-retired-block ONLY to reproduce a historical block result, and record why.")
say(f">>> mesh {args_cli.npz} sha256 {mesh_sha}; {args_cli.substeps} x {args_cli.iterations}, graph {'ON' if args_cli.cuda_graph else 'OFF'}")
# the length the self-contact radius and margin scale from: the mesh's fine dx unless set (2026-10-29, cast walls)
SC_DX = dx if args_cli.self_contact_dx_mm is None else args_cli.self_contact_dx_mm * 1e-3
say(f">>> self-contact {'OFF' if args_cli.no_self_contact else 'ON'}: radius {0.4*SC_DX*1e3:.3f} mm, margin {0.5*SC_DX*1e3:.3f} mm "
    f"(scaled from {'the mesh dx' if args_cli.self_contact_dx_mm is None else '--self-contact-dx-mm'} = {SC_DX*1e3:.3f} mm)")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    dt=frame_dt, device=args_cli.device, gravity=(0.0, 0.0, 0.0),
    physics=NewtonCfg(solver_cfg=VBDSolverCfgBigContact(
        iterations=args_cli.iterations, particle_enable_self_contact=not args_cli.no_self_contact,
        particle_self_contact_radius=0.4 * SC_DX, particle_self_contact_margin=0.5 * SC_DX),
        num_substeps=args_cli.substeps, use_cuda_graph=args_cli.cuda_graph)))
sim_utils.create_prim("/World/env_0", "Xform", translation=(0.0, 0.0, 0.0))
obj = DeformableObject(cfg=DeformableObjectCfg(
    prim_path="/World/env_0/Phantom",
    spawn=sim_utils.UsdFileCfg(usd_path=args_cli.usd, deformable_props=NewtonDeformableBodyPropertiesCfg(),
                               physics_material=NewtonDeformableBodyMaterialCfg(
                                   k_mu=float(our_mat[:, 0].mean()), k_lambda=float(our_mat[:, 1].mean()),
                                   density=args_cli.density, k_damp=0.0)),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)), debug_vis=False))
x_probe = float(d["x_probe"][0]); z_skin = float(d["z_skin"][0]); probe_w = float(d["probe_w"][0]); probe_len = float(d["probe_len"][0])
PUNCH_T = 0.004; STANDOFF = 0.002
punch = RigidObject(cfg=RigidObjectCfg(
    prim_path="/World/env_0/Punch",
    spawn=sim_utils.CuboidCfg(size=(probe_w, probe_len, PUNCH_T), rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                              collision_props=sim_utils.CollisionPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=0.1)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(x_probe, 0.0, z_skin + STANDOFF + PUNCH_T / 2.0))))
cam = None
BODY_H = 0.060                           # visual probe body: 60 mm tall handle standing on the 4 mm plate. RENDER ONLY.
if args_cli.viewer:
    from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
    from pxr import Gf, UsdGeom  # noqa: E402
    vw, vh = (int(v) for v in args_cli.viewer_size.split("x"))
    # item 39: the 2026-09-08 viewer looked face-on at a flat-lit slab and washed out. Oblique view + a
    # directional light for shading + a probe body tall enough to stay visible when the plate is in its notch.
    # Lighting as the 2026-08-25 room render (isaac_sim_scene/sweep_room_probe.py:498): dome 900 tinted + distant 1400.
    # The 2000 + 2500 of the 2026-09-09 §11h view rendered tan skin (0.97, 0.83, 0.55) near white (frame std ~10).
    sim_utils.create_prim("/World/Light", "DomeLight", attributes={"inputs:intensity": 900.0, "inputs:color": (0.62, 0.64, 0.70)})
    sim_utils.create_prim("/World/Sun", "DistantLight", orientation=(0.9239, 0.3827, 0.0, 0.0),
                          attributes={"inputs:intensity": 1400.0, "inputs:angle": 2.0})
    # One probe, render-only: the 4 mm plate and the 60 mm body are children of a single Xform moved as a unit, so they
    # cannot separate. 2026-09-09 §11h drew the body alone at the commanded pose while the visible plate was the physics
    # collider, which stays at theta 0 -- the "decapitated probe" of that video. The physics plate is now hidden from the
    # camera (visibility is render-only; its collider is untouched).
    sim_utils.create_prim("/World/Probe", "Xform", translation=(x_probe, 0.0, z_skin + STANDOFF), orientation=(1.0, 0.0, 0.0, 0.0))
    sim_utils.create_prim("/World/Probe/Plate", "Cube", translation=(0.0, 0.0, PUNCH_T / 2), scale=(probe_w / 2, probe_len / 2, PUNCH_T / 2),
                          attributes={"primvars:displayColor": [(0.75, 0.75, 0.78)]})
    sim_utils.create_prim("/World/Probe/Body", "Cube", translation=(0.0, 0.0, PUNCH_T + BODY_H / 2), scale=(probe_w / 2, probe_len / 2, BODY_H / 2),
                          attributes={"primvars:displayColor": [(0.15, 0.15, 0.18)]})
    UsdGeom.Imageable(sim.stage.GetPrimAtPath("/World/env_0/Punch")).MakeInvisible()
    cam = Camera(cfg=CameraCfg(prim_path="/World/Cam", update_period=0.0, height=vh, width=vw, data_types=["rgb"],
                               spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.01, 10.0))))
    # ---- the V4 leg, render-only (2026-09-09, by direction: "our V4 leg in its full rendering in the Isaac Lab stage").
    # Until today this stage held the FEM block alone, so the room view was a slab in space and a probe flying round a
    # limb that was not there (photos/2026-09-09 §11h). The leg USD's frame (LPS -> world Rx(-90 deg), mm -> m baked in)
    # IS the FEM frame (X = LPS x, Y = LPS z, Z = -LPS y), so it is referenced at identity. Its skin and muscle are hidden
    # and replaced by the slotted copies leg_visual_slot.py cut from the SAME OBJs, because the skin coincides with the
    # block's top face and a pressed block would sink inside an opaque leg. The interior bodies, materials and end caps
    # come from the USD unchanged. Every PhysicsCollisionAPI on the reference is removed before sim.reset(): nothing
    # here may touch the solver.
    leg_manifest = None
    if not args_cli.no_leg_visual:
        from pxr import Sdf, Usd, UsdPhysics, UsdShade  # noqa: E402
        LEGV = pathlib.Path(args_cli.leg_visual); LEGUSD = pathlib.Path(args_cli.leg_usd)
        if not (LEGV / "leg_visual.json").exists() or not LEGUSD.exists():
            raise SystemExit(f"!!! leg visual missing: {LEGV}/leg_visual.json or {LEGUSD}. Run anatomy_meshing/leg_visual_slot.py "
                             f"(anatomy venv) or pass --no-leg-visual for the slab-only view. Not rendering a slab and calling it a leg.")
        leg_manifest = json.load(open(LEGV / "leg_visual.json"))
        rep_json = json.load(open(LEGUSD.with_name(LEGUSD.stem + "_report.json")))
        assert rep_json["frame"].startswith("LPS->world Rx(-90deg)"), rep_json["frame"]
        assert rep_json["obj_dir"].rstrip("/").endswith(str(d["anatomy_dir"][0])), (rep_json["obj_dir"], d["anatomy_dir"][0])
        leg_ref = sim.stage.DefinePrim("/World/Leg", "Xform")
        leg_ref.GetReferences().AddReference(str(LEGUSD.resolve()))
        n_coll = 0
        for prim in Usd.PrimRange(leg_ref):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI); n_coll += 1
            if prim.GetName() in ("SoftTissue", "Muscle") and prim.GetTypeName() == "Mesh":
                UsdGeom.Imageable(prim).MakeInvisible()
        assert n_coll > 0, "expected PhysicsCollisionAPI on the leg reference and found none -- is this the right USD?"
        assert all(not p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(leg_ref)), "collision API survived removal"
        # the slotted skin / muscle / filler, in the FEM frame already, coloured as the USD's own materials
        def _bind(mesh_prim, mat_path):
            mat = UsdShade.Material(sim.stage.GetPrimAtPath(mat_path)); assert mat, mat_path
            UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(mat)
        MAT_OF = {"SoftTissue": "/World/Leg/Looks/SoftTissue_mat", "Muscle": "/World/Leg/Looks/Muscle_mat", "Filler": "/World/Leg/Looks/SoftTissue_mat"}
        for part, info in leg_manifest["parts"].items():
            V, Fc = [], []
            for line in open(LEGV / info["file"]):
                if line.startswith("v "):
                    V.append([float(t) for t in line.split()[1:4]])
                elif line.startswith("f "):
                    Fc.append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
            V = np.asarray(V); assert len(V) == info["vertices"] and len(Fc) == info["faces"], part
            mesh = UsdGeom.Mesh.Define(sim.stage, f"/World/LegSlot/{part}")
            mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in V])
            mesh.CreateFaceVertexCountsAttr([len(f) for f in Fc]); mesh.CreateFaceVertexIndicesAttr([i for f in Fc for i in f])
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateExtentAttr([Gf.Vec3f(*V.min(0)), Gf.Vec3f(*V.max(0))])
            _bind(mesh.GetPrim(), MAT_OF[part.split("_")[0]])
        # the block must be the block this leg was cut for: same mesh, same skin radius, block inside the pocket
        assert leg_manifest["npz_sha256_12"] == mesh_sha, f"leg visual cut for npz {leg_manifest['npz_sha256_12']}, loaded {mesh_sha}"
        _bb = np.array(leg_manifest["block_bbox_m"]); assert np.allclose(_bb[0], our_rest.min(0)) and np.allclose(_bb[1], our_rest.max(0)), "block bbox differs from the manifest"
        assert abs(leg_manifest["z_skin_m"] - z_skin) < 1e-9 and abs(leg_manifest["r_skin_m"] - z_skin) < 0.1e-3
        # a dark table top tangent to the leg (axis at the origin, so top at z = -r_skin), render-only: the limb lies on
        # it as in the room render, and it gives the pale skin something to read against. No collider.
        sim_utils.create_prim("/World/TableTop", "Cube", translation=(0.0, -0.04, -leg_manifest["r_skin_m"] - 0.01),
                              scale=(0.30, 0.22, 0.01), attributes={"primvars:displayColor": [(0.10, 0.10, 0.11)]})
        say(f"  leg: referenced {LEGUSD.name} (removed {n_coll} collision APIs, skin+muscle hidden) + {len(leg_manifest['parts'])} slotted parts from {LEGV}; "
            f"slot y = [{leg_manifest['slot_y_m'][0]*1e3:+.2f}, {leg_manifest['slot_y_m'][1]*1e3:+.2f}] mm, clearance {leg_manifest['clearance_m']*1e3:.1f} mm, render-only")
sim.reset()
if cam is not None:
    # look-at, so no quaternion convention is hand-derived: from the front-right, above, at the skin under the probe
    # further back than the 2026-09-08 viewer (0.20 m): the whole block, its notch and the probe body in one view.
    # ⚠️ The simulated domain is the 72 x 83 x 16.5 mm block in the leg's slot. Everything else the camera sees (the
    # V4 leg either side, the filler ring in the slot, the probe body) is render-only; there is no table.
    # 🪤 Camera.set_world_poses_from_view() is a NO-OP on this Newton build (measured 2026-09-09: pos stays 0,0,0 and
    # the frame is black, the camera being inside the slab). The pose is therefore written to the USD prim in the
    # standard USD camera convention (looks along -Z, +Y up), which needs no Isaac Lab convention at all.
    if args_cli.no_leg_visual and "meta" in d.files and '"kind": "graded_leg"' in str(d["meta"][0]):
        # 2026-09-10: the deformable is the whole leg segment (185 mm across, 120 mm long). Frame it whole, from the
        # front-right, above; a render-only table top under it (axis at the origin, so the top is at z = -r_skin).
        eye = np.array([0.30, -0.42, 0.30]); tgt = np.array([0.0, -0.04, 0.03])
        sim_utils.create_prim("/World/TableTop", "Cube", translation=(0.0, -0.04, -z_skin - 0.01), scale=(0.30, 0.22, 0.01),
                              attributes={"primvars:displayColor": [(0.10, 0.10, 0.11)]})
    elif args_cli.no_leg_visual:
        eye = np.array([0.19, -0.25, 0.20]); tgt = np.array([0.0, 0.0, 0.062])          # the 2026-09-09 §11h framing
    else:
        # the leg is 185 mm across and 120 mm long (y in [-100, +20] mm), the probe's swept reach ~160 mm: from the
        # front-right, above, at ~0.6 m, aimed at the skin over the block. The camera never moves.
        eye = np.array([0.25, -0.34, 0.27]); tgt = np.array([0.0, -0.03, 0.04])
    if args_cli.viewer_eye:
        eye = np.array([float(v) for v in args_cli.viewer_eye.split(",")]); tgt = np.array([float(v) for v in args_cli.viewer_target.split(",")])
    fwd = tgt - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0])); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    Rm = np.column_stack([right, up, -fwd])
    qw = np.sqrt(max(0.0, 1 + Rm[0, 0] + Rm[1, 1] + Rm[2, 2])) / 2
    qx, qy, qz = (Rm[2, 1] - Rm[1, 2]) / (4 * qw), (Rm[0, 2] - Rm[2, 0]) / (4 * qw), (Rm[1, 0] - Rm[0, 1]) / (4 * qw)
    _cp = sim.stage.GetPrimAtPath("/World/Cam"); _ct = _cp.GetAttribute("xformOp:translate"); _co = _cp.GetAttribute("xformOp:orient")
    if _ct.GetTypeName() == "float3":
        _ct.Set(Gf.Vec3f(*eye)); _co.Set(Gf.Quatf(float(qw), Gf.Vec3f(float(qx), float(qy), float(qz))))
    else:
        _ct.Set(Gf.Vec3d(*eye)); _co.Set(Gf.Quatd(float(qw), Gf.Vec3d(float(qx), float(qy), float(qz))))
    say(f"  viewer camera at {eye} looking at {tgt}; quat wxyz {np.round([qw, qx, qy, qz], 4)}")
    body_prim = sim.stage.GetPrimAtPath("/World/Probe")
    _tr = body_prim.GetAttribute("xformOp:translate"); _or = body_prim.GetAttribute("xformOp:orient")


    def move_probe_body(depth_m, theta_deg):
        """Draw the probe (plate + body, one group) at the COMMANDED pose: the group origin is the plate's FACE, at radius
        z_skin - depth from the limb axis, at angle theta round it; the group's +z (plate normal, body axis) points
        radially outward. On the block during the press (theta 0, where it coincides with the physics plate); round the
        limb during the sweep."""
        th = np.radians(theta_deg)
        r = z_skin - depth_m
        pos = (x_probe + r * np.sin(th), 0.0, r * np.cos(th))
        q = (np.cos(th / 2), 0.0, np.sin(th / 2), 0.0)           # rotation about +y by theta (w, x, y, z)
        if _tr.GetTypeName() == "float3":
            _tr.Set(Gf.Vec3f(*pos)); _or.Set(Gf.Quatf(q[0], Gf.Vec3f(q[1], q[2], q[3])))
        else:
            _tr.Set(Gf.Vec3d(*pos)); _or.Set(Gf.Quatd(q[0], Gf.Vec3d(q[1], q[2], q[3])))

rest = obj.data.default_nodal_state_w.torch[0, :, :3].cpu().numpy().astype(np.float64)
model = NewtonManager.get_model()
solver_tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int64)
# node ordering bijection by NEAREST NEIGHBOUR, not by rounded-coordinate keys: the USD stores float32, and for
# unstructured (gmsh) coordinates ~1 % of nodes round to a different 1 um key than their float64 original did.
# The structured block never hit this because its coordinates are exact multiples of dx in float32.
from scipy.spatial import cKDTree  # noqa: E402
_dist, ours_of_solver = cKDTree(our_rest).query(rest)
assert len(rest) == len(our_rest), f"solver has {len(rest):,} nodes, the npz {len(our_rest):,}: the loader changed the mesh"
assert _dist.max() < 1e-6 and len(np.unique(ours_of_solver)) == len(rest), \
    f"node ordering is not a bijection: max match distance {_dist.max():.3e} m, {len(np.unique(ours_of_solver)):,} unique of {len(rest):,}"
say(f"  node bijection OK: {len(rest):,} nodes, max match distance {_dist.max():.2e} m")
solver_of_ours = np.empty(len(rest), dtype=np.int64); solver_of_ours[ours_of_solver] = np.arange(len(rest))
ours_key = {tuple(sorted(t)): k for k, t in enumerate(our_tets.tolist())}
new_mat = model.tet_materials.numpy().copy(); nmatched = 0
our_body = d["tet_body"].astype(str) if "tet_body" in d.files else np.full(len(our_tets), "SoftTissue")
solver_body = np.full(len(solver_tets), "", dtype=our_body.dtype)
for si, st in enumerate(solver_tets):
    k = ours_key.get(tuple(sorted(ours_of_solver[st].tolist())))
    if k is not None:
        new_mat[si, :3] = our_mat[k, :3]; nmatched += 1; solver_body[si] = our_body[k]
assert nmatched == len(solver_tets), "material match incomplete"
cut = None
if cam is not None and args_cli.viewer_cut_y is not None:
    # render-only cut-away: hide the deformable's closed surface, draw per-tissue meshes rebuilt from positions each viewer frame
    from cut_view import COLOR, CutView  # noqa: E402
    cut_tex = None
    if args_cli.viewer_cut_texture:
        cut_tex = json.load(open(pathlib.Path(args_cli.viewer_cut_texture) / "cut_texture.json"))
        assert cut_tex["mesh_sha256_12"] == mesh_sha, f"cut texture made for mesh {cut_tex['mesh_sha256_12']}, loaded {mesh_sha}"
    cut = CutView(solver_tets, solver_body, args_cli.viewer_cut_y, keep_below=(args_cli.viewer_cut_keep == "below"),
                  rest=(rest if cut_tex else None), texture=cut_tex)
    UsdGeom.Imageable(sim.stage.GetPrimAtPath("/World/env_0/Phantom")).MakeInvisible()
    cut_mesh = {}
    for _n in cut.names:
        _m = UsdGeom.Mesh.Define(sim.stage, f"/World/Cut/{_n}")
        _m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none); _m.CreateDoubleSidedAttr(True)
        _m.CreateDisplayColorAttr([Gf.Vec3f(*COLOR[_n])])
        cut_mesh[_n] = _m
    if cut_tex is not None:
        from pxr import Sdf  # noqa: E402
        _m = UsdGeom.Mesh.Define(sim.stage, "/World/Cut/_cutface")
        _m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none); _m.CreateDoubleSidedAttr(True)
        cut_st = UsdGeom.PrimvarsAPI(_m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        from pxr import UsdShade  # noqa: E402
        _mat = UsdShade.Material.Define(sim.stage, "/World/Looks/CutSection")
        _pbr = UsdShade.Shader.Define(sim.stage, "/World/Looks/CutSection/PBR"); _pbr.CreateIdAttr("UsdPreviewSurface")
        _pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        _rd = UsdShade.Shader.Define(sim.stage, "/World/Looks/CutSection/st"); _rd.CreateIdAttr("UsdPrimvarReader_float2")
        _rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st"); _rd.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        _tx = UsdShade.Shader.Define(sim.stage, "/World/Looks/CutSection/tex"); _tx.CreateIdAttr("UsdUVTexture")
        _tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str((pathlib.Path(args_cli.viewer_cut_texture) / "cut_texture.png").resolve()))
        _tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(_rd.ConnectableAPI(), "result")
        _tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        _tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        _pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(_tx.ConnectableAPI(), "rgb")
        _mat.CreateSurfaceOutput().ConnectToSource(_pbr.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(_m.GetPrim()).Bind(_mat)
        cut_mesh["_cutface"] = _m
    def draw_cut(pos):
        from pxr import Vt  # noqa: E402
        parts = cut.build(pos)
        for _n, _m in cut_mesh.items():
            P = parts.get(_n, np.zeros((0, 3), np.float32))
            if _n == "_cutface":
                P, UV = P
                cut_st.Set(Vt.Vec2fArray.FromNumpy(UV))
            _m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P))
            _m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(P) // 3, 3, np.int32)))
            _m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.arange(len(P), dtype=np.int32)))
            if len(P):
                _m.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([P.min(0), P.max(0)])))
    say(f"  viewer cut-away at y = {args_cli.viewer_cut_y*1e3:+.1f} mm, keeping {args_cli.viewer_cut_keep}; "
        f"cut face {'textured from the V4 section ' + args_cli.viewer_cut_texture if cut_tex else 'coloured by element'}; {len(cut.faces):,} boundary faces; tissues {cut.names}")
model.tet_materials.assign(wp.array(new_mat.astype(np.float32), dtype=wp.float32, device=model.device))
model.particle_radius.assign(wp.array(np.full(len(rest), 0.25 * dx, dtype=np.float32), dtype=wp.float32, device=model.device))
model.soft_contact_ke = args_cli.ke
s_pinned = solver_of_ours[pinned_ids]; s_foot = solver_of_ours[footprint]
s_rings = {n: solver_of_ours[v] for n, v in rings.items()}
SLICE = "closure_mode" in d.files and str(d["closure_mode"][0]) == "slice"
if SLICE:
    # graded leg mesh (2026-09-10): closure = plane-slice area of the lumen void at the station, from its oriented
    # boundary triangles -- ONE operator shared with the mesher (closure_ops.py), no node ring.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from closure_ops import slice_area, slice_extent  # noqa: E402
    s_tris = {n: solver_of_ours[d[f"lumen_tris_{n}"].astype(np.int64)] for n in ring_names}
    def lumen_area(pos, n):
        return slice_area(pos, s_tris[n], 0.0)
    def lumen_extent(pos, n):
        """x/z extent of the cut and the lumen boundary's centroid (x, y, z): tells a re-inflating lumen from one drifting off the plane."""
        xmin, xmax, zmin, zmax = slice_extent(pos, s_tris[n], 0.0)
        c = pos[np.unique(s_tris[n].ravel())].mean(0)
        return dict(**{f"{n}_cut_w_mm": (xmax - xmin) * 1e3, f"{n}_cut_h_mm": (zmax - zmin) * 1e3,
                       f"{n}_cx_mm": c[0] * 1e3, f"{n}_cy_mm": c[1] * 1e3, f"{n}_cz_mm": c[2] * 1e3})
    say(f"  closure operator: plane slice at y = 0 of {sum(len(v) for v in s_tris.values()):,} oriented lumen boundary triangles")
else:
    def lumen_area(pos, n):
        return poly_area(pos[s_rings[n]][:, [0, 2]])
    def lumen_extent(pos, n):
        return {}
rest_area = {n: lumen_area(rest, n) for n in ring_names}
say("  rest lumen areas: " + "  ".join(f"{n} {a*1e6:.2f} mm^2" for n, a in rest_area.items()))
rest_t = torch.as_tensor(rest, dtype=torch.float32, device=sim.device)
tgt = obj.data.nodal_kinematic_target.torch.clone()
s_pinned_t = torch.as_tensor(s_pinned, device=tgt.device)
quat_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0], device=sim.device)


def apply_targets(depth_m):
    """Kinematic targets, and the physics plate at the commanded DEPTH, straight above the vessels (theta 0).
    The plate does not follow the sweep angle: during the sweep it hovers 2 mm off the skin at theta 0 and touches
    nothing, exactly as in every run before 2026-09-09. The probe the camera sees is a separate render-only group
    (plate + body, /World/Probe) drawn at the commanded pose -- see move_probe_body. ⚠️ 2026-09-09: writing an
    x-offset and a rotation into this pose read back as the rest pose after the step (log: "physics plate readback");
    that channel is unresolved and nothing in the picture depends on it."""
    t = tgt.clone(); t[0, :, :3] = rest_t; t[0, :, 3] = 1.0; t[0, s_pinned_t, 3] = 0.0
    obj.write_nodal_kinematic_target_to_sim_index(t)
    pose = punch.data.root_state_w[:, :7].clone()
    pose[0, 0] = x_probe; pose[0, 1] = 0.0; pose[0, 2] = z_skin - float(depth_m) + PUNCH_T / 2.0
    pose[0, 3:7] = quat_xyzw
    punch.write_root_pose_to_sim_index(root_pose=pose)


def gpu_sync():
    wp.synchronize_device(str(sim.device)); torch.cuda.synchronize()


# ------------------------------------------------------------------ handshake: the FEM rest nodes define the weights
ready = dict(type="none", note="--no-worker: FEM only")
if conn is not None:
    say(f"  handshake with worker: {len(rest):,} FEM rest nodes, sigma {args_cli.blend_sigma_mm} mm, k {args_cli.k}")
    t0 = time.perf_counter()
    send_msg(conn, dict(type="hello", n=len(rest), sigma_mm=args_cli.blend_sigma_mm, k=args_cli.k, mesh_sha=mesh_sha),
             rest.astype(np.float32).tobytes())
    ready, _ = recv_msg(conn)
    assert ready["type"] == "ready", ready
    say(f"  worker ready in {time.perf_counter()-t0:.1f} s: {ready}")
else:
    say("  FEM only (--no-worker): no B-mode, no images; closure and settle as in the coupled loop")

# warm-up step (graph capture)
apply_targets(0.0); obj.write_data_to_sim(); punch.write_data_to_sim(); sim.step(render=False); gpu_sync()
obj.update(frame_dt); punch.update(frame_dt)
if cam is not None:
    # RTX accumulates over frames and restarts when the scene changes: with the probe warmed up at rest and then jumped
    # to the sweep's first angle, frame 0 of the 2026-09-09 leg run had a noisy silhouette while frame 1 was clean. So
    # warm up with the probe already where the first loop frame will draw it. Nothing is stepped here.
    move_probe_body(-STANDOFF if args_cli.sweep_deg > 0 else 0.0, -args_cli.sweep_deg if args_cli.sweep_deg > 0 else 0.0)
    for _ in range(8):
        sim.render(); cam.update(frame_dt)

# ------------------------------------------------------------------ the coupled press
pc = time.perf_counter
tl, depth_rows, frames_saved = [], [], {}
sim_t, cur, frame_i = 0.0, 0.0, 0
T_START = pc()


pending = None          # pipeline: the frame whose image has been requested but not yet received
HDR_KEYS = ("film_mm", "coupled", "r_face_mm", "fill", "initial_material", "rigid_max_mm")


def _receive(pend):
    """Receive the image for a sent frame and finish its timeline row."""
    hdr, payload = recv_msg(conn); t3 = pc()
    img = np.frombuffer(payload, dtype=np.uint8).reshape(hdr["shape"])
    row = pend["row"]
    row.update(roundtrip_s=t3 - pend["t_sent"], worker_s=hdr["t_worker"], transfer_s=(t3 - pend["t_sent"]) - hdr["t_worker"],
               image_ready_wall=t3 - T_START, image_lag_frames=frame_i - row["frame"] - (0 if not args_cli.pipeline else 0),
               **{k: v for k, v in hdr.items() if k.startswith("t_") or k in HDR_KEYS})
    if row["frame"] % args_cli.save_every == 0:
        frames_saved[f"f{row['frame']:04d}"] = img.copy()
    return img


def one_frame(depth_m, phase, target_mm, theta_deg=0.0):
    """One simulated frame. Returns (positions, image). In --pipeline mode the returned image is the
    PREVIOUS frame's (one simulated frame behind the positions), and the total_s of a row is the
    wall clock from the start of its FEM step to the start of the next frame's FEM step."""
    global sim_t, frame_i, pending
    t0 = pc(); apply_targets(depth_m); obj.write_data_to_sim(); punch.write_data_to_sim(); sim.step(render=False); gpu_sync()
    obj.update(frame_dt); punch.update(frame_dt); t1 = pc()
    pos32 = obj.data.nodal_pos_w.torch[0].cpu().numpy().astype(np.float32); t2 = pc()
    viewer_s = 0.0
    if cam is not None and frame_i % args_cli.viewer_every == 0:
        tv = pc(); move_probe_body(depth_m, theta_deg)
        if cut is not None:
            draw_cut(pos32)
        sim.render(); cam.update(frame_dt)
        # no flip: with the camera posed upright via USD the buffer is upright (diag_viewer_light.py variant F, 2026-09-09).
        # The 2026-09-08 viewer's flipud was compensating for its own rolled camera, not a bottom-up buffer.
        rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
        frames_saved[f"view_f{frame_i:04d}"] = rgb; viewer_s = pc() - tv
        view_std = float(rgb.astype(np.float32).std())      # item 39: per-frame contrast, recorded so a washout can be refused downstream
    else:
        view_std = None
    img = None
    if args_cli.pipeline and pending is not None:
        img = _receive(pending)                       # previous frame's image, rendered while this FEM step ran
        pending["row"]["total_s"] = pc() - pending["row"]["_t0"]
        tl.append(pending["row"]); pending = None
    sim_t += frame_dt
    pos64 = pos32.astype(np.float64)
    row = dict(frame=frame_i, sim_t=sim_t, wall=pc() - T_START, phase=phase, depth_mm=target_mm, theta_deg=float(theta_deg),
               probe_depth_mm=float(depth_m) * 1e3, fem_s=t1 - t0, readback_s=t2 - t1, viewer_s=viewer_s, view_std=view_std, _t0=t0,
               **{f"{n}_pct": 100.0 * lumen_area(pos64, n) / rest_area[n] for n in ring_names},
               **{k: v for n in ring_names for k, v in lumen_extent(pos64, n).items()})
    if conn is None:
        row.update(roundtrip_s=0.0, worker_s=0.0, transfer_s=0.0, total_s=pc() - t0)
        tl.append(row); frame_i += 1
        return pos64, None
    send_msg(conn, dict(type="frame", frame=frame_i, depth_mm=float(depth_m) * 1e3, sim_t=sim_t, theta_deg=float(theta_deg)), pos32.tobytes())
    t_sent = pc()
    if args_cli.pipeline:
        pending = dict(row=row, t_sent=t_sent)
    else:
        img = _receive(dict(row=row, t_sent=t_sent))
        row["total_s"] = pc() - t0
        tl.append(row)
    frame_i += 1
    return pos32.astype(np.float64), img


if args_cli.sweep_deg > 0:
    # -X -> +X -> 0, punch hovering 2 mm above the skin (no contact), tissue at rest: the "find the bundle" arc
    thetas = np.concatenate([np.arange(-args_cli.sweep_deg, args_cli.sweep_deg + 1e-9, args_cli.sweep_step_deg),
                             np.arange(args_cli.sweep_deg - args_cli.sweep_step_deg, -1e-9, -args_cli.sweep_step_deg)])
    say(f"  sweep: {len(thetas)} frames, {-args_cli.sweep_deg:g} -> +{args_cli.sweep_deg:g} -> 0 deg at {args_cli.sweep_step_deg:g} deg/frame")
    for th in thetas:
        one_frame(-STANDOFF, "sweep", 0.0, theta_deg=float(th))
        if th == thetas[0]:
            # instrument note, not a check: what the physics plate reads back after a step at the hover pose (z only).
            pp = punch.data.root_link_pose_w.torch[0].cpu().numpy()
            say(f"  physics plate readback after step (x,y,z | qx,qy,qz,qw): {np.round(pp[:3]*1e3, 3)} mm | {np.round(pp[3:], 4)}; "
                f"commanded (0, 0, {(z_skin + STANDOFF + PUNCH_T/2)*1e3:.3f}) mm, identity")
settled_pos = {}
for target in [float(x) / 1000.0 for x in args_cli.depths.split(",")]:
    nramp = max(1, int(round((target - cur) / (args_cli.ramp_mm_per_frame / 1000.0))))
    for i in range(nramp):
        one_frame(cur + (target - cur) * (i + 1) / nramp, "ramp", target * 1e3)
    cur = target
    ah = {n: deque(maxlen=30) for n in ring_names}; dh = deque(maxlen=30)
    settled_at, veto = None, "max_settle 250 reached"
    for f in range(250):
        pos, img = one_frame(cur, "settle", target * 1e3)
        for n in ring_names:
            ah[n].append(lumen_area(pos, n))
        dh.append(float(np.abs(pos - rest).max()))
        if len(dh) < 30:
            continue
        flat = max((max(ah[n]) - min(ah[n])) / max(abs(np.mean(ah[n])), 1e-15) for n in ring_names)
        dflat = (max(dh) - min(dh)) / max(abs(np.mean(dh)), 1e-15)
        # 🪤 2026-10-29: a NaN run used to settle here -- NaN > 0.02 is False, so both tests fell through to "settled".
        # Two cast runs blew up to NaN at every depth and were logged "settled_at 29, veto none". Non-finite always vetoes.
        if not (np.isfinite(flat) and np.isfinite(dflat) and np.isfinite(pos).all()):
            veto = "NON-FINITE (solver diverged)"
            break
        if flat > 0.02:
            veto = f"area not flat ({100*flat:.2f} %)"
        elif dflat > 0.02:
            veto = f"displacement not flat ({100*dflat:.2f} %)"
        else:
            settled_at, veto = f, "none"; break
    pct = {n: 100.0 * ah[n][-1] / rest_area[n] for n in ring_names}
    if args_cli.save_positions:
        settled_pos[f"pos_{target*1e3:g}mm".replace(".", "p")] = pos.copy()     # solver node order; reordered at save
    if img is not None:
        frames_saved[f"settled_{target*1e3:g}mm"] = img.copy()    # ⚠️ in --pipeline mode this is the frame BEFORE the settled one
    last = tl[-1]
    depth_rows.append(dict(depth_mm=target * 1e3, nramp=nramp, settled_at=settled_at, veto=veto, wall_at_settle_s=pc() - T_START,
                           sim_t_at_settle=sim_t, film_mm=last.get("film_mm"), coupled=last.get("coupled"), fill=last.get("fill"),
                           skin_travel_mm=float(np.abs(rest[s_foot, 2] - pos[s_foot, 2]).max()) * 1e3,
                           **{f"{n}_pct": pct[n] for n in ring_names}))
    _film = last.get("film_mm"); _fill = last.get("fill")
    say(f"  depth {target*1e3:5.1f} mm  ramp {nramp:3d}  settled_at {settled_at}  veto {veto}  vein {pct['vein']:.2f} %  "
        f"artery {pct['artery']:.2f} %  film {'n/a' if _film is None else round(_film, 2)} mm  "
        f"fill {'n/a' if _fill is None else f'{_fill:.3f}'}  wall {pc()-T_START:6.1f} s  sim {sim_t:5.2f} s")

if args_cli.pipeline and pending is not None:
    img = _receive(pending); pending["row"]["total_s"] = pc() - pending["row"]["_t0"]; tl.append(pending["row"]); pending = None
for r in tl:
    r.pop("_t0", None)
if conn is not None:
    send_msg(conn, dict(type="bye")); conn.close()
fs = np.array([r["total_s"] for r in tl]) * 1e3
fem = np.array([r["fem_s"] for r in tl]) * 1e3; wk = np.array([r.get("worker_s", 0.0) for r in tl]) * 1e3; xf = np.array([r.get("transfer_s", 0.0) for r in tl]) * 1e3
say(f"  {len(tl)} frames: total median {np.median(fs):.1f} ms (p95 {np.percentile(fs,95):.1f}, max {fs.max():.1f}); "
    f"FEM {np.median(fem):.1f}  worker {np.median(wk):.1f}  transfer {np.median(xf):.1f} ms")
say(f"  wall {tl[-1]['wall']:.1f} s for {tl[-1]['sim_t']:.2f} s simulated = {tl[-1]['wall']/tl[-1]['sim_t']:.1f}x real time")
out = pathlib.Path(args_cli.out); out.parent.mkdir(parents=True, exist_ok=True)
if args_cli.save_positions:
    # solver order -> mesh-file order: node i of the solver is node ours_of_solver[i] of the npz (bijection asserted above).
    def _to_npz(a):
        b = np.empty_like(a); b[ours_of_solver] = a; return b
    _rest_npz = _to_npz(rest)
    assert np.abs(_rest_npz - our_rest).max() < 1e-6, "reordered rest does not match the mesh file"   # the mapping is right
    np.savez_compressed(args_cli.save_positions, rest=our_rest, tets=our_tets, mesh_sha256_12=mesh_sha,
                        substeps=args_cli.substeps, iterations=args_cli.iterations, cuda_graph=args_cli.cuda_graph,
                        depths_mm=np.array([r["depth_mm"] for r in depth_rows]),
                        **{k: _to_npz(v) for k, v in settled_pos.items()})
    say(f"  saved settled positions (mesh-file order) for {len(settled_pos)} depths -> {args_cli.save_positions}")
json.dump(dict(mode="live_coupled", date=time.strftime("%Y-%m-%d"), viewer_eye=args_cli.viewer_eye, viewer_target=args_cli.viewer_target, viewer_cut_y=args_cli.viewer_cut_y, viewer_cut_keep=(args_cli.viewer_cut_keep if args_cli.viewer_cut_y is not None else None), viewer_cut_texture=args_cli.viewer_cut_texture, leg_visual=(None if (cam is None or args_cli.no_leg_visual) else dict(dir=str(args_cli.leg_visual), usd=str(args_cli.leg_usd), **{k: leg_manifest[k] for k in ("npz_sha256_12", "obj_sha256_12", "slot_y_m", "clearance_m", "r_skin_m")})),
               pipeline=args_cli.pipeline, sweep_deg=args_cli.sweep_deg, sweep_step_deg=args_cli.sweep_step_deg, viewer=args_cli.viewer, viewer_every=args_cli.viewer_every, mesh=args_cli.npz, mesh_sha256_12=mesh_sha, substeps=args_cli.substeps, iterations=args_cli.iterations,
               cuda_graph=args_cli.cuda_graph, ke=args_cli.ke, self_contact=not args_cli.no_self_contact, self_contact_dx_m=SC_DX, blend_sigma_mm=args_cli.blend_sigma_mm, k=args_cli.k,
               worker=ready, timeline=tl, depth_rows=depth_rows, wall_total_s=time.perf_counter() - T_WALL0),
          open(out, "w"), indent=2, default=str)
import csv as _csv  # noqa: E402
def _csv_out(path, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
_csv_out(out.with_suffix(".frames.csv"), tl)          # the record: one row per simulated frame
_csv_out(out.with_suffix(".depths.csv"), depth_rows)  # one row per depth step
np.savez_compressed(out.with_suffix(".frames.npz"), **frames_saved)
say(f">>> wrote {out} and {out.with_suffix('.frames.npz')} ({len(frames_saved)} frames)")
app.close()
