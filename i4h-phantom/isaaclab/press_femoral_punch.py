"""The 2026-08-18 press with the SAME LOAD: a rigid punch in contact, not prescribed displacement.

WHY THIS EXISTS. The first Isaac Lab press drove the 216 footprint skin nodes as prescribed
displacement and closed the vein LESS at every depth (88.9 % vs 79.7 % at 7 mm) while wrecking the
mesh sooner (inverting between 5 and 7 mm against 9-11 mm on the standalone rig). Identical
material and identical mesh must give identical displacements -- so the load could not have been
identical, and it was not. Measured on 2026-08-20:

  - the deformed skin carried a STEP at the footprint edge of up to **1.84 mm**, which a rigid
    punch cannot produce because tissue rides up its face and the surface stays continuous
  - **123 of the 200 worst elements (62 %) at 7 mm sat within 3 mm of a footprint edge**

Both are artefacts of the boundary condition. This version replaces it with the thing
2026-08-18 actually used: a rigid flat punch driven into the tissue, with particle-shape contact
doing the loading. No skin node is prescribed; only the region-of-interest boundary is pinned.

[original header follows]
The 2026-08-18 press, re-run on Isaac Lab's Newton VBD with OUR imported tet mesh.

    env/bin/python press_femoral_punch.py --headless --depths 1,3,5,7 \
        --substeps 640 --iterations 100 \
        --npz out/tetmesh_femoral_v4_gap12_s020_periv.npz \
        --usd out/tetmesh_femoral_v4_gap12_s020_periv.usda

    ⚠️ --npz and --usd are REQUIRED as of 2026-08-25. There is no default mesh: a default
    that names a mesh the run is not using is how the 2026-08-21 press was misread for four
    days (open items 18, 37). --out and --save-positions DERIVE from the --npz stem, so an
    s020 run cannot write a file called ...s030...

WHAT IS BEING REPRODUCED. 2026-08-18 pressed a probe into the femoral phantom at station
s = 6 mm and measured how much each lumen closed: vein 79.7 %, artery 97.4 % of rest area at
6.99 mm past skin.

🚨 CORRECTED 2026-08-25: this said "station s = 30 mm" and that is WRONG. The 2026-08-18 run was
at s = 6 mm. Evidence: `out/femoral_v4_gap12/section_s006.json` carries `station_s_mm: 6.0` and
was written 2026-08-18 16:34; `out/demo_s6/`, `out/verify_s6/` and `out/room_s6/` are all that
day; the 2026-08-18 README quotes these same 79.7 / 97.4 figures at 6.99 mm; and open item 35
states the station as s = 6 mm. The error mattered -- this file's --npz/--usd DEFAULTS also point
at an s030 mesh, so the docstring and the defaults agreed with each other and both disagreed with
the run, which is how the 2026-08-21 press came to be misread. See open item 18 (closed): the
result JSON now records its own mesh, digest and body list.

Closure is a POLYGON AREA through each lumen's ring of wall nodes, so it is read entirely off node
positions and needs no force channel -- which is why this can run while the Isaac Lab force offset
(item 34) is still unattributed.

⚠️ ONE DELIBERATE DIFFERENCE FROM 2026-08-18, STATED BEFORE THE RUN.
2026-08-18 drove a RIGID FLAT PUNCH in soft contact, on a prismatic joint. Here the footprint
skin nodes are driven as KINEMATIC TARGETS -- prescribed displacement. The difference is real:
  - prescribed nodes CANNOT separate from the punch, so there is no rim separation
  - there is no punch EDGE singularity, which is where item 28 measured the inversions seeding
So expect this to be, if anything, better behaved than 2026-08-18 at depth and to give slightly
MORE closure per unit depth. 🚨 It is therefore NOT a like-for-like reproduction of the boundary
condition, only of the anatomy, the materials and the measurement. Say so wherever these numbers
appear next to 2026-08-18's.

📌 TIMESTEP MATCHED ON PURPOSE. Isaac Lab's default is dt = 0.005 with 4 substeps -> 1250 us.
2026-08-18 ran 1/60 s with 512 substeps -> 32.6 us, and femoral_fem already reports 132 of
162,954 tets unintegrable at THAT step. Running 38x coarser would not be a re-run, it would be a
different experiment that diverges. dt and substeps are matched here.
"""

import argparse
import json
import pathlib
import numpy as np
from run_provenance import provenance_block
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True, help="USD companion to --npz. REQUIRED -- no default; see --npz.")
ap.add_argument("--npz", required=True, help="Tet mesh to press. REQUIRED -- no default. Until 2026-08-25 this defaulted to tetmesh_femoral_v4_s030.npz while the work ran at s006/s020, so the default silently named a mesh no run used and the output could not be told apart from it (open items 18, 37). A wrong mesh must be a refusal to start, not a substitution.")
ap.add_argument("--depths", default="1,3,5,7", help="mm past skin")
ap.add_argument("--ramp-mm-per-frame", type=float, default=0.05)
ap.add_argument("--substeps", type=int, default=512)
ap.add_argument("--iterations", type=int, default=10)
ap.add_argument("--density", type=float, default=1000.0)
ap.add_argument("--ke", type=float, default=2.0e3, help="soft_contact_ke; the press used 2e3")
ap.add_argument("--artery-kpa", type=float, default=0.0,
                help="lumen pressure, artery. 2026-08-08 used 12.5 kPa (transmural). 🔴 textbook "
                     "point estimate, logged in sources_needed.md. 2026-08-18 ran 0.")
ap.add_argument("--vein-kpa", type=float, default=0.0,
                help="lumen pressure, vein. 2026-08-08 used 1.0 kPa. 🔴 same status.")
ap.add_argument("--allow-inversion", action="store_true",
                help="Do not let inverted elements VETO the settle test. 🎯 Added 2026-08-26 by "
                     "direction: Newton's stable neo-Hookean is built to stay finite and "
                     "differentiable for J<0 and to RECOVER from inversion rather than forbid it "
                     "(item 28), so a step with a few inverted elements still reaches a "
                     "meaningful configuration. Without this the depth curve hard-walls at the "
                     "first inversion (~9 mm) and we never see how far the tissue actually goes. "
                     "⚠️ `usable` STILL goes false on any inversion -- this changes what the run "
                     "is allowed to MEASURE, never what it is allowed to CLAIM.")
ap.add_argument("--stop-force-n", type=float, default=None,
                help="Abandon the remaining depths once F_tissue exceeds this. Safety cap so a "
                     "deep sequence cannot run away; 30 N by direction 2026-08-26.")
ap.add_argument("--no-cuda-graph", action="store_true",
                help="Run the substep loop eagerly instead of capturing it into one CUDA graph. "
                     "🚨 REQUIRED AT HIGH SUBSTEP COUNTS, measured 2026-08-27. NewtonManager "
                     "captures actuators AND every solver substep into a SINGLE graph "
                     "(newton_manager.py:524-527), so the graph holds substeps x iterations "
                     "worth of launches. The 2026-08-26 mesh study at 896 x 140 (125,440 "
                     "iterations) reached 45.4 GB resident + 7.8 GB swap on a 62 GB machine, "
                     "filled swap, then spent 21 h 49 m at 1.1 CPU cores with the GPU at 0 % and "
                     "never finished its FIRST step of seven. ⚠️ RESULT-NEUTRAL: this is the same "
                     "path the lumen-pressure runs already take (see the use_cuda_graph line "
                     "below), and the graph changes only how kernels are dispatched, never what "
                     "the solver computes. Slower per frame; bounded in memory.")
ap.add_argument("--log-every", type=int, default=25,
                help="print a settle progress line every N frames. 🎯 Added 2026-08-27: the run only\n"
                     "printed at STEP boundaries, so a step that took 76 minutes was\n"
                     "indistinguishable from a hung one from outside. The flatness numbers it\n"
                     "prints are the settle test's own, so the trajectory toward (or away from)\n"
                     "the tolerance is visible while it happens. 0 disables.")
ap.add_argument("--max-settle", type=int, default=250)
ap.add_argument("--settle-window", type=int, default=30)
ap.add_argument("--settle-tol", type=float, default=0.02)
ap.add_argument("--trace", default="", help="npz path: per-frame F_tissue and lumen areas, so CREEP is visible rather than inferred")
ap.add_argument("--out", default=None,
                help="Result JSON. Default: derived from the --npz stem, so the output name can NEVER disagree with the mesh that produced it.")
ap.add_argument("--save-positions", default=None,
                help="deformed node positions per step — the input to the B-mode and the render, "
                     "and the only way to SEE where the mesh is failing")
AppLauncher.add_app_launcher_args(ap)
args_cli = ap.parse_args()

# 🎯 OUTPUT NAMES DERIVE FROM THE MESH, so they cannot disagree with it.
# `press_i40_periv_s006.json` was indistinguishable from an s030 run because the name was a free
# string while --npz had a default nobody passed (open items 18, 37). Deriving the stem means a
# run at s020 CANNOT write a file called ...s030..., whatever anyone types.
import os as _os_derive
_stem = _os_derive.path.basename(args_cli.npz)
for _p in ("tetmesh_", ".npz"):
    _stem = _stem.replace(_p, "")
_outdir = _os_derive.path.dirname(_os_derive.path.abspath(args_cli.npz))
if args_cli.out is None:
    args_cli.out = _os_derive.path.join(_outdir, f"{"press_punch"}_{_stem}.json")
if args_cli.save_positions is None:
    args_cli.save_positions = _os_derive.path.join(_outdir, f"{"press_punch"}_{_stem}_pos.npz")
print(f">>> mesh   {args_cli.npz}", flush=True)
print(f">>> result {args_cli.out}", flush=True)

app = AppLauncher(args_cli).app

import torch  # noqa: E402
import warp as wp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


@configclass
class VBDSolverCfgBigContact(VBDSolverCfg):
    """VBDSolverCfg plus the rigid-body/particle contact buffer, which Isaac Lab does not expose.

    🚨 MEASURED 2026-08-20, AND IT SILENTLY TRUNCATED THE LOAD.
    Running the punch produced, every substep:

        Warning: Per-body particle contact buffer overflowed 2737 > 256.

    The punch touches 2,737 particles; Newton's default per-body capacity is 256, so about 91 %
    of the contacts were DISCARDED and the tissue was being pressed by a fraction of the punch
    face. Nothing raises -- the run completes and produces a smooth, plausible, wrong curve.
    solver_vbd.py:355 states these buffers are "fixed and will not be dynamically resized during
    runtime", so it has to be set at construction.

    📌 It reaches the solver because `VBDManager._build_solver` filters `solver_cfg.to_dict()`
    against `inspect.signature(SolverVBD.__init__)` -- any field the solver accepts is passed
    through, so declaring it here is enough.
    """

    rigid_body_particle_contact_buffer_size: int = 8192
from isaaclab_newton.physics import NewtonCfg, NewtonManager  # noqa: E402
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg  # noqa: E402
from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg  # noqa: E402

import os as _os  # noqa: E402
import sys  # noqa: E402
# The shared libraries (snh_force, femoral_fem, materials, ...) live at the repository
# root, one level up from isaaclab/. Resolved from this file so the repository can be
# cloned anywhere; it used to be an absolute home-directory path, which is what
# folding these scripts into the repository removed.
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from snh_force import TetElasticitySNH  # noqa: E402


def rule(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def poly_area(pts_xz):
    """Shoelace, identical to femoral_fem.Phantom._poly_area."""
    x, z = pts_xz[:, 0], pts_xz[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))))


d = np.load(args_cli.npz, allow_pickle=True)
our_rest = d["rest"].astype(np.float64)
our_tets = d["tets"].astype(np.int64)
our_mat = d["tet_materials"].astype(np.float64)
pinned_ids = d["pinned"].astype(np.int64)
footprint = d["footprint"].astype(np.int64)
ring_names = [str(n) for n in d["ring_names"]]
rings = {n: d[f"ring_{n}"].astype(np.int64) for n in ring_names}
rest_area_file = {n: float(d[f"rest_area_{n}"][0]) for n in ring_names}

rule("SETUP")
print(f"  mesh: nodes={len(our_rest):,} tets={len(our_tets):,}  pinned={len(pinned_ids):,}  "
      f"footprint={len(footprint)}", flush=True)
print(f"  rings: " + "  ".join(f"{n}={len(rings[n])} nodes, rest {rest_area_file[n]*1e6:.2f} mm^2"
                               for n in ring_names), flush=True)

dx = float(d["dx"][0])
frame_dt = 1.0 / 60.0
# 🚨 FOUR SETTINGS THAT ISAAC LAB DEFAULTS DIFFERENTLY FROM THE RIG THAT PRODUCED 2026-08-18.
# The first smoke test ran with Isaac Lab's defaults and detonated: 2,806 inverted tets and
# minJ = -3.87 at ONE millimetre of indentation, with the vein reading 12.9 % -- below its own
# measurement floor. None of that was physics; all four are configuration.
#   1. GRAVITY. femoral_fem builds with `ModelBuilder(up_axis=Z, gravity=0.0)` -- the press is
#      quasi-static and gravity-FREE. Isaac Lab defaults to -9.81. A 92 mm column of 3 kPa soft
#      tissue under its own weight is rho*g*h = 883 Pa against E = 3000 Pa, i.e. ~30 % strain
#      before the probe moves at all.
#   2. particle_radius. Isaac Lab hard-codes 0.008 m in DeformableRegistryEntry; the rig uses
#      0.25*dx = 0.375 mm. At dx = 1.5 mm that default is 21x too large.
#   3. self-contact is OFF by default here and ON in the rig -- and it is not optional: it is
#      what stops opposing lumen walls passing through each other, and it sets the 0.8*dx
#      closure floor that 2026-08-18 reports.
#   4. soft_contact_ke: the press ran 2e3, inside the measured 2e3-2e4 band.
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    dt=frame_dt, device=args_cli.device, gravity=(0.0, 0.0, 0.0),
    physics=NewtonCfg(solver_cfg=VBDSolverCfgBigContact(
        iterations=args_cli.iterations,
        particle_enable_self_contact=True,
        particle_self_contact_radius=0.4 * dx,
        particle_self_contact_margin=0.5 * dx),
        num_substeps=args_cli.substeps,
        # 🚨 CUDA GRAPH CAPTURE BYPASSES PYTHON-LEVEL PATCHES. The substep loop -- including
        # `state.clear_forces()` -- is captured once and replayed on device, so the clear_forces
        # wrapper that re-applies lumen pressure NEVER RUNS during replay. Measured 2026-08-20:
        # with the graph on, turning pressure to 12.5/1.0 kPa changed the closure numbers by
        # EXACTLY ZERO (99.5/93.4 and 98.8/91.2 identical to the unpressurised run, same minJ,
        # same settle frame) -- a silent no-op that looks like "pressure does not matter".
        # Disabled whenever lumen pressure is on; slower, and correct.
        use_cuda_graph=(not args_cli.no_cuda_graph)
                       and not (args_cli.artery_kpa or args_cli.vein_kpa))))
print(f"  gravity=0 (rig builds with gravity=0.0)   self_contact ON r={0.4*dx*1e3:.3f} mm "
      f"margin={0.5*dx*1e3:.3f} mm   dx={dx*1e3:.2f} mm", flush=True)
print(f"  dt={frame_dt*1e3:.3f} ms  substeps={args_cli.substeps}  -> substep "
      f"{frame_dt/args_cli.substeps*1e6:.1f} us  (2026-08-18 ran 32.6 us)", flush=True)
_graph_on = (not args_cli.no_cuda_graph) and not (args_cli.artery_kpa or args_cli.vein_kpa)
print(f"  cuda_graph={'ON' if _graph_on else 'OFF'}  "
      f"({args_cli.substeps * args_cli.iterations:,} solver iterations per frame"
      + ("; the whole loop is captured into one graph -- watch host RSS" if _graph_on
         else "; dispatched eagerly, bounded memory") + ")", flush=True)

sim_utils.create_prim("/World/env_0", "Xform", translation=(0.0, 0.0, 0.0))
obj = DeformableObject(cfg=DeformableObjectCfg(
    prim_path="/World/env_0/Phantom",
    spawn=sim_utils.UsdFileCfg(
        usd_path=args_cli.usd,
        deformable_props=NewtonDeformableBodyPropertiesCfg(),
        physics_material=NewtonDeformableBodyMaterialCfg(
            k_mu=float(our_mat[:, 0].mean()), k_lambda=float(our_mat[:, 1].mean()),
            # 🪤 THIS k_damp IS DEAD, verified 2026-08-21. The per-tet write below copies
            # tet_materials[:, :3] from the mesh file, and column 2 IS k_damp -- 1.0 Pa*s on all
            # 170,148 elements. So the EFFECTIVE damping is 1.0, not the 0.0 written here, and a
            # reader who trusts this line has the wrong number. Left in place rather than changed
            # because 1.0 is what the standalone rig used and what 2026-08-18 ran: editing it would
            # silently break the comparison. Viscous stress vanishes at zero strain rate, so this
            # does not move the settled answer -- only the transient.
            density=args_cli.density, k_damp=0.0)),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    debug_vis=False))
# --- the punch: a rigid flat plate, kinematic, driven down into the tissue ---
x_probe = float(d["x_probe"][0]); z_skin = float(d["z_skin"][0])
probe_w = float(d["probe_w"][0]); probe_len = float(d["probe_len"][0])
PUNCH_T = 0.004                      # 4 mm thick plate; only its lower face ever touches tissue
STANDOFF = 0.002                     # start clear of the skin, as the rig does
punch = RigidObject(cfg=RigidObjectCfg(
    prim_path="/World/env_0/Punch",
    spawn=sim_utils.CuboidCfg(
        size=(probe_w, probe_len, PUNCH_T),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1)),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(x_probe, 0.0, z_skin + STANDOFF + PUNCH_T / 2.0))))
sim.reset()

rest = obj.data.default_nodal_state_w.torch[0, :, :3].cpu().numpy().astype(np.float64)
model = NewtonManager.get_model()
solver_tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int64)

# --- node ordering bijection (1 um tolerance: USD stores float32) ---
key = {tuple(np.round(p, 6)): i for i, p in enumerate(our_rest)}
ours_of_solver = np.array([key.get(tuple(np.round(p, 6)), -1) for p in rest], dtype=np.int64)
assert (ours_of_solver >= 0).all() and len(np.unique(ours_of_solver)) == len(rest), \
    "node ordering is not a bijection -- every downstream index would be wrong"
solver_of_ours = np.empty(len(rest), dtype=np.int64)
solver_of_ours[ours_of_solver] = np.arange(len(rest))
print(f"  ✅ ordering bijection, max residual "
      f"{float(np.abs(rest - our_rest[ours_of_solver]).max()):.3e} m", flush=True)

# --- per-tet materials, matched by CONTENT (Newton may drop rest-degenerate tets) ---
ours_key = {tuple(sorted(t)): k for k, t in enumerate(our_tets.tolist())}
new_mat = model.tet_materials.numpy().copy()
nmatched = 0
for si, st in enumerate(solver_tets):
    k = ours_key.get(tuple(sorted(ours_of_solver[st].tolist())))
    if k is not None:
        new_mat[si, :3] = our_mat[k, :3]
        nmatched += 1
model.tet_materials.assign(wp.array(new_mat.astype(np.float32), dtype=wp.float32, device=model.device))
print(f"  ✅ per-tet materials: {nmatched:,}/{len(solver_tets):,} matched; "
      f"{np.unique(new_mat[:, :2], axis=0).shape[0]} unique rows, "
      f"k_mu {new_mat[:,0].min():.4g}..{new_mat[:,0].max():.4g} Pa", flush=True)

# particle_radius and the contact stiffness are post-spawn writes, like tet_materials --
# neither is reachable through Isaac Lab's config.
pr_before = float(model.particle_radius.numpy().max())
model.particle_radius.assign(wp.array(
    np.full(len(rest), 0.25 * dx, dtype=np.float32), dtype=wp.float32, device=model.device))
model.soft_contact_ke = args_cli.ke
print(f"  ✅ particle_radius {pr_before*1e3:.3f} mm -> {0.25*dx*1e3:.3f} mm (0.25*dx);  "
      f"soft_contact_ke={args_cli.ke:.3g} N/m", flush=True)

# --- index sets in SOLVER order ---
s_pinned = solver_of_ours[pinned_ids]
s_foot = solver_of_ours[footprint]
s_rings = {n: solver_of_ours[v] for n, v in rings.items()}
rest_area = {n: poly_area(rest[s_rings[n]][:, [0, 2]]) for n in ring_names}
print(f"  rest areas recomputed here: " + "  ".join(
    f"{n}={rest_area[n]*1e6:.2f} mm^2 (file {rest_area_file[n]*1e6:.2f})" for n in ring_names), flush=True)

# ---------------------------------------------------------------------------
# F_tissue on the FEMORAL mesh.  Same operator as the cube calibration, same
# arrays the solver integrates: mu and lambda are read back out of
# `model.tet_materials` AFTER the per-tet write, so the operator and Newton
# cannot drift apart.  Gravity is 0 here, so on a FREE node the punch is the
# ONLY external force and F_tissue is exactly the punch load.
#
# 🚨 SCOPE, STATED BEFORE THE RUN.  `f_int + f_ext = 0` holds AT EQUILIBRIUM.
# VBD is iterative; at 4 substeps the cube read +305.7 % high.  So this number
# is a force measurement only where the substep sweep shows it has converged,
# and the sweep is the point of --ftissue-only.
# ---------------------------------------------------------------------------
mat_now = model.tet_materials.numpy()
op = TetElasticitySNH(rest, solver_tets,
                      mat_now[:, 0].astype(np.float64), mat_now[:, 1].astype(np.float64))
s_free = np.ones(len(rest), dtype=bool)
s_free[s_pinned] = False
f_at_rest = op.probe_load_z(rest, s_free)

# ---------------------------------------------------------------------------
# 🎯 THE EQUILIBRIUM RESIDUAL — the convergence criterion this project never had.
#
# Everything so far asked "has the answer stopped changing as I refine?", which needs a
# plateau and an extrapolation and gave neither. This asks the question directly instead:
# a FREE node that the punch is not touching has NO external force on it, so at static
# equilibrium `f_int = 0` on that node, exactly. |f_int| over those nodes is therefore a
# direct measure of HOW FAR FROM EQUILIBRIUM the solve actually is -- absolute, per
# setting, with no reference run and nothing extrapolated.
#
# ⚠️ SCOPE. Nodes near the punch and nodes in self-contact DO carry external forces, so
# they are excluded rather than counted as residual. The exclusion is by distance from the
# punch face, so it is geometric and stated, not fitted. Self-contact is reported as a
# COUNT so a run where it is active can be spotted rather than silently averaged in.
# ---------------------------------------------------------------------------
_interior = s_free.copy()
_interior[s_foot] = False

print(f"  ✅ F_tissue operator: {len(solver_tets):,} tets, {int(s_free.sum()):,} free nodes; "
      f"zero reading at rest = {f_at_rest:+.6f} N (must be ~0)", flush=True)

trace = []
rest_t = torch.as_tensor(rest, dtype=torch.float32, device=sim.device)
tgt = obj.data.nodal_kinematic_target.torch.clone()


def apply_targets(depth_m):
    """Pin ONLY the region-of-interest boundary. Every skin node is FREE -- the punch loads it."""
    t = tgt.clone()
    t[0, :, :3] = rest_t
    t[0, :, 3] = 1.0
    t[0, torch.as_tensor(s_pinned, device=t.device), 3] = 0.0
    obj.write_nodal_kinematic_target_to_sim_index(t)
    # punch lower face at (z_skin - depth): centre sits half a thickness above it
    pose = punch.data.root_state_w[:, :7].clone()
    pose[0, 0] = x_probe
    pose[0, 1] = 0.0
    pose[0, 2] = z_skin - float(depth_m) + PUNCH_T / 2.0
    # 🪤 QUATERNION ORDER. isaaclab_newton's write_root_pose_to_sim_index documents (x, y, z, w).
    # Identity is therefore [0, 0, 0, 1], NOT [1, 0, 0, 0] -- the (w,x,y,z) form used by isaacsim
    # and IsaacLab 2.3. Getting this wrong rotates the punch silently instead of raising, and it
    # is correct at theta = 0 and progressively wrong either side, so a single eyeball check
    # passes it. This project has been caught by exactly this before.
    pose[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=pose.device, dtype=pose.dtype)
    punch.write_root_pose_to_sim_index(root_pose=pose)


def mesh_quality(pos):
    p = pos[solver_tets]
    Ds = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)
    v = np.linalg.det(Ds)
    v0 = np.linalg.det(np.stack([rest[solver_tets][:, 1] - rest[solver_tets][:, 0],
                                 rest[solver_tets][:, 2] - rest[solver_tets][:, 0],
                                 rest[solver_tets][:, 3] - rest[solver_tets][:, 0]], axis=2))
    J = v / v0
    return float(J.min()), int((J <= 0).sum())


# =====================================================================================
# LUMEN PRESSURE — the mechanism that makes the artery hold while the vein yields.
# =====================================================================================
# 🚨 WHY THIS HAD TO BE PORTED. Both lumens are bare VOIDS in the mesh, so with pressure off the
# artery's only defence against collapse is its 600 kPa wall. Real compression sonography
# discriminates the two vessels because the artery is PRESSURISED (~90-100 mmHg) and the vein is
# not (~5-10 mmHg) -- that asymmetry IS the diagnostic signal. On 2026-08-08, turning it on at
# artery 12.5 kPa / vein 1.0 kPa produced "the vein yields ~2.5x the artery at every step".
# ⚠️ 2026-08-18 ran with pressure OFF and says so, calling its 98.2 % artery "the pessimistic
# figure". So this is an IMPROVEMENT on that run, not a reproduction of it -- report separately.
#
# The forces are FOLLOWER forces: recomputed each frame from the CURRENT lumen shape, outward
# normal, length-weighted. Same algorithm as femoral_fem.Phantom._refresh_lumen_pressure.
#
# 🪤 AND THE INJECTION POINT IS NOT OBVIOUS. NewtonManager._run_solver_substeps calls
# `state.clear_forces()` after EVERY substep (newton_manager.py:1471,1481), so a force written
# once per frame would survive 1 substep in 512 -- i.e. 0.2 % of the intended impulse, silently.
# The rig re-applies after each clear; here `clear_forces` is wrapped on both state buffers so the
# pressure buffer is restored every time it is zeroed.
import types  # noqa: E402

_wrap_calls = [0]

LUMEN_P = {"artery": args_cli.artery_kpa * 1e3, "vein": args_cli.vein_kpa * 1e3}
surf_names = [str(n) for n in d["surf_names"]] if "surf_names" in d.files else []
surfs = {n: solver_of_ours[d[f"surf_{n}"].astype(np.int64)] for n in surf_names}
p_scale = float(d["p_scale"][0]) if "p_scale" in d.files else 1.0
use_pressure = any(v for v in LUMEN_P.values()) and surfs
pf_buf = None
if use_pressure:
    pf_buf = wp.zeros(len(rest), dtype=wp.vec3, device=model.device)

    def _wrap(state):
        orig = state.clear_forces

        def clear_and_restore(self):
            orig()
            self.particle_f.assign(pf_buf)
            _wrap_calls[0] += 1
        state.clear_forces = types.MethodType(clear_and_restore, state)

    for _st in (NewtonManager._state_0, NewtonManager._state_1):
        if _st is not None:
            _wrap(_st)
    print(f"  ✅ lumen pressure ON: artery {args_cli.artery_kpa} kPa, vein {args_cli.vein_kpa} kPa "
          f"(p_scale={p_scale}); clear_forces wrapped, cuda_graph DISABLED", flush=True)


def refresh_lumen_pressure(pos):
    """Outward, length-weighted follower forces on every lumen wall node."""
    if not use_pressure:
        return
    f = np.zeros((len(rest), 3), dtype=np.float32)
    nyp = int(d["ny"][0]) + 1
    for name, pa in LUMEN_P.items():
        if not pa or name not in surfs:
            continue
        ring = surfs[name]
        for jj in range(min(nyp, ring.shape[0])):
            ids = ring[jj]
            P = pos[ids][:, [0, 2]]
            e = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
            nrm = 0.5 * np.column_stack([e[:, 1], -e[:, 0]])      # outward, length-weighted
            w = dx * (0.5 if jj in (0, ring.shape[0] - 1) else 1.0)
            fn = (pa * p_scale * w) * nrm
            np.add.at(f, ids, np.column_stack([fn[:, 0], np.zeros(len(ids)), fn[:, 1]]))
    pf_buf.assign(wp.array(f, dtype=wp.vec3, device=model.device))


from collections import deque  # noqa: E402

saved_pos = {}
_rp = rest[solver_tets]
v0_ref = np.linalg.det(np.stack([_rp[:, 1] - _rp[:, 0], _rp[:, 2] - _rp[:, 0],
                                 _rp[:, 3] - _rp[:, 0]], axis=2))

depths = [float(x) / 1000.0 for x in args_cli.depths.split(",")]
rows, cur = [], 0.0

out = pathlib.Path(args_cli.out)


def write_results(final=False):
    """Write the JSON and its CSV. 🚨 CALLED AFTER EVERY STEP, not only at the end (2026-08-27).

    Until today the result file appeared only when the whole depth sequence finished, so a trial
    that was killed, hung, or ran out of time at step 5 of 7 left NOTHING on disk but a log -- five
    good measurements unreadable by the triage and unplottable. A partial file is not a defect: each
    row already carries its own settle verdict, inversion count and force check, so a reader can
    tell a finished sequence from an interrupted one by the rows themselves.
    """
    json.dump(dict(rows=rows, rest_area_mm2={n: rest_area[n] * 1e6 for n in ring_names},
                   complete=final, n_depths_requested=len(depths),
                   **provenance_block(args_cli, d),
                   n_nodes=len(rest), n_tets=len(solver_tets), substeps=args_cli.substeps,
                   boundary_condition="rigid flat punch in particle-shape contact, matching 2026-08-18",
                   artery_kpa=args_cli.artery_kpa, vein_kpa=args_cli.vein_kpa,
                   force_channel="F_tissue = sum over FREE nodes of f_int_z (stable neo-Hookean), "
                                 "window mean over the settle window; gravity is 0 so the punch is "
                                 "the only external force on a free node",
                   f_tissue_zero_at_rest_N=f_at_rest, ramp_mm_per_frame=args_cli.ramp_mm_per_frame,
                   dt_s=frame_dt, iterations=args_cli.iterations, soft_contact_ke=args_cli.ke),
              open(out, "w"), indent=2)
    import csv as _csv
    if rows:
        _cols = list(dict.fromkeys([k for r in rows for k in r]))
        with open(out.with_suffix(".csv"), "w", newline="") as _fh:
            _w = _csv.DictWriter(_fh, fieldnames=_cols, extrasaction="ignore")
            _w.writeheader()
            for _r in rows:
                _w.writerow(_r)
            _fh.write(f"# rows from {out.name} (complete={final}); that JSON holds the provenance, "
                      f"rest areas and the force-channel definition.\n")

# 🚨 THE DEEPEST DEPTH USED TO BE TRUNCATED IN THE KEY, and 2026-09-01 sec 3a said to fix it at
# source. It was fixed in the animation builder only; the key here still read int(round(25.4)) = 25.
# So the solve ran 25.4 mm and every saved-position file called it `pos_25mm`, which put a renderer's
# probe 0.4 mm too shallow for its own tissue and made `closure[25.4]` a silent KeyError-to-None.
# Lossless now: integral depths keep their old names (`pos_3mm`), a fractional one becomes
# `pos_25p4mm`. A consumer looking for the old `pos_25mm` therefore FAILS LOUDLY instead of
# quietly reading the wrong depth -- which is the point (~/.claude/CLAUDE.md sec 10).
# `depths_mm` is written alongside so nothing has to parse key names at all.
def _depth_key(t_m: float) -> str:
    mm = round(t_m * 1e3, 3)
    return str(int(mm)) if float(mm).is_integer() else f"{mm:g}".replace(".", "p")

_saved_depths: set = set()

for target in depths:
    rule(f"STEP: {target*1e3:.2f} mm past skin")
    # ramp
    nramp = max(1, int(round((target - cur) / (args_cli.ramp_mm_per_frame / 1000.0))))
    # 🎯 THE RAMP GETS A PROGRESS LINE TOO, added 2026-08-27. It is the LONG phase -- at 0.05 mm/frame
    # the 3..25.4 mm list is 508 ramp frames against ~210 settle frames -- and until now it printed
    # nothing at all, so the majority of a 17 h trial was invisible from outside while a watcher
    # reported "no new step" and could not say whether that meant slow or stuck.
    if args_cli.log_every:
        print(f"    ramp: {nramp} frames at {args_cli.ramp_mm_per_frame} mm/frame "
              f"({cur*1e3:.2f} -> {target*1e3:.2f} mm)", flush=True)
    for i in range(nramp):
        if args_cli.log_every and i and i % args_cli.log_every == 0:
            print(f"    ramp {i:4d}/{nramp}  probe at "
                  f"{(cur + (target - cur) * i / nramp)*1e3:6.2f} mm", flush=True)
        apply_targets(cur + (target - cur) * (i + 1) / nramp)
        if use_pressure:
            refresh_lumen_pressure(obj.data.nodal_pos_w.torch[0].cpu().numpy().astype(np.float64))
        obj.write_data_to_sim(); punch.write_data_to_sim(); sim.step()
        obj.update(frame_dt); punch.update(frame_dt)
    cur = target
    # settle on SHAPE (no force channel involved)
    ah = {n: deque(maxlen=args_cli.settle_window) for n in ring_names}
    dh = deque(maxlen=args_cli.settle_window)
    fh_t = deque(maxlen=args_cli.settle_window)
    # 🚩 the settle test can only veto on a 30-frame WINDOW; if the body creeps slower than the
    # tolerance it passes while still moving. The trace is what makes that visible.
    settled_at, veto = None, "max_settle reached"
    blew_up = False
    for f in range(args_cli.max_settle):
        apply_targets(cur)
        if use_pressure:
            refresh_lumen_pressure(obj.data.nodal_pos_w.torch[0].cpu().numpy().astype(np.float64))
        obj.write_data_to_sim(); punch.write_data_to_sim(); sim.step()
        obj.update(frame_dt); punch.update(frame_dt)
        pos = obj.data.nodal_pos_w.torch[0].cpu().numpy().astype(np.float64)
        # 🚨 NON-FINITE GUARD, added 2026-08-27. Without it a diverged solve is reported as a
        # CONVERGED step: NaN positions make `flat` and `dflat` NaN, every `nan > tol` is False,
        # so the settle test falls through to its `else` and writes settled_at=f, veto="none" --
        # and mesh_quality's `J < 0` is also False on NaN, so n_inverted reads 0 and `usable`
        # reads True. That is a blow-up wearing a measurement's clothes, and it matters precisely
        # because the depth sequence is now meant to run until the mesh absolutely fails.
        # ⚠️ Inversion is NOT failure -- Newton's stable neo-Hookean is built to carry J < 0 and
        # recover. Non-finite is failure. Keep the two apart.
        if not np.isfinite(pos).all():
            nbad = int((~np.isfinite(pos)).any(axis=1).sum())
            veto = f"NON-FINITE positions: {nbad:,}/{len(pos):,} nodes at settle frame {f}"
            blew_up = True
            break
        for n in ring_names:
            ah[n].append(poly_area(pos[s_rings[n]][:, [0, 2]]))
        dh.append(float(np.abs(pos - rest).max()))
        fh_t.append(op.probe_load_z(pos, s_free))
        if args_cli.trace:
            trace.append([float(target*1e3), float(f), fh_t[-1], dh[-1]] +
                         [ah[n][-1] for n in ring_names])
        if args_cli.log_every and f % args_cli.log_every == 0:
            _fl = (max((max(ah[n]) - min(ah[n])) / max(abs(np.mean(ah[n])), 1e-15)
                       for n in ring_names) if len(dh) >= args_cli.settle_window else float("nan"))
            print(f"    settle {f:4d}/{args_cli.max_settle}  max disp {dh[-1]*1e3:7.3f} mm  "
                  f"F_tissue {fh_t[-1]:+8.4f} N  area flatness {100*_fl:6.2f} % "
                  f"(needs <= {100*args_cli.settle_tol:.0f} %)", flush=True)
        if len(dh) < args_cli.settle_window:
            continue
        minJ, ninv = mesh_quality(pos)
        flat = max((max(ah[n]) - min(ah[n])) / max(abs(np.mean(ah[n])), 1e-15) for n in ring_names)
        dflat = (max(dh) - min(dh)) / max(abs(np.mean(dh)), 1e-15)
        if ninv > 0 and not args_cli.allow_inversion:
            veto = f"n_inverted={ninv}"
        elif flat > args_cli.settle_tol:
            veto = f"area not flat ({100*flat:.2f} %)"
        elif dflat > args_cli.settle_tol:
            veto = f"displacement not flat ({100*dflat:.2f} %)"
        else:
            settled_at, veto = f, "none"
            break
    pos = obj.data.nodal_pos_w.torch[0].cpu().numpy().astype(np.float64)
    minJ, ninv = mesh_quality(pos)
    ar = {n: poly_area(pos[s_rings[n]][:, [0, 2]]) for n in ring_names}
    pct = {n: 100.0 * ar[n] / rest_area[n] for n in ring_names}
    # 🎯 THE INDEPENDENT CHECK THE PRESCRIBED-DISPLACEMENT RUN COULD NOT HAVE: with a punch the
    # skin is FREE, so how far it actually moved is a measurement rather than a restatement of
    # the command. A punch commanded to `depth` that produces near-zero skin travel means the
    # tissue was never loaded -- the 2026-08-08 failure mode.
    skin_u = float(np.abs(rest[s_foot, 2] - pos[s_foot, 2]).max())
    # 📌 WINDOW MEAN, NOT THE LAST SAMPLE.  Residual ringing was 3.2-7.9 % of the
    # reading on 2026-08-20; quoting one frame quotes one phase of it.  The spread
    # travels WITH the number so the two cannot be quoted apart.
    # equilibrium residual on interior free nodes (see the note where _interior is built)
    _fi = op.forces(pos)
    _near = np.abs(pos[:, 2] - (z_skin - cur)) < 2.0 * dx      # within 2*dx of the punch face
    _res_mask = _interior & (~_near)
    _rn = np.linalg.norm(_fi[_res_mask], axis=1)
    _fint_all = np.linalg.norm(_fi[s_free], axis=1)
    res_rms = float(np.sqrt((_rn ** 2).mean())) if _rn.size else float("nan")
    res_max = float(_rn.max()) if _rn.size else float("nan")
    # a residual is only meaningful against the size of the forces in play
    res_rel = res_rms / max(float(_fint_all.mean()), 1e-30)
    # 🎯 THE CHECK THAT ACTUALLY TESTS F_tissue, added 2026-08-27 because the 30 N cap is only a
    # cap if the reading is real. F_tissue is a SUM of f_int_z over free nodes, and it is the
    # probe load ONLY IF every free node away from the probe carries no other external force --
    # i.e. their z-forces SUM to ~0. An RMS cannot see that: a small systematic bias per node
    # cancels in RMS and accumulates in the sum. 28,159 interior nodes each biased by 1e-5 N is
    # 0.28 N, ~30 % of a 0.95 N reading, with RMS 2.7e-4 N looking perfectly healthy throughout.
    # ⚠️ AND `res_rel` ABOVE IS A MISLEADING NORMALISATION -- RMS-over-a-subset divided by
    # mean-over-the-superset exceeds 1 for any spread distribution, so its 240-300 % on the
    # 2026-08-25 presses reads like a failure and is not one. The absolute residual there was
    # 2.7e-4 N against a 0.95 N reading. Keep it for continuity; judge equilibrium on the SUM.
    _res_sum_z = float(_fi[_res_mask][:, 2].sum()) if _rn.size else float("nan")
    # how many particle pairs are close enough for self-contact to be carrying load
    from scipy.spatial import cKDTree as _KD
    _rad = 0.4 * dx
    _pairs = _KD(pos[s_free]).query_pairs(2.0 * _rad, output_type="ndarray")
    n_selfc = int(len(_pairs))
    print(f"  equilibrium residual on {int(_res_mask.sum()):,} interior free nodes: "
          f"RMS {res_rms:.4e} N  max {res_max:.4e} N  ({100*res_rel:.3f} % of mean |f_int|)"
          f"   self-contact pairs {n_selfc:,}", flush=True)
    fa = np.asarray(fh_t, dtype=np.float64)
    f_mean = float(fa.mean()) if len(fa) else float("nan")
    f_last = float(fa[-1]) if len(fa) else float("nan")
    f_spread = float((fa.max() - fa.min()) / max(abs(fa.mean()), 1e-15)) if len(fa) else float("nan")
    print(f"  F_tissue = {f_mean:+.4f} N  (window mean of {len(fa)}; last {f_last:+.4f} N; "
          f"peak-to-peak {100*f_spread:.2f} % of the mean)", flush=True)
    _leak = 100.0 * _res_sum_z / max(abs(f_mean), 1e-15)
    print(f"  ⚖️  independent check on that reading: sum of f_int_z over the "
          f"{int(_res_mask.sum()):,} interior free nodes = {_res_sum_z:+.4e} N "
          f"= {_leak:+.2f} % of F_tissue.  Nothing but the probe loads those nodes, so this "
          f"belongs at ~0; a large value means F_tissue is summing something that is not the "
          f"probe." + ("   🚩 OVER 5 % -- this step's force is NOT a measurement"
                       if abs(_leak) > 5.0 else ""), flush=True)
    row = dict(f_tissue_N=f_mean, f_tissue_last_N=f_last, f_tissue_ripple=f_spread,
               resid_rms_N=res_rms, resid_max_N=res_max, resid_rel=res_rel,
               resid_sum_z_N=_res_sum_z, resid_sum_z_pct_of_F=_leak,
               n_interior=int(_res_mask.sum()), n_selfcontact_pairs=n_selfc,depth_mm=target * 1e3, skin_u_mm=skin_u * 1e3, settled_at=settled_at, veto=veto,
               minJ=minJ, n_inverted=ninv, blew_up=blew_up,
               usable=(veto == "none" and ninv == 0 and not blew_up),
               **{f"{n}_pct": pct[n] for n in ring_names},
               **{f"{n}_mm2": ar[n] * 1e6 for n in ring_names})
    rows.append(row)
    write_results()          # partial file on disk from the first step onward
    # 🎯 AND THE RAW POSITIONS, per step (2026-08-27). The saved node positions are the only thing
    # from which a BETTER closure operator can be computed after the fact -- the lumen ring is grid
    # sampled, so its rest area differs by 41 % (vein) and 77 % (artery) between dx 2.5 and dx 1.25,
    # and any mesh-independent measure has to be rebuilt from the deformation field itself. Writing
    # them only at the end meant an interrupted trial lost exactly the data needed to fix its own
    # measurement. Costs a couple of MB per step.
    if args_cli.save_positions:
        _sp = dict(saved_pos, rest=rest.astype(np.float32), tets=solver_tets.astype(np.int32),
                   depths_mm=np.array(sorted(_saved_depths), dtype=np.float64),
                   footprint=s_foot.astype(np.int32),
                   **{f"ring_{n}": s_rings[n].astype(np.int32) for n in ring_names})
        np.savez_compressed(args_cli.save_positions, **_sp)
    _saved_depths.add(float(target * 1e3))
    saved_pos[f"pos_{_depth_key(target)}mm"] = pos.astype(np.float32)
    # worst elements, so the failure can be LOCATED rather than described
    pq = pos[solver_tets]
    Ds = np.stack([pq[:, 1] - pq[:, 0], pq[:, 2] - pq[:, 0], pq[:, 3] - pq[:, 0]], axis=2)
    Jall = np.linalg.det(Ds) / v0_ref
    worst = np.argsort(Jall)[:200]
    saved_pos[f"worstcen_{_depth_key(target)}mm"] = pos[solver_tets[worst]].mean(axis=1).astype(np.float32)
    saved_pos[f"worstJ_{_depth_key(target)}mm"] = Jall[worst].astype(np.float32)
    # 🎯 THE WITNESS: if the wrapper never fired, the pressure was never applied and any
    # "pressure makes no difference" conclusion would be about the harness, not the physics.
    if use_pressure:
        print(f"  pressure witness: clear_forces wrapper fired {_wrap_calls[0]:,} times "
              f"({'✅ applied' if _wrap_calls[0] > 0 else '🚨 NEVER APPLIED'})", flush=True)
    print(f"  settled_at={settled_at}  veto={veto}  minJ={minJ:.4f}  n_inverted={ninv}", flush=True)
    print("  " + "  ".join(f"{n}={pct[n]:.1f} % ({ar[n]*1e6:.2f} mm^2)" for n in ring_names), flush=True)

    # 🛑 ABSOLUTE FAILURE. Deeper depths after a diverged solve are not measurements of anything,
    # so the sequence ends here -- and the depth it ended at IS the finding this study is after.
    if blew_up:
        print(f"\n🛑 THE SOLVE DIVERGED at {target*1e3:.1f} mm ({veto}). Ending this mesh's "
              f"sequence. The rows already collected stand; THIS row does not (usable=False, and "
              f"its areas are non-finite). This is the mesh's limit, not a run to be retried at "
              f"the same settings.", flush=True)
        break

    # 🛑 SAFETY CAP. A deep sequence with inversion allowed has nothing else to stop it, and the
    # force rises faster than linearly once the tissue is compacted. 30 N by direction
    # 2026-08-26 -- well above the 10 N a compression exam is said to need, and chosen as a
    # runaway guard rather than as a physiological limit.
    if args_cli.stop_force_n is not None and rows[-1]["f_tissue_N"] > args_cli.stop_force_n:
        print(f"\n🛑 F_tissue {rows[-1]['f_tissue_N']:.2f} N exceeds the {args_cli.stop_force_n:.0f} N cap "
              f"at {target*1e3:.1f} mm -- abandoning the remaining depths. This is the cap firing, "
              f"NOT a solver failure; the rows already collected stand.", flush=True)
        break

rule("RESULT")
hdr = f"{'past skin':>10} {'skin moved':>11} " + " ".join(f"{n+' %':>9}" for n in ring_names) + \
      f" {'F_tissue N':>11} {'ripple %':>9} {'minJ':>8} {'n_inv':>6} {'usable':>7}"
print(hdr, flush=True)
for r in rows:
    print(f"{r['depth_mm']:>10.2f} {r['skin_u_mm']:>11.2f} " +
          " ".join(f"{r[n+'_pct']:>9.1f}" for n in ring_names) +
          f" {r['f_tissue_N']:>11.4f} {100*r['f_tissue_ripple']:>9.2f}"
          f" {r['minJ']:>8.4f} {r['n_inverted']:>6} {str(r['usable']):>7}", flush=True)

write_results(final=True)
print(f"\n>>> wrote {out}", flush=True)

if args_cli.trace and trace:
    np.savez_compressed(args_cli.trace, trace=np.asarray(trace, dtype=np.float64),
                        columns=np.array(["depth_mm", "settle_frame", "f_tissue_N", "max_disp_m"]
                                         + [f"area_{n}_m2" for n in ring_names]))
    print(f">>> wrote trace {args_cli.trace} ({len(trace)} frames)", flush=True)
if args_cli.save_positions:
    saved_pos["rest"] = rest.astype(np.float32)
    saved_pos["tets"] = solver_tets.astype(np.int32)
    saved_pos["footprint"] = s_foot.astype(np.int32)
    for n in ring_names:
        saved_pos[f"ring_{n}"] = s_rings[n].astype(np.int32)
    np.savez_compressed(args_cli.save_positions, **saved_pos)
    print(f">>> wrote {args_cli.save_positions}", flush=True)
print("PRESS DONE", flush=True)
app.close()
