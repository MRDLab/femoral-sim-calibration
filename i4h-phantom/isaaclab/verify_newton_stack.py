"""Does the pip-installed Isaac Sim + Newton stack give us what the docker image could not?

Three gates, in dependency order. Each is a thing the docker route failed or could not answer.

  1. VERSIONS   warp new enough for Newton's tile builtins, and newton importable alongside it.
                The docker image deadlocked here: replicator needed warp's pre-1.13 `warp.context`
                layout while Newton's collision kernels need post-1.12 `wp.tile_*`.
  2. COLLISION  CollisionPipeline must actually STEP, not merely construct. The docker smoke test
                passed construction and then failed compiling `narrow_phase` on first step -- with
                no mesh anywhere in the scene, so it was never a VOXEL-MAN problem.
  3. CAMERAS    replicator must survive in the same process. Non-negotiable: the cameras carry the
                policy observations and the B-mode pose bridge. A Newton stack without them is
                useless to this project.

Gate 3 is the one that decides whether tissue and B-mode can live in one scene, or whether we are
stuck with a two-process split.

    python verify_newton_stack.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback

report: dict = {}


def gate(name):
    def deco(fn):
        print(f"\n=== {name} ===", flush=True)
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
            traceback.print_exc()
        report[name] = {"ok": bool(ok), "detail": detail}
        print(f"  --> {name}: {'PASS' if ok else 'FAIL'}  {detail}", flush=True)
        return fn

    return deco


@gate("1_versions")
def _versions():
    import warp as wp

    import newton

    print(f"  warp   {wp.__version__}")
    print(f"  newton {newton.__version__}")
    # ⚠️ Do NOT probe tile builtins with hasattr(wp, ...). Warp registers them in the CODEGEN
    # layer, not as Python attributes, so hasattr returns False even on a warp that has them --
    # it reports False on warp 1.13.0, which demonstrably compiles Newton's collision kernels.
    # Gate 2 (does narrow_phase actually compile and step) is the only valid test.
    ok = tuple(int(x) for x in wp.__version__.split(".")[:2]) >= (1, 13)
    return ok, f"warp {wp.__version__} / newton {newton.__version__} (tile check deferred to gate 2)"


@gate("2_collision_steps")
def _collision():
    """Construct AND step -- construction alone passed in the docker image and still failed."""
    import numpy as np
    import warp as wp

    import newton

    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    b.add_soft_grid(pos=wp.vec3(-0.08, -0.08, 0.0), rot=wp.quat_identity(), vel=wp.vec3(0, 0, 0),
                    dim_x=16, dim_y=16, dim_z=5, cell_x=0.01, cell_y=0.01, cell_z=0.01,
                    density=1050.0, k_mu=10344.8, k_lambda=93103.4, k_damp=1.0,
                    particle_radius=0.0025)
    link = b.add_link(xform=wp.transform(wp.vec3(0, 0, 0.14), wp.quat_identity()), mass=0.2)
    b.add_shape_sphere(body=link, radius=0.04)
    j = b.add_joint_prismatic(
        parent=-1, child=link, axis=wp.vec3(0, 0, 1),
        parent_xform=wp.transform(wp.vec3(0, 0, 0.14), wp.quat_identity()),
        target_pos=0.0, target_ke=2e5, target_kd=2e3,
        actuator_mode=newton.JointTargetMode.POSITION)
    if hasattr(b, "add_articulation"):
        try:
            b.add_articulation([int(j)])
        except TypeError:
            pass
    b.color()
    m = b.finalize()
    s = newton.solvers.SolverVBD(m, iterations=10, rigid_body_particle_contact_buffer_size=20000)
    cp = newton.CollisionPipeline(m, soft_contact_margin=0.01)
    st0, st1, ctl = m.state(), m.state(), m.control()
    c = cp.contacts()
    newton.eval_fk(m, m.joint_q, m.joint_qd, st0)
    t = ctl.joint_target_pos.numpy()
    t[0] = -0.03
    ctl.joint_target_pos.assign(t)
    for _ in range(200):
        cp.collide(st0, c)
        s.step(st0, st1, ctl, c, 1.0 / 1440.0)
        st0, st1 = st1, st0
    pq = st0.particle_q.numpy()
    nc = int(c.soft_contact_count.numpy()[0])
    finite = bool(np.isfinite(pq).all())
    print(f"  stepped 200x | particles {len(pq)} | soft contacts {nc} | finite {finite}")
    return finite, f"{len(pq)} particles, {nc} contacts, finite={finite}"


@gate("3_cameras_with_newton")
def _cameras():
    """replicator must still work in a process that has imported Newton."""
    import newton  # noqa: F401  -- imported FIRST, on purpose

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True, enable_cameras=True).app
    try:
        import omni.replicator.core as rep

        has_create = hasattr(rep, "create")
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import CameraCfg
        from isaaclab.sim import SimulationCfg, SimulationContext

        sim = SimulationContext(SimulationCfg(dt=1 / 60, device="cuda:0"))
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        from isaaclab.sensors import Camera

        cam = Camera(CameraCfg(
            prim_path="/World/cam", update_period=0.0, height=64, width=64,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.1, 20.0)),
        ))
        sim.reset()
        for _ in range(3):
            sim.step()
            cam.update(1 / 60)
        rgb = cam.data.output["rgb"]
        shape = tuple(rgb.shape)
        print(f"  replicator.create: {has_create} | camera rgb shape {shape}")
        return has_create and len(shape) >= 3, f"rep.create={has_create}, rgb={shape}"
    finally:
        try:
            app.close()
        except Exception:
            pass


print("\n================ SUMMARY ================")
for k, v in report.items():
    print(f"  {k:<26} {'PASS' if v['ok'] else 'FAIL'}   {v['detail']}")
json.dump(report, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "verify_report.json"), "w"), indent=2)
print("\nwrote out/verify_report.json")
sys.exit(0 if all(v["ok"] for v in report.values()) else 1)
