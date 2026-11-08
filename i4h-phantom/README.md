# i4h-phantom: the simulation behind the paper

This directory is the code that produced every simulated result, figure, table and video in the paper.
It simulates a probe pressing a femoral leg phantom and renders the ultrasound image live.

- **Tissue:** Newton's Vertex Block Descent (VBD) solver on a tetrahedral leg mesh, inside NVIDIA Isaac Lab.
- **Ultrasound image:** NVIDIA's ray-traced ultrasound simulator (raysim), with one patch of ours.
- **Scene view:** Isaac Sim renders the probe and the cut-open leg.

It builds on NVIDIA's software and sits **beside** it. Nothing is installed inside NVIDIA's repositories.
The only edit to NVIDIA code is a 10-file patch to raysim, applied by a Dockerfile (step 2).

**Contents:** [layout](#1-workspace-layout) · [clone](#2-clone) · [raysim container](#3-the-raysim-container-i4h-workflows) ·
[Python environments](#4-python-environments-isaac-sim-isaac-lab) · [example video](#5-quick-start-the-example-video) ·
[every result](#6-reproducing-every-figure-and-table) · [not included](#7-what-is-not-included) · [files](#8-files)

## Requirements

Tested on one machine. Other configurations may work but are untested.

| | tested | NVIDIA's minimum |
|---|---|---|
| OS | Ubuntu 24.04, x86_64 | Ubuntu 22.04 or 24.04 |
| GPU | RTX 5090, 32 GB, driver 595.58.03 | RTX GPU, compute capability ≥ 8.6 |
| RAM | 62 GB | 64 GB |
| disk | ~110 GB free | ~100 GB (the raysim image alone is 73 GB) |
| software | Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), git, `ffmpeg`, [`uv`](https://docs.astral.sh/uv/) | |

## 1. Workspace layout

Every path in this repository is relative to one workspace directory. `env.sh` assumes this layout:

```
<workspace>/
├── i4h-workflows/            NVIDIA Isaac for Healthcare workflows, commit 0a2affb (builds the raysim image)
├── IsaacLab/                 NVIDIA Isaac Lab, release/3.0.0-beta2, commit ffff603 (unmodified)
├── envs/
│   ├── isaaclab/             Python 3.12: Isaac Sim 6.0.1 + Isaac Lab      (the press and the scene view)
│   ├── physics/              Python 3.13: Newton 1.4.0, gmsh               (mesh building)
│   └── anatomy/              Python 3.13: trimesh, scipy, matplotlib       (geometry, figures, tables, video)
└── femoral-sim-calibration/  this repository
    └── i4h-phantom/          this directory
```

A different layout works: export `PY_ISAACLAB`, `PY_PHYSICS`, `PY_ANATOMY` or `RAYSIM_IMAGE` before running (`env.sh`).

## 2. Clone

```bash
mkdir -p ~/femoral-ws && cd ~/femoral-ws

git clone https://github.com/MRDLab/femoral-sim-calibration.git

git clone https://github.com/isaac-for-healthcare/i4h-workflows.git
git -C i4h-workflows checkout 0a2affbfdb821b3e3e82a59fdb0d705fe824ead6

git clone https://github.com/isaac-sim/IsaacLab.git
git -C IsaacLab checkout ffff603eafc6b74264a5261cc0183d6a65390d78     # branch release/3.0.0-beta2
```

## 3. The raysim container (i4h-workflows)

The ultrasound image comes from raysim, part of NVIDIA's robotic-ultrasound workflow. It runs in that workflow's container.

**3a. Build NVIDIA's container, unchanged.** Follow NVIDIA's guide:
[robotic_ultrasound README at 0a2affb](https://github.com/isaac-for-healthcare/i4h-workflows/blob/0a2affbfdb821b3e3e82a59fdb0d705fe824ead6/workflows/robotic_ultrasound/README.md)
(system requirements and prerequisites).

```bash
cd ~/femoral-ws/i4h-workflows
./i4h build-container robotic_ultrasound          # -> image i4h_build-robotic_ultrasound:main (~73 GB, about an hour)
```

**3b. Apply our edit.** One Dockerfile applies `docker/raysim_femoral.patch` to raysim inside that image and recompiles it.

```bash
cd ~/femoral-ws/femoral-sim-calibration/i4h-phantom
docker build -f docker/Dockerfile.raysim --build-arg BASE=i4h_build-robotic_ultrasound:main \
  -t i4h_raysim:femoral docker/
docker run --rm --gpus all --entrypoint bash i4h_raysim:femoral -lc \
  "python -c 'import raysim.cuda as rs; print(rs.Materials().get_index(\"air\"))'"      # must print 6
```

The patch is against raysim at commit `de8fe6f` (`i4h-sensor-simulation`, Apache-2.0). The Dockerfile refuses any other commit.

| file in `ultrasound-raytracing/` | what the patch adds |
|---|---|
| `csrc/core/material.cpp` | an `air` material. A probe not touching the skin then images black, as a real one does |
| `csrc/core/raytracing_ultrasound_simulator.cpp`, `include/.../raytracing_ultrasound_simulator.hpp` | `SimParams.initial_material`: the medium rays start in (the coupling gel) |
| `csrc/core/hitable.cpp`, `csrc/core/world.cpp`, `csrc/cuda/optix_trace.cu`, their headers | update a mesh's vertices in place on the GPU and rebuild the ray-tracing structure (3 ms) |
| `csrc/python/raysim_bindings.cpp` | Python access: `Mesh.set_vertices(_device)`, `set_normals(_device)`, `get_vertices()`, `rebuild_geometry()` |

⚠️ The stock image also ships Isaac Sim 5.1. On our driver it crashes when rendering (IsaacSim issue #677).
That does not matter here: only raysim is used from this image. The scene view uses Isaac Sim 6.0.1 (step 4).

## 4. Python environments (Isaac Sim, Isaac Lab)

**No edits to Isaac Sim or Isaac Lab.** Both are installed as NVIDIA publishes them, at the versions below.

**4a. Isaac Sim 6.0.1 and Isaac Lab (`envs/isaaclab`).** NVIDIA's guides:
[Isaac Sim pip install](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html) ·
[Isaac Lab pip install](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).

```bash
cd ~/femoral-ws
uv venv --python 3.12 --seed envs/isaaclab && source envs/isaaclab/bin/activate
pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com
cd IsaacLab && ./isaaclab.sh --install && cd ..
deactivate
```

Then check the result against `requirements/isaaclab-frozen.txt`. These versions matter most:
isaacsim 6.0.1.0 · isaaclab 6.1.14 · isaaclab_contrib 0.4.4 (deformable bodies) · newton 1.2.1 · warp-lang 1.13.0.
Complete NVIDIA's first-run steps for Isaac Sim, as its install guide describes, before running anything here.

**4b. The two host environments.** Exact versions, from the environments that produced the paper:

```bash
uv venv --python 3.13 --seed envs/physics && envs/physics/bin/pip install -r femoral-sim-calibration/i4h-phantom/requirements/physics.txt
uv venv --python 3.13 --seed envs/anatomy && envs/anatomy/bin/pip install -r femoral-sim-calibration/i4h-phantom/requirements/anatomy.txt
```

**4c. Check the stack.** Three checks: versions compatible, the collision pipeline steps, cameras work in the same process.

```bash
cd ~/femoral-ws/femoral-sim-calibration/i4h-phantom
source env.sh                                   # warns about any interpreter it cannot find
$PY_ISAACLAB isaaclab/verify_newton_stack.py    # writes out/verify_report.json
```

⚠️ Our `envs/isaaclab` was built in August 2026 and not rebuilt from these commands since.
`requirements/isaaclab-frozen.txt` is its exact package list; a rebuild that differs from it is not the tested stack.

## 5. Quick start: the example video

One live run of the paper's chosen setting, encoded to video. This is the end-to-end test of steps 2 to 4.

```bash
cd ~/femoral-ws/femoral-sim-calibration/i4h-phantom
bash run_scripts/example_video.sh
```

What it does, and what to check:

| step | output | a correct run shows |
|---|---|---|
| 1. rest render set (first run only, ~1 min) | `out/live_render_rest_v5/` | 12 bodies |
| 2. raysim worker | `READY ... scene sha f4dcd823e93b` | exactly that digest: the same anatomy the paper imaged |
| 3. live press | `out/live_bmode/example_s2_i35.{json,log,depths.csv}` | `veto none` at all seven depths; vein ≈ 28 %, artery ≈ 83 % of rest area at 25.4 mm |
| 4. encode | `out/example_video/example_s2_i35_plain_asrun_*fps.mp4` | the paper's video, `clips/final_run/legV5_s2_i35_plain_asrun_18fps.mp4` |

The paper's run read vein 28.27 % and artery 83.40 % at 25.4 mm. A run on a busy GPU can differ by about one point.
The frame rate depends on the GPU and on anything else running on it, so do not compare it with the paper.

The cut-away versions add three flags (`run_scripts/paper_v5_pictures_20261021.sh`, runs `legV5_s2_i35_cuttex` and `_cutsup`).

## 6. Reproducing every figure and table

**6a. Meshes.** The run scripts check every mesh by its sha256 and refuse anything else.

- The V5 anatomy and the V5 leg mesh are in this repository (`out/femoral_v5/`, `out/mesh_leg_v5/`).
- The slab, sensitivity and cast meshes are a release asset. Download it and unpack it here:

```bash
cd ~/femoral-ws/femoral-sim-calibration/i4h-phantom
curl -LO https://github.com/MRDLab/femoral-sim-calibration/releases/download/v1.0/paper-meshes-v1.tar.gz
sha256sum -c paper-meshes-v1.tar.gz.sha256 && tar xzf paper-meshes-v1.tar.gz     # 47 MB; unpacks into out/
```

`run_scripts/build_inputs.sh all` rebuilds the same meshes from the distributed anatomy instead.
⚠️ A rebuilt mesh never has the original's sha256: numpy writes the build time into every `.npz`.
`anatomy_meshing/compare_mesh.py` confirms a rebuild is the same mesh, array by array. We checked this for the 1.5 mm slab.

**6b. Runs.** GPU runs, one at a time. Times are wall clock on our RTX 5090.

| script | produces | time |
|---|---|---|
| `run_scripts/paper_v5_rerun_20261021.sh` | the timed queue: 11 leg settings × 5 repeats, references, sensitivity cases, inversion audit, slab studies → `out/paper_v5/` | 7 days (2026-10-21 to 10-28); `touch out/paper_v5/STOP` stops it cleanly |
| `run_scripts/paper_v5_pictures_20261021.sh` | the three picture runs: plain, cut-away smooth, cut-away element colours → `out/live_bmode/legV5_s2_i35*` | ~5 min |
| `run_scripts/cast_main_20261029.sh`, then `cast_gel_sd_20261029.sh` | the benchtop-cast case: reference and 5 repeats, then gel ±1 SD → `out/paper_cast/` | ~1 h (the reference shares the GPU with the repeats: accuracy only, no timing) |
| `run_scripts/clips_comparison_20261105.sh`, `clips_comparison_cut_20261105.sh`, `clips_compose_computer_time_20261105.sh` | the comparison videos in `clips/` | ~30 min |

`paper_v5_rerun_20261021_order2.sh` is the same queue in the order we finished it (element sizes before the slow references).
Runs on a GPU shared with another process are logged `SHARED` and excluded from the analysis.

**6c. Figures and tables.** Run from `i4h-phantom/` with `$PY_ANATOMY`, in this order. They write to the repository's
`figures/`, `tables/` and `data/`.

```bash
source env.sh
for s in analyze_meshF_repeats headline_numbers analyze_sensitivity audit_slab_inversion audit_leg_inversion \
         fig_leg_vein_vs_artery fig_slab_studies fig_slab_and_sweep make_tables verify_tables; do
  $PY_ANATOMY paper/$s.py || break
done                                            # verify_tables.py must end with "0 failures"
```

| paper | file in `figures/` or table | script |
|---|---|---|
| Fig. 2, Table S5 | `fig_cast_validation.png` | `fig_cast_validation.py` (needs the benchtop record, section 7) |
| Fig. 3, Fig. S6 | `fig_meshF_wallclock_per_frame_top_panel.png`, `fig_meshF_wallclock_residual.png` | `analyze_meshF_repeats.py` |
| Fig. 4 | `fig_leg_vein_vs_artery.png` | `fig_leg_vein_vs_artery.py` |
| Fig. S3 | `fig_slab_and_sweep.png` | `fig_slab_and_sweep.py` |
| Fig. S4, Fig. S5 | `fig_slab_element_size.png`, `fig_slab_timestep_collapse.png` | `fig_slab_studies.py` |
| Fig. S7 | `fig_cast_segmentation_clean_wide.png` | `fig_cast_segmentation.py` (needs the benchtop record) |
| Tables I, III, S1–S4, S6–S8 | `tables/*.tex`, every cell with its source in `tables/values.json` | `make_tables.py`; `verify_tables.py` re-derives every cell |
| abstract and conclusion numbers | `data/headline_numbers.json` | `headline_numbers.py` |
| Section S2: wall clock per solver pass, inverted elements | `data/slab_inversion/`, `data/leg_inversion/`, `data/slab_cost/per_pass.csv` | `audit_slab_inversion.py`, `audit_leg_inversion.py`, `fig_slab_studies.py` |

The `data/` files the supplement cites are not distributed. The scripts above regenerate them from the runs.

## 7. What is not included

- **The benchtop record.** The cast phantom's B-mode frames and lumen measurements are not distributed.
  Fig. 2, Fig. S7 and Table S5 need them (`BENCHTOP_DIR`). Table S5 prints every measured value.
  The measurement code and the operator's picks are included: `analysis/measure_lumen.py` finds the wall echoes on a
  column profile, and `analysis/benchtop_picks.csv` holds the inner-wall pair the operator chose in each frame
  (54 picks: 9 presses × 3 trials × 2 vessels, in image pixels at 22.24 px/mm).
- **Fig. 1, Fig. S1, Fig. S2, Table II.** Fig. 1 is a photo composite and Fig. S2 a drawing, both made by hand.
  Fig. S1 is a TikZ diagram and Table II was written by hand; the paper's LaTeX sources are not distributed.
- **The CAD-to-V5 derivation.** `anatomy_meshing/v4_to_section.py` and `v5_from_v4.py` are included for the record.
  They need the V4 SolidWorks export, which is not distributed. The V5 anatomy they produced is (`out/femoral_v5/`).
- **Raw run outputs.** About 5 GB. Everything in sections 5 and 6 regenerates them.

📌 **Script versions.** These files are a snapshot of our working repository, trimmed to the paper.
`isaaclab/live_press_bmode.py` has gained options since the leg runs (digest `ae1c95e1f35b`): `--no-self-contact`,
`--self-contact-dx-mm`, `--viewer-eye/target`, `--viewer-cut-*`, and a check that a NaN can never count as settled.
Every new option defaults to the earlier behaviour. Each run log records the digest of the script that produced it.

## 8. Files

| path | what it is |
|---|---|
| `env.sh` | paths and interpreters; every run script sources it |
| `materials.py` | the one property table: every tissue's Young's modulus, Poisson ratio and density, with its source |
| `femoral_fem.py`, `closure_ops.py`, `probe_geometry.py`, `v3_section.py`, `snh_force.py`, `repo_paths.py` | shared libraries: the slab mesher, lumen-closure measures, the one probe and gel geometry, the cross-section reader |
| `anatomy_meshing/` | anatomy in, simulator meshes out: leg and slab meshes, sensitivity and cast meshes, the B-mode render set, the gel band |
| `isaaclab/` | the press in Isaac Lab (`live_press_bmode.py`), the slab press (`press_femoral_punch.py`), USD authoring, the cut-away view |
| `raysim_ultrasound/bmode_worker.py` | the live B-mode worker: receives node positions over TCP, warps the render meshes, ray-traces, returns the image |
| `analysis/` | video encoding (`make_live_video_plain.py`, `make_comparison_video.py`, `make_submission_video.py`), cast lumen measurement, cut-face texture |
| `paper/` | every figure and table of the paper and supplement, from the run outputs (`paper_meshes.py` names every input and digest) |
| `run_scripts/` | the exact launchers behind the paper's runs, plus `example_video.sh` and `build_inputs.sh` |
| `docker/` | `Dockerfile.raysim` and `raysim_femoral.patch` (step 3) |
| `requirements/` | the package lists of the three environments |
| `out/femoral_v5/` | the V5 anatomy: 11 bodies as OBJ (LPS, mm), cross-sections at s = 6 and 20 mm |
| `out/mesh_leg_v5/` | the V5 leg mesh: 23,688 nodes, 129,783 tetrahedra, sha256 `d14afb0ae307`, as `.npz` and `.usda` |
| `out/cut_texture_meshV5/` | the cut-face texture for the cut-away videos |
