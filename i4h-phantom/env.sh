# Paths and interpreters for every script in this directory. Every run script sources this file.
# Override any variable by exporting it before running a script.
#
# Default layout (see README.md, "Workspace layout"):
#   <workspace>/
#   ├── i4h-workflows/           NVIDIA Isaac for Healthcare workflows (builds the raysim image)
#   ├── IsaacLab/                NVIDIA Isaac Lab, release/3.0.0-beta2 at ffff603
#   ├── envs/isaaclab/           Python 3.12 venv: Isaac Sim 6.0.1 + Isaac Lab
#   ├── envs/physics/            Python 3.13 venv: Newton 1.4.0, gmsh (mesh building)
#   ├── envs/anatomy/            Python 3.13 venv: trimesh, scipy, matplotlib (geometry, figures, video)
#   └── femoral-sim-calibration/ this repository
#       └── i4h-phantom/         this directory

export I4H_PHANTOM="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$I4H_PHANTOM/.." && pwd)"
export WORKSPACE="${WORKSPACE:-$(cd "$REPO_ROOT/.." && pwd)}"

export PY_ISAACLAB="${PY_ISAACLAB:-$WORKSPACE/envs/isaaclab/bin/python}"
export PY_PHYSICS="${PY_PHYSICS:-$WORKSPACE/envs/physics/bin/python}"
export PY_ANATOMY="${PY_ANATOMY:-$WORKSPACE/envs/anatomy/bin/python}"

# the raysim container built by docker/Dockerfile.raysim (README.md, section 3)
export RAYSIM_IMAGE="${RAYSIM_IMAGE:-i4h_raysim:femoral}"

# the benchtop phantom record (measurements and B-mode frames); not distributed with this repository
export BENCHTOP_DIR="${BENCHTOP_DIR:-}"

for _v in PY_ISAACLAB PY_PHYSICS PY_ANATOMY; do
  [ -x "${!_v}" ] || echo "!!! env.sh: $_v=${!_v} is not an executable interpreter (set it, or follow README.md, section 4)" >&2
done
unset _v
