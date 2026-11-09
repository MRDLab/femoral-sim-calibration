#!/usr/bin/env bash
# Picture runs for the paper on Femoral V5 (2026-10-21), called once by run_scripts/paper_v5_rerun_20261021.sh between
# two timed runs, so they never share the GPU with one. Same settings as the V4 runs they replace:
#   legV5_s2_i35         <- out/live_bmode/legF_s2_i35          (2026-09-09 run, labels 2026-09-30): video, fig_pipeline frame
#   legV5_s2_i35_cuttex  <- out/live_bmode/legF_s2_i35_cuttex   (2026-10-12): cut-away, smooth CAD-section colours
#   legV5_s2_i35_cutsup  <- out/live_bmode/legF_s2_i35_cutsup_final2 (2026-10-12): cut-away, element colours
# All: 2 x 35, CUDA graph, pipeline, +-45 deg sweep then the seven-depth press, Isaac Sim viewer every frame, no leg visual,
# live raysim B-mode from the V5 rest render set (out/live_render_rest_v5, equal to V4's minus the ramus).
#
# ⚠️ Runs from the MAIN checkout ~/i4h-phantom, read only: the cut-away flags of live_press_bmode.py and cut_view.py
# exist there as uncommitted work. The hashes used are logged below. ILLUSTRATION ONLY: no timing or closure from the
# cut-away runs is quoted (cut-away doubles the viewer cost).
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$I4H_PHANTOM"
L=out/paper_v5/pictures_detail.log; : > $L
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $L; }
M=out/mesh_leg_v5/tetmesh_leg_v5_s020_f1p5_c30_w30_bone
TEX=out/cut_texture_meshV5
REST=$PWD/out/live_render_rest_v5/travel_0.0mm
[ "$(sha256sum $M.npz | cut -c1-12)" = d14afb0ae307 ] || { say "!!! leg mesh is not V5 d14afb0ae307"; exit 1; }
python3 -c "import json,sys; sys.exit(0 if json.load(open('$TEX/cut_texture.json'))['mesh_sha256_12']=='d14afb0ae307' else 1)" \
  || { say "!!! cut texture not made for d14afb0ae307"; exit 1; }
[ -f $REST/anatomy_config.json ] && [ ! -f $REST/PubicRamus.obj ] || { say "!!! V5 rest render set missing or holds a ramus"; exit 1; }
say "live_press_bmode $(sha256sum isaaclab/live_press_bmode.py | cut -c1-12); cut_view $(sha256sum isaaclab/cut_view.py | cut -c1-12);" \
    "bmode_worker $(sha256sum raysim_ultrasound/bmode_worker.py | cut -c1-12); main checkout HEAD $(git rev-parse --short HEAD)"
# `=` form: argparse reads a value that starts with '-' as a new option (first launch, 2026-10-21: rc 2 on both cut runs)
CAM="--viewer-eye=-0.126,0.192,0.183 --viewer-target=0.0,-0.01,0.035 --viewer-cut-y=0.0 --viewer-cut-keep=below"
FAILED=0
live() {  # tag extra-args...
  local TAG=$1; shift
  if [ -f out/live_bmode/$TAG.json ]; then say "SKIP $TAG (json exists)"; return; fi
  docker rm -f bmode_worker > /dev/null 2>&1
  REST=$REST RIGID=Femur bash run_scripts/live_bmode_worker.sh > out/live_bmode/worker_$TAG.log 2>&1 &
  local W=0; until docker logs bmode_worker 2>&1 | grep -q READY; do sleep 2; W=$((W+2))
    [ $W -gt 900 ] && { say "!!! $TAG: worker not READY after 900 s -- skipped"; docker rm -f bmode_worker > /dev/null 2>&1; FAILED=$((FAILED+1)); return; }; done
  say "START $TAG (worker ready after $W s)"
  $PY_ISAACLAB -u isaaclab/live_press_bmode.py --headless \
    --npz $M.npz --usd $M.usda --substeps 2 --iterations 35 --cuda-graph --worker-port 5557 --pipeline --sweep-deg 45 \
    --viewer --save-every 1 --no-leg-visual "$@" --out out/live_bmode/$TAG.json > out/live_bmode/$TAG.log 2>&1
  local RC=$?
  docker rm -f bmode_worker > /dev/null 2>&1
  # every frame must be saved: the video composer needs them all (attempts 1-2 saved every 5th, the default)
  if [ $RC -eq 0 ] && ! $PY_ANATOMY -c "import json,sys,numpy as np; t='out/live_bmode/$TAG'; \
       n=sum(k.startswith('f') for k in np.load(t+'.frames.npz').files); m=len(json.load(open(t+'.json'))['timeline']); \
       print(f'saved {n} of {m} frames'); sys.exit(0 if n == m else 1)" >> $L 2>&1; then RC=3; say "!!! $TAG: not every frame saved"; fi
  [ $RC -ne 0 ] && FAILED=$((FAILED+1))
  say "DONE $TAG rc=$RC: $(grep -E 'frames:' out/live_bmode/$TAG.log | tail -1)"
}
live legV5_s2_i35
live legV5_s2_i35_cuttex $CAM --viewer-cut-texture $TEX
live legV5_s2_i35_cutsup $CAM
if [ $FAILED -gt 0 ]; then say "!!! PICTURES: $FAILED run(s) FAILED"; exit 1; fi
say "PICTURES ALL DONE"
