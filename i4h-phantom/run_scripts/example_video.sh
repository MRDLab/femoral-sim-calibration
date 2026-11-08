#!/usr/bin/env bash
# Quick start: one live run of the paper's chosen setting on the V5 leg, encoded to video.
#
#   bash run_scripts/example_video.sh            # writes out/example_video/example_s2_i35_plain_asrun_*fps.mp4
#
# The same run as the paper's video (clips/final_run/legV5_s2_i35_plain_asrun_18fps.mp4): 8.33 ms time step
# (2 substeps x 35 iterations, CUDA graph on), +-45 deg probe sweep round the limb, then the seven-depth press to 25.4 mm,
# with the Isaac Sim view and the live ray-traced B-mode every frame. Flags as run_scripts/paper_v5_pictures_20261021.sh.
#
# Steps: check the V5 leg mesh -> build the B-mode rest render set if missing -> start the raysim worker (docker)
# -> run the press (Isaac Lab) -> stop the worker -> encode.
# ⚠️ Frame rate depends on the GPU and on anything else using it. The paper's 56.4 ms/frame was an RTX 5090, idle.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$I4H_PHANTOM"
TAG=${TAG:-example_s2_i35}
M=out/mesh_leg_v5/tetmesh_leg_v5_s020_f1p5_c30_w30_bone
REST=$I4H_PHANTOM/out/live_render_rest_v5/travel_0.0mm
say(){ echo "[$(date +%F' '%H:%M:%S)] $*"; }

[ "$(sha256sum $M.npz | cut -c1-12)" = d14afb0ae307 ] || { say "!!! $M.npz is not the paper's V5 leg mesh (d14afb0ae307)"; exit 1; }
docker image inspect "$RAYSIM_IMAGE" > /dev/null 2>&1 || { say "!!! docker image $RAYSIM_IMAGE not found: README.md, section 3"; exit 1; }
command -v ffmpeg > /dev/null || { say "!!! ffmpeg not found (sudo apt install ffmpeg)"; exit 1; }

say "1/4 B-mode rest render set"
bash run_scripts/build_inputs.sh render_rest

say "2/4 raysim worker ($RAYSIM_IMAGE)"
mkdir -p out/live_bmode
docker rm -f bmode_worker > /dev/null 2>&1 || true
REST=$REST RIGID=Femur bash run_scripts/live_bmode_worker.sh > out/live_bmode/worker_$TAG.log 2>&1 &
W=0
until docker logs bmode_worker 2>&1 | grep -q READY; do
  sleep 2; W=$((W+2))
  [ $W -gt 900 ] && { say "!!! worker not READY after 900 s: see out/live_bmode/worker_$TAG.log"; docker rm -f bmode_worker > /dev/null 2>&1; exit 1; }
done
say "    $(docker logs bmode_worker 2>&1 | grep READY | sed 's/^\[worker\] //')"
say "    (the paper's render set reads: scene sha f4dcd823e93b)"

say "3/4 live press (Isaac Lab + Newton VBD); log: out/live_bmode/$TAG.log"
set +e
$PY_ISAACLAB -u isaaclab/live_press_bmode.py --headless \
  --npz $M.npz --usd $M.usda --substeps 2 --iterations 35 --cuda-graph --worker-port 5557 --pipeline --sweep-deg 45 \
  --viewer --save-every 1 --no-leg-visual --out out/live_bmode/$TAG.json > out/live_bmode/$TAG.log 2>&1
RC=$?
set -e
docker rm -f bmode_worker > /dev/null 2>&1 || true
[ $RC -eq 0 ] || { say "!!! press failed (rc=$RC): tail out/live_bmode/$TAG.log"; exit $RC; }
say "    $(grep -E 'frames:' out/live_bmode/$TAG.log | tail -1)"

say "4/4 encode"
$PY_ANATOMY analysis/make_live_video_plain.py out/live_bmode/$TAG.json out/example_video
ls -la out/example_video/*.mp4
