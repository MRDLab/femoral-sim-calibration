#!/usr/bin/env bash
# Picture runs for the clip comparison (2026-11-05): same leg, motion, renderer and code; only the solver setting changes.
#   legV5_cmp_s2_i35   8.33 ms step, 35 iterations  (the chosen setting, re-run so all three share this code version)
#   legV5_cmp_s3_i140  5.56 ms step, 140 iterations (Table tab:speed row; predicted ~2.1-2.7 min as run -> the <=3 min clip)
#   legV5_cmp_s8_i140  2.08 ms step, 140 iterations (Table tab:speed row; predicted ~5.2-6.2 min as run -> the long clip)
# Time step = 16.67 ms frame / substeps. Flags otherwise identical to paper_v5_pictures_20261021.sh (legV5_s2_i35).
# ILLUSTRATION ONLY: single runs; no closure value from them is quoted. Wall-clock per frame is what the clips show.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$I4H_PHANTOM"
L=out/live_bmode/clips_comparison_20261105.log; : > $L
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $L; }
M=out/mesh_leg_v5/tetmesh_leg_v5_s020_f1p5_c30_w30_bone
REST=$PWD/out/live_render_rest_v5/travel_0.0mm
[ "$(sha256sum $M.npz | cut -c1-12)" = d14afb0ae307 ] || { say "!!! leg mesh is not V5 d14afb0ae307"; exit 1; }
[ -f $REST/anatomy_config.json ] && [ ! -f $REST/PubicRamus.obj ] || { say "!!! V5 rest render set missing or holds a ramus"; exit 1; }
say "live_press_bmode $(sha256sum isaaclab/live_press_bmode.py | cut -c1-12); bmode_worker $(sha256sum raysim_ultrasound/bmode_worker.py | cut -c1-12); HEAD $(git rev-parse --short HEAD)"
FAILED=0
live() {  # TAG SUBSTEPS ITERATIONS
  local TAG=$1 S=$2 I=$3
  if [ -f out/live_bmode/$TAG.json ]; then say "SKIP $TAG (json exists)"; return; fi
  docker rm -f bmode_worker > /dev/null 2>&1
  REST=$REST RIGID=Femur bash run_scripts/live_bmode_worker.sh > out/live_bmode/worker_$TAG.log 2>&1 &
  local W=0; until docker logs bmode_worker 2>&1 | grep -q READY; do sleep 2; W=$((W+2))
    [ $W -gt 900 ] && { say "!!! $TAG: worker not READY after 900 s -- skipped"; docker rm -f bmode_worker > /dev/null 2>&1; FAILED=$((FAILED+1)); return; }; done
  say "START $TAG substeps=$S iterations=$I (worker ready after $W s)"
  $PY_ISAACLAB -u isaaclab/live_press_bmode.py --headless \
    --npz $M.npz --usd $M.usda --substeps $S --iterations $I --cuda-graph --worker-port 5557 --pipeline --sweep-deg 45 \
    --viewer --save-every 1 --no-leg-visual --out out/live_bmode/$TAG.json > out/live_bmode/$TAG.log 2>&1
  local RC=$?
  docker rm -f bmode_worker > /dev/null 2>&1
  if [ $RC -eq 0 ] && ! $PY_ANATOMY -c "import json,sys,numpy as np; t='out/live_bmode/$TAG'; \
       n=sum(k.startswith('f') for k in np.load(t+'.frames.npz').files); m=len(json.load(open(t+'.json'))['timeline']); \
       print(f'saved {n} of {m} frames'); sys.exit(0 if n == m else 1)" >> $L 2>&1; then RC=3; say "!!! $TAG: not every frame saved"; fi
  [ $RC -ne 0 ] && FAILED=$((FAILED+1))
  say "DONE $TAG rc=$RC: $(grep -E 'frames:' out/live_bmode/$TAG.log | tail -1)"
}
live legV5_cmp_s2_i35 2 35
live legV5_cmp_s3_i140 3 140
live legV5_cmp_s8_i140 8 140
if [ $FAILED -gt 0 ]; then say "!!! COMPARISON RUNS: $FAILED FAILED"; exit 1; fi
say "COMPARISON RUNS ALL DONE"
