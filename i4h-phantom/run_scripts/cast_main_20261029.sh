#!/bin/bash
# THE CAST CASE (2026-10-29): bending-equivalent tube walls (the plan's stated fallback; resolved walls froze), femur free
# at 200 MPa (the cast's printed bone is held only by gel), V5 production settings (8.33 ms x 35, contact 2e3 N/m, self-
# contact as V5). Five repeats, plus the accuracy reference (0.099 ms x 140) on the same mesh. Accuracy only: runs share
# the GPU, so no timing from here is quotable.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
PY=$PY_ISAACLAB
MESH=$I4H_PHANTOM/out/mesh_leg_cast/tetmesh_leg_cast_s020_f1p5_c30_w30_bend1p5_femurfree200
OUT=$I4H_PHANTOM/out/paper_cast/main
mkdir -p $OUT
cd "$I4H_PHANTOM"
run() { local TAG=$1 S=$2 I=$3
  echo "[$(date '+%F %T')] START $TAG" >> $OUT/queue.log
  timeout 21600 $PY -u isaaclab/live_press_bmode.py --headless --npz $MESH.npz --usd $MESH.usda \
    --substeps $S --iterations $I --cuda-graph --no-worker --no-leg-visual --out $OUT/$TAG.json > $OUT/$TAG.log 2>&1
  echo "[$(date '+%F %T')] DONE $TAG rc=$?" >> $OUT/queue.log; }
run cast_ref_s168_i140 168 140 &
for R in 1 2 3 4 5; do run cast_r${R}_s2_i35 2 35; done
echo "[$(date '+%F %T')] REPEATS DONE" >> $OUT/queue.log
wait
echo "[$(date '+%F %T')] ALL DONE" >> $OUT/queue.log
