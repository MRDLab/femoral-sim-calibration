#!/bin/bash
# The plan's supplier-spread cases (2026-10-29): the cast case with both gels at -1 SD and +1 SD (supplier batch spread,
# combined by mass: fat 0.039, muscle 0.055 MPa). Same section, walls, free femur 200 MPa, production settings, 5 repeats.
# Waits for the main queue (its reference run) to finish first. Accuracy only.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
PYP=$PY_PHYSICS
PYL=$PY_ISAACLAB
M=$I4H_PHANTOM/out/mesh_leg_cast
OUT=$I4H_PHANTOM/out/paper_cast/gel_sd
mkdir -p $OUT
cd "$I4H_PHANTOM"
until grep -q "ALL DONE" $I4H_PHANTOM/out/paper_cast/main/queue.log; do sleep 60; done
for SD in -1 1; do
  NAME=tetmesh_leg_cast_s020_f1p5_c30_w30_bend1p5_femurfree200_gel${SD}sd
  echo "[$(date '+%F %T')] BUILD $NAME" >> $OUT/queue.log
  $PYP -u anatomy_meshing/graded_leg_mesh.py --anatomy-dir femoral_cast_bend1p5 --materials cast --femur-free-mpa 200 --gel-sd $SD \
    --station-s 20 --fine-dx-mm 1.5 --coarse-dx-mm 30 --grade-mm 25 --arc-deg 45 --s-window 0 30 --fine-depth-mm auto \
    --sheath-kpa 20 --table-cap-mm 6 --bone-dx-mm 5 --bone-margin-mm 4 --ramus-dx-mm 2.5 --nsub 11 --leg-s-range 0 81 \
    --out $M/$NAME.npz > $M/build_gel${SD}sd.log 2>&1 || { echo "[$(date '+%F %T')] BUILD FAILED $NAME" >> $OUT/queue.log; continue; }
  $PYL isaaclab/author_tetmesh_usd.py --npz $M/$NAME.npz > $M/usd_gel${SD}sd.log 2>&1
  for R in 1 2 3 4 5; do
    TAG=cast_gel${SD}sd_r${R}_s2_i35
    echo "[$(date '+%F %T')] START $TAG" >> $OUT/queue.log
    timeout 7200 $PYL -u isaaclab/live_press_bmode.py --headless --npz $M/$NAME.npz --usd $M/$NAME.usda \
      --substeps 2 --iterations 35 --cuda-graph --no-worker --no-leg-visual --out $OUT/$TAG.json > $OUT/$TAG.log 2>&1
    echo "[$(date '+%F %T')] DONE $TAG rc=$?" >> $OUT/queue.log
  done
done
echo "[$(date '+%F %T')] ALL DONE" >> $OUT/queue.log
