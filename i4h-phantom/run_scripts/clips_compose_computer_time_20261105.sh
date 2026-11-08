#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
# 2026-11-05: comparison clips on COMPUTER time (make_comparison_video.py --clock compute), REPLACING the simulated-time ones (by direction, 2026-11-05: keep only the newest).
# Frames are NOT deleted here.
set -euo pipefail
cd "$I4H_PHANTOM"
PY=$PY_ANATOMY; LB=out/live_bmode; CL=$REPO_ROOT/clips
for pair in "s3_i140 intermediate" "s8_i140 longer"; do set -- $pair; SI=$1; FOLDER=$2
  OUT=$CL/$FOLDER/comparison; mkdir -p $OUT
  CAP=""; [ $FOLDER = intermediate ] && CAP="--cap-s 180 --cap-mb 10"
  for V in "" _cuttex _cutsup; do VIEW=${V#_}; VIEW=${VIEW:-plain}
    $PY analysis/make_comparison_video.py $LB/legV5_cmp_s2_i35${V}.json $LB/legV5_cmp_${SI}${V}.json \
        $OUT/final_vs_${FOLDER}_${VIEW}.mp4 --names "final run,$FOLDER" --clock compute $CAP
  done
done
echo "ALL CLIPS WRITTEN"; ls -la $CL/*/comparison/
