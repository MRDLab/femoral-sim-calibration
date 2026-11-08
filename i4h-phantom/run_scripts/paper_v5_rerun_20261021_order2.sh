#!/usr/bin/env bash
# Every run behind the paper, repeated on Femoral V5 (2026-10-21).
# V5 = V4 with the pubic ramus removed and the limb cut at s = 81 mm (anatomy_meshing/v5_from_v4.py), matching the cast
# gel phantom. Same settings, flags, idle-GPU rule and watchdog as the 2026-09-30 launchers it replaces:
#   paper_repeats_20260930.sh (leg settings), _phase2_ (graph on/off, 0.099 ms reference), _sens_ (material cases),
#   _slab_ (slab time step and element size), _phase4_ (graph-off reference); the inversion audit's 3 position runs.
# Only the meshes change. Each is checked against its digest before anything starts.
#
# ORDER: most of the paper first, at the repeats it states, then the slow slab studies breadth first (every element size
# once before any repeat). Measured solo times from the V4 logs; the V5 leg mesh has 3 % fewer tets.
#    1  leg settings, 11 x 5                         ~3.3 h   cumulative  3.3 h
#    2  leg material sensitivity, 8 x 5              ~0.3 h               3.6 h
#    3  leg graph on/off at 2x35 and 8x35, x 5       ~0.4 h               4.0 h
#    4  leg inversion positions, 2x35, x 3           ~0.1 h               4.1 h
#    5  leg 0.099 ms reference, graph on, x 5        ~4.7 h               8.8 h
#    6  slab time step, 10 settings, repeat 1        ~4.6 h              13.4 h
#    7  slab 1.5 mm at 18.6 us, repeat 1             ~7.4 h              20.8 h   (element-size point AND time-step reference)
#    8  slab 2.5, 2.0, 1.25, 1.0 mm, repeat 1        ~34 h               55 h
#    9  slab time step + 1.5 mm, repeats 2 and 3     ~24 h               79 h
#   10  leg 0.099 ms reference, graph OFF, x 5       ~14 h               93 h
#   11  slab 2.5, 2.0, 1.25, 1.0 mm, repeats 2, 3    ~69 h              162 h
# The first 48 h end inside step 8. The queue keeps going; to stop cleanly, `touch out/paper_v5/STOP` (checked before
# every run; the run in progress finishes).
#
# Pictures: if out/paper_v5/pictures.go exists before a run starts, out/paper_v5/pictures.sh runs once in that gap, so
# picture jobs never share the GPU with a timed run.
#
# Runs without an idle GPU never start: the queue waits with no time limit (2026-10-06 rule). A run killed by the
# watchdog is retried once; a second failure is logged and the queue moves on. Nothing is skipped silently.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
HERE=$(cd "$(dirname "$0")/.." && pwd); cd "$HERE"
O=$(cd out && pwd -P)
BASE=$O/paper_v5; mkdir -p "$BASE"; LOG=$BASE/queue.log; touch "$LOG"
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a "$LOG"; }
PYL=$PY_ISAACLAB

LEG=$O/mesh_leg_v5/tetmesh_leg_v5_s020_f1p5_c30_w30_bone; LEG_SHA=d14afb0ae307
SENS=$O/mesh_leg_v5_sensitivity
declare -A SENS_SHA=([vein_uniform_20kpa]=838817afaa4e [vein_600kpa]=8aa36eaf368d [vein_5p7mpa]=f5334c2e6cbe
                     [nerve_20kpa]=9ee5b4899169 [nerve_80kpa]=31d48d9bd937 [vein_bend_034mm]=51aaeb76c0e2 [vein_bend_050mm]=6db2830f7132)
declare -A SLAB_SHA=([2p5]=5935645c3cd0 [2]=afedb8754e8e [1p5]=3ab9d1302397 [1p25]=2702d8c674f2 [1p0]=c012b923ef7c)
SENS_ORDER="control vein_uniform_20kpa vein_bend_034mm vein_bend_050mm vein_600kpa vein_5p7mpa nerve_20kpa nerve_80kpa"
DEPTHS=3,7,11,15,19,23,25.4

dig(){ sha256sum "$1" | cut -c1-12; }
chk(){ [ -f "$1.npz" ] && [ -f "$1.usda" ] || { say "!!! missing $1 -- regenerate, do not substitute"; exit 1; }
       [ "$(dig "$1.npz")" = "$2" ] || { say "!!! $1.npz digest $(dig "$1.npz"), expected $2 -- refusing"; exit 1; }; }
chk "$LEG" "$LEG_SHA"
for c in "${!SENS_SHA[@]}"; do chk "$SENS/$c" "${SENS_SHA[$c]}"; done
for d in "${!SLAB_SHA[@]}"; do chk "$O/mesh_study_v5/tetmesh_v5_s020_dx$d" "${SLAB_SHA[$d]}"; done
say "START queue. meshes verified: leg $LEG_SHA, 7 sensitivity, 5 slab. phantom HEAD $(git rev-parse --short HEAD);" \
    "live_press_bmode $(dig isaaclab/live_press_bmode.py); press_femoral_punch $(dig isaaclab/press_femoral_punch.py)"

ngpu(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .; }
STARTUP_S=600
# watchdog, as in the 2026-09-30 launchers: startup hang (not on the GPU after STARTUP_S) or a log unchanged for
# the stall limit kills the run, logged as a failure, retried once.
watch() {  # pid logfile tag stall_s
  local PID=$1 LOGF=$2 TAG=$3 STALL=$4 T0=$(date +%s) ONGPU=0 LAST=$(date +%s) SZ=0
  while kill -0 $PID 2>/dev/null; do
    RSS=$(awk '/VmRSS/{print int($2/1e6)}' /proc/$PID/status 2>/dev/null || echo 0)
    if [ "${RSS:-0}" -gt 45 ]; then say "!!! $TAG RSS ${RSS} GB > 45 GB -- KILLED; reported as a failure"; kill -9 $PID; return 8; fi
    [ $ONGPU -eq 0 ] && nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -qw "$PID" && ONGPU=1
    local NOW=$(date +%s) S=$(stat -c %s "$LOGF" 2>/dev/null || echo 0)
    [ "$S" != "$SZ" ] && { SZ=$S; LAST=$NOW; }
    if [ $ONGPU -eq 0 ] && [ $((NOW-T0)) -gt $STARTUP_S ]; then
      say "!!! $TAG WATCHDOG: not on the GPU after $((NOW-T0)) s (startup hang) -- KILLED"; kill -9 $PID; return 9; fi
    if [ $((NOW-LAST)) -gt $STALL ]; then
      say "!!! $TAG WATCHDOG: log unchanged for $((NOW-LAST)) s (mid-run stall) -- KILLED"; kill -9 $PID; return 9; fi
    sleep 10
  done
  return 0
}
gate() {  # before every run: stop file, picture hook, idle GPU
  if [ -f "$BASE/STOP" ]; then say "STOP file present -- queue stopped before the next run"; exit 0; fi
  if [ -f "$BASE/pictures.go" ] && [ -f "$BASE/pictures.sh" ] && [ ! -f "$BASE/pictures.done" ]; then
    local W=0; while [ "$(ngpu)" -gt 0 ]; do sleep 10; W=$((W+1)); done
    say "PICTURES start ($BASE/pictures.sh, sha $(dig "$BASE/pictures.sh"))"
    bash "$BASE/pictures.sh" > "$BASE/pictures.log" 2>&1; local RC=$?
    touch "$BASE/pictures.done"; say "PICTURES done rc=$RC (log $BASE/pictures.log)"
  fi
  local W=0; while [ "$(ngpu)" -gt 0 ]; do sleep 10; W=$((W+1))
    [ $((W % 360)) -eq 0 ] && say "waiting for an idle GPU: $((W/6)) min so far ($(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' '))"; done
  WAITED=$((W*10))
}
launch() {  # tag outdir stall_s json -- command...
  local TAG=$1 OUT=$2 STALL=$3 J=$4; shift 5
  mkdir -p "$OUT"
  if [ -f "$J" ]; then say "SKIP $TAG (json exists)"; return; fi
  gate
  local OTHER=$(ngpu) STATE=IDLE; [ "$OTHER" -gt 0 ] && STATE=SHARED
  local ATT RC=1 T0
  for ATT in 1 2; do
    T0=$(date +%s)
    say "START $TAG (GPU $STATE, other processes $OTHER, waited $WAITED s, attempt $ATT)"
    "$@" > "$OUT/$(basename "$J" .json).log" 2>&1 &
    local PID=$!
    watch $PID "$OUT/$(basename "$J" .json).log" "$TAG" "$STALL"; local WD=$?
    wait $PID 2>/dev/null; RC=$?
    if [ $WD -eq 9 ]; then mv "$OUT/$(basename "$J" .json).log" "$OUT/$(basename "$J" .json).attempt$ATT.log"; rm -f "$J"; RC=137; continue; fi
    break
  done
  say "DONE $TAG rc=$RC GPU $STATE after $(( $(date +%s)-T0 )) s"
}
leg() {  # rep tag substeps iterations graph(on|off) [stall]
  local R=$1 TAG=$2 S=$3 I=$4 G=$5 ST=${6:-2700} OUT=$BASE/leg/r$1; local GF=""; [ "$G" = on ] && GF="--cuda-graph"
  launch "r$R $TAG ($S x $I, graph $G)" "$OUT" "$ST" "$OUT/$TAG.json" -- \
    $PYL -u isaaclab/live_press_bmode.py --headless --npz "$LEG.npz" --usd "$LEG.usda" --substeps "$S" --iterations "$I" $GF \
    --no-worker --out "$OUT/$TAG.json"
}
sens() {  # rep case
  local R=$1 C=$2 M=$SENS/$2 OUT=$BASE/leg_sens/$2/r$1; [ "$C" = control ] && M=$LEG
  launch "r$R sens $C (2 x 35, graph on, no leg visual)" "$OUT" 2700 "$OUT/$C.json" -- \
    $PYL -u isaaclab/live_press_bmode.py --headless --npz "$M.npz" --usd "$M.usda" --substeps 2 --iterations 35 --cuda-graph \
    --no-worker --no-leg-visual --out "$OUT/$C.json"
}
inv() {  # rep
  local R=$1 OUT=$BASE/leg_inversion/r$1
  launch "r$R inversion positions (2 x 35, graph on)" "$OUT" 2700 "$OUT/legV5_s2_i35.json" -- \
    $PYL -u isaaclab/live_press_bmode.py --headless --npz "$LEG.npz" --usd "$LEG.usda" --substeps 2 --iterations 35 --cuda-graph \
    --no-worker --save-positions "$OUT/legV5_s2_i35_pos.npz" --out "$OUT/legV5_s2_i35.json"
}
slab() {  # rep tag mesh-dx substeps iterations force-stop(yes|no)
  local R=$1 TAG=$2 MM=$3 S=$4 I=$5 FS=$6 OUT=$BASE/slab/r$1 M=$O/mesh_study_v5/tetmesh_v5_s020_dx$3
  local FSF=""; [ "$FS" = yes ] && FSF="--stop-force-n 30"
  launch "r$R slab $TAG (dx $MM, $S x $I, graph off, force stop $FS)" "$OUT" 14400 "$OUT/$TAG.json" -- \
    $PYL -u isaaclab/press_femoral_punch.py --headless --depths "$DEPTHS" --substeps "$S" --iterations "$I" \
    --allow-inversion $FSF --no-cuda-graph --ramp-mm-per-frame 0.4 --log-every 25 \
    --npz "$M.npz" --usd "$M.usda" --out "$OUT/$TAG.json" --save-positions "$OUT/${TAG}_pos.npz"
}
slab_timestep(){ local R=$1
  for I in 10 20 35 70; do slab $R ts_s8_i$I 1p5 8 $I no; done
  for S in 2 3 8 17 67 168; do slab $R ts_s${S}_i140 1p5 $S 140 no; done; }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# ORDER 2 (2026-10-22). Everything above is paper_v5_rerun_20261021.sh unchanged: same meshes, digests, flags, watchdog.
# Why: the budget became 96 h (13:15 on 2026-10-25), and V5 slab presses settle more slowly than V4's (432 frames against
# 310 at 1.5 mm), so the original order would spend the budget on slab repeats before the leg graph-off reference. Within
# 96 h this order finishes: every element size once, the leg graph-off reference x5 (the last pending leg cell), and the
# slab time-step repeats 2 and 3. Then the slab 1.5 mm and element-size repeats, as before.
# Hand-over: the order-1 queue was told to stop (STOP file) and finishes its current run first. This waits for it to
# exit, removes the STOP file, makes sure the snapshot loop runs, then continues. Runs already done are skipped (json).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
while pgrep -f "run_scripts/paper_v5_rerun_20261021.sh" > /dev/null; do sleep 60; done
rm -f "$BASE/STOP"
say "ORDER 2 takes over (run_scripts/paper_v5_rerun_20261021_order2.sh, sha $(dig run_scripts/paper_v5_rerun_20261021_order2.sh))"
pgrep -f paper_v5_snapshot_20261021.sh > /dev/null || { setsid -f bash "$HERE/run_scripts/paper_v5_snapshot_20261021.sh" > /dev/null 2>&1 < /dev/null; say "snapshot loop restarted"; }
# 8 the other element sizes, repeat 1 (2.5 mm finished under order 1)
for MM in 2p5 2 1p25 1p0; do slab 1 dx${MM}_s896_i140 $MM 896 140 yes; done; say "SLAB ELEMENT SIZE REPEAT 1 DONE"
# 10 reference, graph off (stall 5400 s: on 2026-09-09 the log went 2484 s between two depths)
for R in 1 2 3 4 5; do leg $R legV5_s168_i140_ref_graphoff 168 140 off 5400; done; say "LEG GRAPH-OFF REFERENCE DONE"
# 9a slab time step, repeats 2-3
for R in 2 3; do slab_timestep $R; done; say "SLAB TIME STEP x3 DONE"
# 9b slab 1.5 mm at 18.6 us, repeats 2-3
for R in 2 3; do slab $R dx1p5_s896_i140 1p5 896 140 yes; done; say "SLAB 1.5 mm x3 DONE"
# 11 element sizes, repeats 2-3
for R in 2 3; do for MM in 2p5 2 1p25 1p0; do slab $R dx${MM}_s896_i140 $MM 896 140 yes; done; done
say "ALL DONE"
