#!/usr/bin/env bash
# Build the derived inputs the paper's runs read, from the distributed V5 anatomy and V5 leg mesh.
#
#   bash run_scripts/build_inputs.sh render_rest     # what the example video needs (README.md, quick start)
#   bash run_scripts/build_inputs.sh all            # every derived input
#
# Targets, in dependency order:
#   slab          the five slab meshes, 2.5 to 1.0 mm (Fig. 5, Tables S1, S2)         physics venv + isaaclab venv (USD)
#   render_rest   the B-mode rest render set (every video and B-mode)                 needs the 1.5 mm slab
#   cut_texture   the cut-face texture for the cut-away videos                       physics venv
#   sensitivity   the seven vessel-wall and nerve stiffness meshes (Table S6)         anatomy venv
#   cast          the benchtop cast meshes: main and gel +/-1 SD (Fig. 3, Table S5)   physics venv + isaaclab venv (USD)
#   leg           rebuild the V5 leg mesh and compare it with the distributed one     physics venv
#
# ⚠️ A rebuilt mesh never has the sha256 of the paper's file: numpy writes the build time into every .npz. The run
# scripts check file digests, so for the timed runs use the distributed meshes (README.md, section 6a). Each mesh target
# here compares its result with the distributed mesh array by array when that mesh is present
# (anatomy_meshing/compare_mesh.py), and fails if any array the press reads differs.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
cd "$I4H_PHANTOM"
REF=${REF_MESHES:-out}          # where the distributed meshes are, to compare against
say(){ echo "[$(date +%F' '%H:%M:%S)] $*"; }
sha(){ sha256sum "$1" | cut -c1-12; }
same(){  # rebuilt reference
  if [ -f "$2" ] && [ "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" != "$(cd "$(dirname "$2")" && pwd)/$(basename "$2")" ]; then
    $PY_ANATOMY anatomy_meshing/compare_mesh.py "$1" "$2" | tail -1
  else say "  (no distributed copy of $(basename "$2") to compare with)"; fi; }

[ "$(sha out/mesh_leg_v5/tetmesh_leg_v5_s020_f1p5_c30_w30_bone.npz)" = d14afb0ae307 ] \
  || { say "!!! out/mesh_leg_v5 is not the paper's V5 leg mesh (d14afb0ae307)"; exit 1; }

slab() {
  mkdir -p out/mesh_study_v5
  for d in 2p5:0.0025 2:0.002 1p5:0.0015 1p25:0.00125 1p0:0.001; do
    local tag=${d%%:*} dx=${d##*:} f=out/mesh_study_v5/tetmesh_v5_s020_dx${d%%:*}
    if [ -f $f.npz ] && [ -f $f.usda ]; then say "slab $tag: present"; continue; fi
    say "slab $tag: building ($dx m)"
    $PY_PHYSICS -u anatomy_meshing/export_tetmesh.py --anatomy-dir femoral_v5 --station-s 20 --dx $dx --substeps 896 \
      --perivascular --out $f.npz > out/mesh_study_v5/build_dx$tag.log 2>&1
    $PY_ISAACLAB isaaclab/author_tetmesh_usd.py --npz $f.npz > out/mesh_study_v5/usd_dx$tag.log 2>&1
  done
}

render_rest() {
  local R=out/live_render_rest_v5 S=out/mesh_study_v5/tetmesh_v5_s020_dx1p5.npz
  if [ -f $R/travel_0.0mm/anatomy_config.json ] && [ -f $R/travel_0.0mm/Gel.obj ]; then say "render_rest: present"; return; fi
  if [ ! -f $S ]; then
    say "render_rest: building the 1.5 mm slab first (its nodes set the imaging slab)"
    mkdir -p out/mesh_study_v5
    $PY_PHYSICS -u anatomy_meshing/export_tetmesh.py --anatomy-dir femoral_v5 --station-s 20 --dx 0.0015 --substeps 896 \
      --perivascular --out $S > out/mesh_study_v5/build_dx1p5.log 2>&1
  fi
  mkdir -p $R
  $PY_ANATOMY anatomy_meshing/zero_state.py $S $R/zero_state_slab_dx1p5.npz
  $PY_ANATOMY anatomy_meshing/deform_meshes.py --states $R/zero_state_slab_dx1p5.npz --obj-dir out/femoral_v5 \
    --out-root $R --section out/femoral_v5/section_s020.json --blend-sigma-mm 12.0 > $R/deform.log 2>&1
  $PY_ANATOMY anatomy_meshing/gel_body.py --obj-dir $R/travel_0.0mm > $R/gel.log 2>&1
  say "render_rest: $(ls $R/travel_0.0mm/*.obj | wc -l) bodies in $R/travel_0.0mm (the paper's: 12, worker scene sha f4dcd823e93b)"
}

cut_texture() {
  local T=out/cut_texture_meshV5
  if [ -f $T/cut_texture.json ]; then say "cut_texture: present"; return; fi
  $PY_PHYSICS analysis/cut_section_texture.py out/mesh_leg_v5/tetmesh_leg_v5_s020_f1p5_c30_w30_bone.npz $T 2048
}

sensitivity() {
  local D=out/mesh_leg_v5_sensitivity
  if [ -f $D/manifest.json ]; then say "sensitivity: present"; return; fi
  $PY_ANATOMY anatomy_meshing/make_sensitivity_meshes.py --src out/mesh_leg_v5/tetmesh_leg_v5_s020_f1p5_c30_w30_bone \
    --src-sha d14afb0ae307 --dst $D
}

cast() {
  local M=out/mesh_leg_cast BASE=tetmesh_leg_cast_s020_f1p5_c30_w30_bend1p5_femurfree200
  mkdir -p $M
  [ -f out/femoral_cast_bend1p5/section_s020.json ] || $PY_ANATOMY anatomy_meshing/cast_section.py --bending-equivalent-mm 1.5
  for SD in none -1 1; do
    local NAME=$BASE EXTRA=""
    [ $SD = none ] || { NAME=${BASE}_gel${SD}sd; EXTRA="--gel-sd $SD"; }
    if [ -f $M/$NAME.npz ] && [ -f $M/$NAME.usda ]; then say "cast $NAME: present"; continue; fi
    say "cast $NAME: building"
    $PY_PHYSICS -u anatomy_meshing/graded_leg_mesh.py --anatomy-dir femoral_cast_bend1p5 --materials cast --femur-free-mpa 200 $EXTRA \
      --station-s 20 --fine-dx-mm 1.5 --coarse-dx-mm 30 --grade-mm 25 --arc-deg 45 --s-window 0 30 --fine-depth-mm auto \
      --sheath-kpa 20 --table-cap-mm 6 --bone-dx-mm 5 --bone-margin-mm 4 --ramus-dx-mm 2.5 --nsub 11 --leg-s-range 0 81 \
      --out $M/$NAME.npz > $M/build_$NAME.log 2>&1
    $PY_ISAACLAB isaaclab/author_tetmesh_usd.py --npz $M/$NAME.npz > $M/usd_$NAME.log 2>&1
  done
}

leg() {
  local N=out/rebuilt/tetmesh_leg_v5_s020_f1p5_c30_w30_bone.npz
  mkdir -p out/rebuilt
  $PY_PHYSICS -u anatomy_meshing/graded_leg_mesh.py --anatomy-dir femoral_v5 --leg-s-range 0 81 \
    $(cat out/mesh_leg_v5/build_args.txt) --out $N > out/rebuilt/build_leg.log 2>&1
  same $N out/mesh_leg_v5/tetmesh_leg_v5_s020_f1p5_c30_w30_bone.npz
}

[ $# -gt 0 ] || { sed -n 2,12p "$0"; exit 1; }
for t in "$@"; do
  case $t in
    all) slab; render_rest; cut_texture; sensitivity; cast ;;
    slab|render_rest|cut_texture|sensitivity|cast|leg) $t ;;
    *) echo "unknown target $t"; exit 1 ;;
  esac
done
say "done: $*"
