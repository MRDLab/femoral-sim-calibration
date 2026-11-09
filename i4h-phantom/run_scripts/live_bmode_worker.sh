#!/usr/bin/env bash
# The live B-mode worker: raysim ($RAYSIM_IMAGE, docker/Dockerfile.raysim) fed FEM node positions over TCP. 2026-09-09.
# Rest render set = the 2026-09-08 chain's tooling at zero displacement (out/live_render_rest/travel_0.0mm).
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
HERE=$(cd "$(dirname "$0")/.." && pwd); cd "$HERE"
REST=${REST:-$HERE/out/live_render_rest/travel_0.0mm}
PORT=${PORT:-5557}
RIGID=${RIGID:-Femur,PubicRamus}     # bone not re-uploaded per frame; the worker logs the displacement it discards
[ -f "$REST/anatomy_config.json" ] && [ -f "$REST/Gel.obj" ] || { echo "!!! rest render set missing at $REST -- regenerate (deform_meshes.py at zero displacement + gel_body.py); do not substitute"; exit 1; }
mkdir -p out/live_bmode
docker rm -f bmode_worker >/dev/null 2>&1
exec docker run --rm --name bmode_worker --gpus all --network host -e PYTHONUNBUFFERED=1 \
  -e PROBE_THETA_DEG=${PROBE_THETA_DEG:-0} -e STATION_S_MM=${STATION_S_MM:-20} \
  -v "$HERE":/w:ro -v "$REST":/rest:ro -v "$HERE/out/live_bmode":/out \
  --entrypoint bash "$RAYSIM_IMAGE" -lc "python /w/raysim_ultrasound/bmode_worker.py --port $PORT --rigid '$RIGID'" \
  2>&1 | grep --line-buffered -vE '^\[20|Assimp' | tee out/live_bmode/worker.log
