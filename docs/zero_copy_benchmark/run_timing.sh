#!/usr/bin/env bash
# Per-request timing: drive load and extract the stub's timing lines.
#   [BASELINE|ARROW|CPP]-TIMING  -> model-side numeric/string section cost (always available)
#   [LRFSM-TIMING]               -> LoadRequestsFromSharedMemory cost, ONLY if the stub was
#                                   built with instrumentation/lrfsm_timer.patch applied.
#
# Usage: run_timing.sh <image> <model>
#   env: FEAT NCAT KEYSPACE VOCAB
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="$1"; MODEL="$2"
NAME="zctime_${MODEL}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run --rm -d --name "$NAME" --gpus all --shm-size=2g -p 8011:8001 \
  -e VOCAB="${VOCAB:-50000}" \
  -v "$SCRIPT_DIR/models":/models:ro \
  "$IMAGE" tritonserver --model-repository=/models --grpc-port=8001 --http-port=8000 >/dev/null
for i in $(seq 1 60); do
  docker exec "$NAME" bash -c 'curl -s -o /dev/null -w "%{http_code}" localhost:8000/v2/health/ready' 2>/dev/null | grep -q 200 && break
  docker ps --format '{{.Names}}' | grep -q "^$NAME$" || { echo "DIED"; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
  sleep 1
done
docker run --rm --network host -v "$SCRIPT_DIR":/work \
  -e MODEL="$MODEL" -e FEAT="${FEAT:-64}" -e NCAT="${NCAT:-20000}" -e KEYSPACE="${KEYSPACE:-60000}" \
  "$IMAGE" python3 /work/loadgen.py localhost:8011 32 8000 "$MODEL" >/dev/null 2>&1
echo "===== [$MODEL] FEAT=${FEAT:-64} NCAT=${NCAT:-20000} VOCAB=${VOCAB:-50000} ====="
docker logs "$NAME" 2>&1 | grep -E 'LRFSM-TIMING|BASELINE-TIMING|ARROW-TIMING|CPP-TIMING' | tail -4
docker rm -f "$NAME" >/dev/null 2>&1 || true
