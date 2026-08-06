#!/usr/bin/env bash
# End-to-end throughput: start a tritonserver from IMAGE serving this suite's
# models, drive the gRPC loadgen against one MODEL, report req/s + latencies.
#
# Usage: run_bench.sh <image> <model> [conc] [total]
#   env: FEAT NCAT KEYSPACE VOCAB  (input sizing; see loadgen.py / model.py)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="$1"; MODEL="$2"; CONC="${3:-32}"; TOTAL="${4:-6000}"
NAME="zcbench_${MODEL}"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run --rm -d --name "$NAME" --gpus all --shm-size=2g -p 8011:8001 \
  -e VOCAB="${VOCAB:-50000}" \
  -v "$SCRIPT_DIR/models":/models:ro \
  "$IMAGE" tritonserver --model-repository=/models \
  --grpc-port=8001 --http-port=8000 >/dev/null

echo "[$MODEL] waiting for server ready..."
for i in $(seq 1 60); do
  docker exec "$NAME" bash -c 'curl -s -o /dev/null -w "%{http_code}" localhost:8000/v2/health/ready' 2>/dev/null | grep -q 200 && { echo "[$MODEL] ready"; break; }
  docker ps --format '{{.Names}}' | grep -q "^$NAME$" || { echo "SERVER DIED:"; docker logs "$NAME" 2>&1 | tail -30; exit 1; }
  sleep 1
done

docker run --rm --network host -v "$SCRIPT_DIR":/work \
  -e MODEL="$MODEL" -e FEAT="${FEAT:-256}" -e NCAT="${NCAT:-20000}" \
  -e KEYSPACE="${KEYSPACE:-60000}" \
  "$IMAGE" python3 /work/loadgen.py localhost:8011 "$CONC" "$TOTAL" "$MODEL"

docker rm -f "$NAME" >/dev/null 2>&1 || true
