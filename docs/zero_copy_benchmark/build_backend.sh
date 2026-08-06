#!/usr/bin/env bash
# Build python_backend (triton_python_backend_stub + libtriton_python.so)
# standalone inside the Triton buildbase image, then extract the two artifacts.
# Swap them into any ABI-matching tritonserver image's
# /opt/tritonserver/backends/python/ to try the optimization without a full
# server rebuild.
#
# Usage: build_backend.sh <python_backend_src_dir> <output_dir>
#   env: REPO_TAG (default r26.06)   BUILDBASE (default tritonserver_buildbase:latest)
set -euo pipefail
SRC="$(cd "$1" && pwd)"
mkdir -p "$2"; OUT="$(cd "$2" && pwd)"
REPO_TAG="${REPO_TAG:-r26.06}"
BUILDBASE="${BUILDBASE:-tritonserver_buildbase:latest}"

docker run --rm --network host \
  -e REPO_TAG="$REPO_TAG" \
  -v "$SRC":/pbsrc:ro -v "$OUT":/out \
  --entrypoint bash "$BUILDBASE" -c '
set -euo pipefail
cp -r /pbsrc /build_src && cd /build_src && rm -rf build
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTRITON_ENABLE_GPU=ON -DTRITON_ENABLE_NVTX=OFF \
  -DTRITON_BACKEND_REPO_TAG="$REPO_TAG" \
  -DTRITON_COMMON_REPO_TAG="$REPO_TAG" \
  -DTRITON_CORE_REPO_TAG="$REPO_TAG" \
  >/out/cmake_configure.log 2>&1
cmake --build build -j"$(nproc)" \
  --target triton-python-backend triton-python-backend-stub >/out/cmake_build.log 2>&1
cp -v build/libtriton_python.so build/triton_python_backend_stub /out/
'
echo "artifacts written to $OUT"
