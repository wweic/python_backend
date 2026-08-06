# Zero-copy input optimization — benchmark suite

Reproduces the before/after numbers in
[`../zero_copy_input_optimization.md`](../zero_copy_input_optimization.md) for the
zero-copy input paths added in this branch (deferred/zero-copy `as_numpy`,
`as_string_buffers`, and `StringLookupTable`).

## What's here

| Path | Purpose |
|------|---------|
| `models/preproc_dict`  | Test preprocessor, **baseline**: `as_numpy()` + `[b.decode()]` + Python `dict.get` |
| `models/preproc_arrow` | Same, using `as_string_buffers()` + `pyarrow.compute.index_in` (rehashes value_set every call) |
| `models/preproc_cpp`   | Same, using `pb_utils.StringLookupTable` (persistent C++ hash, reads shm directly) |
| `loadgen.py`           | gRPC load generator (2 inputs: `features` FP32, `cat_ids` BYTES); pre-serializes payloads so the client is not the bottleneck |
| `run_bench.sh`         | End-to-end throughput (req/s, latency) for one model |
| `run_timing.sh`        | Per-request section timing from the stub log |
| `micro.py`             | Server-free microbenchmark: numeric copy-vs-view and lookup regime map |
| `build_backend.sh`     | Build the stub + backend `.so` standalone and extract the artifacts |
| `instrumentation/lrfsm_timer.patch` | Optional patch to time `LoadRequestsFromSharedMemory` directly |

All three models share the same inputs/outputs and produce **bit-identical**
outputs; they differ only in how the categorical lookup is done.

## The test preprocessor

Each request carries a dense numeric tensor (`features`, FP32) and a categorical
string tensor (`cat_ids`, BYTES). The model maps every string to an int id via a
vocab of `VOCAB` entries (env, default 50000) and folds a reduction of `features`
into the output. Input sizes are controlled by the loadgen via env:
`FEAT` (numeric elements), `NCAT` (strings/request), `KEYSPACE` (id draw range),
`VOCAB` (server-side, passed to the model).

## Prerequisites

- Docker with `--gpus all`.
- A Triton build-base image (`tritonserver_buildbase:latest`) and a runtime
  `tritonserver` image built from the **same** `core`/`common`/`backend` tag
  (default `r26.06`). `build_backend.sh` compiles against that tag so the
  swapped artifacts are ABI-compatible.
- `pyarrow` and `tritonclient[grpc]` in the runtime image (the `arrow` model and
  the loadgen need them):
  `pip install pyarrow tritonclient[grpc]`.

## Build the two backends and swap them into an image

```bash
# baseline = this repo at the base commit; candidate = this branch (with the opt)
./build_backend.sh /path/to/python_backend_baseline out_baseline
./build_backend.sh /path/to/python_backend_candidate out_candidate

# swap artifacts into a runtime image
cat > Dockerfile.cand <<EOF
FROM tritonserver:latest
COPY out_candidate/libtriton_python.so /opt/tritonserver/backends/python/
COPY out_candidate/triton_python_backend_stub /opt/tritonserver/backends/python/
EOF
docker build -f Dockerfile.cand -t tritonserver:candidate .
```

## Run

```bash
# throughput, string-heavy regime (small numeric, 20k strings, vocab 50k)
FEAT=256 NCAT=20000 VOCAB=50000 ./run_bench.sh tritonserver:candidate preproc_cpp
FEAT=256 NCAT=20000 VOCAB=50000 ./run_bench.sh tritonserver:baseline  preproc_dict

# per-request section timing
FEAT=64 NCAT=20000 VOCAB=50000 ./run_timing.sh tritonserver:candidate preproc_cpp

# server-free regime map
docker run --rm -v "$PWD":/w --entrypoint python3 tritonserver:candidate /w/micro.py
```

## Reproducing the LoadRequestsFromSharedMemory numbers

`LoadRequestsFromSharedMemory` is not timed in the shipped source. To measure it,
apply the optional patch and rebuild before running `run_timing.sh`:

```bash
git apply docs/zero_copy_benchmark/instrumentation/lrfsm_timer.patch
./docs/zero_copy_benchmark/build_backend.sh . out_candidate_timed
# ...swap into an image, then run_timing.sh; look for [LRFSM-TIMING] lines
git checkout src/pb_stub.cc   # remove the instrumentation again
```

## Headline results (see the report for the full table)

Single KIND_CPU instance, 32 concurrent clients, string-heavy regime:

- `LoadRequestsFromSharedMemory`: ~240 µs → ~4.75 µs per request.
- Categorical lookup: 6.2× faster than `decode+dict`, and robust to vocab size
  (unlike `pyarrow index_in`, which rehashes its value_set every call).
- End-to-end throughput: **319 → 1670 req/s (5.2×)**, outputs bit-identical.
