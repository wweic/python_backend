#!/usr/bin/env python3
"""gRPC load for the two-input 'preproc' model.
  features: FP32[768] (constant)   cat_ids: BYTES[1000] drawn from cat_0..cat_59999
Reports req/s + latency percentiles. Saves the first response for correctness diff.
Usage: loadgen_preproc.py <url> <conc> <total> [tag]"""
import sys, os, time, threading, random
import numpy as np
import tritonclient.grpc as grpcclient

URL   = sys.argv[1] if len(sys.argv) > 1 else "localhost:8011"
CONC  = int(sys.argv[2]) if len(sys.argv) > 2 else 16
TOTAL = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
TAG   = sys.argv[4] if len(sys.argv) > 4 else "run"
MODEL = os.environ.get("MODEL", "preproc_cpp")
FEAT     = int(os.environ.get("FEAT", "262144"))     # numeric elems (262144 = 1MB fp32)
NCAT     = int(os.environ.get("NCAT", "4000"))       # strings per request
KEYSPACE = int(os.environ.get("KEYSPACE", "4800"))   # ~83% hit rate vs VOCAB=4000
NPOOL    = 64
random.seed(1234)

FEATURES = np.full((FEAT,), 0.5, dtype=np.float32)
# Pre-build and pre-serialize NPOOL distinct InferInput payload pairs ONCE, so the
# client is not re-serializing the (large) BYTES tensor on every request -- that
# would make the benchmark client-bound and hide server-side differences.
OUT = grpcclient.InferRequestedOutput("out")
INPUTS = []
for _ in range(NPOOL):
    arr = np.array([("cat_%d" % random.randint(0, KEYSPACE - 1)).encode() for _ in range(NCAT)],
                   dtype=object)
    fi = grpcclient.InferInput("features", [FEAT], "FP32"); fi.set_data_from_numpy(FEATURES)
    ci = grpcclient.InferInput("cat_ids", [NCAT], "BYTES"); ci.set_data_from_numpy(arr)
    INPUTS.append([fi, ci])

_tls = threading.local()
def client():
    c = getattr(_tls, "c", None)
    if c is None:
        c = _tls.c = grpcclient.InferenceServerClient(url=URL)
    return c

_first = {}
def one(i):
    inp = INPUTS[i % NPOOL]
    t0 = time.perf_counter()
    r = client().infer(MODEL, inputs=inp, outputs=[OUT])
    dt = time.perf_counter() - t0
    if i < NPOOL and i not in _first:
        _first[i] = r.as_numpy("out").copy()
    return dt

def run(n, label, off):
    from concurrent.futures import ThreadPoolExecutor
    lat = []; t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        for dt in ex.map(one, range(off, off + n)):
            lat.append(dt)
    wall = time.perf_counter() - t0; lat.sort()
    p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] * 1e3
    print("[%s/%s] n=%d conc=%d req/s=%.1f lat_ms p50=%.2f p90=%.2f p99=%.2f wall=%.1fs"
          % (TAG, label, n, CONC, n / wall, p(0.5), p(0.9), p(0.99), wall))
    return n / wall

c = grpcclient.InferenceServerClient(url=URL)
print("model ready:", c.is_model_ready(MODEL))
run(CONC * 4, "warmup", 0)
best = max(run(TOTAL, "measured", CONC * 4) for _ in range(2))
# persist first-response fingerprint for correctness comparison across builds
import json
fp = {str(k): [float(v.sum()), float(v[0]), int(v.shape[0])] for k, v in sorted(_first.items())}
open("/work/fingerprint_%s.json" % TAG, "w").write(json.dumps(fp))
print("[%s] BEST req/s=%.1f  fingerprint saved (%d payloads)" % (TAG, best, len(fp)))
