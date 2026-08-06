# Zero-copy Python-backend: before/after measurement

Direct measurement of `triton::backend::python::Stub::LoadRequestsFromSharedMemory`
(pb_stub.cc:653) — the hot function from the flame graph — with an in-function
timer, comparing the **baseline** (r26.06 stock) stub against a **candidate** stub
carrying the DLPack (zero-copy numeric) + Arrow (zero-copy string) changes.

Both stubs are built identically (RelWithDebInfo, r26.06 core/backend/common, same
buildbase, same in-function timer). The ONLY difference is `pb_tensor.cc`/`.h` +
the `as_string_buffers` binding. Runtime = `tritonserver:latest` (2.72.0dev),
Python 3.12, single KIND_CPU instance, 32 concurrent gRPC clients.

## 1. LoadRequestsFromSharedMemory cost (the ask)

| Regime                         | Baseline    | Candidate  | Speedup |
|--------------------------------|-------------|------------|---------|
| numeric-heavy (1MB fp32, 100 strings) | 48.3 µs/req | 4.2 µs/req  | **11.5×** |
| string-heavy  (64 fp32, 20 000 strings) | 240.3 µs/req | 4.75 µs/req | **~50×** |

The candidate's LRFSM cost collapses to a flat ~4–5 µs regardless of tensor size:
the eager shm→NumPy copy (numeric) and the N×PyBytes deserialization (BYTES) are
no longer done inside this function — they are deferred to first use, where the
numeric path never copies (zero-copy view over shm) and the string path does one
bulk compaction instead of N object allocations.

## 2. Full stub-side cost (LRFSM + model execute), per request

The LRFSM number alone excludes deferred work, so here is the honest total
(LRFSM + the numeric + string sections timed inside `execute`):

| Regime         | Baseline total | Candidate total | Net |
|----------------|----------------|-----------------|-----|
| 1MB numeric, 100 strings | 48.3 + 40 + 31 = **119 µs** | 4.2 + 47 + 115 = **166 µs** | **regresses** |
| 64 fp32, 20 000 strings  | 240 + 4 + 2206 = **2450 µs** | 4.75 + 8 + 597 = **610 µs** | **4.0× faster** |

- **Numeric (DLPack / zero-copy view):** copy eliminated, not just moved. The
  reduction reads shm directly. Robust win, scales with tensor size, no downside.
- **String (Arrow):** the N×PyBytes deserialization is removed from LRFSM (240→4.75 µs),
  and for large string batches the vectorized `pc.index_in` lookup is ~3.7× faster
  than decode+dict (2.21 ms → 0.60 ms), giving a 4× total stub-side win.
  **Caveat:** for *small* string batches against a non-trivial vocab, pyarrow rebuilds
  the value_set hash table on every call, so the Arrow lookup is *slower* than a
  CPython dict (the R1 regression, and the earlier 50k-vocab end-to-end regression:
  3313 → 795 req/s). Apply Arrow only for large batches / small-to-moderate vocab,
  or replace `index_in` with a persistent hash table.

## 3. Correctness

Output fingerprints (sum, first element, length) over 64 distinct payloads:
**64/64 identical**, 0 mismatches. The optimization is numerically transparent.

## 4. Why end-to-end throughput barely moves

LRFSM is ~7% of the full gRPC→IPC→execute pipeline (matches the original flame
graph). Removing its copy is a targeted stub-side reduction, not a pipeline-level
game-changer — unless the model is stub-bound on string preprocessing (R2), where
the dominant cost drops 4×.

## 5. The string lookup done in C++ (the right fix for the pc.index_in problem)

`pc.index_in` rebuilds its value_set hash table on **every** call — there is no
way in the pyarrow *Python API* to persist that hash across `execute()` calls
(the machinery exists in Arrow C++ — `SetLookupState`, Acero's SwissTable — but
is not surfaced for reuse). The correct fix is a persistent C++ hash table built
once at init. Added `pb_utils.StringLookupTable(keys, values, default)`
(pb_tensor.cc/.h + binding): the hash is built once, and `lookup(tensor)` reads
the BYTES tensor's length-prefixed shared-memory blob directly in C++
(std::string_view probes, no PyBytes, no compaction copy, no per-call rehash),
returning an int64 NumPy array.

String-processing cost inside `execute`, 20 000 strings/request:

| vocab  | baseline (decode+dict) | Arrow pc.index_in | **C++ StringLookupTable** |
|--------|------------------------|-------------------|---------------------------|
| 4 000  | 2.18 ms                | 0.60 ms           | **0.351 ms** (6.2× vs base) |
| 50 000 | ~2.18 ms               | 1.729 ms          | **0.511 ms** (4.3× vs base) |

- The C++ table is **robust to vocab size** (0.35 → 0.51 ms; O(#strings), not
  O(vocab)); Arrow degrades with vocab (0.60 → 1.73 ms) due to the per-call
  rehash. At vocab 50k the C++ path is **3.4× faster than Arrow**.
- Correctness: C++ lookup outputs are bit-identical to the baseline dict (64/64).
- `LoadRequestsFromSharedMemory` stays ~4 µs (copy/deserialize removed from it);
  the lookup happens in `execute` via the persistent table.

**Recommendation:** use `StringLookupTable` (C++) for the categorical lookup, not
`pc.index_in`. Combine with the numeric zero-copy view (DLPack path). This gives
a robust ~6–7× reduction in stub-side per-request cost for string-heavy
preprocessors, with no vocab-size cliff.

## 6. End-to-end throughput (best optimization: C++ lookup + numeric zero-copy)

String-heavy regime (256 fp32 + 20 000 strings/req, vocab 50 000), 1 KIND_CPU
instance, 32 concurrent gRPC clients, client payloads pre-serialized so the
benchmark is server-bound (not client-bound):

| Build | req/s | p50 latency |
|-------|-------|-------------|
| Baseline (dict)                              | 319   | 100 ms |
| **Optimized (C++ StringLookupTable + numeric zero-copy)** | **1670** | **19 ms** |

**5.2× throughput, 5.2× lower latency**, outputs bit-identical (64/64).

Note on measurement: with the loadgen re-serializing the 20 000-string BYTES
tensor on *every* request, both builds sat at ~252 req/s -- the client was the
bottleneck and hid the server win. Pre-serializing the InferInput payloads once
(as above) exposes the true server-side improvement. Lesson: the copy
optimization only shows up when the pipeline is actually server/stub-bound; when
it is bottlenecked on gRPC/serialization of the same data, the win is invisible
(and the benefit is CPU headroom for more instances instead).

## Microbenchmark regime map (isolated, no server)

Numeric copy vs zero-copy view (copy+consume):
  3KB 21% · 64KB 23% · 256KB 24% · 1MB 35% · 4MB 34% saved.
Arrow index_in vs decode+dict:
  vocab 1000 → 1.9× win · 4000 → 1.6× win · 20000 → 2.1× loss · 50000 → 12× loss · 200000 → 20× loss.
