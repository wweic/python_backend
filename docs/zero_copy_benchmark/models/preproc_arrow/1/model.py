# OPTIMIZED python-backend model: zero-copy inputs.
#  - numeric input : tensor.as_numpy() -> ZERO-COPY view over shm (patched backend);
#                    the reduction reads shm directly, no C++ copy.
#  - string input  : tensor.as_string_buffers() -> (offsets, data) zero-copy over the
#                    compacted shm blob -> pa.StringArray (zero-copy) -> pc.index_in
#                    (vectorized) instead of N PyBytes + python decode + dict.get.
import os, time, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import triton_python_backend_utils as pb_utils

VOCAB = int(os.environ.get("VOCAB", "4000"))
OUTDIM = 256


def build_vocab():
    keys = ["cat_%d" % i for i in range(VOCAB)]
    vals = [(i * 2654435761) % 1000 for i in range(VOCAB)]
    return keys, vals


class TritonPythonModel:
    def initialize(self, args):
        keys, vals = build_vocab()
        # vocab compiled ONCE into an Arrow value_set + parallel values array.
        self.vocab_keys = pa.array(keys, type=pa.string())
        self.vocab_vals = pa.array(vals, type=pa.int64())
        self.t_num = 0.0; self.t_str = 0.0; self.n = 0

    def execute(self, requests):
        responses = []
        for request in requests:
            t0 = time.perf_counter()
            feats = pb_utils.get_input_tensor_by_name(request, "features").as_numpy()  # zero-copy view
            fs = np.float32(feats.sum())
            t1 = time.perf_counter()
            ct = pb_utils.get_input_tensor_by_name(request, "cat_ids")
            offs, data = ct.as_string_buffers()        # int32[M+1], uint8[total]  (zero-copy)
            arr = pa.StringArray.from_buffers(
                len(offs) - 1, pa.py_buffer(offs), pa.py_buffer(data))
            idx = pc.index_in(arr, value_set=self.vocab_keys)   # vectorized lookup
            ids = pc.fill_null(pc.take(self.vocab_vals, idx), -99).to_numpy(zero_copy_only=False)
            t2 = time.perf_counter()
            self.t_num += t1 - t0; self.t_str += t2 - t1; self.n += 1
            if self.n % 1000 == 0:
                sys.stderr.write("[ARROW-TIMING] n=%d numeric=%.3fms/req string=%.3fms/req\n"
                                 % (self.n, self.t_num / self.n * 1e3, self.t_str / self.n * 1e3))
                sys.stderr.flush()
            out = np.full(OUTDIM, fs + np.float32(int(ids.sum()) % 1000), dtype=np.float32)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[pb_utils.Tensor("out", out)]))
        return responses
