import time, numpy as np, pyarrow as pa, pyarrow.compute as pc, random
def t(fn,n=300):
    fn(); t0=time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter()-t0)/n*1e3

print("== NUMERIC: shm->numpy COPY vs zero-copy VIEW, then sum() ==")
for nbytes in [3*1024, 64*1024, 256*1024, 1024*1024, 4*1024*1024]:
    n=nbytes//4; a=np.full(n,0.5,np.float32)
    c=t(lambda:a.copy().sum(), 500); v=t(lambda:a.sum(), 500)
    print("  size=%7dKB  copy+sum=%.4f ms  view+sum=%.4f ms  saved=%.4f ms (%.0f%%)"
          %(nbytes//1024,c,v,c-v,100*(c-v)/c))

print("== STRING lookup: baseline(decode+dict) vs Arrow(index_in) vs no-PyBytes dict ==")
random.seed(1)
for VOCAB,NCAT in [(1000,1000),(4000,2000),(20000,2000),(50000,1000),(200000,2000)]:
    keys=["cat_%d"%i for i in range(VOCAB)]; vals=[(i*2654435761)%1000 for i in range(VOCAB)]
    vocab=dict(zip(keys,vals)); vk=pa.array(keys,pa.string()); vv=pa.array(vals,pa.int64())
    sample=[("cat_%d"%random.randint(0,int(VOCAB*1.2))).encode() for _ in range(NCAT)]
    cats=np.array(sample,dtype=object)
    blob=b"".join(sample); offs=np.zeros(NCAT+1,np.int32)
    for i,s in enumerate(sample): offs[i+1]=offs[i]+len(s)
    dat=np.frombuffer(blob,np.uint8)
    def base():
        ks=[b.decode() for b in cats.tolist()]; g=vocab.get
        return np.fromiter((g(k,-99) for k in ks),np.int64,len(ks))
    def arrow():
        arr=pa.StringArray.from_buffers(NCAT,pa.py_buffer(offs),pa.py_buffer(dat))
        idx=pc.index_in(arr,value_set=vk)
        return pc.fill_null(pc.take(vv,idx),-99).to_numpy(zero_copy_only=False)
    b=t(base); a=t(arrow)
    print("  vocab=%6d ncat=%5d  baseline=%.3f ms  arrow_index_in=%.3f ms  %s"
          %(VOCAB,NCAT,b,a,"ARROW WINS %.1fx"%(b/a) if a<b else "arrow loses %.1fx"%(a/b)))
