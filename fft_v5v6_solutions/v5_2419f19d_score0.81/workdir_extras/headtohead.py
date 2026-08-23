import sys, time, numpy as np, importlib
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/fft_recon/sandbox/workdir')
import base, solution

def timeit(fn, args, reps=3):
    best=1e9
    for _ in range(reps):
        t0=time.perf_counter(); fn(*args); t1=time.perf_counter()
        best=min(best,t1-t0)
    return best

scen = [
 ("m small",  (42, 4,4,4,4,4,2,2,1,  3,3,3,3,3,3,3,3)),
 ("m medium", (42, 4,4,4,4,4,2,2,1,  20,20,20,20,20,20,20,20)),
 ("m large",  (42, 2,2,2,2,2,1,1,1,  60,60,60,60,60,60,60,60)),
 ("bigB",     (42, 16,16,8,8,4,3,2,2, 12,12,12,12,12,12,12,12)),
]
for name, args in scen:
    tb = timeit(base.transform, args, 3)
    ts = timeit(solution.transform, args, 3)
    # verify
    ob = base.transform(*args); os_ = solution.transform(*args)
    n = ob.size//2
    err = np.linalg.norm(os_[:]-ob[:])/np.linalg.norm(ob)
    print(f"{name:10s} base={tb:7.3f}s ours={ts:7.3f}s speedup={tb/ts:5.2f}x relerr(full)={err:.2e}")
