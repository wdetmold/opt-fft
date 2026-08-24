import os, sys, json, subprocess, ctypes, time
import numpy as np
os.sched_setaffinity(0, {0})
SIZES=(6,8,13,17,23,36,45,64)

def build(cfg, tag):
    env = dict(os.environ); env['GENCFG'] = json.dumps(cfg)
    subprocess.run([sys.executable, 'gen.py'], check=True, env=env, capture_output=True)
    so = f"/tmp/impl_{tag}.so"
    subprocess.run(['gcc','-O3','-march=native','-shared','-fPIC','../implementation.c','-o',so,'-lm'], check=True)
    return so

def bench(so, Ls=SIZES, params=None, reps=3):
    lib = ctypes.CDLL(so)
    out = {}
    dflt={6:(64,60),8:(64,60),13:(32,40),17:(16,40),23:(8,40),36:(4,25),45:(2,25),64:(2,15)}
    for L in Ls:
        B, m = (params or dflt)[L]
        n = B*L**3
        rng = np.random.default_rng(5)
        x0 = (rng.standard_normal(2*n)).view(np.float64)
        x0c = np.empty(2*n); x0c[:] = x0
        c = 0.1*rng.standard_normal(2*n)
        o1 = np.empty(2*n); om = np.empty(2*n)
        f = getattr(lib, f'run{L}')
        f.argtypes=[ctypes.c_void_p]*2+[ctypes.c_long]*2+[ctypes.c_void_p]*2
        f(x0c.ctypes.data, c.ctypes.data, B, 2, o1.ctypes.data, om.ctypes.data)  # warm
        best=1e18
        for _ in range(reps):
            t0=time.perf_counter()
            f(x0c.ctypes.data, c.ctypes.data, B, m, o1.ctypes.data, om.ctypes.data)
            t1=time.perf_counter()
            best=min(best, t1-t0)
        # subtract fixed via m=1
        bestf=1e18
        for _ in range(reps):
            t0=time.perf_counter()
            f(x0c.ctypes.data, c.ctypes.data, B, 1, o1.ctypes.data, om.ctypes.data)
            t1=time.perf_counter()
            bestf=min(bestf, t1-t0)
        per=(best-bestf)/(m-1)/n*1e9
        out[L]=per
    return out

if __name__ == "__main__":
    cfgs = {
        'base':   {'pw':{}, 'pf':{}},
        'newton': {'pw':{str(L):'newton' for L in SIZES}, 'pf':{}},
        'nopf':   {'pw':{}, 'pf':{str(L):0 for L in SIZES}},
        'newtnopf': {'pw':{str(L):'newton' for L in SIZES}, 'pf':{str(L):0 for L in SIZES}},
    }
    res = {}
    for tag, cfg in cfgs.items():
        so = build(cfg, tag)
        res[tag] = bench(so)
        print(tag, {L: round(v,3) for L,v in res[tag].items()}, flush=True)
