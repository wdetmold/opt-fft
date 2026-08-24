cd /workdir && nohup python3 - > /tmp/soak2.log 2>&1 <<'EOF' &
import sys, numpy as np, random
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
SIZES=(6,8,13,17,23,36,45,64)
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
rnd = random.Random(99)
fails=0; worst1=0.0
for trial in range(120):
    Bs = tuple(rnd.choice([1,2,3,4,5]) for _ in SIZES)
    ms = tuple(rnd.choice([1,2,3,5,8,12]) for _ in SIZES)
    seed = rnd.randrange(0,10**6)
    args=(seed,)+Bs+ms
    g=solution.transform(*args); r=base.transform(*args)
    off=0
    for L,B in zip(SIZES,Bs):
        n=B*L**3
        e1=np.linalg.norm(g[off:off+n]-r[off:off+n])/np.linalg.norm(r[off:off+n])
        em=np.linalg.norm(g[off+n:off+2*n]-r[off+n:off+2*n])/np.linalg.norm(r[off+n:off+2*n])
        worst1=max(worst1,e1)
        if not (e1<1e-14 and em<gates[L]):
            fails+=1; print(f"trial {trial} FAIL L={L} {e1:.3e} {em:.3e} args={args}", flush=True)
        off+=2*n
    if trial%30==0: print(f"trial {trial}, worst one-step so far {worst1:.3e}", flush=True)
print(f"SOAK2 COMPLETE: {fails} failures / 120 trials, worst one-step {worst1:.3e}", flush=True)
EOF
echo started; sleep 120; tail -4 /tmp/soak2.log