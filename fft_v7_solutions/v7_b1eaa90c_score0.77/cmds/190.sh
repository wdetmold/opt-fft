cd /workdir && nohup python3 - > /tmp/soak.log 2>&1 <<'EOF' &
import sys, numpy as np, random
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution
SIZES=(6,8,13,17,23,36,45,64)
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
rnd = random.Random(12345)
fails = 0
for trial in range(40):
    Bs = tuple(rnd.choice([0,1,2,3,4,5,6,7,8,9,11,13,16,17]) if L<20 else rnd.choice([0,1,2,3,4,5]) for L in SIZES)
    ms = tuple(rnd.choice([1,2,3,4,5,7,10,15,25]) for _ in SIZES)
    seed = rnd.randrange(0, 10**6)
    args=(seed,)+Bs+ms
    g=solution.transform(*args); r=base.transform(*args)
    off=0; ok=True
    for L,B in zip(SIZES,Bs):
        n=B*L**3
        if n:
            e1=np.linalg.norm(g[off:off+n]-r[off:off+n])/np.linalg.norm(r[off:off+n])
            em=np.linalg.norm(g[off+n:off+2*n]-r[off+n:off+2*n])/np.linalg.norm(r[off+n:off+2*n])
            if not (e1<1e-14 and em<gates[L]):
                ok=False
                print(f"trial {trial} FAIL L={L} B={B} m={ms[SIZES.index(L)]} seed={seed}: {e1:.2e} {em:.2e}", flush=True)
    fails += (not ok)
    if trial % 10 == 0: print(f"trial {trial} done", flush=True)
print(f"SOAK COMPLETE: {fails} failures / 40 trials", flush=True)
EOF
echo "soak started in background"; sleep 45; tail -3 /tmp/soak.log