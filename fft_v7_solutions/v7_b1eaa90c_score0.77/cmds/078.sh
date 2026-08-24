cd /workdir && python3 - <<'EOF'
import sys, time, numpy as np
sys.path.insert(0, '/workdir')
import solution
Bs=(2315,977,171,60,24,11,5,2)
m1=(1,)*8
ms=(290,280,180,130,120,110,100,90)
for tag, m in [("m=1", m1), ("m-big", ms)]:
    best=1e9
    for _ in range(2):
        t0=time.perf_counter(); solution.transform(7,*Bs,*m); t1=time.perf_counter()
        best=min(best,t1-t0)
    print(tag, f"{best:.3f}s")
# pure generation time
t0=time.perf_counter()
for L,B in zip((6,8,13,17,23,36,45,64),Bs):
    rng_x = np.random.default_rng(7+L)
    x0 = rng_x.standard_normal((B,L,L,L)) + 1j*rng_x.standard_normal((B,L,L,L))
    rng_c = np.random.default_rng(1000+L)
    c = 0.1*(rng_c.standard_normal((B,L,L,L)) + 1j*rng_c.standard_normal((B,L,L,L)))
t1=time.perf_counter()
print("pure generation:", f"{t1-t0:.3f}s")
EOF