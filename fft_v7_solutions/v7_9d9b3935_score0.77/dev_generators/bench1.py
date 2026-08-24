import sys, time, numpy as np, os
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/replay/workdir/dev'); sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/replay/workdir')
import solution, base

def persize(L, B, m, reps=3):
    rng_x = np.random.default_rng(42 + L)
    x0 = rng_x.standard_normal((B, L, L, L)) + 1j*rng_x.standard_normal((B, L, L, L))
    rng_c = np.random.default_rng(1000 + L)
    c = 0.1*(rng_c.standard_normal((B, L, L, L)) + 1j*rng_c.standard_normal((B, L, L, L)))
    # mine
    best = 9e9
    for _ in range(reps):
        t0 = time.perf_counter(); solution._run(L, B, m, x0, c); t1 = time.perf_counter()
        best = min(best, t1-t0)
    # base-style
    bb = 9e9
    for _ in range(2):
        t0 = time.perf_counter()
        x = x0
        for k in range(m):
            z = np.fft.fftn(x, axes=(1,2,3)) + c
            x = z/(1.0+np.abs(z))
        t1 = time.perf_counter()
        bb = min(bb, t1-t0)
    pts = B*L**3*m
    print(f"L={L:2d} B={B:5d} m={m:3d}: mine {best*1e3:8.2f} ms ({best/pts*1e9:6.2f} ns/pt-step)  base {bb*1e3:8.2f} ms  speedup {bb/best:6.2f}x")
    return best, bb

tot_m = tot_b = 0
for L, B, m in ((6, 4000, 30), (8, 2000, 30), (13, 400, 30), (17, 180, 30), (23, 75, 30), (36, 20, 30), (45, 10, 30), (64, 4, 30)):
    a, b = persize(L, B, m)
    tot_m += a; tot_b += b
print(f"TOTAL: mine {tot_m*1e3:.1f} ms  base {tot_b*1e3:.1f} ms  speedup {tot_b/tot_m:.2f}x")
