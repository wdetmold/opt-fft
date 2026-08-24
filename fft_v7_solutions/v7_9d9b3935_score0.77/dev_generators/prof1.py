import sys, time
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/replay/workdir/dev')
import numpy as np
import solution

Bs = {6:3000, 8:1300, 13:300, 17:140, 23:55, 36:15, 45:8, 64:3}
m = 30
tot_rng = tot_run = 0.0
for L in (6,8,13,17,23,36,45,64):
    B = Bs[L]
    t0 = time.perf_counter()
    rng_x = np.random.default_rng(123 + L)
    x0r = rng_x.standard_normal((B, L, L, L)); x0i = rng_x.standard_normal((B, L, L, L))
    rng_c = np.random.default_rng(1000 + L)
    cr = rng_c.standard_normal((B, L, L, L)); ci = rng_c.standard_normal((B, L, L, L))
    t1 = time.perf_counter()
    one, fin = solution._run(L, B, m, x0r, x0i, cr, ci)
    t2 = time.perf_counter()
    # run again to see warm time
    one, fin = solution._run(L, B, m, x0r, x0i, cr, ci)
    t3 = time.perf_counter()
    print(f"L={L:2d} rng {1e3*(t1-t0):7.1f} ms  run {1e3*(t2-t1):7.1f} ms  run2 {1e3*(t3-t2):7.1f} ms")
    tot_rng += t1-t0; tot_run += t3-t2
print(f"total rng {tot_rng*1e3:.1f} ms   run {tot_run*1e3:.1f} ms")
