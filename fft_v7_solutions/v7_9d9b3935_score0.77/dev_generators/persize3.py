import sys, time
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/replay/workdir/dev')
import numpy as np, solution
def run(L, B, m, reps=4):
    wx = np.random.SeedSequence(3+L).generate_state(4, np.uint64)
    wc = np.random.SeedSequence(1000+L).generate_state(4, np.uint64)
    n = B*L**3
    out = np.empty(2*n, np.complex128)
    PD, PU = solution._PD, solution._PU
    best = 9e9
    for _ in range(reps):
        t0 = time.perf_counter()
        solution._GENS[L](B, m, wx.ctypes.data_as(PU), wc.ctypes.data_as(PU), out[:n].ctypes.data_as(PD), out[n:].ctypes.data_as(PD))
        t1 = time.perf_counter()
        best = min(best, t1-t0)
    return best
for L, B in ((6,3000),(8,1300),(13,300),(17,140),(23,55),(36,15),(45,8),(64,3)):
    t1 = run(L, B, 1)
    t31 = run(L, B, 31)
    step = (t31-t1)/30/(B*L**3)*1e9
    fixed = t1/(B*L**3)*1e9
    print(f"L={L:2d}: marginal step {step:5.2f} ns/pt   fixed(m=1) {fixed:6.2f} ns/pt")
