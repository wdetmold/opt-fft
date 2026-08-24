import sys, time
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/replay/workdir/dev')
import numpy as np, solution, ctypes
def t(L, B, m, reps=3):
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
    print(f"L={L:2d} B={B:4d} m={m}: {best*1e3:8.2f} ms  {best/(B*L**3*m)*1e9:5.2f} ns/pt-step")
for L, B in ((6,3000),(8,1300),(13,300),(17,140),(23,55),(36,15),(45,8),(64,3)):
    t(L, B, 30)
