import sys, numpy as np
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/replay/workdir/dev')
import solution
from acc_test import ld_ref
tols = {6: 1e-4, 8: 3e-6, 13: 1e-9, 17: 1e-10, 23: 1e-10, 36: 1e-10, 45: 1e-10, 64: 1e-10}
for L, m in ((6, 1000), (8, 600), (13, 200), (17, 120), (23, 100), (36, 80), (45, 60), (64, 60)):
    B = 1
    ref1, refm = ld_ref(L, B, m, 3)
    wx = np.random.SeedSequence(3 + L).generate_state(4, np.uint64)
    wc = np.random.SeedSequence(1000 + L).generate_state(4, np.uint64)
    n = B * L**3
    mine = np.empty(2*n, dtype=np.complex128)
    PD = solution._PD
    solution._GENS[L](B, m, wx.ctypes.data_as(solution._PU), wc.ctypes.data_as(solution._PU),
                      mine[:n].ctypes.data_as(PD), mine[n:].ctypes.data_as(PD))
    e_m = np.linalg.norm(mine[n:] - refm)/np.linalg.norm(refm)
    e_1 = np.linalg.norm(mine[:n] - ref1)/np.linalg.norm(ref1)
    print(f"L={L:2d} m={m:4d}: one {e_1:.2e}  m-step {e_m:.3e}  tol {tols[L]:.0e}  {'OK' if e_m < tols[L] else 'FAIL'}")
