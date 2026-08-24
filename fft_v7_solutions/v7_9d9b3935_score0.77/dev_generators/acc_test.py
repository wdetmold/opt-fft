import sys, numpy as np
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/replay/workdir')
import solution, base

_CLD, _LD = np.clongdouble, np.longdouble
_PI = np.longdouble('3.14159265358979323846264338327950288')
def _dftmat(L):
    j = np.arange(L)
    kj = np.outer(j, j) % L
    ang = (-2 * _PI) * kj.astype(_LD) / _LD(L)
    return np.cos(ang).astype(_CLD) + 1j * np.sin(ang).astype(_CLD)
def _fft3(x, W):
    y = np.einsum('bxyz,kx->bkyz', x, W)
    y = np.einsum('bkyz,ly->bklz', y, W)
    return np.einsum('bklz,mz->bklm', y, W)

def ld_ref(L, B, m, seed):
    rng_x = np.random.default_rng(seed + L)
    x = (rng_x.standard_normal((B, L, L, L)) + 1j * rng_x.standard_normal((B, L, L, L))).astype(_CLD)
    rng_c = np.random.default_rng(1000 + L)
    c = (0.1 * (rng_c.standard_normal((B, L, L, L)) + 1j * rng_c.standard_normal((B, L, L, L)))).astype(_CLD)
    W = _dftmat(L)
    one = None
    for k in range(m):
        z = _fft3(x, W) + c
        x = z / (1.0 + np.abs(z))
        if k == 0: one = x.copy()
    return one.ravel().astype(np.complex128), x.ravel().astype(np.complex128)

def base_run(L, B, m, seed):
    rng_x = np.random.default_rng(seed + L)
    x = rng_x.standard_normal((B, L, L, L)) + 1j * rng_x.standard_normal((B, L, L, L))
    rng_c = np.random.default_rng(1000 + L)
    c = 0.1 * (rng_c.standard_normal((B, L, L, L)) + 1j * rng_c.standard_normal((B, L, L, L)))
    one = None
    for k in range(m):
        z = np.fft.fftn(x, axes=(1, 2, 3)) + c
        x = z / (1.0 + np.abs(z))
        if k == 0: one = x.copy()
    return one.ravel(), x.ravel()

import ctypes
tols = {6: 1e-4, 8: 3e-6, 13: 1e-9, 17: 1e-10, 23: 1e-10, 36: 1e-10, 45: 1e-10, 64: 1e-10}
# chain lengths to probe; pick m where base-vs-longdouble divergence ~ 2.5-3 orders below tol
probe = {6: [40, 80, 120], 8: [40, 80, 120], 13: [40, 80], 17: [30, 60], 23: [30, 60], 36: [20, 40], 45: [20, 40], 64: [20, 40]}
for L in (6, 8, 13, 17, 23, 36, 45, 64):
    B = 2 if L <= 23 else 1
    for m in probe[L]:
        ref1, refm = ld_ref(L, B, m, 3)
        b1, bm = base_run(L, B, m, 3)
        wx = np.random.SeedSequence(3 + L).generate_state(4, np.uint64)
        wc = np.random.SeedSequence(1000 + L).generate_state(4, np.uint64)
        n = B * L**3
        mine = np.empty(2*n, dtype=np.complex128)
        PD = solution._PD
        solution._GENS[L](B, m, wx.ctypes.data_as(solution._PU), wc.ctypes.data_as(solution._PU),
                          mine[:n].ctypes.data_as(PD), mine[n:].ctypes.data_as(PD))
        e_base = np.linalg.norm(bm - refm)/np.linalg.norm(refm)
        e_mine = np.linalg.norm(mine[n:] - refm)/np.linalg.norm(refm)
        e_one  = np.linalg.norm(mine[:n] - ref1)/np.linalg.norm(ref1)
        print(f"L={L:2d} m={m:3d}: one-step vs LD {e_one:.2e} | m-step: base {e_base:.2e}  mine {e_mine:.2e}  tol {tols[L]:.0e}  {'OK' if e_mine < tols[L] else 'OVER'}")
