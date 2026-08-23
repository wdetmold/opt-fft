import sys, numpy as np
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/fft_recon/sandbox/workdir')
import solution

_SIZES = (6, 8, 13, 17, 23, 36, 45, 64)
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

def ref_transform(seed, Bs, ms):
    outs = []
    for L in _SIZES:
        B, m = Bs[L], ms[L]
        rng_x = np.random.default_rng(seed + L)
        x = (rng_x.standard_normal((B, L, L, L))
             + 1j * rng_x.standard_normal((B, L, L, L))).astype(_CLD)
        rng_c = np.random.default_rng(1000 + L)
        c = (0.1 * (rng_c.standard_normal((B, L, L, L))
             + 1j * rng_c.standard_normal((B, L, L, L)))).astype(_CLD)
        one = None
        for k in range(m):
            z = _fft3(x, _dftmat(L)) + c
            x = z / (1.0 + np.abs(z))
            if k == 0:
                one = x.copy()
        outs += [one.ravel().astype(np.complex128), x.ravel().astype(np.complex128)]
    return np.concatenate(outs)

Bs={L:1 for L in _SIZES}
ms={L:3 for L in _SIZES}
args=[505]+[Bs[L] for L in _SIZES]+[ms[L] for L in _SIZES]
oref = ref_transform(505, Bs, ms)
ours = solution.transform(*args)
off=0
ok=True
for L in _SIZES:
    n=Bs[L]*L**3
    for kind,lim in (('one',1e-14),('fin',1e-3)):
        x=oref[off:off+n]; y=ours[off:off+n]; off+=n
        e=np.linalg.norm(y-x)/np.linalg.norm(x)
        margin = lim/e
        print(f"L={L:3d} {kind}: rel={e:.3e}  (bar {lim:g}, margin {margin:8.1f}x)")
        if e>=lim: ok=False
print("LONGDOUBLE REFERENCE CHECK:", "PASS" if ok else "FAIL")
