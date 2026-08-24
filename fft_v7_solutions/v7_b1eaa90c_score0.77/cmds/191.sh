cd /workdir && python3 - <<'EOF'
import sys, numpy as np
sys.path.insert(0,'/workdir')
import importlib.util
spec = importlib.util.spec_from_file_location("base", "/workdir/base.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
import solution

# longdouble reference for L=45 one step
_CLD, _LD = np.clongdouble, np.longdouble
_PI = np.longdouble('3.14159265358979323846264338327950288')
def _dftmat(L):
    j = np.arange(L); kj = np.outer(j, j) % L
    ang = (-2 * _PI) * kj.astype(_LD) / _LD(L)
    return np.cos(ang).astype(_CLD) + 1j * np.sin(ang).astype(_CLD)
def _fft3(x, W):
    y = np.einsum('bxyz,kx->bkyz', x, W)
    y = np.einsum('bkyz,ly->bklz', y, W)
    return np.einsum('bklz,mz->bklm', y, W)

seed=212191; L=45; B=4
rng_x = np.random.default_rng(seed + L)
x = (rng_x.standard_normal((B,L,L,L)) + 1j*rng_x.standard_normal((B,L,L,L))).astype(_CLD)
rng_c = np.random.default_rng(1000+L)
c = (0.1*(rng_c.standard_normal((B,L,L,L)) + 1j*rng_c.standard_normal((B,L,L,L)))).astype(_CLD)
z = _fft3(x, _dftmat(L)) + c
ref1 = (z / (1.0 + np.abs(z))).ravel().astype(np.complex128)

Bs=[0]*8; ms=[1]*8; Bs[6]=B; ms[6]=10
g = solution.transform(seed, *Bs, *ms)
r = base.transform(seed, *Bs, *ms)
n = B*L**3
g1 = g[:n]; b1 = r[:n]
for name, arr in (("ours", g1), ("base", b1)):
    e = np.linalg.norm(arr-ref1)/np.linalg.norm(ref1)
    print(f"{name} vs longdouble: {e:.3e}")
print(f"ours vs base: {np.linalg.norm(g1-b1)/np.linalg.norm(b1):.3e}")
d = np.abs(g1-ref1)
i = np.argsort(d)[-5:]
print("worst elems ours-vs-ref:", [(int(k), float(d[k]), complex(ref1[k])) for k in i])
db = np.abs(b1-ref1)
print("same elems base err:", [float(db[k]) for k in i])
print("norm of ref1:", np.linalg.norm(ref1))
EOF