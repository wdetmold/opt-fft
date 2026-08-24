import sys, numpy as np
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/replay/workdir/dev')
sys.path.insert(0, '/private/tmp/claude-501/-Users-wdetmold-MIT-Dropbox-William-Detmold-2026-problems/3e8ea67f-c989-4b31-8f6e-316caac734df/scratchpad/replay/workdir')
import importlib
import solution, base
args = dict(seed=42, B6=2, B8=2, B13=2, B17=2, B23=2, B36=2, B45=1, B64=1,
            m6=5, m8=5, m13=5, m17=5, m23=5, m36=4, m45=3, m64=3)
got = solution.transform(**args)
ref = base.transform(**args)
print("shapes:", got.shape, ref.shape, got.dtype)
# per-block errors
SIZES = (6,8,13,17,23,36,45,64)
Bs = [2,2,2,2,2,2,1,1]
off = 0
ok = True
for L, B in zip(SIZES, Bs):
    n = B * L**3
    for name in ('one', 'final'):
        g = got[off:off+n]; r = ref[off:off+n]
        err = np.linalg.norm(g - r) / np.linalg.norm(r)
        stat = 'OK' if err < 1e-13 else 'FAIL'
        if stat == 'FAIL': ok = False
        print(f"L={L:2d} {name:5s}: rel err {err:.3e} {stat}")
        off += n
print("total off:", off, "expected:", got.size)
assert ok, "CORRECTNESS FAILURE"
print("ALL BLOCKS OK")
