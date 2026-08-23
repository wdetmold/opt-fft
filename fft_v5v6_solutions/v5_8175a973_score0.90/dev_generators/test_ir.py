import numpy as np
from ir import *

for N in (2,3,4,5,6,8,9,13,16,17,23,36,45,64):
    B = Builder()
    xs = [(B.var(f'xr{j}'), B.var(f'xi{j}')) for j in range(N)]
    outs = dft(B, N, xs)
    rng = np.random.default_rng(0)
    xv = rng.standard_normal(N) + 1j*rng.standard_normal(N)
    env = {}
    for j in range(N):
        env[f'xr{j}'] = xv[j].real; env[f'xi{j}'] = xv[j].imag
    got = eval_dag(B, outs, env)
    gotc = np.array([r + 1j*im for r, im in got])
    ref = np.fft.fft(xv)
    err = np.linalg.norm(gotc - ref)/np.linalg.norm(ref)
    ops = count_ops(B, outs)
    tot = ops['add']+ops['sub']+ops['mul']
    print(f"N={N:3d} err={err:.3e} ops={ops} total={tot} per_pt={tot/N:.1f}")
