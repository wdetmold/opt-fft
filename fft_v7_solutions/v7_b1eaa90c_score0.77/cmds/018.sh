cd /workdir && python3 - <<'EOF'
import numpy as np
# divergence growth: perturb x0 by 1 ulp and watch relative L2 error growth per step
_SIZES=(6,8,13,17,23,36,45,64)
gates={6:1e-4,8:3e-6,13:1e-9,17:1e-10,23:1e-10,36:1e-10,45:1e-10,64:1e-10}
for L in _SIZES:
    B=2
    rng_x = np.random.default_rng(7+L)
    x = rng_x.standard_normal((B,L,L,L)) + 1j*rng_x.standard_normal((B,L,L,L))
    rng_c = np.random.default_rng(1000+L)
    c = 0.1*(rng_c.standard_normal((B,L,L,L)) + 1j*rng_c.standard_normal((B,L,L,L)))
    y = x.copy()
    y.flat[0] *= (1+2**-52)
    hist=[]
    m_est=None
    for k in range(1,400):
        z = np.fft.fftn(x,axes=(1,2,3))+c; x = z/(1+np.abs(z))
        z = np.fft.fftn(y,axes=(1,2,3))+c; y = z/(1+np.abs(z))
        rel = np.linalg.norm(x-y)/np.linalg.norm(x)
        hist.append(rel)
        if rel > gates[L]/300 and m_est is None:
            m_est = k
        if rel > gates[L]:  # way beyond
            break
    import math
    # growth rate fit on mid region
    h=np.array(hist)
    print(f"L={L:2d} gate={gates[L]:.0e} est m*(gate/300)={m_est} first few: {h[:3]} growth/step ~ {np.exp(np.mean(np.diff(np.log(h[3:min(len(h),40)]))))}")
EOF