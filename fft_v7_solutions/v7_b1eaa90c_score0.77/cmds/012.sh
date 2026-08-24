cd /workdir && python3 - <<'EOF'
import time, numpy as np
rng = np.random.default_rng(13)
for n in [10**6, 10**7]:
    t0=time.perf_counter(); a = rng.standard_normal(n); t1=time.perf_counter()
    print(f"standard_normal({n}): {t1-t0:.4f}s = {(t1-t0)/n*1e9:.2f} ns/double")
# complex generation like the reference does
B,L=16,64
t0=time.perf_counter()
x0 = rng.standard_normal((B,L,L,L)) + 1j*rng.standard_normal((B,L,L,L))
t1=time.perf_counter()
print(f"complex x0 gen B=16 L=64: {t1-t0:.3f}s for {B*L**3} elems = {(t1-t0)/(B*L**3)*1e9:.1f} ns/complex-elem")
EOF