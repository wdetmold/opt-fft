"""Verify a backend's output against numpy.fft on the same input file.

numpy is the affordable witness here; numpy's own agreement with the from-scratch
definition (python/slow_dft.py) is established separately by python/test_fft3d.py.
"""
import argparse, json, sys, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--L", type=int, required=True)
p.add_argument("--batch", type=int, required=True)
p.add_argument("--tol", type=float, default=1e-12)
p.add_argument("--json", default=None)
a = p.parse_args()

shape = (a.batch, a.L, a.L, a.L)
x = np.fromfile(a.input, dtype=np.complex128).reshape(shape)
got = np.fromfile(a.output, dtype=np.complex128)
if got.size != x.size:
    print(f"FAIL size mismatch: got {got.size} elements, expected {x.size}")
    sys.exit(1)
got = got.reshape(shape)

ref = np.fft.fftn(x, axes=(-3, -2, -1))
denom = np.linalg.norm(ref.reshape(-1))
rel_l2 = float(np.linalg.norm((got - ref).reshape(-1)) / denom)
max_abs = float(np.abs(got - ref).max())
# scale-free per-element measure: the transform of white noise has magnitude ~ sqrt(V)
rel_max = float(max_abs / (np.abs(ref).max()))
ok = bool(np.isfinite(rel_l2) and rel_l2 < a.tol)

result = dict(ok=ok, rel_l2=rel_l2, max_abs=max_abs, rel_max=rel_max, tol=a.tol,
              L=a.L, batch=a.batch)
if a.json:
    with open(a.json, "w") as f:
        json.dump(result, f)
print(f"{'PASS' if ok else 'FAIL'} rel_l2={rel_l2:.3e} rel_max={rel_max:.3e} "
      f"(tol {a.tol:g}) L={a.L} B={a.batch}")
sys.exit(0 if ok else 1)
