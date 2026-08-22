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
p.add_argument("--chain-check", type=int, default=0, metavar="M",
               help="also verify the end state of a unitary-normalized chain of M steps "
                    "(driver --chain M --unitary) against its closed form")
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
if a.chain_check > 0:
    # Closed form of the unitary-normalized chain: each step is FFT then 1/sqrt(V), and
    # FFT^2 = V * index-reversal, so after m steps:
    #   m%4==0: x    1: FFT(x)/sqrt(V)    2: x[-j]    3: FFT(x[-j])/sqrt(V)
    # One reference FFT at most, and any error in any of the m steps compounds into the end
    # state -- a stronger test than a single transform.
    m = a.chain_check
    got_chain = np.fromfile(a.output + ".chain", dtype=np.complex128).reshape(shape)
    ridx = (-np.arange(a.L)) % a.L
    rev = x[:, ridx][:, :, ridx][:, :, :, ridx]
    r = m % 4
    if r == 0:
        want = x
    elif r == 1:
        want = np.fft.fftn(x, axes=(-3, -2, -1)) / a.L ** 1.5
    elif r == 2:
        want = rev
    else:
        want = np.fft.fftn(rev, axes=(-3, -2, -1)) / a.L ** 1.5
    denom = np.linalg.norm(want.reshape(-1))
    rel_chain = float(np.linalg.norm((got_chain - want).reshape(-1)) / (denom if denom else 1.0))
    eff_tol = a.tol * max(1.0, m ** 0.5)   # roundoff accumulates ~sqrt(m)
    chain_ok = bool(np.isfinite(rel_chain) and rel_chain < eff_tol)
    print(f"{'PASS' if chain_ok else 'FAIL'} chain m={m}: rel_l2={rel_chain:.3e} (tol {eff_tol:.1e})")
    ok = ok and chain_ok
    if a.json:
        result["chain_ok"] = chain_ok
        result["chain_rel_l2"] = rel_chain
        with open(a.json, "w") as f:
            json.dump(result, f)

sys.exit(0 if ok else 1)

