"""Verify a backend's output against numpy.fft on the same input file.

numpy is the affordable witness here; numpy's own agreement with the from-scratch
definition (python/slow_dft.py) is established separately by python/test_fft3d.py.
"""
import math
import argparse, json, sys, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--L", type=int, required=True)
p.add_argument("--batch", type=int, required=True)
p.add_argument("--tol", type=float, default=1e-12)
p.add_argument("--json", default=None)
p.add_argument("--map-check", type=int, default=0, metavar="M",
               help="verify the end state of an m-step MAP chain "
                    "(state <- (FFT(state)+c)/(1+|FFT(state)+c|)) against a numpy "
                    "reference chain; needs --cin")
p.add_argument("--cin", default=None, help="the map's c field file")
p.add_argument("--chain-check", type=int, default=0, metavar="M",
               help="also verify the end state of a unitary-normalized chain of M steps "
                    "(driver --chain M --unitary) against its closed form")
a = p.parse_args()

shape = (a.batch, a.L)
x = np.fromfile(a.input, dtype=np.complex128).reshape(shape)
got = np.fromfile(a.output, dtype=np.complex128)
if got.size != x.size:
    print(f"FAIL size mismatch: got {got.size} elements, expected {x.size}")
    sys.exit(1)
got = got.reshape(shape)

ref = np.fft.fft(x, axis=-1)
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
if a.map_check > 0:
    # Two-part gate (see docs/GRADER.md, "Gating a chaotic chain"):
    #   m == 1  ->  ONE-STEP gate, rel L2 < 1e-14 vs the numpy reference step. This is
    #     the precision contract: chaos cannot amplify anything in one step, careful fp64
    #     lands ~1e-15, and every shortcut (fp32-seeded maps ~2.6e-12, fp32 interiors,
    #     skipped work) fails here.
    #   m > 1   ->  CHAIN gate at 300x the honest divergence of two reference paths
    #     (numpy fftn vs V*conj(ifftn(conj(.)))) run over the SAME chain. The map chain is
    #     weakly chaotic: correct fp64 implementations diverge ~exponentially with m
    #     (measured: MKL/FFTW/ducc0 land at 2e-10..1.1e-8 at L=6, m=4856 depending on
    #     seed), so a fixed tolerance either fails honest code or is meaningless. 300x =
    #     30x for a solver legally at the one-step ceiling (1e-14 vs the references'
    #     ~3e-16/step) times a decade of slop, rounded up onto a {1,3}x10^n grid, floored
    #     at 1e-10. Still 4+ orders below an fp32-interior chain and O(1) bugs.
    m = a.map_check
    cf = np.fromfile(a.cin, dtype=np.complex128).reshape(shape)
    got_chain = np.fromfile(a.output + ".chain", dtype=np.complex128).reshape(shape)
    if m <= 2:
        state = x.copy()
        for _ in range(m):
            z = np.fft.fft(state, axis=-1) + cf
            state = z / (1.0 + np.abs(z))
        denom = np.linalg.norm(state.reshape(-1))
        rel_one = float(np.linalg.norm((got_chain - state).reshape(-1)) / (denom if denom else 1.0))
        tol_one = 1.5e-14 * m
        one_ok = bool(np.isfinite(rel_one) and rel_one < tol_one)
        print(f"{'PASS' if one_ok else 'FAIL'} map-{m}-step: rel_l2={rel_one:.3e} (tol {tol_one:.1e})")
        ok = ok and one_ok
        if a.json:
            result["one_ok"] = one_ok
            result["one_rel_l2"] = rel_one
            with open(a.json, "w") as f:
                json.dump(result, f)
    else:
        V = float(a.L)
        sa = x.copy()
        sb = x.copy()
        for _ in range(m):
            za = np.fft.fft(sa, axis=-1) + cf
            sa = za / (1.0 + np.abs(za))
            zb = V * np.conj(np.fft.ifft(np.conj(sb), axis=-1)) + cf
            sb = zb / (1.0 + np.abs(zb))
        denom = np.linalg.norm(sa.reshape(-1))
        rel_chain = float(np.linalg.norm((got_chain - sa).reshape(-1)) / (denom if denom else 1.0))
        anchor = float(np.linalg.norm((sa - sb).reshape(-1)) / (denom if denom else 1.0))
        raw = max(300.0 * anchor, 1e-10)
        exp = math.floor(math.log10(raw))
        mant = raw / 10 ** exp
        eff_tol = (1.0 if mant <= 1.0 else (3.0 if mant <= 3.0 else 10.0)) * 10 ** exp
        chain_ok = bool(np.isfinite(rel_chain) and rel_chain < eff_tol)
        print(f"{'PASS' if chain_ok else 'FAIL'} map-chain m={m}: rel_l2={rel_chain:.3e} "
              f"(anchor {anchor:.3e}, tol {eff_tol:.1e})")
        ok = ok and chain_ok
        if a.json:
            result["chain_ok"] = chain_ok
            result["chain_rel_l2"] = rel_chain
            result["anchor_rel_l2"] = anchor
            result["chain_tol"] = eff_tol
            with open(a.json, "w") as f:
                json.dump(result, f)

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
        want = np.fft.fft(x, axis=-1) / a.L ** 1.5
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

