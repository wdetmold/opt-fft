# gen_bluestein — strategy record

Class: Bluestein chirp-Z existence fallback for ANY L. Never scored on speed alone;
must run everywhere and never fail — it is the library's guarantee that round 6's
surprise sizes always plan. Speed still matters (it sets the library's worst case).

## Round gen_r1

### Starting point
The seed was the dense O(L^4)-per-axis matrix floor (validation stub). Context file
was empty — no other generations' records existed yet this round. Nothing borrowed
from rival entries (there was nothing to borrow); the 8-row SoA lane idea is the same
spirit as the batchlane seed material (L13_rader soa8) applied to Bluestein rows.

### What I built (impl/gen_bluestein.c, complete rewrite)
Real Bluestein: X_k = ch[k] * (a ⊛ b)[k] with a_j = x_j*ch[j], ch[n] = e^{-iπn²/L},
b_n = e^{+iπn²/L} embedded circularly in M = next_pow2(2L-1) (M ≤ 2L padding waste is
2.56x worst case at L=100→M=256; radix-3 sizes like 192 never beat the pow2 choice on
the acceptance list — checked arithmetic, not measured).

Key design decisions, in the order they mattered:

1. **8-row lane batching**: rows are transformed 8 at a time; work buffers are split
   re/im with the 8 row lanes contiguous per element (one zmm of doubles). Every
   butterfly is one 8-wide AVX-512 op. Buffers 2 × M×8 doubles = 32 KiB at M=256,
   L1-resident.
2. **No bit-reversal, ever**: forward = in-place radix-4 DIF (radix-2 tail for odd
   log2 M), inverse = the literal stage-by-stage inverse in mirrored order (radix-4
   DIT). Pointwise multiply by FFT(b)/M happens in the pipeline's own scrambled
   order (bh table computed at create by running my own forward on b_pad), so
   correctness is structural — the scramble permutation never needs to be named.
3. **Intrinsics, not autovectorization**: gcc 12 `-O3 -march=native` REFUSED to
   vectorize the 8-lane butterfly loops ("complicated access pattern" /
   "no vectype") and produced scalar code at 7.5 GF/s real. Hand intrinsics for
   dif4/dit4/r2/pointwise took L=100 B=1 from 80.4 ms to 29.0 ms. Lesson for every
   entry: CHECK `-fopt-info-vec-missed`; also note gcc targets 256-bit by default on
   Ice Lake even when it does vectorize, so intrinsics are the only reliable 512-bit
   path under the fixed harness flags.
4. **Zero-pad pruning (first stage)**: M ≥ 2L-1 ⇒ the upper half of the conv input
   is structurally zero and, since M/4 < L, the first DIF stage's c/d legs vanish and
   the b leg vanishes for j ≥ L-M/4. Specialized `dif4_first` never reads the pad,
   which also kills both work-buffer memsets. **Output pruning (last stage)**: only
   k < L is consumed, and L ≤ M/2, so `dit4_last` never computes/stores the r2/r3
   quarters and drops r1 for j ≥ L-M/4. Together: 28.5 ms (small; the FFT wasn't the
   bottleneck yet — see 6).
5. **Fused convolution middle**: the forward's last stage (len-4, all twiddles = 1,
   or the radix-2 tail for odd log2 M), the pointwise bh multiply, and the inverse's
   first stage all touch the same 4 (resp. 2) consecutive positions — `conv_mid4` /
   `conv_mid2` do all three in registers in one pass. Eliminates two full buffer
   round-trips: 24.4 → 20.5 ms at L=100; L=25 271 → 229 µs; L=10 18.7 → 15.6 µs.
6. **Vectorized gather/scatter** (was the hidden cost): three cases.
   - Axes 0/1: lane groups are 8 *adjacent* rows, so element j of all 8 lanes is 16
     consecutive doubles → 2 loadu + 2 `permutex2var` (deinterleave) per element;
     scatter is the mirror interleave. Falls back to scalar at outer-run seams
     (detected per group: `(r0 % div) + 8 <= div`).
   - Axis 2 (rows contiguous, lanes strided): 8×8 double-block transpose in
     registers (8 unpack + 16 `shuffle_f64x2`, self-inverse network `tr8x8`).
   - Chirp multiply is fused into both gather and scatter. 28.5 → 24.4 ms.
7. **Exact tables**: chirp phase reduced exactly as (j² mod 2L), all trig in long
   double. Twiddles per stage stored in consumption order, ~2M doubles total. bh in
   double via my own forward (create-time `fwd_fft_full`, which must NOT use the
   pruned path — b_pad is not zero-padded; that distinction is load-bearing).

### Operation count (per volume, L=100, M=256)
Per 8-row group: 2 × (4-stage radix-4 M=256 FFT) ≈ 17.4k flops/row-equivalent + fused
middle + chirps ≈ 20k flops/row; 3L² rows ⇒ ~0.6 Gflop/volume. Measured 20.8 ms ⇒
~29 Gflop/s real on one Ice Lake core — about 2.2x over the pure-FMA-port ideal
(~9.5 ms), remainder is load/store port pressure and dep chains, not DRAM (traffic
floor ~8 ms is not binding; see failed experiment below).

### Measured on the node (a80n0, Ice Lake, pinned core, min over samples)
All PASS rel_l2 ≤ 8.8e-16 (tol 1e-12); map-chain gates: L=10 m=1000 1.06e-13
(anchor 4.9e-14), L=31 m=140 4.3e-14 (anchor 2.3e-14), L=100 m=64 4.1e-14
(anchor 2.4e-14) — drift ≤ 2.2x honest, per-step error ~20x under the 1.5e-14
contract. Output bit-repeatable.

| L | B | µs/xform |    | L | B | µs/xform |
|---|---|---------|----|---|---|---------|
| 10 | 1 | 15.5 |  | 27 | 16 | 285 |
| 10 | 64 | 15.6 |  | 31 | 16 | 414 |
| 12 | 64 | 24.1 |  | 32 | 8 | 417 |
| 15 | 32 | 44.3 |  | 40 | 8 | 1307 |
| 20 | 32 | 131.5 |  | 50 | 4 | 2299 |
| 25 | 16 | 229 |  | 100 | 1 | 20815 |

Generality sweep (the class mandate): L = 2, 3, 7, 17, 23, 64, 101, 127, 128 all
PASS at ≤ 9e-16, covering the M=4 special path, odd-log2 (conv_mid2) and even-log2
(conv_mid4) pipelines, and both giant primes. `supports()` accepts 2..2048.
create() is sub-millisecond (driver reports setup=0.000 s) — no plan-budget risk.

Rough MKL yardstick (6 ns/pt/step from the round-1 calibration note): we are ~2.4x
MKL at L=25, ~2.6x at L=10, ~3.4x at L=100. For an any-L fallback that is where it
should be; the class winners should beat me on every scored size, and if one doesn't
(L=31: dense/Rader fight, my 414 µs is the number to beat), that is a signal.

### What did NOT work, with numbers
- **Plane-fused axis-2+1 passes** (do both transforms per (b,x) plane while it is
  L2-hot, saving one full-volume DRAM sweep): L=100 20.5 → 21.0 ms, L=50
  2299 → 2393 µs. The passes are not bandwidth-bound, and per-plane lane tails add
  ~4-12% more groups. Reverted (rlo/rhi range plumbing kept).
- **Twiddle-free len-4 stage as a separate pass** (dif4_notw/dit4_notw): measured
  flat-to-slightly-worse (24.4 → 25.2 ms, within noise but not a win) as a
  standalone pass; the same arithmetic saving only paid off once it was fused with
  the pointwise multiply into conv_mid4. Passes over the buffer, not flops, were
  the binding cost.
- **gcc autovectorization of the lane loops**: total failure, see (3). Do not waste
  a round on `restrict`/pragma games; use intrinsics.

### Tooling note for other implementers
tryout.sh has a `set -u` bug: line 36 uses `$W` before line 38 defines it — export
`W=/home/lqcd/wdetmold/fft/bench/gen/build/tryout/<name>` before calling. Separately,
the check.py map-check inside the ssh heredoc expands `$W` on the REMOTE side (single
quotes inside `$(...)`) where W is unset, so the chain gate never runs from tryout;
run check.py with `--map-check m --cin` manually on the node afterwards. Also:
slurm binaries are not on PATH in this shell — `export
PATH=/opt/software/slurm-19.05.8.1/bin:$PATH` or reserve.sh/tryout.sh's alive-check
fails and claims there is no reservation.

### What I would do next
1. Fuse gather→dif4_first and dit4_last→scatter (one less L-prefix round-trip each;
   est. 5-10%).
2. Radix-16 fusion of the two middle generic stages at M=256/64 (halves buffer
   passes for the biggest sizes; register-pressure experiment needed).
3. Adopt gen_twiddle/gen_layout library layers when they exist; my chirp/twiddle
   builder should defer to the twiddle layer's exact tables.
4. Offer the planner a `bluestein_cost(L)` model (≈ 3L² rows × 2 M-FFTs) so the
   race only tries me when no specialized chain exists.
5. If the fixed harness flags ever allow, `-mprefer-vector-width=512` matters for
   any remaining C loops; with intrinsics it currently does not.
