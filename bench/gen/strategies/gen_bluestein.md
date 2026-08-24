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

## Round gen_r2

### What changed (impl/gen_bluestein.c, three structural changes + two raced-off knobs)

1. **Adopted gen_twiddle** (named borrow; they built the drop-ins for this entry —
   their r1 next-steps item 1): chirp via `tw_chirp` (exact k² mod 2L + octant
   fold, 0.554 → 0.500 ulp), per-stage twiddles via `tw_fill_ct_int_colmajor`
   (their filler for exactly my `*tf++ = cos; *tf++ = sin` order; inverse =
   fill + negate im), and `tw_audit_ct_int_colmajor ≤ 0.51 ulp` asserted in
   create(). `#define GEN_TWIDDLE_LIB_ONLY` + include, zero link footprint.
2. **Gather and scatter fused into the pruned end stages** (my r1 next-step 1).
   The r1 pipeline wrote L chirped elements to w, then dif4_first re-read them
   (mirrored on the output side): ~4L zmm of pure L1 round trip per group.
   Now `first_gather_contig/tr` feed the first-stage butterfly straight from
   the chirped source rows (dual-leg j < jful, single-leg j ≥ jful), and
   `last_scatter_contig/tr` chirp+interleave dit4_last's outputs straight to
   dst. Buffer passes per group at M=256: 9 → 7. The old unfused path remains
   for axis-0/1 seam groups, M < 16, and non-AVX512 builds.
3. **Radix-16 fused middle stages** (r1 next-step 2): `dif16_stage`/`dit16_stage`
   run two radix-4 layers per buffer pass on the register-closed 16-element
   tile p(q,t) = blk + q·(len/4) + t·(len/16) + j2 (layer 1 = legs across q,
   layer 2 = legs across t; second-layer twiddles depend on j2 only and hoist
   out of the q loop). Applied when len ≥ 32, so M ∈ {128, 256, 512, ...}
   collapse their whole generic middle into one pass each way; M ≤ 64 sizes
   are untouched. Passes per group at M=256: 7 → 5. 32 live zmm — gcc spills
   a few, but the spills are L1 traffic we previously paid as full stage
   stores. Arithmetic order is bit-identical to the unfused stages.
   `-DBST_NOR16` restores radix-4 (raced: see numbers).

### The regression I shipped first, and its fix
The first fused `first_gather_tr` handled the block straddling jful (dual→
single leg boundary) per lane, in scalar. At M=32 that block is HALF of S:
L=10 went 15.6 → 16.8, L=25 229 → 244.9 on the node. Fix: the boundary block
loads its b-leg with `_mm512_maskz_loadu_pd` (zero-filled lanes make the dual
butterfly degenerate to the single-leg formula exactly; masked lanes suppress
faults, so the past-row-end addresses are safe), chirp tables get 4 slack
entries (create() fills cos=1/sin=0) for the dead b-chirp indices, and the
partial r1 block in the scatter stores through `_mm512_mask_storeu_pd` after a
zero-padded tr8x8. After the fix: L=10 15.71, L=15 42.64, L=25 226.3 — all at
or better than r1.

### Measured on the node (a80n0, leased core, graded chain cells, min µs/xform)

| L | B | r1 | r2 |    | L | B | r1 | r2 |
|---|---|-----|-----|--|---|---|------|------|
| 10 | 64 | 15.6 | **15.71** |  | 31 | 16 | 414 | **381.6** |
| 12 | 64 | 24.1 | **23.69** |  | 32 | 8 | 417 | **406.9** |
| 15 | 32 | 44.3 | **42.64** |  | 40 | 8 | 1307 | **1260.6** |
| 20 | 32 | 131.5 | **124.98** |  | 50 | 4 | 2299 | **2208** |
| 25 | 16 | 229 | **226.3** |  | 100 | 1 | 20815 | **19905** (19.2–19.4 ms in quiet A/B windows) |
| 27 | 16 | 285 | **273.6** |  | 10 | 1 (m=1000) | — | 16.04 |

Same-window alternating A/B at L=100 B=1 (the honest attribution, 3/3 reps):
r1 binary 20.77/20.82/20.84 ms vs fused gather/scatter 20.04/20.10/19.99
(**+3.8%**); radix-16 on top: 19.61/19.61 vs 19.76/19.75 (**+0.8%**). Net
round gain ~4–8% depending on size (bigger where M ≥ 128 and at the sizes the
r1 tr-boundary was clumsy).

### Gates (shipped binary, on the node)
Singles ≤ 8.9e-16 everywhere (identical digits to r1 — the fused arithmetic is
op-for-op the same). Two-step gate: L=31 2.53e-15, L=100 3.80e-15 (tol 3e-14,
~10x margin). Chains: L=10 m=1000 2.22e-13 (anchor 1.08e-13), L=31 m=140
4.58e-14 (anchor 2.31e-14), L=100 m=64 3.83e-14 (anchor 2.42e-14) — all ≤ 2x
honest drift, tol 1e-10. Generality sweep PASS at
L ∈ {2,3,4,5,7,9,11,13,16,17,23,33,47,63,64,65,96,101,127,128} ≤ 8.9e-16
(covers M=4 special path, odd/even log2, the fused radix-16 pipelines, masked
boundaries, big primes). Scalar (-march=x86-64) fallback build PASS at
{3,10,25,31,100}. create() still ~0 s.

### Where the time actually goes (new this round: BST_AXES dev knob)
Single-execute split at L=100: axis 2 = 7.23 ms, axis 1 = 5.13 ms, axis 0 =
5.20 ms. The strided axes are NOT memory-latency-bound: hand prefetch on the
contig gather/scatter (`-DBST_PF`) LOST same-window (19.67/19.63 with vs
19.48/19.34 without) — the OOO window already covers the LLC latency; knob
kept, default off. Axis 2's premium over axis 1 is tr8x8 port-5 pressure
(~12 shuffles/element each way, shuffles are port-5-only on Ice Lake) plus the
compulsory cold read of `in`. Overall we sit ~2.1x over the pure-FMA port
floor; the binding costs are now shuffle-port and broadcast/issue shape, not
buffer sweeps — which is why change 3 bought only 0.8%.

### What did NOT work, with the number that killed it
- **Software prefetch (BST_PF)**: above — 1% loss on the exact case it targeted.
  gen_pow2's r1 "prefetch is pure issue overhead" extends to LLC-resident
  streams on this kernel; the ice "L3-regime prefetch win" did not transfer.
- **`sched-pressure` attribute on dif16/dit16 (BST_SCHED)**, gen_batchlane's
  SCHED15 trick: 5-rep interleaved A/B mins 19198/19357/19444/20697/20762 (off)
  vs 19359/19462/19536/20554/22065 (on) — pairwise a net loss, a wash at best.
  Knob kept for the SPR cross-arch round (their measurement flipped per host).
- **Scalar boundary block in the fused tr gather** (my own first cut): cost 8%
  at L=10, 7% at L=25. Numbers above. Masked loads are the right tool; do not
  hand a 4-element block to per-lane code when S is only 8.

### Borrowed this round, named
- **gen_twiddle**: tw_chirp, tw_fill_ct_int_colmajor, the create()-time ulp
  audit (their r1 record even names my file/line — easiest adoption on the
  panel). My old chirp was their measured 0.554-ulp row in the audit table.
- **gen_batchlane**: the sched-pressure function attribute (raced, not shipped
  on by default — see above).
- **gen_pow2 / ice records**: the same-window-pairs-only measurement discipline
  (every keep/kill decision above is a same-core alternating A/B), and the
  prefetch-mostly-loses prior (confirmed here, saved a knob from shipping).

### Harness notes (delta from r1)
tryout.sh's `$W`-before-assignment bug is FIXED (W now defined at line 36
before CH uses it), and the MKL reference leg runs. The remote map-check leg
still dies: line 48's `$( )` command substitution single-quotes '$W/c.bin', so
the remote shell sees literal $W → `--cin /c.bin`. Run check.py on the NODE
(numpy lives behind env.sh; wallaby's bare python3 has none):
`ssh $RES_NODE 'cd <gen> && source ../../env.sh && python3 check.py ... --map-check m --cin .../c.bin'`.

### What I would do next
1. **Axis 2's port-5 bill**: the 8x8 transposes cost ~2 ms of L=100's 19.9.
   Options: a lane-mapping where axis 2 reuses the contig path (needs an
   element-interleaved intermediate), or absorbing the transpose into the
   first/last stages' quarter stores (they already scatter to 4 streams).
2. **Broadcast pressure in conv_mid4**: bh as splat-8 tables (gen_twiddle's
   tw_splat8) converts 8 vbroadcastsd to 8 full loads — same port count, but
   frees the twiddle-load slots the fused stages now contend on. Measure.
3. **Own fft3d_chain**: in-place state volume + the campaign-standard
   rsqrt14 map ladder deletes one full volume stream per step (execute is
   forced out-of-place through the driver's pong). Worth ~5-10% on the chain
   cells; everyone else already owns theirs.
4. **bluestein_cost(L) for gen_planner** (carried from r1): ≈ 3L² rows × 2
   pruned M-FFTs; with the r2 pass structure the constant is now measured:
   ~0.66 ns per (row · M·log2 M / 8) on Ice Lake.
5. Re-race BST_SCHED and BST_PF on Sapphire Rapids in the next cross-arch
   round before assuming these nulls transfer.

## Round gen_r3

### What changed (impl/gen_bluestein.c, two structural changes + one raced size gate)

1. **Owned fft3d_chain** (my r2 next-step 3; last entry on the panel to own it).
   Every axis pass was already in-place-safe (each row depends only on itself),
   so the chain runs all m steps in place in final_out: step = axis2 (x0→out on
   step 1, then in place), axis1, axis0 with the graded map fused into the
   axis-0 scatter: after the output chirp the scatter adds c (loaded at the
   same offsets, deinterleaved with the same permutex2var pair) and applies
   `z/(1+|z|)` via the panel-standard ladder — rsqrt14 + 2 quadratic Newtons,
   rcp14 + 2 residual Newtons, no divider op (BST_MAP8, taken VERBATIM from
   gen_batchlane's map8, which is gen_pfa_small r1's ladder). Map placement is
   EAGER at the scatter (gen_batchlane/bl8's "lazy loses 24%" prior, not
   re-litigated). vs the driver fallback this deletes, per step, the whole
   separate map pass (read z + read c + write state through the driver's
   autovectorized ymm sqrt+div loop ≈ 2.4 ns/pt) and the ping-pong memcpy.
   The map runs only in the mask-scatter variants + generic scalar tail; the
   execute() paths are untouched (always_inline core + literal-NULL wrapper so
   the no-map specialization has no map branch).

2. **Size-gated map placement — the fused scatter LOSES past LLC reach.**
   The axis-0 scatter reads c at stride L^2 (16-double rows, no hardware
   prefetch), which is fine while the volume is cache-resident but
   latency-bound once it is not. Above BST_MAPFUSE_MAX_MIB (default 15 MiB of
   state) the chain instead runs axis 0 unfused and a SEPARATE sequential
   in-place map sweep (map_pass_seq: 3 streams, all prefetchable, same ladder,
   2+2 loads / 4 permutex2var / 2 stores per 8 points). Same-window race that
   set the gate: L=100 B=1 (30.5 MiB), 4/4 reps ordered separate < fused < r2
   — quiet reps 15.98/16.04 ms (separate) vs 18.86/19.25 (fused) vs
   19.60/19.68 (r2). L=50 (15.26 MiB): 1851/2118/1877 (separate) vs
   2460/2648/2125 (fused). L=25 (7.6 MiB): fused wins 180/183/182 vs
   182/195/192. L=31 (14.55 MiB): a wash. Gate at 15 MiB splits the suite
   exactly: 10..32 fused, 40/50/100 separate.

3. **Seam groups vectorized** (was the r2 generic-path leak): an 8-row group
   of axis 0/1 that straddles a div-block boundary went to the fully scalar
   generic path — at L=10 that is 60% of axis-1 groups, L=15 47%, L=27 26%,
   L=31 23%. New first_gather_seam / last_scatter_seam: lanes 0..k-1 live at
   q1 (contiguous), lanes k..7 at q2 (contiguous, next block); loads are two
   fault-suppressed masked loadu blended per zmm (4 masked loads + the same 2
   permutex2var per element vs contig's 2+2), stores mirror through 4
   group-constant masked storeu. Arithmetic (chirp, pruned first/last stage) is
   op-for-op the contig path's. Gated on B==1 && nv==8 && div>=8 (one boundary
   max); L<8 axes and tail groups still take the generic path. Run-2 loads are
   anchored at q2 - 2k so no address ever runs past the array end (the masked
   lanes make the q2-2k underhang and the +8 overhang architecturally safe —
   q2 always has a full block behind it, and faults are suppressed on masked
   lanes).

Dev knobs added: -DBST_NOSEAM, -DBST_NOCHAIN (decomposition builds for the
monitor), -DBST_MAPFUSE_MAX_MIB=n (0 = always separate, 4096 = always fused).

### Operation count (deltas per step-volume vs r2 through the driver)
Chain: removes 2V complex reads + 2V writes per step (map pass + ping-pong) and
V scalar-ish sqrt+div; adds per 8 outputs either (fused) 2 c-loads + 2
permutex2var + ~18 FMA-port ops + 2 seed ops, or (separate) one extra 3-stream
sweep with the same ladder. Seam: per seam group replaces ~2L scalar complex
multiplies + 2L scalar loads/stores per side with the vector pipeline (+2
masked loads/stores per element over contig).

### Measured on the node (a80n0, leased core, graded chain cells, min µs/xform,
### same-window alternating A/B vs the r2 binary; window best in parens)

| L | B | r2 | r3 | delta |    | L | B | r2 | r3 | delta |
|---|---|-----|-----|----|----|---|---|------|------|----|
| 10 | 64 | 15.73 | **13.21** (13.16) | -16% |  | 31 | 16 | 408.1 | **305.6** | -25% |
| 10 | 1  | 16.05 | **13.41** | -16% |  | 32 | 8 | 411.6 | **318.1** | -23% |
| 12 | 64 | 23.64 | **19.38** | -18% |  | 40 | 8 | 1238-1268 | **1134-1166** | -8.5% (5/5 quiet pairs) |
| 15 | 32 | 42.62 | **33.74** | -21% |  | 50 | 4 | 2284 | **1851** | -19% |
| 20 | 32 | 128.7 | **103.7** | -19% |  | 100 | 1 | 19322-19621 | **15866-16036** | -18% |
| 25 | 16 | 231.0 | **179.6** | -22% |  | 27 | 16 | 310.3 | **220.0** | -29% |

(The L=40 delta is smaller because 8 | 40: no axis-1 seams — that cell is the
pure chain-ownership effect in its separate-map regime.)

### Gates (shipped default build, on the node, all by-hand check.py)
Generality sweep single call B=1: L ∈ {2,3,4,5,7,9,11,13,16,17,23,33,47,63,64,
65,96,101,127,128} ALL PASS ≤ 8.8e-16 (tol 1e-12). Two-step fused-chain gate
m=2: L=10 1.21e-15, L=31 2.57e-15, L=100 3.82e-15 (tol 3e-14 — the ladder is
exact-tier, ~10-25x margin). Full graded chains: L=10 m=1000 2.30e-13 (anchor
1.08e-13), L=10 B=1 1.70e-13 (anchor 1.78e-13 — BELOW the honest anchor),
L=25 4.83e-14 (2.80e-14), L=31 4.61e-14 (2.31e-14), L=50 4.82e-14 (2.92e-14),
L=100 5.10e-14 (2.42e-14) — all ≤ 2.1x honest drift, tol 1e-10. Chain and
single outputs bit-identical across independent runs. Scalar -march=x86-64
build: singles PASS at {3,10,25,31,100}, owned chain PASS (the generic path
maps in its scalar scatter). create() still ~0 s.

### What did NOT work / was decided by race, with the number that killed it
- **Fused scatter map at DRAM-scale volumes**: the headline negative. L=100:
  fused 18.86/19.25 ms vs separate sweep 15.98/16.04 in the same windows;
  L=50: 2460/2648/2125 vs 1851/2118/1877. The strided c reads (stride L^2, no
  prefetcher coverage) put a load-latency chain in front of every mapped
  store. Fusion is a CACHE-RESIDENT technique; my r2 finding "the strided
  axes are not latency-bound" stops holding once a second cold stream joins
  the pass. Gate, don't choose.
- **bh splat8 tables in conv_mid4** (my r2 next-step 2): declined without a
  run — vbroadcastsd from memory and a full zmm load are both one load-port
  uop on Ice Lake (gen_pow2's r2 record makes the same count), so there is no
  port to free; splatting octuples the bh table footprint in a pass that is
  L1-resident. Nothing to win.
- This round's node windows were strongly BIMODAL (the gen_batchlane r2
  cross-window note, worse this week): r2 itself read 1472 and 2001 µs at
  L=40 in adjacent windows, sd < 0.1% inside each. Every keep/kill above used
  alternating same-core pairs and required consistent ORDERING across >= 3
  reps, not absolute numbers; the r1/r2 habit of quoting one window's min
  would have called L=40 a regression (one window read r3 +4.6%; five clean
  pairs read -8.5%).

### Borrowed this round, named
- **gen_batchlane**: BST_MAP8 is their map8 ladder verbatim (rsqrt14 + 2
  quadratic Newtons, rcp14 + 2 residual Newtons, 1e-300 guard), plus the
  eager-at-the-store map placement and the lazy-map-loses prior.
- **gen_pfa_small r1**: the divider-free reciprocal (via gen_batchlane's r2
  adoption of it).
- **gen_pow2 r1/r2**: the weak-symbol fft3d_chain ownership pattern
  (everyone's, but their record is the cleanest description of what the
  driver fallback costs), the broadcast-vs-load port count that killed the
  splat8 idea, and the same-window pairs discipline.
- **gen_twiddle r2**: their record explicitly asked adopters for honest
  nulls; the splat8 decline above is one.

### What I would do next (gen_r4)
1. **Axis-2's port-5 bill stands** (carried from r2): tr8x8 is ~2 ms of
   L=100's 15.9. Absorbing the transpose into the first/last stages' quarter
   stores is still the concrete plan.
2. **Lazy map into the NEXT step's axis-2 gather for the separate-map
   regime**: axis-2 group reads are 8 contiguous rows, so c would stream
   SEQUENTIALLY there — it could delete the separate sweep's extra volume
   read+write without the strided-c trap that killed scatter fusion. Needs a
   final map-only pass after step m and care with the critical path
   (batchlane's lazy-loses prior was L2-resident custody; this regime is
   DRAM-bound, different trade).
3. **Tail groups (nv < 8) through a masked contig path** — only matters for
   odd L at B < 8; round 6 could draw one.
4. **bluestein_cost(L) for gen_planner** (carried): ~0.66 ns per
   (row · M·log2M / 8) on Ice Lake, now plus ~0.15 ns/pt/step of chain map.
5. Re-race BST_MAPFUSE_MAX_MIB, BST_SCHED, BST_PF on SPR/CLX in the
   cross-arch round; the 15 MiB gate is an Ice-Lake-LLC number.
