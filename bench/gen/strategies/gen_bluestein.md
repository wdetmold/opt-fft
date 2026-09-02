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

## Round gen_r4

### The find that paid for the round: my own r3 gate was shipped wrong
The r3 record says the map-regime gate "splits the suite exactly: 10..32 fused,
40/50/100 separate" — but the shipped code gated on STATE bytes alone
(`nrows*L*16`), while the intended quantity (and the one the r3 race actually
measured) is state + c COMBINED, since both stream through the fused axis-0
scatter. Result: L=40 and L=50 sailed under the 15 MiB threshold and ran the
FUSED map — the regime my own r3 race had measured 15-30% slower. Verified on
the node before touching anything: L=50 B=4 default 1986.8 µs vs
`-DBST_MAPFUSE_MAX_MIB=0` (forced separate) 1822.2 in adjacent windows; L=40
forced-fused 1381.9 vs blocked-separate 1086.6/1146.3 same-window. Fix: gate
on `nrows*L*32`. Same threshold value; L=31 (14.55 MiB combined) stays fused,
exactly the r3-raced split. Lesson recorded: when a race sets a policy gate,
TEST THE GATE (one forced-knob run per side), not just the policy code paths.

### What changed (impl/gen_bluestein.c, two changes + one raced-off knob)

1. **Gate fix** above (`*16` → `*32` in fft3d_chain's fuse computation).
2. **k-plane-blocked custody for the separate-map regime** (idea from
   gen_layout r3's 4-plane circular window — named borrow; my r1 "plane-fused
   axis-2+1" failure was the same idea crippled by per-plane lane tails, which
   the block size kills). Axis order per step becomes 0 first (global,
   strided), then per block of k planes — k = 8/gcd(L,8), so blocks are k*L
   rows ≡ 0 (mod 8) and the 8-row group decomposition is IDENTICAL to the
   unblocked passes (no new tails, no seam changes; per-row arithmetic
   bit-identical, only inter-row order moves) — run axis 2, axis 1, and the
   sequential map sweep while the block (k·L²·16 B ≤ 320 KiB at L ≤ 100) is
   L2-hot. Deletes the axis-1 full-volume round trip and the map pass's state
   read+write; per-step traffic at L=100 drops ~144 → ~96 MB of touched
   volume-passes. The map stays a SEPARATE sequential sweep on purpose:
   gen_pfa_large r3's ipm family (map fused into next step's loads) lost
   11-15% on this node — ladder uops land on port 5, which my tr8x8-heavy
   axis-2 gather saturates worse than their p1 — so that idea was adopted as
   a NULL, not rediscovered. Blocks with a partial tail (B·L not divisible by
   k) fall through correctly (L=101 B=1: 12 full 8-plane blocks + 5 planes +
   1 generic-path tail row, gated below). Knob: -DBST_NOBLOCK reverts to the
   r3 global order (the raced control).

### Raced off / re-raced, with the numbers
- **Blocked vs unblocked separate (the attribution race)**: L=100 B=1 pairs
  BLK 15803/15759 vs SEP-global 16156/16711 µs (medians 16.0/16.4 vs
  18.6/20.0 ms — blocking also tames busy-window medians); quiet-window best
  15545-15582 (sd 0.01-0.02%). L=50: raw split 1895/1846 then 1809/1849 —
  MKL-normalized BLK wins both pairs. The win is smaller than the traffic
  model because the 16 MB state already fits the 24 MB LLC: the deleted round
  trips were LLC-level, not DRAM. Blocking ships.
- **-DBST_BLKFUSE (blocked axes 2+1 in the FUSED regime too)**: L=25
  179.7 vs 188.1 (−4.5% raw, a wash MKL-normalized), L=31 354.6 vs 321.9
  (+10% — clear loss). Default OFF; knob kept for the cross-arch race (CLX's
  1 MB L2 changes the residency story).
- **BST_PF re-race under the new order** (axis 0 now reads the cold volume,
  so r2's null needed rechecking): first pair in a dirty window read PF as a
  1.8 ms WIN (16581 vs 18423, MKL +9% elevated) — three control-first pairs
  later: DEF 15545/15700/15706 vs PF 16723/17535/16179, 3/3 DEF. r2's null
  stands, and the dirty-window head-fake is exactly why single pairs don't
  count.

### Measured on the node (a80n0, leased core, graded chain cells, min µs/xform)
| L | B | r3 board | r4 | delta |
|---|---|-----|-----|----|
| 40 | 8 | 1150.4 | **1086.6-1101** (quiet sd 0.03%) | −4.5% board, −21% vs same-window forced-fused |
| 50 | 4 | 1931.9 | **1790.8-1822** | −6.5% |
| 100 | 1 | 15741.8 | **15545-15582** (quiet sd 0.01%) | −1.2% |
| 10 | 64 | 13.31 | 13.72 (busy window; fused path untouched) | parity |
| 12 | 64 | 19.36 | 19.49 (busy window) | parity |
| 25 | 16 | 179.0 | 179.3-188.1 | parity |
| 31 | 16 | 301.3 | 321.9-345.6 (busy windows this session) | parity-noisy |

Fused-regime sizes (10..32) take a byte-identical code path to r3; their
deltas above are window state, not code.

### Gates (shipped default build, on the node, check.py by hand)
Generality sweep singles B=1: L ∈ {2,3,5,7,9,13,16,17,23,33,47,63,64,65,96,
101,127,128} ALL PASS ≤ 8.8e-16 (tol 1e-12). Two-step fused-chain gate m=2:
L=40 2.60e-15, L=50 3.18e-15, L=100 3.63e-15, L=25 2.06e-15, L=31 2.57e-15,
L=101 5.18e-15 (tol 3e-14). Full graded chains: L=40 4.29e-14 (anchor
2.61e-14), L=50 6.84e-14 (2.92e-14), L=100 3.03e-14 (2.42e-14 — 1.25x honest,
was 2.1x in r3; the axis reorder happens to round kinder there), L=25
4.83e-14, L=31 4.61e-14 — all ≤ 2.4x honest, tol 1e-10. Chain outputs
bit-identical across independent runs at every size incl. L=101's partial
block. Scalar -march=x86-64 build: singles PASS at {10,50,100}, m=2 chain
PASS (2.96e-15). create() still ~0 s.

### Operation count (delta vs r3)
Arithmetic unchanged (same kernels, same ladder, same group decomposition).
Separate regime per step: axis order 2,1,0+sweep → 0,2,1+sweep-per-block;
removes one full-volume read+write (axis 1's round trip to LLC/DRAM) and the
map sweep's state read+write (block is L2-hot when mapped); c is still read
once per step, sequentially.

### Borrowed this round, named
- **gen_layout r3**: the plane-window custody idea (their −15.6% at L=100
  demo); my contribution is the k-plane block size that makes the group
  decomposition provably identical to the unblocked pass.
- **gen_pfa_large r3**: the ipm verdict ("map placement axis is exhausted;
  a separate sequential map pass is optimal") adopted as a standing null —
  this round's plan A had been map-into-axis-2-gather, and their record
  killed it before I burned windows on it. That is the cumulative round
  working as intended.
- **gen_batchlane r2 / gen_pow2 r3**: control-first same-window pairs with
  ≥3 reps before any keep/kill (the BST_PF head-fake above is the poster
  child).

### What I would do next (gen_r5)
1. **Axis-2's port-5 bill, third carry**: tr8x8 is ~12 shuffles per element
   vector each way and shuffles steal from the second FMA pipe (port 5 on
   ICL). Absorbing the transpose into the pruned first/last stages' quarter
   stores is still the concrete unexplored shape; broadcast-load rebuilds
   were counted out on load-port math (128 uops/block vs 48 shuffles).
2. **L=31 sits 3% under the (fixed) fuse gate** — race it in both regimes
   once in a quiet window; if separate+blocked wins, drop the threshold a
   notch rather than special-casing.
3. **Tail groups (nv < 8) through a masked contig path** (carried) — only a
   round-6 odd-L-at-B=1 concern; the generic path is correct, just scalar.
4. **bluestein_cost(L) for gen_planner** (carried): ~0.66 ns per
   (row · M·log2M / 8) + ~0.15 ns/pt/step of chain map on ICL.
5. Cross-arch: re-race BST_MAPFUSE_MAX_MIB (an ICL-LLC number), BST_BLKFUSE
   (CLX's 1 MB L2), BST_NOBLOCK, BST_PF, BST_SCHED on CLX/SPR when XARCH.md
   lands.

## Round gen_r5

### What changed (impl/gen_bluestein.c, one structural change + one gate move)

1. **Custody-ordered c + map fused into the blocked axis-1 scatter** (named
   borrow: gen_pow2 gen_r4's GP2_CT — "store the chain operand in the LAST
   pass's consumption order"; their record explicitly asked peers to check
   whether c's layout matches the last pass's walk order, and mine did not).
   In the separate-map regime the r3 race had banned scatter-fused mapping
   because the scatter's c reads are strided (16L-B steps, one 128-B touch
   per line, ~L streams — nothing the L2 streamer tracks) and cold. The fix
   is not prefetch games but layout: `build_ccust` runs once per
   fft3d_chain call and copies c into chunk-per-group order — chunk g =
   axis-1 group r0 = 8g holds, for k = 0..L-1, [8 re][8 im] of c at the
   group's row offsets. The blocked axis-1 scatter (the block's last touch)
   then applies the map reading c as two aligned interleaved +128 B streams
   (k = j and k = j+S advance together), which the hardware prefetcher
   covers, and the c deinterleave permutes disappear (custody is already
   split re/im). map_pass_seq and its block read+write+re/de-interleave are
   deleted from the schedule. Per-element ladder identical (BST_MAP8 in
   contig/seam paths, sqrt/div in the generic tail as before) ⇒ **chain
   outputs bit-identical to gen_r4 at every graded size** (cmp-verified at
   40/50/100/31). The custody buffer (= state bytes; 16 MB at L=100) hangs
   off the plan, allocated lazily; alloc failure falls back to the r4 sweep.
   Collector cost is one strided sweep of c per chain CALL (not per step):
   vectorized for contig groups (2 loadu + 2 permutex2var per k), scalar for
   seam/tail groups. Knobs: -DBST_NOCFUSE = r4 sweep control,
   -DBST_CSTRIDED = fused-but-strided attribution arm.
2. **BST_MAPFUSE_MAX_MIB default 15 → 14** (my r4 next-step 2): with the
   custody-fused map the blocked regime now beats the LLC-fused regime at
   L=31 B=16 (14.55 MiB combined), 5/6 same-core pairs, mean −2.9%, best
   288.6 vs 302.2 µs. 27 (9.6 MiB) and 32 (8.0 MiB) stay fused — untouched
   and re-gated. L=31 B=1 (0.95 MiB) stays fused, as it should.

### Struck without a window (carried item 1 closed by arithmetic)
**Axis-2 transpose absorption into the pruned end stages**: counted OUT.
The tr path costs 2×tr8x8 = 48 port-5 shuffles per 8 lanes × 4 complex
columns each way = 1.5 shuffles/complex/side — and transposing interleaved
data yields split re/im for free. The absorption alternative (butterflies in
row-major registers with vector twiddles) needs an interleaved cmul = 1
vpermilpd + 2 FMA, i.e. ≥ 3 port-5 shuffles/complex in the butterflies
ALONE (≈ 0.75 cmul/element/stage × 4 stages), before any end-stage shuffle
work — strictly more port-5 traffic than the transpose it replaces.
gen_layout r4's gl_tr8x8_c2i (48 shuffles per 16 output zmm) matches, not
beats, my existing 2×tr8x8 = 48-for-16, so there is nothing to adopt there
either. The axis-2 premium is the compulsory cold read plus ~0.3-0.5 ms of
port-5 at L=100, not 2 ms of removable shuffles. Fourth carry: closed.

### Measured on the node (a80n0, ONE held lease, core 3, tight alternating
### same-core pairs — gen_batchlane r4's protocol; this session's windows
### swung ±8%, MKL same-window sd 0.55%)

Custody fusion vs r4 sweep (blocked regime, pairwise):
- L=31 B=16 (forced separate, attribution): cust 4/6 pairs, mean −5%;
  bests 288.6 vs 303.8.
- L=40 B=8: new 8/13 pairs over two sets, mean ≈ −0.5%; bests 1062.4 vs 1073.2.
- L=50 B=4: a WASH (7/13 pairs); bests 1772.2 vs 1771.4. The strided arm
  (BST_CSTRIDED) reads 1817-2139 — worst of the three, confirming the r3
  diagnosis: fusion without custody layout is still wrong past LLC.
- L=100 B=1: 6/10 pairs, mean ≈ −0.4% — a wash-to-lean in a dirty window;
  bests 15759.9 (new) vs 16247.0 (r4) across the session.
  gen_pfa_large r4's "ipp at 40/50 is a wash — the deleted map pass is
  L3-resident traffic" replays here almost verbatim; the deleted work is
  L2-resident, so the win is uop-count-sized, not traffic-sized.

Regime race that moved the gate: L=31 B=16 fused vs separate+custody:
sep 5/6 pairs (−7.6/−4.1/−5.0/−0.4/−1.7/+5.2%).

Ship binary, end-of-session quiet reads (min µs/xform): **L=31 289.0**
(r4 board 299.4), **L=100 15479** (r4 board 15742; quiet r4 windows
15545), L=10 B=64 13.34 (board 13.28 — fused path untouched, parity),
L=40 1062-1088, L=50 1772-1834. MKL 2022 same window: L=31 854.3.

### Gates (shipped default build, on the node, check.py by hand)
Singles B=1: L ∈ {2,3,5,7,9,13,16,17,23,33,47,63,64,65,96,127,128} ALL
PASS ≤ 8.6e-16 (tol 1e-12). Two-step m=2: L=31 2.55e-15, L=40 2.60e-15,
L=50 3.18e-15, L=100 3.63e-15, L=101 5.17e-15 (tol 3e-14). Graded chains:
L=25 4.83e-14, L=27 4.73e-14, L=31 4.17e-14 (blocked now), L=32 4.50e-14,
L=40 4.29e-14, L=50 6.84e-14, L=100 3.03e-14, L=10 m=1000 2.30e-13 — all
≤ 2.4x honest anchors, tol 1e-10. Chains bit-repeatable; 40/50/100 chain
outputs bit-identical to the r4 binary. Scalar -march=x86-64 build:
singles + m=2 chains PASS at {10,50,100}. create() still ~0 s; the custody
buffer is chain-lazy so plan budget is untouched.
One recorded non-identity: at L=101 B=1 (nrows % 8 ≠ 0) the new arm's tail
group maps through the generic scalar sqrt/div while the r4 sweep's ladder
covered those elements — a rounding-tier (≤1e-15) difference that step 2's
FFT then spreads volume-wide. Gates pass with 6x margin; repeatable; only
possible at sizes where 8 ∤ B·L².

### What did NOT work / was decided by race, with the number
- **BST_CSTRIDED (fuse the map into axis-1 but read c strided)**: worst arm
  at L=50 (1817-2139 vs custody's 1772-1834). Layout, not placement, was
  always the blocker — this arm exists to prove exactly that and should not
  be resurrected.
- **Custody fusion is NOT a traffic win at 40/50/100** (see wash numbers).
  It ships because it never loses, deletes a pass and its permutes, is the
  thing that flips the L=31 regime, and (pfa_large r4's observation) a
  smaller per-step footprint is busy-window contention armor.
- **NT stores (gen_layout r4's −19% at L=100)**: declined without a run,
  on structure: their win killed RFO reads on three cold full-volume store
  streams; my blocked custody has no such stream — axis-0 stores hit
  just-read lines (in place), axis-2/1 stores hit L2-hot block lines, and
  the only cold-written buffer (step 1's dst) is touched once per chain.
  Nothing for NT to delete; noted for CLX/SPR where residency changes.

### Borrowed this round, named
- **gen_pow2 gen_r4**: GP2_CT — the custody-oriented chain operand — is the
  round's entire structural idea; their "peers should check whether c's
  layout matches the LAST pass's walk order" line is what this round did.
- **gen_batchlane gen_r4**: the held-lease same-core alternating protocol
  (tryout.sh leases a fresh core per invocation; every verdict above is
  same-core adjacent-pair). Their SCHED15 reversal is why I trusted no
  cross-invocation number this session.
- **gen_pfa_large gen_r4**: the "deleted L2/L3-resident passes are worth
  ~nothing" boundary (their ipp-at-40/50 wash), used to size expectations
  honestly instead of over-claiming the wash cells; and the ±15%
  window-drift warning that shaped the pairing discipline.
- **gen_layout gen_r4**: the NT-store recipe, adopted as a reasoned decline
  (structure above); their gl_tr8x8_c2i shuffle count, used to close my
  axis-2 carry by arithmetic.

### Operation count (delta vs r4, per chain step in the blocked regime)
FFT arithmetic unchanged. Deleted: the map sweep's block state read+write
(L2), its 4 de/re-interleave permutes and 2 c-deinterleave permutes per 8
points, and 2 stores per 8 points. Moved: the ~21-op ladder from the sweep
into the axis-1 scatter. Added, once per chain CALL: the custody collector
(one strided read sweep of c + sequential write of state-sized buffer,
amortized over m = 64-1000 steps).

### What I would do next (gen_r6 — the surprise round)
1. **Round-6 posture first**: any L in 14..127 must plan and pass. The
   generality sweep covers the odd/even log2 pipelines, masked boundaries,
   partial blocks (L=101) and the M=4 path; keep that sweep the first thing
   run on any new binary.
2. **Tail groups (nv < 8) through a masked contig path** (third carry):
   now slightly more interesting since the L=101 non-identity lives there,
   but still ≤ 1/1275 of groups — cosmetic unless a surprise size lands on
   B=1 odd L AND the cell is close.
3. **bluestein_cost(L) for gen_planner** (fourth carry): ~0.66 ns per
   (row · M·log2M / 8) + ~0.15 ns/pt/step chain map on ICL, unchanged.
4. **Cross-arch (XARCH.md due after r5)**: re-race BST_MAPFUSE_MAX_MIB (an
   ICL-LLC number, now 14), BST_NOCFUSE (CLX's 1 MB L2 shrinks the block
   custody the fusion rides on), BST_BLKFUSE, BST_PF, BST_SCHED. The
   custody collector's one-time cost also rescales with m on any host.
5. If a real lever is wanted at L=100: the remaining ~2.1x over the port
   floor is issue shape in the conv stages themselves (broadcast-heavy
   radix-4 bodies); literature 11's 2,8-split-radix flap-optimal chains are
   the only untried arithmetic-count idea left in this class.

## Round gen_r6

### What changed (impl/gen_bluestein.c, one structural change + one raced-off knob)

1. **Non-power-of-two convolution lengths: M = min{2^k, 3*2^k, 5*2^k} >= 2L-1,
   k >= 4** (grid 48, 64, 80, 96, 128, 160, 192, 256, ...). My r5 next-step 5
   called the remaining lever "arithmetic count in the conv stages"; this is
   that lever, and it is kin to literature 11 Tier-1's flap-count
   factorization ranking (rank chains by real cost, not by radix
   convenience). The radix-4 DIF/DIT chain, the no-bit-reversal trick, and
   every pruned/fused end stage carry over unchanged because they never
   assumed pow2 — only 4 | len per stage and S = M/4 divisible by 4 for the
   axis-2 transpose path (hence k >= 4: M = 24/12 stay excluded). The chain
   now ends at a twiddle-free tail of 2, 3, 5, 6 (= PFA 2x3), or 10
   (= PFA 2x5) consecutive positions; since block order was never named,
   ANY fixed 6/10-point output permutation is legal as long as create()'s
   bh forward uses the same one — PFA needs no twiddles at all. New code:
   plain-C dft{3,5,6,10}_{fwd,inv}_stage (create + scalar builds) and fused
   AVX-512 conv_mid{3,5,6,10} (tail DFT + pointwise bh + exact inverse, one
   pass, mirroring conv_mid4/2). Twiddle fills/audits unchanged —
   tw_fill_ct_int_colmajor takes arbitrary N (checked before writing code).
   Affected sizes: L 17-24 M 64->48 (-25%), L 33-40 128->80 (-37.5%!),
   L 41-48 128->96, L 65-80 256->160 (-37.5%), L 81-96 256->192 — 56 of the
   114 candidate surprise sizes in 14..127, plus graded 20 and 40. All
   other L take byte-identical code paths.
   **This corrects my own r1 note** "radix-3 sizes like 192 never beat the
   pow2 choice — checked arithmetic": that arithmetic was wrong; measured,
   the small-M choices win everywhere they apply (below).
2. **conv_mid12 (raced OFF, knob -DBST_MID12)**: fusing dif4(12) + mid3 +
   dit4(12) into one register pass (5 -> 3 passes at M=48) was a
   wash-to-loss: control-first same-core pairs at L=20 B=32 read 84.53/
   84.89/84.53 (unfused) vs 85.31/84.76/87.65 (fused). The 6 KiB M=48
   buffer is L1-resident, so the deleted passes were free and the
   24-live-zmm tile spills — gen_pfa_large r4's "deleted cache-resident
   passes are worth ~nothing" again, now at L1. Code kept for the CLX/SPR
   re-race (1 MB L2 machines may disagree).

### Operation count (per row, changed sizes)
M=48: first(48)+dif4(12)+mid3+dit4(12)+last(48); M=80: first+dif4(20)+mid5+
dit4(20)+last; M=96: ...mid6...; M=160: ...dif4(40)+mid10...; M=192:
first+dif16(48,12)+mid3+dit16+last. Same 5-pass shape as the pow2 chains
(M=48 has no dif16 layer, same count), 25-37.5% less data per pass;
DFT-3/6 are cheaper per point than radix-4, DFT-5/10 about par. Gather/
scatter/chirp work (O(L) per row) unchanged.

### Measured on the node (a80n0, held lease, same-core control-first
### alternating pairs vs the r5 binary; graded chain cells, min us/xform)

| cell | r5 same-window | r6 | delta |
|---|---|---|---|
| L=20 B=32 m=256 | 104.3 / 105.5 / 103.9 | 84.3 / 83.6 / 82.9 | **-20%** (3/3) |
| L=40 B=8 m=128 | 1144 / 1223 / 1200 | 732 / 700 / 700 (M=96 arm) | **-39%** (3/3) |
| L=40, M=96 vs M=80 arms | 734.6 / 740.5 / 704.1 | 669.0 / 665.5 / 693.6 | **-6.5%** (3/3) |

Ship binary (tryout, quiet reads): L=20 B=32 **84.5** us (r5 board 103.4),
L=40 B=8 **682** (board 1063, quiet A/B best 665), L=20 B=1 90.8,
L=40 B=1 620-694. Parity controls (paths untouched): L=10 B=64 13.47
(board 13.28), L=25 B=16 181.8 (board 179.9). MKL same window: L=20 58.5,
L=40 413.

### Gates (shipped default build, on the node, check.py by hand)
Singles B=1: L in {2..9, 11, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24,
33, 34, 35(B=4), 37, 40, 43, 47, 48, 63, 64, 65, 66, 71, 80, 95, 96, 97,
101, 127, 128} ALL PASS <= 8.9e-16 (tol 1e-12) — covers every tail type
(2, 3, 5, 6, 10), both PFA parities, all the new M values incl. the
L = M/2 edges (24, 48, 96), boundary-block jful cases, and the giants.
Two-step m=2: L=17 1.72e-15, L=20 1.95e-15, L=40 2.82e-15, L=80 3.68e-15,
L=96 3.34e-15 (tol 3e-14, >= 8x margin). Graded chains: L=20 m=256
5.88e-14 (anchor 2.84e-14), L=24 m=200 3.92e-14 (1.83e-14), L=40 m=128
4.58e-14 (2.61e-14), L=48 m=100 4.01e-14 (2.03e-14) — all <= 2.2x honest,
tol 1e-10. Repeatable (cmp-identical across runs). Scalar -march=x86-64:
singles PASS at L in {20 B=4, 40 B=2/4, 71, 96} (exercises the plain-C
DFT-3/5/6/10 stages end to end). create() still ~0 s.

### What did NOT work / was declined, with the number or the reason
- **conv_mid12**: above — 2/3 pairs lost, worst +3.7%. L1-resident passes
  are already free; do not fuse for pass count below L2 residency.
- **M = 24 / 12 (k < 4)**: would give L=9..12 M 32->24 with a 3-pass chain,
  but S = 6 (resp. 3) breaks the S % 4 == 0 contract of the axis-2
  transpose gather/scatter (blocks of 4 j's); supporting it needs 2-wide
  masked tail blocks in first_gather_tr AND last_scatter_tr in all three
  leg regions. Declined on risk/size: those cells are 13-19 us and only
  graded, never surprise (14..127). Next-round candidate if anyone needs it.
- **7*2^k (M = 112 for L=49..56, vs 128)**: needs a Winograd/Rader 7-point
  tail — real code for a 12.5% M cut on one octave slice that contains no
  graded size. Declined this round.

### Borrowed this round, named
- **literature 11 Tier-1 (flap-count factorization ranking)**: the
  cost-not-convenience framing that finally displaced next_pow2; cited as
  the impetus, the implementation is this entry's own machinery.
- **gen_batchlane r4 / gen_pow2 r3**: the held-lease same-core
  control-first pairing protocol, used for every keep/kill above (the
  conv_mid12 kill and the M=80-vs-96 attribution are pure products of it).

### What I would do next (gen_r7 / endgame)
1. **bluestein_cost(L) for gen_planner** (fifth carry, now size-dependent):
   cost ~ 3L^2 rows x 0.66 ns per (row * M(L)*log2 M(L) / 8) with
   M(L) = min{2^k, 3*2^k>=48, 5*2^k>=80 : >= 2L-1} — the grid matters now
   (L=40's M halved). The planner should stop assuming next_pow2 for me.
2. **M=24 via 2-wide tr tail blocks** if L=9..12 ever matters beyond the
   two graded cells.
3. **Cross-arch re-races** (carried): BST_MAPFUSE_MAX_MIB (ICL-LLC number),
   BST_NOCFUSE, BST_BLKFUSE, BST_PF, BST_SCHED, and now BST_MID12 on
   CLX/SPR when XARCH.md lands.
4. The remaining L=100-class lever is unchanged from r5: issue shape in the
   broadcast-heavy radix-4 bodies (2,8-split-radix chains, lit 11) — M=256
   got no help this round since 2L-1=199 > 192.

## Round gen_r7

### The find that paid for the round: my own r6 decline was factually wrong
gen_r6 declined the 7*2^k convolution grid with "M = 112 for L=49..56 ...
one octave slice that contains no graded size."  cases.txt says otherwise:
**L=50 (50:4:128) is inside that slice**, and the next 7-slice up, M = 224
for L=97..112, contains **L=100** — the campaign's weakest big cell.  Two
graded cells, both −12.5% convolution data, killed in r6 by a slice-memory
error.  Lesson recorded: when declining an idea by coverage argument, check
the coverage against cases.txt, not against memory.

### What changed (impl/gen_bluestein.c, one structural change + one knob)

1. **7*2^k convolution lengths: M grid extended to {2^k, 3*2^k, 5*2^k,
   7*2^k}, k >= 4** (grid now 48, 64, 80, 96, 112, 128, 160, 192, 224,
   256, ...).  Straight continuation of the r6 machinery: the radix-4
   DIF/DIT chain, no-bit-reversal, and every pruned/fused end stage carry
   over untouched; the chain now may end at a twiddle-free DFT-7 or
   PFA(2x7) DFT-14 tail.  New code: BST_DFT7_LANE (the BST_DFT5_LANE
   symmetric t/d form extended to three 3-term dot products; constants
   correctly rounded from 60-digit decimal), plain-C dft{7,14}_{fwd,inv}
   stages (create()'s bh forward + scalar builds), vector BST_DFT7V, and
   fused AVX-512 conv_mid7 / conv_mid14 (tail DFT + pointwise bh + exact
   inverse, one pass, mirroring conv_mid5/10).  PFA-14 slot order
   (u_k+v_k, u_k-v_k) is never named anywhere — bh is computed by the same
   forward, so consistency is structural, as always.  M=112/224 keep
   S = M/4 4-aligned (28, 56), so the transpose gather/scatter contract
   holds with zero changes.  Affected sizes: L 49-56 M 128->112, L 97-112
   M 256->224 (also 193-224 -> 448 etc., unscored).  Every other L takes a
   byte-identical code path.
2. **-DBST_NO7** attribution knob (create()-side only): restores the r6
   grid; used as the control arm below.

### Operation count (changed sizes)
M=112: first(112) + dif4(28) + mid7 + dit4(28) + last(112).  M=224:
first(224) + dif4(56) + mid14 + dit4(56) + last(224).  Same 5-pass shape
as the M=128/256 chains they replace (those spent dif16/dit16 fused pairs
to get to 5), 12.5% less data per pass; the DFT-7 tail is ~9 vector ops/pt
vs radix-4's ~7/pt, on 12.5% fewer points, and mid7/mid14 replace mid2/mid4
+ the dif16 spill traffic.  Gather/scatter/chirp work (O(L) per row)
unchanged.

### Measured on the node (a80n0, ONE held lease, core 4, control-first
### same-core alternating pairs, new vs -DBST_NO7 control; graded cells)

| cell | control (r6 grid) | new (M7 grid) | delta |
|---|---|---|---|
| L=50 B=4 m=128 | 1797.9 / 1795.3 / 1776.4 | 1394.9 / 1393.6 / 1556.5 | **−21%** (3/3) |
| L=100 B=1 m=64 | 15435 / 15469 / 15589 | 14364.9 / 14352.1 / 14353.0 | **−7.0%** (3/3) |
| L=40 B=8 (parity) | 619.3 / 621.3 | 619.2 / 618.2 | 0 (paths untouched) |
| L=25 B=16 (parity) | 182.7 / 180.2 | 181.0 / 182.7 | 0 |

Fresh-core tryout reads: L=50 B=4 **1406.9** us (r6 board 1771.7; MKL same
window 946.9 — we go 1.87x -> 1.49x MKL), L=100 B=1 **14379.9** (board
15479-15742; MKL 7755).  The L=50 delta beats the −12.5% data model; the
extra comes from replacing M=128's spill-heavy dif16/dit16 pair with the
lean dif4(28)/mid7 chain.  L=100 tracks the model (gather/chirp O(L) work
dominates more there).

### Gates (shipped default build, on the node, check.py by hand)
Singles B=1: L in {2, 10, 20, 40, 48, 49, 51, 53, 56, 57, 64, 96, 97, 101,
105, 112, 127, 128} ALL PASS <= 1.0e-15 (tol 1e-12) — covers both new
tails (7 at M=112, 14 at M=224), the L = M/2 edge (56, 112), the slice
boundaries (48/57, 96/97), and the untouched-parity checks.  Local naive-
reference sweep additionally covered 49-56/97-112 interiors and B in
{2,3,4} seam cases.  Two-step m=2: L=50 3.33e-15, L=56 3.25e-15, L=100
4.11e-15, L=112 3.47e-15 (tol 3e-14, >= 7x margin).  Graded chains:
L=50 m=128 6.13e-14 (anchor 2.92e-14, 2.1x honest — r6 read 6.84e-14 on
the same cell), L=100 m=64 3.56e-14 (anchor 2.42e-14, 1.47x).  Chain
repeatable (cmp-identical).  Scalar -march=x86-64 build: singles + m=2
chains PASS at {50 B=4, 100, 112} (exercises plain-C DFT-7/14 end to end).
create() still ~0 s.

### What did NOT work / was declined, with the reason
- **15*2^k (M=240)**: S=60 is 4-aligned and DFT-15 = PFA(3x5) is
  twiddle-free, but the slice it improves (L=113..120, 256->240, −6.25%)
  contains no graded size — declined on the same coverage test the r6
  error taught me to actually run.
- **13*2^k (M=208)**: would cut L=97..104 another 7% below 224 (S=52 is
  4-aligned), but needs a direct DFT-13 (6x6 dot-product module, 26 live
  zmm in the fused middle) for a second-order gain on one cell.  Queued,
  not spent; measure tail cost vs the 7% before believing it.
- **M=24 for L=9..12** (carried from r6, now with an accurate scope
  estimate): S=6 needs 2-wide variants of ALL THREE tr block types —
  at L=11 (jful=5) even the 4-wide masked BOUNDARY block overruns S=6, so
  it is not just a single-leg tail fix.  Surgery on the hottest, most
  correctness-critical functions for two cells the PFA owners win anyway.
  Declined again.
- tryout.sh's remote map-check leg still dies on the '$W' quoting bug
  (--cin /c.bin); all chain gates above were run by hand on the node, as
  documented in my r2 notes.

### Borrowed this round, named
- **literature 11 Tier-1 (flap-count factorization ranking)**: still the
  impetus for ranking conv lengths by real cost — this round is the same
  lever as r6, extended to the 7-slice.
- **gen_dense_prime's 5/7-point modules**: checked as the obvious source
  for a 7-point kernel before writing one — their modules are dense fold
  machinery (fold_pass/zpass), not SoA tail stages; nothing adoptable.
  The DFT-7 here is my own DFT-5 symmetric form extended.
- **gen_batchlane r4 / gen_pow2 r3** (standing): the held-lease same-core
  control-first pairing protocol; the -DBST_NO7 control arm exists so the
  A/B is build-for-build, not memory-vs-window.

### What I would do next (gen_r8)
1. **13*2^k (M=208) for L=97..104** if L=100 needs another lever: the
   only remaining M cut for that cell; budget the DFT-13 tail honestly.
2. **bluestein_cost(L) for gen_planner** (sixth carry, grid updated):
   cost ~ 3L^2 rows x 0.66 ns per (row * M(L)*log2 M(L) / 8), now with
   M(L) = min{2^k, 3*2^k>=48, 5*2^k>=80, 7*2^k>=112 : >= 2L-1}.
3. **Cross-arch re-races** (carried): all knobs incl. BST_NO7 on CLX/SPR
   when the next XARCH lands (CLX's 1 MB L2 may reweight the M=112/224
   pass structure).
4. The issue-shape lever (2,8-split-radix, lit 11) still applies only to
   the pow2-M holdout slices (L=25..32, 57..64, 121..128); graded 25/27/
   31/32 sit there at M=64.  That is the remaining arithmetic-count idea
   in this class.

## Round gen_r8

### What changed (impl/gen_bluestein.c: one lever built, raced, and raced OFF)

1. **13*2^(2k) convolution grid (M=208 for L=97..104), my r7 next-step 1 —
   BUILT and KILLED by the race.**  New code, all shipped but create()-gated
   OFF: BST_DFT13_LANE (the DFT-5/7 symmetric t/d form extended to six
   6-term dot products; jk-mod-13 constant/sign rows verified against a
   direct DFT to 1e-14 before compiling), plain-C dft13_{fwd,inv}_stage
   (create()'s bh forward + scalar builds), vector BST_DFT13V_ARR, fused
   conv_mid13, and the grid slice `M13 = 208; while (M13 < 2L-1) M13 <<= 2`
   (<<= 2, not <<= 1, keeps the tail at 13 and never 26, so no PFA-26
   module is needed; S = M/4 stays 4-aligned so the transpose contract
   holds untouched).  The slice is now OPT-IN via -DBST_M13; the default
   grid and every default code path are the gen_r7 ship exactly.
2. Correctness of the (gated-off) 13 machinery, all PASS before the race:
   node+local singles at L in {97, 100, 101, 104} <= 1.1e-15; L=100 m=2
   fused gate 3.75e-15 (tol 3e-14); graded chain m=64 3.24e-14 (anchor
   2.42e-14); chain repeatable; scalar -march=x86-64 build singles + m=2 at
   L=100; and L=385 B=1 (the M=832 = 13*4^3 octave, dif16/dit16 over the
   13-chain) 7.3e-16 — the only sizes that ever pick a 13-grid M.

### The race that killed it (a80n0, ONE held lease, core 3, control-first
### same-core alternating pairs, -DBST_NO13-equivalent control vs M13 arm;
### graded cell L=100 B=1 m=64)

| pair | control (r7 grid, M=224) | M13 arm (M=208) |
|---|---|---|
| 1 | 16287.8 | 17194.1 |
| 2 | 14589.0 | 16844.8 |
| 3 | 14418.8 | 16000.3 |

**3/3 pairs to the control, ~ +10%.**  The -7.1% conv-data cut is real, but
the DFT-13 tail is ~17.5 vector ops/pt vs PFA-14's ~15.1 (+16%), all of it
FMA-port work, and gcc spills the 26-plus-live-zmm fused middle on top
(1039 uops/block vs mid14's 742).  A wallaby (SPR) pre-race read the same
direction (2/3 pairs to control).

### The r8 static analyzers, used as the brief intended (first use in this
### record): llvm-mca predicted this kill before the lease was spent
uiCA's install is broken (missing generated instrData module — recorded for
whoever owns ext/tools).  llvm-mca -mcpu=icelake-server on the gcc codegen
(function bytes extracted from the built binary via objdump, addresses
stripped): conv_mid13 460 cyc/block-iteration vs conv_mid14 313 — +47% on
the mid pass per point, against a 7% cut spread over the four M-scaled
passes.  Model said lose, wallaby leaned lose, the node confirmed 3/3.
Both middles report Port0-saturated (97%) in the ICX model — that model
funnels ALL 512-bit FP into one port (the known blind spot list in
tools/TOOLS.md misses this one), so treat its absolute cycles as ~2x
pessimistic on the 2-FMA-pipe Gold 6326, but the RELATIVE verdict was
correct.  Protocol note: llvm-mca will happily consume objdump -d
--no-show-raw-insn output once addresses and call lines are sed-stripped —
no need to re-compile with clang to analyze the SHIPPING gcc code.

### Ship state (default build = gen_r7 arithmetic and paths exactly)
Chain output at L=100 cmp-IDENTICAL to the control arm.  Node reads, ship
binary: L=100 B=1 m=64 **14387.8-16520** us across the session's windows
(board 14379.9; MKL 2022 same window 7740.1), L=50 B=4 m=128 **1394.9**
(board 1397.6), L=10 B=64 m=1000 **13.44** (board 13.39) — parity
everywhere, as an arithmetic-identical ship must read.  Gates on the ship:
L=100 m=2 4.109e-15 (tol 3e-14), chain m=64 3.564e-14 (anchor 2.416e-14,
1.48x honest — r7's exact digits); generality sweep B=1 singles at L in
{2, 7, 11, 17, 20, 24, 33, 40, 48, 50, 56, 63, 64, 71, 96, 97, 100, 101,
104, 105, 112, 127, 128} ALL PASS <= 1.0e-15 on the node.  create() ~0 s.

### What did NOT work / was decided without a window, with the number
- **M=208 via DFT-13**: the round's headline null, numbers above.  The
  flap-count lesson (lit 11 Tier 1) has a boundary this class just
  measured: ranking conv lengths by data volume stops paying when the
  cheaper length's tail module costs more per point than the data cut
  saves — the {3,5,7}-slices won because DFT-3/5/7 are at-or-below
  radix-4's per-point cost; a 6x6-dot-product DFT-13 is not.  13*2^k
  should stay out of everyone's grids on ICL unless the tail gets a
  fundamentally cheaper form (Rader-13 via an 12-pt cyclic conv is the
  only candidate, and it is a whole sub-engine).
- **Rescheduling conv_mid13 to cut gcc's spills** (split re/im halves
  around the pointwise multiply, ~24 live instead of ~38): declined by
  arithmetic — even a zero-spill mid13 keeps the +16%/pt FMA-port floor,
  and the measured deficit (1.4-2.4 ms/step) is larger than the entire
  modeled spill share (~1 ms).  Not worth a lease slot.
- **Improving conv_mid14 the same way**: declined — llvm-mca shows it
  FP-port-bound, so its spill uops ride the idle load/store ports for
  free (the r6 conv_mid12 lesson from the other side).

### Borrowed this round, named
- **gen_pow2 gen_r7**: the negative-result round as a first-class
  deliverable (their constant-per-site verdict), and the asm-audit/
  model-before-race discipline this round ran on the new analyzers.
  Their r7 also killed constant-per-site twiddle routing on ICL, which
  took that brief item off my list without a window.
- **gen_pfa_large gen_r7**: the two-axes-per-pass closure accounting —
  their traffic argument transfers to my blocked-custody chain (same
  two-sweep floor), closing brief item 1 for this class without code.
- **gen_batchlane r4 / gen_pow2 r3** (standing): held-lease same-core
  control-first pairs; the kill above is 3/3 clean pairs, not a window
  artifact.
- **tools/TOOLS.md (the r8 addition)**: llvm-mca as the pre-lease filter;
  this round it paid for itself (the kill was predicted before the race).

### What I would do next (post-campaign / cross-arch)
1. **Cross-arch re-races** (carried): all knobs — BST_MAPFUSE_MAX_MIB,
   BST_NOCFUSE, BST_BLKFUSE, BST_PF, BST_SCHED, BST_MID12, BST_NO7, and
   now BST_M13 — on CLX/SPR when the next XARCH lands.  BST_M13 on CLX is
   the interesting one: a downclocked port-bound host weights the -7%
   data cut differently, and CLX's 1 MB L2 changes the M=224-vs-208
   buffer story.
2. **bluestein_cost(L) for gen_planner** (seventh carry, grid unchanged
   from r7): ~0.66 ns per (row * M(L)*log2 M(L) / 8) + ~0.15 ns/pt/step
   chain map on ICL, M(L) = min{2^k, 3*2^k>=48, 5*2^k>=80, 7*2^k>=112 :
   >= 2L-1}.
3. **Tail groups (nv < 8) masked contig path** (fourth carry): still
   cosmetic (<= 1/1275 of groups, odd L at B=1 only).
4. The class's arithmetic-count ledger is now closed on ICL: every
   divisor-slice of every octave that can pay has been measured (3, 5, 7
   win; 13 loses; 15*2^k/M=240 fails coverage; M=24/12 fails the
   transpose contract).  What remains at L=100 is issue shape in the
   radix-4 bodies themselves, and the analyzers now say those are
   FP-port-bound at ~2.1 uops/cyc — the honest next lever, if one is
   ever needed, is fewer FLOPs per butterfly (2,8-split-radix chains,
   lit 11), which only applies to the pow2-M holdout slices (graded
   25/27/31/32 at M=64).

## Round gen_r9 (reconstruction, written in gen_r10)

RECORD HYGIENE NOTE: this section was never written in round gen_r9 itself --
the r9 agent log (results/gen_r9/agents/gen_bluestein.log) is EMPTY (1 byte)
though exits.txt says exit=0, and no "Round gen_r9" section was appended.
The shipped r9 change is reconstructed here from `diff impl_8/gen_bluestein.c
impl_9/gen_bluestein.c` (123 lines, one change, fully commented in-source):

**j-outer, blocks-inner loop order in dif4_stage / dit4_stage** (knob
-DBST_NOJX restores blk-outer). Every block of a generic middle stage consumes
the SAME twiddle triple, so the 6 broadcasts hoist out of the block loop;
M/len == 4 at every shipping M's generic middle stage, so 3/4 of the stage's
broadcast uops disappear (gen_rader gen_r8's broadcast amortization, applied
across blocks instead of column pairs). The M == 4*len case additionally
unrolls the 4 blocks explicitly. Per-element arithmetic untouched; outputs
bit-identical. r9 board (NOTE: new host a81n2, absolute numbers not
comparable to the a80n0 boards): L=10 12.83, 25 169.6, 27 202.5, 31 272.6,
32 297.2, 40 605.3, 50 (cut off in context), 100 (not in my context copy) --
all gates ok. This change is load-bearing for r10's radix-16 verdict below.

## Round gen_r10

### What changed (impl/gen_bluestein.c: two adoptions + one raced gate move)

1. **Axis-2 transpose-scatter final layer as 256-bit half stores**
   (`tr8x8_store`, knob -DBST_NOTRST; named borrow: gen_pow2 gen_r9's
   vextractf64x4-TO-MEMORY -- p237+p4, zero p5 uop, 256-bit stores retire
   2/cycle). My last_scatter_tr's transpose feeds MEMORY (unlike gen_pow2's
   TR8-internal shuffles, which their record correctly says cannot take the
   store path), so the third shuffle layer can be deleted entirely: layers
   1+2 become 8 unpacks + 8 vpermt2pd (index vectors chosen so each result's
   256-bit halves are exactly row v's / row v+4's half-rows), layer 3 becomes
   8 vmovupd-ymm (low halves) + 8 vextractf64x4-to-mem (high halves).
   Per full 8x8 block: p5 24 -> 16 uops, stores 8x512 -> 16x256 (bandwidth
   neutral), front-end uop count unchanged. Stored bytes BIT-IDENTICAL
   (cmp-verified vs -DBST_NOTRST at L=25/31/100/21). Applies at every size
   with M >= 16 (the tr path), nv == 8 groups only; nv < 8 and the partial-r1
   boundary block keep the zmm path. objdump audit: 16 memory-destination
   vextractf64x4 + 16 vpermt2pd in the shipped function, exactly as designed.

2. **nv < 8 tail groups through a masked contig pipeline**
   (first_gather_contig_tail / last_scatter_contig_tail, knob -DBST_NOTAIL;
   named borrow: gen_twiddle gen_r9's masked tail forms -- maskz lanes are
   exactly +0.0 = the zero pad). Closes my item carried since gen_r3: the
   last nrows % 8 rows of an axis-0/1 pass (always the tail of the last div
   block, so a single contiguous run) previously took the generic scalar
   path. Now: zero-masked loads -> same permutex2var/chirp/pruned-first-
   stage pipeline; masked-off lanes carry zeros through the linear
   convolution and are never stored. The fused map runs BST_MAP8 in the tail
   too (gen_twiddle r9's divider warning -- "sqrt/div price per OP, not per
   useful lane; keep partial-group divider work scalar" -- does NOT bite:
   the ladder is rsqrt14/rcp14 + Newton, no divider op). Side effect: the
   L=101-type tail non-identity recorded in gen_r5 (scalar sqrt/div tail vs
   ladder elsewhere) is gone -- tails are now ladder-exact like full groups.
   Graded cells all have nrows % 8 == 0, so this is surprise-size coverage,
   not graded speed.

3. **Radix-16 fusion gate raised: len >= 32 -> len >= 64** (forward dif16 and
   the mirrored dit16 condition). The r10 finding that paid for the round:
   after r9's j-outer dif4/dit4, the register-fused radix-16 pairs LOSE at
   len = 32/48. Node, held lease, control-first same-core pairs, 3/3 each:
   - L=63 (M=128): radix-4 2748.3/2744.0/2761.3 vs dif16 2973.3/2983.1/
     2992.7 us = **-7.7%**
   - L=96 (M=192): 10231/10222/10307 vs 11459/11304/11306 = **-9.8%**
   The S2 = len/16 = 2..3 tile loop is too short to amortize the 32-live-zmm
   spills against the now-lean unrolled j-outer dif4. At len = 64 (M = 256,
   32 KiB buffer past L1) the deleted pass still pays: L=127 B=1, 6 pairs
   over two windows, 5/6 to fusion (-1.5..-2% in the clean pairs; one dirty
   34.7-vs-38.8 outlier pair each way). Outputs bit-identical either way
   (the r2 design: fusion is scheduling, not arithmetic), so the gate is
   pure schedule policy. M=128/192 slices cover surprise sizes 57-64 and
   81-96; no graded cell uses dif16 (M = 224/112/80/64/48/32).

### Measured on the node (a81n2 -- the r9 scoring host; ONE held lease per
### battery, control-first same-core alternating pairs, new vs
### -DBST_NOTRST -DBST_NOTAIL control; graded chain cells, min us/xform)

TRST+TAIL vs control (changes 1+2; graded cells have no tails, so these
pairs are effectively the TRST attribution):
- L=25 B=16 m=256: 8 pairs over two windows, new 5/8, median -0.8%
  (172.2/173.6, 173.1/173.0, 170.0/171.3; 169.1/170.8, 169.0/170.3,
  170.3/170.0, 170.0/171.7, 175.5-dirty/169.6)
- L=31 B=16 m=140: 8 pairs, new 5/8, wash-to-lean in a noisy window
  (spread 274-305 us in-window)
- L=50 B=4 m=128: new 4/5, mean **-1.7%** (1374.4/1383.3, 1556.1/1563.2,
  1437.7/1425.2, 1460.5/1502.5, 1419.2/1502.8)
- L=100 B=1 m=64: new 2/3 (14718.7/15878.2-dirty-ctl, 14847.9/14765.4,
  14566.3/14790.6)
Verdict: a small lean win that never consistently loses, strongest where the
blocked-custody chain spends most time in the tr path; same magnitude as
gen_pow2's own r9 measurement of the trick (-1.5% at their L=32).

Tail-only attribution (bin_trst = -DBST_NOTAIL vs full new) at L=21 B=1
m=200 (nrows = 441, one nv=1 tail group per axis-0/1 pass): 3/4 pairs to
trst, all <= 1% (94.6/95.3, 97.3/94.8, 96.4/96.4, 97.1/96.2). A 1/56-groups
change cannot arithmetically cost 1%; this is inside gen_layout r9's
measured 1.5% code-layout confound band (they saw that delta between
binaries executing IDENTICAL tail instructions). Shipped ON for coverage
and tail exactness; knob kept for the cross-arch re-race.

Ship binary, fresh-core tryout reads, same session (MKL 2022 same window):
| cell | gen_bluestein | MKL | vs r9 board (same host) |
|---|---|---|---|
| L=10 B=64 m=1000 | 13.12 | 4.59 | 12.83 (window parity) |
| L=25 B=16 m=256 | 174.4 | 122.2 | 169.6 (window) |
| L=31 B=16 m=140 | 279.4 | 854.4 | 272.6 (window) |
| L=40 B=8 m=128 | 606.8 | 406.4 | 605.3 (parity) |
| L=50 B=4 m=128 | 1380.3 | 947.7 | -- |
| L=100 B=1 m=64 | 14259 | 7838 | -- |
| L=63 B=1 single | 2748 | -- | was 2973-2993 (-7.7%) |
| L=96 B=1 single | 10222 | -- | was 11304-11459 (-9.8%) |

### Gates (shipped default build; singles/2-step/chains on wallaby-SPR by
### hand -- full AVX-512 vector paths per gen_layout r9's host correction --
### singles re-confirmed on the node via tryout at the 6 cells above)
Generality singles B=1: 34 sizes {2,3,5,7,9,11,13,16,17,21,23,24,33,35,40,
47,48,49,56,63,64,65,71,80,96,97,101,104,105,112,113,120,127,128} ALL PASS
<= 1.1e-15 (tol 1e-12) -- covers every tail type, both PFA parities, all M
grids incl. both sides of the new len>=64 fusion gate (63/64: M=128, 96:
M=192, 113-128: M=256), tail-group odd-L sizes (21/35/101/113...), the
L = M/2 edges, and the giants. Two-step m=2: L=25 2.06e-15, L=31 2.55e-15,
L=63 3.76e-15, L=96 3.34e-15, L=100 4.11e-15, L=101 5.86e-15 (tol 3e-14,
>= 5x margin). Graded chains: L=25 4.83e-14 (anchor 2.80e-14), L=31
4.17e-14 (2.31e-14), L=50 6.13e-14 (2.92e-14), L=100 3.56e-14 (2.42e-14),
L=10 m=1000 2.30e-13 (1.08e-13), plus tail-exercising L=21 m=200 3.41e-14
(1.76e-14) and L=101 m=64 5.06e-14 (3.57e-14) -- all <= 2.2x honest, tol
1e-10. All chains bit-repeatable (cmp-identical across runs). Scalar
-march=x86-64 build: singles + m=2 chains PASS at {21, 50 B=4, 100}.
create() still ~0 s. TRST bit-identity and R16-gate bit-identity both
cmp-PROVEN against knob builds (the control arms are flag-disabled builds
of the same source, per gen_rader r9's symlink-trap warning -- never
compare against the impl_N snapshot, `impl` IS the round symlink).

### Brief avenue 1 (bank the picks): N/A for this entry, stated for the
### monitor -- create() is branch-free and deterministic (M grid is a pure
### function of L; no internal race, no wisdom interaction); 5 consecutive
### create() cycles trivially pick identically.

### What did NOT work / was declined without a window, with the reason
- **2,8-split-radix on the pow2-M holdout slices** (my r5/r7/r8 carried
  "last arithmetic lever"): DELETED from the list, not just demoted.
  gen_pow2 r9 measured exactly this class of change on the same
  port-floor-bound engine shape: -7% ops, ZERO wall (their r5 FTW
  subtraction, reaffirmed r9). The op-count ledger for this class is now
  closed on ICL in both directions.
- **ymm/port-1 side-work (PMU audit avenue 4)**: dead by three independent
  r9 records -- gen_pfa_large's portcal3 microbench (8 zmm FMA chains + K
  ymm streams = exactly (8+K)/2 cycles: 256-bit FP steals 512-bit slots
  1:1), gen_batchlane's microarchitecture argument (ICL port 1's FP pipe IS
  the lower half of port 0's fused 512-bit unit), gen_layout's confounded
  tail result. My r10 splits no FP to ymm; the 256-bit stores in change 1
  are store-port work, which is the one resource this argument does not
  touch.
- **Lifted DFT5 (gen_pfa_small r9 / gen_batchlane r7)**: checked against my
  BST_DFT5_LANE/BST_DFT5V as the survey suggested -- NO op delta for my
  form. My q-pair already costs 4 ops/component (2 mul + 2 FMA for
  q1 = S51*d1 + S52*d2, q2 = S52*d1 - S51*d2); the lift (u = d1 - PHI*d2;
  q2 = S52*u; q1 = S51*u + KL5*d2) is also 1 FMA + 1 mul + (1 mul + 1 FMA)
  = 4. Their 8 -> 6 win came off a different starting form. Declined on
  arithmetic; nothing to race.
- **Gather-side 256-bit load restructure** (the mirror of change 1 in
  first_gather_tr): declined without a run -- vinsertf64x4-from-memory is
  p05 + p23 (uops.info), so the "saved" p5 shuffle comes back as a p0/p5
  blend uop in the FP-contended pool. gen_pow2 validated only the store
  side; adopting only what was proven.
- **Tail path at L=21 read +0.6..1% vs the trst-only arm** (numbers above)
  -- kept anyway; see the layout-confound reasoning. If a future round sees
  a real regression at odd-L B=1 cells, -DBST_NOTAIL is the first knob.

### Borrowed this round, named
- **gen_pow2 gen_r9**: vextractf64x4-to-memory through the scatter (change
  1 is their trick at my hottest p5 site), plus their negative ("op cuts
  are wall-neutral on a port-floor-bound engine") which deleted my
  split-radix carry.
- **gen_twiddle gen_r9**: the masked tail forms (change 2's shape) and the
  divider-occupancy boundary I checked my ladder against.
- **gen_pfa_large / gen_batchlane / gen_layout gen_r9**: the three-way
  port-1 kill, adopted as a standing null.
- **gen_rader gen_r9**: the symlink trap -- both control arms this round
  are flag-disabled builds, never the impl_N directory.
- **gen_batchlane r4 / gen_pow2 r3** (standing): held-lease same-core
  control-first pairing for every keep/kill above.

### Operation count (delta vs r9 ship)
Change 1: per full 8x8 scatter block, 8 fewer p5 shuffle uops, +8 store
uops at half width (store-bandwidth and front-end neutral); no FP change;
bytes identical. Change 2: per tail group (<= 1 per axis-0/1 pass, only
when 8 does not divide B*L*L), the 2L-element scalar gather/scatter/chirp
(+ scalar sqrt/div map in chains) becomes the vector pipeline. Change 3:
at M = 128/192 only, one extra buffer pass each way (dif4+dif4 vs dif16)
that removes the fused tile's register spills; arithmetic bit-identical.

### What I would do next
1. **bluestein_cost(L) for gen_planner** (eighth carry): ~0.66 ns per
   (row * M(L)*log2 M(L) / 8) + ~0.15 ns/pt/step chain map on ICL,
   M(L) = min{2^k, 3*2^k>=48, 5*2^k>=80, 7*2^k>=112 : >= 2L-1}.
2. **Cross-arch re-races** (carried, list grows): BST_MAPFUSE_MAX_MIB,
   BST_NOCFUSE, BST_BLKFUSE, BST_PF, BST_SCHED, BST_MID12, BST_NO7,
   BST_M13, BST_NOJX, and now BST_NOTRST / BST_NOTAIL / the len>=64 fusion
   boundary on CLX (1 MB L2 reweights both the M=256 pass economics and the
   store-vs-shuffle trade) and SPR.
3. The len>=64 gate is coupled to the j-outer dif4: if the radix-4 bodies
   change again, re-race the boundary (a one-knob, two-size check).
4. If anyone ever needs the axis-2 gather side too: budget vinsertf64x4's
   p05 cost against the deleted p5 first (a static audit, no lease needed).

## Round gen_r11 (all hands on L=100)

### The mandatory counter protocol, run first -- and it settles the brief's
### open disagreement for this engine class
PMU baseline (a80n0, /tmp/perf, tools/pmu.sh, leased core, whole process =
6 samples + warmup of the graded L=100 B=1 m=64 chain), r10-lineage binary:

| counter | baseline (r10 arith) | after (r11 ship) |
|---|---|---|
| cycles | 39.05G | 40.34G (dirtier window) |
| p0 / cyc | 0.557 | 0.538 |
| p1 / cyc | 0.068 | 0.083 |
| p5 / cyc | 0.579 | 0.560 |
| p2_3 (loads) | 20.82G | **19.73G (-5.2%)** |
| p4_9 (stores) | 11.27G | **10.73G (-4.7%)** |
| p0+p5 / cyc | 1.14 | 1.10 |
| ALL-PORT vector uops / cyc | **2.03** | 1.94 |
| l1d.replacement | 3.26G | 3.12G |
| min us/xform same run | 14102 | **13587** |

**gen_bluestein at L=100 dispatches 2.03 all-port vector uops/cycle -- AT the
node's ~2.1 global cap -- while p0+p5 is only 1.14 of the 2.0 FMA ceiling and
DRAM is nowhere near bound (~80 MB/step at 5.6 GB/s).**  For the brief's
open question (pfa_large r7 "uop-saturated" vs audit "0.82/cyc = headroom"):
BOTH are right, per engine.  The binding resource on this host is TOTAL
vector dispatch, not any port and not traffic, once a kernel is uop-dense
enough to reach the cap; pfa_large's 0.82 p0+p5 cell is traffic-bound
(their l1d numbers), my 1.14-p0+p5 cell is cap-bound.  The actionable rule
is gen_layout r10's, adopted verbatim: at ~2.1 all-port with p0+p5 < 1.6,
port surgery and traffic tricks are dead -- ONLY deleting uops (any port)
pays, and it pays ~proportionally.  Wallaby confirmation of the corollary:
the same A/B is a WASH on SPR (8574/8628/8659 new vs 8736/8595/8552 old,
alternating pairs) -- SPR's second full 512-bit FMA pipe means it is not at
this cap; the win is Ice-Lake-shaped, exactly as the rule predicts.

### Where the 14.1 ms actually lives (new tooling result: /tmp/perf RECORD
### works on the node, -F 2000; report shows raw addresses -- map them
### through `nm <bin> | sort` ranges, the binary's static syms resolve fine)
L=100 B=1 m=64 chain, cycles:u samples: conv_mid14 31.2%, axis_pass
(inlined contig gather/scatter + custody map, axes 0/1) 30.3%,
first_gather_tr 10.0%, conv_rows_mid (inlined dit4) 9.8%, dif4_stage 9.7%,
last_scatter_tr 6.6%.  97.9% attributed.

### What changed (one function): conv_mid14 respilled by hand -- 5 phases
objdump of the shipping r10 conv_mid14: 566 insns per 14-block, of which
102 are rsp spill accesses (~20%) -- the fused DFT-14 holds u[7]+v[7]
(28 zmm) across both DFT-7s plus ~12 dot-product temps + 6 constants =
~40 live, and gcc sprays spills.  My own gen_r8 record declined fixing this
("spills ride the idle load/store ports for free, llvm-mca says FP-bound")
-- WRONG at the cap: at 2.03 all-port, every spill uop is wall time.  The
r8 decline was reasoned from a per-port model exactly where the global-cap
blind spot (TOOLS.md) sits.
Rewrite: five low-liveness phases per block, each staged through the
block's own slots (1.75 KiB, L1-hot): A fwd-DFT7(u even half) in place;
B1 fwd-DFT7(v odd half) in place; B2 memory-to-memory pointwise (load u_k,
v_k, 2-pt butterflies, bh cmuls, store u'_k, v'_k; ~16 live); C1
inv-DFT7(v'); C2 inv-DFT7(u').  Each DFT-7 phase is the exact conv_mid7
shape, which compiles spill-free (its objdump: 0 rsp accesses -- checked
before designing this).  Costs 3x14 extra L1-hot load/store pairs, deletes
the ~102 chaotic spills: **566 -> 479 insns/block, rsp accesses 102 -> 1**
(icelake-server gcc, verified objdump both).  Same ops in the same order:
**outputs bit-identical** (cmp: single + full m=64 chain at L=100 and at
L=105, which adds nv<8 tail groups on the same M=224).  Control arm:
-DBST_MID14F restores the r10 form.  A 3-phase variant (u staged, v+
pointwise+v-inverse fused) measured 497 insns / 38 rsp -- the constants +
k-loop still overflow -- and was superseded by the 5-phase before racing.
conv_mid14 runs ONLY at M=224/448+ (graded: exactly L=100); every other
cell is byte-identical code.

### Measured on the node (a80n0, held lease, same-core control-first
### alternating pairs, -DBST_MID14F control vs ship; L=100 B=1 m=64)
Core 4: 14127.6/13750.9, 14176.4/13690.4, 16162.2/15798.2, 14653.2/14595.8
-- 4/4 to ship (-2.7/-3.4/-2.3/-0.4%).  Core 2 (noisier window):
16348.6/13720.7, 14278.9/14462.7, 16179.9/14085.7 -- 2/3.  Total 6/7 pairs.
Session bests: ship **13586.9** (PMU window) / 13691.5 (fresh-core tryout)
vs control 14102.2/14127.6; board r10 was 14132.  Same-window MKL 7719-7756.
Net ~**-3% at L=100**, mechanism confirmed by counters (loads -1.09G,
stores -0.53G, FP ports unchanged, l1d.replacement -4.6%).  Parity reads
(paths untouched): L=50 B=4 1417.9 (mid7 was already spill-free), L=25 B=16
172.6, L=10 B=64 13.23 -- all board-parity.

### Gates (ship build; sweep on wallaby AVX-512 paths per gen_layout r9,
### graded cells re-confirmed on the node via tryout)
Singles B=1, 35 sizes {2,3,5,7,9,11,13,16,17,21,23,24,33,35,40,47,48,49,56,
63,64,65,71,80,96,97,100,101,104,105,112,113,120,127,128} ALL PASS
<= 1.05e-15 (tol 1e-12).  Two-step m=2: L=100 4.109e-15, L=101 5.858e-15,
L=105 4.212e-15, L=112 3.473e-15, L=50 3.333e-15, L=25 2.059e-15 (tol
3e-14).  Graded chains: L=100 m=64 3.564e-14 (anchor 2.416e-14 -- r8/r10's
exact digits, as a bit-identical change must read), L=50 6.128e-14
(2.922e-14), L=25 4.826e-14 (2.796e-14), L=10 m=1000 2.295e-13 (1.081e-13),
L=105 m=64 3.209e-14 (1.564e-14).  Chain + single outputs cmp-identical
across runs.  Scalar -march=x86-64 build (conv_mid14 is AVX512-only; scalar
takes dft14 stages): singles + m=2 PASS at {21, 50 B=4, 100}.  create()
still ~0 s.

### Sized and SKIPPED, with the numbers (the next uop-deletion ledger)
- **last_scatter_tr spills**: 76 zmm rsp movs per 4-column block (646-insn
  body) -- ro[8]+r1o[8]+LAST_LOAD temps ~30 live.  Ceiling ~1.2M uops/step
  = ~1.3% of wall; needs either recomputing shared legs (u0-u3 feed both
  r0 and r1) or organized r1 staging.  first_gather_tr: 32 zmm spill movs,
  ~0.5% ceiling.  Skipped this round on risk/reward; first candidates next.
- **axis_pass lane-offset idivs**: 8 idiv per 8-row group (objdump), the
  (r/div)*A + (r%div)*B lane loop.  ICL idiv is ~18 cyc; ~0.5-0.8%
  ceiling via incremental (q, rem) tracking.  Touches the group
  decomposition everywhere -- skipped, queued.
- **Split re/im state format between chain steps** (delete the 4
  de/interleave permutes per element in axes 0/1 gather/scatter): measured
  shuffle share is only p5-p0 ~ 0.9G of 22.6G p5 -- ~1-1.5% ceiling for a
  full-surface rewrite of every gather/scatter variant.  Declined.
- **M = 2L = 200 via chirp periodicity** (NEW degree of freedom, nobody has
  used it): b_{n+2L} = b_n exactly, so a CYCLIC convolution at M = any
  multiple of 2L is exact -- the M >= 2L-1 linear-embedding constraint is
  not the only legal grid.  At L=100 that offers M=200 = 8*25 (-10.7% data
  vs 224), but 200's chain needs a 25-point tail (5x5 CT with twiddles,
  ~16-19 vec-ops/pt) or PFA(2x25) at len 50 -- the r8 flap-count boundary
  (DFT-13's +16%/pt killed -7% data) predicts a loss or a wash, and the
  machinery is a round of work.  Documented so the planner/others know the
  constraint is soft; on the existing grid every graded L already has
  M <= 2L+something-small except none -- the trick only matters where 2L
  factors better than the grid value.
- **3-phase conv_mid14**: built, measured statically (497/38), superseded.

### Borrowed this round, named
- **gen_layout gen_r10**: the PMU dashboard method and the decision rule
  ("~2.1 all-port with p0+p5 < 1.0-1.6 => stop rebalancing, delete uops")
  -- this round IS that rule applied to my own r8 wrong turn; also their
  /tmp/perf restage recipe.
- **gen_pfa_large gen_r10**: the NT-at-100 kill and ymm-side-work kill
  (standing nulls I did not re-test), and the 80 MB/step floor accounting
  that let me classify my cell as compute- not traffic-bound.
- **gen_batchlane r4 / gen_pow2 r3** (standing): held-lease same-core
  control-first pairs; both A/B sets above are that protocol.
- **gen_rader r9** (standing): control arm is a flag-disabled build of the
  same source, never an impl_N snapshot.

### Operation count (delta vs r10)
FP arithmetic: zero change (bit-identical outputs).  Per 14-block of
conv_mid14: +42 L1-hot loads +42 L1-hot stores (staging), -102 rsp spill
accesses, net -87 insns (-15%); x16 blocks x 1250 axis-groups x 3 axes
per step at L=100 => ~-5M dispatched uops/step modeled, -1.6G loads+stores
measured process-wide (-3% wall at the cap).

### What I would do next (gen_r12)
1. **last_scatter_tr / first_gather_tr spill diet** (the 1.3% + 0.5%
   sized above) -- same treatment, needs a schedule that either recomputes
   the shared u-legs or stages r1o through the w buffer.
2. **axis_pass incremental lane offsets** (kill the 8 idivs/group).
3. Re-race BST_MID14F on SPR/CLX when the next XARCH lands (SPR measured a
   wash locally -- the knob exists so the race can pick per host; CLX's
   downclocked divider and 1 MB L2 may reweight both directions).
4. bluestein_cost(L) for gen_planner (ninth carry, unchanged constants).
5. The M = 2L periodicity note above if anyone ever builds a cheap 25-tail.

## Round gen_r12 (reconstruction, written in gen_r13)

RECORD HYGIENE NOTE: like gen_r9, this section was never written in round
gen_r12 itself -- the r12 agent log (results/gen_r12/agents/gen_bluestein.log)
is EMPTY (1 byte) though the round shipped a real change.  Reconstructed from
`diff impl_11/gen_bluestein.c impl_12/gen_bluestein.c` (111 lines, two
changes + one static null, all commented in-source) and the r12 leaderboard.

1. **last_scatter_tr spill diet** (r11 next-step 1, the 1.3%-ceiling item):
   the k = j+S outputs are staged through an aligned L1-hot stack slot
   (`stg[64]`) as they are produced and reloaded only after the r0
   transpose-store has retired its register set, instead of accumulating
   ro[8]+r1o[8] (+ BST_LAST_LOAD temps, ~30 live) across the column loop.
   The r11 spill audit had counted 76 chaotic rsp movs here.  Same FP ops in
   the same order: outputs bit-identical.  -DBST_NOSTG restores the r11 form.
2. **axis_pass incremental lane offsets** (r11 next-step 2): incremental
   (rem, off) tracking replaces the per-lane (r/div, r%div) formula -- kills
   8 idivs (~18 cyc each on ICL, unpipelined) + 16 imuls per 8-row group,
   all on the group prologue's critical path.  rem walks 0..div-1 with a
   wrap; off bumps by 2B per row and 2(A - div*B) at each div-block
   boundary.  Offsets identical by construction; bit-identical outputs.
   -DBST_NOINC restores the divide form.
3. Static NULL (in-source comment): the same stack-staging treatment on
   first_gather_tr's a-legs -- 654 -> 657 insns, rsp movs 32 both ways; gcc
   already schedules the gather side cleanly.  Not shipped.

r12 board (a80n0): L=100 B=1 13499.7 us (r11 board 13587-14132 lineage),
L=50 1374.3, L=40 606.5, L=31 274.0, L=27 202.5, L=25 169.2, L=20 80.7,
L=15 33.3, L=12 18.75, L=10 12.65 -- all gates ok (ch <= 5.7e-14 vs 1e-10,
1s <= 3e-15).  No round section was written, so no same-window A/B
attribution numbers exist for the two changes; the board reads are
parity-to-slightly-better everywhere, consistent with two uop-deletion
changes sized ~1-2% at the big cells.

## Round gen_r13 (the B=1 small-L round)

### What changed (impl/gen_bluestein.c, one structural change, ~176 diff lines)

**Sub-32 convolution grid: M = 20, 24, 28, 40, 56 (S = M/4 = 5, 6, 7, 10,
14), via CLAMP-OVERLAPPED transpose blocks.**  The k >= 4 floor on the
{3,5,7}*2^k grid existed for one reason only: the axis-2 transpose
gather/scatter processes j in 4-wide blocks and required S % 4 == 0.  My r6
and r7 records declined M=24 twice ("surgery on the hottest, most
correctness-critical functions... at L=11 even the masked boundary block
overruns S=6") because the fix on the table was 2-wide block variants of
every leg region.  The actual fix is 15 lines of loop schedule, not new
blocks: iterate jb = 0, 4, 8, ... and CLAMP the final block to j = S-4.
The clamped block recomputes up to 3 columns the previous block already
produced -- but every column is a pure function of src (gather) or w
(scatter), never of its own pass's output, so the overlap is IDEMPOTENT and
bit-identical; and each block classifies itself dual/boundary/single
against jful independently, so the r2 masked-b boundary machinery handles
every mixed case (including the L=11 shape that scared r7) unchanged.  The
r7 "overrun" objection dissolves entirely: with the clamp, no address ever
passes S.  Chirp reads reach 2S-1, and every sub-32 pick satisfies
2S-1 <= L+3 (worst cases L=17/M=40: 19<=20 and L=25/M=56: 27<=28), so the
existing L+4 slack covers it -- checked per pick before shipping, noted at
the alloc.  Grid change itself is three constants (M3/M5/M7 start at
24/20/28 instead of 48/80/112).  New picks and their chains:
  L=9,10:  M 32->20  first(5)+mid5+last(5)      -37.5% conv data, 5->3 passes
  L=11,12: M 32->24  first(6)+mid6+last(6)      -25%
  L=13,14: M 32->28  first(7)+mid7+last(7)      -12.5%
  L=17..20: M 48->40 first(10)+mid10+last(10)   -17%  (graded 20:32)
  L=25..28: M 64->56 first(14)+mid14+last(14)   -12.5% (graded 25:16, 27:16)
All five tail modules (mid5/6/7/10/14) already existed from r6/r7 and loop
generically over M; conv_rows_mid's zero-middle-stage dispatch was already
exercised by M=16.  Every other size takes a byte-identical code path
(S % 4 == 0 keeps the original three-phase loops; cmp-PROVEN below).
-DBST_NOSML = the gen_r12 grid (attribution control + CLX/SPR re-race arm).
The 15:32 cell stays at M=32 (2L-1=29 > 24); 29..32 stay at M=64 (M=60
needs a DFT-15 module -- declined below).

### Why this was the round's lever (the brief's B=1 small-L mandate)
The two new scored cells 10:1:16384 and 12:1:12288 run exactly the M=32
slice, where the old grid was weakest relative to size: a 5-pass pipeline
(first + dif4(8) + mid2 + dit4(8) + last) for a 10-element row.  The r6
lesson ("the M-grid cut is the most reliable lever this engine has") applied
directly; the r8 flap-count boundary does not bite because DFT-5/6/7/10/14
are all at-or-near radix-4's per-point cost (unlike the DFT-13 that lost r8).

### Measured on the node (a80n0, ONE held lease per battery, control-first
### same-core alternating pairs, -DBST_NOSML control vs ship; graded chain
### cells incl. the fused map, min us/xform)

| cell | ctl (r12 grid) | new | delta |
|---|---|---|---|
| 10:1 m=16384 | 12.93 / 13.17 (2 dirty 19-20) | 8.64 / 8.71 / 8.71 / 8.87 | **-33%** (4/4) |
| 12:1 m=12288 | 18.07 / 18.14 / 18.18 / 18.29 | 11.99 / 12.20 / 12.48 / 13.03 | **-31%** (4/4) |
| 10:64 m=1000 | 12.77 / 12.81 | 8.75 / 8.84 | **-31%** (2/2) |
| 12:64 m=600 | 18.95 / 19.05 | 13.33 / 13.62 | **-30%** (2/2) |
| 20:32 m=256 | 81.5 / 82.2 / 82.4 | 64.06 / 64.63 / 73.9 | **-20%** (3/3) |
| 25:16 m=256 | 169.9-174.8 | 163.8-165.9 | **-4%** (4/4) |
| 27:16 m=200 | 203.0-204.1 | 195.2-198.8 | **-3.5%** (4/4) |

Singles (fresh-core tryout): 10:1 **9.01** us (r12 binary same window
13.60; MKL 1.47-1.51), 12:1 **12.96** (MKL 2.18).  Parity reads at
untouched cells: 15:32 33.13 (board 33.31), 31:16 276.8 (board 274.0).
The 25/27 win is smaller exactly as the pass-count rule predicts (the
deleted dif4(16)/dit4(16) passes over an 8 KiB buffer were L1-resident ~
free, pfa_large r4's boundary; the win there is the -12.5% on the fused
first/last passes), but it is 8/8 pairs consistent -- kept.
For calibration against the round target: fftw3's benchFFT pace at 10:1 is
~2.1 us/transform, so this entry stays a fallback at these cells (4x off);
the class winners own them.  The fallback just got 1.5x better everywhere
its grid was coarse.

### Gates (ship default build, all on the node by hand, check.py)
Generality singles B=1, 46 sizes = every new-M size {9..14, 17..20, 25..28}
+ the standing regression set {2,3,5,7,15,16,21,23,24,31,33,35,40,47,48,49,
56,63,64,65,71,80,96,97,101,104,105,112,113,120,127,128}: ALL PASS
<= 1.05e-15 (tol 1e-12).  Two-step m=2 fused-chain gate at {9,10,11,12,13,
14 B=3, 17 B=4, 18, 19, 20 B=32, 25 B=16, 26 B=2, 27 B=16, 28}: 1.32e-15
.. 2.60e-15 (tol 3e-14, >= 11x margin).  Graded chains: 10:1 m=16384
2.73e-6 (anchor 9.77e-7, 2.8x honest, tol 3e-4 -- the 16k-step drift scale),
12:1 m=12288 7.69e-9 (anchor 8.00e-9 -- BELOW honest), 10:64 m=1000
2.01e-13 (1.08e-13), 12:64 m=600 9.31e-14 (3.89e-14), 20:32 6.79e-14
(2.84e-14), 25:16 5.10e-14 (2.80e-14), 27:16 4.68e-14 (2.57e-14) -- all
<= 2.8x honest, tol 1e-10.  Chain outputs cmp-repeatable (20:32, 12:1).
PARITY: ship vs -DBST_NOSML outputs cmp-IDENTICAL at 9 untouched sizes
{15:32, 16:4, 23:1, 31:16, 32:8, 40:8, 50:4, 64:1, 100:1} -- the graded
suite outside the five changed slices is provably untouched.  Scalar
-march=x86-64 build: singles PASS at {9..14, 18, 20, 26, 28} + m=2 chain at
12:1 (the plain-C dft5/6/7/10/14 stages end to end).  create() still ~0 s.

### What did NOT work / was declined, with the reason
- Nothing raced off this round -- all five slices kept, 25/29 pairs to the
  ship across seven cells (the 4 losses were all in the two dirty 10:1
  control reads' pairs, which the ship still won).
- **M=60 (15*4) for L=29,30**: S=15 works under the clamp, but needs a new
  conv_mid15 (PFA 3x5, ~30 live zmm) for two never-graded sizes; L=31/32
  (2L-1 = 61,63) stay on 64 regardless.  Same coverage decline as r6's
  15*2^k.
- **M=44 (11*4) for L=21,22 and M=36 (9*4) for L=17,18**: need DFT-11 /
  DFT-9 tail modules that do not exist in this file; the r8 DFT-13 lesson
  (a tail module pricier per point than radix-4 eats a <=10% data cut)
  predicts at best a wash.  Declined without a window.
- The gen_r5-era "M = 2L periodicity" note is now partially moot: at L=10
  the new grid pick M=20 IS 2L, so the cyclic-convolution optimum and the
  linear-embedding grid coincide there.

### Harness notes (delta)
- **tryout.sh's cases.txt lookup breaks at L=10/12 this round**: `awk $1==l`
  now matches BOTH rows (10:64:1000 and 10:1:16384), M becomes two lines,
  `[ $M -gt 1 ]` errors out and the chain leg silently runs m=1.  Chain
  timings/gates for L=10/12 must be run by hand on the node (this round's
  A/B script: build/tryout/gen_bluestein/r13_ab.sh).  Monitor may want the
  awk keyed on $1==l && $2==b.
- r13_verify.sh in the same directory is the full battery (build/sweep/
  parity/gates/scalar) used above; first cut of the repeatability leg ran
  without cd on the remote side and reported NOT-REPEATABLE from stale
  files -- a check that could not pass; loud errors caught it, rerun clean.

### Borrowed this round, named
- **The round brief's known-good list**: item 1 (within-volume pencil
  lanes) is already this engine's native form (8 rows of one volume per
  group since r1), so the round's work went to the M-grid instead.
- **literature 11 Tier-1 flap-count ranking** (standing since r6): the
  cost-not-convenience grid framing; this round is its third application.
- **gen_batchlane r4 / gen_pow2 r3** (standing): held-lease same-core
  control-first pairing for every number above.
- **gen_rader r9** (standing): control arm is the flag-disabled build
  (-DBST_NOSML) of the same source, never an impl_N snapshot.

### Operation count (delta vs r12, per row at the changed sizes)
Buffer passes 5 -> 3 everywhere the grid moved; conv elements per pass
32->20 (L=9,10), 32->24 (11,12), 32->28 (13,14), 48->40 (17..20),
64->56 (25..28).  Transpose block count per tr row-group is UNCHANGED
(ceil(S/4) with overlap = the old S/4 at every pick: 2 vs 2 at 10/12,
3 vs 3 at 20, 4 vs 4 at 25), so the p5 bill did not move; the win is pure
mid-pass deletion + first/last data cut.  Gather/scatter chirp work (O(L)
per row) unchanged.  Overlapped tr columns recompute 1-3 of S columns
(20-60% of one block), bounded by one block per group side.

### What I would do next
1. **Cross-arch re-races** (carried, list grows): BST_NOSML joins
   BST_MAPFUSE_MAX_MIB, BST_NOCFUSE, BST_BLKFUSE, BST_PF, BST_SCHED,
   BST_MID12, BST_NO7, BST_M13, BST_NOJX, BST_NOTRST, BST_NOTAIL,
   BST_NOSTG, BST_NOINC on CLX/SPR when the next XARCH lands.
2. **bluestein_cost(L) for gen_planner** (tenth carry, grid updated):
   ~0.66 ns per (row * M(L)*log2 M(L) / 8) with M(L) now min over
   {2^k, 3*2^k>=24, 5*2^k>=20, 7*2^k>=28} >= 2L-1 -- and a 3-pass constant
   at the sub-32 picks vs 5-pass above; the planner should stop assuming
   the k>=4 grid for me.
3. If the 10:1/12:1 cells ever matter for this entry beyond fallback duty:
   the remaining gap to fftw3 (~4x) is group-prologue overhead (13 groups
   of 8 rows for a 100-row axis, gather/scatter dominated) -- the honest
   fix is a whole-volume-in-registers small-L specialization, which is the
   class winners' territory, not Bluestein's.
4. conv_mid15 (M=60) only if a future round grades L=29/30.

## Round gen_r14 (the execute()/chain() plumbing-seam round)

### What changed (impl/gen_bluestein.c, one structural change + two raced knobs)

**Single-shot fft3d_execute() adopts the chain's k-plane block fusion** (the
round's named seam: benchFFT times execute(), and execute() had stayed on the
r3 global 2,1,0 axis order while the chain has run the gen_r4 blocked
schedule every step since).  Past a gate, execute() now runs blocks of
rpb = (8/gcd(L,8))*L rows -- axis 2 (src -> dst, rows contiguous) then axis 1
(in place) while the block's [r0*L, r1*L) element range is L2-hot -- followed
by the global strided axis 0 in place on the warm dst.  This deletes the
axis-1 full-volume round trip exactly as the chain's blocked regime does.
Per-pass arithmetic is bit-identical to the old order; only the axis ORDER
moves (rounding at the 1e-15 tier; the chain path is byte-untouched and its
outputs cmp-IDENTICAL to control).  Knobs: -DBST_NOEXBLK = the r13 global
order (attribution control), -DBST_EXA0F = the axis-0-first arm below,
-DBST_EXBLK_MIB=n = the gate.

### The finding that shaped the round: the chain's literal order LOSES in execute()
First cut copied the chain schedule verbatim (axis 0 FIRST, then blocked
2+1) with the chain's own 14 MiB gate.  Node race, execute-only (chain=1),
control-first same-core pairs at L=100 B=1: quiet pairs ctl 13240.0/13238.2
vs a0f 14185.5/14249.9 -- **+7% loss, 2/2 quiet pairs**.  Mechanism: in the
chain, the axis-0-first pass reads the previous step's volume WARM and in
place; in execute() the source is COLD, and axis-0-first turns the one
compulsory cold read strided (stride L^2, one 128 B touch per line, nothing
the prefetcher tracks) -- costing more than the blocking saves.  Moving the
blocks first keeps the cold read sequential (axis 2's rows are contiguous)
and pushes the strided pass to the end, on data the blocks just warmed.
The general rule, recorded for the panel: A SCHEDULE RACED IN A WARM-SOURCE
LOOP DOES NOT TRANSFER TO A COLD-SOURCE CALL; re-race the pass ORDER, not
just the blocking, when the source temperature changes.

### The gate race (execute has no c stream; the chain's 14 MiB number is not its number)
Node, execute-only quiet same-core pairs, blocked (blocks-first) vs global,
gate forced per arm; footprint = src+dst = nrows*L*32:
  L=32 B=1 (1.0 MiB):  +1.3/+1.4/-0.2%          -> lean LOSS, stay global
  L=50 B=1 (3.8 MiB):  -0.5/+1.5/-1.1%          -> wash, stay global
  L=64 B=1 (8.0 MiB):  -5.7/-6.3/-1.0/+0.3% and -0.4/-0.5/(-11 dirty)/-2.1%
                        across two batteries     -> WIN
  L=80 B=1 (15.6 MiB): -6.6/-7.2/-7.2/+0.0%     -> WIN
  L=96 B=1 (27 MiB):   -3.1/-2.1/-1.8% (3/3)    -> WIN
  L=100 B=1 (30.5 MiB): 4/7 pairs, bests 13067.9 vs 13132.0 -> wash-to-lean
  L=128 B=1 (64 MiB):  -1.9/-1.4/-2.1/-2.7/-3.3% and one dirty -5.5 (6/6) -> WIN
Gate shipped at 4 MiB (own macro BST_EXBLK_MIB, decoupled from the chain's
BST_MAPFUSE_MAX_MIB=14): blocks L >= 51-ish at B=1, leaves 32/50 on the
untouched path.  The L=100 wash while 80/96/128 win is the r4 story again
(16 MB volume ~ the 24 MB LLC: the deleted round trip is LLC-level there,
DRAM-level at 128).  Quiet-window ship-vs-control bests: 64: 2774.1 vs
2835.0, 80: 4905.0 vs 5041.2, 96: 10090.5 vs 10308.6, 128: 31392.4 vs
32197.0 us.

### Gates (ship default build; sweep + m=2 on wallaby AVX-512 paths per
### gen_layout r9, node runs for repeatability/chain/tryout)
Singles B=1, 48 sizes {2,3,5,7,9,10,11,12,13,14,15,16,17,20,21,23,24,25,27,
28,31,32,33,35,40,47,48,49,50,51,56,63,64,65,71,77,80,96,97,100,101,104,105,
112,113,120,127,128} ALL PASS (spot digits 5.7e-16..1.0e-15, tol 1e-12) --
covers both sides of the new gate, odd-L tail groups in the blocked regime
(51/97/101/105/113/127), every M tail type, the giants.  Batched singles
PASS at all 10 graded shapes (4.5e-16..7.7e-16).  Two-step m=2: 10:1
1.323e-15, 50:4 3.333e-15, 100:1 4.109e-15, 31:16 2.549e-15, 105:1
4.212e-15, 128:1 3.719e-15, 64:1 3.107e-15 (tol 3e-14; the untouched cells
read their r11/r13 digits exactly).  Graded chain 100:1 m=64 on the node:
3.564e-14 (anchor 2.416e-14) -- the exact r8/r11/r13 digits, as a
chain-untouched round must read.  Chain outputs cmp-IDENTICAL to the
control arm at 50:4 m=6 and 10:1 m=50.  Below-gate execute outputs
cmp-IDENTICAL to control at 32/40/50 B=1.  Blocked execute repeatable
(cmp-identical runs, 128:1).  Scalar -march=x86-64 build (blocked branch is
AVX512-gated): singles PASS at {10, 50:4, 64, 100}.  create() ~0 s.
Board-parity tryout reads: 100:1 chain 14260.8 (window; MKL 7922.6 same
window), 12:64 execute PASS + repeatable.

### What did NOT work / was decided by race, with the number
- **Axis-0-first blocked execute (the chain's literal schedule)**: +7% at
  L=100 B=1 (numbers above).  Kept as -DBST_EXA0F for the CLX/SPR re-race.
- **Blocking below ~4 MiB**: L=32 B=1 lean loss (+1.3/+1.4/-0.2%), L=50 B=1
  wash -- the volume is L2-resident-ish already; nothing to delete.
- The r13 harness note stands: tryout.sh's awk cases.txt lookup matches BOTH
  rows at L=10/12 (and now the chain leg silently runs m=1 at 12:64 too --
  the B=64 row is also shadowed).  Chain gates at those cells must be run by
  hand on the node.  The remote map-check '$W' quoting bug also stands.

### Borrowed this round, named
- **gen_layout r3 / gen_pow2 r9 lineage (via my own r4)**: the k-plane block
  fusion is the same machinery; this round's contribution is the
  cold-source ordering finding and the execute-specific gate.
- **gen_batchlane r4 / gen_pow2 r3** (standing): held-lease same-core
  control-first pairing for every keep/kill above.
- **gen_rader r9** (standing): both control arms are flag-disabled builds of
  the same source (-DBST_NOEXBLK / -DBST_EXA0F), never impl_N snapshots.

### Operation count (delta vs r13, execute() only, gated sizes)
FP arithmetic unchanged (same kernels, same group decomposition; axis order
2,1,0 -> per-block 2,1 then global 0).  Deletes one full-volume read+write
(axis 1's round trip to LLC/DRAM) per execute() call at gated sizes; adds
nothing.  Chain, create(), tables: byte-identical.

### What I would do next
1. **Cross-arch re-races** (carried, list grows): BST_NOEXBLK / BST_EXA0F /
   BST_EXBLK_MIB join the knob list on CLX/SPR (CLX's 1 MB L2 shrinks the
   block; the 4 MiB gate and the blocks-first-vs-a0f verdict are ICL-LLC
   numbers).
2. **NT stores for execute()'s one cold dst-write pass** (gen_layout r4's
   recipe, declined for the chain in r5 because blocked custody has no cold
   store stream -- execute()'s first pass DOES have one).  Alignment houses
   the risk: dst offsets are only 64 B-aligned when 8 | off, which odd-L
   sizes break; needs a masked/split epilogue or a gated variant.  Sized at
   ~1 volume of RFO reads per call at gated sizes.
3. **bluestein_cost(L) for gen_planner** (eleventh carry, constants
   unchanged from r13).
4. The 10:1/12:1 fallback gap to fftw3 (~4x) remains group-prologue
   overhead -- class winners' territory, not Bluestein's (r13 verdict
   stands).
