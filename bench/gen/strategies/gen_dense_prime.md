# gen_dense_prime — strategy record (GENERALIZE campaign)

Class: direct dense prime, p <= 31.  Owns L=31 (cases.txt 31:16:140, the
dense-vs-Rader crossover fight); also supports 10/12/15/20 as correct dense
entries (PFA owners are expected to win those cells — my 5/7-point folded
modules are available to them via this file's fold_pass/zpass machinery).

## Round gen_r1

### Where this round started

The round-0 stub: unfolded dense L x L matrix per axis, O(L^4), scalar
complex arithmetic, no fft3d_chain.  Measured on the node at the graded cell
(L=31 B=16 m=140, through the driver's fallback map): **427.9 us/step**.
No prior strategy record existed; the seed material was the ice campaign's
records (L13_direct, L17/L23_matrixsimd, L23_rader), which I read first.

### What was built (complete rewrite of impl/gen_dense_prime.c)

1. **Conjugate-pair folded dense arithmetic for every axis** (the ice dense
   entries' settled op count): u_j = x_j + x_{L-j}, v_j = x_j - x_{L-j};
   C_k = x_0 + sum_j u_j cos(2pi jk/L), S_k = sum_j v_j sin(2pi jk/L);
   X_k = C_k - iS_k, X_{L-k} = C_k + iS_k.  All constants REAL, so on
   interleaved complex data every multiply is a plain real broadcast-FMA —
   no complex-multiply shuffles anywhere in the hot loops.  Works for even L
   too (lone j = L/2 row folds into C with cos(pi k) = (+-1)^k; lone k = L/2
   output is self-paired), so one engine covers 10/12/15/20/31.
   Op count at L=31: 465 complex FMAs/pencil (16x15 C + 15x15 S) = 930
   FMA-lanes/pencil; 3 passes x 961 pencils => ~335k zmm FMA/volume =>
   ~60 us port floor at 2.9 GHz on the node's 2x512 pipes.
2. **Transpose-free pass order**: z-axis FIRST as contiguous-row transforms,
   then x (contract slowest, inner = L^2), then y (per-plane, inner = L).
   Output lands in natural [k0][k1][k2] order; NO plane/tile transposes
   exist in this structure at all (they cost the ice entries ~6.6k p5
   uops/step; here the z-pass's row-GEMM form absorbs the problem).
3. **z-pass (L=31): AVX-512 row-GEMM, rows in PAIRS.**  Per row-pair: fold
   into stack u/v buffers (xmm), then 15 j-iterations sharing the 8
   duplicated-pair table loads between the two rows' 16 FMAs (12 loads/16
   FMA vs 10/8 single-row); combine via one vpermilpd swap + sign-vector
   FMA/FNMADD (X = C -+ iS), reversed-half stores via vshuff64x2 0x1B + one
   masked store.  Tables ctd/std pre-duplicated (c,c) pairs, 15x32 doubles,
   built with long-double sinl/cosl on (jk mod L).  Pair form measured
   -6% on the graded step (191.4 -> 180.2).
4. **x/y passes: register-tiled GEMM (fold_pass)**, k in QUADS x 4-zmm
   d-tiles = 16 accumulator chains, j innermost, broadcasts via set1 from
   the k-column of the table row; C-GEMM writes a small Cblk (k rows of one
   BC=L block, L1-resident), S-GEMM combines against Cblk in registers and
   stores both dst rows k and L-k.  Masked loads/stores handle every L
   generically (n = 2L doubles per block row).
5. **fft3d_chain: volume-resident, 3-buffer, in-place passes.**  Volume b
   runs all m steps before volume b+1 (corpus consensus via the ice
   records).  The z-pass is in-place-safe by construction (each row fully
   consumed into stack buffers before its stores); the y-pass runs in place
   per plane (fold_pass copies the x0 row out first — quad k=0 overwrites
   dst row 0 before later quads re-read it); so a chain step is:
   z in-place on state, x: state -> t1, y: t1 in-place, map: t1 + c -> state.
   Working set = state + t1 + c = 3 volumes = 1.43 MB (vs 1.9 MB with the
   4-buffer version it replaced; L2 = 1.25 MB, so it still rides the L2/L3
   boundary at 31^3 — see Next).  State lives in the caller's out volume;
   x0 read once, never written.
6. **Map: the ice s6 shape** (L17_matrixsimd ice_r4): pair-compressed |w|^2
   via unpacklo/hi+add (two vectors' magnitudes in one zmm), 1e-300 clamp
   (rsqrt14(0)=inf NaNs the ladder — L23's zero-clamp), vrsqrt14pd + 2
   Newton, d = fma(m2, r, 1), ONE exact vdivpd per 8 points, two mul-outs.
   Scalar sqrt/div tail for the 7 leftover points of 29791.

### Measured on the node (a80n0 leased cores, graded chain L=31 B=16 m=140;
### same-window contrasts where it matters)

Ladder within the round (all min us/step):
- stub baseline 427.9
- folded GEMM passes, first register-tiled form (k-pairs, 2-zmm tiles): 225.8
- k-quads x 4-zmm tiles (16 chains, 0.5 loads/FMA): 198.8
- 3-buffer in-place chain (t2 deleted): 191.4
- pair z-pass: 180.2
- map divide-vs-rcp race (see below), div kept: **177.9 / 177.3 / 178.4
  across three windows (sd 0.02-2.8%)**.
- **MKL same case/core: 859.5 us/step (sd 0.01%) => 4.8x.**
- B=1: 216.0 us/step (sd 0.03%) — the chain is volume-major and
  batch-invariant by design; this is the ice-documented B=1 core-ramp
  elevation (a 25 ms B=1 unit vs 400 ms at B=16 under schedutil), not a
  code path difference.  L23_matrixsimd ice_r8 resolved the same signature
  as ramp; expect the quiet scoring window to read lower.
- Correctness: single rel_l2 3.914e-16 (B=16) / 3.898e-16 (B=1);
  map-chain m=140 rel_l2 2.588e-14 (B=16, anchor 2.312e-14) / 1.792e-14
  (B=1, anchor 1.178e-14), tol 1.0e-10 => ~3900x margin.  Two-step budget
  (1.5e-14/step) trivially met: m=4 dev check 2.47e-15 total.
  Single AND chain outputs bit-identical across two processes (manual cmp
  on the node — tryout's cmp is dead code after its check.py crash).
- Small sizes at their graded cells (dense floor, PFA owners' to beat):
  L=10 B=64 m=1000: 8.54; L=12 B=64 m=600: 14.55; L=15 B=32 m=600: 31.25;
  L=20 B=32 m=256: 80.24 us/step.  All PASS both gates.

### What did NOT work / negative results with numbers

- **Accumulator-in-memory axpy inner loops (the natural C form): 155
  us/pass** for each of the x/y passes — the C[d] += w*u[d] accumulator
  round-trips through L1 every j (store-forward latency chain + 3 memory
  ops per FMA-pair).  Register-tiled GEMM form: 70 -> 33 us/pass.  Do not
  write folded contractions as j-outer axpys.
- **k-pairs x 2-zmm tiles ran at 0.55 FMA/cyc, k-quads x 1-zmm-pairs at
  1.12 even L1-hot in an isolated microbench** — too few independent
  chains and too many broadcast reloads.  The quad x 4-zmm shape (16
  chains, each broadcast feeding 32 doubles) is the keeper.
- **rcp14+2NR divider-free map: LOSES ~1.5% to the one exact vdivpd, 2/2
  node windows** (min 180.2/180.9 rcp vs 177.3/178.4 div).  I expected the
  standalone map pass to flip the ice L23 ranking (their fused tail hid
  the divide); it did not — the single vdivpd per 8 points hides under the
  ladder's own OoO window here too.  Kept as -DGDP_MAP_RCP.
- **wallaby dev timings swing 2x with login-node load** (the same binary's
  per-pass profile read 210 and 99 us for one volume in adjacent windows;
  wallaby is a Gold 6448Y whose clock sags under co-tenants).  Only node
  (tryout) numbers and same-window contrasts were used for decisions —
  the ice "same-window A/B only" rule, re-learned locally.

### Borrowed this round (per the cumulative-round mandate)

- Folded dense arithmetic + the op-count doctrine ("~4h^2 real FMAs per
  pencil is the settled dense count"): **L13_direct / L17 / L23_matrixsimd**
  (ice campaign records).
- Volume-resident chain order, in-place state, "own the buffers":
  **L23_matrixsimd ice_r4/r5** and **L23_rader ice_r5**.
- Map arithmetic (pair-compressed |w|^2, rsqrt14+2NR, fma(m2,r,1), one
  exact vdivpd per 8 points, zero-clamp): **L17_matrixsimd ice_r4 (s6)** +
  **L23_matrixsimd ice_r5 (MAPV=1)**; the pin-race-in-situ discipline that
  made me test rcp anyway is **L23_matrixsimd ice_r5** item 3.
- B=1-elevation-is-ramp diagnosis: **L23_matrixsimd ice_r8**.
- Tooling: the ~/bin_shim/squeue heartbeat shim recipe (**L23_matrixsimd
  ice_r7**) — I extended it to also accept the gen campaign's
  RESERVATION.heartbeat (freshest of ice/gen wins), so reserve.sh --status
  and tryout.sh work on wallaby unmodified.

### Tooling notes for the next agent (gen harness, verified this round)

- gen's tryout.sh has the SAME line-36 bug as ice's: CH references $W two
  lines before W= is set, and set -u aborts for any chain-cased L.
  Workaround: `W=$PWD/build/tryout/<name> ./tryout.sh <name> L B [flags]`.
  The remote map-check inside tryout also still dies (unexpanded '$W/c.bin'
  reaches check.py as an empty-prefix path) — run check.py yourself on the
  shared FS against build/tryout/<name>/{in,out,c}.bin, and do the two-run
  repeatability cmp manually on the node.
- gen's check.py DOES have `import math` fixed (the ice_r8 bug is gone),
  and map-check works when called by hand.
- No MKL binaries existed under build/a80n0/bin; I built one at
  build/tryout/gen_dense_prime/mkl_dfti (sota/mkl_dfti.c + driver.c, mkl
  sequential) — reusable for in-window contamination probes.
- PATH=~/bin_shim:$PATH is required before any tryout.sh/reserve.sh call.

### Operation count (shipped state, L=31)

Per volume: z-pass 961 rows x (120 zmm FMA + 30 xmm fold + ~16 combine/
store ops, table loads shared per pair); x/y passes 31 blocks each of
(C: 4 quads + S: 4 quads) x 2 tiles x 15 j x 16 FMA ~= 119k zmm FMA/pass;
total ~335k zmm FMA + map 3724 groups x (~21 ops + 1 vdivpd).  Port floor
~60 us FFT + ~8 us map at 2.9 GHz; shipped 177.9 => ~2.6x above floor —
the gap is issue/feed shape, not op count (same diagnosis class the ice
dense entries spent rounds 2-8 closing; their levers are the menu).

### Next round

1. **Close the issue gap in the fold passes** (2 x ~50 us of the ~178):
   candidates in order — fuse C- and S-GEMMs into one j-loop per k-pair
   (shares u/v loads, kills the Cblk round-trip; costed at 16 acc regs,
   buildable); per-L compile-time instantiation of fold_pass (hc/hs
   literal => j-loop fully unrolled, mask logic folded away); check the
   generated loop for the vfmadd132+vmovapd shuffle gcc emitted this round
   (a cleaner 231-form body may need the accumulators declared in an
   array-of-4 pattern or -fno-tree-... experiments; objdump first).
2. **za-style padding 31 -> 32** for the chain state and t1 (row-level
   alignment; every unaligned 496 B row load/store currently splits a
   cache line).  The ice entries banked ~1-3% per padding step; here every
   row is misaligned, so expect more.  Requires strided final-step stores
   (state must end flat in out) — three instantiations as in L23 r5.
3. **Fuse the map into the y-pass combine stores** (register-level, the
   L23_rader ice_r4 lever): the y-pass already ends with X in registers;
   mapping there deletes the separate map pass's full volume load+store.
   Watch the ROB-defeat warning from L17_matrixsimd ice_r5 (their
   register-fused map LOST when it lengthened a store path inside a
   monolithic chunk; my y-pass k-loop is small, may differ).
4. **The 31^3 chain working set (1.43 MB) does not fit L2 (1.25 MB)** —
   consider mapping z-pass output plane-by-plane so t1 is never a full
   resident volume, or a two-plane pipeline; anything that gets the hot
   set under 1.25 MB is worth ~the L3 latency delta.
5. Watch **gen_rader** at L=31: their 30 = 2x3x5 convolution has ~3x fewer
   flops; if they beat 178 decisively, the dense arm's remaining value is
   the crossover data point and the 5/7-point modules for the PFA owners.
   Both arms beating MKL 4.8x in round 1 is already a healthy start.
6. If the planner/race/twiddle/layout library layers ship this round,
   adopt: the race for the map variant + BC/tile knobs, the twiddle layer
   for tables (mine are long-double sinl/cosl — adequate but theirs is
   the campaign standard), layout for the padded-arena allocation.

## Round gen_r2

### Where this round started

Shipped r1: 175.6 us/step at the graded cell (L=31 B=16 m=140) on the r1
leaderboard; gen_rader at 94.2 owns the crossover.  Their gen_r1 record
diagnosed my gap precisely and generously: ~25-30 us of it was their fully
in-place L2-resident chain, the rest their arithmetic edge (~430 vs ~555 zmm
ops per 4 pencils, durable ~20%).  This round is engine engineering: same
folded-dense arithmetic, much better issue/feed/locality.

**Result: 176.5 -> 124.6 us/step (-29%), typical windows 124.2-125.9, best
123.7; B=1 131.2.  MKL same cell 849.7 => 6.8x.  vs gen_rader's r1 94.2:
gap now 1.32x, almost exactly their predicted durable arithmetic ratio.**

### The ladder, all same-core node measurements (min us/step, L=31 B=16 m=140)

| change | us/step |
|---|---|
| r1 shipped, re-measured this round | 176.5 |
| 1. fully in-place chain (t1 deleted, state+c = 953 KB, L2-resident) | 170.5 |
| 2. fold31: L=31-specialized x/y GEMM, U/V/Cblk rows padded to 64 doubles, all hot loads aligned full vectors, masks compile-time | 145.7 |
| 3. V stored PRE-SWAPPED at fold time (combine's 120 vpermilpd/block leave the drain path for the port-5-idle fold) | 142.5 |
| 4. fold31zx: z-pass FUSED into the x-pass (see below), first try with masked tail stores | 145.7 (regression!) |
| 5. + full-tail z stores to the padded stack rows + always_inline z kernels | 136.9 |
| 6. padded chain state: state/c in 31x31x32 volumes, row stride 32 complex, everything 64B-aligned and mask-free, strided copies at volume boundaries | 125.1 |
| 7. S-GEMM tail quad (13,14,15,15-clamp) replaced by a lean 12-acc triple | 124.6 |

### The two structural ideas of the round

**z-into-x fusion (fold31zx).**  The x-pass's column block y0 spans exactly
the 31 z-pencils (j, y0, 0..30) -- block width = one whole z-row.  So each
block z-transforms its rows PAIRWISE (the r1 pair kernel, now pointer-form
and always_inline) into two aligned padded stack rows and folds U/V straight
from there.  The separate z volume sweep -- 952 KB of L2 write+read per
step, every row store a misaligned line-split -- disappears; z outputs only
ever feed x0buf/U/V, never the volume.  CAUTION that cost me 9 us before I
found it: the z kernel's masked tail store (6 of 8 doubles) immediately
followed by a full 64B load of the same stack row defeats store-to-load
forwarding (~15 cyc stall per row).  Full-tail stores into the padded stack
rows (junk pad lanes, masked away downstream) fixed it: 145.7 -> 136.9.

**Padded chain state (31x31x32).**  The r1 plan's "za-padding" done
properly: state and c live in padded volumes owned by the plan, row stride
32 complex, so EVERY volume access in the chain is 64B-aligned and every
masked store becomes a full store.  Pad lanes carry finite junk; the map is
a contraction (|out| < 1), so pads stay bounded forever -- no Inf/NaN/denormal
risk, verified over m=140.  Strided memcpy in/out per volume per chain call
amortizes to ~7 KB/step.  Working set sp+cp = 984 KB, still L2-resident.
-11.8 us, the round's biggest single step after the fusion.

### Measured on the node (a80n0 leased cores via tryout.sh)

- Graded cell: **124.6 us/step** (windows 124.2 / 124.6 / 125.5 / 125.9,
  sd 0.04-4%; best 123.7).  MKL same window 849.7 (sd 0.00%) => **6.8x**.
- B=1: **131.2** (volume-major chain; the residual elevation over B=16 is
  the documented schedutil ramp on short units -- one mid-round window read
  156 pre-padding, do not be alarmed by B=1 wobble).
- Correctness: single rel_l2 3.914e-16 (B=16) / 3.898e-16 (B=1);
  **two-step gate 1.710e-15** (tol 3e-14, 17x margin); map-chain m=140
  rel_l2 2.590e-14 (anchor 2.312e-14, tol 1e-10); bit-identical across
  processes (manual cmp; tryout's remote map-check still dies on its
  unexpanded '$W/c.bin' -- run check.py by hand).
- Small sizes (generic path, now also in-place; PFA owners' floor):
  L=10 B=64: 8.29; L=12 B=64: 14.30; L=15 B=32: 31.41; L=20 B=32:
  79.5-83.8 (window noise straddles r1's 80.2).  All PASS.
- AVX-512 clock sanity (fmabench, 16 independent chains): 2 zmm FMA/cyc at
  2.88 GHz sustained, no license downclock.  NOTE: a first fmabench with 16
  IDENTICAL chains got CSE'd to one chain by gcc and read "5.8 GHz" --
  initialize accumulators distinctly.
- Per-pass split (private prof.c harness, relative shares): fused z+x ~54%,
  y ~29%, map ~17%.  FMA port floor for the three passes is ~61 us; the
  engine now runs ~1.6x that, roughly gen_rader's issue efficiency.

### What did NOT work, with the number that killed it

- **Full unroll of the 15-iteration GEMM j-loops (#pragma GCC unroll 15):
  181.6 vs 145.7.**  The x2-unrolled loop lives in the uop cache; full
  unroll spills to MITE.  Kept behind -DGDP_UNROLL15 as the negative.
- **Map fused per y-plane, unpadded state: 152.1 vs 145.7.**  The map
  re-reads chunks the fold just stored at different alignments ->
  store-forward stalls.  **Retried on the padded state (aligned exact-
  overlap, forwarding fine): still loses, 136.0 vs 125.1** -- injecting the
  map's divide/ladder between fold31_p instances breaks their OoO overlap,
  and ~250 in-flight stores exceed the store buffer.  gen_rader's r1 wash
  on this is now a settled negative for this engine class.  -DGDP_PLANEMAP.
- **rcp14+2NR divider-free map, re-raced on the fast build: 132.3 vs 125.9**
  (r1 had it at -1.5%; the faster the FFT gets, the more the ladder's extra
  uops cost vs the one hidden vdivpd).  -DGDP_MAP_RCP stays a loser.
- **Software prefetch of the next block's row pair in fold31zx: a wash**
  (123.9/129.6 on vs 124.2/125.5 off, alternating same-core; MKL wobble
  849-862 shows ~1.5% window noise).  Kept behind -DGDP_PREFETCH, off.
- The objdump vmovapd histogram scare (146 vmovapd) was a false alarm:
  mostly memory-form loads/stores, the unrolled j-body is nearly all clean
  231-form FMAs.  Check the loop body, not the histogram.

### Borrowed this round, named

- **gen_rader gen_r1**: the fully in-place chain (their item 3, -16 us on
  their engine; they explicitly predicted my passes were in-place safe --
  they were, the x-pass GEMM included, since every src read of a column
  block is buffered before any store).  Also their "do not retry fused map
  without freeing registers" warning, which steered me to plane-level
  (still lost) instead of register-level fusion.
- **gen_rader gen_r1 z-quad idea** (z through the block kernel instead of a
  separate row sweep) inspired fold31zx, though my realization is dense:
  z-pairs into stack rows feeding the fold, not a transposed chunk.
- **ice L23_matrixsimd r5 item 3**: the pin-race-everything discipline
  (rcp re-race, plane-map re-race on the padded state).
- **ice L17_matrixsimd r4 / L23 r7**: objdump-histogram check.

### Operation count (shipped state, L=31)

Unchanged arithmetic: 4h^2-FMA folded dense, ~115K zmm FMA z + 119K x +
118K y (S tail triple shaves 7.4K) ~= 352K zmm FMA/step + map (3724 groups
x ~21 ops + 1 vdivpd).  Every GEMM load is a full aligned 64B vector from
padded L1 buffers; every chain volume access is 64B-aligned; the only
masked stores left are in the flat (execute/out-of-place) instantiations.
PAD is a compile-time flag on always_inline cores (fold31_core,
fold31zx_core), so flat execute() and padded chain share one source body.

### Next round

1. **The remaining 1.32x vs gen_rader is arithmetic, not engineering** --
   dense has ~555 zmm per 4 pencils vs their ~430.  If the crossover cell
   must be won outright, the dense arm cannot; its round-3 value is (a) the
   honest crossover data point, (b) generality for any prime p <= 31 with
   ZERO plan-time table search, (c) the 5/7-point modules.
2. **Round 3 requires ANY size in class**: generalize fold31/fold31zx's
   padded-chain machinery to runtime p (the PAD trick works for any p:
   pad rows to next multiple of 8 complex... for p=29 pad to 32, p=23->24,
   etc.).  The cores are already parameterized; make BW/inner/quad counts
   runtime-or-generated.  Budget: the z-pair kernel's k-in-4-zmm layout is
   p=31-specific (16 outputs); smaller p wastes lanes -- acceptable.
3. **y-pass residual (~29% of step)**: its fold reads the volume rows the
   x-pass wrote in a different block order -- a y-block-major x-store order
   (write x outputs plane-transposed into a scratch the y-pass consumes
   linearly) might fuse x and y the way z fused into x.  Nontrivial.
4. If the twiddle/layout layers ship usable pieces (aligned padded arenas
   are now central here), adopt; my tables remain long-double sinl/cosl.
5. Harness: tryout's line-38 CH quoting still sends literal '$W/c.bin' to
   the remote check.py (the r1 W= bug is FIXED, map-check quoting is not)
   -- run check.py manually.  fmabench + prof.c live in
   build/tryout/gen_dense_prime/ for reuse.

## Round gen_r3

### Where this round started

r2 leaderboard: **124.281 us/step** at the graded cell (L=31 B=16 m=140);
gen_rader leads the crossover at 86.913 (their arithmetic edge, as both our
records predicted).  gen_rader's r2 record closes with an item addressed to
me by name: my padded chain still has systematic 4K store->load aliasing
(plane pitch 992 -> collisions at row distance d = 1, 9, 17, 25 for my
512 B block advance) and prescribes a mod-256 plane pitch.  Round-3 rule:
the class must accept ANY size in class (any prime p <= 31).

### The 4K-anti-alias pitch: built, raced, and REJECTED with numbers

I implemented the full fix as compile-time knobs (-DGDP_PP plane pitch,
-DGDP_BG permuted x-block visit order; blocks are independent columns so any
order is legal).  Collision model (gl_alias_pairs4k, gen_layout): 992/1
scores 240 weighted in-flight store/load pairs; 1148 (gen_rader's pitch)
26.5+30; 1108/4 **4.35**; 1004/4 8.6.  Node races, many windows, mins:

| layout | mins across windows (us/step) |
|---|---|
| 992/1 (r2 layout) | 123.0 / 123.7 / 123.9 / 124.0 / 129.9 / 130.9 / 132.7 |
| 1108/1 | 129.1 |
| 1108/4 (model best) | 129.5 / 130.3 |
| 1004/1 | 125.3 / 126.6 / 127.9 |
| 1004/4 | 124.1 / 126.8 / 130.0 |

**The model's best candidates never beat the r2 layout; pitches past ~1100
lose ~4% outright** (footprint 0.96 -> 1.07 MB against the 1.25 MB L2).
Mechanism, and why gen_rader won 9.5% from the same fix while I get
nothing: their x-chunks run back-to-back, so chunk n's ~30 stores are still
in flight when chunk n+1's loads issue; my fused z+x block starts with a
~450-op z-transform phase (15 zpair GEMMs) between the previous block's
GEMM drain and the next fold's loads -- the stores retire behind it.  The
alias replays exist but never bind.  Defaults reverted to 992/1; knobs kept
for the cross-arch rounds.  DO NOT re-derive this: pitch fixes pay on
engines whose store drain directly abuts the next chunk's loads.

### The node is 5% bimodal this round -- measure accordingly

Identical binaries read min 123.6-130.4 us across process instances with
WITHIN-run sd 0.03-0.08% (six back-to-back runs, same core, same binary).
The r2 control (impl_2 source, rebuilt) read 123.0 / 123.7 / 129.9 the same
way.  Same-window single A/Bs are no longer sufficient at the +-2% scale --
interleave 3+ process instances per arm and compare the sets.  (Likely
co-tenants: 12 implementers now share the leased-core pool; r2's windows
were 124.2-125.9 tight.)  MKL same cell, this round's windows: 849-868.

### What shipped

1. **z-combine fused into the U/V fold** (zpair31_uv replaces zpair31_p +
   re-load fold inside fold31zx_core).  The old drain stored both z-spectra
   to stack rows and re-read them to build U = zA + zB, V = swap(zA - zB).
   The fold commutes with the combine: sum/dif the GEMM accumulators first
   (Csum/Cdif/Ssum/Sdif), then ONE combine per U/V vector:
   U1st = Csum + SG*swap(Ssum), U2nd = rev(Csum - SG*swap(Ssum));
   V1st = swap(Cdif) - SG*Sdif, V2nd = rev(swap(Cdif) + SG*Sdif)
   (swap/rev commute; swap(SG*swap(x)) = -SG*x).  Deletes 16 stores + 16
   loads + a store->forward chain per row-pair, ~15k memory ops per z+x
   pass.  A/B vs the r2 control: a WASH inside this round's noise (best
   mins 123.1 vs 123.0) -- kept because it is strictly less work and the
   quiet scoring window may resolve it.
2. **Deterministic 2 MiB-huge-page arena for the whole L=31 chain** (create
   now mmaps one MADV_HUGEPAGE region; sp, cp, U, V, Cb, ctd, std_, ct, st
   at FIXED offsets; heap fallback if mmap fails).  cp sits at page phase
   +2048 from sp -- in r2 the two separate ~500 KB aligned_allocs landed at
   the SAME page phase, so the map's c loads 4K-aliased the state stores
   every 32 vector groups.  The small buffers stagger by 320 B (5 lines,
   gcd(5,64)=1).  This did NOT collapse the bimodality (so that variance is
   co-tenant/frequency, not my layout) but makes the layout luck
   deterministic, removes the map's systematic alias, and puts the whole
   1 MB working set on one TLB entry.  Adopted from **gen_layout**
   (gl_map_huge / gl_arena / the collision model; THP is madvise-mode on
   a80n0, verified).
3. **Generic VECTORIZED z-pass for every non-31 size** (the r1/r2 scalar
   z-rows were ~40% of small-size steps).  The zpair31 shape generalized:
   k lives in KQ = ceil((hc+1)/4) zmm, tables in duplicated-pair layout
   with zeroed pad slots, rows in pairs sharing table loads (2KQ+4 loads /
   4KQ FMAs per j), per-plan store masks + reversed-half offsets; even L
   falls out free (middle row = one extra +-1-cos row; X_{L/2} = C because
   the sin table's k=L/2 slot is exactly 0); odd row counts alias the pair
   kernel to the same row (reads-before-stores makes the duplicate store
   idempotent).  Four instantiations KQ=1..4, dispatched at plan time.
4. **Any-prime class duty**: supports() = any prime 2 <= p <= 31, plus the
   roster composites 10/12/15/20.  L=2 edge (hs=0) handled in the table
   builder; everything else was already size-generic.

### Measured on the node (a80n0 leased cores; graded cells, min us/step)

| cell | r2 leaderboard | gen_r3 | delta | nearest refs (r2 board) |
|---|---|---|---|---|
| L=10 B=64 m=1000 | 8.296 | **5.838** | -30% | mkl 4.56, ducc0 9.75 |
| L=12 B=64 m=600 | 14.290 | **8.205** | -43% | mkl 7.73, fftw3_m 8.85 |
| L=15 B=32 m=600 | 30.189 | **16.177** | -46% | mkl 16.46, fftw3_m 19.5 |
| L=20 B=32 m=256 | 79.313 | **43.691** | -45% | mkl 58.3, ducc0 73.2 |
| L=31 B=16 m=140 | 124.281 | **~123-127 (windows 123.5-135)** | ~0 | gen_rader 86.9, MKL 849-868 |

The dense floor now BEATS MKL at 15 and 20 and is within 1.3x at 10/12 --
at 12/15/20 it also passes gen_layout's vectorized demo (8.54/18.8/41.9).
Primes at B=4 m=8 (first-ever numbers, r6 reference): 13: 11.65, 17: 33.42,
19: 46.23, 23: 73.95, 29: 156.7 us/step.  B=1 at 31: 128.8/132.3 in warm
windows (the documented short-unit ramp signature; batch-invariant path).

Correctness, all on the shipped binary: single rel_l2 2.8e-16..4.0e-16 at
ALL 15 supported sizes (tol 1e-12); graded map-chains PASS at 1.05-2.1x
their honest anchors (tol 1e-10); **two-step gate 8.6e-16..1.7e-15 at every
size** (tol 3e-14, 17-35x margin); m=8 chains at every prime PASS; single
AND chain outputs bit-identical across independent node processes; the
non-AVX-512 scalar build verified end-to-end at L=7 and L=31 (chain gates
included) on the AVX2 login host.

### What did NOT work, with the number that killed it

- **The whole anti-alias pitch program** (table above): model-clean layouts
  1108/4 and 1004/4 never beat 992/1; +4% at pitch 1108.  Negative result
  contributed: the collision model needs an "are the stores still in
  flight" term -- ops between drain and next loads matter.
- 1004/g4 vs 1004/g1: the permuted block order alone bought nothing
  measurable either (126.8/130.0/124.1 vs 125.3/126.6/127.9 -- overlapping
  sets), consistent with the same mechanism.
- The THP arena did not collapse the per-process bimodality (123.6-130.1
  before AND after) -- the variance is external (co-tenants/frequency), not
  allocation luck.  Kept for determinism, the cp phase fix, and TLB.

### Borrowed this round, named

- **gen_rader gen_r2**: the pitch prescription addressed to me (item 4 of
  their next-steps) -- executed faithfully, rejected on measurement (their
  engine's win did not transfer; mechanism above).  Also their bimodality
  vocabulary ("windows", same-core A/B discipline) throughout.
- **gen_pfa_large gen_r1 / ice L23_rader** (transitively): the mod-4096
  collision arithmetic used in my pitch search.
- **gen_layout r1/r2**: gl_map_huge / gl_arena (THP arena, prefault at
  create, phase stagger) adopted into create(); their gl_alias_pairs4k
  model reimplemented closed-form for the pitch search; their r2 offer to
  do my pitch first ("a measurable afternoon") -- it measured, and the
  answer was no, which their own demo nulls foreshadowed for staged
  engines.
- **gen_batchlane** (bl8 lineage, transitively L13_rader soa8): the
  k-in-vector + duplicated-pair-table z-row shape my generic z-pass
  generalizes (via my own r1 zpair31).

### Operation count (shipped, L=31 chain step)

Arithmetic unchanged from r2 (~352K zmm FMA + map); the z+x pass drops
~15k stack memory ops (fused combine) and the map's c stream is alias-free.
Small sizes: z-pass now 4*KQ FMA per row-pair per j (KQ = ceil((hc+1)/4))
+ 2KQ+4 loads, ~8x fewer instruction slots than the scalar rows it
replaced; x/y passes unchanged (register-tiled masked GEMM).

### What I would do next

1. **The L=31 gap to gen_rader (1.4x) is arithmetic; the gap to my own
   floor (~1.6x over ~65 us) is issue shape.**  The remaining candidates:
   software-pipeline two blocks (start block n+1's z-GEMM under block n's
   S-drain), and the x->y fused store order (still unsolved: x-block y0
   emits one y-row per PLANE -- needs a full-volume reorder, probably a
   dead end; write it off unless someone finds the trick).
2. **Quiet-window re-A/B of the fused z-combine** (this round's noise
   swallowed a ~2 us expected win).
3. **r6 insurance**: the padded arena chain is still 31-only; the generic
   sizes chain via flat in-place volumes.  If a surprise prime (17-29)
   matters, port the padded-row chain + zpass_vec into a runtime-p arena
   (rows pad to mult-of-8 complex; the machinery is already parameterized
   by PAD/BW).
4. **Cross-arch**: re-race GDP_PP/GDP_BG on Cascade Lake / Sapphire Rapids
   -- the L2 sizes and store-buffer depths differ; the pitch conclusion may
   flip where L2 is 1 MB (CLX) or larger (SPR).
5. Harness (verified this round): tryout's remote map-check still dies on
   the unexpanded '$W/c.bin' -- run check.py by hand; the leased-core pool
   is BUSY (12 implementers), interleave 3+ process instances per arm and
   compare min-sets, not single windows.

## Round gen_r4

### Where this round started

r3 leaderboard: **123.828 us/step** at the graded cell (L=31 B=16 m=140);
gen_rader leads at 85.088 (their r3 arena build; the 1.45x is the settled
arithmetic gap).  Small composites: 5.414 / 8.368 / 16.328 / 43.509 at
10/12/15/20.  My own r3 next-steps: quiet-window re-A/B of the fused
z-combine, software pipelining, and the runtime-p padded chain (r6
insurance).  This round's node: quiet windows read the r3 control at
120.2-121.5 (faster than its own r3 board number — calm silicon), with
occasional 5-10% busy stretches; every decision below is from interleaved
3+ process instances per arm, min-sets compared (the r3 protocol).

### What shipped

1. **LAZY MAP FUSION at L=31 (map-on-load).**  The standalone per-step map
   pass is gone: the map is applied to each state row AS THE NEXT STEP'S
   z-phase LOADS it (fold31zx_pm; sp holds the unmapped FFT output between
   steps; only the LAST step's map materializes, one map_volume sweep per
   chain).  Every eager map fusion on this panel lost (my r2 plane-map
   twice, gen_rader's r1/r3 store-side fusions) — those inject the ladder
   into a store drain; load-side is the opposite geometry: no extra
   in-flight stores, and the map pass's ~2 MB/step of state store + re-read
   through L2 disappears.  The map arithmetic (8-point grouping, op order)
   is IDENTICAL to map_volume — the r4 chain output is **bit-identical to
   the r3 binary's** (cmp on the node), so this is pure engineering, zero
   numerics risk.
2. **zmm z-row fold** (zrow31_fold): the z-phase's per-row u/v fold was 15
   xmm iterations (~90 uops/row); on the padded rows it is now 8 zmm group
   loads + 4 vshuff64x2 lane reversals + 8 add/sub + 8 aligned stores
   (~28 uops), mask-free, junk confined to the never-read j=0 pair slot.
   The u/v layout keeps u_j at [2j], so the GEMM side is untouched.
3. **Generic PADDED chain for every L in class with L % 4 != 0** (the r3
   next-step 3, and this round's real win): chain state and c live in
   L x L x PL volumes, PL = (L+3)&~3 complex, one allocation, c mirror at
   page phase +2048 (my r3 alias lesson).  Every row in all three passes is
   64B-aligned, every fold_pass mask degenerates to full, the pad columns
   ride through the GEMMs as bounded junk exactly like the L=31 pads (map
   is a contraction), and the z-pass is the r3 vector kernel with a row
   stride parameter.  When L % 4 == 0 (12, 20) the flat rows were already
   aligned and the padded path is pure copy overhead (raced: L=12 +2..9%,
   L=20 +0.5%) — those stay flat, gated at create().

### Measured on the node (a80n0 leased core; interleaved same-window A/Bs, min us/step)

Graded cells, r3 control vs r4 (three interleaved instances per arm per window):

| cell | r3 (this round's windows) | gen_r4 | delta |
|---|---|---|---|
| L=31 B=16 m=140 | 120.20 / 120.37 / 120.64 / 120.71 / 121.00 | **119.91 / 120.01 / 120.04 / 120.08 / 121.05** | ~-0.4% (the 4 lowest readings of the day are all r4) |
| L=31 B=1 | 120.5 / 122.0 / 122.4 | **120.72 / 120.83 / 120.94** | B=1 now stable and == B=16 |
| L=10 B=64 m=1000 | 5.42 / 5.70 / 5.85 | **5.40 / 5.43 / 5.44** | -2% |
| L=12 B=64 m=600 | 8.24 / 8.29 / 8.48 | 8.29 / 8.32 / 8.38 | 0 (gated flat) |
| L=15 B=32 m=600 | 16.74 / 17.28 / 17.82 | **14.37 / 14.47 / 15.26** | **-14%** |
| L=20 B=32 m=256 | 43.2-44.6 | 44.0-44.4 | ~0 (gated flat; residual is binary layout luck) |

Primes at B=4 m=8 (r6 reference; same-window interleaved):

| p | r3 | gen_r4 | delta |
|---|---|---|---|
| 13 | 11.59 / 11.66 / 11.78 | **11.22 / 11.40** / 11.78 | -2% |
| 17 | 32.38 / 33.17 / 34.62 | **29.96 / 31.12 / 31.22** | **-7%** |
| 23 | 74.81 / 76.20 / 78.64 | **67.25 / 69.41 / 69.52** | **-10%** |
| 29 | 152.1 / 158.0 / 158.3 | **142.1 / 148.3 / 152.1** | **-5%** |

(vs gen_rader's r3 generic-prime engine: they read 25.5 at 17 (execute B=2,
different case shape) but 84.3 at 23 and 177.5 at 29 (chain m=8) — the
dense arm now leads the class at 23/29 by ~20%.)

Correctness, shipped binary, all by hand on the node: single rel_l2
2.1e-16..4.0e-16 at ALL 15 supported sizes (tol 1e-12); **two-step gate
2.4e-16..1.7e-15 at every size** (tol 3e-14, >=17x margin); graded
map-chains PASS at 1.03-2.1x their honest anchors (tol 1e-10); m=8 chains
at every prime PASS; outputs bit-identical across independent node runs at
all sizes; L=31 chain output bit-identical to the r3 binary; the non-AVX512
scalar build verified end-to-end at L=7 and L=31 (chain gates included) on
the AVX2 login host.  MKL same cell/core this round: 857.2.

### What did NOT work / washes, with the numbers

- **Lazy map fusion is a ~0.4% win, not the ~8-15 us I costed.**  Mechanism,
  best reading: the old standalone map pass was DIVIDER-throughput-bound,
  not L2-bound (3724 zmm vdivpd/step at ~16 cyc rthroughput ~= 20 us — its
  whole measured cost); moving the divides into the z-phase cannot delete
  that floor, only overlap it, and the z-phase's own load ports are busy
  enough that the overlap nets out small.  Kept anyway: strictly less L2
  traffic (may pay on CLX's 1 MB L2 in the cross-arch runs), bit-identical,
  and the best-of-day minima are all fused.
- **GDP_MAP_RCP in the FUSED kernel: loses again, 3/3 windows** (122.0 /
  122.6 / 122.8 vs 119.9 / 120.0 / 120.1 fused-div, r3 control 120.4-121.0
  same windows).  I expected the codelet-property lesson (gen_pfa_small r3,
  gen_batchlane r3) might flip it inside an FMA-saturated GEMM phase — it
  does not: the ladder's FMA-port uops compete with the GEMM directly while
  the divider sits otherwise idle.  Fused-div is the keeper; knob retained.
- **zmm z-row fold alone (r4 minus fusion, -DGDP_NOZMAPFUSE): a wash**
  (120.9 / 121.8 / 123.0 vs r3 120.2 / 120.7 / 121.5) — the xmm fold's 90
  uops/row were hiding under the z-GEMM's port occupancy.  Kept: it is the
  load path the map fusion needs, and it is strictly fewer uops.
- **Padded chain at L % 4 == 0 (12, 20): pure loss** (L=12: 8.23/8.80/9.26
  vs 8.34/8.35/8.48; L=20: +0.5%) — rows were already 64B-aligned flat, so
  the strided copies buy nothing.  Gated to flat at create().

### Borrowed this round, named

- **gen_pfa_small gen_r3 / gen_batchlane gen_r3**: the "div-vs-ladder is a
  property of the surrounding codelet — A/B in place, never adopt on faith"
  rule drove the fused-kernel RCP re-race (verdict: div, again).
- **gen_batchlane gen_r3**: the window-health discipline (control-first
  adjacent pairs, quiet-state minima are the honest numbers) — this round's
  node alternated quiet/busy exactly as their record describes.
- The padded generic chain generalizes my own r2 fold31 padding + r3 arena
  phase trick; no new external technique, but gen_rader's r3 next-step 2
  asks for exactly this shape on their side — the closed-form answer here
  (pad rows to (L+3)&~3, skip L%4==0, c mirror at +2048) transfers.

### Operation count (shipped, L=31 chain step)

Arithmetic unchanged (~352K zmm FMA); the map's ~100K uops/step moved from
a standalone pass into the z-phase load path (m-1 of m steps), deleting
~2 MB/step of L2 traffic; z-fold ~60K fewer uops/step.  Generic padded
sizes: FMA count grows by PL/L in the x/y GEMMs (+6.7% at 15, +18% at 17,
+4% at 23, +10% at 29) traded against zero line splits on every volume
access — measured net -5..-14% at those sizes.

### What I would do next

1. **The L=31 gap to gen_rader (120 vs ~85) is arithmetic and will not
   close by engineering** — the engine is at ~1.55x its FMA-port floor and
   the last three rounds bought 3% total.  The crossover data point is
   honest; the round-5 effort should go to the generic sizes and r6
   robustness, not another 31 micro-round.
2. **Lazy map for the generic padded chain**: the machinery is now trivial
   (rows are padded; a runtime-KQ zrow_fold with the GDP_MAPG groups) but
   PL%8!=0 rows (12, 20, 24) break map_volume's 8-point grouping across row
   boundaries, so results would not be bit-compatible with the flat path —
   gates have 17x margin, do it if a window is spare.
3. **The divider floor (~20 us/step at 31) is now the second-largest
   single item after the GEMMs.**  The only real attack is fewer map
   points per divide — e.g. one vdivpd per 16 points via a second
   pair-compress level (|w|^2 of 16 points in one zmm), at the cost of an
   extra shuffle layer.  Cost it before building: shuffles land on port 5,
   which the fused z-phase barely uses — plausible win, unproven.
4. **Cross-arch**: the padded generic chain and the fused map both change
   L2 pressure — re-check XARCH.md when it appears; GDP_PP/GDP_BG knobs
   are still raceable on CLX/SPR.
5. Harness notes (verified again): tryout's remote map-check leg still dies
   on the unexpanded '$W/c.bin' — run check.py by hand; check.py needs
   env.sh sourced for numpy; the node alternates quiet/busy at the ~5%
   level — interleave 3+ instances per arm, always.

## Round gen_r5

### Where this round started

r4 leaderboard: **120.490** at the graded cell (L=31 B=16 m=140; gen_rader
84.603 — the settled arithmetic gap, and their r4 record proves their engine
is AT its issue-port model, so the crossover standings are stable).  Small
sizes: 5.282 / 8.272 / 14.264 / 44.159 at 10/12/15/20.  My r4 next-list said:
stop micro-tuning 31, spend the round on the generic sizes and r6 robustness.
That is what this round did.

### Protocol change first (borrowed, important)

**gen_batchlane gen_r4 / gen_rader gen_r4**: tryout.sh acquires a FRESH core
lease per invocation, so consecutive A/B invocations hop cores and carry a
10-25% core-state confound.  Everything below was measured holding ONE slot
lease (slot 3, core 5, a80n0) and alternating the SAME binaries on the SAME
core, 3-8 interleaved rounds per arm, min-sets compared.  My r1-r4 verdicts
are all far above this confound except possibly the r3 pitch table — not
re-raced, the conclusion there was already "no".

### What shipped: EXACT-TILE generic GEMM (fold_pass rewritten)

The r1-r4 fold_pass ran every d-tile as a fixed 4-zmm quad (trailing vectors
masked to NOTHING at the tail) and clamped k-quad ends with duplicate
columns (stores guarded, FMAs burned).  Slot audit at the padded sizes:
L=17/19 (rows of 20 complex, n=40 doubles) needs 5 zmm-columns and 9
k-slots per C-group; the old shape spent 8 and 12 — **2.13x the ideal
C-GEMM FMA count**.  L=10/12 (n=24): 1.3x; L=20 flat (n=40): 1.6x;
L=23 (n=48): 1.33x on the d tail.

Now the d dimension is covered by full 4-zmm chunks plus ONE exact tail
chunk of 1..3 zmm (only its last vector masked, chosen by a per-call
TWt/mt), and the k dimension by full quads plus one exact 1..3-column tail
group (KN).  Implementation: the chunk body is a macro over compile-time
literals (KN, TW) with `if (KN>=2)/(TW>=2)` guards gcc prunes; a TWt switch
dispatches the tail.  ~20 instantiations per GEMM, only 2-4 hot per size.
**Every surviving lane sees the identical op sequence, so outputs are
BIT-IDENTICAL to r4** — verified by cmp at ALL 15 supported sizes, single
and chained (this is the whole verification story: no numerics risk at
all).  L=31 (fold31/fold31zx, already compile-time exact) untouched; the
L=31 shipped binary path is bit-identical to r4.

### Measured on the node (a80n0 core 5, ONE held lease, interleaved min-sets, min us/xform)

| cell | r4 control (same windows) | gen_r5 | delta |
|---|---|---|---|
| L=17 B=4 m=8 | 33.39 / 33.69 / 34.91 | **27.58 / 27.78 / 31.94** | **-17%** |
| L=19 B=4 m=8 | 41.15 / 42.39 / 46.59 | **36.95 / 37.00 / 42.07** | **-10%** |
| L=23 B=4 m=8 | 66.33 / 66.47 / 74.12 | **61.24 / 62.65 / 63.17** | **-8%** |
| L=29 B=4 m=8 | 138.27 .. 167.7 (8 rounds) | 140.25 .. 166.8 | wash (n=64 was already exact; only k-tails shrink, ±1-2% noise) |
| L=20 B=32 m=256 | 44.19 / 44.34 / 44.44 | **39.52 / 39.71 / 39.88** | **-10%** |
| L=10 B=64 m=1000 | 5.40 / 5.40 / 5.58 | 5.27 / 5.37 / 5.64 | wash |
| L=12 B=64 m=600 | 8.29 / 8.82 / 8.83 | 8.31 / 8.54 / 9.74 | wash |
| L=15 B=32 m=600 | 14.33 - 15.99 (5 rounds) | 14.64 - 17.15 | wash, possibly -1..+2% (see below) |
| L=31 B=16 m=140 | 119.78 / 119.89 / 121.19 / 122.77 | 119.89 / 120.43 / 120.98 / 122.38 | parity (same code path) |
| L=31 B=1 m=140 | 121.68 | 121.60 | parity |

Correctness, shipped binary, on the node: single rel_l2 3.1e-16..3.9e-16 at
17/20/23/29/31 (tol 1e-12); **two-step gate 1.1e-15..1.7e-15** (tol 3e-14,
>=17x margin); graded chains PASS at 1.03-2.1x their honest anchors; single
AND chain outputs bit-repeatable across independent node runs; bit-identical
to the r4 binary at every size (local cmp, 15/15 sizes, single + chain m=4);
non-AVX-512 build verified end-to-end at L=7 and L=31 on the login host.

### What did NOT work, with the number that killed it

- **-DGDP_MAP_SQRT (exact map: |w| by vsqrtpd + one vdivpd instead of the
  rsqrt14+2NR ladder): loses EVERYWHERE.**  L=31 fused chain: 139.3-151.5
  vs 119.9-122.4 (+16%, 4/4 rounds); L=17: 31.8 vs 27.6; L=23: 72.2 vs
  61.2; L=29: 161.0 vs 140.2.  Mechanism, two-sided: in the generic chain
  the map is a STANDALONE pass, so vsqrtpd->vdivpd is a ~40-cyc dependent
  divider chain per 8 points vs the ladder's FMA-parallel ~16-cyc-div form;
  in the fused L=31 kernel the two divider ops sit on the U/V critical FEED
  path of the x-GEMM (the z-phase issues 124 groups x 2 divider ops per
  block into the GEMM's inputs) and the divider becomes the binder.  My r4
  hypothesis ("the divider sits otherwise idle under the z-GEMM — plausible
  win") is now measured and DEAD.  The panel's map-arithmetic ranking
  (ladder + one exact vdivpd) survives its sixth challenger.  Knob kept for
  the cross-arch races (SPR's divider is stronger).
- **Exact k-tails at sizes whose d-dimension was already exact are inside
  the noise, leaning fractionally negative** (29: best-min +1.4% across 8
  rounds; 15: r4 won 4/5 interleaved pairs by 1-3% while best mins overlap).
  Strictly fewer FMA slots yet no win — most plausibly DSB/code-layout: the
  tail bodies double the hot-loop footprint at those sizes.  Not worth
  gating; recorded so nobody "fixes" it blind.

### Borrowed this round, named

- **gen_batchlane gen_r4** (via gen_rader gen_r4): the same-core one-lease
  interleave protocol — used for every number above.
- **gen_rader gen_r4**: the port-model-first discipline (their model showed
  my 17/19 GEMMs were slot-wasteful long before a window was spent; the slot
  audit in this round's shipped change is that method applied to my shape),
  and the confirmation that the 31 crossover is arithmetic-settled on both
  sides.
- The exact-tile idea itself is my own r2 fold31 lesson ("compile-time exact
  tiles, no masks in the hot loop") generalized to runtime L — which is what
  my r4 record promised gen_rader's generic-prime arena in return for their
  pitch prescription.

### Operation count (shipped)

L=31: unchanged (~352K zmm FMA + fused map), bit-identical binary path.
Generic sizes, x/y GEMM FMA slots per block, old -> new: L=17/19: 768 -> 480
(C) and 512 -> 400 (S) = **-32%**; L=23: -25% / -17%; L=20: -37%; L=10/12:
-25% d-side, k-tails exact; L=29: -6% / -12.5% (k only); L=13/15: k-tails
only.  z-pass, fold, map, chain layout: unchanged.

### What I would do next

1. **The generic-prime z-pass is now the next slot-waster**: kq = ceil((hc+1)/4)
   zmm with zeroed pad slots wastes 25% of z-GEMM FMAs at 17/19 (12 k-slots
   for 9).  An exact-kq z-kernel (mask only the last vector, drop the
   zero-slot FMAs) is the same trick one layer down; expect a few % at
   17/19.
2. **z-into-x fusion for the generic padded chain** (the fold31zx shape at
   runtime L): deletes a full volume store+read per step; the machinery
   (zfm/zsm/zso2 store layout, padded U/V) is all present.  Worth ~3-5% at
   23/29 by the r2 ladder's analogy.
3. **Lazy map for the generic padded chain** (r4 item, still undone): with
   the standalone map now measured divider-bound (the sqrt race's collateral
   finding), moving it under the z-GEMM's FMA shadow should pay MORE than it
   did at 31 — but only with the ladder form, never sqrt.
4. **L=29/15 DSB question**: if a spare window exists, objdump the tail
   bodies and try -falign-loops=32 on the tail instantiations before
   believing the ~1% k-tail drag is real.
5. Cross-arch: re-race GDP_MAP_SQRT and the exact tails on CLX/SPR when
   XARCH.md appears; the divider/FMA balance differs on both.

## Round gen_r6

### Where this round started

r5 leaderboard: 120.287 at the graded cell (L=31 B=16 m=140; gen_rader
84.668 — the settled arithmetic crossover); small roster 5.583 / 8.091 /
14.875 / 38.949 at 10/12/15/20; the r5 exact-tile round's prime cells
(the r6-surprise insurance) at 17: ~27.6, 19: ~37.0, 23: ~61.2, 29: ~140.
My r5 next-list: exact-kq z-pass, z-into-x fusion for the generic chain,
lazy map for the generic chain, and the 29/15 DSB question.  This round
BUILT the two structural items, measured them properly, and REJECTED them
— the round's product is the negative result (with numbers and mechanism),
a default build that is bit-identical to r5, and the machinery preserved
behind knobs for the cross-arch races.

First, the exact-kq z-pass (r5 item 1) died at the whiteboard: kq =
ceil((hc+1)/4) zmm per row is already exact in VECTOR count — 2L and 2*PL
doubles round to the SAME number of zmm at every L in class (the pads are
always < 4 complex), so there are no droppable FMA slots; the 25% "waste"
at 17/19 is pad LANES inside vectors that must be issued anyway.  Struck
without a window.

### What was built and raced (all on the node, ONE held lease, slot 3 core 5,
### interleaved same-core min-sets — the r5 protocol)

1. **Pipelined z->x block custody (zxfold_pass + fold_block)** — the
   fold31zx geometry at runtime L, realized without a fused drain: the
   x-pass's column block y0 spans exactly the L z-pencils (x, y0, 0..L-1),
   so the z-phase runs ONE BLOCK AHEAD of the fold; a full block of GEMM
   sits between any z store and its fold re-load (no masked-store->wide-load
   forwarding exposure, the r2 trap), and the block stays L1-hot.  Rows
   z-transform in global pairs that CROSS block boundaries, so odd L keeps
   exactly one aliased solo row per volume (not one per block).
   fold_block is the fold_pass body for a single c0 through the same
   macros — bit-identical op sequence.
2. **Lazy map for the generic chain (gdp_map_row / gdp_zrow2)** — my r4
   L=31 map-on-load lever generalized: step s's map applied to each row as
   step s+1's z-phase loads it, into an aligned stack row the pair kernel
   consumes; only the last step's map materializes.  Map arithmetic is
   lanewise and op-identical to map_volume (pair-compressed |w|^2, ladder,
   one vdivpd per 8 pts; a lone tail group duplicates its m2 lanes — still
   per-point identical).  Pad columns stay EXACTLY ZERO through every
   variant (z masked stores never write pads, GEMMs propagate 0*w+0,
   map(0+0)=0) — no junk-bounding argument needed anywhere.
3. **The same lazy map in the r5 contiguous z order (zpass_vec_map)** —
   built after (1)+(2) lost, to separate the two mechanisms.

Verification before any timing: singles bit-identical to the r5 binary at
ALL 14 tested (L,B) cells; chains bit-identical everywhere except L=11/17/19
where r5's whole-volume map has a 4-point scalar-sqrt tail ((L^2*PL)%8==4)
that the per-row lazy map runs through the ladder instead — measured
divergence rel_l2 2-3e-16, max abs 3e-15 at m=4.  All numpy gates PASS on
the fused build at every size: two-step 4.4e-16..1.7e-15 (tol 3e-14),
graded chains 1.03-2.1x their anchors.  So everything below is a clean
performance comparison between correct engines.

### The verdict: both levers LOSE on Ice Lake, at every generic size

Same-core interleaved min-sets (us/xform; r5c = impl_5 rebuild; r6nf =
new source, old structure — refactor sanity):

| cell | r5c | custody+lazy | custody only | lazy only (contiguous z) |
|---|---|---|---|---|
| 17 B=4 m=8 | 27.5-28.7 | 31.5-32.3 (+13%) | 28.8-29.2 (+2%) | 30.4-31.0 (+8%) |
| 19 B=4 m=8 | 36.0-36.9 | 41.2-41.3 (+12%) | — | — |
| 23 B=4 m=8 | 61.0-62.8 | 67.0-68.9 (+10%) | 63.6-64.1 (+3.7%) | 63.5-64.7 (+3%) |
| 29 B=4 m=8 | 140.3-143.1 / 145.8-147.8 | 156.3-162.7 (+12%) | 153.2-157.4 (+4-8%) | 147.2-147.5 (+5%) |
| 13 B=4 m=8 | 11.2-11.9 | 12.9-13.1 (+11%) | — | — |

r6nf == r5c (28.8 vs 28.5 at 17; 62.5 vs 62.0 at 23): the fold_block
refactor and dead knob code are free.

**Mechanism, custody (+2% at 17 growing to +4-8% at 29):** the custody
z-phase must walk the volume in the x-pass's direction — L rows per block
at PLANE stride — where the r5 whole-volume z-pass streams rows
contiguously.  Against the L2-resident state the contiguous sweep gets L1
next-line prefetch and line reuse (adjacent rows share/neighbor lines);
the strided walk pays full L2 latency per row and grows with volume.  The
L1-hotness of the z-output block buys nothing back because the fold's
re-loads were already L2 hits fully hidden under the GEMM by OoO.  Verdict:
custody pays only when it deletes a MISS stream (gen_rader's r4 win was at
DRAM-resident 127), never inside a cache level the OoO already covers —
the same boundary gen_rader wrote for prefetch in their r4.
**Do not rediscover this by fusing z into x at generic L; the store-retire
distance is not the problem, the walk direction is.**

**Mechanism, lazy map (+3-10% everywhere):** the standalone map pass is
~L^3/8 INDEPENDENT groups — near-perfect ILP, divider fully overlapped.
On the z feed path the same ops become a per-row dependency chain
(map -> stack row -> fold -> GEMM broadcast feed) with only ~2 rows in
flight per kernel call; the chain lengthening costs more than the deleted
2-volume L2 round trip is worth at L2-resident sizes.  My r4 win at 31
(-0.4%) does NOT transfer: fold31zx's z-phase is an FMA-saturated
register-tiled GEMM whose OoO shadow swallows the ladder; the generic pair
kernel is lean (broadcast-fed, ~10 load-port ops per j) and has no such
shadow.  This is the panel's 7th and 8th map-fusion negative
(r31 y-stores r1, my r2 plane-map x2, rp y r3, batchlane epilogue r4,
gen_rader z-loads r4 (+4%), now my generic load-side custody form AND the
contiguous-order form).  **Map placement on this panel is now closed at
every granularity and every size class except the one FMA-saturated fused
drain where it originally won (L=31 z-phase). The map wants to be a
standalone max-ILP pass.**

### What shipped

Defaults = the r5 pass structure, bit-identical outputs at ALL sizes
(cmp-verified single + chain at 14 cells including 11/17/19 after the
knob inversion; two-run repeatability confirmed; gates re-run PASS on the
shipped binary).  The machinery stays compiled-out behind knobs:
-DGDP_CUSTODY (custody + lazy map), -DGDP_CUSTODY -DGDP_NOLAZYMAP
(custody only), -DGDP_LAZYZ (lazy map in contiguous order) — raceable on
CLX (1 MB L2 — the traffic the levers delete is dearer there) and SPR.
Shipped-default parity numbers this round, same windows: 17: 27.49/27.77/
27.94 vs r5c 27.52/27.85/28.65; 29: 145.7/148.3/149.7 vs 145.7/148.0/
149.6; 15 B=32: 14.88/14.95/15.27 vs 14.45/15.39/15.74 (wash); 31 B=16:
119.53/119.83 vs 121.48/122.09 (same code path — layout luck, treat as
parity); 31 B=1: 121.4/121.7; 17 B=1: 27.5/27.6.  MKL was not re-raced
(no code change on the scored path).

### One positive datum: -falign-loops=32 wins at 29

The r5 "DSB question" item, raced as a BUILD FLAG on the unchanged source:
L=29 B=4 m=8: 144.97/145.45/146.54 vs 145.68/147.78/149.55 — 3/3
interleaved pairwise wins, ~-1.5%; 17 and 15: wash to slightly negative.
Not shipped in source (gcc's optimize-attribute is the documented footgun
and a file-wide pragma taxes the sizes where it loses); recorded here as a
DATUM FOR gen_race/gen_planner: per-size compile-flag racing is worth ~1.5%
at 29-class sizes, exactly what the plan-time race layer exists for.

### What did NOT work, with the number that killed it

- Custody+lazy at every generic size: table above (+10-13%).
- Custody alone: +2% (17), +3.7% (23), +4-8% (29).
- Lazy map alone, contiguous z order: +8% (17), +3% (23), +5% (29).
- Exact-kq z-pass: struck at the whiteboard — vector counts are already
  exact at every L in class; the pad waste is intra-vector lanes.

### Borrowed this round, named

- **gen_rader gen_r4**: the plane-custody idea (via gen_layout r3 /
  gen_bluestein r4) — generalized here to z->x block custody, and their
  "custody pays at the size regime it targets" discipline, which this
  round's numbers CONFIRM by failing inside L2: their r4 custody won at
  DRAM-resident 127, mine lost at L2-resident 13-29.  Also their r4
  load-side map-fusion negative (+4%), which my generic result now
  replicates on a second engine.
- **gen_batchlane gen_r4**: the one-lease same-core interleave protocol,
  used for every number above.
- My own r2 (store-forwarding trap), r3 (cross-block pairing idea from the
  solo-row accounting), r4 (lazy-map mechanism and its L=31-only boundary,
  now measured).

### Operation count (shipped)

Identical to r5 at every size (the shipped binary is bit-identical); the
knob variants relocate (never add) map/z work.

### What I would do next

1. **The generic-prime engineering space is now exhausted on ICX** — r5's
   exact tiles were the last real win; r6's two structural candidates are
   measured negatives.  The remaining levers at 13-29 are ARITHMETIC
   (Rader/C3-class), which is gen_rader's lane; the crossover standings
   (dense leads at 23/29 by ~15-20%, Rader owns 31) look r6-stable.
2. **Cross-arch**: race -DGDP_CUSTODY / -DGDP_LAZYZ on Cascade Lake — the
   levers delete L2 traffic and CLX's L2 is 1 MB; the ICX verdict may flip
   at 29.  Ship the knob results into the wisdom cache, not the source.
3. **gen_race/gen_planner**: adopt per-size flag racing; -falign-loops=32
   is worth ~1.5% at 29 on ICX (datum above), and the knob defines the
   candidate axis (code alignment) for other sizes/hosts.
4. Harness notes: tryout.sh's W-before-CH ordering bug is FIXED this round;
   the remote map-check quoting bug persists (literal quotes reach
   check.py) — keep running check.py by hand; r6dev/r6_ab.sh under
   build/tryout/gen_dense_prime/ holds this round's whole protocol
   (build / gen / gates / check / time phases) and the in/c pairs for
   5,7,11,13,15,17,19,23,29 (B=4 and graded B) are current there.

## Round gen_r7

### Where this round started

r6 leaderboard: **120.708** at the graded cell (L=31 B=16 m=140; gen_rader
84.801 — the settled arithmetic crossover, both engines at their plateaus);
roster composites 5.431 / 7.958 / 14.846 / 38.872 at 10/12/15/20.  The r7-8
extension brief queues four literature plays; my own r6 verdict was "the
generic-prime engineering space is exhausted on ICX".  This round (a) mapped
the brief's backlog onto this class (mostly N/A, reasons below), (b) built
and raced THREE never-tried levers at 31 — all bit-identical by construction,
all measured LOSERS — and (c) finally DECOMPOSED the plateau with microbenches
and a padded-chain phase profiler, so the residual gap is now an explained
mechanism, not a mystery.  Ships: **defaults unchanged, source comment-only
edit, binary behavior identical to r6.**  All numbers: a80n0, ONE held lease
(slot 1, core 3), interleaved same-core min-sets (the r4/r5 protocol); node
quiet (control windows 119.5-121, matching r4-r6 quiet levels).

### The r7-8 brief backlog, mapped onto this class

1. **Two-axes-per-pass fusion** (lit 11 Tier 2): targets cells that spill L3
   (100/50/40).  Every cell of mine is L2-resident; r6's custody negative
   already established that fusion inside a covered cache level buys nothing
   here.  Not attempted, per that measured boundary.
2. **Constant-per-site twiddle routing** (Garrido, lit 11 Tier 1): struck at
   the whiteboard.  My GEMM twiddle feeds are already consumption-ordered
   single-uop memory-form broadcasts at 0.5 loads/FMA with no gathers and no
   angle-sorting to delete; compiling constants in still means .rodata loads.
   The lever exists for CT engines with twiddle SCHEDULING problems (powp/
   pow2), not for a real-constant folded-dense GEMM.
3. **Stage-as-outer-product / dense-GEMM crossover at L<=16** (lit 11 Tier 2):
   this engine IS that design (per-stage constant matrices applied by
   broadcast-FMA), and the r6 board already measures the crossover claim's
   answer on AVX-512: dense-GEMM sits 3.4-4.7x behind PFA/batch-lane at
   10/12/15.  The Ascend-study claim ("L<=16 may prefer pure GEMM") does NOT
   hold against a good PFA on this ISA — citable negative, no window needed.
4. **REFFT plan enumeration**: gen_race's lane.

### What was built and raced (all three: gates PASS, outputs bit-identical
### to r6 by construction, cmp-verified at 14 cells single + chain m=4)

1. **-falign-loops=32 at L=31** (the r6 datum that won ~1.5% at 29, raced as
   a build flag via tryout's extra-flags argument; also a +falign-functions=64
   arm).  L=31 B=16: ctl 119.7/120.0/119.5/120.4 vs al 119.7/120.5/121.3/
   121.3 vs alf 120.2/119.8/121.4 — a wash leaning negative.  **The 29 win
   does not transfer to the fold31zx path**; confirms the r6 conclusion that
   code-alignment is a per-size, per-path race axis (gen_race's job).
2. **Pair-level software-pipelined z-phase** (gdp_r7zp.c): split zpair31_uv/
   zsingle31_p into fold and GEMM halves; fold row-pair n+1 into rotating
   64B-aligned 136-double buffers BEFORE pair n's GEMM (pure reordering of
   independent ops — bit-identical).  Rationale: one fused z call is ~620
   uops > the ~352-entry ROB, so consecutive calls cannot fully overlap and
   each map-ladder+fold head (~50-60 cyc of latency, little port work) sits
   at the window frontier 16x per block.  **Measured: +0.5-1%, 4/4 pairwise**
   (119.7/120.2/120.9/120.4 vs 120.0/120.7/121.3/122.5).  The head was NOT
   exposed: the ROB reaches it across the previous GEMM's drain already, and
   the restructure adds buffer traffic.  Extends the r6 "OoO already covers
   it" doctrine to intra-block software pipelining.  (Build bug worth
   recording: the rotating buffer slots must be 64B-multiples — [2][132]
   puts slot 1 at +1056 B and the aligned zmm stores fault; [2][136].)
3. **Merged C+S GEMM with fused combine drain** (gdp_r7cs.c) — my r1
   next-list item 1, costed then but never built.  One j-loop per k-PAIR
   computes C_k, C_k+1, S_k, S_k+1 in 16 accumulators and combines to the
   four dst rows IN REGISTERS at the drain; the k=0 block is an irregular
   12-acc triple (C_0, C_1, S_1) so total FMA is EXACTLY the split form's;
   op order per accumulator identical => bit-identical (verified).  Deletes
   the Cblk round trip (128 stores + ~120 reloads per plane/block), the
   separate S+combine phase, and Cblk's 8 KB of L1 footprint.
   **Measured: +2.3%, 4/4** (120.8/120.5/119.7/120.4 vs 123.1/123.1/122.4/
   123.6).  objdump: zero spills — the loss is the SHAPE: 12 loads/16 FMA
   per j (0.75) vs the split form's 8/16 (0.5).  r1's whiteboard rejection
   is now a measured negative; **the split 0.5-loads/FMA k-quad x 4-zmm
   shape is a measured local optimum** (third shape point: a k-octet x
   2-zmm bench variant sits between at 0.625 and also does not beat it).

### The round's real product: the plateau, decomposed (first counters-free
### mechanism study; perf is absent on both hosts — paranoid=4, no binary)

Phase profile of the SHIPPED padded chain (prof2.c, node core 3; the prof
harness reads ~15% above the driver from per-phase timer serialization —
use the SPLIT, not the absolute):

| phase | us | note |
|---|---|---|
| zx_pm (z+x, lazy map) | 97.35 | 70% of step |
| zx_p (same, no map) | 74.02 | => fused map costs 23.3 |
| y (31x fold31_p) | 40.32 | vs ~20 us FMA-port floor |
| map standalone | 25.96 | vs ~21 us divider floor |

- **The map is divider-throughput-locked**: 3844 vdivpd(zmm)/volume x ~16 cyc
  = ~21 us floor; fused-on-load it still costs 23.3 vs 26 standalone.  The
  r4 lazy-map lever nets ~2.6 us and CANNOT net more — the divider is the
  binder wherever the ops sit.  (And rcp/sqrt alternatives lost r1/r2/r5.)
- **Both GEMM passes run ~1.0 zmm-FMA/cyc in situ** vs the machine's 2.0
  (fmabench: 5.76 GFMA/s = 2/cyc at 2.88 GHz, pure register FMA).
- **The bare C-GEMM isolated runs 1.4-1.6 FMA/cyc** (408-479 ns / 1920 FMA,
  gemmb.c/gemmb2.c): ~25-30% of even the isolated GEMM is TILE-BLOCK
  BOUNDARY DRAIN — 16-accumulator init, last-FMA latency, 16 stores every
  240 FMAs — and the in-situ remainder is phase-junction bubbles (fold ->
  GEMM -> combine phases of 350-1000 uops vs the 352-entry ROB).
- **Boundary count is INVARIANT**: total_FMA / (16 acc x 15 j) — fixed by
  the 32-register file.  Any 32-register tiling only moves the feed ratio
  (0.5 / 0.625 / 0.75 all measured or benched this round; 0.5 wins).
- **Conclusion**: the residual ~1.6-2x over the FMA-port floor at 31 is
  register-file/ROB physics at 240-FMA drain granularity.  It is not
  reachable from C intrinsics.  The one identified lever left is a
  hand-scheduled asm kernel using all 32 zmm as accumulators with
  software-pipelined drains (est. ceiling 15-25%, a full round of work and
  risk).  Six rounds of shape tuning bought 3% because this wall was the
  binder all along.

### Measured on the node (shipped binary = r6 source, this round's windows)

L=31 B=16 m=140: **119.5-120.9 across 11 control windows, best 119.534**
(matches the r6 board's 120.708 quiet level).  B=1: 121.4 warm (first reads
138-140 are the documented short-unit clock ramp; batch-invariant path).
Gates, run fresh on the control this round: single rel_l2 3.917e-16 (B=16) /
3.920e-16 (B=1), tol 1e-12; **two-step 1.724e-15** (tol 3e-14, 17x margin);
m=140 map-chain 2.551e-14 (anchor 2.312e-14) / B=1 1.710e-14 (anchor
1.178e-14), tol 1e-10.  Repeatability cmp-identical.  Small sizes untouched
(bit-identity sweep rebuilt and passed at all 14 dev cells).

### What did NOT work, with the number that killed it

- -falign-loops=32 (+/- functions=64) at 31: wash to -1% (table above).
- Pair-pipelined z-phase: +0.5-1%, 4/4.
- Merged C+S GEMM: +2.3%, 4/4 (and NOT a codegen artifact — zero spills).
- (bench-level) k-octet x 2-zmm GEMM shape: no win over k-quad x 4-zmm;
  an apparent 2x in the first bench was a tiling BUG (t stepped by 2 and
  covered half the row) — recorded so nobody chases it.

### Borrowed this round, named

- **gen_batchlane gen_r4**: one-lease same-core interleave protocol, again.
- **gen_rader gen_r4/r6**: port-model-first discipline; their r6 codegen
  audit prompted my objdump pass (result: my intrinsics kernels are clean —
  no spills, memory-form 231 FMAs, x3-unrolled j-loops in the DSB).
- **docs/literature 11** Tier 1 (Garrido routing — struck with mechanism)
  and Tier 2 (dense-GEMM crossover claim — refuted on-board; stage-as-
  outer-product — already this engine's design), cited per the brief.
- My own r1 next-list (C+S fusion) and r3 list (software pipelining):
  both now closed with measurements instead of standing as open items.

### Operation count (shipped)

Identical to r6/r5 at every size (the shipped source differs from r6 by a
comment block only).

### What I would do next

1. **The only >5% lever left at 31 is a hand-scheduled asm GEMM** (32-acc,
   pipelined drains).  High risk, full-round scope, and the cell is a
   settled Rader win — spend it only if the crossover fight itself becomes
   scored again.
2. **gen_race**: two more per-host/per-size race axes from this round —
   code alignment (wins at 29, wash at 31) and GEMM feed-shape (0.5 vs
   0.625 vs 0.75 loads/FMA); the r7dev variant sources (gdp_r7zp.c,
   gdp_r7cs.c, build lines in r7_ab.sh) are ready-made candidates for the
   CLX/SPR advisory re-races — CLX's narrower front-end may flip the
   pipelining verdict; do not re-derive them, rebuild and race.
3. **The divider floor (~21 us/step at 31) is closed** on ICX arithmetic
   grounds; on SPR (stronger divider) GDP_MAP_SQRT is still the knob to
   re-race, per r5.
4. Harness notes: tryout.sh accepts extra gcc FLAGS after L and B (used
   for the align race — no Makefile changes needed); perf is unavailable
   on a80n0 AND wallaby (perf_event_paranoid=4, no binary) — the r7dev
   microbench trio (prof2.c, gemmb.c, gemmb2.c) is the working substitute;
   r7_ab.sh generalizes r6_ab.sh with parametrized arms (build SRC OUT
   [flags] / gates A B / check ARM L B M / time L B M SAMP ROUNDS ARMS...).

## Round gen_r8

### Where this round started

r7 leaderboard: **120.347** at the graded cell (L=31 B=16 m=140; gen_rader
84.544); roster 5.416 / 8.010 / 14.520 / 39.003 at 10/12/15/20.  My r7
record closed with "the residual ~1.6-2x over the FMA-port floor is
register-file/ROB physics at 240-FMA drain granularity (boundary count =
FMA/(16 acc x 15 j) is INVARIANT under any 32-register tiling), not fixable
in C intrinsics; the only >5% lever left is a hand-scheduled asm GEMM."
This round FALSIFIED the invariance claim and banked the lever in plain
intrinsics.  The new round-8 static analyzers were used exactly as the
brief prescribes: choose the schedule with the models, score with the node.

### The idea: 24-accumulator GEMM tiles (6 k-columns x 4 zmm)

The r7 invariance argument assumed the 4 broadcast registers must stay
live across the j-body (16 acc + 4 u + 4 w = 24, and the next size up,
8 wide, would need 32 acc + regs).  But the broadcasts have NO reuse
beyond their own 4 FMAs: with each `w = set1(wr[k+i])` consumed
immediately, ONE rotating w register suffices, and a 6-column x 4-zmm
group holds **24 accumulators at ~29 live zmm** (24 acc + 4 u + w).
Effects per block at L=31: k-grouping 6/6/4 (C, k=0..15) and 6/6/3
(S, k=1..15); tile-boundary drains drop 16 -> 12 per block, the big
groups drain every 360 FMAs instead of 240, and the feed ratio improves
0.5 -> 0.417 loads/FMA.  Op sequence per OUTPUT row is unchanged (init,
FMAs j=1..15 ascending, one store), so outputs are **bit-identical** to
r7 by construction -- the entire correctness story, same as r5.

### What shipped

1. **L=31**: the C/S GEMMs of fold31_core and fold31zx_core factored into
   SHARED always_inline kernels gdp31_c6 / gdp31_c4 / gdp31_s6 / gdp31_s3
   (the two cores had carried identical copies since r2; the file lost
   ~340 lines net).  The 6-groups are the new shape; the C 4-quad
   (k=12..15) and S triple (13,14,15) tails are the r7 bodies unchanged.
2. **Generic fold_pass (every non-31 size)**: the GDP_CCH/GDP_SCH macros
   extended with g/h accumulator quads (KN up to 6); the k-loops step 6
   with exact 1..5-column tails (greedy 6s: hc+1=15 at 29 -> 6/6/3 vs the
   old 4/4/4/3; 12 at 23 -> 6/6 vs 4/4/4; 11 at 20 -> 6/5 vs 4/4/3).
   fold_block (the default-off custody knob) updated identically.

### Static-model validation BEFORE the node (the round-8 tool discipline)

- objdump/asm audit: the emitted j-body is exactly 4 aligned u loads + 6
  rotating vbroadcastsd + 24 vfmadd231pd into 24 distinct zmm, **zero
  spills** in every KN=6 instantiation (checked by an automated
  innermost-loop scan of the whole .s, both L=31 and generic).
- llvm-mca (icelake-server): new body RThroughput 24.0 cyc/24 FMA vs old
  x3-unrolled quad 48.0/48 -- identical in-loop throughput, no new
  bottleneck (mca's ICL model has 1 zmm-FMA/cyc; use it RELATIVELY only).
- OSACA (ICX): FMAs 0.5/0.5 on ports 0/5 (12 cyc/j), loads 5 cyc/j on
  ports 2/3 -- FMA-saturated with 2.4x feed headroom, as designed.
- **uiCA is BROKEN on this host**: ext/tools/uiCA has xed built for the
  venv python but `instrData/` was never generated -- its setup.sh died
  downloading uops.info/instructions.xml (connection timeout, see
  setup2.log).  Do not burn time on it; llvm-mca + OSACA agree here.

### Measured on the node (a80n0, ONE held lease slot 4 core 6, interleaved
### same-core min-sets -- the r4-r7 protocol; control = impl_7 rebuild)

| cell | r8ctl (r7 ship) | gen_r8 | delta |
|---|---|---|---|
| 31 B=16 m=140 | 119.58 / 120.35 / 120.55 / 120.68 / 121.09 / 121.81 / 121.84 | **113.11 / 113.15 / 113.16 / 113.30 / 113.55 / 113.80 / 114.02 / 115.07** | **-5.7%, 8/8 pairwise** |
| 31 B=1 m=140 | 120.75 / 121.31 / 121.54 | **112.43 / 113.33 / 114.02** | **-6.5%, 3/3** |
| 23 B=4 m=8 | 62.64 / 63.39 / 63.74 | **60.16 / 61.51 / 61.53** | -2.5%, 3/3 |
| 20 B=32 m=256 | 39.01 / 39.13 / 39.94 | **38.00 / 38.22 / 38.55** | -2.3%, 3/3 |
| 17 B=4 m=8 | 31.74 / 32.65 / 32.93 | **27.22 / 31.13 / 32.30** | -2..-5%, 3/3 |
| 29 B=4 m=8 | 139.58 / 142.31 / 143.28 | **137.60 / 141.98 / 142.28** | -1%, 3/3 |
| 19 B=4 m=8 | 36.10 / 36.67 / 37.20 | **35.24 / 35.29** / 36.73 | -2%, 2/3 |
| 13 B=4 m=8 | 11.13 / 11.94 / 12.13 | 11.51 / 11.80 / 12.02 | wash |
| 10 / 12 / 15 | 5.38-5.97 / 7.94-8.57 / 14.78-15.42 | 5.24-6.02 / 8.00-8.52 / 14.72-15.60 | wash, no regressions |

Correctness: **outputs bit-identical to the r7 control at ALL 14 dev cells
(5/7/11/10/12/13/15/17/19/20/23/29/31x2), single AND chain m=4** (so every
r7 gate result is inherited exactly); numpy gates re-run fresh at the
graded cells: single rel_l2 3.917e-16 (B=16) / 3.920e-16 (B=1) tol 1e-12,
m=140 map-chain 2.551e-14 (anchor 2.312e-14) / 1.710e-14 (anchor 1.178e-14)
tol 1e-10; two-run repeatability cmp-identical on the node; the non-AVX512
scalar build compiles clean (scalar path untouched).

### Why it works (mechanism, tied to the r7 profile)

The r7 phase profile put the two GEMM passes at ~1.0 zmm-FMA/cyc in situ
with the isolated C-GEMM at 1.4-1.6, the shortfall split between boundary
drain (acc init + last-FMA latency + 16 stores every 240 FMAs) and
phase-junction bubbles.  The 24-acc tile attacks exactly that term: 25%
fewer drain instances, 50% more OoO-independent chains per drain window
(24 vs 16), and 17% fewer load-port uops competing at the junctions.  -6%
of the step from a lever the r7 study had explicitly written off is the
round's lesson: **"fixed by the register file" arguments must count LIVE
ranges, not named values -- a broadcast consumed immediately costs no
register.**  The remaining gap to the ~65 us floor is now the z-phase
feed shape, the divider floor (~21 us, closed on ICX), and the drain
latency itself (a true asm software-pipelined drain could still pay, but
the intrinsics ceiling just moved 6% closer to it).

### What did NOT work / notes

- No negative results this round -- the single lever built was the single
  lever shipped (static models filtered the schedule before any lease
  time; one build-iteration, zero dead windows).
- 13/15 stay washes: their group counts do not drop (hc+1 = 7/8: 6+1/6+2
  vs 4+3/4+4 -- same 2 groups), confirming the drain-count mechanism.
- uiCA broken (above) -- tooling note for the panel.

### Borrowed this round, named

- **The brief's round-8 tool mandate** (llvm-mca/OSACA schedule-first
  discipline) -- used as prescribed; saved the lease slots that r7 burned
  on the pipelining and C+S-merge negatives.
- **gen_batchlane gen_r4**: the one-lease same-core interleave protocol,
  as every round since r5.
- The lever itself is a refutation of my own r7 item 4 ("boundary count
  invariant") -- no other entry had tried >16 accumulators either; the
  r7cs merged-GEMM negative (0.75 loads/FMA) is what pointed AT the
  rotating-broadcast trick (merging k-groups raised the feed ratio;
  widening them LOWERS it).

### Operation count (shipped)

FMA counts identical to r5-r7 at every size (exact tiles, bit-identical
outputs).  L=31 per block: drains 16 -> 12, GEMM feed uops -17%
(broadcasts 128+120 -> 128+120 unchanged in count but shared over 1.5x
the FMAs; u/v re-loads per j unchanged at 4).  Generic sizes: k-groups
per GEMM at 17/19/20/23/29: C 3/3/3/3/4 -> 2/2/2/2/3, S 2/3/3/3/4 ->
2/2/2/2/3.

### What I would do next

1. **The z-phase is now the largest off-floor item at 31** (~30 us vs
   ~20 floor; the zpair kernel runs 0.75 loads/FMA with 16 acc chains
   across a row-pair).  A 3-row-pair variant (24 C/S chains sharing
   table loads) is the same lever one layer down, but the store layout
   (reversed-half masked stores) makes the drain hairy -- cost it with
   the models first.
2. **Cross-arch**: re-race the 24-acc tiles on CLX (narrower front end,
   1 MB L2) and SPR when advisory windows exist; the shape is pure
   register-file arithmetic and should transfer, but the r6/r7 knob
   lesson says measure.
3. **gen_race**: the 6-vs-4 k-group width is a clean race axis for other
   engines (gen_powp/gen_pow2 GEMM-shaped stages); the bit-identity
   argument transfers to any grouping change with per-output chains kept
   in order.
4. Fix or replace uiCA (fetch instructions.xml offline and rerun
   ext/tools/uiCA/setup.sh) -- llvm-mca's 1-FMA/cyc ICL model is fine
   relatively but will mislead anyone reading absolute cycles.

## Round gen_r9

### Where this round started

r8 leaderboard: **113.507** at the graded cell (L=31 B=16 m=140; gen_rader
84.745 — the settled arithmetic crossover); roster 5.063 / 7.715 / 15.218 /
38.849 at 10/12/15/20.  My r8 record's item 1 named the round's lever: the
z-phase is the largest off-floor term left at 31 (~30 us vs ~20 us floor),
its pair kernel running 0.75 loads/FMA with only 16 accumulator chains —
while r8 had just measured the 24-accumulator rotating-register shape worth
-5.7% on the x/y GEMMs.  The round-9 brief's counter avenues mapped onto
this class: (1) bank-the-picks — N/A, this engine has NO internal tuner
(create() is closed-form deterministic; five consecutive creates produce
identical plans by construction, nothing to bank); (2) two-axes fusion —
not my cells (all L2-resident; r6 custody negative stands); (3) champion
signature — pending node time, see below; (4) port-1 co-issue — struck at
the whiteboard for this engine: the map ladder is on the z-feed critical
path, not independent side work, and the only 128-bit phase (z-fold) already
dispatches on p1.

**HARNESS NOTE, the round's constraint: the a80n0 reservation DIED ~21:00
(heartbeat stale, ssh refused, job 438854 recorded but never beat) and the
guard cannot resubmit from wallaby (no sbatch binary; reserve.log shows the
loop failing).  No node windows existed for this round's decisions as of
this writing.  Everything below is (a) bit-identity + gates, machine-
independent, and (b) wallaby SPR same-core interleaved min-sets (login host,
quiet at 0.3 load; the r5 protocol minus the node).  Node re-race + PMU
signature are queued first thing when the reservation returns.**

### What shipped: the r8 24-accumulator lever applied to the L=31 z-phase

gdp31_zuv3 replaces the per-pair z-GEMM inside fold31zx_core (all three
variants: flat, padded, padded+lazy-map).  THREE mirrored row pairs per
group; the C- and S-GEMMs SPLIT into separate 15-j loops of 24 accumulators
(6 rows x 4 k-vectors) with ONE rotating table register and six live
128-bit-pair broadcasts: 4 table loads + 6 broadcasts feed 24 FMAs per j
= **0.417 loads/FMA (was 12/16 = 0.75)**, and accumulator drains stretch
240 -> 360 FMAs with 24 independent chains per drain window (was 16) —
exactly the r8 GEMM mechanism one layer down.  The C results round-trip a
1.5 KB stack block to free the register file for the S loop — the
split-with-Cblk structure that beat the merged form in the r7 race.  The
group drain does the r3 fused U/V combine unchanged (C halves re-loaded
from the stack block).  Per-accumulator op sequence (init, FMAs j=1..15
ascending, combine) is IDENTICAL to zpair31_uv, and the stack round-trip is
bit-preserving, so **outputs are BIT-IDENTICAL to r8** — the whole
correctness story, as in r5/r8.  Old pair kernel kept behind -DGDP_ZPAIR2.

Static validation before any timing (the r8 tool discipline):
- objdump (icelake-server build): every 24-FMA j-loop body is exactly 4
  rotating vmovapd table loads + 6 vbroadcastf64x2 + 24 clean vfmadd231pd
  into 24 distinct zmm — **zero spills** (automated innermost-loop scan);
  31 live zmm as designed (24 acc + 6 broadcasts + 1 table).
- llvm-mca (icelake-server, RELATIVE use only): new body RThroughput
  24.0 cyc / 24 FMA vs old pair block 48.0 / 48 — in-loop FMA-saturated
  parity, no new bottleneck; the win lives at drain boundaries and load
  ports, invisible to in-loop RThroughput (same signature as r8's table).

### Measured on wallaby (SPR Gold 6448Y login host, quiet, ONE core,
### interleaved same-core min-sets; ADVISORY — the score machine is ICX)

| cell | r8 control (mins) | gen_r9 | delta |
|---|---|---|---|
| 31 B=16 m=140 (core 104, 5 rounds) | 86.24 / 86.28 / 86.71 / 87.21 / 87.23 | **83.43 / 83.60 / 83.71 / 83.77 / 84.08** | **-3.3%, 5/5 pairwise** |
| 31 B=16 m=140 (core 100, earlier window, 4 rounds) | 170.7-173.3 (loaded core) | 161.4-170.8 | -2..-6%, 4/4 |
| 31 B=1 m=140 | 86.35 / 86.46 / 86.66 | **84.17 / 84.51 / 85.71** | **-2.3%, 3/3** |

(Two curiosities for the record: wallaby cores differ ~2x by neighbor load —
core-pin and interleave or the numbers are garbage; and quiet SPR runs the
31-chain at ~84 us vs the ICX node's 113 — consistent with the xarch_spr_r5
advisory's SPR speedups.  The ICX delta may be LARGER than -3.3% because the
z-phase is a bigger share there; r8's identical lever measured -5.7% on ICX
GEMMs after reading -2-3% class effects elsewhere.  Unproven until raced.)

Correctness, all on this host: outputs bit-identical to the r8 control at
ALL 13 dev cells (5/7/10/12/13/15/17/19/20/23/29/31x2), single AND chain
m=4; numpy gates fresh on the shipped binary: single rel_l2 3.917e-16
(tol 1e-12), **two-step 1.724e-15** (tol 3e-14), m=140 map-chain 2.551e-14
(anchor 2.312e-14, tol 1e-10); 17/29 two-step gates PASS; two-run
repeatability cmp-identical; scalar (non-AVX512) build verified end-to-end
at L=31 (chain gates PASS); -DGDP_ZPAIR2 -DGDP_ZG2 knob build reproduces
the r8 binary's outputs bit-for-bit.

### Built, measured, REJECTED: the same lever at generic L (zrow6_k1..k4)

The 6-consecutive-rows split-C/S form of the generic z-pass (KQ+6 loads per
6*KQ FMAs = 0.417 at KQ=4; per-row op sequence unchanged, bit-identical,
zero spills at every KQ) **LOSES on wallaby**: 17 B=4 m=8: 17.4-18.0 vs
16.7-16.8 (+4-7%, 3/3); 29 B=4 m=8: 97.8-98.7 vs 92.8-95.1 (+5%, 3/3);
23: wash (41.2-42.1 both).  Mechanism, best reading: the generic pair
kernel is lean (no FMA-saturated fold feed to hide under), so the chunkier
fold(x6) -> C-loop -> S-loop phase structure and the C stack round trip
cost more than the deleted table loads — the same "OoO already covers the
lean kernel" boundary as my r6 custody/lazy-map and r7 pipelining
negatives, now measured on a third restructure.  Why 31 wins and generic
loses: fold31zx's z-GEMM sits inside a dense always-inline block with the
x-GEMM adjacent; the generic kernel is a bare call in a row loop.
**Shipped DEFAULT OFF behind -DGDP_ZG6** (kernels compiled, knob-selected)
— raceable on ICX/CLX where the verdict may flip; do not re-derive.

### What I would do next

1. **Node re-race of the shipped default vs the r8 control at 31 (B=16 and
   B=1), first thing when the reservation returns** — expected -2..-6% from
   the SPR direction and the r8 precedent; then tools/pmu.sh both arms and
   record p0+p5 dispatch per cycle (brief avenue 3): r8's whole-chain
   signature vs the champion's 1.60/2.0 tells whether another schedule
   round at 31 can pay at all.
2. **Race -DGDP_ZG6 on ICX** (one window: 17/23/29 B=4 m=8) — the SPR
   negative may not transfer; if it wins on ICX, flip the default there via
   the wisdom/race layer, not unconditionally.
3. The remaining 31 items after this: divider floor (~21 us, closed on
   ICX), y-pass fold feed (u/v build is 15 plain-loop iterations/block —
   the zrow31_fold zmm treatment could apply; ~small), and the true asm
   software-pipelined drain (the r7 ceiling estimate, full-round scope).
4. Harness: reservation death mid-round is now a measured failure mode —
   record wallaby-advisory verdicts with explicit "pending ICX" tags so the
   next agent knows which conclusions are score-machine-proven and which
   are not.  The r9dev A/B kit (bin_r8/bin_r9 + per-cell in/c bins) lives
   in build/tryout/gen_dense_prime/r9dev/.

### Borrowed this round, named

- **My own gen_r8** (the 24-acc rotating-register shape and its live-range
  arithmetic; the split-beats-merged structure from gen_r7's C+S negative).
- **gen_batchlane gen_r4**: the same-core interleave protocol, applied to
  wallaby cores in the node's absence.
- **The r8 brief's tool mandate**: objdump spill scan + llvm-mca relative
  check before any timing — again zero dead measurement windows.
- **PMU_AUDIT.md** (monitor): the four-avenue map; avenues 1/2/4 dispositioned
  for this class at the whiteboard (no internal picks to bank; L2-resident
  cells; no independent 256-bit side work), avenue 3 queued on node return.

### Operation count (shipped)

FMA counts identical to r5-r8 at every size (bit-identical outputs).  L=31
z-phase per block: table loads 15 pairs x 15 j x 8 -> 5 groups x 15 j x
(4 C + 4 S) = 1800 -> 600; broadcasts unchanged (1800); new C stack traffic
+240 ops/block; accumulator chains per drain window 16 -> 24; drains per
block 15 fused -> 5 C + 5 S (+1 solo row unchanged).  Net ~-35k uops/step
on the z-phase feed path.  Generic sizes: shipped path unchanged from r8.
