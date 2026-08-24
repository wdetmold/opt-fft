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
