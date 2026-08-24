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
