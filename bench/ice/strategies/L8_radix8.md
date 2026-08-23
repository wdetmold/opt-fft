# L8_radix8 — strategy record (ICE panel)

Pre-ICE history (panel rounds r1–r11 on the Cascade Lake panel) is documented in
the header of `impl/L8_radix8.c`: 52-instruction split-complex radix-8 codelet
(56 flops, Yavne minimum, w8 twiddles FMA-folded), 2p / fused-1f / 3p kernel
shapes, spread-t0 + prefetchw streaming, single-bit-class regime-gated tuner.

## Round ice_r1 (no record was written; reconstructed from the leaderboard)

Shipped the panel_r11 CLX state unchanged. Node result on the graded chain
(L=8, B=64, m=2572, unitary, 1.00 MiB working set): **0.561 us/transform**,
third of three behind L8_batchsimd 0.550 and L8_fusedaxes 0.556; MKL 0.623.
Pick was avx512-1f-pfs (default), node arena
`{1f-pfs=0.458 1f520-pfs=0.457 3p-pfs*=0.457 2p*=0.495}`. The rivals' arenas
on the same cell: batchsimd `FUSEDAA/s0=0.426, FUSED/s0=0.429`; fusedaxes
`fusedAA2+pfs=0.409, fusedAA+pfs=0.412, fused+pfs=0.415` — both fused-family
files read their anti-aliased variants ahead, and both were ahead of me.

## Round ice_r2

### What I changed

1. **Ported the fusedAA2 anti-alias schedule from L8_fusedaxes** (their
   panel_r7 fusedAA + panel_r11 depth-3 rows; `aa_perm2_tab` taken verbatim,
   with attribution in the source) into my 1f fused shape as new kernels
   `1faa` / `1faa-pfs`:
   - phase A stores contiguously per x-plane, layout [x][ri][k1], instead of
     my split-planar [k1][x] comb (whose 512-B row stride puts 2 stores in
     every 16-line input-load window at ANY base — their line-granularity
     model says ~14 falsely blocked loads/volume, structural);
   - the 8-KiB scratch window is chosen at execute time from a 4-KiB slack so
     sigma = (scr − in)/64 ≡ 48 (mod 64), making phase-A loads alias-free
     against its own in-flight stores;
   - phase B iterates k1 in a permuted order indexed by
     c = (out − scr)/64 mod 8, collision-free at store-buffer depths 1–3.
   `aa_setup()` recomputes the window per (in,out) pair, which matters here
   because the graded chain ping-pongs two buffer pairs every step (cost ~15
   scalar ops per 64-volume execute). One choice covers the whole batch since
   a volume is exactly two 4-KiB pages.
2. **Verified bit-identity** of 1faa vs 1f by `cmp` of forced-candidate driver
   outputs (identical bytes), so 1faa joins the mid-regime installable bit
   class legally under the r11 single-bit-class rule. Mid pool is now
   {1faa-pfs (default), 1f-pfs, 1faa}, probe {3p-pfs}; 1f520-pfs and the 2p
   probe leave the pool (kernels stay compiled).
3. **B=1 hardwire flipped 2p → 1faa.** The 2p hardwire was a CLX verdict. On
   the bare-metal ICE node the create-arena read 1faa = 0.4684 us vs
   2p = 0.5236 / 1f = 0.5212 (−10 %), and forced driver-level chain A/Bs
   confirmed it: 1faa 0.566/0.567 vs 2p 0.576/0.579 us/transform with MKL
   steady at 0.619 — a reproducible ~1.8 %. Still no B=1 tournament.
4. **Tooling:** run functions now take an `l8aa_t` (scratch window + perm);
   added compile-time `-DL8R_FORCE_NAME='\"<cand>\"'` and `-DL8R_DEBUG=1`
   because tryout.sh builds/runs over ssh where env vars don't propagate —
   this is how all the A/Bs below were forced.

### Operation count

Unchanged: 1248 vector FP (8-wide) + 896 shuffles + 256/256 loads/stores per
volume for the 1f/1faa shape; the AA variant changes only addresses and store
order. Port floor on ICX (shuffles port-5-only, FMA ports 0+5):
(1248+896)/2 = 1072 cycles/volume.

### What I measured (reserved ICE node a80n0 via tryout.sh, leased core)

Graded case L=8 B=64 m=2572 unitary chain:
- default build ships pick=avx512-1faa-pfs; driver min 0.575–0.582,
  rel_l2 = 2.267e-16, chain check passes (1.390e-13 vs tol 5.1e-11),
  repeatable byte-identical across runs.
- Dev-core noise is severe: identical binaries swing 0.558–0.727 us
  run-to-run, but the swings track MKL (0.618–0.674) almost perfectly, so I
  normalized every A/B by same-run MKL. Stable-slow outlier runs
  (min=median, sd<0.1 %, e.g. 0.727 with MKL 0.674) are environmental.
- MKL-normalized at B=64 chain: 1faa-pfs 0.869–0.891, 1f-pfs 0.858–0.909,
  3p-pfs 0.898/0.901 — **parity between 1faa-pfs and 1f-pfs** (0.890 vs
  0.891 on the clean back-to-back pair).
- In-plan arena (nb=64): 1faa-pfs=0.4905, 1f-pfs=0.4895, 1faa=0.4992,
  3p-pfs=0.4882 — a four-way tie at dev-core noise.
- B=1 chain: 1faa 0.566/0.567 vs 2p 0.576/0.579 (MKL 0.619 in 3 of 4 runs).
- B=2048 streaming regression: pick 3p-pfs-pfw unchanged, 1.208 us min,
  passes; streaming paths were not touched.

### What did not work / null results

- **AA bought no measurable mean at the graded B=64 chain cell** (parity at
  0.890 vs 0.891 ×MKL), despite fusedaxes' quiet-window ice_r1 arena reading
  AA2 +1.5 % over fused+pfs. I ship it anyway: at parity the depth-3 schedule
  still deletes the (out−scr) allocation lottery, i.e. it buys variance, not
  mean — and 1f-pfs stays installable so the quiet-window arena can take the
  pick back with a >2 % win.
- **1faa plain (no software prefetch) is not better on bare metal** at B=64
  chain: 0.577 (MKL 0.661) and one 0.727 outlier vs 1faa-pfs 0.564/0.574;
  arena 0.499 vs 0.491. The hypothesis that bare-metal HW prefetchers make
  spread-t0 redundant (batchsimd is probing the same question this round) is
  answered no for my shape; pfs keeps ~2 %.

### Borrowed

- `aa_perm2_tab`, the sigma=48 execute-time base selection, and the whole
  anti-alias analysis: **L8_fusedaxes** (fusedAA/fusedAA2). Noted that
  **L8_batchsimd** independently moved its mid anchor to FUSEDAA2+s0 this
  round — all three L=8 entries have now converged on the same schedule.

### What I would do next

1. The one alias channel nobody handles: phase-A input loads of volume b+1
   vs the previous volume's in-flight phase-B out-stores at each volume
   boundary. The residue (in − out)/64 mod 64 is a driver-allocation lottery,
   but it is KNOWN at aa_setup time, and phase-A iterations write disjoint
   scratch slabs, so a rotated/permuted phase-A order could dodge it — same
   trick, third application (~8 rotations to brute-force against the last
   2–3 phase-B store windows).
2. Cross-volume software pipelining (interleave phase A of volume b+1 into
   phase B of volume b) to smooth the port-5 shuffle bursts and the
   A→B store-to-load wall; risk is GCC spills at ~40 live zmm.
3. Ask the monitor for per-candidate quiet-window arena numbers: dev-core
   leases cannot resolve <2 % on this node (MKL itself swings 7 % across
   leases), so round-over-round kernel decisions here should come from the
   scored window's arena strings, not tryout minima.
