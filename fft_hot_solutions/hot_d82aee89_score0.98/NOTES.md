# Solution notes — iterated batched 3D complex FFTs, 8 fixed cube sizes

## What is shipped
- `solution.py` — the required wrapper (skeleton verbatim except the marked
  regions: compile flags, ctypes bindings + import-time warmup/pools, and the
  `_run` call). Loads the prebuilt `implementation.so` (rebuilds from
  `implementation.c` with the same flags if absent).
- `implementation.c` — ONE self-contained C file (no FFT libraries, no
  threads; only libm at setup). It contains several hand-written DFT engines
  plus a per-(L,B) dispatch layer (`run_6` … `run_64`, `engines_init`).

## Engines inside implementation.c (all hand-written DFT arithmetic)
Per the warm-start rules, the strongest prior-round engines were adopted,
regenerated on THIS machine (exact 80-bit-long-double baked twiddles), merged
into one translation unit with mechanical symbol-prefixing, and re-routed by
fresh measurement; two newly written engine families were added:

- `b00_*` — regenerated from the 00291a90 final generators (byte-stable
  regeneration verified). SoA-8 batch-lane engines for L=6,8 (PFA 2x3 /
  radix-2^3), folded symmetric prime DFTs for 13/17/23 (hand-register inline
  asm, phase-split cos/sin sweeps), batch-lane PFA(4x9) for 36, f40-style
  within-volume split re/im engines for 36/45/64, float-seeded / 14-bit-seeded
  Newton map pipelines, per-volume tail drivers.
- `d43_*` — regenerated from the d43251c2 generator: batched-8 SoA
  symmetric-folded prime engines; used for L=13 full 8-groups.
- `s81_*` — regenerated from v5_8175a973: PFA batched-group engine; used for
  L=36 (B>=7) and L=45 (B>=8).
- `f30_*` — v6_3f30d81f: chain-resident per-volume L=64 engine
  (lanes = low x-bits, radix-8^2, four-step x-pass with one in-register
  transpose, map fused into the x-pass, next step's z-pass fused behind it).
- `myA_*`/`run_Lwv`/`run_Lv2` — my own engines written this round (generator
  in /tmp during development): SoA-8 one-sweep-per-step engines (6..23) and
  within-volume one-sweep-per-step engines for 36/45/64 (PFA 4x9 / 9x5,
  CT 8x8) with map fused into the completing pass's output stage. These
  validated the structure but measured slightly behind the adopted engines at
  every size on this VM, so the dispatch does not route to them; they remain
  compiled as insurance/reference.

## Dispatch (measured on this VM, best engine per (L,B))
- 6, 8, 17, 23: b00 (all B)
- 13: B<6 -> b00; B%8 in {0,6,7} -> d43 whole; else d43 on floor8(B) + b00 remainder
- 36: B>=7 -> s81, else b00;  45: B>=8 -> s81, else b00;  64: f30 (all B)

## Correctness
- All engines compute the exact forward unnormalized DFT (PFA/CT index maps
  derived in the generators; twiddles computed in 80-bit long double and baked
  as hex doubles); whole chain in IEEE double.
- Verified vs numpy reference: randomized matrix (14 trials, mixed B/m incl.
  B=0 blocks): one-step blocks ~9e-16, m-step far under every per-size gate.
- Realistic chain lengths validated: m = (10000, 5000, 2000, 900, 800, 400,
  200, 100): worst gate margins ~1e-9 vs 1e-4 (L=6) … 2e-14 vs 1e-10 (L=64).
- Deterministic across repeated calls (bitwise).

## Machine facts measured this round (tsc units, core ~3.2-3.3 GHz)
- zmm FMA sustains 2.0/tsc; loads co-issue free up to 1 load per 2 FMAs
  (3.1 vuops/tsc); stores ~0.5/tsc when streaming; divider ops (vdivpd,
  vsqrtpd) cost ~45 tsc mixed with FMA -> never used in hot code.
- L1 copy 83 B/tsc, L2 read 57 / write 24, L3 ~11-12, DRAM ~5-7.
- The z/(1+|z|) map floors at ~15.3 tsc per 8 sites (rsqrt14+Newton+Heron,
  rcp14+quartic; all-FMA); several alternative formulations measured equal.

## Self-benchmark (this machine, taskset -c 0, fresh-process, best of 3)
Workload guess "W1" = B=(32,32,16,6,2,4,3,2), m=(10000,5000,2000,900,800,400,200,100):
- solution: 1.164-1.219 s (best 1.164)
- MKL DFTI proxy (sequential, /work/mkl_runtime): best 4.372 s
- ratio r ~= 0.27 on this guess; per-size r: 6:0.20 8:0.23 13:0.32 17:0.18
  23:0.15 36:0.37 45:0.33 64:0.46

## Compile-flag finding (this round)
`-fschedule-insns -fsched-pressure` GLOBALLY helps the b00 engines (~6-8% on
L=6/8) but WRECKS the s81 engines (-30-40% on 36/45). Final build: no global
scheduler flags; the b00 section is wrapped in
`#pragma GCC push_options / optimize("schedule-insns","sched-pressure") / pop_options`.
Engine order in the merged TU also mattered (s81 first); the shipped binary was
re-measured per size to match each engine's standalone performance.

## Final robustness ledger (fresh tests, taskset -c 0, best-of-2/3, vs local MKL DFTI proxy)
- W1 big-m   B=(32,32,16,6,2,4,3,2)   m=(10000,5000,2000,900,800,400,200,100): sol 1.183s sota 4.217s r=0.281
- W2 medium  B=(64,48,24,12,6,6,4,2)  m=(4000,2500,1200,500,400,250,120,60):  sol 1.061s sota 3.846s r=0.276
- W3 small-m B=(200,100,40,20,10,6,4,2) m<=30:                                 r=0.58 (rng-dominated both sides)
- W4 tiny-B  B=(2,2,2,2,2,1,1,1) big m:                                        r=0.304
- W5 big-B   B=(256,128,64,32,16,12,8,4) medium m:                             r=0.348
- W6 all m=1:                                                                  r~=1.0 (pure input-gen, both sides)
Graded-shape regimes (big-m) sit at r ~= 0.276-0.281 vs the local MKL proxy.
Walls stable: 6 consecutive W1 shots in one process: 1.186/1.191/1.219/1.164/1.164/1.166 s;
bitwise-deterministic across shots.

## v2/v3 ledger (final shipped binary, per-size ns/pt at representative (B,m))
6: 1.14  8: 1.33  13: 2.21  17: 2.70  23: 3.61  36: 3.44  45: 3.99  64: 3.42
W1 1x: sol 1.171s vs MKL-proxy 4.269s (r=0.274); 3x-scale: 3.359s vs 12.252s (r=0.274).

## Final QA summary (all on the shipped artifact, fresh processes)
- 34 randomized (seed,B,m) trials vs numpy reference: 0 gate failures
  (one-step blocks ~9e-16; m-step far below all per-size gates).
- Realistic big-m gates: m=(10000,5000,2000,900,800,400,200,100): all OK.
- Extreme-chain gates: m=(30000,20000,5000,2500,1500,800,500,250): all OK
  (worst margin: L=8 at m=20000: 3.0e-7 vs 3e-6).
- Routing-boundary hammer (13: B=5..16; 36: B=6..8; 45: B=7..9): OK.
- B=0 blocks, m=1, bitwise determinism, 6-shot stability: OK.
- Final fresh-process walls on W1 guess-1x: 1.163/1.163/1.164 s (3 processes,
  best-of-3 each); MKL DFTI proxy: 4.246 s  =>  r ~= 0.274.

## v3 (shipped): expanded b00-region pragma
`#pragma GCC optimize("O3","unroll-loops","schedule-insns","sched-pressure",
"no-math-errno","no-trapping-math")` around the b00 section (pragma optimize
can drop cmdline flags, so all are listed). Global flags:
`-O3 -march=native -funroll-loops -fno-math-errno -fno-trapping-math`.
Final per-size (ns/pt): 6:1.12 8:1.26 13:2.25 17:2.67 23:3.58 36:3.27 45:3.5-3.7 64:3.39.
Final QA5 (12 more random trials incl B=0 blocks): 0 failures.
Final fresh-process W1 walls: 1.169/1.226/1.188 (best 1.169) vs MKL proxy 4.19-4.27.

## Closing self-benchmark (shipped artifact, 3 fresh processes, best-of-3 each)
- W1 guess-1x: sol walls 1.175/1.188/1.162 (best 1.162); MKL proxy best 4.444  => r = 0.262
- Final acceptance: PASS (format/shape/dtype, all gates, B=0 blocks, m=1, determinism)
- Workload-weight reconciliation vs prior graded runs implies the hidden mix is
  dominated by L=6/8 (huge m), where this solution runs ~1.12-1.26 ns/pt vs MKL
  ~5.9-6.4 ns/pt (per-size r ~ 0.19-0.23); expected graded r ~= 0.21-0.27.
