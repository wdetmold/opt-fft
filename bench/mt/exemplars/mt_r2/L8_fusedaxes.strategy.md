# L8_fusedaxes — strategy record (multicore phase)

Phase-1 history (how the serial kernel got its shape, panel_r1..r11) lives in
`../../geom/strategies/L8_fusedaxes.md`.  This file starts at the 32-core
phase.  Phase-1 node baselines for parallel efficiency (panel_r11 scored,
single-thread Cascade Lake): B=1 0.555 us, B=64 0.578, B=2048 0.922,
B=16384 1.251 us/transform.  Scored mt cells: B = 1, 2048, 32768.

## Round mt_r1

### What was built

The phase-1 kernel (fused three-axis 8^3, split-complex, spatial axis in the
SIMD lanes, 52-instr codelet, AA store schedules — arithmetic untouched, all
17 variants still compiled and forceable) got a threading layer:

1. **Volume-parallel batched path.**  The batch is cut into nt contiguous
   chunks; thread t runs the untouched serial variant loop on
   [B·t/nt, B·(t+1)/nt) with its OWN 40 KiB scratch lane (classic scr + both
   AA arenas), allocated and first-touched by thread t inside one
   create-time OpenMP region — NUMA-local under the harness's close/cores
   binding.  Chunk bases are whole volumes (8 KiB = 128 lines, ≡ 0 mod 64
   lines), so (a) no two threads ever share an out cache line, and (b) every
   thread sees the same line residues against its own AA scratch as the
   serial kernel, so `aa_setup` per lane needs no new analysis.  Lane structs
   are 128-B aligned/padded.

2. **Pinned spin-wait pthread pool** instead of an OMP region per execute.
   Protocol adopted from **L36_pfa mt_r1** (epoch release word published by
   main, per-worker 128-B-padded ack lines, main = participant 0,
   publish-then-scan), including **L17_rader mt_r1's** hard-won invariant:
   main waits for ALL workers to ack every epoch — team members and idle
   alike — because that is what makes rewriting the job descriptor for the
   next dispatch safe (their team-only-ack attempt double-ran jobs; I did
   not rediscover this).  Workers are created in fft3d_create, pinned to the
   exact CPUs the harness's OMP mapping chose (read back via sched_getcpu in
   a throwaway region), spin ~25 ms after their last job, then park on a
   condvar.  One hardening of my own: a seq_cst fence between the epoch
   store and the `parked` read in dispatch — TSO lets the load pass the
   store, and a just-parked worker could otherwise be missed (main would
   spin on its ack forever).  After the tuner picks, the pool is SHRUNK to
   the picked team (L36_pfa: unpicked spinners drag the all-core clock).
   Execute never creates a thread.  `L8_POOL=0` forces the omp-region
   fallback for a node A/B; `L8_TEAM=<n>` forces the team size (1 = serial).
   **Measured A/B, wallaby, B=2048: pool 31.7 us/call vs omp region
   35.3 us/call (−10.4%), 3/3 both sides.**  From the B=32 cell (call
   1.148 us, each thread computing one 0.33 us volume), dispatch+join at
   T=32 is ~0.8 us — against 2.7–8.3 us for a hot omp region (L17_rader /
   L23_matrixsimd measurements, reused not re-derived).

3. **B=1 stays serial — measured answer: it does not parallelise at L=8.**
   One volume is 0.555 us on the node.  The cheapest dispatch on this panel
   is my own pool's ~0.8 us join at T=32 (measured above); L17_rader's pool
   handshake is ~1.5–2.5 us and an OMP region 2.7–8.3 us.  Every known
   dispatch ≥ 1.4× the ENTIRE serial budget before any work happens, and the
   fused shape needs a barrier between phases A and B on top.  A B=1 plan
   therefore never allocates lanes, never builds a pool, never enters an
   OpenMP region: it is the phase-1 path (tiny-band serial tuner + B1DIRECT
   dispatch) bit-identical.  Wallaby confirms 0.327–0.328 us = the phase-1
   r8–r11 floor exactly.

4. **NT stores re-admitted to the batched race.**  Phase 1 retired NT on
   five rounds of single-core node evidence ("hide the RFO with prefetchw").
   Three entries measured the rule INVERTING at 32 threads this round —
   L23_matrixsimd, L36_pfa (−17% at B=512), L17_rader (−14% at B=4096) — 
   because the batched cells are DRAM-bound and the RFO is a third of the
   traffic.  Confirmed here: wallaby B=32768 arena, fused-nt+pfs/32 0.0760
   vs fused+pfs+pfw/32 0.1085 us/t (−30%).  At the L3-resident B=2048 on
   wallaby NT rightly loses ~2× and the tuner keeps plain — per-machine
   race, exactly as designed.

5. **Tuner = racing a (variant × team) grid under the real dispatch.**
   For batch ≥ 64: {fused+pfs, fused+pfs+pfw, fused-nt+pfs, seq3-nt+pfs} at
   the full team, plus the leading plain and NT shapes at 24 and 16 threads
   (the two-socket question for the node; on wallaby it caught DRAM
   saturation instead — fused-nt+pfs/24 beat /32, 0.0454 vs 0.0519, before
   the arena-depth fix below).  For 2 ≤ batch < 64: serial is the anchor and
   a team must beat it by 3%.  Anchor for streaming ws (>1.5·L3) is
   fused-nt+pfs/32; else fused+pfs/32; 2% hysteresis, min-of-5, one untimed
   own-cache-state pass per trial.  The arena is filled AND memset by the
   MAIN thread, deliberately: the driver freads `in` and memsets `out` on
   its main thread, so the caller's pages are one-socket, and the race must
   see that placement (**L23_matrixsimd mt_r1's lesson**, taken as stated).
   The full grid is published in `arena{}` in the description every run.

### Operation count

Unchanged from phase 1 per volume: 1248 vector FP (24 × 52, Yavne minimum,
FMA-folded twiddles) + 896 shuffles; 256+256 loads/stores (fused/AA shapes).
Parallelisation only reassigns whole volumes to threads — output is
bit-identical to the serial kernel at the same variant, which is why the
tuner needs no correctness gate and pick-flips are check-safe.

### Measured (wallaby, Gold 6448Y SPR, 32 threads close/cores on a shared
login node — dev numbers, relative only; quiet-window minima)

| cell | mt_r1 | pick (debug runs) | vs wallaby 1-thread | note |
|---|---|---|---|---|
| B=1     | **0.327 us** | fused (serial, phase-1 path) | 1.00× by construction | sd 0.06% |
| B=2048  | **31.4–31.9 us/call = 0.0154 us/t** | fused+pfs+pfw/32 or fused+pfs/24 (≤8% apart in-arena) | ~0.50/0.0154 ≈ 32× | L3-resident on wallaby's 60 MB; superlinear-looking because the 1-thread kernel was MLP-bound (~10 LFBs), as the brief predicted |
| B=32768 | **2466–2515 us/call = 0.0753–0.0767 us/t** | fused-nt+pfs/32 (stable 2/2 after arena fix) | ~1.0/0.076 ≈ 13× | DRAM wall: 16 KiB/vol at 0.0758 us/t = 216 GB/s with the RFO deleted; matches L17_rader's 245 GB/s NT ceiling on the same host |

Correctness: PASS (rel_l2 ≈ 2.2–2.3e-16, tol 1e-12) at B = 1, 2, 3, 5, 31,
32, 33, 63, 65, 100, 2048, 32768; repeatable bit-identical across runs at
every size; ~270k pool dispatches in one driver run without a hang; builds
warning-free (-Wall -Wextra) at native (SPR), cascadelake, and haswell
(AVX2) targets.

Parallel efficiency against the phase-1 NODE numbers must wait for the node:
the honest statement from wallaby is ~32×/100% at the L3-resident batched
cell and ~13×/40% at the DRAM wall, plus a reasoned 1.00×/serial at B=1.

### What did NOT work, with the numbers that killed it

1. **Streaming surrogate at bsur=8192 (128 MiB)**: wallaby's 60 MB L3 held
   half the arena and crowned fused+pfs+pfw/32 at arena 0.0425 us/t — which
   the driver then ran at 0.105 us/t at the real 512 MiB, while the
   NT-at-24 pick the arena rejected drove 0.0767.  Wrong pick by 27%.  Fix:
   surrogate capped at 8× L3 (clamped [8192, 32768] volumes) so the arena is
   in the driver's residency regime; pick became fused-nt+pfs/32, stable
   2/2, driver 2483–2515 us.  (On the node the same cap gives 11264 volumes
   = 176 MiB vs 22 MiB L3.)
2. **reps=1 timed trials (0.4 ms) in the grid tuner**: coin-flipped a 0.5%
   tie between fused+pfs+pfw/32 and fused-nt+pfs/24 across runs.  Raised the
   target to ≥32768 volumes per trial (reps ≥ 2, ~2+ ms) — pick strings
   became identical across runs.  Same lesson as phase-1 r5/r11, re-learned
   at 32 threads where a pass is 32× shorter.
3. **One omp parallel region per execute** (the v1 dispatcher): 35.3 vs
   31.7 us/call at B=2048 — a 10.4% tax at the cell where the call is short.
   Kept as the fallback path and behind L8_POOL=0, replaced by the pool.

### Borrowed, and from whom

* **Spin pool protocol** (epoch word, flag-array acks, publish-then-scan,
  park-after-idle, pool-shrink-after-pick): **L36_pfa mt_r1**; the
  all-workers-ack invariant and its double-run failure mode: **L17_rader
  mt_r1** (not rediscovered).  The seq_cst fence in dispatch is my addition.
* **Volume-parallel contiguous blocks + owner-first-touch lanes**:
  L13_direct / L17_winograd mt_r1 via L36_pfa's record.
* **NT re-race at 32 threads**: L23_matrixsimd mt_r1's inversion result,
  confirmed by L36_pfa and L17_rader; re-admitted here and confirmed −30%
  in-arena at B=32768.
* **Serial arena fill so the tuner sees the driver's NUMA placement**:
  L23_matrixsimd mt_r1.
* From my own phase-1 lineage: every kernel variant, the AA machinery, and
  the tuner discipline (round-robin, own-cache-state warm pass, min-of-N,
  hysteresis toward the anchor) carried forward unchanged.

### Node predictions (stated to be scored)

* **B=1: 0.545–0.556 us, pick fused-family serial** — byte-identical code
  path to panel_r11; any movement is machine state.
* **B=2048 (ws 32 MiB vs 22 MiB L3, both buffers socket-0)**: the genuinely
  open cell.  If the two-socket UPI path binds, the 24/16-thread candidates
  or NT take it; plausible range 0.06–0.20 us/t.  Read the `arena{}` string:
  plain-vs-NT and 32-vs-24-vs-16 are both published every run.
* **B=32768: pick fused-nt+pfs at 32 or 24**, per-transform set by node
  DRAM: 16 KiB/vol ÷ (aggregate GB/s).  At 200 GB/s that is ~0.082 us/t
  (~15× phase-1); if socket-0 residency of in/out halves effective
  bandwidth, ~0.16 us/t and the T=16 or weighted-partition question becomes
  next round's first item.

### Next round

1. **Read the node's arena{} strings first** — they carry the full
   (variant × team) grid at both batched cells plus the pool/omp and
   pick provenance.
2. **Weighted partition** (L17_rader mt_r1 item 5): both caller buffers are
   socket-0, so the far socket's threads stream over UPI and equal chunks
   run at the slowest thread's rate.  L17's measured per-thread re-cut is
   the template; build it only if the node shows the imbalance (compare
   T=32 vs T=16 in arena{}).
3. **B=1**: the only lever left would be a sub-0.2 us barrier, which does
   not exist on this hardware; the serial answer stands unless someone's
   record proves otherwise.
4. If the node's B=2048 pick lands NT while B=32768 lands plain (or any
   cross-regime surprise), re-derive the anchor rule from the node's own
   pick history, not wallaby's.

## Round mt_r2

### Where the round started (node, mt_r1)

B=1 0.554 us (tied best, 1.19× MKL), B=2048 0.0262 us/t (won, 1.40× MKL),
**B=32768 0.176 us/t — LOST to fftw3_patient's 0.161** (min-of-min; my
medians beat fftw's by up to 1.67×, but the loss stands as scored).  My pick
there was fused-nt+pfs/nt16 = 93 GB/s of useful traffic on socket 0's DRAM.
The VERDICT calls the cell "a tuning fix, not a kernel fix" and its §4 last
paragraph falsifies the round's shared premise: the caller's pages are NOT
reliably socket-0 by the time samples run (L6_pfa sustained 175 GB/s =
both sockets' DRAM on driver-touched buffers; fftw's winning samples are a
~102 GB/s minority mode against a 55 GB/s majority mode; nine entries show
cross-process placement/pick lottery).  The decisive numbers: 0.161 needs
>102 GB/s, one socket practically tops out near there, and my nt16 pick
**forecloses ever using the far socket** — while the create-time surrogate
race can never see the real buffers' placement, so it cannot make this call.

### What changed (arithmetic untouched, all r1 variants intact)

1. **Variant 17 `seq3AA-nt+pfs`** — the seq3AA shape (AA phase A, permuted
   B1, scr2 pinned so pass B2 is alias-free) with NT stores and the seq3
   spread-prefetch cadence.  Rationale: at L=8 the volume stride is
   8192 B ≡ 0 (mod 4096), so phase A's loads of volume v+1 land on exactly
   the bits-11:6 residues of the fused shape's *scattered* phase-B
   out-stores of volume v — and at DRAM-bound drain rates the 56-entry
   store buffer is effectively always full, so this 4K-alias false
   dependence (the §4.5 counter the VERDICT flagged, never yet counted)
   recurs at every volume boundary and throttles load MLP exactly where
   bandwidth is made.  seq3's B2 stores the volume *ascending*: the
   in-flight store tail when A(v+1) begins is the high residues while the
   first loads are the low ones — the streams chase and never collide.
   Node r1 arena already had seq3-nt+pfs/32 beating fused-nt+pfs/32
   (0.203 vs 0.208) but the seq3 family was never offered below T=32.
2. **Streaming create-grid reworked** (ws > 1.5·L3 only; the B=2048 mid
   grid is byte-identical to mt_r1): NT shapes {fused, seq3, seq3AA} at
   T=32 and 16, fused-nt & seq3AA-nt at 24, plain fused+pfs+pfw/32 kept as
   the NT veto; anchor = fused-nt+pfs/16 (the node's r1 winner).
3. **Execute-time streaming governor** (armed when ws > 6·L3, batch ≥ 4096,
   pool built, no forcing knobs).  The plan races a 4-config set **on the
   caller's real buffers** across the first ~18–30 execute calls and locks
   the winner (1.5% hysteresis toward the create pick).  This is legal and
   free under the harness statistic: every probe call is a full correct
   execute, and min-of-min ignores slow warmup/calibration/early-sample
   calls.  Configs: cfg0 = create pick; cfg1 = seq3AA-nt+pfs at the half
   team; cfg2/3 = fused-nt / seq3AA-nt at the FULL team with **weighted
   adaptive cuts** — per-thread chunk sizes re-cut every call from each
   thread's own measured chunk time (damped √(t_avg/t_t), per-step clamp
   [0.75,1.33], weight clamp [0.02,4]), so if the far socket owns pages (or
   they migrate) it converges to a both-socket split, and if socket 0 owns
   everything it converges toward the half team with ~0.1%-of-B probe
   chunks still measured (so it can recover).  Probing budgets self-extend
   (+2 calls while a config still improves >2%) up to 12 calls; after
   locking, every 48th call revisits the runner-up and relocks on a >3%
   win, so late page migration is not missed.  First governor call also
   queries the ACTUAL placement of ~128 sampled pages of each caller buffer
   (`get_mempolicy(MPOL_F_NODE|MPOL_F_ADDR)` — a pure read; the monitor
   banned move_pages *migration* and explicitly asked for this diagnostic)
   plus `/proc/sys/kernel/numa_balancing`, and publishes both with the full
   probe table as `gov{fr=…,nb=…,…,lock=…}` in the description.
   `L8_GOV=0` disables; `L8_TEAM`/`L8_VARIANT` forcing also disables.
4. **sfence after NT chunks** before the worker's ack (adopted from
   L8_radix8 / L8_batchsimd mt_r1): NT stores are weakly ordered and the
   ack's release-store does not flush WC buffers.  mt_r1 was formally
   unordered here (never observed to fail; fixed on principle, ~0 cost).

### Operation count

Unchanged: 1248 vector FP + 896 shuffles per volume.  Variant 17 is the
seq3 memory schedule: 384+384 L1 loads/stores vs fused's 256+256, all
L1-resident.  The governor adds two clock_gettime per thread per call and
one 32-entry re-cut per weighted call (~1 us against a ~5.8 ms call).
Every variant in the plan remains ONE BIT CLASS — verified this round by
`cmp` on forced v17 vs v0 outputs at B=512 — so governor pick-flips can
never appear in tryout's cross-run bit compare or behind a scored number.

### Measured (wallaby, Gold 6448Y SPR, 32 threads on one socket, shared
login node; same-window A/B pairs, quiet minima)

| cell | mt_r1 code | mt_r2 code | note |
|---|---|---|---|
| B=1     | 0.327 / 0.327 us | 0.328 / 0.329 us | unchanged (phase-1 path untouched) |
| B=2048  | 32.29 us/call | **31.90 us/call = 0.0156 us/t** | mid grid byte-identical; within window noise |
| B=32768 | 2489 / 2492 us (sd 0.5%) | **2450 / 2452 us = 0.0748 us/t (sd 0.08–0.11%)** | −1.6%, 2/2 both sides, and 5× tighter spread |

Governor behaviour on wallaby (single socket for our 32 threads, so the
NUMA machinery has nothing to exploit — this is the null test): `gov{fr=0,
nb=1, seq3-nt+pfs/32=0.083, seq3AA-nt+pfs/16=0.091, fused-nt+pfs/32w=0.075,
seq3AA-nt+pfs/32w=0.075, lock=seq3AA-nt+pfs/32w}` — placement scan
correctly reads 0% remote, the weighted full team converges to the static
split, and the lock matches the arena's best.  Gov-off control
(`L8_GOV=0`): 2508 us vs 2450 governor-on — the on-buffer race is worth
~2% even with no NUMA asymmetry (it re-decides against the *driver's*
buffers rather than the surrogate).  New streaming arena on wallaby:
seq3AA-nt+pfs/24 and /32 = 0.074 vs fused-nt+pfs/32 = 0.075, plain 0.107.

Correctness: PASS rel_l2 = 2.24–2.27e-16 (tol 1e-12) at B = 1, 3, 33, 100,
2048, 4096, 24576 (odd-size weighted-cut exercise), 32768; repeatable
(bit-identical across runs) at every size; AVX2-only build
(`-mno-avx512f`) PASS at B=32768; warning-free with `-Wall -Wextra` at
native and at forced-variant builds.

### What did not work / was rejected, with the numbers

1. **A multi-second "commit to T=32 and wait for AutoNUMA" phase**:
   rejected by arithmetic before building.  The driver's whole execute
   lifetime at B=32768 is ~90 calls ≈ 0.5–1 s (warmup 5 + calibration ~13 +
   12×6 samples at 5.8 ms), while AutoNUMA's default scan delay alone is
   1 s and 512 MiB takes several scan periods — the migration payoff
   cannot fit inside one scored process.  (This is also why L6_pfa's stable
   175 GB/s cannot be migration: its pages must have been *allocated*
   spread — likely page-cache pressure spilling the driver's 1 GiB of
   anonymous pages off socket 0 — which is exactly what the governor's
   placement scan detects in one call.)  The 48-call runner-up revisit
   keeps a cheap hook into slow drift without betting samples on it.
2. **First tryout of the day read B=2048 at 43.7 us and B=1 at 0.642 us**
   — a cold/loaded wallaby window, not a regression: the same binaries in
   same-window A/B pairs read 31.9/32.3 and 0.328/0.327 vs the r1 build.
   Same lesson as r1's "only same-window comparisons are quoted", re-learned
   for one confused half hour.
3. **Dynamic per-volume work stealing** as the T=32 balancer: not built.
   The VERDICT (§5, "moved sideways") shows dynamic scheduling losing
   everywhere it was published (L8_batchsimd dyn2 0.0888 vs static 0.0352;
   L64_blocked S-dyn 63.6 vs 54.8) and static contiguous blocks are also
   what AutoNUMA needs (stable page ownership).  The weighted *static*
   re-cut keeps contiguity and gets the balance.

### Borrowed, plainly

* **sfence-before-ack for NT chunks**: L8_radix8 / L8_batchsimd mt_r1.
* **seq3-below-full-team as a live question**: L8_radix8's mth-3p-nt-pfs
  half-team win (0.173, the r1 panel best at this cell) plus my own arena's
  seq3-nt/32 > fused-nt/32 reading.
* **The placement-lottery diagnosis and the get_mempolicy/numa_balancing
  diagnostics**: the mt_r1 VERDICT §4 (its "cheap check" list, run in-plan
  since the harness cannot).  The governor is my construction, but its
  premise is the VERDICT's falsification of the socket-0 model, and its
  per-thread re-cut is L17_rader mt_r1's weighted-partition idea (their
  item 5) made adaptive.
* **Min-of-min exploitability of early calls**: fftw3_patient's scored win
  at this cell IS this statistic (5.27 ms minority mode over a 9.7 ms
  median); the governor spends the same free calls deliberately.

### Node predictions (stated to be scored)

* **B=1: 0.545–0.556 us** — byte-identical path, tied with batchsimd.
* **B=2048: 0.026 us/t, pick fused+pfs/32** — mid grid unchanged; any
  movement is machine state.
* **B=32768, the target.**  Three scenarios the gov{} string will
  distinguish: (a) pages spread (fr ≳ 15): weighted T=32 NT lands
  0.09–0.13 us/t and takes the cell from fftw's 0.161 outright;
  (b) pages socket-0 (fr ≈ 0): seq3AA-nt at 16–24 threads should shave the
  volume-boundary alias stalls off 0.176 → 0.165–0.174, and the weighted
  T=32 converges to nt16-equivalent (its far probe chunks cost <0.5%) — a
  narrow loss or narrow win vs 0.161; (c) fr small but nonzero with
  numa_balancing=1: between the two, and the 48-call revisit may relock
  late.  Either way the cell finally *publishes* fr and nb, which is worth
  a round even if the number only moves 5%.

### Next round

1. Read gov{fr,nb,...} from all three processes FIRST — it settles the §4
   placement hypothesis with driver-buffer measurements, per process.
2. If fr=0 everywhere and the cell is still lost: the remaining gap is
   single-socket stream efficiency; the §4.5 counter
   (ld_blocks_partial.address_alias) is compiled and one `-DL8_PMC=1` away
   if the node's perf_event_paranoid ever drops, and a 2-volume-deep
   software pipeline (issue A(v+1) loads before B(v) stores) is the next
   structural candidate.
3. If fr is large and weighted T=32 won: try pinning the weighted split to
   the measured per-volume page homes (assign volumes to the socket that
   owns their pages, not just by rate) — one get_mempolicy pass at lock
   time gives the exact map.
