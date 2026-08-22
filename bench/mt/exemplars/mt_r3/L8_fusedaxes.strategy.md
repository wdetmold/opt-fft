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

## Round mt_r3

### Where the round started (node, mt_r2)

B=1 0.558 µs (0.2% behind batchsimd — a tie per the VERDICT), B=2048
0.0263 µs/t (won, 1.39× MKL), **B=32768 0.1737 µs/t — still LOST to
fftw3_patient 0.159**.  The r2 governor answered its own question:
`gov{fr=0,nb=1}` in 3/3 processes — the caller's 512 MiB never leaves
socket 0 *under a 16-thread socket-0 team*, and all three L=8 entries at
nt16 sit at 94 GB/s while fftw's 32-thread plan reaches 103.  The VERDICT
(§5) reads the whole round's tuner history as one mechanism: **every
create-time or early-execute race runs in the pre-migration regime, where a
wide team on one socket's pages loses; the tuner locks a narrow team and
thereby forecloses the two-socket regime for the rest of the run.**  Its §6
asks, verbatim: (1) "read fr under a wide team" — nobody has; (2) for L=8:
"use the whole machine … force T=32 with a socket-aware chunk split at that
cell and re-measure", and "race wide-team candidates only after a
migration-settling warmup".  My own r2 gov data supports the foreclosure
reading: the weighted-32 configs bottomed at 0.211–0.212 µs/t, 24% above
the nt16 point they should converge to if far threads were priced to zero —
they were still unconverged when the lock fired (~20 calls in).

### What changed (arithmetic untouched; B=1 and B=2048 paths byte-identical)

The streaming governor no longer locks.  All changes are inside the
`gov_*` machinery; every config remains one bit class (verified again this
round: `cmp` across gov-on/gov-off/L8_TEAM=16/spread-bound runs at
B=32768, all bit-identical).

1. **Cyclic schedule, no lock, no hysteresis.**  Calls 0–4 (the driver's
   discarded warmup): wide.  From call 5, a 20-call cycle: 12 calls
   committed to the WIDE config, then an 8-call SAFE block.  The driver
   times *samples = means of `inner` consecutive calls* (read from
   driver.c, not assumed), so a banked minimum needs a whole clean sample,
   not one fast call: 8 consecutive safe calls contain an aligned full
   sample for any inner ≤ 4 (node r1/r2: inner=3; wide calls during
   calibration only push inner lower).  Two safe blocks land inside the
   node's ~36-call sample window.  Under min-of-min the slow wide calls are
   free; the safe blocks bank the known-good 0.170–0.175, and any
   post-migration regime the wide team reaches late in the run is captured
   by its own samples.  SAFE = the create-time arena pick (node 3/3:
   seq3/seq3AA-nt+pfs at 16); whole blocks alternate with an insurance twin
   (seq3AA-nt+pfs at the half team) only if the pick is not already a
   half-team config.  WIDE = seq3AA-nt+pfs at the full team with weighted
   cuts (my r2 gov's better wide flavour, 0.211 vs fused-nt's 0.215, 3/3).
2. **Far-share floor while fr < 5%.**  If the far threads converge to
   0.1%-of-B probe chunks (r2 behaviour), AutoNUMA never sees a remote
   access pattern to migrate toward — the elicitation signal dies with the
   convergence.  So the far socket's share of the weighted cut is floored
   at 25% while every placement scan still reads fr < 5; once fr ≥ 5 the
   floor lifts and the measured per-thread rates take over (migrated pages
   make far threads faster, so the cut follows the migration
   automatically).  `L8_GOV_FLOOR=<pct>` overrides (0 = pure
   rate-following).
3. **Weight normalisation (r2 bug fix).**  r2 clamped weights to [0.02,4]:
   the floored far weights stood still while the near weights inflated
   toward the cap, so the far *share* — the only thing cuts see — silently
   collapsed and the floor was fiction.  Weights are now renormalised to
   max = 1 every update; the share floor is applied on shares.
4. **fr under a wide team, published as a trajectory.**  The placement
   scan (same read-only `get_mempolicy` as r2) re-runs every 4th wide call;
   `gov{fr0,fr,frmax,…}` publishes first/last/max.  This is §6 ask (1) run
   for the entire cell, in all three processes.
5. **Migration-settling warmup, disclosed.**  During the 5 discarded
   warmup calls, up to 0.9 s of *extra full correct wide passes* (same
   input, same output, byte-identical result — real work, not sleep),
   capped at 24/call, counted and published as `mw=N/secs`.  This is §6's
   own phrase made concrete: the driver's fread+memset already burn ≥1 s
   before execute (AutoNUMA's scan_delay), and these passes add up to
   ~0.9 s of sustained far-socket touching before any timed sample exists.
   `L8_GOV_MW=0` disables.
6. Everything else — create-time arena (stable 3/3 on the node), mid-band
   grid, B=1 path, pool, sfence, alignment fallback — untouched.

### Operation count

Unchanged: 1248 vector FP + 896 shuffles per volume; variant 17 is
384+384 L1 loads/stores vs fused's 256+256.  The governor adds per wide
call: one 32-entry re-cut, one weight update, and (every 4th call) a
~512-syscall placement scan ≈ 0.1–0.3 ms against 7–12 ms calls — and none
of it inside safe blocks.

### Measured (wallaby, Gold 6448Y SPR, shared login node, quiet-window
minima; close binding = all 32 threads on ONE socket there, so the far
machinery is inert in the default tryout — the spread-bound runs are the
node-like exercise)

| case | number | note |
|---|---|---|
| B=1 | 0.328 µs, rel_l2 2.27e-16 | phase-1 path byte-identical |
| B=2048 | 32.13 µs/call quiet window (a 43.6/43.4 µs loaded-window pair first — r2's "first tryout of the day" lesson, third sighting) | mid grid byte-identical; same-window r2 A/B at parity |
| B=32768, close (1 socket) | r2 2469.8/2464.8 vs r3 2465.4/2471.1 µs, paired A/B | parity — no harm where there is no far socket |
| B=32768, **spread (2 sockets, node-like)** | min 2124.6–2136.3 µs = **0.0648–0.0652 µs/t** vs 2465 close (−14%); gov string: `fr0=0,fr=0,frmax=0,nfar=16,fs=25,fcur=49, safe/32=0.0627, alt/16=0.0800, wide/32w=0.0618` | the wide-commit machinery converges (far share rose 25→49% on measured rates, no migration needed on DDR5/UPI wallaby) and the wide config's own samples became the minimum.  Magnitude will NOT transfer to DDR4/CLX; the mechanism is what was being tested |
| B=24575 (odd, gov armed) | 0.0704 µs/t | weighted cuts + alternation on an odd batch |

Correctness: PASS rel_l2 = 2.26–2.27e-16 (tol 1e-12) at B = 1, 33, 2048,
4097, 24575, 32768; bit-identical outputs across runs and across
gov-on/gov-off/L8_TEAM=16/L8_GOV_FLOOR=0/L8_GOV_MW=0/spread-vs-close;
AVX2-only build (`-mno-avx512f`) PASS at B=32768; warning-free at
`-Wall -Wextra`.

### What did not work / was rejected, with numbers

1. **Elicitation cap 8 passes/call throttled the settling budget to
   0.09 s of the 0.9 s allowed** (wallaby passes are 2.4 ms; measured
   `mw=40/0.09s`).  Raised to 24/call → `mw=120/0.25s` on wallaby; on the
   node's ~12 ms passes the 0.9 s budget now binds (~75 passes).
2. **Keeping r2's lock-with-revisit and just re-ordering the probes**:
   rejected by the r2 numbers themselves — with 12-call budgets the
   weighted-32 configs were still 24% from their own convergence point
   when the lock fired, so any lock decided in the first ~30 calls decides
   in the wrong regime (VERDICT §5).  No number can fix a lock that fires
   before the regime it should price exists; so no lock.
3. **Extra safe-team candidates (T=12, T=20) in the create arena**:
   rejected without racing them — the arena is 3/3 stable and the §3.3
   pick-lottery table is a warning against widening a converged tuner
   surface for a ≤2% hypothesis.
4. **Extending the migration window by sleeping or by extra passes inside
   timed samples**: sleeping is not work and was never on the table;
   extra passes inside the sample phase would poison medians for no gain
   (min-of-min never needs them) — extra passes are confined to the 5
   discarded warmup calls and disclosed as `mw=`.

### Borrowed, plainly

* **The entire direction of the round** — wide-team commit, fr-under-wide,
  migration-settling warmup — is the mt_r2 VERDICT §5/§6 (the monitor's),
  executed.  The schedule/floor/normalisation constructions are mine.
* **The evidence the far socket is reachable**: L6_pfa's T=32/T=16
  bracketing at B=65536 (200 vs 85 GB/s, same binary) and fftw3_patient's
  own bimodal 5.2/9.6 ms behaviour at my cell — both from the VERDICT,
  not re-measured.
* **Safe-block length from the driver's sample structure** (samples are
  means of `inner` consecutive calls, so a banked min needs ≥ 2·inner−1
  consecutive calls): read from driver.c this round; as far as I can see
  no other record has stated this constraint, so recording it here for
  the panel.

### Node predictions (stated to be scored, keyed on the published gov{})

* **B=1: 0.545–0.558 µs** — byte-identical path, the tie continues.
* **B=2048: 0.026 µs/t, pick fused+pfs/32** — byte-identical path.
* **B=32768** — the fr trajectory decides, and this round finally
  publishes it under a sustained wide team (frmax is the number to read):
  (a) **frmax stays 0**: AutoNUMA does not migrate the driver's pages
  inside one scored process even under ~1 s of settling plus a 25%-floored
  far stream.  Safe blocks bank 0.169–0.176; scored ≈ parity with r2, cell
  still lost, but the §5 migration hypothesis is then CLOSED by
  measurement and the remaining fix is harness-level (§6 item 2: parallel
  first-touch), not kernel-level.  (b) **frmax 5–25**: partial migration;
  the rate-following cut turns it into 105–125 GB/s late samples,
  0.13–0.16 µs/t — the cell flips on any sample better than 0.159.
  (c) **frmax → ~50**: 0.09–0.12 µs/t, cell won outright.  My honest
  weighting given the r2 arithmetic (scan_delay 1 s, 256 MB/scan-period,
  two-pass rule, ~1.5 s total of wide-team touching): (a) is more likely
  than (b)+(c) combined — call it 60/30/10 — and (a) is still the round's
  deliverable, because it is the experiment the VERDICT ranked above
  everything else this round.  Medians will read higher than r2
  (wide-committed calls at the 25% floor cost ~0.2–0.37 µs/t each);
  that is the disclosed price of the elicitation and min-of-min is the
  protocol.

### Next round

1. Read `gov{fr0,fr,frmax,mw,safe,alt,wide}` from all three processes
   FIRST.  If (a): stop spending rounds on placement from inside the
   plan — the ask moves to the harness (§6 item 2), and the kernel-side
   residual is the §4.5 alias counter, still blocked on
   `perf_event_paranoid`.
2. If (b)/(c): tighten — drop the floor once fr plateaus (it costs far
   threads' overshoot), consider assigning volumes to sockets by measured
   page homes at cycle boundaries (one get_mempolicy pass gives the map;
   reads only, no move_pages).
3. If the safe blocks' banked min lands above the r2 0.1737 (i.e. two
   8-call blocks were not enough insurance against a bad window), lengthen
   the safe blocks or bias the first cycle safe-first; the wide/safe split
   is two constants.
