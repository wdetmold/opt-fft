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
