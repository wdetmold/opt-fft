# L64_radix8 — multicore phase strategy record

Phase-1 (single-core) history: `../../geom/strategies/L64_radix8.md` (11 rounds;
final form: radix-8^2 per axis, split-complex AVX-512, odd-line-padded scratch,
fused pass2+3, tuner over store modes / prefetch mechanisms; node B=1 949.9 µs,
B=8 1262.9 µs/vol, fastest in all three phase-1 cells).  That kernel is kept
verbatim as the per-thread body; this file records the 32-core layer.

## Round mt_r1

### What changed

First multicore round.  Arithmetic, layout, padding, codelets: untouched
(1190 flops/line, ≈1.59 M vector FP instr/volume, same shuffle bill).  The
passes were range-parameterized (pass 1 takes an x-plane range, pass 2+3 a
ky-slab range, both take a per-thread line/xb buffer) and four schedulers were
built around them:

1. **split (mt=1)** — within-volume split, volumes sequential.  Pass 1
   `omp for` over x-plane pairs into a SHARED SC (each pair owns disjoint
   SCXS rows), barrier, pass 2+3 `omp for` over ky pairs (in-place SC columns,
   disjoint 16-line output rows), barrier.  Every partition boundary ≥ 1 cache
   line: no false sharing.  Shared SC first-touched in create() by the same
   static x-map pass 1 uses.
2. **vol (mt=2) / vdyn (mt=3)** — volume-parallel on per-thread scratch,
   zero sync; dynamic,1 twin for batch > team (borrowed from L36_mixedradix
   mt_r1: driver first-touches caller buffers on the main thread → socket-0
   pages → remote threads run slower → static split parks the near half).
3. **gang (mt=4, gsz ∈ {4,8,16}) — the round's winner at batch.**  The team
   splits into tsz/gsz gangs of ADJACENT threads (same socket under
   PROC_BIND=close when gsz | 16); volumes round-robin over gangs; within a
   volume each lane takes 64/gsz x-planes then 64/gsz ky-slabs, synchronized
   by a per-gang central-counter spin barrier (1 cache line per gang, phase
   counter, reusable without reset).  The gang's shared SC is lane 0's
   region: one socket owns it entirely.  This is the middle ground the brief
   hints at: split's cross-core transpose traffic drops by tsz/gsz, and
   unlike vol mode all 32 threads are busy at B < 32.  Direction credited to
   L36_mixedradix mt_r1 "Next" item 3 (per-socket gangs for B in [2,32));
   the spin-barrier gang implementation is mine.
4. **Scratch**: one mapping = shared SC + 32 per-thread regions (SC 4.46 MB +
   xb 68 KB + lb 8 KB, rounded to 2-MiB slices, MADV_HUGEPAGE), each region
   first-touched by its pinned owner inside create(), where the OMP pool is
   spun up; execute() re-enters the warm pool and creates nothing.
5. **Tuner**: create() races {split(T32), split(T16)+p1pf rows at B=1;
   split / vol-fused / vol-tiled / gang-g4/g8/g16 / vdyn at B≥2} × {store
   mode plain/NT/pfw} with the REAL threaded paths at bt=min(B,32) volumes
   (arena filled SERIALLY to reproduce the driver's single-socket page
   placement — L36_mixedradix's arena-fidelity rule), then slabpf {0,1,2}
   and scst {0,1,2} A/Bs on the winner (incumbents defend at 1%; grid
   non-first beats cand[0] by 2%).  Phase-1 micro-picks that read 0 on the
   node (propf, xb, fout) default off but stay env-forcible.
   New env: `FFT64R_MT=0|1|2|3|4`, `FFT64R_GSZ=n`, `FFT64R_T=n`; all phase-1
   flags keep their meanings.

### Operation count

Per-volume arithmetic identical to phase-1 r11.  Sync cost: split = 2 OMP
barriers/volume; gang = 2 spin barriers/volume (measured indirectly: gang-g4
B=8 per-call ≈ 337 µs for one volume per gang, barrier contribution not
visible above noise); vol/vdyn = 0.  Scratch = 33 × 6 MiB ≈ 198 MiB + the
tuner's transient bt×8 MB arena.

### What was measured (wallaby, 2×32-core SPR, gcc 11.4 — NOTE from
### L36_mixedradix mt_r1: with OMP_NUM_THREADS=32 and PLACES=cores the team
### lands entirely on socket 0 there, so wallaby prices neither UPI nor the
### two-socket barrier; node picks may differ, esp. T16 rows and gang-vs-vol)

All PASS rel_l2 = 4.459e-16…4.464e-16 vs numpy at B=1/2/5/8/32/128,
bit-identical re-runs everywhere (parallel bodies run the same codelets on
the same data in a different order; output is bit-identical to serial by
construction).  Scalar fallback (`-mno-avx512f`) PASSES at B=8 (68.1 ms/vol).
Builds warning-free under `-Wall -Wextra` with and without AVX-512.

| cell | pick (in-process tuner) | best µs/transform | notes |
|---|---|---|---|
| B=1   | split-T32-plain+p1pf1+slabpf0 | **64.2** (median 64.3, sd 0.2%) | MKL-threaded same host: 68.0 min / 89.5 median (sd 26%) |
| B=8   | gang-g4 (mode pfw/nt, coin flip) | **331.9/8 = 41.5** (repeats 333.1, 336.6) | MKL 274.5/8 = **34.3 — MKL still 1.21× ahead here** |
| B=128 | gang-g4-nt+slabpf1 | **6693.9/128 = 52.3** | MKL 14670/128 = 114.6 → **2.19×** |

In-process tuner tables (FFT64R_TUNEDBG, µs/vol, same process):

* B=1: split-T32 149–165 over modes, T16 rows 219–240 (wallaby's one socket:
  wider is simply better — do NOT read as the node's answer), serial 1228;
  slabpf A/B **0 123.4 / 1 135.5 / 2 144.9 → slabpf=0**: the phase-1
  single-core slabpf win INVERTS under 32 threads at B=1 (32 threads issuing
  slab prefetches into a shared L3 is pure contention, and with 2 kys/thread
  half the prefetches land in a neighbour's chunk).  scst plain 3/3 again.
* B=8: **gang-g4 50.8 / gang-g8 53.0 / gang-g16 77.0 / split 94.0 /
  vol 126.7 / tiled 155.7** (best mode each) — the gang layer is worth ~2×
  over split at B=8 exactly as the cross-core-transpose model predicts
  (split: 32 cores exchange every volume; gang-4: 4).
* B=128: **gang-g4-nt 50.6 / gang-g8-nt 51.4 / vol-nt 85.1 / vdyn-nt 85.4 /
  split-nt 108.8** — gang wins the streaming cell too, and the reason is
  residency, not sync: 8 gangs × 4.5 MB shared SC = 36 MB of live scratch
  (fits L3) against vol's 32 × 4.5 = 144 MB (thrashes it).  NT output
  stores, which the node REJECTED three times single-core, win by 20–30%
  at batch under 32-thread bandwidth contention — the expected regime flip,
  now measured.  vdyn ties vol static on wallaby's single socket (nothing
  to rebalance there); it stays in the pool for the node.
* Parallel efficiency (per brief, vs same-process serial twin): B=8
  eff 0.76 (1132/46.3/32), B=128 eff 0.50 (805/50.0/32), B=1 eff 0.32
  in-tuner; against phase-1's quiet-window serial 950 µs the benchmark B=1
  is 950/64.2 ≈ 14.8× ⇒ ~0.46.  Where the other cores go at B=1: the
  irreducible cross-core 3D-transpose traffic (pass-1 x-owners → pass-23
  ky-owners is all-to-all) plus two barriers against ~30 µs of arithmetic.

### What did NOT work / negatives with numbers

1. **xb under gangs at B=8** (theory: SC read-only in pass 2+3 cuts
   cross-lane coherence): 348.2 vs 337.3 µs — loses on wallaby exactly like
   phase-1 r6/r11.  Env-forcible for the node (`FFT64R_XB=1`), not defaulted.
2. **p1pf forced in gang mode at B=8**: 371.8 vs 343.0 µs — the next-plane
   input prefetch is contention, not latency-hiding, when 32 threads share
   the memory system.  Kept as tuner rows at B=1 only.
3. **slabpf at B=1 under 32 threads**: 135.5 vs 123.4 (see above) — phase-1
   prefetch wins do not survive 32-way contention by default; every one of
   them must be re-raced threaded (this round did: slabpf, scst, mode).
4. B=8 remains the weak cell vs MKL-threaded on wallaby (42.1 vs 34.3).
   Not closed: see Next.

### Borrowed

* **L36_mixedradix mt_r1**: the split/vol/dyn scheme taxonomy, the
  socket-0-pages argument for dynamic scheduling, per-thread first touch in
  create(), the serial-arena-fill fidelity rule, wallaby's
  "32 threads = one socket" caveat, and the gang direction (their "Next"
  item 3 — per-socket gangs, within-volume split inside a gang).  Their
  move_pages(2) analysis (legal-but-unsporting; not shipped) is adopted
  unchanged here: not shipped either.
* Phase-1 protocol carried over: greedy A/B chain with defend-at-1% bars,
  bit-identical twins only, env forcing for every axis.

### Node predictions (to be scored; wallaby→node band was 1.74–1.95×
single-core, but MT adds NUMA physics wallaby cannot price)

* B=1: split picked; T16 may beat T32 (two-socket barrier + UPI transpose);
  95–160 µs.  Watch the T16 rows in the pick string.
* B=8: gang-g4 or gang-g8; g8 = exactly one gang per node socket-half…
  the interesting question is whether gangs of 4 straddle nothing (16/4=4
  gangs per socket, clean) and whether NT or pfw wins there; 60–110 µs/vol.
* B=128: gang-g4-nt or g8-nt.  Node L3 = 2×22 MB; g8's 4×4.5 = 18 MB of
  live SC fits one socket's L3, g4's 36 MB needs both — g8 may flip ahead
  on the node.  90–160 µs/vol.  vdyn only wins if the UPI asymmetry is as
  large as L36 predicts.

### Next

1. Read the node pick strings (`pick[B=…]=…` in the description) before
   touching anything: T16-vs-T32 at B=1, g4-vs-g8 at B=8/128, nt-vs-pfw —
   they answer the NUMA questions wallaby cannot.
2. B=8 vs MKL: the remaining 1.27× (wallaby) is inside the gang — one
   volume through a 4-lane gang costs ~337 µs ≈ 2.8× serial speedup on 4
   cores.  The named residual is the gang-internal all-to-all (each lane
   reads 3/4 of its pass-23 input from sibling lanes).  Candidate move: a
   lane-blocked pass-1 store layout so each lane writes the ky-columns its
   OWN pass-23 chunk will read (reorder SC so pass 1 scatters by ky-owner),
   turning the all-to-all into lane-local writes + one read stream — a real
   layout change, one round of work.
3. If the node's B=1 pick is T16: try a 2-gang B=1 hybrid (one gang per
   socket, each doing half the volume's planes/slabs with a shared output)
   before spending on spin barriers for the full team.
4. Do not re-open: xb (lost threaded AND single-core on two machines),
   p1pf-at-batch, forced slabpf at B=1 T32.

## Round mt_r2

### What changed

The arithmetic, layout, codelets and passes are again untouched; this round
rebuilt the SYNC STRUCTURE of the gang layer around what the mt_r1 node
leaderboard proved.  The node data that drove it: L64_blocked beat me
95.7-vs-146.9 µs/vol at B=128 with the SAME 8-gangs-of-4 decomposition and
the same NT stores — the entire 1.54× was their pipeline (one barrier per
volume, double-buffered group scratch, pass-2+3 stragglers overlapping the
next volume's pass 1) against my two barriers on a single SC.  Adopted, and
extended one step they designed but did not ship:

1. **mt=5 "gangp" — pipelined gang** (structure from L64_blocked mt_r1
   eng=2, stated plainly).  ONE spin barrier per volume.  The gang's SC
   double-buffers between lane 0's and lane 1's scratch regions (both
   first-touched by their same-socket owners in create(); zero new memory —
   lanes 1..gsz-1's SC regions were idle in mt_r1).  Volume k uses buffer
   k&1; the pass23(v_k)/pass1(v_{k+2}) WAR hazard on a buffer is ordered by
   the pass-1 barrier of v_{k+1}, which every lane crosses only after its
   own pass23(v_k).  A lane that finishes its ky-slice flows straight into
   the next volume's pass-1 input reads — on the node those reads are
   UPI-remote for socket-1 gangs (driver first-touches in/out on socket 0),
   which is exactly the high-variance straggling the overlap absorbs.
2. **mt=6 "gangd" — gangp + dynamic volume assignment.**  L64_blocked's
   "Next" item 2, unshipped there because of a leader-overwrite race; the
   race disappears when the claim is folded INTO the barrier: the
   last-arriving lane claims the gang's next volume with one relaxed
   fetch_add on a global counter (own cache line in the plan) and writes it
   into the barrier's own line BEFORE the phase release, so every waiter
   reads it race-free after its acquire.  Static round-robin makes the call
   wait on the slowest (UPI-remote) gang: with t_remote ≈ 1.5 t_local the
   work-conserving claim is worth ~20% at B=128 on paper.  Costs one
   atomic per volume per gang; identical output bit-for-bit (assignment
   changes which scratch a volume flows through, not the arithmetic).
3. **mt=7 "splitf" — flat-barrier split for B=1** (borrowed from
   L17_winograd mt_r1 via L36_pencilfused mt_r1): each arriver writes its
   OWN padded line, thread 0 scans (misses overlap) and publishes one
   release word — NOT the central-counter barrier L64_blocked measured
   losing at 32 waiters (89.6 vs 69.5).  B=1 is 2 barriers per ~30 µs of
   arithmetic, so barrier latency is on the critical path; raced at T32 and
   T16 (the node's mt_r1 pick), modes plain/pfw.
4. Both split variants now SKIP the trailing barrier of the last volume
   (the parallel-region join already orders it) — one less T32 barrier per
   call at B=1.
5. Tuner: gang rows race pipelined (mt=5) always, dynamic (mt=6) when a
   gang gets >1 volume, and keep two legacy two-barrier rows (g4/g8,
   nt/pfw) as insurance in case double-buffering thrashes the node's
   22-MB/socket L3; splitf rows at B=1.  cv widened 32→40 (34 rows live at
   B=128).  New env: FFT64R_MT=5|6|7 on top of the old meanings.

**Bug that cost half the round**: the first flat-barrier version seeded each
thread's epoch from its OWN flag line.  The tuner races T16 after T32, so
threads 16..31 kept stale epochs and the next T32 candidate livelocked
(thread 0 scanning for an epoch the stale flags could never reach — 15
cores spinning, found via ps on wallaby).  Fix: seed every thread's epoch
from the RELEASE line, which is the global maximum and identical for all
entrants.  If you build a persistent-epoch barrier raced at multiple team
sizes, seed from the release line, not per-thread state.

### Operation count

Per-volume arithmetic identical to phase-1 r11 (1190 flops/line, ≈1.59 M
vector FP instr/volume).  Sync per volume: gangp/gangd = 1 gang spin
barrier (was 2) + for gangd one relaxed fetch_add per gang; split/splitf =
2 barriers (last volume 1).  Live scratch per gang doubles to 2×4.46 MB
(uses lanes' otherwise-idle regions; mapping unchanged at 33×6 MiB).

### What was measured (wallaby, 2×32-core SPR, one-socket team — it prices
### neither UPI stragglers nor the two-socket barrier, i.e. NONE of what
### this round targets; the rows are raced so the node decides)

All PASS rel_l2 = 4.460e-16 (B=8/128) / 4.462e-16 (B=1) vs numpy,
bit-identical re-runs, every forced path checked independently
(mt=4/5/6 × g4/g8 at B=128, mt=6-g4 at B=8, mt=7-T32): all PASS 4.460e-16.
Scalar fallback (-mno-avx512f) PASS at B=8 (25.6 ms/vol).  Warning-free
under -Wall -Wextra with and without AVX-512.

| cell | pick (wallaby tuner) | best µs/transform | mt_r1 same host |
|---|---|---|---|
| B=1   | split-T32-plain+p1pf0+slabpf0 | **67.2** (median 68.0, sd 1.4%) | 64.2 (diff window) |
| B=8   | gang-g4-nt+slabpf1 (legacy)   | **336.2/8 = 42.0** | 41.5 |
| B=128 | gang-g4-nt+slabpf1 (legacy)   | **6863.9/128 = 53.6** | 52.3 |

In-process tuner tables (µs/vol, one window each):

* B=8: legacy gang-g4-nt 43.3 / gangp-g4-nt 44.0 / gangp-g8-nt 50.4 /
  gangd-g8-nt 51.4 / legacy g8-nt 51.2 / split-nt 81.7 / vol-nt 122.9.
  On one socket the pipeline neither wins nor loses at g4 (±1.6%): there
  are no NUMA stragglers to absorb, and 8 gangs × 2 volumes leaves little
  to rebalance.  Expected; the node physics is the target.
* B=128: legacy gang-g4-nt 49.4 / gangp-g8-nt 51.2 / gangd-g8-nt 51.4 /
  legacy g8-nt 53.8 / gangp-g4-nt 56.2 / gangd-g4-nt 56.3 / vdyn-nt 83.3 /
  vol-nt 85.0.  Two real findings: (a) **gangp-g4 loses 14% to legacy g4
  on wallaby** — 8 gangs × 2 × 4.46 MB = 71 MB of live SC against a 60-MB
  L3, the residency cost the insurance rows exist for; (b) **gangp-g8
  BEATS legacy g8** (51.2 vs 53.8) — 4 gangs × 8.9 = 36 MB fits, and the
  barrier halving shows.  On the node g8-pipelined is 2 gangs/socket ×
  8.9 = 17.8 MB per 22-MB socket L3: it fits THERE at the gang size whose
  legacy form won B=8, which is why I expect the node picks gangp/gangd.
* B=1: split-T32-plain 85.5 / splitf-T32 88.5 / split-T16 129.5 /
  splitf-T16 129.4–131.4 / serial 1036.5.  GOMP's tree barrier on one
  32-core socket is already good; splitf is for the node's split-T16
  one-socket pick, where a 16-flag scan on one socket should undercut a
  GOMP barrier built for 32.  slabpf A/B 75.9/82.4/88.3 → slabpf=0 and
  scst plain again at B=1; batch A/Bs kept slabpf=1, scst=0.

### What did NOT work / negatives with numbers

1. **Flat barrier with per-thread epoch seeding**: livelock, see above.
   (Not a performance negative — a correctness landmine documented so the
   next generation does not re-step on it.)
2. **gangp-g4 on a 60-MB-L3 one-socket machine at B=128**: 56.2 vs legacy
   49.4 — double-buffering is NOT free; it is a residency-for-overlap
   trade, and on a box with no stragglers it is pure cost at g4.  Do not
   delete the legacy rows.
3. gangd ≈ gangp on wallaby everywhere (51.4 vs 51.2 at g8-B=128): one
   socket has nothing to rebalance.  Its case rests entirely on the node's
   socket-0 page asymmetry; if the node tuner still picks static, the
   UPI-imbalance theory of my B=128 loss is wrong and the residual is
   elsewhere (measure before theorizing further).

### Borrowed

* **L64_blocked mt_r1** (the round's model): the one-barrier-per-volume
  pipelined gang with parity double-buffered scratch (their eng=2), the
  L3-residency accounting style for gang×buffer sizing, and their unshipped
  dynamic-volume-scheduling direction — shipped here with the claim folded
  into the barrier release to kill the leader-overwrite race they named.
* **L17_winograd mt_r1 via L36_pencilfused mt_r1**: the flat
  arrival-flag/release-word barrier design, adopted wholesale including
  their negative (no central-counter barrier at wide teams).
* Protocol carriers: same-window-only wallaby comparisons (L64_blocked),
  serial-arena-fill fidelity + env forcing for every axis (phase 1 /
  L36_mixedradix).

### Node predictions (to be scored)

* B=128: gangp-g8-nt or gangd-g8-nt, 65–100 µs/vol (blocked's 95.7 is the
  mark to beat; their pipeline + my slabpf/mode tuning and the g8-fits-L3
  argument are the case for beating it).  If gangd wins over gangp, the
  UPI-imbalance model is confirmed and worth a line in the verdict.
* B=8: gangp-g8-pfw or gangd-g8-pfw, 70–90 µs/vol; MKL's 73.7 is the bar.
  The claim-per-volume atomic is once per ~350 µs of gang work — noise.
* B=1: split-T32/T16 or splitf-T16, 120–140 µs; splitf only pays if the
  node's 2-socket GOMP barrier is the reason T16 won there.

### Next

1. Read the node picks: gangp-vs-gangd-vs-legacy at B=8/128 (the UPI
   question), splitf-vs-split at B=1 (the barrier question).
2. If gangd wins B=128 but MKL still leads B=8: the residual is
   gang-internal (the 4-lane all-to-all through L2/L3); the lane-blocked
   SC store layout from mt_r1 "Next" item 2 remains the one untried
   structural move.  Cost it against a perf-stat ask first.
3. If gangp-g4 wins on the node despite 22-MB L3: residency matters less
   than barrier count there; consider triple-buffering g8 to decouple two
   volumes of lookahead.
4. Do not re-open: xb, p1pf-at-batch, forced slabpf-at-B=1, central-counter
   wide barriers, per-thread epoch seeding (this round's landmine).
