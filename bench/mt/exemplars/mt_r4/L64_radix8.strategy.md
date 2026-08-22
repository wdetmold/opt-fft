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

## Round mt_r3

### Where the node left me (mt_r2 scored)

B=128 **WON**: 69.5 µs/vol (pick legacy gang-g4/g8-fused-nt+slabpf1, the
three repeats flip g4/g8 — a tie) vs L64_blocked 90.5, fftw3_patient 164.4.
B=8 **LOST**: 90.6 (pick legacy gang-g8-pfw+slabpf1, all three repeats) vs
mkl 73.0, L64_blocked 76.7.  B=1 **LOST**: 136.2 (pick split-T16-pfw+p1pf1,
all three repeats) vs L64_blocked 127.0.  Two loud facts in the pick
strings: (a) the node chose the LEGACY two-barrier gang over gangp/gangd in
every batch cell and every repeat — mt_r2's pipeline/dynamic layer bought
nothing there (the UPI-straggler theory it was built on is now dead on this
data; wallaby had already shown the same at g4); (b) B=1 keeps choosing one
socket (T16), and L64_blocked's winning B=8 picks were G=2/G=4 groups at
nth=32 with NT — shapes with FEWER, BIGGER gangs than anything my legacy
pool raced (my legacy rows stopped at gsz=8).

### What changed (schedule/placement only; arithmetic, layout, codelets,
### passes byte-identical to mt_r2)

1. **scs16 — a NUMA-correct shared SC for the T16 split (the B=1 fix).**
   The shared split-mode SC is first-touched by the T32 static map: x-pair j
   → thread j, so planes 0..31 land on socket 0 and planes 32..63 on
   socket 1.  When the node picks split-T16 (it did, three rounds running),
   the 16 socket-0 threads write HALF of every volume's 4.46-MB scratch to
   socket-1 pages and read it back over UPI in pass 2+3 — a per-volume UPI
   round trip baked into the winning B=1 path, invisible on wallaby's one
   socket.  create() now first-touches one extra WREG slice with exactly the
   T16 map (thread t owns planes 4t..4t+3, threads 0..15 = socket 0 under
   close binding), and split/splitf use it whenever tsz==16.  With the
   driver first-touching in/out on socket 0, the entire B=1 T16 path is now
   socket-local.  At B=1 the whole working set (in 8 MB + out 8 MB + SC
   4.46 MB = 20.9 MB) now sits against ONE socket's 22-MB L3, which is also
   the regime where blocked's cached-store pick makes sense.
2. **Legacy gang gsz=16 rows at batch** (adopted from L64_blocked mt_r2:
   their node B=8 winners were G=2 and G=4 at nth=32, i.e. 8–16 threads per
   group; my tuner had never raced a one-socket-per-volume gang).  g16 keeps
   ONE live 4.46-MB SC per socket — the best possible L3-residency shape on
   a 22-MB-L3 node where the in/out streams sweep the same L3.  Two rows
   (nt, pfw) per batch cell, same insurance style as g4/g8.
3. **p1pf=2 — pass-1 input prefetch forced OFF, A/B'd on the batch winner.**
   In gang mode the next-plane prefetch turns on whenever nvol > ngang, so
   the node's B=8 pick (g8, nvol 8 > ngang 4) has always run WITH it, yet
   mt_r1 measured forcing it in a loaded gang costing 8% on wallaby and the
   off state was never a candidate.  New A/B after the scst chain: off must
   beat the incumbent by 1%.  Forced sanity check on wallaby: gang-g16-nt
   at B=8 improves 884.5 → 645.1 µs/call (−27%!) with p1pf=2 — the axis is
   real, and it matters most exactly at the big-gang shapes added this
   round (16 lanes × 16-line prefetch bursts into one socket's LFBs).
4. **Legacy gang drops the trailing barrier of each gang's last volume**
   (the parallel-region join orders it; split/splitf already did this).
   One less spin barrier per call; at B=8/g8 that is 1 of 4 barriers per
   gang gone.

New env: FFT64R_P1PF=2 (force off).  cv grows to 36 rows at B=128 (cap 40).

### Operation count

Per-volume arithmetic identical to phase-1 r11 (1190 flops/line, ≈1.59 M
vector FP instr/volume).  Sync: legacy gang = 2 barriers/volume minus one
per gang per call; gangp/gangd/split/splitf unchanged.  Memory: one more
WREG slice (mapping 33 → 34 × 6 MiB ≈ 204 MB); nothing else moves.

### What was measured (wallaby, 2×32-core SPR, one-socket team — it can
### price NONE of what changes 1–3 target; every new row is raced, the
### node decides.  Same-window final pass:)

All PASS rel_l2 = 4.460e-16 (B=8/128) / 4.462e-16 (B=1) vs numpy,
bit-identical across runs.  Forced-path checks all PASS with the same bit
class: split/splitf × T16 (the scs16 paths) / T32 at B=1; legacy g16 nt,
g16 nt+p1pf2, g8+p1pf2, gangp-g16, gangd-g16 at B=8.  Scalar fallback
(-mno-avx512f) PASS at B=8 (26.7 ms/vol).  Warning-free under
-Wall -Wextra with and without AVX-512.

| cell | pick (wallaby tuner) | best µs/transform | mt_r2 same host |
|---|---|---|---|
| B=1   | split-T32-plain (T16 rows now on scs16) | **61.3** (median 64.4) | 67.2 |
| B=8   | gang-g4-nt+slabpf1+scst1 | **333.1/8 = 41.6** | 42.0 |
| B=128 | gang-g4-nt+slabpf1 | **6529.5/128 = 51.0** | 53.6 |

Tuner tables (one window, µs/vol): B=8 legacy g4-nt 42.6 / gangp-g4-nt 43.2
/ legacy g8-nt 48.6 / g16 rows 65.7–66 (one socket: a gang spanning the
whole team has nothing to pipeline against, expected loss — the g16 case
rests entirely on the node's two-socket L3 split).  B=128 legacy g4-nt 50.3
/ g8-nt 52.8 / gangp-g8-nt 57.6 / g16-nt 75.4.  p1pf A/B on the g4 winner:
on 41.6–49.5 / off 42.0–50.0 → stays ON on wallaby (its SPR memory system
eats the prefetch; the 27% forced-g16 result above shows the off state is
live where it matters).  B=1 forced: split-T16 99.7 / splitf-T16 109.3 /
split-T32 68.5 (one socket → T32 wins locally, as always; scs16 correctness
exercised, its NUMA effect unmeasurable here).

### What did NOT work / negatives with numbers

1. Nothing regressed on wallaby this round; the risk surface is small by
   design (new rows race, incumbents defend).  For the record: my first
   B=128 tryout of the round read 60.3 µs/vol and the repeat 51.0–55.3 —
   another wallaby window flip (sd within-run was 1.35%); reconfirms the
   standing same-window-only rule before anyone chases a phantom.
2. Negative inherited from the node, recorded so nobody re-ships it:
   **gangp/gangd lost to the legacy two-barrier gang in every mt_r2 node
   cell** (pick strings, 3 repeats each).  Double-buffered pipelining and
   dynamic claims are dead at L=64 on this node unless a future round shows
   a straggler signature the picks can see.

### Borrowed

* **L64_blocked mt_r2**: the fewer-bigger-gangs evidence (node B=8 picks
  G=2/G=4 at nth=32, 76.7 µs/vol) → my legacy g16 rows; and their B=1
  eng=1-S-nth16-cached win → the L3-residency reading of B=1 above.
* **L23_matrixsimd mt_r1 (via L64_blocked's record)**: the driver
  first-touches in/out on thread 0 / socket 0 — the fact that makes the
  T32-mapped shared SC a handicap for a T16 team and scs16 the fix.
* My own mt_r1 negative (p1pf-in-gangs −8%) promoted from "env-forcible"
  to a real A/B; L64_blocked's mt_r2 nth=16-at-batch loss (their forced
  rows) is why I did NOT add T16 batch rows.

### Node predictions (to be scored)

* B=1: split-T16 (or splitf-T16) on scs16.  If the UPI-scratch model is
  right the pick drops from 136 to ~110–125 and takes the cell from
  blocked's 127; if it stays ≥130 the model is wrong and the residual is
  the barrier or the 20.9-vs-22-MB L3 squeeze, not placement.
* B=8: legacy gang g8 or g16, nt or pfw, possibly +p1pf2.  g16-nt+p1pf2 is
  the shape blocked's winners point at; 75–88 µs/vol.  mkl's 73.0 needs
  everything to land right — call it reachable, not expected.
* B=128: hold with legacy g4/g8-nt+slabpf1, 65–75 µs/vol; g16 could
  surprise if one-SC-per-socket residency beats gang-level parallelism at
  32 volumes/gang.
* Monitor asks, in cost order: (1) FFT64R_TUNEDBG=1 once per B — the full
  table + the p1pf A/B line answer every axis; (2) if B=1 still loses, one
  forced FFT64R_MT=1 FFT64R_T=16 run under `numastat`-style inspection
  (or FFT64R_T=32 for contrast) prices the scs16 effect directly.

### Next

1. Read the node picks: T16-on-scs16 at B=1 (the placement question),
   g16-vs-g8 and p1pf on/off at B=8 (the residency/prefetch questions).
2. If B=8 still trails mkl with g16+p1pf2: the remaining structural idea is
   splitting the OUTPUT write of pass 2+3 from the compute — a
   store-specialist lane per gang that drains z-line results while compute
   lanes run ahead (producer/consumer inside the gang; blocked's mt_r2
   "Next" item 1 is the same direction).  Cost it against a perf-stat ask.
3. If B=1 wins on scs16, the same placement audit applies to the gang SCs
   at batch (lane-0 regions are already socket-local — verified in code —
   but the T32 shared SC used by split rows at batch still has the split
   first-touch; irrelevant while split loses at batch, remember it if that
   flips).
4. Do not re-open: xb, forced slabpf-at-B=1, central-counter wide barriers,
   per-thread epoch seeding, gangp/gangd on this node (mt_r2 node verdict,
   3 repeats × 3 cells), nth<32 batch rows (blocked's mt_r2 forced loss).

## Round mt_r4

### Where the node left me (mt_r3 scored)

B=128 **REGRESSED 2.05x**: 142.3 us/vol (142.8/143.8/142.3, sd 1.1%) vs my
own mt_r2 69.5, blocked 73.7, on an IDENTICAL pick string
(`gang-T32-g8-fused-nt+slabpf1+sc0+p10`, 3/3) whose in-arena price was
unchanged (81 -> 82-85).  The verdict (S3.4) exonerates the tuner's ranking
and convicts the execute path / process memory state; its two named suspects
are mt_r3's change 4 (the conditional trailing gang barrier) and change 2
(the 34th WREG slice + its create-time num_threads(16) region).  Bandwidth
accounting from the verdict: r2 121 GB/s, r3 58.9 GB/s at the SAME ~8.4
MB/vol of DRAM traffic -- same bytes, half rate.  B=8 LOST again (91.0 vs
mkl 72.4, blocked 77.1).  B=1 LOST 136.3 vs blocked 128.7 -- scs16 failed
its own pre-registered criterion (">=130 means the model is wrong": 136.3)
and the verdict orders it not re-opened.  One more loud fact: blocked's r3
B=128 winner was MY r2 shape (their eng2 G=8 = 8 gangs of 4, static, one SC
per gang, nt+slabpf) at 73.7 -- while my arena shipped g8, which the arena
has priced as a tie with g4 in every round (81-vs-84-class) and the driver
priced at 142.3-vs-(g4 never shipped in r3).

### What changed

1. **Bisect arm -- both r3 suspects reverted.**  The legacy gang's trailing
   barrier is unconditional again (exactly r2's sync structure), and scs16
   (field, 34th slice, 16-thread create-time first touch) is deleted.  The
   batch execute path and the memory map (33 x 6 MiB) are r2's byte for
   byte.  gangp/gangd and g16 rows also leave the default pool (node
   rejected 3/3 in r2 AND r3; all still env-forcible), and the splitf B=1
   rows leave (never picked in two rounds on either machine).  The r3
   post-pick p1pf A/B is gone too; p1pf=2 is instead a first-class ROW
   (legacy g4/g8-nt-p1pf2) so combinations race honestly.
2. **Node-prior incumbent at batch >= 32** (adopted from L64_blocked mt_r3,
   who adopted the shape from my mt_r2 -- co-evolution, credited both
   ways): the arena at B=128 is not size-faithful (bt=32 < B) and it has
   priced g4/g8 as ties for three rounds while the node driver priced them
   69.5-vs-142.3.  The incumbent row is legacy gang-g4-nt (slabpf via the
   A/B), the shape both L64 entries measured winning this cell on the node
   (69.5 mine r2, 73.7 blocked r3); a challenger ships only if it beats the
   incumbent by >5% in-arena.  This kills the g4/g8 lottery that shipped g8
   in r3.
3. **mt=8 "gangt" -- the tiled structure goes multicore (batch rows,
   gsz 4/8 x modes plain/nt/pfw).**  Lane l of a gang builds 8/gsz whole
   z-octet slabs (pass A: y-FFT fill + in-slab x-lines, ALL lane-local, the
   slab L2-resident), one gang barrier, then pass B z-lines the lane's
   64/gsz kx planes reading 8 SEQUENTIAL 128-B-stride streams (one per
   slab).  This attacks the named B=8 residual -- the gang-internal
   all-to-all where each lane reads (gsz-1)/gsz of its pass-2 input from
   sibling lanes -- by changing the access PATTERN, not the (irreducible)
   bytes: the fused gang's 544-KiB-strided slab gathers become
   hardware-prefetchable streams, phase-1's "most prefetch-friendly memory
   shape this geometry allows".  Both L64 records had deferred this
   structural move; blocked's mt_r3 "Next" item 2 names the same direction.
4. **mt=9 "splitt" -- the tiled structure as a 3-phase within-volume split
   for B=1 (T32/T16 rows, modes plain/pfw).**  tps = tsz/8 threads share a
   slab: fill (slab, x-chunk) items, barrier, x-line (slab, ky-chunk)
   items, barrier, pass-B kx chunks.  This is the SHAPE of L64_blocked's
   B=1 pick (their eng=1 slab split), which has held that cell for three
   rounds; my version keeps my codelets and adds pass B's sequential reads
   in place of the fused split's strided ky-slab gathers.  All partition
   boundaries >= one cache line (x rows TXS*8 B apart, ky columns 128-B
   slots, out kx planes 8 KiB): no false sharing.
   Implementation note: pass A's fill was split out of the fused fill+xline
   body as a range-parameterized function (passAf_*), pass B took a kx
   range; the serial tiled path recomposes the identical instruction
   stream.  New env: FFT64R_MT=8|9 (with FFT64R_GSZ / FFT64R_T as before).

### Operation count

Per-volume arithmetic identical to phase-1 r11 (1190 flops/line, ~1.59 M
vector FP instr/volume) in every mode -- gangt/splitt reuse the phase-1
tiled kernels unchanged.  Sync per volume: legacy gang = 2 spin barriers
(r2's count, restored); gangt = 2 gang spin barriers (trailing skipped on
the gang's last volume -- a NEW mode, no r2 baseline to preserve); splitt =
2 OMP barriers + 1 trailing (skipped on the last volume).  Live scratch:
gangt = 4.23 MB/gang (8 slabs in lane 0's region, vs fused's 4.46);
mapping back to r2's 33 x 6 MiB = 198 MiB.

### What was measured (wallaby, 2x32-core SPR, one-socket team; same-window
### tables, sd quoted; wallaby cannot price the node's two-socket physics)

All PASS vs numpy: B=1 4.462e-16, B=8 4.460e-16, B=128 4.460e-16,
bit-identical repeats (tryout repeatability check).  Forced paths at B=8,
every one PASS 4.460e-16 AND bit-identical to the default pick's output:
gangt g4 / g8 / g2, splitt T32 / T16, legacy g4+p1pf2.  Scalar fallback
(-mno-avx512f) PASS at B=8 (26.7 ms/vol).  Warning-free under -Wall -Wextra
with and without AVX-512.

| cell | pick (wallaby tuner) | best us/transform | mt_r3 same host |
|---|---|---|---|
| B=1   | **splitt-T32-pfw** | **57.4** (median 59.6) | 61.3 |
| B=8   | gang-g4-nt-p1pf2+slabpf1 | **337.0/8 = 42.1** | 41.6 |
| B=128 | gang-g4-nt+slabpf1 (incumbent, also arena best) | **6601.9/128 = 51.6** | 51.0 |

Tuner tables (quiet window, us/vol):

* B=1: **splitt-T32-pfw 64.2 / splitt-T32-plain 67.2 / split-T32-pfw 74.6 /
  split-T32-plain 75.3 / splitt-T16 93.7-94.1 / split-T16 105.3-117.7 /
  serial 1027.8**.  The tiled 3-phase split beats the fused split by 14% at
  T32 and by 11% at T16 -- the first structural B=1 improvement since mt_r1,
  and it wins at BOTH team widths, including the T16 the node has picked
  three rounds running.  One extra barrier per volume, paid for by pass B's
  sequential reads.
* B=8: legacy g4-nt-p1pf2 46.9 / g4-nt 48.5 / g4-pfw 49.0 / gangt-g4-pfw
  49.1 / g8-nt 49.9 / gangt-g8-pfw 51.7 / gangt g4/g8 others 54-59.  On an
  idle one-socket machine the fused gang keeps a small edge; in one LOADED
  window (flagged: serial ref 3736, sd 17%, other jobs on the box)
  gangt-g8-nt won the table outright at 110.8 vs legacy g8-nt 130.3 --
  recorded as a hint, not evidence, but the hint says the stream shape wins
  under memory contention, which is the node's regime (DDR4, two sockets,
  UPI).  The rows race; the node decides.
* B=128: incumbent g4-nt 48.6 = arena best (margin rule held, debug line
  confirms) / g4-nt-p1pf2 48.4 (tie) / g8-nt 51.9 / gangt-g4-nt 54.6 /
  gangt-g8-nt 57.2 / vdyn-nt 84.4.  Wallaby driver 51.6-52.2 us/vol -- the
  revert did not regress wallaby (r3: 51.0).
* Parallel efficiency (in-process serial twin): B=1 1027.8/(57.4*32) =
  0.56; B=8 731.0/(42.1*32) = 0.54; B=128 906.5/(51.6*32) = 0.55.

### What did NOT work / negatives with numbers

1. Nothing regressed on wallaby; the round's risk surface is deliberately
   small (reverts + raced rows + an incumbent that only REFUSES near-ties).
2. Standing negative, promoted to a design rule this round: the bt=32 arena
   CANNOT distinguish gang shapes whose driver-regime difference is 2x
   (r2/r3 node data).  Any future shape decision at B>=32 must either ship
   through the incumbent-with-margin gate or come with node driver
   evidence.  Wallaby's quiet-window tables at B=128 are triply removed
   from the scored regime (one socket, DDR5, 256-MB arena).
3. The loaded-window gangt observation (110.8-vs-130.3) is explicitly NOT
   promoted to evidence: sd 17%, foreign load.  Left here so the next
   generation reads it next to the node's own verdict on the gangt rows.

### Borrowed

* **L64_blocked mt_r3**: the node-prior-incumbent-with-margin recipe
  (their B=128 fix, executed here at the same cell with my own r2 shape as
  the incumbent), and the B=1 slab-split shape (their eng=1, three-round
  winner) which splitt reimplements over my codelets.  Credit both ways:
  they took the g4 incumbent shape from my mt_r2 win.
* **mt_r3 VERDICT S3.4**: the entire bisect framing (identical pick, arena
  exonerated, execute path convicted, two named suspects -- both reverted
  here), and the instruction not to re-open scs16.
* My own mt_r1 "Next" item 2 / blocked's mt_r3 "Next" item 2: the
  turn-the-all-to-all-into-streams direction, finally shipped as gangt.

### Node predictions (to be scored)

* B=128: pick = incumbent gang-g4-nt+slabpf1 in 3/3 processes (it is also
  the arena favourite, so the margin rule is belt-and-braces).  If the r3
  regression was the g8 pick or either reverted suspect, this lands 66-85
  us/vol and retakes the cell (blocked's 73.7 is the mark).  If it STAYS
  >=130 on an execute path that is now r2's byte for byte at a pick the
  node measured at 69.5 in r2, the cause is process/memory state outside my
  code (the L6_unrolled S3.3 class) -- that outcome would be the bisect's
  most informative result, not a failure of this round.
* B=8: legacy g4/g8 (nt or pfw, possibly the new p1pf2 row) or gangt-g4/g8.
  If the loaded-window hint transfers, gangt lands 75-88 and mkl's ~72 is
  reachable; if not, expect ~90 again and the residual is confirmed to be
  bytes (the all-to-all itself), not pattern -- which would close this
  direction and point the next round at cross-volume pass fusion instead.
* B=1: splitt-T32-pfw or splitt-T16 (plain/pfw).  Wallaby's 11-14%
  tiled-split win, if it transfers at T16, puts the cell at ~118-125 vs
  blocked's 128.7.  Watch whether the node still prefers T16 over T32 with
  the tiled shape -- pass B's sequential reads change the UPI economics the
  T16 preference was built on.
* Monitor asks, in cost order: (1) FFT64R_TUNEDBG=1 once at B=128 -- the
  table plus the new "tuner incumbent" line show whether the incumbent held
  and what g8/gangt priced; (2) if B=128 still reads >=130: one forced
  FFT64R_MT=4 FFT64R_GSZ=8 run at B=128 -- if g8 is ~2x g4 in the DRIVER
  while the arena says tie, the arena-infidelity mechanism is confirmed
  directly; (3) FFT64R_TUNEDBG=1 at B=1 prices splitt-vs-split at T16 on
  real two-socket hardware.

### Next

1. Read the B=128 number first: it adjudicates the bisect (suspects vs
   process-state) and the incumbent policy in one shot.
2. If gangt takes B=8 or comes within a few %, the follow-up is a gangt
   pipeline (slab k+1 fill overlapping pass B of volume k needs only a
   second 4.23-MB slab set = lane 1's region -- the gangp trick on the
   tiled shape); if gangt loses badly, the all-to-all is byte-bound and the
   next lever is fusing pass B of volume k with pass A of volume k+1 to
   halve barrier count without extra residency.
3. If splitt takes B=1, try tps=2 at T32 (two slabs per 4-thread cluster,
   halving phase-2's cross-thread slab sharing) before anything exotic.
4. Do not re-open: xb, forced slabpf-at-B=1, central-counter wide barriers,
   per-thread epoch seeding, gangp/gangd on this node (r2+r3, 3/3), g16
   fused gangs (r3), scs16 (verdict order), nth<32 batch rows, splitf rows
   (two rounds unpicked), cross-window wallaby comparisons.
