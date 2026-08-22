# L45_mixedradix — strategy record (multicore phase)

Phase-1 history (how the serial PFA 9x5 kernel reached its current form,
rounds panel_r6..r11) lives in `../../geom/strategies/L45_mixedradix.md`.
This file starts fresh for the 32-core phase.  Scored cells: B=1, 16, 256.
Phase-1 baselines used below: node (CLX Gold 5218, panel_r11 scored) B=1
312.9 µs, B=2 310.3, B=16 405.2 µs/vol; wallaby (SPR Gold 6448Y, one-socket
32-thread runs) ST B=1 170.1 µs fast-window min, B=64 243.7 µs/vol streaming.

## Round mt_r1

### What I built

The serial kernel is arithmetically UNTOUCHED — same PFA 9x5 line codelet
(344 FMA-port ops + 78 shuffles per PW lines), same two-sweep schedule, same
r11 xmm tail lines, same 514,968 zmm FMA-port ops per volume.  What changed
is structural: the monolithic `body()` was split into two **range kernels**
per (width, mechanism) —

* `p1r(vin, vout, plane, plane2, x0, x1, ...)`: phase 1 (z pass + y pass)
  for x-planes [x0, x1) of one volume, per-thread plane scratch;
* `p2r(vout, t0, t1)`: phase 2 (in-place x pass) for flat (y,z) tiles
  [t0, t1), the last tile index being the one masked tail call —

and a threading layer was put around them:

**1. Persistent pinned pthread pool, not OpenMP regions** — BORROWED
wholesale from **L23_matrixsimd mt_r1**: their record measured one GOMP
`parallel`+join at 6.2–8.2 µs (T=8..32), and L17_winograd independently
measured ~5 µs and had its tuner reject OMP regions at B=1.  I did not
re-measure a dead idea; `fft3d_create()` spawns 31 workers once, pins each
to the CPU one throwaway OMP region says `OMP_PROC_BIND=close /
OMP_PLACES=cores` would use (their read-back trick, including the
unbound-run guard), workers first-touch their own 68 KiB scratch after
pinning (NUMA-local on the two-socket node), and `execute()` publishes a job
and release-stores one atomic generation counter.  Barriers are their
flag-array shape: per-thread padded arrival flags + one release word per
group, leader scans; epochs derive from (generation, in-job volume index) so
they only ever increase across tuner jobs of different shapes.

**2. Three decompositions, raced at plan time** (per-candidate min over
interleaved rounds, hysteresis 3% cached / 2% streaming, incumbent first):

* **serial** — the phase-1 kernel on the main thread (the B=1 floor; if one
  volume does not parallelise, the pick says so rather than faking it);
* **vol** — thread t owns volumes [B·t/T, B·(t+1)/T), full serial schedule
  on its own scratch, join-only sync (T ∈ {32, 16}, capped at B);
* **grp** — G threads per volume: member m does x-planes [45m/G, 45(m+1)/G),
  ONE group barrier, then phase-2 tiles [NTT·m/G, NTT·(m+1)/G).  One barrier
  per volume only: a member's phase 1 of volume b+1 touches only out[b+1]
  and private scratch, so it never waits on a sibling still in volume b's
  phase 2.  B=1 is grp with a single group (T swept over {32, 23, 16, 8});
  at batch cells grp G=2/4 also divides the number of volumes in flight by
  2/4 — the aggregate phase1-out → phase2-in reuse window, which is the
  node-L3-residency lever (32 volumes in flight = 44.5 MiB vs the node's
  22 MiB L3; 8 volumes = 11 MiB fits).

**3. Mechanisms as per-thread candidates**, from the serial streaming
winner: m0 none, m1 pfin+pfw (paced next-plane T1 input prefetch +
write-intent output prefetch), m2 cpy (y pass into an L1-hot plane image +
one ERMS rep-movsb per plane — deletes the cold-out RFO), m3 = m1 + phase-2
45-stream poke, m4 cpy+pfin, m5 cpy+pfin+poke.  pfw is never combined with
cpy: a prefetchw on `out` would reintroduce exactly the RFO that cpy
deletes.  DROPPED this round to shrink the surface: the r11 odd-column
tournament axis (LDCOL is now the fixed form), the phase-2 next-volume
re-cover, the perf/phase-split probes.  All three are restorable from the
phase-1 file if a node cell regresses ≥2% at an unchanged pick.

### Bit class (one)

Every decomposition assigns whole lines to threads and changes no per-line
arithmetic or accumulation order, and the PW variants apply the same scalar
DAG per lane, so all (mode × T × G × width × mech) combinations are one bit
class.  Evidence: the create()-time gate (every candidate vs a serial v0
reference at 1e-13) passed for all candidates in all cells; rel_l2 is
4.050e-16 at B=2 on BOTH wombat (AVX2 path) and wallaby (512-bit path),
identical to phase 1; the driver's two-run cmp is bit-identical everywhere.

### Operation count

Serial arithmetic unchanged: 1497 full-width calls × 344 zmm FMA-port ops +
90 xmm tail lines per volume.  Parallel overhead per execute: one dispatch
store + T arrival stores + (grp mode) one release word per volume per group.
Zero added FP work.

### Measured (wallaby, Gold 6448Y, 32 threads close/cores = one socket; dev
numbers, relative only)

| cell | this round | wallaby ST same kernel | speedup / par. eff. |
|---|---|---|---|
| B=1   | **33.3–34.8 µs** (pick grp T=23 v1) | 170.1 µs | 4.9–5.1× / 21% at T=23 |
| B=16  | **163.8–168.3 µs = 10.2–10.5 µs/vol** (pick grp16×2 v1) | 198.4 µs/vol | 19.4× / 61% |
| B=256 | **6433 µs = 25.1 µs/vol** (pick vol32 v2 pfpp) | 243.7 µs/vol (B=64) | 9.7× / 30% |

rel_l2 4.050–4.065e-16 in every cell (identical to phase 1), repeatable
(bit-identical across runs) at B=1, 2, 16, 256.  Setup 0.1–0.7 s.  AVX2
host (wombat, no AVX-512): B=2 63.2 µs/vol, PASS, repeatable.

In-arena tuner tables (wallaby, µs/vol, per-candidate min):

* B=1: grp23-v1 **38.3**, grp16-v1 37.7 (within the 3% gate → T=23 kept),
  grp32-v1 46.4, grp23-v2 47.5, grp8-v1 63.7, serial 239.9.  B=1
  parallelises ~5×, not 32×: at T=23 the compute floor is ~7.4 µs, measured
  34.6 — the residue is the plane-granularity critical path (2 planes ≈
  5 µs), the mid barrier, and 2.78 MiB of in+out crossing 23 L2s.  Honest
  accounting per the brief: B=1 is latency-bound; T=32 is *worse* than
  T=23 (barrier cost grows, critical path doesn't shrink: 45 planes still
  means 2-plane stragglers at 32 threads).
* B=16 (CACHED on wallaby's 60 MiB L3 — 44.5 MiB fits; the node's 22 MiB
  makes this cell streaming there): **grp16×2-v1 13.1** vs vol16-v1 22.3 —
  pairing threads on volumes is a 1.7× win here, from using all 32 threads
  (vol mode idles 16) and halving the volumes in flight.
* B=256 (streaming, nt=86 arena = 4× L3): vol32-v2-pfpp **18.5**, -pp 19.1,
  v1-pp 19.0, -cpinpf 20.5, -cpin 20.8, -cpy 22.9, grp16×2-pp 23.5,
  vol16-pp 28.9, -m0 29.1, grp8×4-pp 34.0.  Reading: paced prefetch is
  worth 36% even at 32 threads on this DDR5 socket (m0 29.1 vs 18.5); the
  RFO-deleting cpy family loses ~10% here; grouped modes lose when the
  whole socket's L3 is this big.  None of those three verdicts is expected
  to transfer to the node unmodified (DDR4, half the L3, caller pages all
  on socket 0), which is why all of them are fielded.

### What did NOT work / traps hit, with numbers

* **A 2.5×-L3 streaming tuning arena** (the phase-1 rule) ranked candidates
  ~45% faster than the driver's real B=256 (13.7 vs 25.1 µs/vol) — the
  arena was partially cache-resident and would have mis-priced exactly the
  RFO/prefetch mechanisms the regime decides.  Raised to 4× L3, clamp
  [32, 96] volumes; the floor of 32 also guarantees a T=32 vol candidate is
  exercised at full width.
* **grp G=4 on wallaby streaming: 34.0 vs 18.5 µs/vol** — halving volumes
  in flight buys nothing when the socket's L3 is 60 MiB, and the extra
  barrier + shorter per-thread streams cost 80%.  Kept in the pool because
  the node's 22 MiB L3 is the case it was built for.
* **One tryout window read B=256 at 16.0 ms min, sd 8.9%** (vs 6.43 ms,
  sd 0.04% minutes later) — wallaby's slow-window trap, fourth appearance
  in this entry's records; never read one window.
* Nothing failed correctness at any point: the range-kernel refactor +
  pool passed the gate and numpy on the first successful build.

### Borrowed from other entries (as the round demands)

* **L23_matrixsimd mt_r1**: the persistent pinned spin pool, flag-array
  arrival/release barriers, the GOMP-fork-costs-6–8-µs measurement that
  justifies both, the OMP thread→CPU mapping read-back (incl. unbound-run
  guard), the serially-filled tuning arena rationale (caller pages are
  socket-0 on the node), and the serial-row-as-B=1-floor tuner discipline.
* **L17_winograd mt_r1**: independent corroboration of the OMP-region
  failure and the flat-barrier win; their `omp_get_max_threads()`=128 trap
  on a raw wallaby shell → hard cap at 32 in create().
* From my own phase-1 record: the kernel, mechanisms, PPITCH, hysteresis
  and min-over-interleaved-rounds tuner discipline.

### Env knobs for node A/Bs (harness sets none)

`FFT45_MODE` (0 serial / 1 vol / 2 grp), `FFT45_T`, `FFT45_G`, `FFT45_V`
(0/1/2), `FFT45_MECH` (0–5), `FFT45_VERBOSE` (tuner table on stderr);
compile flags `-DFFT45_R10TAIL`, `-DPPITCH=48` still live.

### Node predictions (Gold 5218, 2 sockets × 16 cores — 32 threads SPAN
BOTH SOCKETS, caller's in/out entirely on socket 0)

* **B=1**: cross-socket barrier + UPI reads should hurt the wide teams;
  expect the T sweep to pick 16 or 23 and land **45–90 µs** (3.5–7× over
  312.9).  If serial wins outright, B=1 does not parallelise on CLX and the
  pick will say so.
* **B=16** (streaming there, unlike wallaby): expect a grp pick (G=2 or 4)
  with pp; **30–65 µs/vol**.  This is the cell where grp-G=4's 11-MiB
  reuse window and the cpin mechanisms (RFO deletion on DDR4) can invert
  the wallaby table — that inversion, either way, is the round's main
  node question.
* **B=256**: all pages on socket 0, so socket 0's memory controllers carry
  everything and 16 threads pay UPI; compulsory traffic is 2.78 MiB/vol +
  RFO/eviction extras.  Expect **45–110 µs/vol** and nt=32 tuning.  If the
  pick is vol16 (socket 0 only), the far socket is a net loss and r2 should
  build the NUMA-asymmetric split (socket-0 threads own all `out` stores)
  sketched in L23's record.

### Next round

1. Read the node's picks and the in-description candidate table off the
   leaderboard; re-head the pools to the node's choices and retire what
   took no picks.
2. B=1 residue: the 2-plane critical path and the single mid barrier are
   ~half the measured time.  Candidates: half-plane phase-1 units (z pass
   and y pass as separate dependent units on per-plane scratch with a
   2-thread mini-sync), and a socket-tree barrier if the node's flat scan
   is the cost.
3. If cpin/grp-G=4 win node B=16/256, pursue traffic further: a 64B-aligned
   staging buffer for phase 2 that enables true NT final stores (the
   current in-place x pass cannot NT: odd rows are 16-mod-64 misaligned).
4. Report parallel efficiency against 312.9/T from the node leaderboard, as
   the brief requires.
