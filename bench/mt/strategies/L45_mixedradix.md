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

## Round mt_r2

### Where r1 landed on the node, and the diagnosis

Node (Gold 5218, mt_r1 scored): B=1 **58.3 µs — won the cell**; B=16
**16.87 µs/vol — won**; B=256 **79.07 µs/vol — LOST 2.94×** to L45_pfa's
26.9.  The B=256 loss is a traffic story, not an arithmetic one: the vol
mode writes `out` in phase 1 (RFO read + later eviction write-back), then
phase 2 re-reads and rewrites it in place — and with 32 volumes in flight
(44.5 MiB) overflowing the node's 22 MiB L3, phase-1 `out` is evicted
before phase 2 returns to it.  That is ~5 DRAM/UPI passes of the caller's
buffers per volume (~6.9 MiB) against the compulsory 2.78.  L45_pfa's mtn
class (both phases in a private mid volume, one NT burst to `out`) pays
the compulsory 2 passes and their 26.9 µs/vol shows the node cell is
bandwidth-bound, so the ~2.5× traffic ratio is the ~2.9× time ratio.
Compounding it, my r1 tuning arena (4×L3 clamp [32,96] → 32 volumes on the
node) priced the picked candidate at 25.4 µs/vol where the driver's B=256
measured 79.1 — the exact mispricing L45_pfa hit at 32 volumes and fixed
at 128.

### What changed (this round)

1. **New decomposition `vnt`** — BORROWED from **L45_pfa mt_r1** (their
   mtn class, the node B=256 winner), with one addition of my own:
   * volumes are claimed **dynamically** off one atomic counter
     (`fetch_add`, ≤ B claims/job) instead of contiguous static blocks —
     whichever socket holds more remote caller pages runs slower per
     volume, and the counter rebalances it for free;
   * the claiming thread runs phase 1 AND phase 2 inside its private,
     NUMA-local 1.46 MB mid volume M (first-touched by the owner after
     pinning, 64B-aligned);
   * one linear `vmovntpd` burst flushes M → `out` (16 B head peel + 16 B
     xmm tail — their r1 lesson: NT demands 64 B alignment and out volume
     bases rotate 16 B since NVOL·16 mod 64 = 16), one `sfence` per thread
     per job before the join's release store (NT stores are not ordered by
     release semantics).
   `out` is touched exactly once per volume with no RFO: 2 caller-buffer
   passes per volume instead of ~5.
2. **New mechanisms m6 (pfin only) / m7 (pfin + phase-2 poke)** for vnt:
   M is cache-resident, so pfw/cpy pointed at it is a pure uop tax
   (L45_pfa / L23_matrixsimd r1 lesson); only the DRAM `in` stream is
   paced.  m7's poke covers the part of M that spills the node's 1 MiB L2
   (M fits wallaby's 2 MiB, so m7 can only win on the node).
3. **Streaming arena 32 → 128 volumes** (356 MB in+out) — adopting
   L45_pfa's measured fix verbatim rather than rediscovering it.
4. **Streaming pool re-headed**: vnt32-{pf,pfpk,m0} lead (the
   lowest-traffic class is the most robust to any residual arena-vs-driver
   mispricing, so it wins hysteresis ties), old incumbents vol32-v2-pfpp/pp
   stay as controls, vnt16 is the socket-0-only NUMA probe (wins only
   outright), grp G=2/4 keep the volumes-in-flight lever, and the full
   mech ladder (m0..m6) stays fielded for the node's B=16 streaming cell.
5. **B=1 T-sweep widened** with grp T=28 and T=20 (the best cross-socket
   team size is unanswerable on wallaby's one socket; let the node race
   say).  No other B=1 change: I analysed the half-plane split-sweep from
   my r1 notes and L45_pfa's, and at the picked team sizes (16–23) the
   critical path math gives ~zero: 90 half-units / 23 threads = 4 units ≈
   the same 2-plane path, plus a second barrier and a cross-thread
   slot handoff that converts my L2-hot plane reuse into L3/UPI traffic.
   It only pays at T=32; parked unless the node says B=1 picked T=32.

### Bit class (still one)

vnt reassigns whole volumes (dynamically, but assignment does not touch
per-volume arithmetic) and the NT flush is a copy, so every candidate
remains one bit class with the serial kernel.  Evidence: create()-gate at
1e-13 passed for all 15 streaming candidates; rel_l2 4.050–4.065e-16 at
B=1/2/16/64/256 (identical to r1 and phase 1); driver two-run cmp
bit-identical at every cell INCLUDING B=256 under dynamic scheduling, on
wallaby (512-bit) and wombat (AVX2, exercising the nt_flush_256 path).

### Operation count

Serial arithmetic unchanged (514,968 zmm FMA-port ops + 90 xmm tail lines
per volume).  vnt adds per volume: one staged copy of 182,250 doubles
(~22.8k NT stores, zero FP, replacing — not adding to — the phase-1 direct
`out` stores) and one relaxed fetch_add; per job per thread: one sfence.
Deleted per volume at B=256 scale: the `out` RFO, the phase-1→phase-2
eviction write-back, and the phase-2 re-read of `out` ≈ 4.2 MiB of
DRAM/UPI traffic (6.9 → 2.78 MiB compulsory).

### Measured (wallaby, Gold 6448Y, 32 threads; dev numbers, relative only)

| cell | mt_r2 | mt_r1 same host | wallaby ST | par. eff. |
|---|---|---|---|---|
| B=1   | 35.4–36.5 µs (pick grp T=16/23 v1) | 33.3–34.8 | 170.1 | 4.8× / 21–30% |
| B=16  | 195.8–252 µs = 12.2–15.8 µs/vol (pick grp16×2-v1) | 163.8–168.3 | 198.4/vol | ~16× / 51% |
| B=256 | **4428.7–4690 µs = 17.3–18.3 µs/vol** (pick vnt32-v2-pf) | 6433 = 25.1/vol | 243.7/vol (B=64) | 14.1× / 44% |

Same-window forced A/B at B=256 (the honest one; wallaby drifted between
windows all session): vnt32-pf **4451.6 µs** vs r1 incumbent vol32-pfpp
**6449.7 µs**, three rounds, −31% every round.  In-arena (nt=128):
vnt32-v2-pf **16.5**, vnt32-pfpk 16.8, vnt32-v1-pf 16.9, vnt32-m0 18.0,
vol32-pfpp 22.7, vol32-pp 22.7, vol32-cpin 22.9, grp16×2-pp 24.4,
vol32-pf 24.9, vnt16-pf 26.0 (half the cores of one socket — expected
loser here; it is a node probe), vol16-pp 28.4, grp8×4-pp 28.7.  The
arena (16.5) now sits within ~5% of the good-window driver (17.3), against
r1's 45% gap — the 128-volume arena fixed the pricing as L45_pfa said it
would.  B=1/B=16 paths are unchanged code; their spread vs r1 is
login-node window noise (the picks are identical to r1's).

### What did NOT work / traps, with numbers

* **vnt in the cached regime**: B=16 wallaby arena 46–59 µs/vol vs
  grp16×2's 21 — NT bypasses an L3 that is doing useful work, the same
  +55% failure L45_pfa and L17_rader measured.  vnt is fielded in the
  cached pool as a probe but is expected to lose there; do not force it.
* **Slow-window trap, fifth appearance**: one full tryout read B=256 at
  6153 µs (sd 0.6% — looked clean!) minutes before the same binary read
  4429; and a later window read 5478 at sd 10%.  A single window with low
  sd is still not trustworthy on wallaby; only same-window A/Bs and
  cross-window minima are.
* Nothing failed correctness at any point: the vnt mode, the head-peel NT
  flush and both ISA paths passed the gate and numpy on the first build.

### Borrowed from other entries (as the round demands)

* **L45_pfa mt_r1** (the bulk of this round): the NT-staged private-volume
  class that beat me 2.94× on the node — private mid volume, linear NT
  flush with 16 B head peel, sfence discipline, no-prefetch-on-M; and the
  128-volume streaming arena fix.  My additions on top: the dynamic volume
  counter (they use static contiguous blocks) and keeping the flush inside
  the existing pinned spin pool rather than OMP.
* **L23_matrixsimd / L17_rader mt_r1** (transitively): prefetch on
  cache-resident lines is a pure uop tax — shaped m6/m7.
* Retained from my own r1: the pool, barriers, tuner discipline, env
  forcing (FFT45_MODE=3 now selects vnt; FFT45_MECH up to 7).

### Node predictions (falsifiable)

* **B=256**: vnt32-pf or -pfpk picked, **24–30 µs/vol** (the cell is
  bandwidth-bound at pfa's 26.9; my dynamic balancing and m7's L2-spill
  poke are the only edges either way).  If vnt16 wins instead, the caller
  pages really are all socket-0 and r3 should build far-socket input
  staging.  If the pick lands >40, the arena is still lying and r3's fix
  is tuning on the driver's own first calls.
* **B=16** (streaming on the node): grp or cpin family retained; vnt may
  invert it — either way the volumes-in-flight vs NT question gets its
  node answer.
* **B=1**: unchanged code path; expect 55–60 µs, T∈{16,20,23,28} — the new
  T values tell the cross-socket barrier story for free.

## Round mt_r3

### Where r2 landed on the node, and the diagnosis

Node (Gold 5218, mt_r2 scored): B=1 **57.549 µs, 2nd by 0.6%** (pfa 57.197 —
a tie); B=16 **16.822 µs/vol, 2nd by 1.07×** (pfa's g2-pfw, i.e. MY r1 pair
decomposition on their kernel with their pfin+pfw+poke+PFNX mechanism, won
at 15.751); B=256 **50.216 µs/vol, 2nd by 1.10×** (pfa's static-block mtn at
45.446).  My r2 repair worked (79.07 → 50.22, arena now honest: 51.5–53.5
in-arena vs 50.2 driver), but two facts frame this round:

1. **The mt_r2 VERDICT's central finding (§5): two page-placement regimes
   exist on the node.**  The driver fills `in`/`out` on its main thread
   (all pages socket-0; L8_fusedaxes *measured* fr=0 there), and a cell
   runs at ~64–94 GB/s in that regime — but L=6 B=65536 sustained
   **200 GB/s** on a T=32 static-block schedule, i.e. the caller's pages
   can end up effectively spread during the timed loop (AutoNUMA;
   `numa_balancing=1` is confirmed on).  pfa's r1 B=256 number, **26.9
   µs/vol ≈ 108 GB/s**, is only reachable in the spread regime; their r2
   regression to 45.4 (1.69×, "the round's largest") and my 50.2 are both
   ~64 GB/s = the socket-0 regime.
2. AutoNUMA can only migrate a page toward the thread that touches it if
   the page→thread map is **stable across calls**.  My r2 vnt claims
   volumes off a dynamic atomic counter — a page is touched from socket 0
   on one call and socket 1 on the next, so there is nothing for the
   balancer to settle.  pfa's mtn comment says it outright: "each thread
   touches the same volumes every call (prefetcher- and NUMA-stable)".

### What changed (serial kernel untouched again; zero new FP)

1. **New mode `vns`** — vnt with STATIC contiguous volume blocks
   (thread t owns [B·t/T, B·(t+1)/T)), i.e. exactly **L45_pfa's mtn shape,
   BORROWED with attribution**.  Same private NUMA-local mid volume M, same
   linear NT flush + per-thread sfence; only the volume assignment changes,
   so it stays one bit class.  vns rows LEAD the streaming pool
   (earliest-wins hysteresis): static and dynamic tie in the pre-migration
   regime the create-time arena races in, and static is the only one that
   can get faster during the driver's loop — this is the VERDICT §6
   "make the right class the incumbent, don't re-race it in the wrong
   regime" rule applied to my pool ordering.  vnt32-v2-m0 (the r2 node
   pick) stays as the dynamic control.
2. **New mechanisms m8/m9/m10 with PFNX** — BORROWED from L45_pfa's
   node-winning mechanisms (their `pfw` at B=16, `pfi` at B=256): during
   phase 2, pre-cover the first ~63 KB of the NEXT volume's input with two
   paced T1 prefetch lines per tile (phase 2 + the NT flush otherwise leave
   the DRAM read stream idle for ~half the volume time).  m8 =
   pfin+pfw+poke+pfnx (their g2-pfw analog, for grp), m9 = pfin+pfnx
   (their mtn-pfi analog), m10 = pfnx alone — fielded because the node's
   r2 vnt table disliked MY pfin pacing (m0 52.2 vs pf 56.8 µs/vol; my
   pfin bursts ~42 T1 lines per group vs their fine-grained 32-KB-ahead
   pacing, a plausible LFB-swamping difference on CLX I did not try to
   re-tune this round).  In grp mode each member pre-covers its OWN
   phase-1 plane range [x0,x1) of the next volume.
3. **Placement diagnostic** — BORROWED from **L8_fusedaxes mt_r2** (the
   instrument the VERDICT praised and asked to be pointed at a 32-thread
   streaming team): a read-only `get_mempolicy(MPOL_F_NODE|MPOL_F_ADDR)`
   scan of ~64 sampled pages of each caller buffer, on the 1st and every
   24th execute, only when an NT-staged mode (vns/vnt) was picked.  ~128
   syscalls ≈ 0.1 ms once per 24 calls — invisible to min-over-samples.
   Published as `gov{nb,fr0,fr,sc,n}` appended to the description, which
   the driver writes to JSON AFTER the timed loop, so `fr` is the
   post-loop placement.  This directly runs the experiment VERDICT §5
   names ("nobody has yet read fr under a 32-thread team at a streaming
   cell"): fr>0 at B=256 confirms the migration mechanism; fr=0 with a
   slow number says the spread regime needs something else entirely.
4. **Streaming pool re-headed** (10 rows): vns32-v2-{nx,m0,pfnx},
   vnt32-v2-m0 (control), vns32-v1-nx (512-bit probe), grp16x2-{v2,v1}-
   {ppnx,pp} (node B=16; m8 = pfa's winning mech, and the v1 width
   restored — my r1 B=16 winner was v1 and pfa's r2 win was pw4, while my
   r2 pool only fielded v2 grp rows), vnt32-v2-pfpk (r2 runner-up
   control).  **DROPPED** (zero picks in r1+r2, ≥10% off in every node
   table): all mode-1 vol rows, grp8x4, vnt16, and the cpy-family
   mechanisms from the fielded set (code kept, restorable).  Cached pool
   unchanged except one grp16x2-v2-ppnx probe; B=1 pool untouched (the
   cell is a 0.6% tie and the T-sweep answer was stable across r1/r2).

### Operation count

Serial arithmetic unchanged: 514,968 zmm FMA-port ops + 90 xmm tail lines
per volume.  vns = vnt's traffic exactly (2 compulsory caller-buffer passes
per volume, one fetch_add less per volume).  pfnx adds ≤1012 prefetch
instructions per volume (2/tile), zero FP.  The gov scan adds 128 read-only
syscalls per 24 executes, main thread only, modes 3/4 only.

### Measured (wallaby, Gold 6448Y, 32 threads; dev numbers, relative only)

| cell | mt_r3 | mt_r2 same host | pick |
|---|---|---|---|
| B=1   | 35.4 µs (min), PASS, repeatable | 35.4–36.5 | grp T=16/23 v1 (unchanged path) |
| B=16  | 193.6 µs = 12.1 µs/vol, PASS | 195.8–252 | grp16×2 (cached pool there) |
| B=64  | 1114.9 µs = 17.4 µs/vol, PASS, repeatable | — | vns32-v1-nx (that window) |
| B=256 | **4479.0–4626.9 µs = 17.5–18.1 µs/vol**, PASS, repeatable | 4428.7–4690 | vns32-v2-pfnx |

In-arena (nv=128, B=256, one window): vns32-v2-pfnx **17.3**, vnt32-v2-pfpk
17.7, vns32-v1-nx 18.1, vns32-v2-nx 19.7, vns32-v2-m0 20.3, vnt32-v2-m0
20.4, grp16x2-v1-pp 25.0, grp16x2-v2-pp 25.1, grp16x2-v1-ppnx 25.4,
grp16x2-v2-ppnx 26.4.  Reading: **pfnx is worth −15% vs m0 on wallaby's
DDR5** (17.3 vs 20.3) and static ≥ dynamic at equal mechanism (nx 19.7 vs
m0-dynamic 20.4; both within window noise — the static-vs-dynamic verdict
can only come from the node, where migration exists).  rel_l2
4.055–4.065e-16 at B=1/4/16/32/64/256 (identical to r1/r2/phase 1), gate at
1e-13 passed for all 10 streaming candidates, driver two-run cmp
bit-identical everywhere including the vns pick.  wombat (AVX2): B=4
autotuned PASS; FORCED vns32-v0-nx at B=32 PASS (exercises nt_flush_256 +
pfnx on the 2-lane path), 97–101 µs/vol.

### What did NOT work / traps, with numbers

* **m8 (poke+pfnx) loses on wallaby streaming**: grp16x2-v2-ppnx 26.4 vs
  -pp 25.1 at B=256, 22.5 vs 20.5 at B=64 — the phase-2 poke on a 60-MiB-L3
  socket is a uop tax, as every prior record predicted.  Fielded anyway,
  ordered FIRST among grp rows, because the node is where it won for pfa
  (g2-pfw 15.6 vs g2-pf0 17.3 in their node arena) and node B=16 is the
  cell being chased; if the node also prefers pp, the hysteresis order
  costs me nothing (pp sits right behind it).
* **Wallaby pick lottery, again (sixth appearance of the window trap)**:
  one B=64 window returned grp16x2-v1-pp=19.6 as its best row with
  vnt-pfpk at 19.3 and the vns rows at 20–22.5 — a whole-table inversion
  vs the window an hour earlier.  Same binary, same arena.  Nothing new to
  fix (min-over-interleaved-rounds already hardens the in-window race);
  recorded so nobody reads one wallaby table as a ranking.
* Nothing failed correctness at any point: vns, pfnx on all three widths,
  and the gov scan passed the gate, numpy, and the two-run cmp on the
  first successful build.

### Borrowed this round (attributions)

* **L45_pfa mt_r1/mt_r2**: the static-block NT-staged decomposition (vns is
  their mtn shape verbatim, including the NUMA-stability rationale their
  impl_2 comment states), and the PFNX next-volume-input pre-cover plus the
  poke+pfnx B=16 mechanism (m8/m9/m10 are transcriptions of their pfw/pfi).
* **L8_fusedaxes mt_r2**: the read-only get_mempolicy placement scan and
  the publish-in-description discipline (their `gov{fr,nb}` instrument),
  re-cadenced to 1st + every-24th call.
* **mt_r2 VERDICT §5/§6**: the two-regime mechanism this round's pool
  ordering is built around (static/lowest-traffic as incumbent, dynamic as
  challenger), and the directive to read fr under a wide team.

### Node predictions (pre-registered, falsifiable)

* **B=256**: pick vns32-v2-{nx or pfnx} (pfnx if my pfin pacing was the
  CLX problem rather than prefetch per se; nx if not).  If AutoNUMA
  settles during the loop: **gov fr rises above 0 and the cell lands
  27–38 µs/vol**, taking it back.  If fr stays 0: **44–50 µs/vol** (the
  socket-0 ceiling, pfnx worth a few % of the 50.2), and the gov line
  becomes the round's diagnostic payload: it would falsify the migration
  hypothesis for L=45 and point r4 at explicit wide-team warmup or
  harness-level first-touch instead.
* **B=16**: pick grp16x2 (m8 if the node likes the poke as it did for pfa,
  pp otherwise; v1/v2 is a coin flip there), **14.5–16.5 µs/vol** —
  closing most of the 1.07× via pfnx.  vns/vnt stay losers at this cell
  (nb=16 halves their team and NT bypasses a useful L3 half the time).
* **B=1**: unchanged code path, **56–59 µs**, pick grp T∈{23,28,32} —
  still a coin-flip cell against pfa.

### Next round

1. Read gov{fr0,fr} at B=256 first — it decides everything at that cell.
   fr>0: pursue the spread regime harder (e.g. an execute-time
   static-partition REMAP that reassigns block ownership once, keyed on a
   mid-run fr read, so both sockets' controllers carry their own pages).
   fr=0: the 26.9 target needs the harness first-touch fix the VERDICT
   recommends; say so and stop burning rounds on it.
2. If the node picked nx over pfnx, my pfin pacing is convicted on CLX:
   rebuild it as pfa's fine-grained 32-KB-ahead cursor (FFT45_PFD-style)
   instead of per-group plane bursts.
3. B=1 remains a tie decided by window luck; the half-plane phase-1 split
   only pays at T=32 (analysis in r2, unchanged).  Build it only if r3's
   node B=1 pick lands on T=32.

## Round mt_r4

### Where r3 landed on the node, and the diagnosis

Node (Gold 5218, mt_r3 scored): B=1 **58.798 µs, 2nd by 1.04x** (pfa's
mtf-t23 56.569); B=16 **15.305 µs/vol, 2nd by 1.04x** (pfa's mtf-blk
14.742); B=256 **47.687 µs/vol, LOST 1.78x** (pfa's omtn-pfi 26.763).
The round's instrument triple did exactly what it was built to do:

1. **My gov read fr=0 in all three processes** (`gov{nb=1,fr0=0,fr=0}`):
   under my spinning pool the caller's pages never moved and the cell sat
   at the socket-0 ~64 GB/s ceiling (arena 50.4-53.5, driver 47.7 --
   arena honest, regime wrong).  My r3 prediction pre-registered exactly
   this outcome and its number (44-50 µs/vol).
2. **pfa's gov read i=16/16, o=16/16** at the same cell: under their omtn
   -- the SAME static-block NT-staged decomposition, dispatched by one
   `#pragma omp parallel` per execute with their pool NAPPED -- the pages
   DID spread mid-run and the cell settled at ~108 GB/s.  Same node, same
   hour, same decomposition shape, same kernel family: the dispatch/
   process-state is the whole 1.78x.
3. The VERDICT (S3.3/S4.4) has independent evidence for the mechanism:
   L17_matrixsimd's own probe read clk512=2.29 GHz in its spin-pool
   process vs 2.89 in L17_winograd's OMP process, and the round's two
   biggest regressions both implicate live spinners.

I also ran the VERDICT-ordered five-minute check for pfa's 16-mod-64 `Pt`
misalignment in my own scratch: **negative** -- plane at block offset 0,
plane2 at 37440 B, mid at 69888 B, plane rows at 832 B pitch, all = 0 mod
64.  My r4 is therefore all dispatch, not alignment.

### What changed (serial kernel untouched, fourth round running; zero new FP)

1. **New mode `vno`** -- BORROWED from **L45_pfa mt_r3's omtn**, the node
   B=256 winner: the existing vns static-block NT-staged work (private
   NUMA-local mid volume, one linear NT burst to `out`, per-thread
   sfence), dispatched by ONE OMP parallel region per execute instead of
   the spin pool.  ctx[t] is still first-touched by pool worker t pinned
   to the CPU OMP thread t reports under close/cores, so scratch
   placement is identical under either dispatch.  vno rows LEAD the deep-
   streaming pool (earliest-wins hysteresis): the create-time arena races
   both forms in the pre-settled regime where they tie (measured: 20.2 vs
   20.2 µs/vol at equal mechanism, same window), and only the OMP form
   has node evidence of settling -- pfa's S6 rule, their measured twins.
   vns32/vnt32 rows stay fielded as the pool-dispatch controls.
2. **Pool NAP machinery** -- pfa's mutex+condvar nap with their counted
   quiesce (an OMP team never timeshares with a still-spinning worker),
   EXTENDED per-thread: mt_run sets a nap policy "nap all tids >= lo"
   every call (steady state = one integer compare).  vno naps the whole
   pool for the plan's life; pool picks with T < 32 nap the unused
   workers (B=1's grp T=23 pick used to leave 9 spinners dragging the
   all-core clock per S4.4); the serial row runs on idle cores.  A napped
   worker sleeps through generations it is excluded from, so on wake it
   resyncs its local generation under the mutex BEFORE the quiesce lets
   main publish another job (the one subtlety my per-thread variant adds
   over pfa's all-or-nothing form).
3. **New mechanism m11 "pfi"** -- pfa's FINE-paced input prefetch,
   transcribed: the next plane's 508 lines issued as 2*NGRP small bursts
   (24 lines at PW=4, 12 at PW=2) spread across BOTH phase-1 subpass
   group loops, instead of my old 12 bursts of 43 in the z pass only.
   The node convicted my coarse pacing in r2 (vnt-m0 52.2 vs -pf 56.8);
   pfa's fine pacing WON the settled regime (pfi 27.8 vs pf0 29.8).  My
   r3 next-round note ordered exactly this rebuild.
4. **B=1 pool reordered**: T=23 now leads.  The node arena ranked
   grp1x23 fastest in every r1/r3 process (56.8 vs the T=32 incumbent's
   57.9 in r3) but incumbent-first 3% hysteresis kept T=32; pfa won the
   cell at the T=23 level three rounds running.  Dropped grp1x8 (106.6,
   dead three rounds) and grp1x32-v2 (65.6).  Plus the partial nap above.
5. **B=16 (small-streaming) pool rebuilt**: the r3 node picks
   (grp16x2-v1-ppnx/pp) stay incumbents; behind them a G sweep --
   grp4x8 (4 volumes in flight = 11.1 MiB, the one that FITS the node's
   22 MiB L3), grp2x16, grp1x32 (pfa's whole-team-per-volume mtf-blk
   analog) -- for the node's arena to price.  vns16/vnt rows dropped from
   this cell (57.3-66.7 vs 15.2 in every r3 node table).
6. Env: FFT45_MODE=5 forces vno, FFT45_MECH up to 11 (pfi); gov scan now
   covers modes 3/4/5.

### Bit class (still one)

vno reassigns nothing arithmetically (same static blocks, same per-volume
schedule, same NT flush as vns; only the dispatcher changes); nap is pure
scheduling; pfi is prefetch-only.  Evidence: create()-gate at 1e-13
passed for every candidate in every cell on wallaby AND wombat; rel_l2
4.050-4.065e-16 at B=1/2/4/16/32/64/256 (identical to r1/r2/r3/phase 1);
driver two-run cmp bit-identical everywhere including the vno pick at
B=256 and forced vno-v0 on the AVX2 host.

### Operation count

Serial arithmetic unchanged: 514,968 zmm FMA-port ops + 90 xmm tail lines
per volume.  vno = vns traffic exactly (2 compulsory caller-buffer passes
per volume); it trades the pool's ~0.3 µs dispatch for one GOMP region
(6-8 µs per L23's measurement, 0.05% of a 12 ms B=256 call).  pfi issues
<= 528 T1 prefetch lines per volume phase 1 (same count as m6, smaller
bursts), zero FP.  Nap costs one integer compare per execute in steady
state and nothing during the timed region.

### Measured (wallaby, Gold 6448Y, 32 threads; dev numbers, relative only)

| cell | mt_r4 | mt_r3 same host | pick |
|---|---|---|---|
| B=1   | 35.5-39.1 µs, PASS, repeatable | 35.4 | grp T=23/16 v1 (reordered pool) |
| B=16  | 235.4 µs = 14.7 µs/vol, PASS | 244.9 = 15.3 | grp16x2 (cached pool here) |
| B=64  | 1132.3 µs = 17.7 µs/vol, PASS | 1114.9 = 17.4 | vno (window-equivalent) |
| B=256 | **4411.7 µs = 17.2 µs/vol**, PASS, repeatable | 4479-4627 = 17.5-18.1 | **vno32-v2-pfi** |

Same-window forced A/B at B=256, three rounds, order stable in all three:
vno32-v2-pfi **4408-4481 µs** (17.2-17.5 µs/vol), vno32-v2-pfnx 4526-4634
(+2.6%), vno32-v2-nx 5166-5191 (+17%), vns32-v2-nx 5165-5204 (pool
control -- the twins TIE at equal mechanism on this one-socket host,
exactly as pfa measured; the dispatch A/B is decided on the node or not
at all), vno32-v2-m0 5362-5460 (+22%).  In-arena (nv=128): vno-pfi 17.31,
vno-pfnx 17.29, vns-nx 17.71, vnt-m0 17.89, vno-nx 19.91, vno-m0 20.73,
vno16-nx 32.4 (expected on one socket; it is a node probe).  Reading:
**fine-paced pfi is worth -15% vs nx and -18% vs m0 on wallaby's DDR5**,
recovering everything my coarse pfin lost and more.  wombat (AVX2): B=4
autotuned PASS 64.3 µs/vol; B=32 autotuned PASS 74.2 µs/vol; forced
vno32-v0-pfi at B=32 PASS (exercises the OMP region + nt_flush_256 + the
2-lane pfi path).  Setup 0.1-1.1 s.

### What did NOT work / refuted, with numbers

* **My grp1x32 transcription of pfa's B=16-winning mtf-blk is refuted on
  wallaby**: same-window forced A/B at B=16 read grp1x32-v1-pp at
  579-746 µs/call (36-47 µs/vol) vs grp16x2-pp at 181-221 (11.3-13.8) --
  3x, in the cached regime where pfa's mtf ties their g2.  Cause, by
  inspection: my phase 1 splits 45 PLANES over the members (2:1
  imbalanced at G=32) where their mtf splits 540 (plane, lane-group)
  units.  The class is right (their node 14.74 proves it), my granularity
  is too coarse for it.  Fielded at the tail of the node pool anyway
  (the node's streaming regime is qualitatively different and the rows
  are cheap), but the honest fix is unit-granularity phase-1 range
  kernels, which is next round's item if B=16 is still lost.
* **grp4x8 / grp2x16 lose on wallaby** (251-288 / 315-361 µs vs G2's
  181-221, same window) -- expected: wallaby's 60 MiB L3 never asks the
  volumes-in-flight question.  They are node probes; r1's B=256 version
  of this lesson said the same ("none of those verdicts transfer").
* **pfnx vs pfi at B=256**: statistically tied in-arena (17.29 vs 17.31),
  pfi won all three same-window driver rounds by 2.6%.  Both fielded;
  pfi leads.
* **Wallaby window lottery, seventh appearance**: one B=1 arena window
  read every candidate 2-3x slow (grp1x23 81.8, T=16 77.9) minutes from
  a window reading the driver at 35.5 µs.  Same binary.  Nothing new to
  fix; recorded so nobody reads one wallaby table as a ranking.
* Nothing failed correctness at any point: vno, the nap protocol, pfi on
  all three widths, and both ISA paths passed the gate, numpy, and the
  two-run cmp on the first successful build.

### Borrowed this round (attributions)

* **L45_pfa mt_r3** (the round's core): the omtn OMP-region dispatch with
  the pool napped, including the nap_on counted-quiesce discipline and
  the rank-the-OMP-twin-first pool ordering; the fine-paced pfi input
  prefetch (m11 transcribes their 22-line-burst pacing); and their gov
  i=16/16 reading, which is the direct node evidence this round bets on.
* **L13_rader mt_r2 (intent) / L45_pfa mt_r3 (implementation)**: the
  pool-nap idea lineage; my per-thread nap-above-T variant extends it.
* **mt_r3 VERDICT**: S3.3/S4.4 (idle spinners drag the all-core clock --
  the reason the nap is per-thread, not just for OMP picks), the ordered
  Pt-alignment check (run: negative), and S6's "make the right class the
  incumbent" applied to both streaming pools.

### Node predictions (pre-registered, falsifiable)

* **B=256**: pick vno32-v2-pfi (or -pfnx within noise).  If the omtn
  mechanism transfers to my kernel, gov reads i/o spreading toward 16/16
  during the loop and the cell lands **26-31 µs/vol**, closing most or
  all of the 1.78x (pfa's same-shape number is 26.8; my pfi and their
  pfi are now the same mechanism).  If gov stays fr=0 under a NAPPED
  pool and a sleeping GOMP team, the spinner hypothesis is falsified for
  my process and the residual is something else in their runtime setup;
  the twins table (vno vs vns rows, same window, published in the
  description) will say which half of the change mattered.
* **B=16**: grp16x2-ppnx/pp retained unless the G sweep inverts on the
  node's L3; **14.5-15.5 µs/vol** (the pfi/nap changes barely touch this
  cell).  If grp4x8 or grp1x32 wins the arena there, the volumes-in-
  flight lever is confirmed and r5 builds unit-granularity fused ranges.
* **B=1**: pick grp1x23-v1-m0 (reorder makes the three-round arena
  favourite installable), 9 workers napped; **55.5-57.5 µs** -- the tie
  broken by ~1-2% of hysteresis correction plus whatever the nap buys.

### Next round

1. Read gov and the twins table at B=256 first: vno vs vns rows in the
   same node arena, and the driver pick's settled number.  If vno settled
   and won, retire the vns/vnt pool rows and consider the weighted
   socket-0-heavy partition on top of the spread regime.
2. If B=16 is still second: build unit-granularity phase-1 ranges (the
   (plane, group) work list pfa's mtf uses), which also unlocks an honest
   whole-team-per-volume row and the B=1 T=32 case in one change.
3. If B=1's nap+reorder did not close the 0.6-2%: the remaining levers
   are the half-plane hp split (pfa's B=1 arena had hp beating rr in
   every window) at T=32, or accepting the tie.
