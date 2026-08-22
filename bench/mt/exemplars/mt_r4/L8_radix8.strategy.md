# L8_radix8 — multicore phase strategy record

Phase-1 history (rounds 1–11, single core) lives in
`../../geom/strategies/L8_radix8.md`; this file starts at mt_r1.

## Round mt_r1

### What changed

The kernel arithmetic is untouched — same 52-instruction radix-8 codelet, same
copy-free AVX-512 transpose networks, same 2p/1f/3p shapes.  What was added is
a threading layer, deliberately minimal:

1. **B=1 ships SERIAL**, no parallel region at all, bit-identical to phase 1
   (fingerprint 1.308e-16, the hardwired 2p pick).  One volume is 8 KiB and
   ~0.6 µs of work; L13_direct mt_r1 measured the GOMP fork alone at 3–5 µs
   and monotone losses from intra-volume splits at a 4× larger volume
   (t2g2 +71 % … t32g32 +624 %), and L36_pfa/L23_matrixsimd measured
   5–8 µs GOMP region entry.  At a quarter of L13's per-volume work there is
   nothing to discuss: I did not spend the round rediscovering their result.
   B=1 parallel efficiency is therefore deliberately —: the phase-1 time IS
   the multicore time (wallaby 0.325 µs this round vs 0.324 in panel_r11).
2. **B>1 is batch-parallel over contiguous volume chunks**: thread t of a
   `min(32, batch)` team owns volumes `[nb·t/T, nb·(t+1)/T)` (ranges computed
   from the team OpenMP actually delivers) and runs the UNTOUCHED serial 3p
   per-chunk run function on it.  Zero synchronisation inside the region; one
   fork+join per execute.  Chunk boundaries are whole 8-KiB volumes, so no two
   threads ever share an output cache line, and the per-chunk next-volume
   spread prefetch works unchanged (only each chunk's last volume loses its
   prefetch: 1/64 of volumes at B=2048).
3. **Per-thread scratch = page-aligned 12-KiB slots** (3 whole pages each, no
   shared pages ⇒ no false sharing), allocated in `fft3d_create()` and
   **first-touched by the owning thread** inside a full-width parallel region
   that also spins up the OpenMP pool — the first timed execute creates no
   thread.  Slot 0 doubles as the serial paths' scratch.
4. **NT stores RETURN to the B>1 pool** (they were retired in phase 1, four
   rounds running, "hide the RFO, do not avoid it").  That was a single-core
   verdict: one core is fill-buffer-bound, not DRAM-bound.  32 cores are
   aggregate-bandwidth-bound, where NT cuts mandatory traffic from
   24 KB/volume (in + RFO + out) to 16 — and L64_blocked mt_r1 measured NT
   beating cached stores ~25 % at every engine at the memory wall.  A second,
   L=8-specific mechanism: at the node's B=2048 cell (in = 16 MiB, out =
   16 MiB, L3 = 22 MiB), NT keeps `out` OUT of L3 entirely, which lets `in`
   stay L3-resident across the driver's timing loop instead of 32 MiB
   thrashing 22.  Each thread issues its own `sfence` before the join barrier
   (a fence only orders the issuing core's stores).
5. **Tuner pool for B>1** (raced at create at the real batch size, min-of-7
   interleaved, 2 % hysteresis, unchanged protocol): `mt-3p-nt-pfs` (default),
   `mt-3p-pfs-pfw` (the phase-1 streaming winner, so the node can veto the NT
   inversion), `mt-3p-pfs`, `mt-3p-nt`, `mth-3p-nt-pfs`/`mth-3p-pfs`
   (HALF-team twins — the two-socket question: the driver first-touches all
   caller pages on one socket, so the far 16 threads of a close-bound
   32-thread team pay UPI for every caller line; wallaby's single 32-core
   socket cannot price this, the node tuner does), and `st-3p-pfs` (serial,
   offered only at batch ≤ 4×team where the fork cost could conceivably
   exceed the work).  **Every candidate is the 3p shape**, and threading,
   team size, NT stores and prefetch flavor change no arithmetic — the whole
   pool is ONE BIT CLASS, so the r11 single-class invariant survives the
   phase by construction (verified by `cmp`, below, not just argued).
   Arena strings still publish through `fft3d_description()`; `L8R_FORCE`
   still overrides for A/Bs; `L8R_TUNE_DEBUG=1` prints the table.

### Operation count

Unchanged per volume in every shape: 1248 vector FP (24 × 52-instr codelets,
the 56-flop optimum), 896 shuffles, 384+384 L1 loads/stores (3p).  The mt
layer adds zero arithmetic and one fork+join per execute.

### What was measured on wallaby (Gold 6448Y, one 32-core socket, 32 threads
close/cores; shared login node with multi-hour clock windows — the two
B=2048 driver runs this round read 44.2 and 73.8 µs for the same pick, so
only same-window and same-table comparisons are quoted)

| cell | phase-1 1-thread (wallaby) | mt_r1 | speedup | par. eff. |
|---|---|---|---|---|
| B=1 | 0.324 µs | **0.325 µs** (serial by choice) | 1.0× | — (deliberate, see §1) |
| B=8 | ~2.8 µs/call | 1.947 µs/call | ~1.4× | 18 % on 8 threads (fork-bound) |
| B=2048 | 0.429 µs/vol | **44.8 µs/call = 0.0219 µs/vol** | **19.6×** | **61 % on 32 threads** |
| B=32768 | 0.908 µs/vol (B=16384) | **2482 µs/call = 0.0758 µs/vol** | **~12×** | 37 % |

Where the missing cores went, honestly: at B=32768 the pick was cached-store
`mt-3p-pfs`, whose 24 KB/volume ÷ 0.0758 µs = 317 GB/s ≈ wallaby's DDR5 wall —
the cell is at the memory wall, exactly the brief's "more cores = more fill
buffers" regime, and no decomposition buys more DRAM.  At B=2048 wallaby's
60-MB L3 holds the whole 32-MB working set (the node's 22-MB L3 will not), so
its 19.6× is compute/L3-side; the node number will be bandwidth-shaped.
At B=8 the ~2–4 µs GOMP region is comparable to the work; `st-3p-pfs` and the
small-team splits trade places across windows — unscored cell, not chased.

Tuner tables (same-process, within-table valid; us/vol):
* B=2048 (nb=2048): mt-3p-pfs 0.0345 < mt-3p-pfs-pfw 0.0392 < mth-3p-pfs
  0.0473 < mth-3p-nt-pfs 0.1004 < mt-3p-nt 0.1129 < mt-3p-nt-pfs 0.1166.
* B=32768 (nb=8192): mt-3p-pfs 0.0275 < mt-3p-pfs-pfw 0.0300 < mth-3p-nt-pfs
  0.0453 < mt-3p-nt-pfs 0.0507 < mt-3p-nt 0.0494 < mth-3p-pfs 0.1056.

Two wallaby-specific readings, flagged as such: (a) **NT loses 1.8–3.4× on
wallaby** — its 60-MB L3 makes even the nb=8192 arena partially resident, and
NT forfeits every L3 hit.  This does NOT price the node, whose 22-MB L3 gets
~4 % residency at B=32768; there the 24→16 KB/volume traffic cut is the whole
game.  Same epistemic class as phase-1's wallaby/node inversions: the node
tuner decides, both families are installable, and either outcome is safe
(one bit class).  (b) pfw loses slightly to plain pfs on wallaby in both
tables — when DRAM-bound, prefetchw only moves the RFO earlier in a queue
that is already full.  Also kept installable; the node decides.

Correctness: PASS rel_l2 ≤ 1.92e-16 (tol 1e-12) at B = 1, 4, 8, 33, 64, 2048,
32768; repeatable (bit-identical) across runs at every size.  **Every B>1
candidate force-run via `L8R_FORCE` at B=2048 and checked against numpy
individually — all seven PASS at rel_l2 1.915e-16, and the outputs are
`cmp`-bit-identical across NT vs cached, full vs half team, and mt vs
serial**, so a cross-process pick flip can never put unchecked bits behind a
leaderboard number.  AVX2-only build (`-mno-avx512f`, B=64) and portable
build (`-mno-avx512f -mno-avx2 -mno-fma`, B=8) PASS and repeatable.
Warning-free under `-Wall -Wextra`.

### What was tried and did NOT work

Nothing failed outright this round — the two negative results are the
within-table NT and pfw losses on wallaby above, both recorded as
wallaby-can't-price-it rather than acted on (candidates stay installable).
Intra-volume splitting at B=1 was not attempted at all, on L13_direct's
numbers (their fork alone exceeds my whole transform by ~6×); that is an
adopted negative result, not an untested idea.

### Borrowed, plainly

* **L13_direct mt_r1**: the entire decomposition pattern — contiguous
  volume chunks from the actually-delivered team, per-thread page-slot
  scratch first-touched by its owner in create, B=1 serial-by-evidence.
* **L64_blocked mt_r1**: NT stores returning to the pool at the memory wall
  (their NT-beats-cached-25 % measurement), and the create-region pool warm.
* **L36_pfa / L23_matrixsimd mt_r1**: the 5–8 µs GOMP region cost numbers I
  used to justify serial B=1 and the `st-3p-pfs` small-batch candidate
  without re-measuring.
* From my own phase-1 lineage: the 3p streaming shape, spread prefetch
  cadences, tuner protocol (interleaved min-of-7, hysteresis, installable/
  probe split, arena publication) — all unchanged.

### Node predictions (stated to be scored)

* **B=1: same as phase-1's r11 number (~0.57), pick string `avx512-2p
  (fixed)`.**  Serial by design; any regression is code-layout noise.
* **B=2048: pick is the real experiment.**  If NT's keep-`in`-resident
  mechanism is real on a 22-MB L3, `mt-3p-nt-pfs` wins and the cell lands
  near out-write-bound (~8 KB/vol over ~100 GB/s socket-0 bandwidth
  ≈ 0.08–0.12 µs/vol plus fork).  If not, `mt-3p-pfs`/`mt-3p-pfs-pfw` at
  ~0.15–0.25 µs/vol.  Either way the arena string publishes the whole table.
* **B=32768: expect NT or cached 3p at 0.12–0.20 µs/vol**, bounded below by
  socket-0's DRAM (all caller pages live there; the far socket's cores add
  fill buffers but not channels).  Watch `mth-*` — if half team ties full
  team here, the UPI tax is confirmed and r2 should try an in-region
  socket-0-only team with the far cores idle.

### Next round

1. **Persistent pinned spin pool** (L36_pfa built one; L17_winograd's
   flag-array barrier numbers): the ~2–4 µs fork+join is ~5–8 % of a 44-µs
   B=2048 call and the whole gap at B≤64.  Adopt theirs rather than invent.
2. **Honest NT pricing for B=32768**: the arena cap (min(batch, ~5632–8192)
   volumes) gives the create-time race more L3 residency than the scored
   32768-volume run has, which biases the race toward cached stores.  If the
   node arena shows NT close behind cached at the cap, raise the cap for the
   race (create time is free) before believing the cached pick.
3. **Read the half-team arena numbers on the node** before doing anything
   NUMA-clever; wallaby cannot see UPI at all.
4. Still do not touch the codelet, the transpose networks, or the prefetch
   cadences.

## Round mt_r2

### Where mt_r1 landed, read before changing anything

Node (p51n1, mt_r1 leaderboard + my t_*.json arena strings): B=1 0.581
(pick avx512-2p fixed, arena 2p 0.572 < 1f520 0.577 < 1f 0.578 — B=1 stays
closed).  B=2048 scored 0.035 µs/vol (arena pick mt-3p-pfs at 0.046) against
L8_fusedaxes 0.026 and L8_batchsimd 0.028 — both rivals ran the FUSED kernel
shape over a persistent spin pool; NT lost 3.5× at this cell in all three of
my runs (mt-3p-nt-pfs 0.177–0.179 vs 0.046).  B=32768 scored 0.173 with a
**71 % spread**: the pick flipped across runs between mt-3p-pfs (0.152
in-arena) and mth-3p-nt-pfs (0.153) — a dead tie under default-first
hysteresis — and the cell's economics say the cached full-team pick is the
outlier run: all caller pages are socket-0 (driver fread/memset on one
thread), fftw's winning 0.161 is exactly 16 KB/vol over ~100 GB/s, cached
stores cost 24 KB/vol ≈ 0.24+, and L8_fusedaxes' scored pick was
fused-nt+pfs at T=16 (their arena: nt/16 0.176 < nt/24 0.186 < nt/32 0.208).

### What changed (three things, each answering a specific mt_r1 node number)

1. **The B>1 chunk kernel is now the fused 1f shape; 3p is demoted to
   probes.**  The 1f kernels were already in this file (ported from
   L8_fusedaxes in phase 1) — mt_r1 simply never threaded them.  Added an NT
   twin (`kernel1f_nt`): every 1f output store is one full aligned 64-B line
   via the f_off table, so `_mm512_stream_pd` drops in with no
   write-combining games.  The installable pool is ONE BIT CLASS by
   construction (y,x,z axis order; threading/team/NT/prefetch change no
   arithmetic), fingerprint rel_l2 2.27e-16; the 3p probes (z,x,y order,
   1.91e-16) are timed and published with '*' but can never be picked.
2. **Dispatch is a persistent pinned spin-wait pthread pool** built in
   fft3d_create, replacing the per-execute OMP region (which stays as the
   fallback and behind `L8R_POOL=0`).  Protocol taken whole from
   L8_fusedaxes mt_r1 (which took it from L36_pfa): epoch release word,
   per-worker 128-B ack lines, main = participant 0, publish-then-scan,
   park-on-condvar after 25 ms idle, plus L17_rader's all-workers-ack
   invariant and fusedaxes' own seq_cst fence between the epoch store and
   the parked read.  Workers pin to the exact CPUs read back from the
   harness's close/cores OMP mapping in the same create-time region that
   first-touches the per-thread scratch slots, so NUMA locality carries
   over.  After the tuner picks, the pool is SHRUNK to the picked team
   (serial pick tears it down; half-team pick rebuilds with 15 workers on
   the first close CPUs = socket 0 on the node).  B=1 builds no pool at all.
3. **The deep-streaming regime (ws > 3×L3, i.e. the B=32768 cell) defaults
   to NT + half team** (`mth-1f-nt-pfs`), executing what the r1 arena and
   fusedaxes' scored pick both said; cached full team stays installable so
   the node can veto.  The mid regime (B=2048 cell) keeps cached full team
   first (`mt-1f-pfs`) — NT's 3.5× node loss there is settled.  Also
   adopted: arena surrogate raised from 4×L3 to 8×L3 of volumes (clamp
   [8192, 16384]; fusedaxes' 2×L3 surrogate mispicked by 27 % in their r1),
   and per-trial work raised 1.5 → 3 ms with reps ≥ 2 (their reps=1
   coin-flip lesson).

### Operation count

Unchanged per volume: 1248 vector FP (24 × 52-instr codelets, the 56-flop
optimum), 896 shuffles; the 1f chunk shape does 256+256 L1 loads/stores per
volume where 3p did 384+384 — that difference, times 32 threads, is most of
the B=2048 gap to the rivals.  The pool adds 31 ack-line transfers per
execute and zero arithmetic.

### Measured on wallaby (Gold 6448Y, one 32-core socket, shared login node;
same-window comparisons only)

| cell | mt_r1 | mt_r2 | pick | note |
|---|---|---|---|---|
| B=1 | 0.325 µs | **0.324 µs** | avx512-2p (fixed) | serial, bit-identical to phase 1 (rel_l2 1.308e-16) |
| B=8 | 1.947 µs/call | **0.989 µs/call** | pool | fork tax halved by the spin pool |
| B=2048 | 44.8 µs/call (0.0219/vol) | **32.8–33.1 µs/call = 0.0160 µs/vol** | mt-1f-pfs (tuned) | −27 %; matches fusedaxes' wallaby 31.4–31.9 |
| B=32768 | 2482 µs/call (0.0758/vol) | **2476–2512 µs/call = 0.0756–0.0767 µs/vol** | mt-1f-nt-pfs | at wallaby's DDR5 wall already in r1; no change expected here |

Parallel efficiency (vs wallaby phase-1 single-thread): B=2048
0.429/0.0160 ≈ 26.8× on 32 threads (84 %); B=32768 ~12× (37 %) — the
missing cores are the DRAM wall, same arithmetic as r1 (16 KB/vol at
0.0756 µs/vol = 212 GB/s aggregate).

Pool vs OMP-region A/B (`L8R_POOL=0`, B=2048, three same-window pairs):
44.1 vs 48.5, 34.6 vs 36.9, 33.1 vs 38.1 µs — pool wins 3/3, −6 to −13 %,
consistent with fusedaxes' −10.4 %.

Wallaby tuner tables (same-process, µs/vol): B=2048 nb=2048: mt-1f-pfs
0.0215 (picked) ≈ mt-3p-pfs* 0.0215 < mt-1f-pfw 0.0217 < mt-1f 0.0219 <<
mth-1f-pfs 0.0451 < mt-1f-nt-pfs 0.0529 (wallaby's 60-MB L3 hides the 3p
scratch tax the node's 22-MB L3 exposed, and makes NT/half-team rightly
lose — the node re-races).  B=32768 nb=16384: mt-1f-nt-pfs 0.0670 (picked)
< mth-3p-nt-pfs* 0.0746 < mth-1f-nt-pfs 0.0761 << mt-3p-pfs* 0.1048 <
mt-1f 0.1101 ≈ mt-1f-pfs 0.1100 < mth-1f-pfs 0.1495.  Note NT now WINS the
wallaby streaming arena where it lost in r1 — the deeper 8×L3 surrogate
(16384 vols = 256 MiB vs 60-MB L3) removed the residency bias that r1's
8192-vol arena had; that was the exact mechanism of fusedaxes' 27 %
mispick, confirmed here.  Half team loses on wallaby's single socket at
both cells, as it must — only the node can price the UPI question.

Correctness: PASS (tol 1e-12) at B = 1, 3, 8, 33, 64, 100, 555, 2048,
32768; repeatable (bit-identical across runs) at every size checked.
**Every installable B>1 candidate force-run via `L8R_FORCE` at B=2048 AND
B=32768 and checked against numpy individually — all PASS, and all 1f
candidates (mt/mth/st × pfs/pfw/none × NT/cached) are `cmp`-bit-identical
(2.271e-16 / 2.272e-16), while the 3p probes pass in their own class
(1.914e-16 / 1.915e-16) and can never be picked.**  AVX2-only build
(`-mno-avx512f`, B=64) and portable build (`-mno-avx512f -mno-avx2
-mno-fma`, B=8) PASS and repeatable.  Warning-free under `-Wall -Wextra`
at the cascadelake target.

### What was tried and did NOT work

Nothing failed outright this round; the design was assembled from other
entries' already-paid-for negatives instead of new ones.  Specifically NOT
tried, with the number that pre-killed each: intra-volume splitting at any
B (L8_batchsimd's poolrt 0.407 µs at T=2 alone exceeds the ~0.27 µs
maximum saving); per-execute OMP region as the primary dispatcher (my own
A/B above, 3/3 loss); NT in the mid regime as default (node r1, 0.177 vs
0.046, 3/3); prefetchw combined with NT output (L8_fusedaxes r2's
fill-buffer clog, 1.6×).  One residual risk accepted knowingly: the
B=2048 wallaby arena cannot rank 1f vs 3p (both 0.0215 on a 60-MB L3), so
the 1f promotion rests entirely on the node's r1 cross-entry evidence —
if the node disagrees, the mt-3p-pfs probe published in every arena string
will say so without costing a pick.

### Borrowed, plainly

* **L8_fusedaxes mt_r1**: the entire spin-pool implementation (their code,
  lightly renamed; protocol lineage L36_pfa → them), the pool-vs-omp A/B
  design, the 8×L3 arena-depth rule, the ≥2-reps trial rule, and the
  B=32768 NT+T=16 target the streaming default now encodes.
* **L8_batchsimd mt_r1**: the no-prefetch threaded candidate (their
  T32/none won the B=32768 arena; prefetch +72 % at 32 threads at B=64),
  and the poolrt numbers that keep B=1 serial without re-measurement.
* **L17_rader mt_r1** (via fusedaxes): the all-workers-ack invariant and
  its double-run failure mode — not rediscovered.
* **L36_pfa mt_r1**: pool shrink after pick (idle spinners drag the
  all-core clock).
* From my own phase-1 lineage: the 1f fused kernels themselves (ported
  from L8_fusedaxes in panel_r5), the f_off full-line store table that
  makes the NT twin legal, spread prefetch cadences, and the
  installable/probe single-bit-class tuner discipline.

### Node predictions (stated to be scored)

* **B=1: unchanged, ~0.57–0.58, pick `avx512-2p (fixed)`** — byte-identical
  serial path; any movement is machine state.
* **B=2048: 0.026–0.031 µs/vol, pick mt-1f-pfs (default) with ",pool" in
  the description.**  The claim decomposes as: fused shape −20 %, pool
  −10 % off my r1 0.035; fusedaxes' 0.026 is the floor since my chunk
  kernel is now their shape over their dispatcher.
* **B=32768: 0.15–0.18 µs/vol and — the real prediction — the 71 % spread
  collapses to single digits**, because the cached full-team pick that
  produced the ~0.30 outlier run can no longer win a tie: the default IS
  mth-1f-nt-pfs and displacing it needs >2 %.  Whether mth (16 close = one
  socket) or mt-1f-nt-pfs (32) takes it prices UPI; either is installable
  and bit-identical.

### Next round

1. If B=2048 lands at 0.026–0.028, the shape/dispatch gap is closed and the
   next lever is L8_batchsimd's leaner-collect idea (tree release, or
   releasing only the caller's-socket workers): the 31-flag collect is
   ~1 µs of a ~55 µs call, so only worth it if the cell tightens further.
2. If the node's B=32768 pick is mth-1f-nt-pfs and it beats fftw's 0.161,
   the socket-0 story is confirmed; the follow-up is a weighted partition
   (L17_rader item 5) giving socket-0 threads larger slices, which wallaby
   cannot develop — build it only on node evidence.
3. If mt-3p-pfs's published probe time ties mt-1f-pfs on the node at
   B=2048, the 1f promotion argument was wrong in an interesting way —
   re-read the rivals' r2 records before touching anything.
4. B=1 stays closed unless someone's record shows a sub-0.2 µs dispatch.

## Round mt_r3

### Where mt_r2 landed on the node, and what this round is aimed at

B=1 0.581 (pick avx512-2p fixed; batchsimd/fusedaxes 0.557/0.558 with
same-shape fused serial kernels — a code-layout gap eleven phase-1 rounds
could not move, left closed).  B=2048 0.0279 us/t (57.19 us/call, pick
mt-1f-pfs 3/3) vs fusedaxes 0.0263 — and batchsimd's r2 proved the gap is
NOT the handshake (their tree collect landed 56.9, still 3 us behind
fusedaxes' flat-collect 53.9), so I did not spend this round on the pool.
**B=32768 0.1737, a three-way panel tie at 94 GB/s, LOST to fftw3_patient's
0.1591 (103 GB/s)** — the mt_r2 VERDICT's clearest single action ("L=8 —
use the whole machine") and this round's target.

### What changed: an execute-time governor (create arena untouched)

The mt_r2 evidence chain, restated because the design is built on it:
all three L=8 entries picked 16-thread single-socket teams; fusedaxes'
gov{fr=0,nb=1} proved the caller's 512 MiB sits on socket 0 the whole run;
their weighted-T32 probe read 0.211–0.217 ON THE REAL BUFFERS (so naive
full-team loses 22% when pages are socket-0); yet fftw's 32-thread plan is
bimodal WITHIN every process (median 9.6 ms, min 5.2) — its fast mode
ARRIVES DURING the run.  The coherent mechanism (VERDICT §5): with
numa_balancing=1, a wide team's far-socket threads keep faulting on
scanner-marked pages and AutoNUMA migrates them, unlocking the second
socket's controllers (L=6 B=65536 sustains 200 GB/s the same way).  A
create-time surrogate race can never see this, and fusedaxes' r2 governor
gave wide only ~12 calls (~70 ms) before locking away from it.  So:

1. **DEEP governor (ws > 3×L3, batch ≥ 4096, team ≥ 8, i.e. the B=32768
   cell).**  DWELL the first 40 executes at FULL-team NT with static
   contiguous cuts — socket-1 threads re-touch the upper half every ~7 ms,
   maximal migration pressure — then probe the half team 4 calls, lock the
   faster on recent-window minima (2% hysteresis toward half, the proven r2
   config), and REVISIT the loser with PAIRED probes every ~24 calls
   (pairs, because AutoNUMA migrates a private page on its second same-node
   fault), relocking on a >3% win.  The driver takes 30 samples (~130–195
   executes, ~0.9–1.2 s at this cell), so the lock at call ~44 leaves ~22
   clean samples, and a migration that completes late still flips the last
   ~half of the samples.  Probe/dwell calls are full correct executes and
   are free under the min-of-samples statistic — fftw's scored win at this
   cell IS that statistic, spent deliberately (fusedaxes' r2 argument,
   monitor-endorsed).  Placement is measured, not assumed: fr (% of 64
   sampled caller pages off node 0, raw get_mempolicy syscall — a pure
   read, the diagnostic the VERDICT asked to propagate) and numa_balancing
   publish as gov{fr0,nb,w,h,lock,fl,fr1}.  **fr1 — read after a sustained
   wide dwell — is exactly the "nobody has yet read fr under a 32-thread
   team at a streaming cell" experiment of VERDICT §5(4), run in-plan.**
2. **MID governor (the B=2048 cell): a short on-buffer race** over the
   cached full-team 1f trio {pfs, 520-pfs, no-pf} — 2 settle + 9 race
   calls, all inside the driver's ~350-call calibration phase, lock needs
   >1.5% over the create pick.  Rationale: the phase-B out-store 4K-alias
   count is set by (scr − out) mod 4096, an allocation lottery the
   surrogate re-rolls (LITERATURE §4.5, fusedaxes' fusedAA analysis), so
   only the real buffers can price the trio.  fusedaxes' gov-off control
   measured the on-buffer re-decision worth ~2% even with no NUMA
   asymmetry.  mt-1f520-pfs also joins the create-time installables (same
   bit class).
3. **A governed plan keeps the full spin pool** (the governor switches team
   width per call via job.T); the r2 node arena priced 31 idle spinners at
   ~0 at these DRAM-bound cells (mth in-arena 0.173 with full pool ==
   0.174 scored with a shrunk one).  L8R_GOV=0 disables, L8R_DWELL resizes,
   L8R_FORCE implies disabled.  B=1 and the AVX2/portable backends are
   untouched (governor compiles only under _OPENMP && __AVX512F__ &&
   __linux__).

**Bit-class invariant preserved by construction and re-verified by cmp:**
every config either governor can install is the fused 1f shape (y,x,z
order); at B=2048 all five reachable configs force-run and compared
bit-identical (rel_l2 2.271e-16), at B=32768 the deep pair likewise
(2.273e-16), and a gov-off run is bit-identical to both — so a lock flip
can never appear in a correctness, repeatability, or cross-process
comparison.

### Operation count

Unchanged per volume: 1248 vector FP (24 × 52-instr codelets, the 56-flop
optimum), 896 shuffles, 256+256 L1 loads/stores (1f).  The governor adds
two clock_gettime per execute (~50 ns against 5.7 ms deep / 57 us mid
calls) and three one-time 64-syscall page scans; zero arithmetic.

### Measured on wallaby (Gold 6448Y, 32 threads on one socket, shared login
node; same-window comparisons only — one loaded window this round read
B=2048 at 42.9 and B=1 median at 0.63, quiet windows below)

| cell | mt_r2 | mt_r3 | governor behaviour |
|---|---|---|---|
| B=1 | 0.324 us | **0.325 us** (min; loaded median 0.63) | none (serial 2p, bit-identical, rel_l2 1.308e-16) |
| B=2048 | 32.8–33.1 us/call | **32.4–33.2 us/call = 0.0158–0.0162 us/t** | govm{mt-1f-pfs=0.0225, mt-1f520-pfs=0.0225, mt-1f=0.0232, lock=mt-1f-pfs} — locked the default; 520 ties on wallaby's buffers |
| B=32768 | 2450–2512 us/call | **2471–2497 us/call = 0.0754–0.0762 us/t** | gov{fr0=0, nb=1, w=0.075, h=0.079, lock=mt-1f-nt-pfs, fl=0, fr1=0} |

Wallaby is the null test only: our 32 threads sit on one socket, so fr0=0,
wide wins trivially, and no flip ever fires — it validates the machinery
(dwell → probe → lock → ~7 probe pairs per run, min and output unchanged),
not the hypothesis.  A dwell=4 stress run and the L8R_POOL/GOV knobs all
behaved; the flip branch itself has executed only in code-trace, never in a
run where it should fire — flagged honestly.

Correctness: PASS (tol 1e-12) at B = 1, 8, 33, 64, 100, 2048, 32768;
repeatable (bit-identical across runs) at every size; AVX2-only build
(-mno-avx512f, B=64, 1.910e-16) and portable build (-mno-avx512f -mno-avx2
-mno-fma, B=8, 1.881e-16) PASS and repeatable; warning-free under -Wall
-Wextra at the cascadelake target.  Parallel efficiency unchanged from r2
(kernels untouched): ~84% on 32 threads at B=2048 on wallaby, ~37% at the
B=32768 DRAM wall.

### What was tried and did NOT work / rejected with numbers

* **Chasing the B=2048 kernel gap to fusedaxes (0.0263 vs 0.0279)**:
  deliberately not attempted structurally.  batchsimd's r2 killed the
  handshake theory (tree collect: 56.9, no change); their base "fused" beat
  their own fusedAA in-arena at B=1 (0.556 vs 0.560), so the AA schedule is
  not the edge either; the residual is unexplained same-shape difference
  and the mid governor's on-buffer race is this round's cheap bite at it.
* **Chunk-head prefetch of each thread's first volume**: rejected by
  inspection — a t0 burst issued immediately before its own demand loads
  hides nothing (the r2 spread cadence already covers volumes 2..N).
* **A multi-second commit-to-wide wait**: still rejected (fusedaxes r2's
  arithmetic); the dwell is 0.28 s + an 8% probe duty afterwards, sized so
  a no-migration outcome costs ~8 of 30 samples (median inflates ~15–20%,
  the scored min keeps ~22 clean samples of the r2 baseline).
* One dev slip worth recording: a stale tryout binary (built -mno-avx512f
  for the portable check) produced a "DIFFERS + mt-3p pick" scare in the
  dwell stress test; rebuilt native, everything bit-identical.  Rebuild
  before every A/B.

### Borrowed, plainly

* **L8_fusedaxes mt_r2**: the execute-time governor concept and its
  legality argument, the get_mempolicy fr scan and numa_balancing
  diagnostic, the on-buffer-race-worth-~2% control, and the measured
  0.211–0.217 weighted-T32-on-socket-0-pages number that shaped the
  lock-toward-half hysteresis.
* **mt_r2 VERDICT §5/§6**: the migration mechanism, the L=6 T32/T16
  200-vs-85 GB/s bracket, fftw's bimodality reading, and the literal ask
  ("race wide-team candidates only after a migration-settling warmup") the
  dwell implements.  The paired-probe trick (AutoNUMA's two-fault rule) is
  my own addition.
* **L8_batchsimd mt_r2**: the adopted negative result that the collect is
  not the B=2048 gap.
* From my own lineage: kernels, pool, tuner discipline unchanged.

### Node predictions (stated to be scored)

* **B=1: 0.575–0.585, pick avx512-2p (fixed)** — untouched path.
* **B=2048: 0.027–0.028 us/t, pick mt-1f-pfs with govm{} published.**  If
  the real-buffer race finds the alias lottery (520 or no-pf locking in
  any process), up to ~2% better and the govm string says so; if all three
  processes lock the default, the trio is settled as equivalent on the
  node and the residual gap to fusedaxes is real kernel text.
* **B=32768 — the experiment, three pre-registered branches read off
  gov{fr1,fl,lock}:**
  (a) **migration real**: lock ends mt-1f-nt-pfs (fl ≥ 1 or wide from the
  start), fr1 > 0; min lands 0.08–0.15 us/t and the cell is taken from
  fftw's 0.159.  (b) **no migration**: fr1 = 0, fl = 0, lock mth; min
  0.170–0.176 (r2 preserved; medians ~15% worse from the dwell — accepted,
  min-of-min is the statistic).  Then the migration hypothesis is
  FALSIFIED for a 0.3 s dwell + 8% duty, which is §5's missing measurement
  either way.  (c) **partial**: fr1 small nonzero, possibly a late flip;
  min 0.15–0.174.  I consider (a) and (b) about equally likely; the round
  is worth it because fr1 publishes regardless.

### Next round

1. **Read gov{fr0,fr1,fl,lock} from all three processes first.**  Branch
   (a): tune dwell/duty (longer dwell buys nothing once flipped; consider
   locking wide permanently after any flip).  Branch (b): fftw's 103 GB/s
   on socket-0 pages is then raw T=32 stream efficiency — try full-team NT
   with socket-weighted static cuts biased ~2:1 toward socket 0 (between
   fusedaxes' equal-cut 0.21 and mth's 0.174 there may be a minimum), and
   re-request perf_event_paranoid for the §4.5 alias counter.
2. If any B=2048 process locks 520/no-pf, promote it to the create default
   with the govm numbers as evidence.
3. B=1 stays closed unless a rival's record explains their 0.557 vs my
   0.575 on same-shape kernels.

## Round mt_r4

### Where mt_r3 landed on the node, and what it settled

B=1 0.576 (fusedaxes 0.551, batchsimd 0.559).  B=2048 0.0278 us/t 3/3 (fusedaxes
0.0261) — the mid governor locked the create default in all three processes
(govm{pfs=0.0279, 520=0.0278–0.0282, none=0.0288–0.0298}), so the r3 trio is
settled as equivalent on the node and the gap to fusedaxes is elsewhere.
**B=32768 0.172 in-arena / 0.1737 scored, and the deep experiment came back
branch (b): gov{fr0=0, nb=1, w=0.233, h=0.175, lock=mth, fl=0, fr1=0} in 3/3
processes.**  A 40-call full-team NT dwell with maximal re-touch pressure
migrated NOTHING; my own instrument falsified the migration hypothesis my r3
round was built on.  The VERDICT (s5) confirms it four ways panel-wide, orders
"stop building placement instruments", and names the survivor for L=8
verbatim (s6): the remaining 7% to fftw3_patient is s4.5 — 4K aliasing — with
two sanctioned moves that need no perf counter: ducc0's odd-line scratch
offset, swept; and measuring by construction at several (scr − out) mod 4096
values.  Separately, s4.4/s3.3 build a three-entry case (L17_matrixsimd clk512
2.29 vs 2.89 GHz, L36_pfa's nap, L17_rader's 1.71x from pool→OMP) that live
spin-pool workers poison a streaming process — and my r3 deep plan kept 31
spinners alive all run for the governor's sake.

### What changed (arithmetic untouched; every installable still the 1f bit class, B=1 still 2p bits)

1. **The 4K-alias residue is now a controlled constant.**  My per-thread
   scratch slots were page-aligned with a 12-KiB stride, so
   (scr − out) mod 4096 == 0 deterministically — the relation fusedaxes'
   fusedAA analysis calls an allocation lottery was, in my file, PINNED at a
   worst-tier value in every mode including B=1.  Slots grew to 4 pages
   (16 KiB, stride still ≡ 0 mod 4096 so all threads share one residue; no
   shared pages) with 4 KiB in-slot slack, and fft3d_execute() re-bases the
   slot array against the caller's real out so (scr − out)/64 mod 64 hits a
   target theta: **56 for the 1f family, 58 for 2p**, minima of a line-residue
   model (load blocked iff bits 11:6 match a store from the previous ≤3
   iterations' 16-store windows — the same model fusedaxes brute-forced its
   aa tables from).  Model counts, steady-state batched 1f: 118 blocked
   loads/volume at the mt_r3 layout → 110 at theta=56; B=1 2p: 104 → 90.
   Deterministic per buffer, so repeatable; L8R_SCROFF=<0..63|raw> overrides.
2. **Permuted-iteration twins** (pass iterations are independent, so
   reordering is bit-identical — fusedaxes' aa_perm_tab trick, orders
   re-derived for my layouts by exhaustive search at theta*): **1fp** =
   phase-B k1 order (6,5,7,1,3,2,0,4), model 83 vs 110 at theta=56 (D=3;
   59 vs 83 at D=2); **2pp** = pass orders (0,2,1,7,3,6,5,4)/(3,7,2,0,1,4,5,6),
   model 68 vs 90 at theta=58.  1fp joins the mid AND deep installable pools
   (cached-pfs and NT-pfs twins, full and half team); 2pp is a **B=1 probe
   only** — timed and published so the node prices it, never picked, because
   wallaby reads it +0.6% and B=1 stays closed until a node number says
   otherwise.
3. **The deep dwell/probe governor is deleted** — falsified by its own r3
   numbers above.  Deep installs from the create arena (stable 3/3 in r2 and
   r3) and the post-pick pool shrink now applies to EVERY plan: with the mth
   pick that removes the 15 far-socket spinners r3 kept alive through every
   scored call (the s4.4 clk512 evidence), and medians stop paying the
   40-call wide dwell.
4. **The execute-time governor is rebuilt as a residue race** (now
   team-width-invariant, which is what lets the shrink compose with it):
   cfg0 = the create pick at the RAW (mt_r3) layout — the proven incumbent —
   vs the pick at theta*, plus (mid, cached full-team picks) the 1fp twin at
   theta*.  2 settle + 3 trials per config inside the driver's calibration,
   lock needs >1.5% over raw, published as govm{name@theta=...}.  The create
   arena also times all candidates AT theta* against its surrogate out, so
   the arena's ranking and the driver's run finally see the same alias
   geometry — the fidelity hole the r3 mid governor existed to patch is
   closed at the source instead.

### Operation count

Unchanged per volume: 1248 vector FP (24 × 52-instr codelets, the 56-flop
optimum), 896 shuffles, 256+256 L1 loads/stores (1f).  Permutation and
re-basing add ZERO instructions to any kernel (the orders are compile-time
tables; the re-base is ~2 ns of address arithmetic per execute).

### Measured on wallaby (Gold 6448Y SPR, 32 threads = one socket, shared login
node; same-window interleaved pairs only; SPR is the machine that HIDES 4K
alias and alignment stalls — L45_pfa's r3 lesson — so parity here is the
expected result and the node is where the model is priced)

| cell | mt_r3 (same window) | mt_r4 | note |
|---|---|---|---|
| B=1 | 0.324 us | **0.324–0.325 us** | raw vs theta58: 0.324/0.325 ×3 — SPR-invisible, as expected; rel_l2 1.308e-16 unchanged |
| B=2048 | 32.20/32.63/32.94 us/call | **31.89/32.54/32.77 us/call** (0.0156–0.0160 us/t) | parity, r3-vs-r4 outputs BIT-IDENTICAL; first-run-of-the-day 43.6-median artifact seen again (4th sighting) |
| B=32768 | 2488/2497/2475 us | **2454/2476/2472 us** (0.0749–0.0756 us/t) | r4 ≤ r3 in 3/3 pairs (−0.6%); outputs bit-identical |

Arena and governor strings (wallaby): B=2048 pick mt-1f-pfs (default,pool,
so=56), arena{mt-1f-pfs=0.021 mt-1fp-pfs=0.021 mt-1f520-pfs=0.021 ...},
govm{mt-1f-pfs@raw=0.0220, @56=0.0221, mt-1fp-pfs@56=0.0221, lock=raw} — the
wallaby race is a three-way tie, i.e. the null result SPR should give.
**B=32768 pick mt-1fp-nt-pfs (tuned, so=56): the permuted kernel won the
wallaby arena at BOTH team widths (mt-1fp-nt 0.058 < mt-1f-nt 0.059;
mth-1fp-nt 0.071 < mth-1f-nt 0.073)** — the first machine-measured sign the
order matters; govm{@raw=0.0795, @56=0.0790, lock=raw} (0.6%, under the 1.5%
bar).  Correctness: PASS (tol 1e-12) at B = 1, 8, 33, 64, 100, 2048, 32768;
repeatable at every size; **bit-identity battery: every installable forced
via L8R_FORCE at B=2048 (6 configs) and B=32768 (6 configs) plus
L8R_SCROFF={raw,13} plus L8R_POOL=0 all cmp-identical to the default; 2pp
cmp-identical to 2p at B=1; r4 output cmp-identical to the r3 binary's at
B=2048 and B=32768.**  AVX2-only (-mno-avx512f, B=64, 1.910e-16), portable
(-mno-avx512f -mno-avx2 -mno-fma, B=8, 1.881e-16) and cascadelake-target
builds PASS, warning-free at -Wall -Wextra.

### What was tried / did NOT work, with numbers

* **2pp at B=1 on wallaby: +0.6% (0.326–0.327 vs 0.325, 3/3 pairs).**  Kept
  as a probe, not promoted; the model says CLX should like it, wallaby says
  SPR does not care, the node arena will publish the answer either way.
* **theta at B=1 on wallaby: nothing (0.324 vs 0.325).**  Expected (SPR);
  shipped anyway because the model is strict-improvement, the cost is zero
  instructions, and the mechanism class has two prior NODE wins at exactly
  this cell (batchsimd r9 si520 −1.0% median; the r10 fused pick variance
  matching fusedAA's depth analysis).  Flagged honestly: if the node shows
  theta56/58 ≥ raw everywhere, the model is miscalibrated for CLX and r5
  should revert to raw and keep only 1fp.
* **Deep half-team races on wallaby**: not run — single socket, mth is
  structurally wrong there (r1–r3 lesson, not rediscovered).
* Not attempted, with the killing number from the corpus: any wide-team or
  placement machinery (my own r3 fr1=0/fl=0 3/3 + VERDICT s5's four
  independent refutations); pfw on NT output (fusedaxes r2, 1.6x); prefetch
  at cache-resident sizes (batchsimd, +72%).

### Borrowed, plainly

* **L8_fusedaxes (phase-1 r7/r11 fusedAA/fusedAA2)**: the entire 4K-alias
  line-residue model, the insight that (scr − out) mod 4096 sets the phase-B
  blocked-load count, and the permuted-iteration-order trick (their
  aa_perm_tab); my orders re-derived for my layouts by the same brute force.
* **mt_r3 VERDICT s6**: the ducc0 odd-line-guard directive and
  "measure by construction at several (scr − out) mod 4096 values" — the
  residue race is exactly that, run on the real buffers.
* **mt_r3 VERDICT s4.4 / L36_pfa / L17_rader**: the live-spinner clk512
  evidence behind extending the pool shrink to governed plans.
* **L8_batchsimd r9**: the si520 precedent (already in this file) that this
  stall class is worth real node time at L=8.
* My own r3 falsification (fr1=0, fl=0, 3/3) is what licenses deleting the
  deep governor rather than arguing with it.

### Node predictions (stated to be scored)

* **B=1: 0.570–0.578 if theta58 does nothing, 0.555–0.570 if the model is
  right** (the 2p pass-2 aliases it removes are ~14–16 blocked loads/volume
  of a ~360-cycle gap to the port floor).  Pick string `avx512-2p
  (fixed,so=58)`; the 2pp probe's arena time is the number to read for r5.
* **B=2048: pick mt-1f-pfs or mt-1fp-pfs, 0.0265–0.0279 us/t.**  The govm
  string is the experiment: if @56 or 1fp@56 beats raw by >1.5% on the real
  buffers in any process, s4.5 is finally measured-by-construction at L=8;
  if all three processes lock raw, the alias model does not price CLX at
  this cell and the fusedaxes gap is something else again.
* **B=32768: 0.166–0.174 us/t.**  Decomposition of the hoped-for gain:
  shrunk pool (15 far spinners gone) is the mechanism with independent node
  evidence; theta/1fp is the model bet (wallaby arena already prefers 1fp at
  both widths).  Medians should drop a lot (no 40-call dwell) — on medians
  the panel already beat fftw 1.39x here, so this mostly cleans the record.
  If the min lands ≤ 0.159, s4.5 + the spinner tax WAS the fftw gap; if it
  sits at 0.172–0.174 again, the residue is fftw's plan shape, not our
  allocation, and r5 should look at their two-pass structure instead.

### Next round

1. Read govm{} and the deep arena (1fp vs 1f at mth) from all three
   processes FIRST; promote or revert theta per the pre-registered branches
   above.  If 2pp's B=1 arena time beats 2p's 3/3, promote it (same bits,
   zero risk).
2. If the deep cell is still lost with the spinners gone and the residue
   controlled, the s4.5 hypothesis is exhausted at the allocation level;
   the remaining asks are administrative (perf_event_paranoid for the
   counter) or structural (fftw's shape).  Do not re-open team width.
3. If theta pays at mid but the fusedaxes gap persists, the dispatch is
   already identical (full-team pick = all 31 workers working, same epoch
   protocol) — the next diff target is their kernel text itself against my
   1f port, instruction by instruction, since every layer above it is now
   measured equal.
