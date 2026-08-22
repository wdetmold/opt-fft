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
