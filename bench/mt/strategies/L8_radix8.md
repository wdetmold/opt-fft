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
