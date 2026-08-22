# L13_direct — multicore strategy record

Phase-1 history (how the serial kernel got its shape) lives in
`../../geom/strategies/L13_direct.md`. This file starts at the multicore
phase. Phase-1 baselines used below: node (CLX, panel_r11 scored) B=1
5.739 µs, B=16 5.957 µs/vol, B=512 7.965 µs/vol; wallaby (SPR, pinned,
driver-level) 2.533 / 2.530 / 3.083 µs per volume.

## Round mt_r1

### What changed

The kernel arithmetic is untouched — same conjugate-folded dense 13×13
passes, zsolid-Y + xmm-tail mixed-width shape, same 14.3k vector-op cycles
per volume (per-volume operation count identical to panel_r11; the parallel
version does zero extra arithmetic and the output is bit-identical to the
serial exec at every team size, since every volume's DAG is the same no
matter which thread runs it). What was added is a threading layer:

1. **All exec functions became range execs**: `(plan, in, out, b0, b1, tls)`
   where `tls` is per-thread scratch (`pb`/`t1`/`sb`, one page-aligned
   42 KiB slot per thread). Slots are allocated **and first-touched by
   their owning thread** inside a full-width parallel region in
   `fft3d_create()` — that region also spins up the OpenMP pool, so the
   first timed execute pays no thread creation. Page-sized slots: no false
   sharing between threads' scratch, and each slot is NUMA-local to the
   core that uses it (PANEL_BRIEF "NUMA": we control our scratch's first
   touch; we cannot control the caller's buffers, which the driver
   first-touches single-threaded, i.e. one socket owns in/out).
2. **Batch-parallel decomposition**: team of `min(batch, 32)` threads,
   thread t owns the contiguous volume block [nb·t/T, nb·(t+1)/T). No
   synchronisation inside the region at all — the only barrier is the
   implicit join. Contiguous blocks keep each thread's in/out streams
   hardware-prefetcher-friendly and give the streaming-pf exec the same
   next-volume prefetch pattern as phase 1. Ranges are computed from the
   *actual* team OpenMP delivers, so a squeezed team still computes the
   whole batch.
3. **B=1 ships SERIAL** (`nuse=1`; execute has no parallel region at all,
   so the phase-1 single-core time is preserved exactly). See "did not
   work" for the intra-volume split evidence.
4. **Intra-volume worker built and kept behind `-DL13_B1T=n` / `-DL13_GSM=n`**:
   G threads split one volume — the X pass's 43 chunk units (42 zmm + xmm
   tail, each unit owning disjoint 64-byte tiles of the shared t1, so no
   false sharing by construction), one team barrier, then the 13 kx-planes
   (zsolid-Y into the thread's own pb, Z into out). Bit-identical output.
5. **The phase-1 in-plan discriminator was replaced by a decomposition
   sweep** (still INSTRUMENT ONLY, never changes the pick): it races
   `l13_run` on private buffers across (team, gsplit) candidates and
   appends e.g. `ab[B1]=t1g1:...,t2g2:...` to the description. At B=1 it
   prints the node's own team-size curve t1→t32 on every leaderboard line;
   at mid batches it prices full team vs half team vs per-volume split; at
   B≥32 it prices t32 vs t16 vs t8 vs t4 (that last one is the NUMA
   question: in/out live on one socket, so the far 16 threads pay UPI —
   wallaby can't answer this, its 32-thread close team is one socket).
   Phase-1 caveat stands: read these as kernel-relative, not
   cell-predictive (the r11 record documents an inversion).
6. Dev knobs: `-DL13_TCAP=n` caps the team (node NUMA A/B), `-DL13_B1T=n`
   re-enables the B=1 split, `-DL13_GSM=g` arms per-volume splitting at
   small batches. `L13_FORCE` 0–14 still selects serial kernel shapes (a
   forced exec runs batch-parallel, never intra-split). Determinism: the
   decomposition is a pure function of batch and the harness-fixed thread
   count — no runtime tuner, no pick lottery.

### What was measured on wallaby (Gold 6448Y, 32 threads close/cores, driver-level, unpinned — a shared login node, so mid-team numbers carry scheduling noise; sd quoted)

| case | serial (phase-1, pinned) | mt_r1 | speedup | parallel efficiency |
|---|---|---|---|---|
| B=1 | 2.533 µs | **2.497 µs** (serial by choice, sd 9.9% on min-stat run) | 1.0× | — (deliberate; see below) |
| B=3 | 7.60 µs/call | 5.71 µs/call | 1.3× | 44% on 3 threads |
| B=16 | 40.47 µs/call | **7.81 µs/call** (0.49 µs/vol, sd 5%) | 5.2× | 32% on 16 threads |
| B=33 | — | 13.66 µs/call | — | (1–2 vols/thread boundary case, PASS) |
| B=512 | 1578 µs/call | **84.2 µs/call** (0.164 µs/vol, sd 0.2%) | **18.7×** | **59% on 32 threads** |
| B=2048 (pf path, ws>L3) | ~9.4 ms/call | 526 µs/call (0.257 µs/vol) | ~17.9× | 56% |

Correctness: PASS rel_l2 = 2.8e-16 at B = 1, 3, 16, 33, 512, 2048;
repeatable (bit-identical across runs) everywhere; compiles clean with
AVX-512, AVX2-only, and without OpenMP.

Where the missing cores went, honestly: at B=512 the call does 512
volumes × ~74 KB of in+out traffic ≈ 34 MB in 84 µs ≈ 400 GB/s aggregate —
this is bandwidth-shared territory (the brief's "more cores = more fill
buffers" is exactly what the 18.7× shows), not a sync problem (sd 0.2%,
zero barriers). At B=16 the 5.2× is fork/join-bound: ~2.5 µs of work per
thread against ~4–5 µs of GOMP region overhead on a *shared* login node
(sd 5–9% and a 7.0-vs-7.8 µs min spread across runs say scheduler noise
dominates; the exclusive node should do better).

### What did NOT work, with the numbers that killed it

1. **Splitting one volume across threads (B=1).** In-plan sweep on
   wallaby, min of 7 trials: t1g1 5091 ns → t2g2 8702 (+71%) → t4g4 9905
   (+95%) → t8g8 11816 (+132%) → t16g16 19970 → t32g32 36895 ns/vol.
   Driver-level confirmation: 3.396 µs at g8 vs 2.521 µs serial (+35%).
   Monotone in G: the OpenMP fork + one barrier costs more than the entire
   2.5 µs transform at every team size. B=1 does not parallelise at L=13
   under a GOMP fork-per-call regime — the volume is 34 KiB, L1-resident,
   ~7k cycles of work. It *might* parallelise under a persistent spin-pool
   with sub-µs dispatch (next-round item), but not for free with `omp
   parallel`. The worker is kept compiled (and node-priceable via the
   ab[B1] sweep every leaderboard line) rather than deleted.
2. **Per-volume split at mid batch (G=2 at B=16, 32 threads).** In-plan:
   t32g2 9603 vs t16g1 1212 ns/vol (+7.9×, login-node polluted but the
   direction is unambiguous); the barrier plus the doubled team costs far
   more than halving each thread's 2.5 µs of work. Shipped default is one
   volume per thread.
3. **Capping the team at B=512 (`-DL13_TCAP=16`)**: 108 µs min, sd 17%
   vs 84 µs at t32 — on wallaby's single 32-core socket, 32 threads win
   outright. This does NOT answer the node's two-socket question (there
   t16 = one socket = all-local vs t32 = 16 remote threads over UPI);
   the ab[B64] instrument line answers it there.

### Borrowed

The threading layer itself is new this phase (context.md is empty — no
other mt records existed to borrow from yet). From my own phase-1 lineage:
the per-thread scratch keeps the panel_r8 t1 pad and panel_r11 zsolid
shape verbatim; the instrument-only discipline (never let a timed race
change the pick) is L6_unrolled r9's ab1 pattern carried forward.

### Next round

1. **Persistent spin-barrier pool** (pthreads created in create(), pinned
   to the harness's cores, sense-reversing spin barrier): would cut the
   ~3–5 µs GOMP fork/join to sub-µs, which is the whole gap at B=16
   (0.49 µs/vol vs the 0.16 µs/vol the same kernel does at B=512) and is
   the only thing that could make B=1 splitting viable (2.5 µs / 4 threads
   + ~0.5 µs sync ≈ 1.1 µs — a real target). Do it only after the node's
   ab[B1]/ab[B16] lines confirm the fork cost is real on an idle machine.
2. **NUMA at B=512**: read the node's ab[B64] t32-vs-t16 field. If t16
   wins (UPI-bound), the fix is not fewer threads but making the far
   socket's threads *copy in then compute locally* — their volumes' in
   data staged through their local t1 (one extra 34 KiB read per volume is
   free if it converts remote RFOs to local ones). Only worth building on
   node evidence.
3. If the node's B=512 scales like wallaby (~19×), expect ~0.42 µs/vol
   there; the cell then becomes DRAM-bound and the phase-1 streaming
   tricks (pf schedule, staged Z) deserve a re-A/B under 32-thread
   contention — `-DL13_PW=0`/`-DL13_PFIN=0` still build.

## Round mt_r2

### Where round mt_r1 left this entry

Node (Gold 5218, scored): B=1 **5.670 µs** (won, 1.35× over mkl2026),
B=512 **0.308 µs/vol** (won, 1.17×), B=8192 **1.275 µs/vol — LOST**, 1.31×
behind L13_rader (0.976) and 1.62× behind fftw3_patient (0.787). The
verdict names the cell's disease exactly: 72–95 GB/s achieved where FFTW
gets 89 useful, because every panel L13 store to `out` paid the
write-allocate RFO — and six other entries measured the NT-store inversion
at 32 threads this round (20–30% at every streaming cell). Phase 1's
eleven-round "hide the RFO with prefetchw, never NT" verdict was a
one-core fill-buffer result; it does not survive 32 threads at the DRAM
wall.

### What changed

One structural addition, nothing else touched: a third batch tier. When
the working set exceeds **4× the socket L3** (node: B=8192's 549 MB ≫
88 MB; B=512's 34 MB stays below it), execute switches to
`l13_exec_xsnt_pf_mx` — X-first into t1, per kx-plane zsolid-Y → pb,
zsolid-Z → the hot 2.7 KB `sb` staging plane (both groups pure zmm so the
copy's 64 B reads never straddle narrow stores), then the plane is
appended to `out` with **non-temporal stores**.

The appender is the one part that is mine rather than borrowed: since a
thread's volumes are contiguous in `out` and a volume's planes are
contiguous too, the whole per-thread output range is ONE byte stream, so
the appender carries the partial cache line across plane and volume
boundaries (a 13³ volume is 549.25 lines; plane stride 2704 B ≡ 16 mod
64). Every line of the range is therefore written by exactly one
full-line 64 B NT burst — zero partial-line WC flushes at the ~13 plane
junctions per volume that a per-plane copy would pay, and none at volume
junctions either (L13_rader's per-volume staging pays 2 per volume).
Head/tail of the range use 16 B `_mm_stream_pd` (volume bases are 16 B
aligned). One `sfence` per thread before the join (L8_radix8's rule: a
fence orders only the issuing core's stores). prefetchw is deleted in
this exec — with NT stores nothing RFOs `out` — while the paced
read-prefetch of the next volume's input stays. Intrinsics are confined
to this one `__AVX512F__`-only block; AVX2-only and no-OpenMP builds are
unchanged (verified).

Knobs: `-DL13_NT=0` rolls the tier back to the r1 pf exec; `L13_FORCE=15`
pins the NT exec at any batch. Determinism: the gate is a pure function
of batch and sysconf L3, the copy writes identical bits, so output stays
bit-identical to every other exec at every team size (verified
"repeatable: identical output across runs" at B=1..8192).

### Operation count

Arithmetic unchanged (14.3k vector-op cycles/volume, zsolid + xmm-tail
census from panel_r11). The NT tier adds a per-volume staging pass: 13
planes × 2704 B written to L1-hot sb and re-read for the NT burst
(~4.4 KB of extra L1 traffic per plane, zero extra DRAM traffic), and
deletes the 35 KB/volume out-RFO: compulsory DRAM traffic drops
105 KB → 70 KB per volume.

### Measured on wallaby (Gold 6448Y, 32 threads close/cores, driver-level, shared login node)

| case | mt_r1 | mt_r2 | note |
|---|---|---|---|
| B=1 | 2.497 µs | **2.501 µs** | serial path untouched |
| B=16 | 7.81 µs/call | **7.01 µs/call** | untouched path, login noise |
| B=512 | 84.2 µs/call | **83.2 µs/call** (0.162 µs/vol) | pf exec, untouched |
| B=2048 | 526 µs/call | 540 µs/call | pf path (137 MB < 4×60 MB on wallaby) |
| B=4096 | — | **1206 µs/call** (0.295 µs/vol) | NT tier |
| B=8192 | — (node 1.275 µs/vol) | **2788 µs/call = 0.340 µs/vol**, sd 0.3% | NT tier |

A/Bs at B=8192, same host, same build flags otherwise:
* `-DL13_NT=0` (the r1 exec): 3721 µs → **NT is −25%**, matching the
  panel-wide inversion and L13_rader's −21%.
* `-DL13_PFIN=0` (NT but no input read-prefetch): 3023 µs → the paced
  next-volume input prefetch is worth **−8%** even under NT (consistent
  with L13_rader's node race pricing pf-off at +13%). Keep both.
* Correctness: PASS rel_l2 = 2.83e-16 (B=1) … 2.87e-16 (B=512) …
  2.86e-16 (B=8192), tol 1e-12; bit-identical across runs everywhere.

For reference, L13_rader's mt_r1 wallaby number at B=8192 was 2948 µs
(0.360 µs/vol); this exec is 5% under it on the same host, with the same
70 KB/volume compulsory traffic, so the difference is junction/WC
behaviour and the cheaper kernel. Parallel efficiency at B=8192: phase-1
serial streaming was ~4.6 µs/vol on wallaby (r8 B=2048 reading), so
0.340 µs/vol ≈ 13.5× on 32 threads (42%) — bandwidth-bound, as the
achieved 358 GF/s ≈ 206 GB/s aggregate says directly.

Node prediction, pre-registered: mt_r1's wallaby→node factor for the NT
streaming regime was 2.7× (L13_rader 0.360 → 0.976). That puts this exec
at **~0.85–1.0 µs/vol at B=8192** on the unchanged harness — ahead of
L13_rader, still possibly behind fftw3_patient's 0.787 (which sits at the
node's useful-bandwidth roof). If the monitor adopts the verdict §6
harness fix (parallel first-touch / explicit page policy), the socket-0
residency handicap disappears and this should drop well below 0.787.

### What did NOT work / was deliberately declined, with numbers

1. **Flipping B=1 to the t2g2 intra split.** The node's ab[B1] instrument
   read t2g2:13136 vs t1g1:14222 (−7.6%) — the first time any split beat
   serial anywhere. Declined this round: it is ONE reading from an
   instrument whose absolute scale is 2.5× off the driver (14.2 µs vs
   5.67 µs measured), the r11 record documents an instrument/cell
   inversion, and B=1 is a cell we currently WIN by 8% (5.670 vs
   L13_rader's 6.143). Losing it to chase −7.6% is a bad trade on one
   noisy bit of evidence. The ab[B1] sweep still prints every round: if
   mt_r2's leaderboard shows t2g2 winning again, flip it in mt_r3.
2. **Half team (t16) at the streaming cell.** Not re-measured:
   L13_rader's node race already priced n16 at B=8192 (1086 vs 976
   ns/vol, +11%) — the far socket contributes bandwidth, not just UPI
   (verdict §4: T=32 won every batched cell). The ab[B64] t16 reading
   (1157 vs t32 6842) contradicts this at mid batch, but B=64 is not a
   scored cell and the t32 arena reading there is 5× off the driver's
   B=512 per-volume time, i.e. the arena is mis-pricing something at
   small tb; noted for the monitor, not acted on.
3. **NT at the L3-adjacent batch (B=512).** Not rediscovered:
   L13_rader's node race read nt +33% at B=512. The 4×L3 gate exists to
   keep NT out of exactly that cell; `-DL13_NT=0` exists in case the node
   says the gate is still wrong in the other direction.

### Borrowed, explicitly

* **NT staged stores and the 4×-socket-L3 gate: L13_rader mt_r1** (its
  nts mode, −21% wallaby, node race nt-off +30%), which itself overturned
  my phase-1 prefetchw verdict. Via the mt_r1 VERDICT §4/§6, which
  aggregated the same inversion from L23_matrixsimd, L36_pfa, L17_rader,
  L8_fusedaxes, L45_pfa, L64_*.
* **Per-thread sfence before the join: L8_radix8 mt_r1.**
* **"pw is pointless under NT (no RFO to hide)": L23_matrixsimd mt_r1**,
  confirmed here by construction (pfw deleted from the NT exec).
* The carried-partial-line streaming appender is new here; rivals stage
  per volume and pay 2 partial-line WC flushes per volume, this pays 2
  per thread RANGE (256 volumes at B=8192).

### Next round

1. **Read the node's B=8192 number first.** If it lands ≥1.0 µs/vol while
   wallaby says 0.340, the gap is page placement, not code — push on the
   verdict §6 harness question before touching the kernel.
2. **B=1 t2g2**: flip if the mt_r2 leaderboard's ab[B1] again shows t2g2
   beating t1g1 on the node (two consistent readings = act).
3. If fftw3_patient still leads at B=8192, the remaining levers are (a) a
   per-thread software-pipelined X-pass that overlaps the next volume's
   X with the current volume's Y/Z (deeper overlap of the read stream
   with the NT drain), and (b) trying the NT tier one gate step earlier
   (2×L3) on the node only — wallaby B=2048 pf vs NT is 540 vs untested.
4. The B=16-band fork/join cost (~4–5 µs GOMP region) is still the whole
   gap at small batch; a create()-time spin pool (L17_matrixsimd has one
   to borrow) is the fix if a scored cell ever lands in that band —
   B=512/8192 don't need it (region cost is <2% there).
