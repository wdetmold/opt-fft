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
