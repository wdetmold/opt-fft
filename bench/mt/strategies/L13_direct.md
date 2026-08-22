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

## Round mt_r3

### Where round mt_r2 left this entry

Node (Gold 5218, scored): B=1 **5.734 µs** (won, 1.33× over mkl2026),
B=512 **0.309 µs/vol** (won, 1.17×), B=8192 **0.984 µs/vol — LOST** to
fftw3_patient's 0.603 as scored (1.24× on medians), with `L13_rader` absent
(build failure). My B=8192 landed inside the pre-registered 0.85–1.0 band,
which means the code did what wallaby said it would and the residue is the
machine's placement regime: the VERDICT (§5) measured that the caller's
buffers can sit 100% on socket 0 (`fr=0` at L=8, all three processes) or be
effectively spread (L=6 sustains 200 GB/s at T=32), that the difference is
worth more than every schedule decision in the round combined, and that
**nobody has yet read `fr` under a 32-thread team at a streaming cell**. I
achieve 71 GB/s at that cell; FFTW's 32-thread plan gets 89–103; L=6 gets
200. The node's ab[B1] instrument also read t2g2 < t1g1 again (−11.8% and
−8.2% in two processes, +5.2% in one), the second consecutive round, which
triggers my mt_r2 pre-registration "two consistent readings = act".

### What changed

One structural addition; every kernel, tier gate, and create-time
decomposition default is untouched. **The plan now carries an execute-time
governor, adopted from L13_GOV L8_fusedaxes mt_r2** (the entry whose
`gov{fr,nb}` placement instrument the VERDICT called "the single most
valuable diagnostic produced this round"). The create-time surrogate arena
cannot see the real buffers' page placement — my own record has documented
its 2.5× absolute mis-scale at B=1 and a 5× arena/driver gap at small tb
since mt_r1 — so at the armed cells the plan races a small set of
**bit-identical** configs on the driver's real buffers across the first
execute calls and locks the winner. Probe calls are full correct executes;
the harness statistic (min over processes of min sample) ignores them —
L8_fusedaxes' argument, and fftw3_patient's scored wins at both lost cells
are the same free-early-calls statistic.

* **Streaming cells (ws > socket L3, i.e. node B=512 and B=8192):** configs
  are `tf` (create pick, full team, static contiguous cuts), `th` (half
  team = one socket under close binding), `tfw` (full team with
  **weighted adaptive cuts** — L8_fusedaxes' damped re-cut: per-thread
  chunk timing, weight × clamp(√(v_avg/v_t), 0.75..1.33), weight clamp
  [0.02,4], floor one volume per thread so a starved far socket keeps a
  probe chunk and can recover), and the opposite store discipline (`pf`
  when the NT tier is picked, `nt` when the pf tier is) so the r2 NT gate
  is finally priced on the real buffers rather than a surrogate. Budget 4
  probe calls per config, self-extending +2 while a config improves >2%
  (weighted cuts need calls to converge), cap 8; lock needs to beat the
  create pick by 1.5%; after locking, every 48th call revisits the
  runner-up (≤6 times) and relocks on a >3% win, so late page migration is
  not missed. The **first armed call reads the real buffers' page homes**
  (`get_mempolicy(MPOL_F_NODE|MPOL_F_ADDR)` over ~96 sampled pages of `in`
  and `out` — a pure read; the mt_r1 ruling allows reads and bans
  move_pages) plus `/proc/sys/kernel/numa_balancing`, and publishes
  `gov{fr=<in>/<out>,frl=<latest>,nb=..,tf:..,th:..,tfw:..,pf:..,lock=..}`
  ns/vol in the description — the §5 experiment the VERDICT asked for, now
  answered at L=13 in every process.
* **B=1:** configs are `t1g1` (serial, the incumbent) and `t2g2` (the
  intra-volume split). 64 probe calls each, alternated call-by-call so the
  powersave frequency ramp hits both equally; the challenger must win by
  3%. This **honours the pre-registered "act" with a better instrument
  than a blind flip**: the flip the ab[B1] sweep priced at −8..−12% is now
  decided in the driver's own hot-loop regime (the thing the 2.5×-off
  surrogate could never see), per process, and if t2g2's win is an
  instrument artifact the lock stays serial at zero scored cost. I
  deliberately did NOT hard-flip the default: B=1 is a cell we win by
  1.33×, the instrument's absolute scale is documented-wrong, and risking
  a won cell on 3-of-4 noisy readings when the governor can take the win
  iff it is real is a strictly better trade.
* Determinism note, updated honestly in the header: the **output** remains
  bit-identical at every batch (each volume's DAG is the same under every
  raced config — the r1 property, so no lock flip can ever appear in
  `cmp`/correctness), but the **decomposition** at armed cells is now
  measurement-locked per process. That is the exact adaptation the VERDICT
  endorsed (§4.6: "racing a surrogate is not searching"; §5's two-regime
  finding makes a create-time constant wrong in at least one regime).
  `-DL13_GOV=0` removes the governor entirely; any `-DL13_FORCE` also
  disables it. Non-AVX512 and non-OpenMP builds never compile it.

### Operation count

Kernel arithmetic unchanged (14.3k vector-op cycles/volume, zsolid +
xmm-tail census from panel_r11; NT tier traffic 70 KB/volume compulsory).
Governor overhead: probing ≤ 32 calls (streaming) / 128 calls (B=1), all
min-immune; after lock, 4 pointer/int stores per call, plus for `tfw` only
2 `clock_gettime` per thread per call (~0.05 µs against a ≥160 µs call)
and one 32-entry re-cut on the main thread; the fr scan is ~192 syscalls
(~0.2 ms) on probe/revisit calls only. Zero extra arithmetic, zero extra
DRAM traffic.

### Measured on wallaby (Gold 6448Y, 32 threads close/cores, driver-level, shared login node — this round it was intermittently loaded; minima quoted, same-session A/Bs)

| case | mt_r2 | mt_r3 | note |
|---|---|---|---|
| B=1 | 2.501 µs | **2.494 µs** | gov locks t1g1 (probe window: t1g1 6975 vs t2g2 8387 ns — t2g2 +20% on SPR, as its 2-FMA units predict); serial path preserved exactly |
| B=16 | 7.01 µs/call | **5.86 µs/call** | untouched path (gov off below L3), login noise |
| B=512 | 83.2 µs/call | **83.9 µs/call** (0.164 µs/vol) | gov off on wallaby (34 MB < 60 MB L3); node will arm it |
| B=2048 | 540 µs/call | **533 µs/call** | pf tier, gov armed, locks tf |
| B=4096 | 1206 µs/call | **1132 µs/call** (0.276 µs/vol) | NT tier; gov: `fr=0/0,nb=1,tf:356,th:504,tfw:486,pf:404,lock=tf` |
| B=8192 | 2788 µs/call | **2803 µs/call** (0.342 µs/vol) | parity — wallaby is the null test (one socket, fr=0), where the governor must and does change nothing |

The B=4096 gov line is the wallaby null test passing end-to-end: the fr
scan correctly reads 0% remote for 32 close threads on one socket, the
half team prices +42% (one socket's fill buffers halved), NT-off (`pf`)
prices +13% worse — same direction as mt_r2's −25% NT A/B — and the lock
stays with the create pick through the 1.5% hysteresis.

Correctness: PASS rel_l2 = 2.83e-16 (B=1) … 2.88e-16 across B = 1, 16, 33,
512, 2048, 4096, 8192, tol 1e-12; **bit-identical output across runs at
every batch** (the governor cannot break this by construction — verified by
tryout's cross-run cmp everywhere). Builds and passes: AVX2-only
(`-mno-avx512f`), no-OpenMP (`-fno-openmp`), `-DL13_GOV=0`,
`-DL13_FORCE=15`. No new warnings at `-Wall -Wextra`.

### What did NOT work / was deliberately declined, with numbers

1. **A long-setup "arm AutoNUMA so the timed loop migrates" scheme** — I
   had talked myself into this from the setup-time correlation at the
   B=8192 table (every ≥1 s-setup backend is fast there: fftw3_patient
   1.348 s → 0.603; every ≤0.1 s-setup backend sits at ~55–71 GB/s), but
   **L8_fusedaxes mt_r2 already rejected it by arithmetic** (their "did
   not work" item 1: the driver's whole execute lifetime is ~0.5–1 s while
   AutoNUMA's scan delay alone is 1 s and 512 MB takes several scan
   periods; L6_pfa's stable 200 GB/s is therefore allocation-time
   spillover, not migration). Adopted their rejection instead of spending
   the round rediscovering it; the governor's 48-call runner-up revisit is
   the cheap hook into slow drift, and the fr/frl pair will now measure
   directly whether ANY migration happens inside a scored L=13 process.
2. **Hard-flipping B=1 to t2g2** (my own mt_r2 pre-registration): declined
   in favour of the on-buffer race, reasons above. The wallaby gov reading
   (t2g2 +20%) confirms a blind flip would have been wrong on at least one
   host class; the node decides for itself per process.
3. **`tfw` on wallaby prices +36% over `tf` in its 8-call probe window**
   (486 vs 356 ns/vol) where L8's weighted variant won its null test by
   ~2%. Two differences: my weights re-cut from only 4–8 calls at 128-vol
   chunks on a loaded login node, and my floor is 1 volume where theirs is
   ~0.1% of B in probe chunks. Not tuned further this round — the
   hysteresis keeps it from ever locking unless it genuinely wins by 1.5%,
   and its interesting regime (fr=0, far socket UPI-throttled) does not
   exist on wallaby to tune against.

### Borrowed, explicitly

* **The whole governor concept, its min-of-min-immunity argument, the
  get_mempolicy/numa_balancing diagnostics, the weighted adaptive re-cut
  recipe (damped √, clamps), the self-extending probe budgets, and the
  48-call runner-up revisit: L8_fusedaxes mt_r2.** Also its rejection of
  the AutoNUMA-arming scheme (their §"what did not work" item 1).
* **"Init race buffers serially to reproduce the driver's first touch" as
  the reason surrogates mis-lead, and the aggregate-cache arena lesson:
  L6_pfa mt_r1 via L6_unrolled mt_r2** — used here as the justification
  for racing on the real buffers instead of enlarging my surrogate.
* The bit-identical-configs discipline that makes governor flips invisible
  to correctness is my own mt_r1 property, kept deliberately.

### Node predictions, pre-registered

* **B=1**: if the node's t2g2 advantage is real in the driver regime, the
  gov locks t2g2 in ≥2 of 3 processes and B=1 lands **5.1–5.5 µs**; if it
  was an instrument artifact, locks stay t1g1 and B=1 stays **5.70–5.78**.
  Either way the cell should stay won (mkl2026 7.64).
* **B=512**: gov armed (34 MB > 22 MB). Expect `fr=0/0`, lock tf in most
  processes, **0.30–0.31 µs/vol** (unchanged); if any process reads fr>0,
  tfw/th may lock and it could dip below 0.30.
* **B=8192, the target.** Scenarios the published gov{} string will
  distinguish per process: (a) `fr≈0` everywhere: lock tf or tfw,
  **0.90–1.00 µs/vol** (tfw rebalancing the UPI-throttled far threads is
  worth 0–5%); the cell stays lost to fftw3_patient's minority-mode min
  and the panel finally has the fr=0 measurement at a 32-thread streaming
  cell. (b) any process has `fr` well above 0 (the L6 regime): tf/tfw
  exploit both sockets and that process lands **0.45–0.75 µs/vol**, which
  under min-of-min scoring takes the cell back from 0.603. (c) `frl` >>
  `fr0` (migration inside the run): the revisit relocks late — worth
  publishing even if the number barely moves.

### Next round

1. **Read the three gov{} strings first.** They are the round's product as
   much as the times: fr at a 32-thread streaming cell settles VERDICT §5
   at L=13 whichever way it reads.
2. If fr=0 held and B=8192 stayed ~0.95: the honest residue is one-socket
   NT-drain efficiency (71 GB/s vs FFTW's ~103 with regular stores).
   Candidates, in order: race `th`+NT vs `th`+pf (16 local threads may
   prefer regular stores + prefetchw, the phase-1 per-core verdict, since
   one socket alone is concurrency- not bandwidth-bound); and a
   socket-split hybrid (near-socket threads NT, far threads regular) which
   is only worth building on a measured fr=0 with th ≈ tf.
3. If t2g2 locked at B=1 and won: fold the same on-buffer race idea down
   to a t4g4 config (the node ab curve says t4g4 is the next candidate,
   26887 vs t2g2's 13903 — unlikely, but the race prices it for free).
4. If `L13_rader` rebuilds, its 0.976 at B=8192 plus my governor are
   complementary — whoever is behind should adopt the other's exec.

## Round mt_r4

### Where round mt_r3 left this entry

Node (Gold 5218, scored): B=1 **5.868 µs** (won, 1.30× over mkl2026 — but a
2.3% REGRESSION from 5.734, and the VERDICT §3.5 names the cause: my B=1
governor read t2g2 10% faster than serial in its 64-call probe window and
locked it in all three processes, and the cell got slower). B=512
**0.305 µs/vol** (the honest winner — L13_rader's listed 0.302 is a 1-in-3
pick lottery the verdict says not to quote). B=8192 **0.980 µs/vol — LOST**
to fftw3_patient's 0.603 (itself a 1-in-3 fast mode; on medians the panel
wins 1.19×). The round's big findings that bear on me: (a) VERDICT §5 killed
the page-placement mechanism outright — fr=0 everywhere including my own
`gov{fr=0/0,frl=0}` at B=8192, so tfw's target regime does not exist;
(b) §4.3/§6: the per-thread-L2-tile construction ("in read once, out written
once, all three axes inside the tile") measured 137–151 GB/s at L=36 and
129 at L=17 where both L=13 entries sit at 69–72 GB/s; the verdict's L=13
order is to build it; (c) four independent on-buffer races (L8×3, L36_pfa)
priced the half team 19–35% FASTER at their deep-streaming cells.

### What changed

1. **The B=1 governor race is DELETED; serial is unconditional again.**
   The mechanism of the r3 regression, now understood: at B=1 the race's 64
   alternated probe calls sit in the driver's frequency-ramp/warm-up window,
   while the incumbent it displaces would have sampled the min over
   thousands of post-lock calls — the probe statistic is structurally biased
   toward *whichever config is measured only early*, and serial's true min
   (5.734) was better than t2g2's whole-run min. Probe-window minima at B=1
   are not comparable to lock-window minima; I will not race that cell
   again. (The ab[B1] create-time instrument still prints the t-curve.)

2. **Staged-input L2-tile execs** (`l13_exec_xcnt_mx` NT tier, FORCE=16;
   `l13_exec_xcpf_mx` pf tier, FORCE=17) — the VERDICT §6 construction
   ported from **L36_pencilfused / L36_mixedradix mt_r3** (their paced read
   cursor / `vol32-sntp` shape). Per-thread scratch gains a `tin` volume
   tile (13 x-rows repadded 338→344 doubles; slot now 77.6 KiB, still inside
   one node core's 1 MiB L2). The exec pipelines on one buffer: prologue
   copies volume b0's 13 rows; the X pass runs from tin; between the Y and Z
   groups of plane x it copies x-row x of the NEXT volume (inline vector
   copy — a memcpy call would spill the 12 pinned zmm constants). Two
   mechanisms in one change: the volume's DRAM read becomes ONE linear
   demand stream at full MLP (replacing 13 concurrent 2.7 KB-strided kernel
   streams plus droppable prefetcht1 hints — pfr43 is gone from the staged
   exec, the copy IS the read), and every X-pass load becomes 64 B-aligned
   (at the caller's 338-double row stride, 3/4 of the X pass's zmm loads
   split a cache line; tin's 344 stride and 64 B base make them all whole-
   line). Output bit-identical by construction (the copy preserves values).

3. **Governor slots reworked** (still min-of-min-immune, all configs
   bit-identical): `tfw` and the weighted re-cut race are DELETED (VERDICT
   §5: fr=0 everywhere; my own r3 wallaby reading had it +36% anyway). NT
   tier now races `tf` (unstaged full team, the create pick), `th`
   (unstaged half team), `st` (staged full), `sh` (staged half) — the full
   2×2 the r3 races never covered, since the L8-family's T16 wins and my
   MLP theory could interact. pf tier (node B=512) races `tf`, `th`,
   `st`=xcpf, `nt`=xsnt. The one-time fr read still rides the desc string
   (zero new machinery, keeps the record comparable).

4. **Default stays UNSTAGED** at both streaming tiers (`-DL13_STG=1` flips
   it): wallaby priced the staged NT exec +15% (below), and under min-of-min
   scoring a wrong default costs only the probe calls while a right
   challenger still takes the cell — the asymmetric-risk choice.

### Operation count

Kernel arithmetic unchanged (14.3k vector-op cycles/volume, zsolid +
xmm-tail census from panel_r11). The staged execs add one 35 KB volume copy
per volume: +35 KB of L1/L2 store traffic and +~4.4k copy instructions;
compulsory DRAM traffic UNCHANGED at 70 KB/volume (the copy's reads replace
the kernel's reads of `in` one-for-one), and ~40 line-splitting zmm loads
per volume are deleted from the X pass. Governor: 4 configs × 4–8 probes of
~53 calls at B=8192, then locked; ≤2 pointer stores per call after lock.

### Measured on wallaby (Gold 6448Y, 32 threads close/cores, driver-level, shared login node — session had visible load spikes; clean-window minima quoted, sd shown where it matters)

| case | mt_r3 | mt_r4 | note |
|---|---|---|---|
| B=1 | 2.494 µs | **2.499 µs** (sd 0.15%) | serial restored; one loaded-window reading of 4.89 µs discarded (repeated at 2.50) |
| B=16 | 5.86 µs/call | **7.42 µs/call** | untouched path, pure login noise (session drift below) |
| B=512 | 83.9 µs/call | **83.7 µs/call** (0.164 µs/vol) | untouched tier on wallaby (34 MB < 60 MB L3) |
| B=2048 | 533 µs/call | **529 µs/call** | pf tier, gov armed |
| B=4096 | 1132 µs/call | **1159–1505 µs** | NT tier, 4-config gov; loaded windows, unreadable beyond parity |
| B=8192 | 2803 µs/call | **2781–2852 µs = 0.339 µs/vol** | gov locks tf (unstaged), as wallaby should |

**The round's controlled A/B** (forced execs, same build flags, interleaved
runs): B=8192 unstaged `FORCE=15` **2786 / 2811 µs (sd 0.21/0.23%)** vs
staged `FORCE=16` **3201 / 3240 µs (sd 11.9/0.10%)** — **staged is +15% on
wallaby**. Read honestly: wallaby's 32 close threads are one DDR5 socket
already sustaining 206 GB/s on the unstaged exec, so there is no MLP famine
to fix and the tile's extra L2 round trip is pure cost. The staged form's
target regime — the node's 71 GB/s of a socket's ~128, with reads crossing
UPI for 16 threads — does not exist on wallaby. That is exactly why it
ships as a governor challenger, not the default.

A pf-tier A/B at B=2048 was attempted and is UNREADABLE: identical unstaged
builds minutes apart read 535 and 686 µs (28% session drift); the staged
pf read 644–655 between them. The node governor prices it instead.

Correctness: PASS rel_l2 = 2.83e-16 (B=1) … 2.88e-16 across B = 1, 16, 33,
512, 2048, 4096, 8192, tol 1e-12; bit-identical output across runs at every
batch (tryout cmp). Builds and passes: AVX2-only (`-mno-avx512f`),
no-OpenMP (`-fno-openmp`), `-DL13_GOV=0`, `-DL13_STG=1`, `-DL13_FORCE=16/17`.
No new warning *classes* (+24 instances of the pre-existing dead-constant-
load pattern from the two new execs' shared decl macro; gcc deletes them).

### What did NOT work / was deliberately declined, with numbers

1. **Staged-input NT as the create default** — killed by the wallaby A/B
   above (+15%, sd ≤0.23% on three of four runs). Kept as gov challenger.
2. **Re-racing B=1 in any form** (including a longer probe window): the r3
   node data is three processes of the same 2.3% loss, and the bias
   mechanism (probe-window vs lock-window minima) is structural, not a
   budget problem. Serial, unconditionally.
3. **tfw / weighted cuts**: deleted without re-measurement — VERDICT §5's
   seven-entry fr table killed the regime it was built for (fr=0 at every
   L=13-relevant cell, migration changes nothing where it happens); my own
   r3 wallaby probe had it +36% over tf.

### Borrowed, explicitly

* **The staged-input L2-tile construction and the paced read cursor:
  L36_pencilfused mt_r3 (mode-2/istream) and L36_mixedradix mt_r3
  (`vol32-sntp`), via the VERDICT §4.3/§6 port order** — re-derived at L=13
  scale (the volume is 34 KiB, so the tile holds one volume and the cursor
  paces at x-row granularity between kernel groups).
* **The staged×half-team `sh` config exists because of the L8-family's
  three independent T16-wins-by-19-35% races (L8_fusedaxes, L8_batchsimd,
  L8_radix8 mt_r3) plus L36_pfa's s16** — none of which raced narrow WITH a
  changed read discipline.
* The governor frame remains L8_fusedaxes mt_r2's (r3 borrow, carried).
* "Publish both configs' real-buffer costs instead of hiding a lock"
  (L8_fusedaxes's no-lock reporting, praised in VERDICT §6) — the gov desc
  string keeps printing every config's best ns/vol after locking.

### Node predictions, pre-registered

* **B=1**: serial restored ⇒ **5.70–5.78 µs**, cell stays won (mkl2026
  7.64). Anything at 5.87+ means the r3 regression was not the lock.
* **B=512**: gov armed, 4 configs. Expect lock=tf and **0.30–0.31 µs/vol**;
  if the X pass's split loads matter at the L3-adjacent tier, `st` locks
  and it dips a few %. `nt` should price +25–35% as it always has there.
* **B=8192, the target.** The gov string is the experiment: if the node's
  71 GB/s is MLP/latency-starved strided reading (my theory), `st` or `sh`
  locks with a double-digit win and the cell lands **0.65–0.85 µs/vol**;
  if the node behaves like wallaby (staging is pure overhead), lock stays
  tf at **0.95–1.00** and the honest conclusion is that the read pattern
  was never the binding constraint — in which case the next lever is the
  NT drain itself (th/sh price that too). Either outcome closes the
  question the verdict posed for L=13.

### Next round

1. Read the B=8192 gov string first: which of {tf,th,st,sh} locked, and the
   four best-ns/vol numbers. They decide between "read-side MLP" and
   "NT-drain/other" for the remaining 71→129 GB/s gap.
2. If st/sh won: fold staging into the default (working-set gate, no race —
   the verdict's "install from the working set" rule) and try the same
   cursor at the pf tier default for B=512.
3. If tf held at ~0.98: the read side is exonerated; the remaining suspects
   are the NT drain's fine-grained interleave (2.7 KB bursts between kernel
   groups vs L36's whole-volume drains — try draining sb per VOLUME from a
   13-plane staging buffer, 35 KB, still L2-resident) and the per-volume
   junction rate (34 KiB volumes = 4.5× more stream restarts than L=17).
4. B=1 has ~0.4 µs of standing headroom vs the phase-1 pinned floor
   (5.73 node vs 2.53 wallaby ≈ 2.26×, the largest B=1 wallaby→node factor
   among my cells); if L13_rader's pool lands a stable ~5.0, adopt their
   two-phase split ONLY under a persistent pool with sub-µs dispatch —
   the GOMP fork evidence from mt_r1 still stands.
