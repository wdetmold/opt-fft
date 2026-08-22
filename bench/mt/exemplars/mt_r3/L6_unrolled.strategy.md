# L6_unrolled — multicore strategy record

Phase-1 history (how the serial kernel got its shape over panel_r1–r11) lives
in `../../geom/strategies/L6_unrolled.md`. This file starts at the multicore
phase. Phase-1 baselines used below — node (CLX, panel_r11 scored):
B=1 0.209–0.221 µs, B=4096 0.394 µs/vol, B=32768 0.563 µs/vol; wallaby (SPR,
driver-level, session-band): B=1 0.129 µs, B=4096 0.193 µs/vol, B=32768
0.254–0.261 µs/vol.

## Round mt_r1

### What changed

The per-volume kernels are untouched — same PFA 2×3 arithmetic, same
operation count (972 vector arithmetic uops + 108 in-codelet shuffles per
volume, 5184 real flops), and the parallel output is **bit-identical to the
serial execute at every team size** (each volume's DAG is unchanged; only
which thread runs it changes). What was added:

1. **Batch-parallel decomposition.** Thread t owns the contiguous volume
   block [B·t/T, B·(t+1)/T) and runs the chosen kernel over it with its own
   t1/t2 (per-thread arena with the same 4 KiB placement slack as the serial
   one, allocated and **first-touched by the owning thread** inside a
   full-width parallel region in `fft3d_create()`, which is also what spins
   up the pool — `fft3d_execute()` never creates a thread). No
   synchronisation inside the region; the only barrier is the implicit
   join, plus a per-thread `sfence` when the picked kernel streams. Volumes
   are 3456 B = 54 whole cache lines, so chunk boundaries can never
   false-share the caller's buffers. Ranges are computed from the actual
   team OpenMP delivers, so a squeezed team still computes the whole batch.
2. **B=1 ships SERIAL** (`nthr=1`; execute has no parallel region at all, so
   the phase-1 path and its node-tuned pick machinery — race, ab1, kclk,
   chosen-kernel dwell — are preserved exactly; wallaby confirms 0.129 µs,
   identical to panel_r11). I did not rebuild an intra-volume splitter:
   L13_direct's mt_r1 record measured a GOMP fork + one barrier at +71%
   over serial on a 2.5 µs volume; mine is 0.21 µs — 12× less work against
   the same overhead. Adopted their conclusion instead of rediscovering it.
   `L6_FORCE_T=2` at B=1 remains as the node instrument if anyone wants the
   fork-cost number measured there.
3. **The plan-time kernel tournament runs THREADED when batch > 1**, through
   the exact `l6_mt_call` path execute() uses (fork/join included), at full
   width. This matters because the 32-thread winner is not the 1-thread
   winner — see the NT result below. The race buffers are initialised
   serially on the main thread, reproducing the driver's `fread` first-touch
   (one socket owns in/out), so the race sees the driver's NUMA layout.
   batch=4096 races at its real size (27 MiB — on the node's two sockets the
   combined 44 MiB L3 can hold it; a genuinely different regime from
   B=65536), larger batches race at the 16384-volume / 113 MiB cap.
4. **Team-size race** with the chosen kernel over T ∈ {1,2,4,8,16,24,32},
   round-robin, per-T minimum, smallest T within 2% wins. The full curve is
   published on every leaderboard line as `tm=1:…,…,32:…ns` — on the node,
   T=16 under `OMP_PROC_BIND=close` is one socket (every thread local to the
   driver-touched buffers) vs T=32 adding the far socket's fill buffers but
   paying UPI. That is the round's NUMA question and the node now answers it
   in every cell. `L6_FORCE_T=<n>` forces the team (trailing `!` on nthr).
5. **Four NT-store fused kernels** join the race as trailing 2.5%-margin
   challengers: `fused_nt`, `fused_nt_pf`, `fused_zp_nt_pf`,
   `fused_nt_pfnta` (NTA input prefetch: read-once input, don't thrash the
   shared L3). Phase 1 rejected NT 0-for-4 rounds **single-core** (one core
   is concurrency-bound, not bandwidth-bound); 32 threads at B=65536 are
   DRAM-bound where the write-allocate RFO is a third of all traffic. The
   threaded race decides per cell instead of me guessing.
6. **Pruned** (questions answered on the node in phase 1, and dead code is
   not free in this file — r5: +3.5% B=1 from unused zmm bodies): the
   panel_r11 abL DRAM codelet A/B, its probe-only VD63/fused3_pfw kernel
   (abL=f524.0,f3529.6 closed it: VD6 wins the DRAM regime too), and ~250 ms
   + 113 MiB of per-plan setup with it. ab1/kclk/chosen-dwell now run only
   on the batch==1 path where their question lives.

### Operation count

Unchanged from panel_r11: 48 real flops / 36 arithmetic instructions per
6-point line, 108 line-DFTs per volume, 972 vector uops + 108 shuffles per
volume in ymm. The threading layer adds zero arithmetic; per call it adds
one GOMP fork/join and T placement lookups on the first execute only.

### What was measured on wallaby (Gold 6448Y, 32 threads close = ONE socket, shared login node — session bands quoted where seen)

| case | phase-1 serial (wallaby) | mt_r1 | speedup | parallel efficiency |
|---|---|---|---|---|
| B=1 | 0.129 µs | **0.129 µs** (sd 0.02%; one displaced session read 0.253 — known wallaby session hazard, reproduced 0.129 twice after) | 1.0× | — (serial by choice) |
| B=4096 | 0.193 µs/vol | **9.4–12.3 ns/vol** (38.5–50.5 µs/call, sd 0.2–0.3%) | 15.7–20.6× | 49–64% on 32 |
| B=65536 | ~0.26 µs/vol (from B=32768) | **43.2 ns/vol** (2828 µs/call, sd 0.5%; one session 3539) | ~6.0× | ~19% on 32 |

Correctness: PASS rel_l2 = 2.34e-16 / 2.43e-16 / 2.42e-16 at B=1/4096/65536,
also B=2, 7, 33 (odd chunking); repeatable (bit-identical) everywhere; builds
and passes without OpenMP and without AVX2.

Picks and curves (wallaby): B=4096 → `fused_zp nthr=32`,
tm=1:274,2:134,4:87,8:44,16:20,24:12,32:9ns — near-linear, the 27 MiB
working set is L3/DDR5-friendly on one SPR socket. B=65536 →
`fused_zp_pf nthr=32`, tm=1:613,2:334,4:168,8:92,16:54,24:38,32:30ns —
sublinear from T=16 on: bandwidth. Where the missing cores went at B=65536,
honestly: 43.2 ns/vol × 65536 vols moving 10.4 KB/vol (in + out + RFO) is
~240 GB/s aggregate — this is the socket's memory system, not sync (sd
0.5%, zero barriers, 2048 volumes per thread). The brief's "more cores =
more fill buffers" is exactly the 6× (one core sustains ~40 GB/s here).

### What did NOT work, with the numbers that killed it

1. **NT stores on wallaby, even DRAM-bound.** Threaded race at nt=16384
   (113 MiB > 60 MiB L3): fused_pf 0.0189 µs/vol vs fused_nt_pf 0.0314
   (+66%), fused_nt 0.0338, fused_zp_nt_pf 0.0325, 3pass_nt_pf 0.0308. The
   RFO-traffic argument says NT should win by up to 1.5×; on one SPR socket
   with DDR5 it loses by 1.6× instead — 32 threads × 32-byte ymm streaming
   stores apparently fight over WC buffers harder than the RFOs cost. NOT
   extrapolated to the node: CLX has less bandwidth per socket and the far
   socket's NT writes skip the remote-RFO round trip entirely; the kernels
   stay in the race and the node decides. Watch the mt_r1 pick strings.
2. **NTA input prefetch: worst of all 14 candidates** (0.0445 µs/vol, +135%
   over fused_pf, same race). SPR appears to punish `prefetchnta` here;
   kept raced (never pickable unless it wins by 2.5%) for the node reading.
3. **One displaced wallaby session** read B=1 at 0.253 µs and B=65536 at
   3539 µs with tight per-sample sd — whole-session displacement on a
   shared login node, not a code property (0.129 reproduced twice
   immediately after). Same hazard my phase-1 record flags since r4;
   scored numbers come from the exclusive node.

### Borrowed

- **L13_direct (mt_r1)**: the threading layer's shape — per-thread scratch
  allocated and first-touched by its owner inside create()'s full-width
  pool-warming region; contiguous volume blocks; ranges from the actual
  team; B=1 serial with no region; a forced-team dev knob (their
  `L13_TCAP`); and their measured fork/join numbers, which I used to skip
  rebuilding an intra-volume splitter at 12× less work per volume.
- **From my own phase-1 lineage**: the round-robin licence-fair race, the
  4K-aliasing placement (now per thread), the safest-first margins, and the
  instrument-through-description discipline (the tm= curve is L36_pfa's r8
  in-plan-probe pattern applied to team size).

### Next round

1. **Read the node's tm= curves.** If t16 ≈ t32 at B=65536, the far socket
   is pure UPI waste; the fix worth trying is not fewer threads but staging:
   far-socket threads copy their input volume through their NUMA-local t1
   (one extra 3.4 KB local read converts every remote access to local) —
   L13_direct's record proposes the same and neither of us has node evidence
   yet. If t32 clearly wins, leave it alone.
2. **Read the node's pick strings for NT.** If a fused_nt* shape wins the
   B=65536 cell there, consider an NT variant of the zp x-order and
   dropping the wallaby-motivated pf variants; if NT loses on the node too,
   delete all four next round (my own r10 prune lesson).
3. If the node's B=4096 cell shows the 44 MiB combined-L3 effect (per-vol
   time well under the B=65536 cell's), consider racing an interleaved
   (round-robin) chunk assignment at that size only — it spreads each
   thread's footprint across both sockets' L3 slices. Not worth it at
   B=65536 where DRAM streams want contiguity.
4. B=1 intra-volume splitting stays dead unless someone demonstrates a
   sub-µs persistent spin-barrier pool; even then the budget is ~0.2 µs of
   work total.

## Round mt_r2

### Where I stood

Second in all three node cells behind L6_pfa: B=1 0.221 vs 0.210 µs, B=4096
11.6 vs 9.4 ns/vol, B=65536 72.1 vs 39.5 ns/vol (1.83×). Their mt_r1 record
explains all of it, so this round is deliberate adoption, not invention.

### What changed (all three adopted from L6_pfa's mt_r1, attributed)

1. **The multicore race-arena cap rose 16384 → 65536 volumes (452 MiB).**
   Their round's central lesson: once threaded, the *aggregate* cache (node
   ~76 MiB = 2×22 L3 + 32×1 L2; wallaby ~124 MiB), not the largest single
   L3, sets the race-arena floor. My 113 MiB arena raced a
   near-cache-resident problem for a DRAM-bound batch: it rejected the NT
   kernels AND flattened the team curve (T=16 75 ns ≈ T=32 78 ns raced), so
   the smallest-T rule shipped T=16 = one socket's ~96 GB/s = the whole
   1.83× B=65536 gap. The node's mt_r1 evidence that this was the bug:
   L6_pfa's winner `fused_pf_nt_xa_d2` T=32 is *structurally my
   fused_nt_pf* (ascending x + T0 pf + the VD6 graph they adopted from me
   in r9 + NT fused stores) — same kernel class, right regime, 39.5 ns.
   The serial arm keeps 16384.
2. **Spin-wait pthread dispatch pool**, raced sequenced against OMP
   dispatch on the chosen (kernel, T): epoch broadcast + padded per-worker
   done flags, workers pinned to the affinity masks the OMP threads report
   in the pool-warm region (same core, same socket as their first-touched
   scratch). OMP is timed first with no pool alive; the pool then warms
   ~3 ms (GOMP spinners futex-sleep) before its timed trials; kept only if
   >2% faster, else torn down before create() returns. My chunk split
   (b0 = nvol·t/T) is used by both dispatches, so the output is
   bit-identical either way. Raced result rides the description as
   od=<omp>,<pool>ns and the shipped choice as disp=omp|pool.
3. **sp2 software-pipelined fused y/z stage** (their panel_r5 shape; won
   their node B=4096 cell in mt_r1): plane registers double-buffered P/Q,
   next plane's 18 loads + 3 y-codelets interleaved by thirds into the
   current plane's z-chunks, plane-pair loop kept rolled for DSB residency.
   Raced as fused_sp2 / fused_sp2_pf (full 2.5% new-shape margin), with
   their caveat carried: their sp2 won with the radix-3-first codelet and
   its d2 twin (= my VD6) "composed badly with the interleave" (−3.3%
   wallaby), so mine may lose — the race decides.
4. **Threaded end-to-end gate** (their point 7): the final (kernel, T,
   dispatch) config must reproduce the scalar reference on an odd
   61-volume batch (uneven chunks, idle threads, the pool handoff protocol
   on real data) at rel L2 ≤ 1e-13 or the plan falls back to serial.

### Operation count

Unchanged: 48 real flops / 36 arithmetic instructions per 6-point line, 972
ymm arithmetic uops + 108 shuffles per volume. sp2 reorders the fused
stage's instructions, it adds none. The pool adds zero arithmetic; per
execute it replaces one GOMP fork/join (node fork= probes: 13.5–18.3 µs)
with one epoch store + T−1 flag scans (~1–2 µs).

### What was measured on wallaby (Gold 6448Y; the node was busy with rival
### sessions the whole round, load 15–40 — session bands and min-of-samples
### quoted; the forced A/Bs are load-immune, the unforced picks are not)

| case | mt_r1 | mt_r2 | pick |
|---|---|---|---|
| B=1 | 0.129 µs | **0.137 µs min, identical to the mt_r1 binary interleaved in the same session** (8 alternating runs, every one 0.137 — the 0.129 vs 0.137 delta is the session, not the code; no layout regression from +~500 lines) | serial, fused_zp, unchanged |
| B=4096 | 9.4–12.3 ns/vol | **7.8 ns/vol** best unforced (31.8 µs/call); forced (fused_zp_pf, T=32) same-session A/B: base 30.3 µs vs new 30.0 µs — parity | under load the race locks T=16 (~11.5–13 ns/vol, tight sd) — shared-node artifact; exclusive node will resolve |
| B=65536 | 43.2 ns/vol (2828 µs) | **30.9–31.5 ns/vol** (2024.8–2063.3 µs, sd 0.25%, two processes) = **−28%**, and DRAM-bound enough to be load-immune | `3pass_nt_pf`, T=24/32 (within 2% of each other), disp=omp |

Correctness: PASS rel_l2 = 2.34–2.50e-16 at B=1/2/7/8/33/4096/65536,
repeatable (bit-identical across runs) everywhere; builds and passes
without OpenMP and without AVX2 (both checked at B=7 this round).
Parallel efficiency at B=65536: 301 ns (T=1 raced) / 31.1 ns (T=32) = 9.7×
on 32 — the socket's memory system, as mt_r1 argued, now with both sockets'
fill buffers in play; at ~10.4 KB/vol moved that is ~330 GB/s aggregate
DDR5, so wallaby is near the two-socket wall (it ran one socket in mt_r1
only because my race arena lied about the regime).

### The round's headline result

**mt_r1's "NT stores lose on wallaby even DRAM-bound" is FALSIFIED — it
was my own capped-arena artifact, exactly as L6_pfa's record predicted.**
At the 65536-volume race arena the NT kernels win their race outright:
3pass_nt_pf 31.1 ns/vol vs best regular-store kernel (fused_zp_pfw) 43.1
(−28%). The mt_r1 numbers that "killed" NT (fused_nt_pf +66% at nt=16384)
were measured in a cache-resident arena where streaming past the cache is
correctly poison. I nearly deleted all four NT kernels this round on that
reading (my own r10 prune rule); the only reason they survived is that the
node picked 3pass_nt_pf in mt_r1 anyway. Lesson recorded: **a prune
decision is only as good as the regime of the race that produced its
number.**

### What did NOT work / observations, with numbers

1. **The dispatch pool is a wash at B=65536** (od=31.4,31.1 and 31.1,31.1
   ns — under the 2% keep margin, correctly not kept): one fork per 2 ms
   call is 0.5%. It pays at B=4096: od=14.2,13.7 and 13.4,12.8 (−3.5 to
   −4.5%, kept both times it mattered) — consistent with L6_pfa's node
   measurement (−6%).
2. **sp2 + VD6 does not win on SPR**: raced 9.0–9.1 ns vs 8.9 (fused_zp_pf)
   at B=4096, 46.0 vs 43.1 at B=65536 — consistent with L6_pfa's "d2
   composed badly with the interleave" caveat. Kept raced for the node:
   CLX's smaller ROB is where sp2's argument actually lives, and their
   radix-3-first sp2 won there. If it takes zero picks on the node too,
   delete both next round (r10 prune rule).
3. **Wallaby was contended all round** (load 15–40, rival panel sessions;
   one displaced unforced B=4096 plan picked T=16 disp=pool and sat at
   ~13 ns/vol with sd 0.2% — tight, repeatable, and wrong for a quiet
   machine). All picks quoted above are therefore load-conditioned;
   the node's exclusive race is the one that counts. Forced-config A/Bs
   (L6_FORCE + L6_FORCE_T) were used to get load-immune comparisons.

### Borrowed

Everything material this round is from **L6_pfa (mt_r1)**: the 65536-volume
MT race cap (their point 5, "the aggregate cache sets the race-arena
floor"), the spin-pool dispatch design including the sequenced race and
>2% keep rule (their point 6), the sp2 fused stage (their panel_r5 kernel),
and the threaded end-to-end gate (their point 7). The pool's chunk split
and the gate's fallback wiring are mine, to keep output bit-identical to
the OMP path.

### Next round

1. **Read the node's B=65536 cell first.** Expected ~39–42 ns at T=32 with
   an NT pick (L6_pfa's regime, now raced honestly). If my 3pass_nt_pf vs
   their fused NT differ materially on the node, the 3pass-vs-fused NT
   question is one controlled twin away (fused_nt_pf is already in the
   table).
2. **B=4096 on the node**: if the pool ships (disp=pool) and the cell still
   trails L6_pfa, the residual is kernel (their sp2/radix-3-first vs my
   VD6) — consider a radix-3-first codelet twin for the sp2 shape only
   (VD63 exists in the geom history; it lost in cache as a *plain* fused
   codelet, which says nothing about the interleaved shape).
3. **NUMA staging stays unbuilt** until the node's tm= curve at the honest
   arena shows T=16 ≥ T=32 (it will not, on this evidence).
4. If sp2 takes zero node picks, delete it; if the pool loses everywhere,
   delete it too — dead code is not free in this file (r5: +3.5% B=1).

## Round mt_r3

### Where I stood

mt_r2 node: first in both batched cells — B=4096 9.36 ns/vol (38.34 µs/call,
pick `fused_zp_pf nthr=32 disp=pool`, od=16.2→9.3 ns: the pool wins big on
CLX) and B=65536 34.6 ns/vol (2269 µs, `3pass_nt_pf nthr=32 disp=omp`,
200 GB/s, sd ≤ 0.4% — the VERDICT calls it "the only large-batch cell in the
round that is both fast and reproducible" and the best wallaby→node transfer
of the round, 1.11×). B=1 0.222 vs L6_pfa's 0.220 — a scored 1% tie. The
VERDICT's L=6 judgment: "L=6 is otherwise finished", with two explicit asks
for the L=6 tuners (the `fr` placement instrument, and the mechanical
wide-team fix) — so this round is consolidation: race integrity, the
verdict's asks, and pruning, not new kernels.

### What changed

1. **The tournament and the team race now RUN THROUGH THE SPIN POOL**
   (ADOPTED FROM L6_pfa mt_r2, their round's payload). The pool is created
   before the race; every T>1 cell dispatches via `l6_run_pool` (idle
   workers spinning, exactly as in a scored pool run); T=1 cells stay
   direct calls; OMP is only the race fallback if pool creation fails. My
   mt_r2 race timed every cell through the GOMP fork/join (13–18 µs on the
   node) while the scored B=4096 run shipped the ~1–2 µs pool — ranking
   configurations under a dispatch the winner never uses, the exact defect
   behind L6_pfa's mt_r1 sp2/T=24 mis-pick (14.4 vs 9.4 ns/vol). Measured
   effect on race fidelity (wallaby, B=4096): the mt_r2 OMP-raced team
   curve read 23 ns/vol at T=32 for a 9.3 ns shipped config (2.5× off);
   the pool-raced curve reads 8.6 ns for an 8.6 ns ship — **the arena now
   prices the shipped configuration exactly**. The dispatch race is
   re-sequenced to match (also their design): pool timed first (warm from
   the tournament), then DESTROYED — workers joined, cores genuinely idle,
   GOMP's team futex-slept through the multi-second tournament — then OMP
   timed, pool recreated only if it won by >2%. The description gains
   `rd=pool|omp` (which dispatch the tournament ran under).
2. **Pool join fixes** (both ADOPTED FROM L6_pfa mt_r2, who found the bug
   in the pool design I had copied from their mt_r1): (a) my done-flag
   elements were `{_Atomic long; char pad[48]}` = **56 B** — `aligned(64)`
   on the array does not align elements, so adjacent workers' done flags
   shared cache lines and the join's release-stores ping-ponged; now 64 B
   each. (b) The master prefetches all done-flag lines before scanning, so
   the cross-core misses overlap in the fill buffers instead of
   serializing.
3. **Wide-team incumbency at streaming cells** (the mt_r2 VERDICT §6
   mechanical fix for the panel-wide T=32→T=16 pick lottery — L6_pfa lost
   2.06× at B=65536 to exactly this). When the real working set exceeds
   128 MiB (beyond any aggregate cache here: node ~76 MiB, wallaby
   ~124 MiB), the widest T is the team-race incumbent and a narrower team
   must beat it by >2%; cache-resident cells keep smallest-T-within-2%.
   This fired immediately on wallaby at B=65536: the raced curve read
   T=24 30 ns vs T=32 31 ns — the old smallest-T rule would have shipped
   T=24; the new rule kept T=32, which is what the node's curve
   (24:52 vs 32:39) says is right where it counts.
4. **`fr=` placement instrument** (the VERDICT §6 ask for the L=6 tuner;
   ADOPTED FROM L8_fusedaxes mt_r2's `gov_scan_remote`, attributed): a
   read-only `get_mempolicy(MPOL_F_NODE|MPOL_F_ADDR)` sample of ~128 pages
   of each caller buffer, on the 1st and 49th threaded execute (the
   49th-call rescan is the "fr under a wide team during the timed loop"
   reading the VERDICT says nobody has taken), published as
   `fr=<pct0>/<pct48>,nb=<autonuma>` through the description. Reading page
   homes is explicitly allowed; migrating them is not. Cost: ~256 syscalls
   on exactly two calls — invisible to min-of-samples scoring.
5. **Pruned** (my r10 rule — zero node picks closes a question, and dead
   code is not free in this file): the sp2 twins (0 node picks in mt_r2,
   lost every wallaby race, and L6_pfa's own caveat said VD6 composes badly
   with the interleave), `fused_nt`, `fused_zp_nt_pf`, `fused_nt_pfnta`
   (0 node picks each in 2 rounds; pfnta was the worst of all 14 wallaby
   candidates, +135%). The candidate table drops 16 → 11. NT keeps
   `3pass_nt_pf` (the node's B=65536 winner) and `fused_nt_pf` (the
   controlled 3pass-vs-fused NT twin, still an open question).

### Operation count

Unchanged: 48 real flops / 36 arithmetic instructions per 6-point line, 108
line-DFTs per volume, 972 ymm arithmetic uops + 108 in-codelet shuffles per
volume. This round adds zero arithmetic; per execute the only new work is
two 256-syscall page scans on calls 1 and 49, and the join scan gains 31
prefetches.

### What was measured on wallaby (Gold 6448Y, 32 threads close; load ~2.4,
### near-idle — but the B=1 displacement hazard still fired once, see below)

| case | mt_r2 | mt_r3 | pick |
|---|---|---|---|
| B=1 | 0.129/0.137 µs (session bands) | **0.129–0.130 µs** modal, interleaved A/B vs the mt_r2 binary: old {0.130, 0.130, 0.193, 0.130}, new {0.129, 0.323, 0.130, 0.129} — parity; the 0.193/0.323 blips hit BOTH binaries (shared-node displacement, not code). No layout regression from the prune. | serial `fused_zp`, path untouched |
| B=4096 | 7.8 ns/vol best unforced | **8.1–8.6 ns/vol** (33.3–35.2 µs/call, sd 0.07–0.15%), same band; od=9.5,8.6 ns (pool by ~9%, kept), rd=pool | `fused_zp` / `fused_zp_pf`, T=32, disp=pool |
| B=65536 | 30.9–31.5 ns/vol | **30.7 ns/vol** (2013.5 µs min, median 2021), setup 2.7 s → **1.4 s** (pool-raced cells don't pay the fork) | `3pass_nt_pf`, T=32 kept by the streaming rule over a raced T=24 within noise, disp=omp (od=30.7,30.5 — a wash, correctly not kept) |

Correctness: PASS rel_l2 = 2.34–2.43e-16 at B=1/7/33/4096/65536, repeatable
(bit-identical across runs) everywhere; builds and runs clean without
OpenMP and compiles without AVX2 (checked this round at B=7). The verbose
B=4096 race table is clean and monotone: fused family 8.0–8.1, 3pass family
8.6–9.1, NT correctly poison at 19.7–21.2 (cache-resident 27 MiB), team
curve 267/134/70/34/18/11/8.6 ns — near-linear to T=32 through the pool.
`fr=0/0,nb=1` on wallaby at both batched cells (all sampled caller pages
local to the main thread's node, no migration during the loop) — the
instrument's real reading comes on the two-socket node.

### What did NOT work / observations, with numbers

1. **Nothing regressed, but nothing on wallaby moved much either** — the
   round's changes are all aimed at node-side race integrity (the fork tax
   is 13–18 µs on CLX vs 4.5–5.5 on SPR, so the pool-racing payoff is
   mostly there) and at pick stability, which by construction cannot show
   on a machine that was already picking right. The measurable wallaby
   wins: race-arena fidelity 2.5×→1.0× (23→8.6 ns raced vs 8.6 shipped)
   and setup time −48% at B=65536.
2. **The B=1 displacement hazard on wallaby persists** (one 0.323 blip in
   8 interleaved runs, hitting both old and new binaries equally). Same
   hazard my record has flagged since phase-1 r4; scored numbers come from
   the exclusive node.

### Borrowed

- **L6_pfa (mt_r2)**: the pool-raced tournament, the re-sequenced dispatch
  race (pool first, destroy, OMP on idle cores, recreate on win), the
  64-byte done-flag fix, and the join-scan prefetch. All attributed inline.
- **L8_fusedaxes (mt_r2)**: the `fr`/`nb` read-only placement scan
  (`gov_scan_remote`), ported as `l6_scan_remote` per the VERDICT §6 ask.
- **The mt_r2 VERDICT §6**: the streaming-cell wide-team incumbency rule.

### Next round

1. **Read the node's `fr=` values first** — B=65536 under T=32 is the
   measurement the VERDICT says settles §5's mechanism (arena races in the
   pre-spread placement regime). If `fr>0` appears in the /48 rescan, the
   next move is a migration-settling warmup before the team race, not a
   kernel change.
2. **B=4096 on the node**: rd=pool should firm the pick (expected
   `fused_zp_pf`-family T=32 pool again, maybe a hair under 9.3 ns since
   the tm race no longer pays the fork tax anywhere). If L6_pfa's node
   number drops materially below mine here, the residual is kernel, and
   the sp2/radix-3-first question reopens — but only with THEIR codelet,
   not VD6 (three strikes on that combination now).
3. **B=65536 is at the wall** (200 GB/s on DDR4-2666 ×2 sockets); expect
   ~34–35 ns again. The only lever left there is placement luck, which
   `fr=` now measures instead of guesses.
4. If `fused_nt_pf` takes zero node picks again this round, delete it and
   declare 3pass the settled NT shape at L=6 (r10 rule).
