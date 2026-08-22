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
