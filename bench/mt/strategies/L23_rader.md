# L23_rader — multicore strategy record

Phase-1 history (how the serial kernel got its shape, and the closed list of
streaming nulls that must not be rebuilt) lives in
`../../geom/strategies/L23_rader.md`. This file starts at the multicore phase.
Phase-1 baselines used below: node (CLX, panel_r11 scored) B=1 47.469 µs,
B=4 49.678, B=128 64.793 µs/vol; wallaby (SPR, driver-level) 21.35 / 23.34 /
24.3 µs per volume.

## Round mt_r1

### What changed

The arithmetic is untouched: same conjugate-folded Rader pair (297 vector FP
ops per line-group, 409 zmm chunks per volume, 943 kflop), same two-sweep
pinned-constant kernel, same X-first pass order, same 1064-double t1 plane
stride, sine-table-folded -i rotation. Every volume's DAG is identical no
matter which thread runs it, so output is bit-identical to the serial exec at
every team size and for every tuner cell (cmp-VERIFIED, see below). What was
added is a threading layer, and what was *deleted* is every phase-1 exec
variant the verdicts had already declared a null (rp-t1, deferred-Z, both
pipelining schedules, pf=1, krn_il, P-parking, X-last) — the MT file races
modes, not corpses.

1. **Range execs** `(plan, in, out, b0, b1, slot)` over per-thread page-sized
   scratch slots (t1 24472 + pb 1104 + NT staging plane 1064 doubles, padded
   to 53 pages = 217 KiB), allocated once and **first-touched by the owning
   thread** in a full-width parallel region in `fft3d_create()` — NUMA-local
   scratch, no false sharing by page construction, and the region spins up
   the OpenMP pool so the first timed execute creates no threads.
2. **mode=batch**: team of min(batch, T) threads, thread t owns the contiguous
   volume block [nb·t/T, nb·(t+1)/T), ranges computed from the team OpenMP
   actually delivers; zero synchronisation except the join. pf=2 (in-pass X
   prefetch) and pw=1 (write-intent out-plane prefetch) kept as per-thread
   knobs.
3. **mode=batchNT**: same, but each finished plane is staged and streamed to
   `out` with NT stores. This was a four-round phase-1 null; it was raced
   again on the argument that with 32 cores sharing DRAM, the RFO third of
   the out traffic is a bigger slice of a scarcer resource. **The null
   flipped**: on wallaby at nv=403 (true streaming), batchNT pf2 = 1.74 µs/t
   vs plain-batch best 2.00 — the round's one transferable finding (other
   entries whose NT variants lost single-threaded should re-race them).
4. **mode=fused** (batch ≤ nslots): omp-for over all batch·133 X chunks
   (chunk i of volume v writes whole distinct 64-B lines of slot v's t1; the
   overlapping tail chunk rewrites bit-identical values, benign cross-thread),
   implicit barrier — the one true global dependency, since every X chunk
   writes one (y,z) column of *all 23* kx-planes — then omp-for `nowait` over
   all batch·23 (v,x) planes, each Y→Z on the executing thread's own pb.
   At B=1 this is the intra-volume split.
5. **Deterministic joint-cell hysteretic tuner** (phase-1 r9 mechanism, cells
   extended to (mode, width, team, pf, pw)): one canonical list per regime,
   two fixed-order sweeps, per-cell min, >2% to displace, licence-honest
   1.5 ms warmups. Heads by the fastest-known-head rule: B=1 serial (the
   node-measured phase-1 pick), 1<B<32 one-volume-per-thread, B≥32 full-team
   batch pf2 pw1. team=16 cells sit in the B≥32 list as the node's
   two-socket question (driver first-touches in/out single-threaded → one
   socket owns them; 16 close-bound threads = one CLX socket, all-local).
   Small batches tune at the TRUE batch (the decomposition is a function of
   it), streaming batches on the L3-relative arena. Since all cells are one
   bit class, no pick can change the output — the pick lottery is
   structurally harmless here, unlike phase-1 r8.
6. Env knobs: `L23R_FORCE` (cell index), `L23R_TEAM`, `L23R_PF`, `L23R_PW`,
   `L23R_VERBOSE`. Tuner telemetry (pick + list-head µs/t, arena size) baked
   into the description string as before.

### Operation count

Per volume unchanged from panel_r11: 3·529 line-groups × 297 vector FP ops =
943 kflop, 409 zmm chunks (flat layout only now). The MT layer adds zero
arithmetic; fused adds two team barriers per call, batch adds one fork/join.

### What was measured on wallaby (Gold 6448Y SPR, 32 threads close/cores, driver-level, shared login node)

| case | phase-1 serial (same machine) | mt_r1 | per-vol | speedup | parallel efficiency |
|---|---|---|---|---|---|
| B=1 | 21.35 µs | **7.65–8.1 µs** (fused team=23) | 7.65 µs | 2.8× | 12% on 23 threads |
| B=3 | ~22.8 µs/vol | 16.5 µs/call | 5.5 µs | 4.1× | — |
| B=8 | ~23.3 µs/vol | 24.2 µs/call | 3.02 µs | 7.7× | 97% on 8 threads |
| B=16 | ~23.3 µs/vol | 47.6 µs/call (batch-16 pf2 pw1) | 2.98 µs | 7.8× | 49% on 16 threads |
| B=32 | ~24.3 µs/vol | 45.0 µs/call | 1.41 µs | 17× | 54% on 32 |
| B=33 | — | 68.4 µs/call (2-vols/thread boundary, PASS) | 2.07 µs | — | — |
| B=128 | 24.3 µs/vol | **140.8 µs/call** (batchNT) | **1.10 µs** | **22×** | **69% on 32** |
| B=512 | 24.3 µs/vol | 892 µs/call (batchNT) | 1.74 µs | 14× | 44% |

Correctness: PASS rel_l2 = 3.77e-16 … 3.81e-16 at B = 1, 3, 4, 8, 16, 32,
33, 128, 512; repeatable (bit-identical across runs) everywhere.
Bit-class verification: forced cells batch / batch+pf2pw1 / fused-32 /
fused-16 / serial / batch-w2 all cmp-identical on full B=3 outputs.

Key tuner tables (one quiet-ish run each):
* B=1 (nv=1): serial 32.07; fused t32 9.83, **t23 9.07 ← kept**, t16 10.48,
  t8 13.46, t4 23.24. The 23-plane phase sets the natural team.
* B=16 (nv=16): batch-16 pf0 3.35, **pf2 pw1 2.96 ← kept**; fused t16 3.41,
  t32 6.15 (cross-thread t1 handoff: X writers ≠ plane readers, so fused
  loses to thread-private t1 once every thread has a whole volume).
* B=512 (nv=403): batch t32 pf2pw1 2.07, pf0pw1 2.00; **batchNT pf2 1.74 ←
  kept** (−13% vs plain best); batch t16 3.49 (wallaby is one socket at 32
  threads, so t16 just halves the compute — the node must re-ask this);
  serial 32.05.

Where the missing cores went, honestly: at B=128 the call moves ≥128 × 0.95
MB (in + out + RFO-or-NT + 2×t1) in 141 µs ≈ 550 GB/s aggregate including
cache reuse — bandwidth-shared territory, and exactly where NT's RFO savings
pay. At B=1 the 7.7 µs decomposes as ~1.4 µs of longest-thread work plus
~6 µs of GOMP fork + one barrier + join on a *shared* login node (L13_direct
measured the same 4–5 µs GOMP region cost); the fused split still buys 2.8×
because L=23's 21 µs of work dwarfs it — the opposite balance from
L13_direct's B=1 (2.5 µs of work, split lost). Both results support the same
rule: intra-volume splitting pays iff work ≫ region cost.

### What did NOT work / observations with numbers

1. **fused at team=32 loses to team=23 at B=1** (9.83 vs 9.07): the plane
   phase has exactly 23 tasks, so threads 24–31 only help the X phase and
   then idle into the barrier; the wider fork costs more than the X-phase
   help. Kept team=23 in the canonical list ahead of 32? No — list order is
   head-first only; both are raced, hysteresis decides per machine.
2. **fused at mid batch** (B=16): 6.15 µs/t at t32 vs 3.35 for plain
   batch-16 — volume v's t1 is written by many threads and read by others
   (dirty-line migration every plane), where batch mode keeps t1 L2-private.
   Fused is only for batch < team.
3. **The second fused barrier was pure waste**: `nowait` on the plane loop
   (join is the sync) took B=1 from 8.68 to 7.69 µs, −11%. One barrier ≈
   1 µs at team 23 on this machine.
4. **The raw-ssh trap** (documented five times in the phase-1 record, and I
   did it AGAIN, twice): one-off remote commands must be
   `ssh wallaby 'source ~/fft/env.sh; cd ~/fft/bench/mt; …'` with absolute
   paths, `test -s` before any cmp, `&&`-chained.

### Borrowed this round (attributions)

**L13_direct mt_r1** (the only other MT record when this round started): the
whole threading-layer design — range execs with per-thread slots first-touched
by their owner inside a create()-time full-width region, ranges computed from
the delivered team, contiguous volume blocks, one-volume-per-thread as the
mid-batch head, and the "B=1 splits iff work ≫ region cost" framing that
predicted my fused win. From my own phase-1 lineage: krn_ts, pf=2/pw=1,
NT-copy machinery, the joint-cell hysteretic tuner, fastest-known-head rule,
licence-honest warmups, L3-relative arena sizing.

### Node prediction (pre-registered)

Picks: B=1 fused (team 23 or 32 — CLX has 2×16 cores, so team 16 may also
surface: it is all-one-socket there); B=4 fused t32; B=128 **batchNT** if the
node's DRAM contention resembles wallaby's, else batch pf2 pw1. Levels are
guesses across a µarch gap: B=1 8–15 µs (fork cost on the idle pinned node
should be *below* wallaby's), B=128 1.7–2.6 µs/vol (node phase-1 streaming
was 2.7× wallaby's per-vol time). The t16-vs-t32 field in the B≥32 tuner
telemetry is the round's NUMA instrument — read it before believing any
UPI story.

### Next

1. **Persistent spin-pool** (pthreads pinned to the harness's cores,
   sense-reversing barrier, created in fft3d_create) — the B=1 cell is
   ~80% synchronisation; cutting fork+barrier+join from ~6 µs to ~1 µs is
   worth ~3× there and helps every batch < 8. L13_direct's next-round item
   too; whoever builds it first, the other should copy. Only build it on
   node confirmation that the fork cost survives on idle pinned hardware.
2. If the node's B≥32 telemetry shows t16 ≥ t32 (UPI-bound): don't cap the
   team — make the far socket's threads stage their volumes' `in` through
   local scratch (one extra 190 KiB local read converts every remote load
   into a single remote stream), and consider NT for the far socket only.
3. Race a fused variant whose X phase assigns chunks so that the thread that
   writes t1 columns of plane x is the one that later reads plane x — a
   chunk→plane affinity map could cut the mid-batch dirty-line handoff and
   make fused competitive at 2 vols/thread (the B=33-boundary 2.07 µs/vol
   is the target).
